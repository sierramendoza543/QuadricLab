import { AI_SCHEMA, AI_SYSTEM, AiPayload, AiReview } from "@/lib/ml/ai";

export const runtime = "nodejs";
export const maxDuration = 60;

// a small in-memory limit per client; use a shared store if you deploy several instances
const hits = new Map<string, number[]>();
const LIMIT = 12;
const WINDOW = 60 * 60 * 1000;

const clip = (s: unknown, n = 300) => (typeof s === "string" ? s.slice(0, n) : "");

export async function GET() {
  return Response.json({ available: Boolean(process.env.GEMINI_API_KEY) });
}

export async function POST(req: Request) {
  const key = process.env.GEMINI_API_KEY;
  if (!key) return Response.json({ error: "not_configured" }, { status: 503 });

  const ip = req.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? "local";
  const now = Date.now();
  const recent = (hits.get(ip) ?? []).filter((t) => now - t < WINDOW);
  if (recent.length >= LIMIT) return Response.json({ error: "rate_limited" }, { status: 429 });
  hits.set(ip, [...recent, now]);

  const text = await req.text();
  if (text.length > 80_000) return Response.json({ error: "too_large" }, { status: 413 });
  let payload: AiPayload;
  try {
    payload = JSON.parse(text);
  } catch {
    return Response.json({ error: "bad_json" }, { status: 400 });
  }
  if (!Array.isArray(payload.columns) || payload.columns.length === 0 || payload.columns.length > 60) return Response.json({ error: "bad_payload" }, { status: 400 });

  try {
    const model = process.env.GEMINI_MODEL ?? "gemini-3.5-flash-lite";
    const res = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify({
        systemInstruction: { parts: [{ text: AI_SYSTEM }] },
        contents: [{ role: "user", parts: [{ text: JSON.stringify(payload) }] }],
        generationConfig: { responseMimeType: "application/json", responseJsonSchema: AI_SCHEMA, temperature: 0.2, maxOutputTokens: 8000, thinkingConfig: { thinkingLevel: "low" } },
      }),
      signal: AbortSignal.timeout(40_000),
    });
    if (res.status === 429) return Response.json({ error: "upstream_busy" }, { status: 503 });
    if (res.status === 400 || res.status === 403) return Response.json({ error: "bad_key_or_request" }, { status: 502 });
    if (!res.ok) return Response.json({ error: "upstream", status: res.status }, { status: 502 });

    const j = await res.json();
    if (j?.promptFeedback?.blockReason) return Response.json({ error: "refused" }, { status: 502 });
    const out = (j?.candidates?.[0]?.content?.parts ?? []).map((p: { text?: string }) => p.text ?? "").join("");
    if (!out) return Response.json({ error: "empty" }, { status: 502 });
    const raw = JSON.parse(out) as AiReview;

    const names = new Set(payload.columns.map((c) => c.name));
    const sources = new Set(payload.columns.map((c) => c.from));
    const known = (n: string) => names.has(n) || sources.has(n);
    const review: AiReview = {
      summary: clip(raw.summary, 700),
      target: names.has(raw.target) ? raw.target : payload.candidateTarget,
      predictors: (raw.predictors ?? []).filter(known).slice(0, 60),
      column_notes: (raw.column_notes ?? []).filter((c) => known(c.name)).slice(0, 60).map((c) => ({ name: c.name, role: c.role, note: clip(c.note) })),
      extra_terms: (raw.extra_terms ?? [])
        .filter((e) => Array.isArray(e.columns) && e.columns.every((c) => names.has(c)) && (e.kind === "square" ? e.columns.length === 1 : e.columns.length === 2))
        .slice(0, 5)
        .map((e) => ({ kind: e.kind, columns: e.columns, why: clip(e.why) })),
      surface_pair: { terms: (raw.surface_pair?.terms ?? []).filter((c) => names.has(c)).slice(0, 2), why: clip(raw.surface_pair?.why) },
      warnings: (raw.warnings ?? []).slice(0, 6).map((w) => clip(w)),
      reasoning: clip(raw.reasoning, 500),
    };
    return Response.json({ review });
  } catch {
    return Response.json({ error: "failed" }, { status: 500 });
  }
}
