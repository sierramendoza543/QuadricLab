"use client";

import { useEffect, useState } from "react";
import { AiReview, buildPayload } from "@/lib/ml/ai";
import { Table, targetColumns } from "@/lib/ml/datasets";
import { chooseModel } from "@/lib/ml/lesson";

interface Props {
  table: Table;
  review: AiReview | null;
  onReview: (r: AiReview | null) => void;
}

export default function ReviewPanel({ table, review, onReview }: Props) {
  const [available, setAvailable] = useState<boolean | null>(null);
  const [consent, setConsent] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/analyze")
      .then((r) => r.json())
      .then((j) => setAvailable(Boolean(j.available)))
      .catch(() => setAvailable(false));
  }, []);

  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      const targets = targetColumns(table);
      const guess = targets[targets.length - 1];
      const feats = table.columns.map((_, i) => i).filter((i) => i !== guess);
      const c = chooseModel(table, guess, feats);
      const payload = buildPayload(table, { linear: c.search.r2Linear, searched: c.search.r2Final, extra: c.terms.slice(c.search.base).map((t) => t.name) });
      const res = await fetch("/api/analyze", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
      const j = await res.json();
      if (!res.ok) throw new Error(j.error === "rate_limited" ? "Too many reviews from this connection. Try again later." : "The review could not be completed, so the built-in analysis is being used.");
      onReview(j.review as AiReview);
    } catch (e) {
      setError(e instanceof Error ? e.message : "The review failed.");
    } finally {
      setBusy(false);
    }
  };

  if (available === null) return null;

  if (!review) {
    return (
      <div className="mt-4 border border-gray-300 p-4">
        <h3 className="mb-1 font-bold">AI review (optional)</h3>
        {available ? (
          <>
            <p className="mb-2 text-[16px] text-gray-700">
              The AI reads a summary of your data and suggests what each column is, which variable to predict, and which interaction or squared terms are worth testing. Every suggestion is scored on held-out rows before it is used.
            </p>
            <label className="mb-2 flex items-start gap-2 text-[15px]">
              <input type="checkbox" className="mt-1" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
              <span>Send column names, summary statistics and 8 sample rows to Google Gemini. The full file is not sent.</span>
            </label>
            <button disabled={!consent || busy} onClick={run} className="border border-blue-800 bg-blue-700 px-4 py-1.5 font-bold text-white hover:bg-blue-800 disabled:border-gray-300 disabled:bg-gray-200 disabled:text-gray-500">
              {busy ? "Reviewing…" : "Review my data"}
            </button>
          </>
        ) : (
          <p className="text-[16px] text-gray-700">The AI review is not turned on for this server, so the built-in analysis is used: it encodes text columns, then scores squared and interaction terms on held-out rows.</p>
        )}
        {error && <p className="mt-2 text-red-700">{error}</p>}
      </div>
    );
  }

  return (
    <div className="mt-4 border border-blue-300 bg-blue-50 p-4">
      <div className="mb-1 flex items-baseline justify-between">
        <h3 className="font-bold">AI review</h3>
        <button className="text-sm text-blue-700 underline" onClick={() => onReview(null)}>
          Discard
        </button>
      </div>
      <p className="mb-2 text-[16px]">{review.summary}</p>
      <p className="mb-2 text-[16px]">
        <b>Predict:</b> {review.target}. <b>Using:</b> {review.predictors.join(", ") || "(none listed)"}.
      </p>
      {review.extra_terms.length > 0 && (
        <>
          <div className="font-bold">Terms to test</div>
          <ul className="mb-2 list-disc pl-6 text-[16px]">
            {review.extra_terms.map((e, i) => (
              <li key={i}>
                <b>{e.kind === "square" ? `${e.columns[0]}²` : e.columns.join(" × ")}</b>: {e.why}
              </li>
            ))}
          </ul>
        </>
      )}
      {review.warnings.length > 0 && (
        <>
          <div className="font-bold text-amber-800">Watch out for</div>
          <ul className="mb-2 list-disc pl-6 text-[16px]">
            {review.warnings.map((w, i) => (
              <li key={i}>{w}</li>
            ))}
          </ul>
        </>
      )}
      <p className="text-[15px] text-gray-700">{review.reasoning}</p>
    </div>
  );
}
