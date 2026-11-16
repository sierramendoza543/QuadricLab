import { Source, Table } from "./datasets";

export interface ParsedCSV {
  table: Table;
  droppedColumns: string[];
  droppedRows: number;
  sampled: boolean;
  encoded: string[];
}

const MAX_ROWS = 5000;
const MAX_COLS = 40;
const MAX_LEVELS = 12;

function splitRows(text: string, delim: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (ch === '"') quoted = false;
      else cell += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === delim) {
      row.push(cell);
      cell = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell);
      cell = "";
      if (row.some((c) => c.trim() !== "")) rows.push(row);
      row = [];
    } else cell += ch;
  }
  row.push(cell);
  if (row.some((c) => c.trim() !== "")) rows.push(row);
  return rows;
}

const asNumber = (s: string) => {
  const t = s.trim();
  if (t === "") return NaN;
  return Number(t);
};

export function parseCSV(text: string, name: string): ParsedCSV {
  const firstLine = text.split(/\r?\n/, 1)[0] ?? "";
  const delim = [",", ";", "\t"].map((d) => ({ d, n: firstLine.split(d).length })).sort((a, b) => b.n - a.n)[0].d;
  const raw = splitRows(text, delim);
  if (raw.length < 2) throw new Error("That file looks empty.");

  const hasHeader = raw[0].some((c) => c.trim() !== "" && Number.isNaN(asNumber(c)));
  const names = hasHeader ? raw[0].map((c, i) => c.trim() || `col${i + 1}`) : raw[0].map((_, i) => `col${i + 1}`);
  const body = hasHeader ? raw.slice(1) : raw;

  type Plan = { j: number; kind: "numeric" | "categorical"; levels?: string[] };
  const plans: Plan[] = [];
  const dropped: string[] = [];

  names.forEach((n, j) => {
    const cells = body.map((r) => (r[j] ?? "").trim());
    const filled = cells.filter((c) => c !== "");
    if (filled.length / body.length < 0.9) return void dropped.push(n);
    const nums = filled.filter((c) => !Number.isNaN(Number(c)));

    if (nums.length / filled.length >= 0.9) {
      const distinct = new Set(nums.map(Number));
      if (distinct.size < 2) return void dropped.push(n);
      const ints = nums.every((c) => Number.isInteger(Number(c)));
      const idLike = ints && distinct.size === nums.length && (/(^|_|\s)(id|index|idx)$/i.test(n) || /^unnamed/i.test(n) || n === "");
      if (idLike) return void dropped.push(`${n} (row ID)`);
      plans.push({ j, kind: "numeric" });
      return;
    }
    const counts = new Map<string, number>();
    for (const c of filled) counts.set(c, (counts.get(c) ?? 0) + 1);
    if (counts.size >= 2 && counts.size <= MAX_LEVELS && counts.size <= Math.max(2, body.length / 4)) {
      const levels = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([l]) => l);
      plans.push({ j, kind: "categorical", levels });
    } else dropped.push(n);
  });

  if (!plans.some((p) => p.kind === "numeric")) throw new Error("I need at least one numeric column to predict.");
  if (plans.length < 2) throw new Error("I need at least two usable columns (one to predict, one to learn from).");

  // encoded layout, in the original column order
  const columns: string[] = [];
  const sources: Source[] = [];
  const notes: Record<string, string> = {};
  const encoded: string[] = [];
  const emit: { plan: Plan; cols: { name: string; level?: string }[] }[] = [];
  for (const p of plans) {
    const n = names[p.j];
    let cols: { name: string; level?: string }[];
    if (p.kind === "numeric") cols = [{ name: n }];
    else cols = p.levels!.slice(1).map((l) => ({ name: `${n}_${l}`, level: l }));
    if (columns.length + cols.length > MAX_COLS) {
      dropped.push(n);
      continue;
    }
    columns.push(...cols.map((c) => c.name));
    emit.push({ plan: p, cols });
  }

  let rows: number[][] = [];
  let droppedRows = 0;
  for (const r of body) {
    const out: number[] = [];
    let ok = true;
    for (const { plan: p, cols } of emit) {
      const cell = (r[p.j] ?? "").trim();
      if (p.kind === "numeric") {
        const v = asNumber(cell);
        if (!Number.isFinite(v)) ok = false;
        out.push(v);
      } else {
        if (cell === "" || !p.levels!.includes(cell)) ok = false;
        for (const c of cols) out.push(cell === c.level ? 1 : 0);
      }
    }
    if (ok) rows.push(out);
    else droppedRows++;
  }
  if (rows.length < 20) throw new Error("After removing rows with missing values, fewer than 20 rows were left.");

  const preview = {
    columns: emit.map((e) => names[e.plan.j]),
    rows: body.slice(0, 8).map((r) => emit.map((e) => (r[e.plan.j] ?? "").trim())),
  };

  let sampled = false;
  if (rows.length > MAX_ROWS) {
    const stride = rows.length / MAX_ROWS;
    rows = Array.from({ length: MAX_ROWS }, (_, i) => rows[Math.floor(i * stride)]);
    sampled = true;
  }

  let at = 0;
  for (const { plan: p, cols } of emit) {
    const n = names[p.j];
    if (p.kind === "numeric") {
      const col = rows.map((r) => r[at]);
      sources.push({ name: n, kind: "numeric", columns: [n], mean: col.reduce((s, v) => s + v, 0) / col.length, min: Math.min(...col), max: Math.max(...col) });
    } else {
      sources.push({
        name: n,
        kind: "categorical",
        columns: cols.map((c) => c.name),
        levels: p.levels,
        baseline: p.levels![0],
        dummies: cols.map((c) => ({ column: c.name, level: c.level! })),
      });
      for (const c of cols) {
        notes[c.name] = `1 if ${n} is "${c.level}", otherwise 0 (compared with "${p.levels![0]}")`;
        encoded.push(c.name);
      }
    }
    at += cols.length;
  }

  return {
    table: { name, columns, rows, notes, sources, preview },
    droppedColumns: dropped,
    droppedRows,
    sampled,
    encoded,
  };
}
