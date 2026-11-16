import { Table, sourcesOf, targetColumns } from "./datasets";
import { AiTermSpec } from "./search";

export interface AiColumnNote {
  name: string;
  role: "target" | "predictor" | "ignore";
  note: string;
}

export interface AiReview {
  summary: string;
  target: string;
  predictors: string[];
  column_notes: AiColumnNote[];
  extra_terms: { kind: "square" | "interaction"; columns: string[]; why: string }[];
  surface_pair: { terms: string[]; why: string };
  warnings: string[];
  reasoning: string;
}

export interface AiPayload {
  dataset: string;
  rows: number;
  candidateTarget: string;
  columns: {
    name: string;
    from: string;
    kind: "numeric" | "0/1 indicator";
    mean: number;
    std: number;
    min: number;
    max: number;
    distinct: number;
    corrWithTarget: number;
    note?: string;
  }[];
  sample: { columns: string[]; rows: (string | number)[][] };
  localAnalysis: { linearHeldOutR2: number; searchedHeldOutR2: number; extraTermsFound: string[] };
}

const round = (v: number, d = 3) => Math.round(v * 10 ** d) / 10 ** d;

// a compact description of the data: statistics and a few rows, never the whole file
export function buildPayload(table: Table, local: { linear: number; searched: number; extra: string[] }): AiPayload {
  const targets = targetColumns(table);
  const guess = targets[targets.length - 1];
  const y = table.rows.map((r) => r[guess]);
  const my = y.reduce((s, v) => s + v, 0) / y.length;
  const sy = Math.sqrt(y.reduce((s, v) => s + (v - my) ** 2, 0) / y.length) || 1;
  const src = new Map<string, string>();
  sourcesOf(table).forEach((s) => s.columns.forEach((c) => src.set(c, s.name)));

  const columns = table.columns.map((name, j) => {
    const col = table.rows.map((r) => r[j]);
    const m = col.reduce((s, v) => s + v, 0) / col.length;
    const sd = Math.sqrt(col.reduce((s, v) => s + (v - m) ** 2, 0) / col.length) || 1e-12;
    const cov = col.reduce((s, v, i) => s + (v - m) * (y[i] - my), 0) / col.length;
    const dummy = table.sources?.some((s) => s.kind === "categorical" && s.columns.includes(name));
    return {
      name,
      from: src.get(name) ?? name,
      kind: dummy ? ("0/1 indicator" as const) : ("numeric" as const),
      mean: round(m),
      std: round(sd),
      min: round(Math.min(...col)),
      max: round(Math.max(...col)),
      distinct: new Set(col).size,
      corrWithTarget: round(cov / (sd * sy)),
      note: table.notes?.[name],
    };
  });
  const sample = table.preview
    ? { columns: table.preview.columns, rows: table.preview.rows }
    : { columns: table.columns, rows: table.rows.slice(0, 8).map((r) => r.map((v) => round(v, 2))) };
  return {
    dataset: table.name,
    rows: table.rows.length,
    candidateTarget: table.columns[guess],
    columns,
    sample,
    localAnalysis: { linearHeldOutR2: round(local.linear), searchedHeldOutR2: round(local.searched), extraTermsFound: local.extra },
  };
}

// map the review onto real columns; anything that does not exist is ignored
export function reviewSelection(table: Table, review: AiReview) {
  const byName = new Map(table.columns.map((c, j) => [c, j]));
  const srcCols = new Map(sourcesOf(table).map((s) => [s.name, s.columns]));
  const targets = new Set(targetColumns(table));
  const t = byName.get(review.target);
  const target = t !== undefined && targets.has(t) ? t : null;
  const feats = new Set<number>();
  for (const p of review.predictors) {
    if (byName.has(p)) feats.add(byName.get(p)!);
    else for (const c of srcCols.get(p) ?? []) feats.add(byName.get(c)!);
  }
  if (target !== null) feats.delete(target);
  const extras: AiTermSpec[] = review.extra_terms.map((e) => ({ kind: e.kind, columns: e.columns, why: e.why }));
  return { target, feats: [...feats].sort((a, b) => a - b), extras };
}

export const AI_SCHEMA = {
  type: "object",
  additionalProperties: false,
  required: ["summary", "target", "predictors", "column_notes", "extra_terms", "surface_pair", "warnings", "reasoning"],
  properties: {
    summary: { type: "string" },
    target: { type: "string" },
    predictors: { type: "array", items: { type: "string" } },
    column_notes: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["name", "role", "note"],
        properties: { name: { type: "string" }, role: { type: "string", enum: ["target", "predictor", "ignore"] }, note: { type: "string" } },
      },
    },
    extra_terms: {
      type: "array",
      items: {
        type: "object",
        additionalProperties: false,
        required: ["kind", "columns", "why"],
        properties: { kind: { type: "string", enum: ["square", "interaction"] }, columns: { type: "array", items: { type: "string" } }, why: { type: "string" } },
      },
    },
    surface_pair: {
      type: "object",
      additionalProperties: false,
      required: ["terms", "why"],
      properties: { terms: { type: "array", items: { type: "string" } }, why: { type: "string" } },
    },
    warnings: { type: "array", items: { type: "string" } },
    reasoning: { type: "string" },
  },
};

export const AI_SYSTEM = `You are a careful statistics teacher reviewing a dataset before a student fits a regression model to it.

You receive a JSON summary: per-column statistics, the correlation of each column with a candidate target, a few raw sample rows, and what a built-in analysis already found. Columns that look like "smoker_yes" are 0/1 indicators made from a text column.

Decide, using only column names that appear in the summary:
- summary: two or three plain sentences saying what the data appears to be (what one row is, what the columns measure).
- target: the numeric column that is most sensible to predict. It must not be a 0/1 indicator.
- predictors: the columns to learn from. Leave out row IDs, near-duplicates of the target, and anything that would leak the answer.
- column_notes: one short note per column (role target, predictor or ignore).
- extra_terms: at most 5 squares of continuous predictors or products of two predictors that have a real-world reason to matter (for example an effect that is much stronger in one group). Use exact column names. Do not invent terms without a reason.
- surface_pair: two terms whose weights are worth drawing as a 3D loss surface. Prefer two that are correlated, because their weights trade off and the surface is visibly stretched. Otherwise pick the two most important.
- warnings: genuine data problems a student should know about (skew, outliers, tiny groups, possible leakage). Empty if there are none.
- reasoning: two sentences on why this model fits this data.

The model is linear in its weights, a weighted sum of terms, fitted by gradient descent on mean squared error. Write for a student: plain words, no filler. Keep every string under 240 characters.`;
