import { Source, Table, sourcesOf } from "./datasets";
import { Prepared, Term, termValue } from "./profile";

export type Inputs = Record<string, number | string>;

export interface Contribution {
  name: string;
  z: number;
  w: number;
  effect: number; // in the units of the response
}

// the original columns the fitted model actually reads
export function usedSources(table: Table, terms: Term[], target: number): Source[] {
  const used = new Set(terms.flatMap((t) => t.parts).map((j) => table.columns[j]));
  const tcol = table.columns[target];
  return sourcesOf(table).filter((s) => !s.columns.includes(tcol) && s.columns.some((c) => used.has(c)));
}

export function defaultInputs(sources: Source[]): Inputs {
  const out: Inputs = {};
  for (const s of sources) out[s.name] = s.kind === "numeric" ? Math.round((s.mean ?? 0) * 100) / 100 : s.baseline ?? "";
  return out;
}

export function encodeRow(table: Table, sources: Source[], values: Inputs): number[] {
  const row = new Array<number>(table.columns.length).fill(0);
  for (const s of sources) {
    if (s.kind === "numeric") row[table.columns.indexOf(s.columns[0])] = Number(values[s.name]);
    else for (const d of s.dummies ?? []) row[table.columns.indexOf(d.column)] = values[s.name] === d.level ? 1 : 0;
  }
  return row;
}

export function predictRow(table: Table, sources: Source[], prepared: Prepared, terms: Term[], theta: Float64Array, values: Inputs) {
  const row = encodeRow(table, sources, values);
  const d = terms.length;
  const contributions: Contribution[] = terms.map((tm, k) => {
    const z = (termValue(row, tm) - prepared.xMean[k]) / prepared.xStd[k];
    return { name: tm.name, z, w: theta[k], effect: theta[k] * z * prepared.yStd };
  });
  const base = prepared.yMean + theta[d] * prepared.yStd;
  return { base, yhat: base + contributions.reduce((s, c) => s + c.effect, 0), contributions };
}
