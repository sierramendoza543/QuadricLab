import { Table, classInfo } from "./datasets";
import type { Task } from "./engine";
import { rng } from "./random";

export interface ColumnStat {
  name: string;
  mean: number;
  std: number;
  min: number;
  max: number;
}

export interface Profile {
  names: string[];
  stats: ColumnStat[];
  corr: number[][];
  observations: string[];
}

// a model term is a product of columns: x, x^2, x*z, ...
export interface Term {
  name: string;
  parts: number[];
}

export interface Prepared {
  task: Task;
  classLabel: string | null;
  names: string[];
  parts: number[][];
  target: string;
  d: number;
  ntr: number;
  nva: number;
  yMean: number;
  yStd: number;
  xMean: number[];
  xStd: number[];
  Xtr: Float64Array;
  ytr: Float64Array;
  Xva: Float64Array;
  yva: Float64Array;
}

const column = (t: Table, j: number) => t.rows.map((r) => r[j]);
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / a.length;
const sd = (a: number[], m: number) => Math.sqrt(a.reduce((s, v) => s + (v - m) * (v - m), 0) / a.length) || 1e-12;

function pearson(a: number[], b: number[]) {
  const ma = mean(a);
  const mb = mean(b);
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return num / (Math.sqrt(da * db) || 1e-12);
}

export function buildTerms(t: Table, feats: number[], degree: number): Term[] {
  const nm = (j: number) => t.columns[j];
  const terms: Term[] = feats.map((j) => ({ name: nm(j), parts: [j] }));
  if (degree >= 2) {
    for (let a = 0; a < feats.length; a++)
      for (let b = a; b < feats.length; b++)
        terms.push({ name: a === b ? `${nm(feats[a])}²` : `${nm(feats[a])}·${nm(feats[b])}`, parts: [feats[a], feats[b]] });
  }
  if (degree >= 3 && feats.length === 1) terms.push({ name: `${nm(feats[0])}³`, parts: [feats[0], feats[0], feats[0]] });
  return terms;
}

export const termValue = (row: number[], term: Term) => term.parts.reduce((p, j) => p * row[j], 1);

const f2 = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

export function profileData(t: Table, target: number, feats: number[], r2Linear: number | null): Profile {
  const idx = [...feats, target];
  const names = idx.map((j) => t.columns[j]);
  const cols = idx.map((j) => column(t, j));
  const stats: ColumnStat[] = cols.map((c, i) => {
    const m = mean(c);
    return { name: names[i], mean: m, std: sd(c, m), min: Math.min(...c), max: Math.max(...c) };
  });
  const corr = cols.map((a) => cols.map((b) => pearson(a, b)));
  const observations: string[] = [];

  const fs = stats.slice(0, feats.length);
  if (fs.length > 1) {
    const big = fs.reduce((a, b) => (b.std > a.std ? b : a));
    const small = fs.reduce((a, b) => (b.std < a.std ? b : a));
    if (big.std / small.std > 5)
      observations.push(
        `${big.name} varies much more than ${small.name} (standard deviation ${f2(big.std)} versus ${f2(small.std)}). Each predictor is standardized before fitting, so the weights are on a comparable scale.`
      );
  }

  let worst = { a: 0, b: 0, r: 0 };
  for (let i = 0; i < feats.length; i++)
    for (let k = i + 1; k < feats.length; k++) if (Math.abs(corr[i][k]) > Math.abs(worst.r)) worst = { a: i, b: k, r: corr[i][k] };
  if (Math.abs(worst.r) > 0.95)
    observations.push(
      `${names[worst.a]} and ${names[worst.b]} are almost perfectly correlated (r = ${worst.r.toFixed(3)}), so they carry the same information. The weights are then not uniquely determined, and the loss surface will show it.`
    );

  const ci = classInfo(t, target);
  if (ci) {
    const share = ci.n1 / (ci.n1 + ci.n0);
    observations.push(
      `${names[feats.length]} has two values, so this is a classification problem. ${Math.round(share * 100)}% of the rows have ${ci.label}, so always guessing the more common class is right ${Math.round(Math.max(share, 1 - share) * 100)}% of the time. A useful model has to beat that.`
    );
  } else if (r2Linear !== null)
    observations.push(
      r2Linear >= 0.9
        ? `A linear model explains about ${Math.round(r2Linear * 100)}% of the variance in ${names[feats.length]}.`
        : `A linear model explains only about ${Math.max(0, Math.round(r2Linear * 100))}% of the variance in ${names[feats.length]}, so the relationship is not linear.`
    );
  if (t.rows.length < 100) observations.push(`The sample is small (${t.rows.length} rows), so expect noisy estimates.`);

  return { names, stats, corr, observations };
}

// the same 80/20 split everywhere, so searching for a model and fitting it agree
export function splitIndices(n: number) {
  const order = Array.from({ length: n }, (_, i) => i);
  const r = rng(7);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const nva = Math.max(2, Math.round(n * 0.2));
  return { tr: order.slice(0, n - nva), va: order.slice(n - nva) };
}

export function prepare(t: Table, target: number, terms: Term[]): Prepared {
  const { tr, va } = splitIndices(t.rows.length);
  const ntr = tr.length;
  const nva = va.length;
  const d = terms.length;

  const vals = (rows: number[]) => terms.map((tm) => rows.map((i) => termValue(t.rows[i], tm)));
  const trVals = vals(tr);
  const xm = trVals.map(mean);
  const xs = trVals.map((c, k) => sd(c, xm[k]));
  const ci = classInfo(t, target);
  const ytrRaw = tr.map((i) => t.rows[i][target]);
  // a classification target stays 0/1; a regression target is standardized
  const ym = ci ? 0 : mean(ytrRaw);
  const ys = ci ? 1 : sd(ytrRaw, ym);

  const build = (ids: number[]) => {
    const v = vals(ids);
    const X = new Float64Array(ids.length * d);
    const y = new Float64Array(ids.length);
    ids.forEach((row, a) => {
      for (let k = 0; k < d; k++) X[a * d + k] = (v[k][a] - xm[k]) / xs[k];
      y[a] = ci ? (t.rows[row][target] === ci.hi ? 1 : 0) : (t.rows[row][target] - ym) / ys;
    });
    return { X, y };
  };
  const trn = build(tr);
  const val = build(va);
  return {
    task: ci ? "classification" : "regression",
    classLabel: ci ? ci.label : null,
    names: terms.map((tm) => tm.name),
    parts: terms.map((tm) => tm.parts),
    target: t.columns[target],
    d,
    ntr,
    nva,
    yMean: ym,
    yStd: ys,
    xMean: xm,
    xStd: xs,
    Xtr: trn.X,
    ytr: trn.y,
    Xva: val.X,
    yva: val.y,
  };
}
