import { Net, Plan, Task, TrainConfig, TrainResult, logLoss, sigmoid } from "./engine";
import { Prepared, Term } from "./profile";
import { Table, classInfo } from "./datasets";
import { Grid, QuadricFit, Slice, computeGrid, fitQuadric, weightSlice } from "./landscape";
import { AiTermSpec, SearchResult, describeSearch, searchTerms } from "./search";

export type Vec2 = [number, number];

export interface ModelChoice {
  kind: "linear" | "polynomial";
  degree: number;
  terms: Term[];
  r2Linear: number | null;
  reason: string;
  cfg: TrainConfig;
  search: SearchResult;
}

export function chooseModel(t: Table, target: number, feats: number[], ai: AiTermSpec[] = []): ModelChoice {
  const search = searchTerms(t, target, feats, ai);
  const degree = Math.max(...search.terms.map((x) => x.parts.length));
  return {
    kind: search.terms.length > search.base ? "polynomial" : "linear",
    degree,
    terms: search.terms,
    r2Linear: search.r2Linear,
    reason: describeSearch(search, t.columns[target], classInfo(t, target)),
    cfg: { optimizer: "adam", lr: 0.05, epochs: classInfo(t, target) ? 500 : 400, seed: 3 },
    search,
  };
}

export interface PairInfo {
  i: number;
  j: number;
  rho: number;
  cond: number;
  kind: "circular" | "elliptic" | "cylinder";
  reason: string;
}

const shapeOf = (rho: number) => {
  const ratio = (1 - Math.abs(rho)) / (1 + Math.abs(rho));
  return ratio > 0.85 ? "circular" : ratio < 0.015 ? "cylinder" : "elliptic";
};

function pairReason(names: string[], i: number, j: number, rho: number, important: boolean) {
  const a = names[i];
  const b = names[j];
  if (Math.abs(rho) > 0.3)
    return `${a} and ${b} are correlated (r = ${rho.toFixed(2)}). Weights on correlated terms trade off against each other, which stretches the bowl into a valley.`;
  return important ? `${a} and ${b} have the largest weights in the fitted model.` : `${a} and ${b}.`;
}

// the pairs of weights worth drawing: both matter, and the more correlated the more interesting the bowl
export function rankPairs(data: Prepared, theta: Float64Array): PairInfo[] {
  const d = data.d;
  if (d < 2) return [];
  const w = Array.from({ length: d }, (_, k) => Math.abs(theta[k]));
  const top = Array.from({ length: d }, (_, k) => k).sort((a, b) => w[b] - w[a]).slice(0, 7);
  const wmax = Math.max(...w, 1e-9);
  const out: (PairInfo & { score: number })[] = [];
  for (let x = 0; x < top.length; x++)
    for (let y = x + 1; y < top.length; y++) {
      const [i, j] = [top[x], top[y]];
      let g = 0;
      for (let r = 0; r < data.ntr; r++) g += data.Xtr[r * d + i] * data.Xtr[r * d + j];
      const rho = g / data.ntr;
      const imp = Math.min(w[i], w[j]) / wmax;
      // powers of one variable (x and x²) are correlated by construction, so they make a dull example
      const nested = new Set([...data.parts[i], ...data.parts[j]]).size === 1;
      const score = (imp >= 0.25 ? Math.abs(rho) + 0.1 * imp : 0.1 * imp) * (nested ? 0.5 : 1);
      out.push({ i, j, rho, cond: (1 + Math.abs(rho)) / Math.max(1 - Math.abs(rho), 1e-9), kind: shapeOf(rho), reason: pairReason(data.names, i, j, rho, true), score });
    }
  out.sort((a, b) => b.score - a.score);
  return out;
}

export interface PathPoint {
  alpha: number;
  beta: number;
  z: number;
}

export interface SampleRow {
  z: number; // the score before the sigmoid (classification)
  y: number;
  yhat: number;
  miss: number;
  miss2: number;
  x1: number;
  x2: number;
}

export interface Lesson {
  task: Task;
  hess: [number, number, number] | null; // exact second derivatives at the minimum, classification only
  n: number;
  names: [string, string];
  frozen: boolean;
  best: Vec2;
  rho: number;
  Lmin: number;
  start: Vec2;
  L0: number;
  rows: SampleRow[];
  h: number;
  L1: number;
  L2: number;
  d1: number;
  d2: number;
  g1: number;
  g2: number;
  exact: Vec2;
  alpha: number;
  next: Vec2;
  Lnext: number;
  steps: { k: number; w1: number; w2: number; L: number }[];
  quad: { A: number; B: number; C: number; D: number; E: number; F: number };
}

export interface MathBundle {
  task: Task;
  slice: Slice;
  grid: Grid;
  fit: QuadricFit;
  lesson: Lesson;
  path: PathPoint[];
  pair: [number, number];
  pairNote: string;
}

interface Dial {
  idx: number;
  name: string;
  col: number; // which model term feeds this weight, -1 for the bias
}

function dialsFor(plan: Plan, names: string[], pair: [number, number] | null, theta: Float64Array): [Dial, Dial] {
  const l = plan.layers[0];
  if (l.inDim >= 2) {
    let [a, b] = pair ?? [-1, -1];
    if (a < 0 || b < 0 || a === b) [a, b] = Array.from({ length: l.inDim }, (_, k) => k).sort((x, y) => Math.abs(theta[l.wOff + y]) - Math.abs(theta[l.wOff + x]));
    return [
      { idx: l.wOff + a, name: names[a], col: a },
      { idx: l.wOff + b, name: names[b], col: b },
    ];
  }
  return [
    { idx: l.wOff, name: names[0], col: 0 },
    { idx: l.bOff, name: "the intercept b", col: -1 },
  ];
}

function subsample(data: Prepared, cap: number) {
  if (data.ntr <= cap) return { X: data.Xtr, y: data.ytr, n: data.ntr };
  const stride = data.ntr / cap;
  const X = new Float64Array(cap * data.d);
  const y = new Float64Array(cap);
  for (let r = 0; r < cap; r++) {
    const src = Math.floor(r * stride);
    for (let k = 0; k < data.d; k++) X[r * data.d + k] = data.Xtr[src * data.d + k];
    y[r] = data.ytr[src];
  }
  return { X, y, n: cap };
}

const ALPHAS = [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005];

export async function buildMath(plan: Plan, data: Prepared, result: TrainResult, cancelled: () => boolean, pairChoice: [number, number] | null = null): Promise<MathBundle | null> {
  const task = data.task;
  const net = new Net(plan, task);
  const theta = result.theta;
  const { X, y, n } = subsample(data, 1500);
  const ranked = rankPairs(data, theta);
  const top2 = ranked.length && Math.abs(ranked[0].rho) > 0.3 ? ([ranked[0].i, ranked[0].j] as [number, number]) : null;
  const dials = dialsFor(plan, data.names, pairChoice ?? top2, theta);
  const [i, j] = [dials[0].idx, dials[1].idx];
  const slice = weightSlice(theta, i, j);

  const grid = await computeGrid(net, theta, slice, X, y, n, 41, () => {}, cancelled);
  if (!grid) return null;
  const fit = fitQuadric(grid);
  if (!fit) return null;

  const tmp = new Float64Array(theta.length);
  const lossAt = (a: number, b: number) => {
    tmp.set(theta);
    tmp[i] = a;
    tmp[j] = b;
    return net.loss(tmp, X, y, n);
  };
  const gradAt = (a: number, b: number, e = 1e-5): Vec2 => [
    (lossAt(a + e, b) - lossAt(a - e, b)) / (2 * e),
    (lossAt(a, b + e) - lossAt(a, b - e)) / (2 * e),
  ];

  const best: Vec2 = [theta[i], theta[j]];
  const r = slice.range;
  // a starting guess uphill from the minimum, on neither principal axis, so both slopes are nonzero
  const [vc, vs] = [Math.cos(fit.angle), Math.sin(fit.angle)];
  const start: Vec2 = [best[0] + 0.5 * r * (0.8 * vc - 0.6 * vs), best[1] + 0.5 * r * (0.8 * vs + 0.6 * vc)];
  const L0 = lossAt(start[0], start[1]);

  tmp.set(theta);
  tmp[i] = start[0];
  tmp[j] = start[1];
  const pred = Float64Array.from(net.forward(tmp, X, n));
  const input = (k: number, col: number) => (col === -1 ? 1 : X[k * data.d + col]);
  const cls = task === "classification";
  // what the model predicts for a row: a value for regression, a probability for classification
  const out = (k: number) => (cls ? sigmoid(pred[k]) : pred[k]);
  const rows: SampleRow[] = [0, 1, 2].map((k) => ({
    z: pred[k],
    y: y[k],
    yhat: out(k),
    miss: y[k] - out(k),
    miss2: cls ? logLoss(pred[k], y[k]) : (y[k] - pred[k]) ** 2,
    x1: input(k, dials[0].col),
    x2: input(k, dials[1].col),
  }));

  let s1 = 0;
  let s2 = 0;
  for (let k = 0; k < n; k++) {
    s1 += (out(k) - y[k]) * input(k, dials[0].col);
    s2 += (out(k) - y[k]) * input(k, dials[1].col);
  }
  const scale = cls ? 1 / n : 2 / n;
  const exact: Vec2 = [scale * s1, scale * s2];

  // classification: exact curvature of the log-loss at the fitted weights, H = (1/N) sum p(1-p) x x^T
  let hess: Lesson["hess"] = null;
  if (cls) {
    const atBest = Float64Array.from(net.forward(theta, X, n));
    let h11 = 0;
    let h12 = 0;
    let h22 = 0;
    for (let k = 0; k < n; k++) {
      const p = sigmoid(atBest[k]);
      const a = input(k, dials[0].col);
      const c = input(k, dials[1].col);
      h11 += p * (1 - p) * a * a;
      h12 += p * (1 - p) * a * c;
      h22 += p * (1 - p) * c * c;
    }
    hess = [h11 / n, h12 / n, h22 / n];
  }

  const h = 0.01;
  const L1 = lossAt(start[0] + h, start[1]);
  const L2 = lossAt(start[0], start[1] + h);
  const [g1, g2] = gradAt(start[0], start[1]);

  const curvature = Math.max(fit.mu1, 1e-6);
  const alpha = ALPHAS.find((a) => a * curvature <= 0.5) ?? 0.005;
  const next: Vec2 = [start[0] - alpha * g1, start[1] - alpha * g2];

  // gradient descent from the starting guess, so the table and the 3D path match the algebra
  const path: PathPoint[] = [];
  const steps: Lesson["steps"] = [];
  let w: Vec2 = [...start];
  for (let k = 0; k <= 120; k++) {
    const L = lossAt(w[0], w[1]);
    if (Math.abs(w[0] - best[0]) > slice.range || Math.abs(w[1] - best[1]) > slice.range) break;
    path.push({ alpha: w[0] - best[0], beta: w[1] - best[1], z: L });
    steps.push({ k, w1: w[0], w2: w[1], L });
    const g = gradAt(w[0], w[1]);
    w = [w[0] - alpha * g[0], w[1] - alpha * g[1]];
  }

  // absolute coefficients of L = A w1^2 + B w1 w2 + C w2^2 + D w1 + E w2 + F
  const [c0, ca, cb, caa, cab, cbb] = fit.coef;
  const [b1, b2] = best;
  const quad = {
    A: caa,
    B: cab,
    C: cbb,
    D: ca - 2 * caa * b1 - cab * b2,
    E: cb - cab * b1 - 2 * cbb * b2,
    F: c0 - ca * b1 - cb * b2 + caa * b1 * b1 + cab * b1 * b2 + cbb * b2 * b2,
  };

  let rho = 0;
  if (dials[0].col >= 0 && dials[1].col >= 0) {
    let g = 0;
    for (let r = 0; r < data.ntr; r++) g += data.Xtr[r * data.d + dials[0].col] * data.Xtr[r * data.d + dials[1].col];
    rho = g / data.ntr;
  }
  const lesson: Lesson = {
    task,
    hess,
    n,
    rho,
    names: [dials[0].name, dials[1].name],
    frozen: theta.length > 2,
    best,
    Lmin: lossAt(best[0], best[1]),
    start,
    L0,
    rows,
    h,
    L1,
    L2,
    d1: (L1 - L0) / h,
    d2: (L2 - L0) / h,
    g1,
    g2,
    exact,
    alpha,
    next,
    Lnext: lossAt(next[0], next[1]),
    steps,
    quad,
  };
  const pair: [number, number] = [dials[0].col, dials[1].col];
  const pairNote =
    dials[1].col === -1
      ? `The model has one predictor, so we draw its weight against the intercept.`
      : pairChoice
        ? pairReason(data.names, pair[0], pair[1], rho, false)
        : top2
          ? pairReason(data.names, pair[0], pair[1], rho, true)
          : `${data.names[pair[0]]} and ${data.names[pair[1]]} have the largest weights in the fitted model.`;
  return { task, slice, grid, fit, lesson, path, pair, pairNote };
}
