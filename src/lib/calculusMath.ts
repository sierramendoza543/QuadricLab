export type SurfaceKind = "circular" | "elliptic" | "cylinder" | "saddle";
export type Mode = "regression" | "minimax";
export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

export interface Params {
  ratio: number; // sigma1 / sigma2
  rho: number;
  mode: Mode;
}

export interface Dataset {
  n: number;
  x1: number[];
  x2: number[];
  y: number[];
}

export interface Model {
  params: Params;
  data: Dataset;
  p: number; // H = [[p, q], [q, s]]
  q: number;
  s: number;
  bvec: Vec2;
  lam1: number;
  lam2: number;
  theta: number; // angle of the eigenvector for lam1
  center: Vec2;
  kind: SurfaceKind;
  semiA: number;
  semiB: number;
  noiseFloor: number;
  zMin: number;
  zMax: number;
}

export type AgentKind = "gd" | "momentum";
export type AgentStatus = "running" | "converged" | "escaped";

export interface Agent {
  id: AgentKind;
  pos: Vec2;
  vel: Vec2;
  step: Vec2 | null;
  path: Vec2[];
  iter: number;
  status: AgentStatus;
}

export const DOMAIN = 3.5;
export const TRUE_WEIGHTS: Vec2 = [1.2, -0.85];
const N_POINTS = 20;
const MAX_ITER = 4000;
const MAX_PATH = 500;

function rng(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand: () => number) {
  const u = Math.max(rand(), 1e-12);
  const v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const mean = (a: number[]) => a.reduce((t, v) => t + v, 0) / a.length;
const dotN = (a: number[], b: number[]) => a.reduce((t, v, i) => t + v * b[i], 0) / a.length;

// whitened basis is generated once so the sample covariance of X is exactly
// the requested one, which keeps the surface classification clean
let basisCache: { z1: number[]; z2: number[]; e: number[] } | null = null;

function basis() {
  if (basisCache) return basisCache;
  const rand = rng(20240917);
  const z1 = Array.from({ length: N_POINTS }, () => gaussian(rand));
  const z2 = Array.from({ length: N_POINTS }, () => gaussian(rand));
  const e = Array.from({ length: N_POINTS }, () => gaussian(rand));

  const center = (v: number[]) => {
    const m = mean(v);
    return v.map((x) => x - m);
  };
  let a = center(z1);
  let b = center(z2);
  let r = center(e);

  const sa = Math.sqrt(dotN(a, a));
  a = a.map((x) => x / sa);
  const proj = dotN(b, a);
  b = b.map((x, i) => x - proj * a[i]);
  const sb = Math.sqrt(dotN(b, b));
  b = b.map((x) => x / sb);

  const ra = dotN(r, a);
  const rb = dotN(r, b);
  r = r.map((x, i) => x - ra * a[i] - rb * b[i]);
  const sr = Math.sqrt(dotN(r, r));
  r = r.map((x) => (0.4 * x) / sr);

  basisCache = { z1: a, z2: b, e: r };
  return basisCache;
}

export function makeDataset(ratio: number, rho: number): Dataset {
  const { z1, z2, e } = basis();
  const s1 = Math.sqrt(ratio);
  const s2 = 1 / Math.sqrt(ratio);
  const k = Math.sqrt(Math.max(0, 1 - rho * rho));
  const x1 = z1.map((v) => s1 * v);
  const x2 = z1.map((v, i) => s2 * (rho * v + k * z2[i]));
  const y = x1.map((v, i) => TRUE_WEIGHTS[0] * v + TRUE_WEIGHTS[1] * x2[i] + e[i]);
  return { n: N_POINTS, x1, x2, y };
}

export function mse(data: Dataset, w: Vec2) {
  let t = 0;
  for (let i = 0; i < data.n; i++) {
    const r = w[0] * data.x1[i] + w[1] * data.x2[i] - data.y[i];
    t += r * r;
  }
  return t / data.n;
}

export function buildModel(params: Params): Model {
  const data = makeDataset(params.ratio, params.rho);
  const n = data.n;
  let p = 0;
  let q = 0;
  let s = 0;
  let b1 = 0;
  let b2 = 0;
  for (let i = 0; i < n; i++) {
    p += data.x1[i] * data.x1[i];
    q += data.x1[i] * data.x2[i];
    s += data.x2[i] * data.x2[i];
    b1 += data.x1[i] * data.y[i];
    b2 += data.x2[i] * data.y[i];
  }
  p /= n;
  q /= n;
  s /= n;
  b1 /= n;
  b2 /= n;

  const m = (p + s) / 2;
  const d = Math.hypot((p - s) / 2, q);
  const lam1 = m + d;
  let lam2 = m - d;
  if (lam2 < 1e-9 * lam1) lam2 = 0;
  const theta = 0.5 * Math.atan2(2 * q, p - s);
  const v1: Vec2 = [Math.cos(theta), Math.sin(theta)];
  const v2: Vec2 = [-Math.sin(theta), Math.cos(theta)];

  // minimum-norm minimiser, works when H is singular
  let h = 0;
  let k = 0;
  const c1 = (v1[0] * b1 + v1[1] * b2) / lam1;
  h += c1 * v1[0];
  k += c1 * v1[1];
  if (lam2 > 0) {
    const c2 = (v2[0] * b1 + v2[1] * b2) / lam2;
    h += c2 * v2[0];
    k += c2 * v2[1];
  }

  const ratio = lam2 / lam1;
  let kind: SurfaceKind;
  if (params.mode === "minimax") kind = "saddle";
  else if (ratio < 0.01) kind = "cylinder";
  else if (ratio > 0.97) kind = "circular";
  else kind = "elliptic";

  const lamBar = (lam1 + lam2) / 2;
  const semiA = kind === "circular" ? 1 / Math.sqrt(lamBar) : 1 / Math.sqrt(lam1);
  const semiB = kind === "circular" ? semiA : 1 / Math.sqrt(Math.max(lam2, 1e-3));

  const model: Model = {
    params,
    data,
    p,
    q,
    s,
    bvec: [b1, b2],
    lam1,
    lam2,
    theta,
    center: [h, k],
    kind,
    semiA,
    semiB,
    noiseFloor: mse(data, [h, k]),
    zMin: 0,
    zMax: 1,
  };

  let lo = Infinity;
  let hi = -Infinity;
  const steps = 36;
  for (let i = 0; i <= steps; i++) {
    for (let j = 0; j <= steps; j++) {
      const z = surfaceValue(model, -DOMAIN + (2 * DOMAIN * i) / steps, -DOMAIN + (2 * DOMAIN * j) / steps);
      if (z < lo) lo = z;
      if (z > hi) hi = z;
    }
  }
  model.zMin = lo;
  model.zMax = hi;
  return model;
}

function rotated(m: Model, w1: number, w2: number): Vec2 {
  const dx = w1 - m.center[0];
  const dy = w2 - m.center[1];
  const c = Math.cos(m.theta);
  const s = Math.sin(m.theta);
  return [dx * c + dy * s, -dx * s + dy * c];
}

export function surfaceValue(m: Model, w1: number, w2: number) {
  const [u1, u2] = rotated(m, w1, w2);
  if (m.kind === "saddle") return m.lam2 * u2 * u2 - m.lam1 * u1 * u1;
  return m.lam1 * u1 * u1 + m.lam2 * u2 * u2;
}

export function gradient(m: Model, w1: number, w2: number): Vec2 {
  const [u1, u2] = rotated(m, w1, w2);
  const c = Math.cos(m.theta);
  const s = Math.sin(m.theta);
  const g1 = m.kind === "saddle" ? -2 * m.lam1 * u1 : 2 * m.lam1 * u1;
  const g2 = 2 * m.lam2 * u2;
  return [g1 * c - g2 * s, g1 * s + g2 * c];
}

// plain descent in regression mode; gradient descent-ascent in minimax mode,
// where the max player moves uphill along the first eigendirection
export function updateDirection(m: Model, g: Vec2): Vec2 {
  if (m.kind !== "saddle") return [-g[0], -g[1]];
  const c = Math.cos(m.theta);
  const s = Math.sin(m.theta);
  const gu1 = g[0] * c + g[1] * s;
  const gu2 = -g[0] * s + g[1] * c;
  return [gu1 * c + gu2 * s, gu1 * s - gu2 * c];
}

export function surfaceNormal(g: Vec2): Vec3 {
  const len = Math.hypot(g[0], g[1], 1);
  return [-g[0] / len, -g[1] / len, 1 / len];
}

export function norm(v: Vec2) {
  return Math.hypot(v[0], v[1]);
}

export function angleBetween(a: Vec2, b: Vec2) {
  const na = norm(a);
  const nb = norm(b);
  if (na < 1e-12 || nb < 1e-12) return null;
  const c = Math.max(-1, Math.min(1, (a[0] * b[0] + a[1] * b[1]) / (na * nb)));
  return (Math.acos(c) * 180) / Math.PI;
}

export function newAgent(id: AgentKind, start: Vec2): Agent {
  return { id, pos: start, vel: [0, 0], step: null, path: [start], iter: 0, status: "running" };
}

export function proposedStep(m: Model, agent: Agent, alpha: number, beta: number): Vec2 {
  const g = gradient(m, agent.pos[0], agent.pos[1]);
  const dir = updateDirection(m, g);
  if (agent.id === "gd") return [alpha * dir[0], alpha * dir[1]];
  return [beta * agent.vel[0] + alpha * dir[0], beta * agent.vel[1] + alpha * dir[1]];
}

export function stepAgent(m: Model, agent: Agent, alpha: number, beta: number): Agent {
  if (agent.status !== "running") return agent;
  const step = proposedStep(m, agent, alpha, beta);
  const pos: Vec2 = [agent.pos[0] + step[0], agent.pos[1] + step[1]];
  const path = agent.path.length >= MAX_PATH ? agent.path.slice(1) : agent.path.slice();
  path.push(pos);
  const iter = agent.iter + 1;

  let status: AgentStatus = "running";
  if (!Number.isFinite(pos[0]) || Math.abs(pos[0]) > DOMAIN || Math.abs(pos[1]) > DOMAIN) status = "escaped";
  else {
    const g = gradient(m, pos[0], pos[1]);
    if ((norm(g) < 1e-3 && norm(step) < 1e-3) || iter >= MAX_ITER) status = "converged";
  }
  return { id: agent.id, pos, vel: agent.id === "momentum" ? step : [0, 0], step, path, iter, status };
}

export function traceCurve(m: Model, axis: "x" | "y", k: number, n = 120): Vec3[] {
  const pts: Vec3[] = [];
  for (let i = 0; i <= n; i++) {
    const t = -DOMAIN + (2 * DOMAIN * i) / n;
    const w1 = axis === "x" ? k : t;
    const w2 = axis === "x" ? t : k;
    pts.push([w1, w2, surfaceValue(m, w1, w2)]);
  }
  return pts;
}

// marching squares on f(w1, w2) = level
export function levelSegments(m: Model, level: number, n = 90): [Vec2, Vec2][] {
  const out: [Vec2, Vec2][] = [];
  const h = (2 * DOMAIN) / n;
  const f = (i: number, j: number) => surfaceValue(m, -DOMAIN + i * h, -DOMAIN + j * h) - level;
  const grid: number[][] = [];
  for (let i = 0; i <= n; i++) {
    grid.push([]);
    for (let j = 0; j <= n; j++) grid[i].push(f(i, j));
  }
  const lerp = (a: number, b: number) => a / (a - b);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v00 = grid[i][j];
      const v10 = grid[i + 1][j];
      const v11 = grid[i + 1][j + 1];
      const v01 = grid[i][j + 1];
      const x0 = -DOMAIN + i * h;
      const y0 = -DOMAIN + j * h;
      const pts: Vec2[] = [];
      if (v00 * v10 < 0) pts.push([x0 + lerp(v00, v10) * h, y0]);
      if (v10 * v11 < 0) pts.push([x0 + h, y0 + lerp(v10, v11) * h]);
      if (v01 * v11 < 0) pts.push([x0 + lerp(v01, v11) * h, y0 + h]);
      if (v00 * v01 < 0) pts.push([x0, y0 + lerp(v00, v01) * h]);
      if (pts.length === 2) out.push([pts[0], pts[1]]);
      else if (pts.length === 4) {
        out.push([pts[0], pts[1]]);
        out.push([pts[2], pts[3]]);
      }
    }
  }
  return out;
}

// along x = k (axis "x") or y = k (axis "y") the surface is a quadratic in the free variable
export function traceQuadratic(m: Model, axis: "x" | "y", k: number) {
  const at = (t: number) => (axis === "x" ? surfaceValue(m, k, t) : surfaceValue(m, t, k));
  const c = at(0);
  const fp = at(1);
  const fm = at(-1);
  return { A: (fp + fm) / 2 - c, B: (fp - fm) / 2, C: c };
}

export function fitPredictions(m: Model, w: Vec2) {
  return m.data.x1.map((v, i) => w[0] * v + w[1] * m.data.x2[i]);
}

export const PRESETS = [
  {
    id: "normalized",
    title: "Normalized features",
    note: "Equal scale, uncorrelated",
    result: "Circular paraboloid",
    params: { ratio: 1, rho: 0, mode: "regression" as Mode },
    alpha: 0.2,
    beta: 0.8,
  },
  {
    id: "unscaled",
    title: "Unscaled units (sq ft vs. bedrooms)",
    note: "One feature dominates the other",
    result: "Elliptic paraboloid",
    params: { ratio: 4, rho: 0.2, mode: "regression" as Mode },
    alpha: 0.2,
    beta: 0.9,
  },
  {
    id: "redundant",
    title: "Redundant features (Celsius and Fahrenheit)",
    note: "Perfectly collinear columns",
    result: "Parabolic cylinder",
    params: { ratio: 1, rho: 1, mode: "regression" as Mode },
    alpha: 0.15,
    beta: 0.8,
  },
  {
    id: "gan",
    title: "Adversarial GAN minimax",
    note: "Generator vs. discriminator",
    result: "Hyperbolic paraboloid",
    params: { ratio: 1.5, rho: 0.3, mode: "minimax" as Mode },
    alpha: 0.1,
    beta: 0.5,
  },
];

export function zScaleFor(m: Model) {
  const span = Math.max(Math.abs(m.zMax), Math.abs(m.zMin), 1e-6);
  return Math.min(0.5, 4.5 / span);
}
