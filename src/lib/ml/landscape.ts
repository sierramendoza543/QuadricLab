import { Net } from "./engine";
import { solve } from "./linalg";

export interface Slice {
  a: Float64Array;
  b: Float64Array;
  range: number;
}

export interface Grid {
  n: number;
  range: number;
  z: Float64Array; // index iy * n + ix, alpha along ix, beta along iy
  zMin: number;
  zMax: number;
}

const dot = (a: Float64Array, b: Float64Array) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

export function weightSlice(theta: Float64Array, i: number, j: number): Slice {
  const a = new Float64Array(theta.length);
  const b = new Float64Array(theta.length);
  a[i] = 1;
  b[j] = 1;
  const range = 1.5 * Math.max(Math.abs(theta[i]), Math.abs(theta[j]), 1);
  return { a, b, range };
}

export async function computeGrid(
  net: Net,
  theta: Float64Array,
  slice: Slice,
  X: Float64Array,
  y: Float64Array,
  n: number,
  N: number,
  onProgress: (f: number) => void,
  cancelled: () => boolean
): Promise<Grid | null> {
  const z = new Float64Array(N * N);
  const tmp = new Float64Array(theta.length);
  let lo = Infinity;
  let hi = -Infinity;
  let clock = performance.now();
  for (let iy = 0; iy < N; iy++) {
    const beta = -slice.range + (2 * slice.range * iy) / (N - 1);
    for (let ix = 0; ix < N; ix++) {
      const alpha = -slice.range + (2 * slice.range * ix) / (N - 1);
      for (let k = 0; k < tmp.length; k++) tmp[k] = theta[k] + alpha * slice.a[k] + beta * slice.b[k];
      const v = net.loss(tmp, X, y, n);
      z[iy * N + ix] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (performance.now() - clock > 16) {
      onProgress((iy + 1) / N);
      await new Promise((r) => setTimeout(r, 0));
      clock = performance.now();
      if (cancelled()) return null;
    }
  }
  onProgress(1);
  return { n: N, range: slice.range, z, zMin: lo, zMax: hi };
}

export function sampleGrid(g: Grid, alpha: number, beta: number): number | null {
  const fx = ((alpha + g.range) / (2 * g.range)) * (g.n - 1);
  const fy = ((beta + g.range) / (2 * g.range)) * (g.n - 1);
  if (fx < 0 || fy < 0 || fx > g.n - 1 || fy > g.n - 1) return null;
  const x0 = Math.min(Math.floor(fx), g.n - 2);
  const y0 = Math.min(Math.floor(fy), g.n - 2);
  const tx = fx - x0;
  const ty = fy - y0;
  const at = (x: number, y: number) => g.z[y * g.n + x];
  return (
    at(x0, y0) * (1 - tx) * (1 - ty) + at(x0 + 1, y0) * tx * (1 - ty) + at(x0, y0 + 1) * (1 - tx) * ty + at(x0 + 1, y0 + 1) * tx * ty
  );
}

export type FitKind = "circular" | "elliptic" | "cylinder" | "saddle" | "inverted" | "flat";

export interface QuadricFit {
  kind: FitKind;
  r2: number;
  mu1: number;
  mu2: number;
  angle: number;
  center: [number, number]; // in eigen coordinates
  zVertex: number;
  semiA: number;
  semiB: number;
  condition: number;
  coef: [number, number, number, number, number, number];
}

export const FIT_LABEL: Record<FitKind, string> = {
  circular: "Circular paraboloid",
  elliptic: "Elliptic paraboloid",
  cylinder: "Parabolic cylinder",
  saddle: "Hyperbolic paraboloid",
  inverted: "Inverted paraboloid",
  flat: "Nearly flat plane",
};

export function quadValue(f: QuadricFit, alpha: number, beta: number) {
  const [c0, ca, cb, caa, cab, cbb] = f.coef;
  return c0 + ca * alpha + cb * beta + caa * alpha * alpha + cab * alpha * beta + cbb * beta * beta;
}

// least squares quadratic over the central window of the slice
export function fitQuadric(g: Grid, frac = 0.5): QuadricFit | null {
  const R = g.range;
  const lim = R * frac;
  const basis = (x: number, y: number) => [1, x, y, x * x, x * y, y * y];
  const A = Array.from({ length: 6 }, () => new Array<number>(6).fill(0));
  const b = new Array<number>(6).fill(0);
  const pts: [number, number, number][] = [];
  for (let iy = 0; iy < g.n; iy++) {
    for (let ix = 0; ix < g.n; ix++) {
      const al = -R + (2 * R * ix) / (g.n - 1);
      const be = -R + (2 * R * iy) / (g.n - 1);
      if (Math.abs(al) > lim || Math.abs(be) > lim) continue;
      const z = g.z[iy * g.n + ix];
      pts.push([al / R, be / R, z]);
      const f = basis(al / R, be / R);
      for (let i = 0; i < 6; i++) {
        b[i] += f[i] * z;
        for (let k = 0; k < 6; k++) A[i][k] += f[i] * f[k];
      }
    }
  }
  const c = solve(A, b);
  if (!c) return null;

  let mean = 0;
  for (const p of pts) mean += p[2];
  mean /= pts.length;
  let sse = 0;
  let sst = 0;
  for (const [x, y, z] of pts) {
    const f = basis(x, y);
    const pred = f.reduce((s, v, i) => s + v * c[i], 0);
    sse += (z - pred) ** 2;
    sst += (z - mean) ** 2;
  }
  const r2 = sst > 1e-18 ? 1 - sse / sst : 1;

  const coef: QuadricFit["coef"] = [c[0], c[1] / R, c[2] / R, c[3] / (R * R), c[4] / (R * R), c[5] / (R * R)];
  const h11 = 2 * coef[3];
  const h22 = 2 * coef[5];
  const h12 = coef[4];
  const m = (h11 + h22) / 2;
  const d = Math.hypot((h11 - h22) / 2, h12);
  const mu1 = m + d;
  const mu2 = m - d;
  const angle = 0.5 * Math.atan2(2 * h12, h11 - h22);
  const cs = Math.cos(angle);
  const sn = Math.sin(angle);
  const g1 = coef[1] * cs + coef[2] * sn;
  const g2 = -coef[1] * sn + coef[2] * cs;

  const mx = Math.max(Math.abs(mu1), Math.abs(mu2));
  const tol = 0.05 * mx;
  let kind: FitKind;
  if (mx < 1e-12) kind = "flat";
  else if (mu2 > tol) kind = mu2 / mu1 > 0.85 ? "circular" : "elliptic";
  else if (Math.abs(mu2) <= tol && mu1 > tol) kind = "cylinder";
  else if (mu1 > tol && mu2 < -tol) kind = "saddle";
  else kind = "inverted";

  const c1 = Math.abs(mu1) > tol ? -g1 / mu1 : 0;
  const c2 = Math.abs(mu2) > tol ? -g2 / mu2 : 0;
  let zVertex = coef[0];
  if (Math.abs(mu1) > tol) zVertex += g1 * c1 + 0.5 * mu1 * c1 * c1;
  if (Math.abs(mu2) > tol) zVertex += g2 * c2 + 0.5 * mu2 * c2 * c2;

  return {
    kind,
    r2,
    mu1,
    mu2,
    angle,
    center: [c1, c2],
    zVertex,
    semiA: Math.abs(mu1) > 1e-12 ? Math.sqrt(2 / Math.abs(mu1)) : Infinity,
    semiB: Math.abs(mu2) > 1e-12 ? Math.sqrt(2 / Math.abs(mu2)) : Infinity,
    condition: Math.abs(mu2) > 1e-12 ? Math.abs(mu1 / mu2) : Infinity,
    coef,
  };
}

export function fitQuality(r2: number) {
  if (r2 >= 0.95) return { label: "strong match", tone: "#34d399" };
  if (r2 >= 0.8) return { label: "rough match", tone: "#fbbf24" };
  return { label: "weak match", tone: "#fb7185" };
}

export function fitEquation(f: QuadricFit) {
  const n = (x: number) => x.toFixed(3);
  const z0 = n(f.zVertex);
  const a = n(f.semiA);
  const b = n(f.semiB);
  switch (f.kind) {
    case "circular":
    case "elliptic":
      return `z - ${z0} = \\frac{u_1^2}{${a}^2} + \\frac{u_2^2}{${b}^2}`;
    case "cylinder":
      return `z - ${z0} = \\frac{u_1^2}{${a}^2}`;
    case "saddle":
      return `z - ${z0} = \\frac{u_1^2}{${a}^2} - \\frac{u_2^2}{${b}^2}`;
    case "inverted":
      return Number.isFinite(f.semiB)
        ? `z - ${z0} = -\\frac{u_1^2}{${a}^2} - \\frac{u_2^2}{${b}^2}`
        : `z - ${z0} = -\\frac{u_1^2}{${a}^2}`;
    default:
      return `z \\approx ${z0}`;
  }
}

// what the shape means for minimizing the loss
export function shapeMeaning(f: QuadricFit): string {
  switch (f.kind) {
    case "circular":
      return "The surface curves equally in every direction, so the steepest-descent direction points straight at the minimum.";
    case "elliptic":
      return `The surface is about ${Number.isFinite(f.condition) ? f.condition.toFixed(1) : "much"} times steeper along one axis than the other. Steepest descent then tends to zig-zag across the steep direction instead of heading straight for the minimum.`;
    case "cylinder":
      return "The surface does not curve along one direction, so there is a whole line of weights with the same minimum loss. The weights are not uniquely determined.";
    case "saddle":
      return "The surface curves up in one direction and down in the other, so there is no minimum in this slice.";
    case "inverted":
      return "The surface curves downward, so this slice has a maximum rather than a minimum.";
    default:
      return "The loss barely changes here, so the surface is nearly a plane.";
  }
}
