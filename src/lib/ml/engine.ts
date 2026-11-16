import { gaussian, rng } from "./random";
import { Prepared } from "./profile";

export type Task = "regression" | "classification";

export type Activation = "relu" | "tanh" | "sigmoid";

export type Block =
  | { id: number; kind: "dense"; units: number }
  | { id: number; kind: "act"; fn: Activation };

export interface Layer {
  inDim: number;
  outDim: number;
  act: Activation | null;
  wOff: number;
  bOff: number;
}

export interface Plan {
  layers: Layer[];
  nParams: number;
  warnings: string[];
}

export function planNetwork(blocks: Block[], d: number): Plan {
  const layers: Layer[] = [];
  const warnings: string[] = [];
  let width = d;
  let off = 0;
  let openDense = false;

  const push = (units: number) => {
    layers.push({ inDim: width, outDim: units, act: null, wOff: off, bOff: off + width * units });
    off += width * units + units;
    width = units;
  };

  for (const b of blocks) {
    if (b.kind === "dense") {
      if (openDense) warnings.push("Two Dense layers in a row with no activation between them collapse into one linear map.");
      push(b.units);
      openDense = true;
    } else {
      const last = layers[layers.length - 1];
      if (!last) warnings.push("An activation needs a Dense layer in front of it, so it was skipped.");
      else if (last.act) warnings.push("Two activations in a row: only the first one is used.");
      else {
        last.act = b.fn;
        openDense = false;
      }
    }
  }
  if (openDense) warnings.push("The last hidden Dense layer has no activation, so it merges with the output layer.");
  push(1);
  return { layers, nParams: off, warnings: Array.from(new Set(warnings)) };
}

export function initParams(plan: Plan, seed: number): Float64Array {
  const r = rng(seed);
  const theta = new Float64Array(plan.nParams);
  for (const l of plan.layers) {
    const scale = Math.sqrt((l.act === "relu" ? 2 : 1) / l.inDim);
    for (let i = 0; i < l.inDim * l.outDim; i++) theta[l.wOff + i] = gaussian(r) * scale;
  }
  return theta;
}

const act = {
  relu: (s: number) => (s > 0 ? s : 0),
  tanh: Math.tanh,
  sigmoid: (s: number) => 1 / (1 + Math.exp(-s)),
};

const sigmoid = (z: number) => (z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)));
// cross-entropy of one row, written so large scores do not overflow
export const logLoss = (z: number, y: number) => Math.max(z, 0) - y * z + Math.log1p(Math.exp(-Math.abs(z)));
export { sigmoid };

const slope = {
  relu: (a: number) => (a > 0 ? 1 : 0),
  tanh: (a: number) => 1 - a * a,
  sigmoid: (a: number) => a * (1 - a),
};

export class Net {
  private buffers = new Map<number, Float64Array[]>();

  constructor(public plan: Plan, public task: Task = "regression") {}

  private acts(n: number) {
    let b = this.buffers.get(n);
    if (!b) {
      b = this.plan.layers.map((l) => new Float64Array(n * l.outDim));
      this.buffers.set(n, b);
    }
    return b;
  }

  forward(theta: Float64Array, X: Float64Array, n: number): Float64Array {
    const bufs = this.acts(n);
    let input = X;
    this.plan.layers.forEach((l, li) => {
      const out = bufs[li];
      const f = l.act ? act[l.act] : null;
      for (let r = 0; r < n; r++) {
        const xr = r * l.inDim;
        for (let o = 0; o < l.outDim; o++) {
          const w = l.wOff + o * l.inDim;
          let s = theta[l.bOff + o];
          for (let i = 0; i < l.inDim; i++) s += theta[w + i] * input[xr + i];
          out[r * l.outDim + o] = f ? f(s) : s;
        }
      }
      input = out;
    });
    return bufs[bufs.length - 1];
  }

  loss(theta: Float64Array, X: Float64Array, y: Float64Array, n: number): number {
    const out = this.forward(theta, X, n);
    let s = 0;
    if (this.task === "classification") {
      for (let r = 0; r < n; r++) s += logLoss(out[r], y[r]);
      return s / n;
    }
    for (let r = 0; r < n; r++) {
      const e = out[r] - y[r];
      s += e * e;
    }
    return s / n;
  }

  lossAndGrad(theta: Float64Array, X: Float64Array, y: Float64Array, n: number, grad: Float64Array): number {
    const out = this.forward(theta, X, n);
    const bufs = this.acts(n);
    grad.fill(0);
    let loss = 0;
    let da = new Float64Array(n);
    if (this.task === "classification") {
      for (let r = 0; r < n; r++) {
        loss += logLoss(out[r], y[r]);
        da[r] = (sigmoid(out[r]) - y[r]) / n;
      }
    } else {
      for (let r = 0; r < n; r++) {
        const e = out[r] - y[r];
        loss += e * e;
        da[r] = (2 * e) / n;
      }
    }
    for (let li = this.plan.layers.length - 1; li >= 0; li--) {
      const l = this.plan.layers[li];
      const a = bufs[li];
      const input = li > 0 ? bufs[li - 1] : X;
      if (l.act) {
        const d = slope[l.act];
        for (let k = 0; k < n * l.outDim; k++) da[k] *= d(a[k]);
      }
      const prev = li > 0 ? new Float64Array(n * l.inDim) : null;
      for (let r = 0; r < n; r++) {
        const xr = r * l.inDim;
        for (let o = 0; o < l.outDim; o++) {
          const dz = da[r * l.outDim + o];
          if (dz === 0) continue;
          grad[l.bOff + o] += dz;
          const w = l.wOff + o * l.inDim;
          for (let i = 0; i < l.inDim; i++) {
            grad[w + i] += dz * input[xr + i];
            if (prev) prev[xr + i] += dz * theta[w + i];
          }
        }
      }
      if (prev) da = prev;
    }
    return loss / n;
  }
}

export type OptimizerKind = "sgd" | "momentum" | "adam";

export interface TrainConfig {
  optimizer: OptimizerKind;
  lr: number;
  epochs: number;
  seed: number;
}

export interface TrainResult {
  theta: Float64Array;
  start: Float64Array;
  checkpoints: Float64Array[];
  trainLoss: number[];
  valLoss: number[];
  diverged: boolean;
  r2Train: number;
  r2Val: number;
  cls: { accTrain: number; accVal: number; baseVal: number; lossTrain: number; lossVal: number } | null;
}

const variance = (y: Float64Array) => {
  let m = 0;
  for (const v of y) m += v;
  m /= y.length;
  let s = 0;
  for (const v of y) s += (v - m) * (v - m);
  return s / y.length || 1;
};

export async function train(
  net: Net,
  data: Prepared,
  cfg: TrainConfig,
  onProgress: (epoch: number, train: number[], val: number[]) => void,
  cancelled: () => boolean
): Promise<TrainResult> {
  const theta = initParams(net.plan, cfg.seed);
  const start = Float64Array.from(theta);
  const P = theta.length;
  const grad = new Float64Array(P);
  const m = new Float64Array(P);
  const v = new Float64Array(P);
  const trainLoss: number[] = [];
  const valLoss: number[] = [];
  const checkpoints: Float64Array[] = [Float64Array.from(theta)];
  const every = Math.max(1, Math.floor(cfg.epochs / 100));
  let diverged = false;
  let clock = performance.now();

  for (let e = 1; e <= cfg.epochs; e++) {
    const loss = net.lossAndGrad(theta, data.Xtr, data.ytr, data.ntr, grad);
    if (!Number.isFinite(loss) || loss > 1e8) {
      diverged = true;
      break;
    }
    for (let i = 0; i < P; i++) {
      if (cfg.optimizer === "sgd") theta[i] -= cfg.lr * grad[i];
      else if (cfg.optimizer === "momentum") {
        m[i] = 0.9 * m[i] + grad[i];
        theta[i] -= cfg.lr * m[i];
      } else {
        m[i] = 0.9 * m[i] + 0.1 * grad[i];
        v[i] = 0.999 * v[i] + 0.001 * grad[i] * grad[i];
        const mh = m[i] / (1 - Math.pow(0.9, e));
        const vh = v[i] / (1 - Math.pow(0.999, e));
        theta[i] -= (cfg.lr * mh) / (Math.sqrt(vh) + 1e-8);
      }
    }
    trainLoss.push(loss);
    valLoss.push(net.loss(theta, data.Xva, data.yva, data.nva));
    if (e % every === 0) checkpoints.push(Float64Array.from(theta));

    if (performance.now() - clock > 16) {
      onProgress(e, trainLoss, valLoss);
      await new Promise((r) => setTimeout(r, 0));
      clock = performance.now();
      if (cancelled()) break;
    }
  }
  if (!diverged) checkpoints.push(Float64Array.from(theta));
  onProgress(trainLoss.length, trainLoss, valLoss);

  const lt = net.loss(theta, data.Xtr, data.ytr, data.ntr);
  const lv = net.loss(theta, data.Xva, data.yva, data.nva);
  let cls: TrainResult["cls"] = null;
  if (net.task === "classification") {
    const score = (X: Float64Array, y: Float64Array, n: number) => {
      const z = net.forward(theta, X, n);
      let ok = 0;
      let pos = 0;
      for (let r = 0; r < n; r++) {
        if ((z[r] > 0 ? 1 : 0) === y[r]) ok++;
        pos += y[r];
      }
      return { acc: ok / n, base: Math.max(pos, n - pos) / n };
    };
    const a = score(data.Xtr, data.ytr, data.ntr);
    const b = score(data.Xva, data.yva, data.nva);
    cls = { accTrain: a.acc, accVal: b.acc, baseVal: b.base, lossTrain: lt, lossVal: lv };
  }
  return {
    theta,
    start,
    checkpoints,
    trainLoss,
    valLoss,
    diverged,
    r2Train: cls ? 0 : 1 - lt / variance(data.ytr),
    r2Val: cls ? 0 : 1 - lv / variance(data.yva),
    cls,
  };
}
