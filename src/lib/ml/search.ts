import { Table } from "./datasets";
import { Term, splitIndices, termValue } from "./profile";
import { solve } from "./linalg";

export interface SearchStep {
  label: string;
  r2: number;
}

export interface AiTermSpec {
  kind: "square" | "interaction";
  columns: string[];
  why?: string;
}

export interface SearchResult {
  terms: Term[];
  base: number; // number of leading terms that are plain predictors
  steps: SearchStep[];
  r2Linear: number;
  r2Final: number;
  chosen: "search" | "ai" | "linear";
  compared: SearchStep[];
}

const keyOf = (t: Term) => [...t.parts].sort((a, b) => a - b).join(",");

// scores any subset of a pool of terms on the held-out rows, using precomputed Gram matrices
function evaluator(t: Table, target: number, pool: Term[]) {
  const { tr, va } = splitIndices(t.rows.length);
  const m = pool.length;
  const raw = (ids: number[]) => pool.map((tm) => ids.map((i) => termValue(t.rows[i], tm)));
  const trCols = raw(tr);
  const vaCols = raw(va);
  const mean = trCols.map((c) => c.reduce((s, v) => s + v, 0) / c.length);
  const sd = trCols.map((c, k) => Math.sqrt(c.reduce((s, v) => s + (v - mean[k]) ** 2, 0) / c.length) || 1);
  const ztr = trCols.map((c, k) => c.map((v) => (v - mean[k]) / sd[k]));
  const zva = vaCols.map((c, k) => c.map((v) => (v - mean[k]) / sd[k]));
  const ytr = tr.map((i) => t.rows[i][target]);
  const yva = va.map((i) => t.rows[i][target]);
  const ym = ytr.reduce((s, v) => s + v, 0) / ytr.length;
  const yvm = yva.reduce((s, v) => s + v, 0) / yva.length;
  const yc = ytr.map((v) => v - ym);
  const yvc = yva.map((v) => v - ym);
  const sst = yva.reduce((s, v) => s + (v - yvm) ** 2, 0) || 1;
  const yvSS = yvc.reduce((s, v) => s + v * v, 0);

  const dotp = (a: number[], b: number[]) => a.reduce((s, v, i) => s + v * b[i], 0);
  const G = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  const Gv = Array.from({ length: m }, () => new Array<number>(m).fill(0));
  for (let a = 0; a < m; a++)
    for (let b = a; b < m; b++) {
      G[a][b] = G[b][a] = dotp(ztr[a], ztr[b]);
      Gv[a][b] = Gv[b][a] = dotp(zva[a], zva[b]);
    }
  const bt = ztr.map((c) => dotp(c, yc));
  const bv = zva.map((c) => dotp(c, yvc));

  const r2 = (sel: number[]) => {
    const A = sel.map((a, i) => sel.map((b, j) => G[a][b] + (i === j ? 1e-6 * ytr.length : 0)));
    const w = solve(A, sel.map((a) => bt[a]));
    if (!w) return -Infinity;
    let quad = 0;
    for (let i = 0; i < sel.length; i++) for (let j = 0; j < sel.length; j++) quad += w[i] * w[j] * Gv[sel[i]][sel[j]];
    const sse = yvSS - 2 * sel.reduce((s, a, i) => s + w[i] * bv[a], 0) + quad;
    return 1 - sse / sst;
  };
  const corrY = ztr.map((c) => Math.abs(dotp(c, yc)) / (Math.sqrt(dotp(yc, yc)) * Math.sqrt(dotp(c, c)) || 1));
  return { r2, corrY, constant: sd.map((v, k) => v < 1e-9 || trCols[k].every((x) => x === trCols[k][0])) };
}

const pct = (v: number) => `${Math.max(0, Math.round(v * 100))}%`;

export function searchTerms(t: Table, target: number, feats: number[], ai: AiTermSpec[] = []): SearchResult {
  const nm = (j: number) => t.columns[j];
  const baseTerms: Term[] = feats.map((j) => ({ name: nm(j), parts: [j] }));
  const isBinary = (j: number) => t.rows.every((r) => r[j] === 0 || r[j] === 1);

  // pool of candidates, kept small enough to score every one of them
  const cand: Term[] = [];
  const seen = new Set(baseTerms.map(keyOf));
  const add = (tm: Term) => {
    if (!seen.has(keyOf(tm))) {
      seen.add(keyOf(tm));
      cand.push(tm);
    }
  };
  const mk = (...js: number[]): Term => ({ name: js.length === 2 && js[0] === js[1] ? `${nm(js[0])}²` : js.map(nm).join("·"), parts: js });

  if (feats.length === 1) {
    for (let p = 2; p <= 5; p++) add({ name: `${nm(feats[0])}${["", "", "²", "³", "⁴", "⁵"][p]}`, parts: Array(p).fill(feats[0]) });
  } else {
    const probe = evaluator(t, target, baseTerms);
    const rank = (js: number[]) => js.map((j) => ({ j, c: probe.corrY[feats.indexOf(j)] })).sort((a, b) => b.c - a.c).map((x) => x.j);
    const cont = rank(feats.filter((j) => !isBinary(j))).slice(0, 5);
    const bin = rank(feats.filter((j) => isBinary(j))).slice(0, 4);
    cont.forEach((a) => add(mk(a, a)));
    for (let i = 0; i < Math.min(cont.length, 4); i++) for (let k = i + 1; k < Math.min(cont.length, 4); k++) add(mk(cont[i], cont[k]));
    cont.slice(0, 4).forEach((a) => bin.forEach((b) => add(mk(a, b))));
    for (let i = 0; i < Math.min(bin.length, 3); i++) for (let k = i + 1; k < Math.min(bin.length, 3); k++) add(mk(bin[i], bin[k]));
  }

  // terms the AI proposed, added to the pool if they are real columns
  const aiTerms: Term[] = [];
  for (const spec of ai) {
    const js = spec.columns.map((c) => feats.find((j) => nm(j) === c));
    if (js.some((j) => j === undefined)) continue;
    if (spec.kind === "square" && js.length === 1) aiTerms.push(mk(js[0]!, js[0]!));
    else if (spec.kind === "interaction" && js.length === 2) aiTerms.push(mk(js[0]!, js[1]!));
  }
  const pool = [...baseTerms, ...cand];
  aiTerms.forEach((tm) => {
    if (!pool.some((p) => keyOf(p) === keyOf(tm))) pool.push(tm);
  });
  const ev = evaluator(t, target, pool);
  const idx = (tm: Term) => pool.findIndex((p) => keyOf(p) === keyOf(tm));
  const baseIdx = baseTerms.map((_, i) => i);
  const r2Linear = ev.r2(baseIdx);
  const compared: SearchStep[] = [{ label: "Linear in all predictors", r2: r2Linear }];

  let chosenIdx: number[] = baseIdx;
  const steps: SearchStep[] = [];

  if (feats.length === 1) {
    // powers of one predictor: take the lowest degree that is close to the best
    const fits = [1, 2, 3, 4, 5].map((deg) => ({ deg, r2: ev.r2(Array.from({ length: deg }, (_, i) => i)) }));
    const best = Math.max(...fits.map((f) => f.r2));
    const pick = fits.find((f) => f.r2 >= best - 0.01)!;
    fits.forEach((f) => steps.push({ label: f.deg === 1 ? "Straight line" : `Polynomial of degree ${f.deg}`, r2: f.r2 }));
    chosenIdx = Array.from({ length: pick.deg }, (_, i) => i);
  } else {
    let cur = r2Linear;
    const chosen = [...baseIdx];
    for (let round = 0; round < 8; round++) {
      let bestC = -1;
      let bestR = cur;
      for (let c = baseTerms.length; c < pool.length; c++) {
        if (chosen.includes(c) || ev.constant[c]) continue;
        const r = ev.r2([...chosen, c]);
        if (r > bestR) {
          bestR = r;
          bestC = c;
        }
      }
      if (bestC < 0 || bestR - cur < 0.004) break;
      chosen.push(bestC);
      cur = bestR;
      steps.push({ label: `Add ${pool[bestC].name}`, r2: cur });
    }
    chosenIdx = chosen;
  }
  const searchR2 = ev.r2(chosenIdx);
  compared.push({ label: "Best model from the term search", r2: searchR2 });

  let chosen: SearchResult["chosen"] = chosenIdx.length > baseIdx.length || feats.length === 1 ? "search" : "linear";
  let finalIdx = chosenIdx;
  let finalR2 = searchR2;

  if (aiTerms.length) {
    const aiIdx = [...baseIdx, ...aiTerms.map(idx)].filter((v, i, a) => a.indexOf(v) === i);
    const aiR2 = ev.r2(aiIdx);
    compared.push({ label: "Model suggested by the AI review", r2: aiR2 });
    if (aiR2 >= searchR2 - 0.003) {
      finalIdx = aiIdx;
      finalR2 = aiR2;
      chosen = "ai";
    }
  }
  return { terms: finalIdx.map((i) => pool[i]), base: baseTerms.length, steps, r2Linear, r2Final: finalR2, chosen, compared };
}

export function describeSearch(r: SearchResult, target: string): string {
  const extra = r.terms.slice(r.base);
  if (!extra.length) return `A linear model explains ${pct(r.r2Linear)} of the variance in ${target} on held-out rows, and no extra terms improved it, so we keep the model linear.`;
  const names = extra.map((e) => e.name).join(", ");
  const who = r.chosen === "ai" ? "The AI review suggested, and held-out scoring confirmed," : "A search over squared and interaction terms found";
  return `A linear model explains ${pct(r.r2Linear)} of the variance in ${target} on held-out rows. ${who} these extra terms: ${names}. They raise it to ${pct(r.r2Final)}. The model is still linear in its weights, which keeps the calculus that follows exact.`;
}
