"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import LossChart from "@/components/builder/LossChart";
import Paper from "@/components/builder/Paper";
import Scatter from "@/components/builder/Scatter";
import Tex from "@/components/Tex";
import type { TraceMode } from "@/components/explorer/TraceBuildCanvas";
import {
  Agent,
  Vec2,
  buildModel,
  gradient,
  mse,
  newAgent,
  stepAgent,
  surfaceValue,
  updateDirection,
  zScaleFor,
} from "@/lib/calculusMath";
import { KIND_LABEL, canonicalLatex, levelLatex, traceLatex } from "@/lib/equations";
import { KINDS, Kind, ML_MEANING, STORIES, lossCoefficients, standardForm, standardTrace } from "@/lib/explorer";
import { traceQuadratic } from "@/lib/calculusMath";
import { Slices } from "@/lib/types";

const QuadricCanvas = dynamic(() => import("@/components/QuadricCanvas"), { ssr: false });
const TraceBuildCanvas = dynamic(() => import("@/components/explorer/TraceBuildCanvas"), { ssr: false });

const STEPS = ["Surface", "Equation", "Build", "Dataset", "Loss", "Point", "Traces", "Training", "In ML", "Summary"];
const T = String.raw;
const f = (x: number, d = 3) => (Math.abs(x) < 0.5 * Math.pow(10, -d) ? 0 : x).toFixed(d);
const NO_VECTORS = { normal: false, grad: false, momentum: false };
const P = ({ children }: { children: React.ReactNode }) => <p className="my-3 text-[17px] leading-relaxed">{children}</p>;
const H = ({ n, children }: { n: number; children: React.ReactNode }) => <h2 className="mb-3 text-2xl font-bold">
    {n}. {children}
  </h2>;
const btn = "border border-gray-400 bg-white px-4 py-1.5 hover:bg-gray-100 disabled:opacity-40";
const primary = "border border-blue-800 bg-blue-700 px-4 py-1.5 font-bold text-white hover:bg-blue-800 disabled:opacity-40";

export default function ExplorerPage() {
  const [kind, setKind] = useState<Kind | null>(null);
  const [preview, setPreview] = useState<Kind | null>(null);
  const [step, setStep] = useState(0);
  const [reached, setReached] = useState(0);
  const scroller = useRef<HTMLDivElement>(null);

  const story = kind ? STORIES[kind] : null;
  const model = useMemo(() => (story ? buildModel(story.params) : null), [story]);
  const zScale = useMemo(() => (model ? zScaleFor(model) : 1), [model]);
  const std = useMemo(() => (model && kind ? standardForm(kind, model) : null), [model, kind]);

  // build animation
  const [bmode, setBmode] = useState<TraceMode>("z");
  const [t, setT] = useState(0);
  const [bplay, setBplay] = useState(false);
  // a single model (a dot) and the trace through it
  const [dot, setDot] = useState<Vec2>([-2, 1.8]);
  const [tmode, setTmode] = useState<TraceMode>("x");
  const [tk, setTk] = useState(0);
  // training
  const [alpha, setAlpha] = useState(0.2);
  const [agent, setAgent] = useState<Agent>(() => newAgent("gd", [-2, 1.8]));
  const [tplay, setTplay] = useState(false);

  const liveModel = useRef(model);
  liveModel.current = model;

  const chooseKind = (k: Kind) => {
    const s = STORIES[k];
    setKind(k);
    setDot(s.start);
    setAlpha(s.alpha);
    setAgent(newAgent("gd", s.start));
    setTplay(false);
    setBmode("z");
    setT(0);
    setBplay(false);
    setReached(1);
    setStep(1);
  };

  const goTo = (i: number) => {
    setStep(i);
    setReached((r) => Math.max(r, i));
    scroller.current?.scrollTo({ top: 0 });
  };

  // start the build animation whenever its screen opens or the plane direction changes
  useEffect(() => {
    if (step === 2) {
      setT(0);
      setBplay(true);
    } else setBplay(false);
  }, [step, bmode]);

  useEffect(() => {
    if (!bplay) return;
    const id = setInterval(() => {
      setT((v) => {
        if (v >= 1) {
          setBplay(false);
          return 1;
        }
        return Math.min(1, v + 0.007);
      });
    }, 40);
    return () => clearInterval(id);
  }, [bplay]);

  // set the trace plane through the current dot
  useEffect(() => {
    if (!model) return;
    setTk(tmode === "x" ? dot[0] : tmode === "y" ? dot[1] : surfaceValue(model, dot[0], dot[1]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tmode, model, step === 6]);

  useEffect(() => {
    if (!tplay) return;
    const id = setInterval(() => {
      const m = liveModel.current;
      if (!m) return;
      setAgent((a) => {
        const nx = stepAgent(m, a, alpha, 0);
        if (nx.status !== "running") setTplay(false);
        return nx;
      });
    }, 650);
    return () => clearInterval(id);
  }, [tplay, alpha]);

  const resetTrain = (start: Vec2, a = alpha) => {
    setTplay(false);
    setAlpha(a);
    setAgent(newAgent("gd", start));
  };

  // ---------- screens ----------
  const slices: Slices = useMemo(
    () => ({
      x: { on: tmode === "x", k: tk },
      y: { on: tmode === "y", k: tk },
      z: { on: tmode === "z", k: tk },
    }),
    [tmode, tk]
  );

  const pv = useMemo(() => {
    if (!preview) return null;
    const m = buildModel(STORIES[preview].params);
    return { model: m, zScale: zScaleFor(m) };
  }, [preview]);

  let body: React.ReactNode = null;

  if (step === 0 || !story || !model || !std) {
    body = (
      <div>
        <h2 className="mb-2 text-2xl font-bold">Choose a quadric surface</h2>
        <p className="mb-5 text-[17px] leading-relaxed">
          Each of these four surfaces from Stewart §12.6 shows up as the loss surface of a real machine learning problem. Pick one to see how its equation is built, what a point and a trace mean for a model, and how training moves across it.
        </p>
        <div className="grid gap-5 lg:grid-cols-[1fr_1.1fr]">
          <div className="space-y-3">
            {KINDS.map((k) => (
              <button
                key={k}
                onClick={() => setPreview(k)}
                className={`block w-full border p-4 text-left hover:border-blue-700 hover:bg-blue-50 ${preview === k ? "border-blue-700 bg-blue-50" : "border-gray-300 bg-white"}`}
              >
                <div className="text-lg font-bold">{STORIES[k].name}</div>
                <div className="my-1 overflow-x-auto text-[17px]">
                  <Tex tex={STORIES[k].stewart} />
                </div>
                <div className="text-[15px] text-gray-700">
                  <b>Machine learning example:</b> {STORIES[k].scenario}. {STORIES[k].oneLine}
                </div>
              </button>
            ))}
          </div>
          <div>
            {pv ? (
              <>
                <div className="mb-2 flex flex-wrap items-center gap-3">
                  <button className={primary} onClick={() => chooseKind(preview!)}>
                    Explore the {STORIES[preview!].name.toLowerCase()}
                  </button>
                  <span className="text-[15px] text-gray-600">{STORIES[preview!].scenario}: {STORIES[preview!].oneLine}</span>
                </div>
                <QuadricCanvas className="h-[360px] border border-gray-300" model={pv.model} zScale={pv.zScale} agents={[]} focusId="gd" vectors={NO_VECTORS} slices={{ x: { on: false, k: 0 }, y: { on: false, k: 0 }, z: { on: false, k: 0 } }} onPick={() => {}} />
                <p className="mt-1 text-sm text-gray-500">Drag to rotate, scroll to zoom.</p>
              </>
            ) : (
              <div className="flex h-[420px] items-center justify-center border border-dashed border-gray-300 text-gray-500">Select a surface to preview it here.</div>
            )}
          </div>
        </div>
      </div>
    );
  } else if (step === 1) {
    const ex = (mode: TraceMode) => standardTrace(kind!, mode, 1, std.a, std.b);
    body = (
      <div>
        <H n={2}>The equation of {/^[aeiou]/i.test(story.name) ? "an" : "a"} {story.name.toLowerCase()}</H>
        <P>
          In Stewart §12.6, a quadric surface is the graph of a second-degree equation in x, y and z. This one, in standard form, is:
        </P>
        <Paper lines={[story.stewart]} />
        <P>
          Here the constants are <Tex tex={`a=${f(std.a, 2)}`} />
          {kind === "circular" ? ", the same in both directions" : kind === "cylinder" ? "" : <>, <Tex tex={`b=${f(std.b, 2)}`} /></>}. A smaller constant means the surface rises more steeply in that direction.
        </P>
        <P>
          To understand a surface, Stewart looks at its <b>traces</b>: the curves where it meets the planes x = k, y = k and z = k. Here is each trace for k = 1:
        </P>
        <table className="mb-3 w-full border border-gray-300 text-left text-[16px]">
          <thead className="bg-gray-100">
            <tr>
              {["Plane", "Trace equation", "Curve"].map((h) => (
                <th key={h} className="border-b border-gray-300 px-3 py-1.5">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(["z", "x", "y"] as const).map((m) => (
              <tr key={m} className="border-b border-gray-200 last:border-0">
                <td className="px-3 py-1.5">{m} = 1</td>
                <td className="px-3">
                  <Tex tex={ex(m).tex} />
                </td>
                <td className="px-3">{ex(m).name}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <P>The next screen builds the surface by sliding a plane through space and keeping every trace it leaves behind.</P>
      </div>
    );
  } else if (step === 2) {
    const zr = (() => {
      let lo = Infinity;
      let hi = -Infinity;
      for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) {
        const z = std.f(-2.5 + (5 * i) / 40, -2.5 + (5 * j) / 40);
        lo = Math.min(lo, z);
        hi = Math.max(hi, z);
      }
      return [lo, hi];
    })();
    const kcur = bmode === "z" ? zr[0] + (zr[1] - zr[0]) * t : -2.5 + 5 * t;
    const tr = standardTrace(kind!, bmode, kcur, std.a, std.b);
    body = (
      <div>
        <H n={3}>Build the surface from its traces</H>
        <P>A plane sweeps through the region. Each time it cuts the surface it leaves a curve. When the plane has passed through everything, the stack of traces is the surface.</P>
        <div className="mb-3 flex flex-wrap gap-4 text-[16px]">
          {(
            [
              ["z", "Horizontal planes z = k"],
              ["x", "Vertical planes x = k"],
              ["y", "Vertical planes y = k"],
            ] as const
          ).map(([m, label]) => (
            <label key={m} className="flex cursor-pointer items-center gap-2">
              <input type="radio" name="bmode" checked={bmode === m} onChange={() => setBmode(m)} />
              {label}
            </label>
          ))}
        </div>
        <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
          <TraceBuildCanvas className="h-[440px] border border-gray-300" f={std.f} D={2.5} mode={bmode} t={t} />
          <div className="text-[16px]">
            <div className="mb-1 font-bold">Current plane</div>
            <div className="mb-3 border border-gray-300 bg-gray-50 px-3 py-2 text-[18px]">
              <Tex tex={`${bmode} = ${f(kcur, 2)}`} />
            </div>
            <div className="mb-1 font-bold">Trace it leaves</div>
            <div className="mb-1 overflow-x-auto border border-gray-300 bg-gray-50 px-3 py-2 text-[18px]">
              <Tex tex={tr.tex} />
            </div>
            <div className="mb-4 text-gray-700">{tr.name}</div>
            <input type="range" min={0} max={1} step={0.005} value={t} onChange={(e) => { setBplay(false); setT(parseFloat(e.target.value)); }} className="w-full" />
            <div className="mt-2 flex gap-2">
              <button className={primary} onClick={() => { setT(0); setBplay(true); }}>
                Replay
              </button>
              <button className={btn} onClick={() => setBplay((v) => !v)} disabled={t >= 1}>
                {bplay ? "Pause" : "Play"}
              </button>
            </div>
            <p className="mt-3 text-sm text-gray-500">Drag the view to rotate it while it builds.</p>
          </div>
        </div>
      </div>
    );
  } else if (step === 3) {
    const rows = model.data.x1.slice(0, 6).map((v, i) => [v, model.data.x2[i], model.data.y[i]]);
    body = (
      <div>
        <H n={4}>The machine learning problem: {story.dataTitle.toLowerCase()}</H>
        <P>{story.dataNote}</P>
        {kind !== "saddle" ? (
          <>
            <table className="mb-1 w-full max-w-xl border border-gray-300 text-left text-[16px]">
              <thead className="bg-gray-100">
                <tr>
                  {story.cols.map((c, i) => (
                    <th key={c} className="border-b border-gray-300 px-3 py-1.5">
                      {i < 2 ? `x${i === 0 ? "₁" : "₂"}: ` : "y: "}
                      {c}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {rows.map((r, i) => (
                  <tr key={i} className="border-b border-gray-200 last:border-0">
                    {r.map((v, j) => (
                      <td key={j} className="px-3 py-1">
                        {f(v, 2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mb-2 text-sm text-gray-500">{model.data.n} rows in total, centered so the averages are 0.</p>
            <P>The model predicts y from a weighted sum of the two predictors. The loss is the mean squared error:</P>
            <Paper lines={[T`\hat y_i &= w_1x_{i1}+w_2x_{i2}`, T`L(w_1,w_2) &= \frac{1}{N}\sum_{i=1}^{N}\left(y_i-w_1x_{i1}-w_2x_{i2}\right)^2`]} />
            <P>Every pair of weights (w₁, w₂) is a different model with its own loss. The surface we are about to study is the graph of L over all those pairs.</P>
          </>
        ) : (
          <>
            <P>In a GAN, a generator and a discriminator train against each other. Simplify each to a single weight: the generator picks w₁, the discriminator picks w₂, and both look at one payoff.</P>
            <Paper lines={[T`\text{generator: } \min_{w_1} z, \qquad \text{discriminator: } \max_{w_2} z`]} />
            <P>The payoff z = z(w₁, w₂) is a function of two variables, so it has a graph over the (w₁, w₂) plane. Its shape decides whether the game settles down.</P>
          </>
        )}
      </div>
    );
  } else if (step === 4) {
    const c = lossCoefficients(model);
    const mu1 = 2 * model.lam1;
    const mu2 = 2 * model.lam2;
    const canon = canonicalLatex(model);
    body = (
      <div>
        <H n={5}>The loss is a quadric</H>
        {kind !== "saddle" ? (
          <>
            <P>Multiply out the square in the loss. Every term has degree 2 or less in w₁ and w₂, which is the form of a quadric surface:</P>
            <Paper lines={[T`L(w_1,w_2) &= A\,w_1^2 + B\,w_1w_2 + C\,w_2^2 + D\,w_1 + E\,w_2 + F`]} />
            <P>The coefficients are averages over the data (a bar means the average over the rows):</P>
            <Paper
              lines={[
                T`A &= \overline{x_1^2} = ${f(c.A)}`,
                T`B &= 2\,\overline{x_1x_2} = ${f(c.B)}`,
                T`C &= \overline{x_2^2} = ${f(c.C)}`,
                T`D &= -2\,\overline{x_1y} = ${f(c.D)}`,
                T`E &= -2\,\overline{x_2y} = ${f(c.E)}`,
                T`F &= \overline{y^2} = ${f(c.F)}`,
              ]}
            />
            <P>The second-degree part (A, B, C) sets the shape. Its Hessian matrix and eigenvalues are:</P>
            <Paper
              lines={[
                T`H &= \begin{bmatrix} 2A & B \\ B & 2C \end{bmatrix} = \begin{bmatrix} ${f(2 * c.A)} & ${f(c.B)} \\ ${f(c.B)} & ${f(2 * c.C)} \end{bmatrix}`,
                T`\mu_1 &= ${f(mu1)}, \qquad \mu_2 = ${f(mu2)}`,
              ]}
            />
            <P>
              Completing the square puts it in standard form, centered at the minimum <Tex tex={`(h,k)=(${f(model.center[0], 2)},\\,${f(model.center[1], 2)})`} />:
            </P>
            <Paper lines={[T`a &= \sqrt{2/\mu_1} = ${f(model.semiA, 3)}`, Number.isFinite(model.semiB) ? T`b &= \sqrt{2/\mu_2} = ${f(model.semiB, 3)}` : T`\mu_2 &\approx 0 \;\Rightarrow\; \text{no } b`, canon[0].replace(" = ", " &= ")]} />
            <P>
              {kind === "circular" && "Both eigenvalues are positive and equal, so a = b: a circular paraboloid."}
              {kind === "elliptic" && "Both eigenvalues are positive but different, so a ≠ b: an elliptic paraboloid. The smaller eigenvalue sits on the longer axis."}
              {kind === "cylinder" && "One eigenvalue is 0, so the loss does not change along that direction: a parabolic cylinder. (The two predictors are identical, so only the combination w₁ + w₂ matters.)"}
            </P>
          </>
        ) : (
          <>
            <P>Here the payoff is itself written as a quadratic. One direction curves up and the other curves down:</P>
            <Paper lines={[T`z &= \lambda_2u_2^2-\lambda_1u_1^2`, T`\lambda_1 &= ${f(model.lam1)}, \qquad \lambda_2 = ${f(model.lam2)}`, canon[0].replace(" = ", " &= ")]} />
            <P>The signs are opposite, so this is a hyperbolic paraboloid. The saddle point is at (h, k) = ({f(model.center[0], 2)}, {f(model.center[1], 2)}).</P>
          </>
        )}
      </div>
    );
  } else if (step === 5) {
    const [w1, w2] = dot;
    const z = surfaceValue(model, w1, w2);
    const reg = kind !== "saddle";
    const preds = reg ? model.data.x1.map((v, i) => ({ x: w1 * v + w2 * model.data.x2[i], y: model.data.y[i] })) : [];
    body = (
      <div>
        <H n={6}>A point on the surface is one model</H>
        <P>
          {reg
            ? "Choose two weights and you have chosen a model. The dot sits above that pair on the floor, and its height is the loss: how badly that model fits the data. Move the dot and watch the predictions change."
            : "Choose a value for each player and you have chosen one outcome of the game. The dot sits above that pair, and its height is the payoff z."}
        </P>
        <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
          <QuadricCanvas className="h-[440px] border border-gray-300" model={model} zScale={zScale} agents={[newAgent("gd", dot)]} focusId="gd" vectors={NO_VECTORS} slices={{ x: { on: false, k: 0 }, y: { on: false, k: 0 }, z: { on: false, k: 0 } }} onPick={setDot} />
          <div className="text-[16px]">
            {(
              [
                ["w₁", w1, 0],
                ["w₂", w2, 1],
              ] as const
            ).map(([name, val, idx]) => (
              <label key={name} className="mb-2 block">
                <div className="flex justify-between">
                  <b>{name}</b>
                  <span className="tabular-nums">{f(val, 2)}</span>
                </div>
                <input type="range" min={-3.4} max={3.4} step={0.05} value={val} onChange={(e) => setDot(idx === 0 ? [parseFloat(e.target.value), w2] : [w1, parseFloat(e.target.value)])} className="w-full" />
              </label>
            ))}
            <div className="mb-3 border border-gray-300 bg-gray-50 px-3 py-2 text-[17px]">
              {reg && <Tex tex={`\\hat y=${f(w1, 2)}\\,x_1${w2 < 0 ? "-" : "+"}${f(Math.abs(w2), 2)}\\,x_2`} />}
            </div>
            <table className="mb-3 w-full text-[16px]">
              <tbody>
                {reg && (
                  <tr className="border-b border-gray-200">
                    <td className="py-1">loss L (mean squared error)</td>
                    <td className="text-right tabular-nums">{f(mse(model.data, dot), 3)}</td>
                  </tr>
                )}
                <tr className="border-b border-gray-200">
                  <td className="py-1">{reg ? "height z = L − L_min" : "payoff z"}</td>
                  <td className="text-right tabular-nums">{f(z, 3)}</td>
                </tr>
              </tbody>
            </table>
            {reg && <Scatter points={preds} xLabel="prediction ŷ" yLabel="actual y" line={{ a: 1, b: 0 }} />}
            {reg && <p className="mt-1 text-sm text-gray-500">Points on the red line would be perfect predictions.</p>}
            <p className="mt-2 text-sm text-gray-500">You can also click the surface to place the dot.</p>
          </div>
        </div>
      </div>
    );
  } else if (step === 6) {
    const reg = kind !== "saddle";
    const zlo = model.kind === "saddle" ? model.zMin * 0.8 : 0;
    const zhi = model.zMax * 0.8;
    const lo = tmode === "z" ? zlo : -3.4;
    const hi = tmode === "z" ? zhi : 3.4;
    const q = tmode === "z" ? null : traceQuadratic(model, tmode, tk);
    const tx = tmode === "z" ? null : traceLatex(model, tmode, tk);
    const lv = tmode === "z" ? levelLatex(model, tk) : null;
    const free = tmode === "x" ? "w₂" : "w₁";
    const held = tmode === "x" ? "w₁" : "w₂";
    let meaning = "";
    if (tmode !== "z" && q) {
      const vertex = Math.abs(q.A) > 1e-6 ? -q.B / (2 * q.A) : null;
      meaning = `Hold ${held} fixed at ${f(tk, 2)} and let only ${free} change. The curve shows how the ${reg ? "loss" : "payoff"} responds to ${free} alone.`;
      if (vertex !== null)
        meaning += q.A > 0
          ? ` Its lowest point is the best ${free} for this ${held}: ${free} = ${f(vertex, 2)}. (Optimizing one weight at a time like this is called coordinate descent.)`
          : ` It opens downward, so ${free} = ${f(vertex, 2)} gives the highest payoff for this ${held}.`;
    } else if (reg) {
      meaning = `Every pair of weights on this curve gives the same loss (z = ${f(tk, 2)} above the minimum). Smaller z means a better model. At z = 0 the curve shrinks to one point, the best model.`;
      if (kind === "cylinder") meaning += " Here it is a pair of parallel lines, so many different weight pairs tie.";
    } else meaning = `Every pair on this curve gives the same payoff z = ${f(tk, 2)}.`;
    body = (
      <div>
        <H n={7}>Traces are slices of the loss</H>
        <P>A trace fixes one thing and shows what is left. Pick a plane, move it with the slider, and read what the curve means for the model. The plane starts at your dot from the previous screen.</P>
        <div className="mb-3 flex flex-wrap gap-4 text-[16px]">
          {(
            [
              ["x", "x = k (fix w₁)"],
              ["y", "y = k (fix w₂)"],
              ["z", "z = k (fix the loss)"],
            ] as const
          ).map(([m, label]) => (
            <label key={m} className="flex cursor-pointer items-center gap-2">
              <input type="radio" name="tmode" checked={tmode === m} onChange={() => setTmode(m)} />
              {label}
            </label>
          ))}
        </div>
        <div className="grid gap-4 lg:grid-cols-[1fr_340px]">
          <QuadricCanvas className="h-[440px] border border-gray-300" model={model} zScale={zScale} agents={[newAgent("gd", dot)]} focusId="gd" vectors={NO_VECTORS} slices={slices} onPick={setDot} />
          <div className="text-[16px]">
            <label className="mb-3 block">
              <div className="flex justify-between">
                <b>k</b>
                <span className="tabular-nums">{f(tk, 2)}</span>
              </div>
              <input type="range" min={lo} max={hi} step={(hi - lo) / 200} value={Math.min(hi, Math.max(lo, tk))} onChange={(e) => setTk(parseFloat(e.target.value))} className="w-full" />
            </label>
            <div className="mb-1 font-bold">Trace</div>
            <div className="mb-1 overflow-x-auto border border-gray-300 bg-gray-50 px-3 py-2 text-[17px]">
              <Tex tex={tx ? tx.tex : lv!.tex} />
            </div>
            <div className="mb-3 text-gray-700">{tx ? tx.kind : lv!.kind}</div>
            <div className="mb-1 font-bold">What it means</div>
            <p className="leading-relaxed">{meaning}</p>
          </div>
        </div>
      </div>
    );
  } else if (step === 7) {
    const reg = kind !== "saddle";
    const path = agent.path;
    const last = path.length - 1;
    const rowsShown = path.map((p, k) => ({ k, p })).slice(Math.max(0, last - 7));
    const zOf = (p: Vec2) => surfaceValue(model, p[0], p[1]);
    const valOf = (p: Vec2) => (reg ? mse(model.data, p) : zOf(p));
    const lossSeries = reg ? path.map((p) => mse(model.data, p)) : [];
    let lines: string[] = [];
    if (last >= 1) {
      const p0 = path[last - 1];
      const p1 = path[last];
      const g = gradient(model, p0[0], p0[1]);
      if (kind !== "saddle") {
        lines = [
          T`\nabla f(${f(p0[0])},\,${f(p0[1])}) &= \langle ${f(g[0])},\ ${f(g[1])}\rangle`,
          T`w_1 &= ${f(p0[0])} - ${alpha}\,(${f(g[0])}) = ${f(p1[0])}`,
          T`w_2 &= ${f(p0[1])} - ${alpha}\,(${f(g[1])}) = ${f(p1[1])}`,
        ];
      } else {
        const d = updateDirection(model, g);
        lines = [
          T`\nabla f(${f(p0[0])},\,${f(p0[1])}) &= \langle ${f(g[0])},\ ${f(g[1])}\rangle`,
          T`w_1 &= ${f(p0[0])} + ${alpha}\,(${f(d[0])}) = ${f(p1[0])}`,
          T`w_2 &= ${f(p0[1])} + ${alpha}\,(${f(d[1])}) = ${f(p1[1])}`,
        ];
      }
    }
    body = (
      <div>
        <H n={8}>Training, one epoch at a time</H>
        <P>
          Training starts at an arbitrary pair of weights. Each epoch computes the gradient at the current point and moves the weights a small step. Press “Next epoch” and watch the ball move on the surface.{" "}
          {kind === "saddle" ? "In a min–max game each weight moves its own way: w₂ goes downhill and w₁ goes uphill." : ""}
        </P>
        <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
          <div>
            <QuadricCanvas className="h-[440px] border border-gray-300" model={model} zScale={zScale} agents={[agent]} focusId="gd" vectors={NO_VECTORS} slices={{ x: { on: false, k: 0 }, y: { on: false, k: 0 }, z: { on: false, k: 0 } }} onPick={(w) => resetTrain(w)} />
            <p className="mt-1 text-sm text-gray-500">Click the surface to choose a new starting point.</p>
          </div>
          <div className="text-[16px]">
            <div className="mb-3 flex flex-wrap gap-2">
              <button className={primary} disabled={agent.status !== "running"} onClick={() => setAgent((a) => stepAgent(model, a, alpha, 0))}>
                Next epoch
              </button>
              <button className={btn} disabled={agent.status !== "running"} onClick={() => setTplay((v) => !v)}>
                {tplay ? "Pause" : "Play"}
              </button>
              <button className={btn} onClick={() => resetTrain(path[0])}>
                Reset
              </button>
            </div>
            <label className="mb-3 block">
              <div className="flex justify-between">
                <b>Learning rate α</b>
                <span className="tabular-nums">{alpha.toFixed(2)}</span>
              </div>
              <input type="range" min={0.02} max={0.6} step={0.01} value={alpha} onChange={(e) => resetTrain(path[0], parseFloat(e.target.value))} className="w-full" />
            </label>
            <table className="mb-3 w-full border border-gray-300 text-left text-[15px]">
              <thead className="bg-gray-100">
                <tr>
                  {["epoch", "w₁", "w₂", reg ? "loss L" : "payoff z"].map((h) => (
                    <th key={h} className="border-b border-gray-300 px-2 py-1">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {rowsShown.map(({ k, p }) => (
                  <tr key={k} className={`border-b border-gray-200 last:border-0 ${k === last ? "bg-yellow-100 font-bold" : ""}`}>
                    <td className="px-2 py-0.5">{k}</td>
                    <td>{f(p[0])}</td>
                    <td>{f(p[1])}</td>
                    <td>{f(valOf(p), 3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {last >= 1 ? <Paper small lines={lines} /> : <p className="text-gray-600">Epoch 0 is the starting guess. Press “Next epoch” to take the first step.</p>}
          </div>
        </div>
        {reg && lossSeries.length > 2 && (
          <div className="mt-4 max-w-xl">
            <LossChart loss={lossSeries} />
          </div>
        )}
        <p className="mt-4 text-[17px] leading-relaxed">
          <b>What to notice.</b> {story.notice}
          {kind === "cylinder" && agent.status !== "running" && ` It stopped with w₁ + w₂ = ${f(agent.pos[0] + agent.pos[1], 3)}, a point on the trough.`}
          {agent.status === "escaped" && " This step size sent the weights off the plotted region. Try a smaller learning rate."}
        </p>
      </div>
    );
  } else if (step === 8) {
    const mm = ML_MEANING[kind!];
    const mu1 = 2 * model.lam1;
    const mu2 = 2 * model.lam2;
    const kappa = mu2 > 1e-6 ? mu1 / mu2 : Infinity;
    body = (
      <div>
        <H n={9}>What the {story.name.toLowerCase()} means in machine learning</H>
        <p className="mb-2 text-xl font-bold">{mm.heading}</p>
        <h3 className="mt-5 text-lg font-bold">Why the loss has this shape</h3>
        {kind !== "saddle" && (
          <Paper
            lines={[T`L(w) &= \frac{1}{N}\lVert y-Xw\rVert^2`, T`H &= \nabla^2L = \frac{2}{N}X^{\top}X = 2\,\Sigma`]}
          />
        )}
        {mm.why.map((t) => (
          <P key={t}>{t}</P>
        ))}
        <p className="my-2">
          Condition on the Hessian eigenvalues: <Tex tex={mm.condition} />
        </p>
        {kind !== "saddle" ? (
          <p className="mb-2 text-gray-700">
            Here: μ₁ = {f(mu1)}, μ₂ = {f(mu2)}, so κ = μ₁/μ₂ {Number.isFinite(kappa) ? `= ${f(kappa, 2)}` : "is infinite"}.
          </p>
        ) : (
          <p className="mb-2 text-gray-700">
            Here: λ₁ = {f(model.lam1)} (curves down), λ₂ = {f(model.lam2)} (curves up). The signs differ.
          </p>
        )}
        <h3 className="mt-5 text-lg font-bold">Where you meet it</h3>
        <ul className="list-disc space-y-1 pl-6 text-[17px] leading-relaxed">
          {mm.where.map((t) => (
            <li key={t}>{t}</li>
          ))}
        </ul>
        <h3 className="mt-5 text-lg font-bold">What it does to training</h3>
        {mm.training.map((t) => (
          <P key={t}>{t}</P>
        ))}
        <h3 className="mt-5 text-lg font-bold">What to do about it</h3>
        <P>{mm.remedy}</P>
      </div>
    );
  } else {
    body = (
      <div>
        <H n={10}>Summary: the {story.name.toLowerCase()}</H>
        <ol className="list-decimal space-y-3 pl-6 text-[17px] leading-relaxed">
          <li>
            Standard form: <Tex tex={story.stewart} />. Its traces are {kind === "circular" ? "circles and parabolas" : kind === "elliptic" ? "ellipses and parabolas" : kind === "cylinder" ? "parallel lines and parabolas" : "hyperbolas and parabolas"}.
          </li>
          <li>
            {kind !== "saddle"
              ? "The mean squared error of a two-weight model is a second-degree polynomial in w₁ and w₂, so its graph is a quadric. The eigenvalues of its Hessian decide which one."
              : "A min–max payoff with one upward and one downward direction is a quadric with eigenvalues of opposite sign."}
          </li>
          <li>A point on the surface is one choice of weights. Its height is the {kind !== "saddle" ? "loss of that model" : "payoff"}.</li>
          <li>A trace fixes one weight (or the loss) and shows what remains: the best value of the other weight, or all the models that tie.</li>
          <li>{story.notice}</li>
        </ol>
        <div className="mt-6 flex gap-3">
          <button className={primary} onClick={() => { setKind(null); setStep(0); setReached(0); }}>
            Explore another surface
          </button>
          <Link href="/" className={btn}>
            Back to Quadric Lab
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div ref={scroller} className="fixed inset-0 overflow-y-auto bg-white text-[#1a1a1a]">
      <div className="flex min-h-full flex-col">
        <header className="border-b border-gray-300">
          <div className="mx-auto flex max-w-5xl items-baseline justify-between px-6 pt-4">
            <div>
              <span className="text-xl font-bold">Quadric Explorer</span>
              <span className="ml-3 text-gray-500">{kind ? KIND_LABEL[kind] : "how each Chapter 12 surface appears in machine learning"}</span>
            </div>
            <Link href="/" className="text-blue-700 underline">
              Quadric Lab
            </Link>
          </div>
          <nav className="mx-auto flex max-w-5xl gap-1 overflow-x-auto px-6 pt-3 text-[15px]">
            {STEPS.map((s, i) => {
              const open = kind !== null && i <= reached;
              return (
                <button
                  key={s}
                  disabled={!open}
                  onClick={() => goTo(i)}
                  className={`whitespace-nowrap border-b-2 px-2 pb-2 ${
                    i === step ? "border-blue-700 font-bold text-blue-800" : open ? "border-transparent text-gray-800 hover:text-blue-800" : "border-transparent text-gray-400"
                  }`}
                >
                  {i + 1}. {s}
                </button>
              );
            })}
          </nav>
        </header>

        <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-7">{body}</main>

        <footer className="sticky bottom-0 z-30 border-t border-gray-300 bg-white">
          <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3">
            <button disabled={step === 0} onClick={() => goTo(step - 1)} className="border border-gray-400 bg-white px-5 py-1.5 hover:bg-gray-100 disabled:opacity-40">
              Back
            </button>
            <span className="text-gray-500">
              Step {step + 1} of {STEPS.length}
            </span>
            <button
              disabled={step === 0 || step === STEPS.length - 1}
              onClick={() => goTo(step + 1)}
              className="border border-blue-800 bg-blue-700 px-5 py-1.5 font-bold text-white hover:bg-blue-800 disabled:border-gray-300 disabled:bg-gray-200 disabled:text-gray-500"
            >
              Next
            </button>
          </div>
        </footer>
      </div>
    </div>
  );
}
