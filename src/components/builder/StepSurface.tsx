"use client";

import dynamic from "next/dynamic";
import { useEffect, useState } from "react";
import { FIT_LABEL, fitEquation, shapeMeaning } from "@/lib/ml/landscape";
import { MathBundle } from "@/lib/ml/lesson";
import Tex from "../Tex";

const LandscapeCanvas = dynamic(() => import("./LandscapeCanvas"), { ssr: false });

const f = (x: number, d = 3) => x.toFixed(d);

function useReplay(length: number) {
  const [i, setI] = useState<number | null>(null);
  useEffect(() => {
    if (i === null) return;
    if (i >= length - 1) {
      const t = setTimeout(() => setI(null), 800);
      return () => clearTimeout(t);
    }
    const t = setTimeout(() => setI(i + 1), 60);
    return () => clearTimeout(t);
  }, [i, length]);
  return [i, () => setI(0)] as const;
}

export function StepSurface({ b, names, onPair }: { b: MathBundle; names: string[]; onPair: (p: [number, number]) => void }) {
  const { lesson: L, grid, fit, path } = b;
  const [replay, start] = useReplay(path.length);
  return (
    <div>
      <h2 className="mb-3 text-2xl font-bold">6. The loss surface</h2>
      <p className="mb-2 text-[17px] leading-relaxed">
        Every choice of weights is a different model. Take two weights, <Tex tex={"w_1"} /> (on {L.names[0]}) and <Tex tex={"w_2"} /> (on {L.names[1]}), and hold the others at their fitted values. Each point <Tex tex={"(w_1,w_2)"} /> on the floor is then one model, and the height above it is that model&apos;s loss <Tex tex={"L(w_1,w_2)"} />.
      </p>
      <p className="mb-3 text-[17px] leading-relaxed">Fitting a model means finding the lowest point of this surface.</p>
      {b.pair[0] >= 0 && b.pair[1] >= 0 && (
        <div className="mb-4 border border-gray-300 bg-gray-50 p-3 text-[16px]">
          <div className="mb-1">
            <b>Why these two weights?</b> {b.pairNote}
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span>Show weights for</span>
            {[0, 1].map((k) => (
              <select
                key={k}
                value={b.pair[k]}
                onChange={(e) => {
                  const v = parseInt(e.target.value, 10);
                  const next: [number, number] = k === 0 ? [v, b.pair[1]] : [b.pair[0], v];
                  if (next[0] !== next[1]) onPair(next);
                }}
                className="border border-gray-400 bg-white px-2 py-1"
              >
                {names.map((n, i) => (
                  <option key={n} value={i}>
                    {n}
                  </option>
                ))}
              </select>
            ))}
          </div>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-[1fr_260px]">
        <LandscapeCanvas className="h-[460px] border border-gray-300" grid={grid} fit={fit} showFit={false} path={path} replay={replay} showStart />
        <div className="space-y-3 text-[16px]">
          <div>
            <span className="mr-2 inline-block h-3 w-3 rounded-full bg-green-700" />
            <b>Fitted weights</b>
            <div className="pl-5 tabular-nums">
              ({f(L.best[0])}, {f(L.best[1])})<br />L = {f(L.Lmin, 3)}
            </div>
          </div>
          <div>
            <span className="mr-2 inline-block h-3 w-3 rounded-full bg-orange-600" />
            <b>Starting guess</b>
            <div className="pl-5 tabular-nums">
              ({f(L.start[0])}, {f(L.start[1])})<br />L = {f(L.L0, 3)}
            </div>
          </div>
          <div>
            <span className="mr-2 inline-block h-[3px] w-3 bg-black align-middle" />
            <b>Gradient descent path</b>
            <div className="pl-5 text-gray-700">from the guess down to the minimum. The next screens work out one step.</div>
          </div>
          <button className="w-full border border-gray-500 bg-gray-100 px-3 py-1.5 hover:bg-gray-200" onClick={start}>
            Replay descent
          </button>
          <p className="text-sm text-gray-500">Drag to rotate, scroll to zoom.</p>
        </div>
      </div>
    </div>
  );
}

export function StepSummary({ b }: { b: MathBundle }) {
  const { lesson: L, grid, fit, path } = b;
  const [show, setShow] = useState(true);
  return (
    <div>
      <h2 className="mb-3 text-2xl font-bold">11. Summary</h2>
      <p className="mb-4 text-[17px] leading-relaxed">
        The loss surface of this model is {/^[aeiou]/i.test(FIT_LABEL[fit.kind]) ? "an" : "a"} <b>{FIT_LABEL[fit.kind].toLowerCase()}</b>. The orange lines are the quadric surface from the previous two screens, drawn over the real loss values.
      </p>
      <div className="grid gap-4 lg:grid-cols-[1fr_300px]">
        <div>
          <LandscapeCanvas className="h-[460px] border border-gray-300" grid={grid} fit={fit} showFit={show} path={path} replay={null} showStart={false} />
          <label className="mt-2 flex items-center gap-2 text-[16px]">
            <input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} />
            Show the fitted quadric (orange)
          </label>
        </div>
        <div className="space-y-3 text-[16px] leading-relaxed">
          <div className="border border-gray-300 p-3">
            <div className="font-bold">{FIT_LABEL[fit.kind]}</div>
            <div className="overflow-x-auto text-[17px]">
              <Tex tex={fitEquation(fit)} />
            </div>
            <div className="mt-1 text-sm text-gray-600">fit R² = {fit.r2.toFixed(3)}</div>
          </div>
          <ol className="list-decimal space-y-2 pl-5">
            <li>
              {b.task === "classification"
                ? "Log-loss is not exactly a polynomial, but near its minimum it is well approximated by a second-degree one, so its graph is close to a quadric surface."
                : "L is a second-degree polynomial in the weights, so its graph is a quadric surface."}
            </li>
            <li>
              The Hessian has eigenvalues μ₁ = {f(fit.mu1)} and μ₂ = {f(fit.mu2)}, which fixes the type.
            </li>
            <li>{shapeMeaning(fit)}</li>
          </ol>
          <p className="text-sm text-gray-500">Minimum at (w₁, w₂) = ({f(L.best[0])}, {f(L.best[1])}).</p>
        </div>
      </div>
    </div>
  );
}
