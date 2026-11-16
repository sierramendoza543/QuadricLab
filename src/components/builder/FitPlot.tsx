"use client";

import { useMemo, useState } from "react";
import { Table } from "@/lib/ml/datasets";
import { Prepared, Term } from "@/lib/ml/profile";
import { defaultInputs, predictRow, usedSources } from "@/lib/ml/predict";
import Scatter from "./Scatter";

interface Props {
  table: Table;
  target: number;
  terms: Term[];
  prepared: Prepared;
  theta: Float64Array | null;
}

function corr(a: number[], b: number[]) {
  const n = a.length;
  const ma = a.reduce((s, v) => s + v, 0) / n;
  const mb = b.reduce((s, v) => s + v, 0) / n;
  let sab = 0;
  let saa = 0;
  let sbb = 0;
  for (let i = 0; i < n; i++) {
    sab += (a[i] - ma) * (b[i] - mb);
    saa += (a[i] - ma) ** 2;
    sbb += (b[i] - mb) ** 2;
  }
  return saa && sbb ? sab / Math.sqrt(saa * sbb) : 0;
}

// the data as points, and the fitted function drawn over it along one predictor
export default function FitPlot({ table, target, terms, prepared, theta }: Props) {
  const sources = useMemo(() => usedSources(table, terms, target), [table, terms, target]);
  const numeric = sources.filter((s) => s.kind === "numeric");
  const yi = target;

  const ranked = useMemo(
    () =>
      numeric
        .map((s) => ({ s, r: Math.abs(corr(table.rows.map((r) => r[table.columns.indexOf(s.columns[0])]), table.rows.map((r) => r[yi]))) }))
        .sort((a, b) => b.r - a.r),
    [numeric, table, yi]
  );
  const [pick, setPick] = useState<string | null>(null);
  const xs = ranked.find((r) => r.s.name === pick)?.s ?? ranked[0]?.s;

  const view = useMemo(() => {
    if (!xs) return null;
    const xc = table.columns.indexOf(xs.columns[0]);
    const step = Math.max(1, Math.ceil(table.rows.length / 400));
    const points = table.rows.filter((_, i) => i % step === 0).map((r) => ({ x: r[xc], y: r[yi] }));
    let curve: { x: number; y: number }[] | null = null;
    if (theta) {
      const lo = Math.min(...table.rows.map((r) => r[xc]));
      const hi = Math.max(...table.rows.map((r) => r[xc]));
      const base = defaultInputs(sources);
      curve = Array.from({ length: 81 }, (_, i) => {
        const x = lo + ((hi - lo) * i) / 80;
        return { x, y: predictRow(table, sources, prepared, terms, theta, { ...base, [xs.name]: x }).yhat };
      });
    }
    return { points, curve };
  }, [xs, table, yi, theta, sources, prepared, terms]);

  if (!xs || !view) return null;
  const others = sources.length > 1;

  return (
    <div className="mb-6 max-w-2xl">
      <h3 className="mb-1 text-lg font-bold">The data and the function we chose</h3>
      <p className="mb-2 text-[15px] text-gray-700">
        Each dot is one row: {xs.name} against {prepared.target}.{" "}
        {theta ? "The red curve is the fitted model." : "Press “Fit model” and gradient descent will draw the function over these points."}
      </p>
      {ranked.length > 1 && (
        <label className="mb-2 block text-[15px]">
          Predictor on the x-axis:{" "}
          <select className="border border-gray-400 bg-white px-2 py-0.5" value={xs.name} onChange={(e) => setPick(e.target.value)}>
            {ranked.map((r) => (
              <option key={r.s.name} value={r.s.name}>
                {r.s.name}
              </option>
            ))}
          </select>
        </label>
      )}
      <Scatter points={view.points} curve={view.curve} xLabel={xs.name} yLabel={prepared.target} />
      {theta && others && (
        <p className="mt-1 text-[14px] text-gray-600">
          The curve is the model’s prediction as {xs.name} changes, with every other predictor held at its average (or its baseline category). The dots also differ in those other predictors, which is why they scatter around the curve.
        </p>
      )}
    </div>
  );
}
