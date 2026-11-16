"use client";

import { useMemo, useState } from "react";
import { Table } from "@/lib/ml/datasets";
import { profileData } from "@/lib/ml/profile";
import Scatter from "./Scatter";

interface Props {
  table: Table;
  target: number;
  feats: number[];
  r2Linear: number | null;
}

const fmt = (v: number) => (Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

export default function StepExplore({ table, target, feats, r2Linear }: Props) {
  const p = useMemo(() => profileData(table, target, feats, r2Linear), [table, target, feats, r2Linear]);
  const [pick, setPick] = useState(0);
  const k = Math.min(pick, feats.length - 1);
  const j = feats[k];

  const pts = table.rows.map((r) => ({ x: r[j], y: r[target] }));
  const mx = pts.reduce((s, q) => s + q.x, 0) / pts.length;
  const my = pts.reduce((s, q) => s + q.y, 0) / pts.length;
  let sxx = 0;
  let sxy = 0;
  pts.forEach((q) => {
    sxx += (q.x - mx) ** 2;
    sxy += (q.x - mx) * (q.y - my);
  });
  const slope = sxx > 0 ? sxy / sxx : 0;

  return (
    <div>
      <h2 className="mb-2 text-2xl font-bold">3. Explore the data</h2>
      <p className="mb-4 text-[17px]">Summary statistics for each variable, and a scatterplot of the response against one predictor.</p>

      <table className="mb-6 w-full border border-gray-300 text-left text-[15px]">
        <thead className="bg-gray-100">
          <tr>
            {["Variable", "Mean", "Std. dev.", "Min", "Max"].map((h) => (
              <th key={h} className="border-b border-gray-300 px-3 py-1.5">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {p.stats.map((s, i) => (
            <tr key={s.name} className="border-b border-gray-200 last:border-0">
              <td className="px-3 py-1">
                {s.name}
                {i === p.stats.length - 1 && <span className="text-gray-500"> (response)</span>}
              </td>
              <td className="tabular-nums">{fmt(s.mean)}</td>
              <td className="tabular-nums">{fmt(s.std)}</td>
              <td className="tabular-nums">{fmt(s.min)}</td>
              <td className="tabular-nums">{fmt(s.max)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="mb-2 flex flex-wrap items-center gap-3">
        <label className="font-bold" htmlFor="pred">
          Predictor:
        </label>
        <select id="pred" className="border border-gray-400 bg-white px-2 py-1" value={k} onChange={(e) => setPick(parseInt(e.target.value, 10))}>
          {feats.map((f, i) => (
            <option key={f} value={i}>
              {table.columns[f]}
            </option>
          ))}
        </select>
        <span>
          correlation r = <b>{p.corr[k][feats.length].toFixed(3)}</b>
        </span>
      </div>
      <div className="max-w-xl">
        <Scatter points={pts} xLabel={table.columns[j]} yLabel={table.columns[target]} line={{ a: slope, b: my - slope * mx }} />
        <p className="mt-1 text-sm text-gray-500">Red line: least-squares line for this predictor alone.</p>
      </div>

      {p.observations.length > 0 && (
        <>
          <h3 className="mb-1 mt-6 text-lg font-bold">Observations</h3>
          <ul className="list-disc space-y-1.5 pl-6 text-[17px]">
            {p.observations.map((o) => (
              <li key={o}>{o}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
