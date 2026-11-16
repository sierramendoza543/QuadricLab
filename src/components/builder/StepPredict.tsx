"use client";

import { useMemo, useState } from "react";
import { Table } from "@/lib/ml/datasets";
import { Prepared, Term } from "@/lib/ml/profile";
import { sigmoid } from "@/lib/ml/engine";
import { Inputs, defaultInputs, predictRow, usedSources } from "@/lib/ml/predict";

interface Props {
  table: Table;
  target: number;
  terms: Term[];
  prepared: Prepared;
  theta: Float64Array;
}

const money = (v: number) => {
  const a = Math.abs(v);
  const d = a >= 100 ? 0 : a >= 10 ? 1 : 2;
  return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
};
const signed = (v: number) => (v >= 0 ? "+" : "−") + money(Math.abs(v));

export default function StepPredict({ table, target, terms, prepared, theta }: Props) {
  const sources = useMemo(() => usedSources(table, terms, target), [table, terms, target]);
  const [values, setValues] = useState<Inputs>(() => defaultInputs(sources));
  const p = useMemo(() => predictRow(table, sources, prepared, terms, theta, values), [table, sources, prepared, terms, theta, values]);
  const name = table.columns[target];
  const cls = prepared.task === "classification";
  const prob = sigmoid(p.yhat);

  const sorted = [...p.contributions].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect));
  const shown = sorted.slice(0, 8);
  const rest = sorted.slice(8).reduce((s, c) => s + c.effect, 0);
  const maxAbs = Math.max(...sorted.map((c) => Math.abs(c.effect)), Math.abs(rest), 1e-9);
  const set = (k: string, v: number | string) => setValues((cur) => ({ ...cur, [k]: v }));

  return (
    <div>
      <h2 className="mb-2 text-2xl font-bold">12. Try the model</h2>
      <p className="mb-4 text-[17px] leading-relaxed">
        {cls ? (
          <>
            Enter values for the predictors and the fitted model gives the probability that <b>{prepared.classLabel}</b>. Each bar below is one term: its weight times how far your input is from average, in log-odds. The bars add up to the score z, and the sigmoid turns z into the probability.
          </>
        ) : (
          <>
            Enter values for the predictors and the fitted model predicts <b>{name}</b>. Each bar below is one term: its weight times how far your input is from average. The bars add up to the prediction.
          </>
        )}
      </p>

      <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
        <div>
          {sources.map((s) => (
            <div key={s.name} className="mb-3">
              <div className="mb-0.5 flex justify-between">
                <b>{s.name}</b>
                {s.kind === "numeric" && (s.min !== undefined && (Number(values[s.name]) < s.min || Number(values[s.name]) > (s.max ?? Infinity))) && <span className="text-sm text-amber-700">outside the data</span>}
              </div>
              {s.kind === "numeric" ? (
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={s.min}
                    max={s.max}
                    step={((s.max ?? 1) - (s.min ?? 0)) / 200 || 0.01}
                    value={Math.min(s.max ?? 0, Math.max(s.min ?? 0, Number(values[s.name])))}
                    onChange={(e) => set(s.name, Math.round(parseFloat(e.target.value) * 100) / 100)}
                    className="flex-1"
                  />
                  <input type="number" value={values[s.name]} onChange={(e) => set(s.name, parseFloat(e.target.value) || 0)} className="w-24 border border-gray-400 px-2 py-1 tabular-nums" />
                </div>
              ) : (
                <select value={String(values[s.name])} onChange={(e) => set(s.name, e.target.value)} className="w-full border border-gray-400 bg-white px-2 py-1">
                  {s.levels?.map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))}
                </select>
              )}
              {s.kind === "numeric" && s.min !== undefined && (
                <div className="text-xs text-gray-500">
                  data range {money(s.min)} to {money(s.max ?? 0)}
                </div>
              )}
            </div>
          ))}
          <button className="border border-gray-400 bg-white px-3 py-1.5 hover:bg-gray-100" onClick={() => setValues(defaultInputs(sources))}>
            Reset to typical values
          </button>
        </div>

        <div>
          <div className="mb-4 border border-gray-300 bg-gray-50 px-4 py-3">
            {cls ? (
              <>
                <div className="text-[15px] text-gray-600">Probability that {prepared.classLabel}</div>
                <div className="text-[34px] font-bold tabular-nums">{(prob * 100).toFixed(1)}%</div>
                <div className="text-[15px] text-gray-700">
                  Predicted label: <b>{prob >= 0.5 ? 1 : 0}</b> (the model says {prepared.classLabel} is {prob >= 0.5 ? "more" : "less"} likely than not) · score z = {p.yhat.toFixed(2)}
                </div>
                <div className="mt-2 h-3 w-full bg-gray-200">
                  <div className="h-3 bg-blue-600 transition-all duration-300" style={{ width: `${prob * 100}%` }} />
                </div>
              </>
            ) : (
              <>
                <div className="text-[15px] text-gray-600">Predicted {name}</div>
                <div className="text-[34px] font-bold tabular-nums">{money(p.yhat)}</div>
              </>
            )}
          </div>

          <div className="mb-1 flex justify-between text-[15px]">
            <span>Starting point: {cls ? "the score z for an average row" : "the model's prediction for an average row"}</span>
            <b className="tabular-nums">{money(p.base)}</b>
          </div>
          <div className="mb-2 border-t border-gray-300" />
          {shown.map((c) => (
            <div key={c.name} className="mb-1.5 grid grid-cols-[170px_1fr_90px] items-center gap-2 text-[15px]">
              <span className="truncate" title={c.name}>
                {c.name}
              </span>
              <div className="relative h-4">
                <div className="absolute inset-y-0 left-1/2 w-px bg-gray-400" />
                <div
                  className={`absolute top-0.5 h-3 transition-all duration-300 ${c.effect >= 0 ? "bg-blue-600" : "bg-red-500"}`}
                  style={{ left: c.effect >= 0 ? "50%" : `${50 - (Math.abs(c.effect) / maxAbs) * 50}%`, width: `${(Math.abs(c.effect) / maxAbs) * 50}%` }}
                />
              </div>
              <span className="text-right tabular-nums">{signed(c.effect)}</span>
            </div>
          ))}
          {sorted.length > 8 && (
            <div className="mb-1.5 grid grid-cols-[170px_1fr_90px] items-center gap-2 text-[15px] text-gray-600">
              <span>{sorted.length - 8} smaller terms</span>
              <span />
              <span className="text-right tabular-nums">{signed(rest)}</span>
            </div>
          )}
          <div className="mt-2 flex justify-between border-t border-gray-300 pt-2 font-bold">
            <span>{cls ? "Score z, then P = σ(z)" : "Prediction"}</span>
            <span className="tabular-nums">{cls ? `${p.yhat.toFixed(2)} → ${(prob * 100).toFixed(1)}%` : money(p.yhat)}</span>
          </div>

          <details className="mt-4 border border-gray-300 p-3">
            <summary className="cursor-pointer font-bold">Show the arithmetic</summary>
            <table className="mt-2 w-full text-left text-[14px]">
              <thead className="bg-gray-100">
                <tr>
                  {["Term", "Standardized input z", "Weight w", cls ? "Effect w·z (log-odds)" : `Effect w·z (in ${name} units)`].map((h) => (
                    <th key={h} className="border-b border-gray-300 px-2 py-1">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="tabular-nums">
                {p.contributions.map((c) => (
                  <tr key={c.name} className="border-b border-gray-200 last:border-0">
                    <td className="px-2 py-0.5">{c.name}</td>
                    <td>{c.z.toFixed(3)}</td>
                    <td>{c.w.toFixed(3)}</td>
                    <td>{signed(c.effect)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="mt-2 text-sm text-gray-600">z = (term value − its average) ÷ its standard deviation. {cls ? "Effects are in log-odds, the units of the score z." : `Effects are the weights converted back to ${name} units.`}</p>
          </details>
        </div>
      </div>
    </div>
  );
}
