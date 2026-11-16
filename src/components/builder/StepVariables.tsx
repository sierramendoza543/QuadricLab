"use client";

import { AiReview } from "@/lib/ml/ai";
import { Table, sourcesOf, targetColumns } from "@/lib/ml/datasets";

interface Props {
  table: Table;
  target: number;
  feats: number[];
  review: AiReview | null;
  onTarget: (j: number) => void;
  onFeats: (f: number[]) => void;
}

const fmt = (v: number) => (Number.isInteger(v) || Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

export default function StepVariables({ table, target, feats, review, onTarget, onFeats }: Props) {
  const sources = sourcesOf(table);
  const idx = (c: string) => table.columns.indexOf(c);
  const targetSource = sources.find((s) => s.columns.includes(table.columns[target]));
  const note = (name: string) => review?.column_notes.find((n) => n.name === name)?.note ?? table.notes?.[name];

  const toggle = (cols: number[]) => {
    const on = cols.every((c) => feats.includes(c));
    onFeats(on ? feats.filter((f) => !cols.includes(f)) : [...new Set([...feats, ...cols])].sort((a, b) => a - b));
  };

  const header = table.preview ? table.preview.columns : table.columns;
  const rows = table.preview ? table.preview.rows.slice(0, 6) : table.rows.slice(0, 6).map((r) => r.map((v) => fmt(v)));

  return (
    <div>
      <h2 className="mb-2 text-2xl font-bold">2. Choose the variables</h2>
      <p className="mb-4 text-[17px]">
        Each row is {table.rowMeaning ?? "one observation"} ({table.rows.length} rows in total). The <b>response variable</b> y is what we want to predict. The <b>predictor variables</b> x are the columns we use to predict it.
      </p>
      {review && (
        <div className="mb-4 border border-blue-300 bg-blue-50 px-3 py-2 text-[16px]">
          <b>The AI review</b> picked the response and predictors below and added its extra terms to the model search. You can change any of it.
        </div>
      )}

      <div className="mb-5 overflow-x-auto border border-gray-300">
        <table className="w-full text-left text-[15px]">
          <thead className="bg-gray-100">
            <tr>
              {header.map((c) => (
                <th key={c} className="border-b border-gray-300 px-3 py-1.5">
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className="border-b border-gray-200 last:border-0">
                {r.map((v, j) => (
                  <td key={j} className="px-3 py-1 tabular-nums">
                    {v}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid gap-6 sm:grid-cols-2">
        <div>
          <h3 className="mb-1 font-bold">Response variable (y)</h3>
          {targetColumns(table).map((j) => (
            <label key={j} className="flex cursor-pointer items-start gap-2 py-0.5">
              <input type="radio" name="response" className="mt-1.5" checked={target === j} onChange={() => onTarget(j)} />
              <span>
                {table.columns[j]}
                {note(table.columns[j]) && <span className="text-gray-500"> – {note(table.columns[j])}</span>}
              </span>
            </label>
          ))}
        </div>
        <div>
          <h3 className="mb-1 font-bold">Predictor variables (x)</h3>
          {sources
            .filter((s) => s !== targetSource)
            .map((s) => {
              const cols = s.columns.map(idx);
              return (
                <label key={s.name} className="flex cursor-pointer items-start gap-2 py-0.5">
                  <input type="checkbox" className="mt-1.5" checked={cols.every((c) => feats.includes(c))} onChange={() => toggle(cols)} />
                  <span>
                    {s.name}
                    {s.kind === "categorical" && <span className="text-gray-500"> (text: {s.levels?.join(", ")})</span>}
                    {note(s.name) && <span className="text-gray-500"> – {note(s.name)}</span>}
                  </span>
                </label>
              );
            })}
        </div>
      </div>

      <p className="mt-6 border-t border-gray-300 pt-3 text-[17px]">
        {feats.length === 0 ? (
          <span className="text-red-700">Select at least one predictor.</span>
        ) : (
          <>
            Model: predict <b>{table.columns[target]}</b> from <b>{feats.map((j) => table.columns[j]).join(", ")}</b>.
          </>
        )}
      </p>
    </div>
  );
}
