"use client";

import { useState } from "react";
import { AiReview } from "@/lib/ml/ai";
import { BUNDLED, DATA_SOURCES, Table } from "@/lib/ml/datasets";
import { parseCSV } from "@/lib/ml/csv";
import ReviewPanel from "./ReviewPanel";

interface Props {
  table: Table | null;
  note: string | null;
  review: AiReview | null;
  onTable: (t: Table, note: string | null) => void;
  onReview: (r: AiReview | null) => void;
}

export default function StepData({ table, note, review, onTable, onReview }: Props) {
  const [error, setError] = useState<string | null>(null);

  const readFile = async (f: File | undefined) => {
    if (!f) return;
    setError(null);
    try {
      if (f.size > 8_000_000) throw new Error("That file is larger than 8 MB. Reduce it to a few thousand rows and try again.");
      const res = parseCSV(await f.text(), f.name.replace(/\.[^.]+$/, ""));
      const bits = [`${res.table.rows.length} rows, ${res.table.columns.length} columns after encoding.`];
      const text = res.table.sources?.filter((s) => s.kind === "categorical") ?? [];
      if (text.length) bits.push(`Text columns turned into 0/1 indicators: ${text.map((s) => `${s.name} (${s.columns.join(", ")})`).join("; ")}.`);
      if (res.droppedColumns.length) bits.push(`Not used: ${res.droppedColumns.slice(0, 6).join(", ")}${res.droppedColumns.length > 6 ? ", …" : ""}.`);
      if (res.droppedRows) bits.push(`${res.droppedRows} rows with missing values removed.`);
      if (res.sampled) bits.push("Sampled down to 5,000 rows.");
      onReview(null);
      onTable(res.table, bits.join(" "));
    } catch (e) {
      setError(e instanceof Error ? e.message : "The file could not be read.");
    }
  };

  return (
    <div>
      <h2 className="mb-2 text-2xl font-bold">1. Choose a dataset</h2>
      <p className="mb-4 text-[17px]">Upload a CSV file, or select a sample dataset. Text columns such as yes/no or region are converted to 0/1 indicator columns so they can be used.</p>

      <label className="mb-5 flex flex-wrap items-center gap-3 border border-gray-300 p-3">
        <span className="font-bold">Upload CSV</span>
        <input type="file" accept=".csv,.tsv,.txt" onChange={(e) => readFile(e.target.files?.[0])} />
        <span className="text-[15px] text-gray-600">A response column with two values is treated as classification.</span>
      </label>

      <h3 className="mb-2 text-lg font-bold">Or choose a sample dataset</h3>
      <div className="grid gap-5 md:grid-cols-2">
        {(
          [
            ["regression", "Regression", "Predict a number"],
            ["classification", "Classification", "Predict a yes or no"],
          ] as const
        ).map(([task, title, sub]) => (
          <div key={task}>
            <h3 className="font-bold">{title}</h3>
            <p className="mb-1 text-[15px] text-gray-600">{sub}</p>
            <div className="divide-y divide-gray-300 border border-gray-300">
              {BUNDLED.filter((d) => d.task === task).map((d) => (
                <label key={d.id} className="flex cursor-pointer items-start gap-3 p-3 hover:bg-gray-50">
                  <input type="radio" name="dataset" className="mt-1.5" checked={table?.name === d.title} onChange={() => { onReview(null); onTable(d.build(), null); }} />
                  <span>
                    <span className="block font-bold">{d.title}</span>
                    <span className="block text-gray-700">{d.blurb}</span>
                    <span className="block text-sm text-gray-500">{d.size}</span>
                  </span>
                </label>
              ))}
            </div>
          </div>
        ))}
      </div>

      {error && <p className="mt-3 text-red-700">{error}</p>}
      {table && !error && (
        <p className="mt-3">
          Loaded <b>{table.name}</b>. {note ?? `${table.rows.length} rows, ${table.columns.length} columns.`}
        </p>
      )}
      {table?.preview && <ReviewPanel table={table} review={review} onReview={onReview} />}

      <h3 className="mb-1 mt-8 text-lg font-bold">Where to find more data</h3>
      <ul className="list-disc space-y-0.5 pl-6">
        {DATA_SOURCES.map((s) => (
          <li key={s.name}>
            <a href={s.url} target="_blank" rel="noopener noreferrer" className="text-blue-700 underline">
              {s.name}
            </a>
            <span className="text-gray-600"> – {s.note}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
