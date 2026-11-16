"use client";

import { useState } from "react";
import { TrainResult } from "@/lib/ml/engine";
import { ModelChoice } from "@/lib/ml/lesson";
import { Prepared } from "@/lib/ml/profile";
import { Table } from "@/lib/ml/datasets";
import CodeView from "./CodeView";
import FitPlot from "./FitPlot";
import LossChart from "./LossChart";
import Tex from "../Tex";

interface Props {
  table: Table;
  target: number;
  choice: ModelChoice;
  prepared: Prepared;
  code: string;
  running: boolean;
  loss: number[];
  result: TrainResult | null;
  rmse: number | null;
  onRun: () => void;
  onStop: () => void;
  onTry: () => void;
}

export default function StepFit({ table, target, choice, prepared, code, running, loss, result, rmse, onRun, onStop, onTry }: Props) {
  const [showCode, setShowCode] = useState(false);
  const d = prepared.d;
  const w3 = (v: number) => (Math.abs(v) < 0.0005 ? 0 : v).toFixed(3);
  const shown = prepared.names.slice(0, 5);
  const cls = prepared.task === "classification";
  const eq = String.raw`${cls ? "z" : "\\hat y"} = b + ` + shown.map((_, i) => String.raw`w_{${i + 1}}x_{${i + 1}}`).join(" + ") + (d > 5 ? String.raw` + \cdots + w_{${d}}x_{${d}}` : "");

  const sig = String.raw`\hat p = \sigma(z) = \frac{1}{1+e^{-z}}`;
  const l = result?.theta;
  const layer = { wOff: 0, bOff: d };

  return (
    <div>
      <h2 className="mb-2 text-2xl font-bold">4. Fit the model</h2>
      <p className="mb-2 text-[17px]">{choice.reason}</p>
      <details className="mb-3 border border-gray-300 p-3">
        <summary className="cursor-pointer font-bold">How the model was chosen</summary>
        <p className="mt-2 text-[15px] text-gray-700">Each candidate is fitted on 80% of the rows and scored {cls ? "(R² of a least-squares fit to the 0/1 labels)" : "(R²)"} on the other 20%, so a model cannot win just by memorizing.</p>
        <table className="mt-2 w-full max-w-xl border border-gray-300 text-left text-[15px]">
          <thead className="bg-gray-100">
            <tr>
              <th className="border-b border-gray-300 px-3 py-1">Candidate</th>
              <th className="border-b border-gray-300 px-3 py-1">{cls ? "Held-out score" : "Held-out R²"}</th>
            </tr>
          </thead>
          <tbody className="tabular-nums">
            {[...choice.search.compared, ...choice.search.steps.map((st) => ({ label: `   ${st.label}`, r2: st.r2 }))].map((c, i) => (
              <tr key={i} className="border-b border-gray-200 last:border-0">
                <td className="whitespace-pre px-3 py-0.5">{c.label}</td>
                <td className="px-3">{c.r2.toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      <p className="text-[17px]">
        {cls
          ? `The model computes a score z from a weighted sum of the predictors. The sigmoid function then squeezes z into a probability between 0 and 1 that ${prepared.classLabel}:`
          : "The model predicts the response with a weighted sum of the predictors:"}
      </p>
      <div className="my-2 border-l-2 border-gray-300 pl-4 text-[18px]">
        <Tex block left tex={eq} />
        {cls && <Tex block left tex={sig} />}
      </div>
      <p className="mb-4 text-[15px] text-gray-600">
        {prepared.names.slice(0, 8).map((n, i) => `x${i + 1} = ${n}`).join(", ")}
        {d > 8 ? ", …" : ""}. {cls ? "The predictors are standardized (mean 0, standard deviation 1)." : "All variables are standardized (mean 0, standard deviation 1)."}
      </p>
      <p className="mb-4 text-[17px]">
        The weights are found by <b>gradient descent</b>: start with arbitrary weights and repeatedly move them in the direction that lowers the {cls ? "cross-entropy (log-loss)" : "mean squared error"}.
      </p>

      <div className="mb-3 flex flex-wrap gap-3">
        {running ? (
          <button className="border border-gray-500 bg-gray-100 px-4 py-1.5 hover:bg-gray-200" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="border border-blue-800 bg-blue-700 px-4 py-1.5 font-bold text-white hover:bg-blue-800" onClick={onRun}>
            {result ? "Fit again" : "Fit model"}
          </button>
        )}
        {result && !result.diverged && (
          <button className="border border-gray-400 bg-white px-4 py-1.5 hover:bg-gray-100" onClick={onTry}>
            Try the model on your own inputs
          </button>
        )}
        <button className="border border-gray-400 bg-white px-4 py-1.5 hover:bg-gray-100" onClick={() => setShowCode((v) => !v)}>
          {showCode ? "Hide" : "Show"} PyTorch code
        </button>
      </div>

      <FitPlot table={table} target={target} terms={choice.terms} prepared={prepared} theta={result && !result.diverged ? result.theta : null} />

      {showCode && <div className="mb-4"><CodeView code={code} /></div>}

      <div className="grid gap-6 lg:grid-cols-2">
        <LossChart loss={loss} />
        {result && !result.diverged && l && (
          <div>
            <table className="mb-3 w-full border border-gray-300 text-left text-[15px]">
              <thead className="bg-gray-100">
                <tr>
                  <th className="border-b border-gray-300 px-3 py-1.5">Term</th>
                  <th className="border-b border-gray-300 px-3 py-1.5">Fitted weight</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-200">
                  <td className="px-3 py-1">b (intercept)</td>
                  <td className="tabular-nums">{w3(l[layer.bOff])}</td>
                </tr>
                {prepared.names.slice(0, 8).map((n, i) => (
                  <tr key={n} className="border-b border-gray-200 last:border-0">
                    <td className="px-3 py-1">
                      w<sub>{i + 1}</sub> ({n})
                    </td>
                    <td className="tabular-nums">{w3(l[layer.wOff + i])}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {result.cls ? (
              <p>
                Accuracy (training) = <b>{(result.cls.accTrain * 100).toFixed(1)}%</b>
                <br />
                Accuracy (held-out 20%) = <b>{(result.cls.accVal * 100).toFixed(1)}%</b>
                <br />
                Always guessing the common class = <b>{(result.cls.baseVal * 100).toFixed(1)}%</b>
                <br />
                Log-loss (held-out) = <b>{result.cls.lossVal.toFixed(3)}</b>
                {result.cls.lossTrain < 0.05 && (
                  <span className="mt-2 block text-[15px] text-gray-700">
                    The classes are almost perfectly separable, so more training keeps pushing the weights larger and the loss toward 0. The true minimum is at infinity, and the loss surface will look like a trough that keeps getting flatter.
                  </span>
                )}
              </p>
            ) : (
              <p>
                R² (training) = <b>{result.r2Train.toFixed(3)}</b>
                <br />
                R² (held-out 20%) = <b>{result.r2Val.toFixed(3)}</b>
                {rmse !== null && (
                  <>
                    <br />
                    RMSE (held-out) = <b>{rmse >= 100 ? rmse.toFixed(0) : rmse.toFixed(2)}</b> in the units of {prepared.target}
                  </>
                )}
              </p>
            )}
          </div>
        )}
      </div>
      {result?.diverged && <p className="mt-3 text-red-700">The loss diverged. Try fitting again.</p>}
    </div>
  );
}
