"use client";

import { Agent, AgentKind, Model, gradient, norm, surfaceValue } from "@/lib/calculusMath";
import { KIND_COLOR, KIND_LABEL, KIND_MEANING, canonicalLatex, levelLatex, num, traceLatex } from "@/lib/equations";
import { Slices } from "@/lib/types";
import Tex from "./Tex";

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between border-b border-gray-200 py-1 text-[14px]">
      <span className="text-gray-600">{label}</span>
      <span className="tabular-nums text-gray-900">{value}</span>
    </div>
  );
}

function Title({ children, note }: { children: React.ReactNode; note?: string }) {
  return (
    <div className="mb-2 mt-5">
      <h3 className="text-[15px] font-bold text-gray-900">{children}</h3>
      {note && <p className="text-[13px] text-gray-500">{note}</p>}
    </div>
  );
}

interface Props {
  model: Model;
  agents: Agent[];
  focusId: AgentKind;
  onFocus: (k: AgentKind) => void;
  slices: Slices;
}

export default function TelemetryPanel({ model, agents, focusId, onFocus, slices }: Props) {
  const agent = agents.find((a) => a.id === focusId) ?? agents[0];
  const lines = canonicalLatex(model);
  const traceX = traceLatex(model, "x", 0);
  const traceY = traceLatex(model, "y", 0);
  const level = levelLatex(model, slices.z.k);
  const color = KIND_COLOR[model.kind];

  return (
    <div className="px-4 pb-5 pt-4">
      <h2 className="text-[17px] font-bold text-gray-900">What the surface is</h2>

      <div className="mt-3 inline-block border px-2.5 py-1 text-[13px] font-bold" style={{ color, borderColor: color }}>
        {KIND_LABEL[model.kind]} (Stewart §12.6)
      </div>
      <div className="mt-3 overflow-x-auto border border-gray-300 bg-gray-50 p-3 text-[15px]">
        {lines.map((l, i) => (
          <div key={i} className={i === 0 ? "" : "mt-1 text-[13px] text-gray-700"}>
            <Tex tex={l} />
          </div>
        ))}
      </div>
      <p className="mt-2 text-[14px] leading-snug text-gray-700">{KIND_MEANING[model.kind]}</p>
      <p className="mt-1 text-[12px] text-gray-500">
        Curvatures of the data: λ₁ = {num(model.lam1, 3)}, λ₂ = {num(model.lam2, 3)}.
      </p>

      <Title note="Cut the surface with a plane and look at the curve it leaves.">Traces</Title>
      <div className="space-y-2 text-[14px]">
        <div className="border-l-4 border-emerald-600 bg-gray-50 px-3 py-2">
          <div className="text-[12px] font-bold text-emerald-700">x = 0: {traceX.kind}</div>
          <Tex tex={traceX.tex} />
        </div>
        <div className="border-l-4 border-pink-600 bg-gray-50 px-3 py-2">
          <div className="text-[12px] font-bold text-pink-700">y = 0: {traceY.kind}</div>
          <Tex tex={traceY.tex} />
        </div>
        <div className="overflow-x-auto border-l-4 border-amber-600 bg-gray-50 px-3 py-2">
          <div className="text-[12px] font-bold text-amber-700">
            z = {num(slices.z.k)}: {level.kind}
          </div>
          <Tex tex={level.tex} />
        </div>
      </div>

      <Title note="Click the surface to drop a ball, then press Play.">The ball</Title>
      {!agent ? (
        <p className="border border-dashed border-gray-300 p-3 text-[14px] text-gray-600">No ball yet.</p>
      ) : (
        <>
          {agents.length > 1 && (
            <div className="mb-2 flex gap-1 bg-gray-100 p-1 text-[13px]">
              {agents.map((a) => (
                <button
                  key={a.id}
                  onClick={() => onFocus(a.id)}
                  className={`flex-1 py-1 font-semibold ${focusId === a.id ? "bg-white text-gray-900 shadow-sm" : "text-gray-600 hover:text-gray-900"}`}
                >
                  {a.id === "gd" ? "Ball A: gradient descent" : "Ball B: momentum"}
                </button>
              ))}
            </div>
          )}
          {(() => {
            const [w1, w2] = agent.pos;
            const g = gradient(model, w1, w2);
            return (
              <>
                <Row label="epoch" value={`${agent.iter}${agent.status === "running" ? "" : " (" + agent.status + ")"}`} />
                <Row label="weights (w₁, w₂)" value={`(${num(w1, 3)}, ${num(w2, 3)})`} />
                <Row label="height z (loss above the minimum)" value={num(surfaceValue(model, w1, w2), 4)} />
                <Row label="gradient ∇f" value={`⟨${num(g[0], 3)}, ${num(g[1], 3)}⟩`} />
                <Row label="gradient length |∇f|" value={num(norm(g), 4)} />
              </>
            );
          })()}
        </>
      )}
    </div>
  );
}
