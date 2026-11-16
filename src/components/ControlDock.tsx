"use client";

import { ChevronRight, Pause, Play, RotateCcw, StepForward } from "lucide-react";
import { Model, PRESETS, Params, AgentKind, Vec2, fitPredictions, mse } from "@/lib/calculusMath";
import { Hyper, Slices, VectorToggles } from "@/lib/types";
import { DOMAIN } from "@/lib/calculusMath";

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  digits?: number;
  disabled?: boolean;
  onChange: (v: number) => void;
}

function Slider({ label, value, min, max, step, digits = 2, disabled, onChange }: SliderProps) {
  return (
    <label className={`block ${disabled ? "opacity-40" : ""}`}>
      <div className="flex items-baseline justify-between text-[13px]">
        <span className="text-gray-700">{label}</span>
        <span className="tabular-nums text-blue-800">{value.toFixed(digits)}</span>
      </div>
      <input
        type="range"
        value={value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

interface SwitchProps {
  on: boolean;
  color?: string;
  onClick: () => void;
  label: string;
}

function Switch({ on, color = "#1d4ed8", onClick, label }: SwitchProps) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={on}
      aria-label={label}
      onClick={onClick}
      className="relative h-[18px] w-8 shrink-0 rounded-full border border-gray-300 transition-colors"
      style={{ background: on ? color : "#d1d5db" }}
    >
      <span
        className="absolute top-[2px] h-3 w-3 rounded-full bg-white transition-all"
        style={{ left: on ? 16 : 2 }}
      />
    </button>
  );
}

function SectionTitle({ children, note }: { children: React.ReactNode; note?: string }) {
  return (
    <div className="mb-2 mt-6">
      <h3 className="text-[15px] font-bold text-gray-900">{children}</h3>
      {note && <p className="text-[13px] leading-snug text-gray-500">{note}</p>}
    </div>
  );
}

function FitPlot({ model, weights }: { model: Model; weights: Vec2 }) {
  const W = 296;
  const H = 128;
  const pad = 14;
  const { x1, y } = model.data;
  const preds = fitPredictions(model, weights);
  const xs = x1;
  const ys = [...y, ...preds];
  const [x0, x1m] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1m] = [Math.min(...ys), Math.max(...ys)];
  const r2 = (v: number) => Math.round(v * 100) / 100;
  const sx = (v: number) => r2(pad + ((v - x0) / (x1m - x0 || 1)) * (W - 2 * pad));
  const sy = (v: number) => r2(H - pad - ((v - y0) / (y1m - y0 || 1)) * (H - 2 * pad));

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-md border border-gray-300 bg-gray-50">
        <line x1={pad} y1={sy(0)} x2={W - pad} y2={sy(0)} stroke="#d1d5db" />
        <line x1={sx(0)} y1={pad} x2={sx(0)} y2={H - pad} stroke="#d1d5db" />
        {x1.map((v, i) => (
          <line key={`r${i}`} x1={sx(v)} y1={sy(y[i])} x2={sx(v)} y2={sy(preds[i])} stroke="#d97706" strokeOpacity={0.45} />
        ))}
        {x1.map((v, i) => (
          <circle key={`p${i}`} cx={sx(v)} cy={sy(preds[i])} r={2.6} fill="none" stroke="#15803d" strokeWidth={1.2} />
        ))}
        {x1.map((v, i) => (
          <circle key={`d${i}`} cx={sx(v)} cy={sy(y[i])} r={2.6} fill="#1d4ed8" />
        ))}
      </svg>
      <div className="mt-1 flex justify-between text-[12px] text-gray-600">
        <span>
          <span className="text-blue-700">●</span> y &nbsp;<span className="text-green-700">○</span> ŷ &nbsp;vs. x₁
        </span>
        <span className="tabular-nums">MSE {mse(model.data, weights).toFixed(3)}</span>
      </div>
    </div>
  );
}

interface ControlDockProps {
  model: Model;
  params: Params;
  hyper: Hyper;
  slices: Slices;
  vectors: VectorToggles;
  race: boolean;
  solo: AgentKind;
  playing: boolean;
  hasBall: boolean;
  activePreset: string | null;
  fitWeights: Vec2;
  onPreset: (id: string) => void;
  onParams: (p: Partial<Params>) => void;
  onHyper: (h: Partial<Hyper>) => void;
  onSlices: (s: Slices) => void;
  onVectors: (v: VectorToggles) => void;
  onRace: (r: boolean) => void;
  onSolo: (k: AgentKind) => void;
  onPlay: () => void;
  onStep: () => void;
  onReset: () => void;
}

export default function ControlDock(p: ControlDockProps) {
  const { model, slices } = p;
  const zMax = model.zMax * 0.8;
  const zMin = model.kind === "saddle" ? model.zMin * 0.8 : 0;
  const setSlice = (axis: "x" | "y" | "z", patch: Partial<{ on: boolean; k: number }>) =>
    p.onSlices({ ...slices, [axis]: { ...slices[axis], ...patch } });

  return (
    <div className="px-4 pb-5 pt-4">
      <h2 className="text-[17px] font-bold text-gray-900">Controls</h2>
      <p className="mt-0.5 text-[14px] leading-snug text-gray-600">The loss of a 2-weight model, drawn as a surface in 3D.</p>

      <SectionTitle note="Each example builds a different quadric surface from different data.">1. Pick an example</SectionTitle>
      <div className="space-y-1.5">
        {PRESETS.map((pr) => {
          const active = p.activePreset === pr.id;
          return (
            <button
              key={pr.id}
              onClick={() => p.onPreset(pr.id)}
              className={`group flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left transition-colors ${
                active
                  ? "border-blue-600 bg-blue-50"
                  : "border-gray-300 bg-white hover:border-gray-500"
              }`}
            >
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[14px] font-semibold text-gray-900">{pr.title}</span>
                <span className="block text-[12px] text-gray-600">
                  {pr.note} · <span className="text-blue-800">{pr.result}</span>
                </span>
              </span>
              <ChevronRight size={15} className="shrink-0 text-gray-500 group-hover:text-gray-900" />
            </button>
          );
        })}
      </div>

      <SectionTitle note="Drag a slider and watch the surface reshape.">Change the data</SectionTitle>
      <div className="space-y-3">
        <Slider
          label="Feature scale ratio  σ₁ / σ₂"
          value={p.params.ratio}
          min={0.25}
          max={4}
          step={0.05}
          onChange={(v) => p.onParams({ ratio: v })}
        />
        <Slider
          label="Correlation  ρ"
          value={p.params.rho}
          min={0}
          max={1}
          step={0.01}
          onChange={(v) => p.onParams({ rho: v })}
        />
        <div className="flex gap-1 rounded-lg bg-gray-100 p-1 text-[13px]">
          {(["regression", "minimax"] as const).map((m) => (
            <button
              key={m}
              onClick={() => p.onParams({ mode: m })}
              className={`flex-1 rounded-md py-1 font-semibold transition-colors ${
                p.params.mode === m ? "bg-gray-200 text-gray-900" : "text-gray-600 hover:text-gray-900"
              }`}
            >
              {m === "regression" ? "Convex MSE" : "GAN saddle"}
            </button>
          ))}
        </div>
      </div>

      <SectionTitle note="Blue dots: actual y. Green rings: predictions for the ball\u2019s weights.">How well the weights fit</SectionTitle>
      <FitPlot model={model} weights={p.fitWeights} />

      <SectionTitle note="Click the surface to drop a ball, then press Play.">2. Gradient descent</SectionTitle>
      <div className="space-y-3">
        <Slider label="Learning rate  α" value={p.hyper.alpha} min={0.01} max={0.6} step={0.005} digits={3} onChange={(v) => p.onHyper({ alpha: v })} />
        <Slider label="Momentum  β" value={p.hyper.beta} min={0} max={0.99} step={0.01} onChange={(v) => p.onHyper({ beta: v })} />
        <Slider label="Speed (steps / s)" value={p.hyper.speed} min={2} max={60} step={1} digits={0} onChange={(v) => p.onHyper({ speed: v })} />

        <div className="flex items-center justify-between text-[13px] text-gray-700">
          <span>Race: gradient descent vs. momentum</span>
          <Switch on={p.race} onClick={() => p.onRace(!p.race)} label="Race mode" />
        </div>
        {!p.race && (
          <div className="flex gap-1 rounded-lg bg-gray-100 p-1 text-[13px]">
            {(["gd", "momentum"] as const).map((k) => (
              <button
                key={k}
                onClick={() => p.onSolo(k)}
                className={`flex-1 rounded-md py-1 font-semibold transition-colors ${
                  p.solo === k ? "bg-gray-200 text-gray-900" : "text-gray-600 hover:text-gray-900"
                }`}
              >
                {k === "gd" ? "Vanilla GD" : "Momentum"}
              </button>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <button
            onClick={p.onPlay}
            disabled={!p.hasBall}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg bg-blue-700 py-2 text-[14px] font-bold text-white transition-colors hover:bg-blue-800 disabled:cursor-not-allowed disabled:bg-gray-200 disabled:text-gray-500"
          >
            {p.playing ? <Pause size={15} /> : <Play size={15} />}
            {p.playing ? "Pause" : "Play"}
          </button>
          <button
            onClick={p.onStep}
            disabled={!p.hasBall}
            title="Advance one iteration"
            className="flex items-center justify-center rounded-lg border border-gray-300 px-3 text-gray-800 hover:border-gray-500 disabled:opacity-40"
          >
            <StepForward size={16} />
          </button>
          <button
            onClick={p.onReset}
            disabled={!p.hasBall}
            title="Return the ball to its starting point"
            className="flex items-center justify-center rounded-lg border border-gray-300 px-3 text-gray-800 hover:border-gray-500 disabled:opacity-40"
          >
            <RotateCcw size={16} />
          </button>
        </div>
        {!p.hasBall && <p className="text-[12px] text-gray-500">Click anywhere on the surface to drop a ball.</p>}
      </div>

      <SectionTitle note="Cut the surface with a plane to see its traces.">3. Traces</SectionTitle>
      <div className="space-y-3">
        {(
          [
            ["x", "x = k", "#059669", -DOMAIN, DOMAIN],
            ["y", "y = k", "#db2777", -DOMAIN, DOMAIN],
            ["z", "z = k", "#d97706", zMin, zMax],
          ] as const
        ).map(([axis, label, color, lo, hi]) => (
          <div key={axis}>
            <div className="mb-0.5 flex items-center justify-between">
              <span className="text-[13px] font-semibold" style={{ color }}>
                {label}
              </span>
              <Switch on={slices[axis].on} color={color} onClick={() => setSlice(axis, { on: !slices[axis].on })} label={`${label} plane`} />
            </div>
            <Slider
              label="k"
              value={slices[axis].k}
              min={lo}
              max={hi}
              step={(hi - lo) / 200}
              disabled={!slices[axis].on}
              onChange={(v) => setSlice(axis, { k: v })}
            />
          </div>
        ))}
      </div>

      <details className="mt-6 border-t border-gray-300 pt-3">
        <summary className="cursor-pointer text-[15px] font-bold text-gray-900">Arrows on the ball (optional)</summary>
      <div className="mt-2 space-y-2 text-[13px] text-gray-700">
        {(
          [
            ["normal", "Surface normal", "#0e7490"],
            ["grad", "Downhill direction (−∇f)", "#15803d"],
            ["momentum", "Previous step", "#c2410c"],
          ] as const
        ).map(([key, label, color]) => (
          <div key={key} className="flex items-center justify-between">
            <span>
              <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ background: color }} />
              {label}
            </span>
            <Switch on={p.vectors[key]} color={color} onClick={() => p.onVectors({ ...p.vectors, [key]: !p.vectors[key] })} label={label} />
          </div>
        ))}
      </div>
      </details>
    </div>
  );
}
