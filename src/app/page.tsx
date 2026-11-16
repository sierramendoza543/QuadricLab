"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, FlaskConical, Shapes, SlidersHorizontal } from "lucide-react";
import ControlDock from "@/components/ControlDock";
import TelemetryPanel from "@/components/TelemetryPanel";
import {
  Agent,
  AgentKind,
  Params,
  PRESETS,
  Vec2,
  buildModel,
  newAgent,
  stepAgent,
  zScaleFor,
} from "@/lib/calculusMath";
import { Hyper, Slices, VectorToggles } from "@/lib/types";

const QuadricCanvas = dynamic(() => import("@/components/QuadricCanvas"), {
  ssr: false,
  loading: () => <div className="absolute inset-0 bg-[#060913]" />,
});

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function seed(start: Vec2 | null, race: boolean, solo: AgentKind): Agent[] {
  if (!start) return [];
  if (race) return [newAgent("gd", start), newAgent("momentum", start)];
  return [newAgent(solo, start)];
}

export default function Page() {
  const [params, setParams] = useState<Params>(PRESETS[1].params);
  const [hyper, setHyper] = useState<Hyper>({ alpha: PRESETS[1].alpha, beta: PRESETS[1].beta, speed: 14 });
  const [activePreset, setActivePreset] = useState<string | null>(PRESETS[1].id);
  const [slices, setSlices] = useState<Slices>({
    x: { on: false, k: 0 },
    y: { on: false, k: 0 },
    z: { on: false, k: 2 },
  });
  const [vectors, setVectors] = useState<VectorToggles>({ normal: true, grad: true, momentum: true });
  const [race, setRace] = useState(false);
  const [solo, setSolo] = useState<AgentKind>("gd");
  const [start, setStart] = useState<Vec2 | null>(null);
  const [agents, setAgents] = useState<Agent[]>([]);
  const [playing, setPlaying] = useState(false);
  const [focusId, setFocusId] = useState<AgentKind>("gd");
  const [leftOpen, setLeftOpen] = useState(true);
  const [rightOpen, setRightOpen] = useState(true);

  const model = useMemo(() => buildModel(params), [params]);
  const zScale = useMemo(() => zScaleFor(model), [model]);

  const live = useRef({ model, hyper });
  live.current = { model, hyper };

  useEffect(() => {
    if (window.innerWidth < 1024) {
      setLeftOpen(false);
      setRightOpen(false);
    }
  }, []);

  useEffect(() => {
    setAgents(seed(start, race, solo));
  }, [start, race, solo, model]);

  useEffect(() => {
    const lo = model.kind === "saddle" ? model.zMin * 0.8 : 0;
    const hi = model.zMax * 0.8;
    setSlices((s) => {
      const k = clamp(s.z.k, lo, hi);
      return k === s.z.k ? s : { ...s, z: { ...s.z, k } };
    });
  }, [model]);

  const advance = useCallback(() => {
    const { model: m, hyper: h } = live.current;
    setAgents((prev) => prev.map((a) => stepAgent(m, a, h.alpha, h.beta)));
  }, []);

  useEffect(() => {
    if (!playing) return;
    const id = setInterval(advance, 1000 / hyper.speed);
    return () => clearInterval(id);
  }, [playing, hyper.speed, advance]);

  useEffect(() => {
    if (playing && agents.length > 0 && agents.every((a) => a.status !== "running")) setPlaying(false);
  }, [agents, playing]);

  const focus: AgentKind = agents.some((a) => a.id === focusId) ? focusId : agents[0]?.id ?? "gd";
  const focused = agents.find((a) => a.id === focus);

  const applyPreset = useCallback((id: string) => {
    const pr = PRESETS.find((p) => p.id === id);
    if (!pr) return;
    const next = buildModel(pr.params);
    setParams(pr.params);
    setHyper((h) => ({ ...h, alpha: pr.alpha, beta: pr.beta }));
    setActivePreset(id);
    setSlices((s) => ({ ...s, z: { ...s.z, k: next.zMax * 0.25 } }));
    setPlaying(false);
    return next;
  }, []);

  const handlePick = useCallback((w: Vec2) => {
    setStart(w);
    setPlaying(true);
  }, []);

  return (
    <main className="flex h-screen w-screen flex-col overflow-hidden bg-white text-gray-900">
      <header className="flex items-center gap-3 border-b border-gray-300 px-4 py-2">
        <button
          onClick={() => setLeftOpen((v) => !v)}
          aria-label="Toggle controls"
          className={`border px-2 py-1.5 ${leftOpen ? "border-blue-700 bg-blue-50 text-blue-800" : "border-gray-300 text-gray-600 hover:text-black"}`}
        >
          <SlidersHorizontal size={16} />
        </button>
        <div>
          <span className="text-xl font-bold">Quadric Lab</span>
          <span className="ml-3 hidden text-gray-500 sm:inline">Chapter 12 surfaces and gradient descent</span>
        </div>
        <div className="ml-auto flex items-center gap-2 text-[15px]">
          <Link href="/builder" className="flex items-center gap-2 border border-gray-300 px-3 py-1.5 hover:bg-gray-100">
            <FlaskConical size={15} />
            <span className="hidden sm:inline">Build your own model</span>
            <span className="sm:hidden">Builder</span>
          </Link>
          <Link href="/explorer" className="flex items-center gap-2 border border-gray-300 px-3 py-1.5 hover:bg-gray-100">
            <Shapes size={15} />
            <span className="hidden sm:inline">Quadric explorer</span>
            <span className="sm:hidden">Explorer</span>
          </Link>
          <a href="https://github.com/sierramendoza543/QuadricLab" target="_blank" rel="noopener noreferrer" className="border border-gray-300 px-3 py-1.5 hover:bg-gray-100">
            GitHub
          </a>
          <button
            onClick={() => setRightOpen((v) => !v)}
            aria-label="Toggle telemetry"
            className={`border px-2 py-1.5 ${rightOpen ? "border-blue-700 bg-blue-50 text-blue-800" : "border-gray-300 text-gray-600 hover:text-black"}`}
          >
            <Activity size={16} />
          </button>
        </div>
      </header>

      <div className="relative flex min-h-0 flex-1">
        {leftOpen && (
          <aside className="absolute inset-y-0 left-0 z-10 w-[min(320px,100vw)] overflow-y-auto border-r border-gray-300 bg-white lg:static lg:w-[320px] lg:shrink-0">
            <ControlDock
              model={model}
              params={params}
              hyper={hyper}
              slices={slices}
              vectors={vectors}
              race={race}
              solo={solo}
              playing={playing}
              hasBall={agents.length > 0}
              activePreset={activePreset}
              fitWeights={focused ? focused.pos : [0, 0]}
              onPreset={applyPreset}
              onParams={(p) => {
                setParams((prev) => ({ ...prev, ...p }));
                setActivePreset(null);
              }}
              onHyper={(h) => setHyper((prev) => ({ ...prev, ...h }))}
              onSlices={setSlices}
              onVectors={setVectors}
              onRace={setRace}
              onSolo={setSolo}
              onPlay={() => setPlaying((v) => !v)}
              onStep={() => {
                setPlaying(false);
                advance();
              }}
              onReset={() => {
                setPlaying(false);
                setAgents(seed(start, race, solo));
              }}
            />
          </aside>
        )}

        <div className="relative min-w-0 flex-1">
          <QuadricCanvas
            className="absolute inset-0"
            model={model}
            zScale={zScale}
            agents={agents}
            focusId={focus}
            vectors={vectors}
            slices={slices}
            onPick={handlePick}
          />
          <div className="pointer-events-none absolute bottom-2 left-1/2 -translate-x-1/2 text-[13px] text-gray-500">
            drag to rotate · scroll to zoom · click the surface to drop a ball
          </div>
        </div>

        {rightOpen && (
          <aside className="absolute inset-y-0 right-0 z-10 w-[min(350px,100vw)] overflow-y-auto border-l border-gray-300 bg-white lg:static lg:w-[350px] lg:shrink-0">
            <TelemetryPanel
              model={model}
              agents={agents}
              focusId={focus}
              onFocus={setFocusId}
              slices={slices}
            />
          </aside>
        )}
      </div>

    </main>
  );
}
