"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import StepData from "@/components/builder/StepData";
import StepVariables from "@/components/builder/StepVariables";
import StepExplore from "@/components/builder/StepExplore";
import StepFit from "@/components/builder/StepFit";
import StepPredict from "@/components/builder/StepPredict";
import { MathClassify, MathDescent, MathLoss, MathQuadratic, MathSlope } from "@/components/builder/StepMath";
import { StepSummary, StepSurface } from "@/components/builder/StepSurface";
import { Table, targetColumns } from "@/lib/ml/datasets";
import { AiReview, reviewSelection } from "@/lib/ml/ai";
import { Net, TrainResult, planNetwork, train } from "@/lib/ml/engine";
import { prepare } from "@/lib/ml/profile";
import { MathBundle, buildMath, chooseModel } from "@/lib/ml/lesson";
import { pytorchCode } from "@/lib/ml/codegen";

const STEPS = ["Data", "Variables", "Explore", "Fit", "Loss", "Surface", "Derivatives", "Descent", "Quadratic", "Classify", "Summary", "Predict"];

interface Trained {
  key: string;
  result: TrainResult;
}

export default function BuilderPage() {
  const [step, setStep] = useState(0);
  const [reached, setReached] = useState(0);
  const [table, setTable] = useState<Table | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [target, setTarget] = useState(0);
  const [feats, setFeats] = useState<number[]>([]);
  const [running, setRunning] = useState(false);
  const [loss, setLoss] = useState<number[]>([]);
  const [trained, setTrained] = useState<Trained | null>(null);
  const [math, setMath] = useState<MathBundle | null>(null);
  const [review, setReview] = useState<AiReview | null>(null);
  const [pair, setPair] = useState<[number, number] | null>(null);
  const cancel = useRef(false);
  const scroller = useRef<HTMLDivElement>(null);

  const choice = useMemo(() => (table && feats.length ? chooseModel(table, target, feats, review?.extra_terms ?? []) : null), [table, target, feats, review]);
  const prepared = useMemo(() => (table && choice ? prepare(table, target, choice.terms) : null), [table, target, choice]);
  const plan = useMemo(() => planNetwork([], prepared?.d ?? 1), [prepared]);
  const key = `${table?.name}|${target}|${feats.join(",")}`;
  const fresh = trained && trained.key === key ? trained.result : null;
  const ready = !!fresh && !fresh.diverged;

  const code = useMemo(() => (prepared && choice ? pytorchCode(prepared.names, prepared.target, choice.degree, choice.cfg) : ""), [prepared, choice]);

  const rmse = useMemo(() => {
    if (!fresh || !prepared || fresh.diverged) return null;
    const out = new Net(plan).forward(fresh.theta, prepared.Xva, prepared.nva);
    let s = 0;
    for (let k = 0; k < prepared.nva; k++) s += (out[k] - prepared.yva[k]) ** 2;
    return Math.sqrt(s / prepared.nva) * prepared.yStd;
  }, [fresh, prepared, plan]);

  // the calculus numbers are worked out in the background once the model is fitted
  useEffect(() => {
    setMath(null);
    if (!fresh || fresh.diverged || !prepared) return;
    let stale = false;
    buildMath(plan, prepared, fresh, () => stale, pair).then((b) => {
      if (!stale && b) setMath(b);
    });
    return () => {
      stale = true;
    };
  }, [fresh, prepared, plan, pair]);

  const canNext = [!!table, feats.length > 0, true, ready && !!math, true, true, true, true, true, true, true, false][step];

  const goTo = (i: number) => {
    setStep(i);
    setReached((r) => Math.max(r, i));
    scroller.current?.scrollTo({ top: 0 });
  };

  const loadTable = (t: Table, n: string | null) => {
    const tc = targetColumns(t);
    const tj = tc[tc.length - 1];
    setTable(t);
    setNote(n);
    setPair(null);
    setTarget(tj);
    setFeats(t.columns.map((_, i) => i).filter((i) => i !== tj));
    setTrained(null);
    setLoss([]);
    setReached(0);
  };

  const applyReview = (r: AiReview | null) => {
    setReview(r);
    setTrained(null);
    setPair(null);
    if (r && table) {
      const sel = reviewSelection(table, r);
      if (sel.target !== null && sel.feats.length) {
        setTarget(sel.target);
        setFeats(sel.feats);
      }
    }
  };

  const changeTarget = (j: number) => {
    if (!table) return;
    setTarget(j);
    setFeats(table.columns.map((_, i) => i).filter((i) => i !== j));
  };

  const runTrain = async () => {
    if (!prepared || !choice) return;
    cancel.current = false;
    setRunning(true);
    setTrained(null);
    setLoss([]);
    const result = await train(new Net(plan), prepared, choice.cfg, (_e, t) => setLoss(t.slice()), () => cancel.current);
    setLoss(result.trainLoss);
    setTrained({ key, result });
    setRunning(false);
  };

  const waiting = <p className="text-gray-600">Working out the numbers…</p>;
  let body: React.ReactNode = null;
  if (step === 0) body = <StepData table={table} note={note} review={review} onTable={loadTable} onReview={applyReview} />;
  else if (step === 1 && table) body = <StepVariables table={table} target={target} feats={feats} review={review} onTarget={changeTarget} onFeats={setFeats} />;
  else if (step === 2 && table && choice) body = <StepExplore table={table} target={target} feats={feats} r2Linear={choice.r2Linear} />;
  else if (step === 3 && prepared && choice)
    body = (
      <StepFit
        choice={choice}
        prepared={prepared}
        code={code}
        running={running}
        loss={fresh ? fresh.trainLoss : loss}
        result={fresh}
        rmse={rmse}
        onRun={runTrain}
        onStop={() => (cancel.current = true)}
        onTry={() => goTo(11)}
      />
    );
  else if (step === 11 && table && prepared && fresh && !fresh.diverged)
    body = <StepPredict key={key} table={table} target={target} terms={choice!.terms} prepared={prepared} theta={fresh.theta} />;
  else if (step >= 4 && step <= 10) {
    if (!math) body = waiting;
    else if (step === 4) body = <MathLoss b={math} />;
    else if (step === 5) body = <StepSurface b={math} names={prepared?.names ?? []} onPair={setPair} />;
    else if (step === 6) body = <MathSlope b={math} />;
    else if (step === 7) body = <MathDescent b={math} />;
    else if (step === 8) body = <MathQuadratic b={math} />;
    else if (step === 9) body = <MathClassify b={math} />;
    else body = <StepSummary b={math} />;
  }

  const nextLabel = step === 3 && ready && !math ? "Working out the numbers…" : "Next";

  return (
    <div ref={scroller} className="fixed inset-0 overflow-y-auto bg-white text-[#1a1a1a]">
      <div className="flex min-h-full flex-col">
      <header className="border-b border-gray-300">
        <div className="mx-auto flex max-w-5xl items-baseline justify-between px-6 pt-4">
          <div>
            <span className="text-xl font-bold">Regression and Loss Surfaces</span>
            <span className="ml-3 text-gray-500">a multivariable calculus applet</span>
          </div>
          <Link href="/" className="text-blue-700 underline">
            Quadric Lab
          </Link>
        </div>
        <nav className="mx-auto flex max-w-5xl gap-1 overflow-x-auto px-6 pt-3 text-[15px]">
          {STEPS.map((s, i) => {
            const open = i <= reached || (i === 11 && ready);
            return (
              <button
                key={s}
                disabled={!open}
                onClick={() => goTo(i)}
                className={`whitespace-nowrap border-b-2 px-2 pb-2 ${
                  i === step ? "border-blue-700 font-bold text-blue-800" : open ? "border-transparent text-gray-800 hover:text-blue-800" : "border-transparent text-gray-400"
                }`}
              >
                {i + 1}. {s}
              </button>
            );
          })}
        </nav>
      </header>

      <main className="mx-auto w-full max-w-5xl flex-1 px-6 py-7">{body}</main>

      <footer className="sticky bottom-0 z-30 border-t border-gray-300 bg-white">
        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-3">
          <button disabled={step === 0} onClick={() => goTo(step - 1)} className="border border-gray-400 bg-white px-5 py-1.5 hover:bg-gray-100 disabled:opacity-40">
            Back
          </button>
          <span className="text-gray-500">
            Step {step + 1} of {STEPS.length}
          </span>
          <button
            disabled={!canNext || step === STEPS.length - 1}
            onClick={() => goTo(step + 1)}
            className="border border-blue-800 bg-blue-700 px-5 py-1.5 font-bold text-white hover:bg-blue-800 disabled:border-gray-300 disabled:bg-gray-200 disabled:text-gray-500"
          >
            {nextLabel}
          </button>
        </div>
      </footer>
      </div>
    </div>
  );
}
