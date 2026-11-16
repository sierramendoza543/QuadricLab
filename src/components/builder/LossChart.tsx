const W = 560;
const H = 260;
const PAD = { l: 56, r: 14, t: 12, b: 40 };

export default function LossChart({ loss }: { loss: number[] }) {
  const vals = loss.filter((v) => v > 0 && Number.isFinite(v));
  if (loss.length < 2 || vals.length === 0) {
    return <div className="flex h-[260px] items-center justify-center border border-dashed border-gray-300 text-gray-500">Loss versus iteration will appear here.</div>;
  }
  const lo = Math.log10(Math.min(...vals));
  const hi = Math.log10(Math.max(...vals));
  const span = Math.max(hi - lo, 0.3);
  const n = loss.length;
  const x = (i: number) => PAD.l + (i / (n - 1)) * (W - PAD.l - PAD.r);
  const y = (v: number) => PAD.t + (1 - (Math.log10(Math.max(v, 1e-12)) - lo) / span) * (H - PAD.t - PAD.b);
  const d = loss.map((v, i) => `${i === 0 ? "M" : "L"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const ticks = [0, 0.5, 1].map((f) => lo + span * f);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full border border-gray-300 bg-white">
      {ticks.map((t) => {
        const yy = PAD.t + (1 - (t - lo) / span) * (H - PAD.t - PAD.b);
        const v = Math.pow(10, t);
        return (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={yy} y2={yy} stroke="#e5e5e5" />
            <text x={PAD.l - 6} y={yy + 4} textAnchor="end" fontSize="12" fill="#555">
              {v >= 0.01 ? v.toFixed(2) : v.toExponential(0)}
            </text>
          </g>
        );
      })}
      <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="#555" />
      <path d={d} fill="none" stroke="#1d4ed8" strokeWidth={2} />
      <text x={PAD.l} y={H - 22} fontSize="12" fill="#555">
        1
      </text>
      <text x={W - PAD.r} y={H - 22} fontSize="12" fill="#555" textAnchor="end">
        {n}
      </text>
      <text x={(PAD.l + W - PAD.r) / 2} y={H - 6} fontSize="14" fill="#222" textAnchor="middle">
        iteration
      </text>
      <text x={14} y={(PAD.t + H - PAD.b) / 2} fontSize="14" fill="#222" textAnchor="middle" transform={`rotate(-90 14 ${(PAD.t + H - PAD.b) / 2})`}>
        loss L (log scale)
      </text>
    </svg>
  );
}
