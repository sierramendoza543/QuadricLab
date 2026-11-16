interface Point {
  x: number;
  y: number;
}

interface ScatterProps {
  points: Point[];
  xLabel: string;
  yLabel: string;
  line?: { a: number; b: number } | null;
  curve?: Point[] | null;
}

const W = 560;
const H = 320;
const PAD = { l: 62, r: 14, t: 12, b: 46 };

const fmt = (v: number) => (Number.isInteger(v) || Math.abs(v) >= 1000 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

export default function Scatter({ points, xLabel, yLabel, line, curve }: ScatterProps) {
  const all = curve ? [...points, ...curve] : points;
  const xs = all.map((p) => p.x);
  const ys = all.map((p) => p.y);
  let [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  let [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const px = (x1 - x0) * 0.05 || 1;
  const py = (y1 - y0) * 0.05 || 1;
  x0 -= px;
  x1 += px;
  y0 -= py;
  y1 += py;
  const sx = (v: number) => PAD.l + ((v - x0) / (x1 - x0)) * (W - PAD.l - PAD.r);
  const sy = (v: number) => H - PAD.b - ((v - y0) / (y1 - y0)) * (H - PAD.t - PAD.b);
  const r1 = (v: number) => Math.round(v * 10) / 10;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full border border-gray-300 bg-white">
      <line x1={PAD.l} y1={H - PAD.b} x2={W - PAD.r} y2={H - PAD.b} stroke="#555" />
      <line x1={PAD.l} y1={PAD.t} x2={PAD.l} y2={H - PAD.b} stroke="#555" />
      {points.map((p, i) => (
        <circle key={i} cx={r1(sx(p.x))} cy={r1(sy(p.y))} r={3} fill="#1d4ed8" fillOpacity={0.6} />
      ))}
      {curve && curve.length > 1 && (
        <polyline fill="none" stroke="#c00000" strokeWidth={2.5} points={curve.map((p) => `${r1(sx(p.x))},${r1(sy(p.y))}`).join(" ")} />
      )}
      {line && <line x1={r1(sx(x0))} y1={r1(sy(line.a * x0 + line.b))} x2={r1(sx(x1))} y2={r1(sy(line.a * x1 + line.b))} stroke="#c00000" strokeWidth={2} />}
      <text x={PAD.l} y={H - PAD.b + 16} fontSize="12" fill="#555">
        {fmt(x0 + px)}
      </text>
      <text x={W - PAD.r} y={H - PAD.b + 16} fontSize="12" fill="#555" textAnchor="end">
        {fmt(x1 - px)}
      </text>
      <text x={(PAD.l + W - PAD.r) / 2} y={H - 8} fontSize="14" fill="#222" textAnchor="middle">
        {xLabel}
      </text>
      <text x={PAD.l - 6} y={H - PAD.b} fontSize="12" fill="#555" textAnchor="end">
        {fmt(y0 + py)}
      </text>
      <text x={PAD.l - 6} y={PAD.t + 10} fontSize="12" fill="#555" textAnchor="end">
        {fmt(y1 - py)}
      </text>
      <text x={14} y={(PAD.t + H - PAD.b) / 2} fontSize="14" fill="#222" textAnchor="middle" transform={`rotate(-90 14 ${(PAD.t + H - PAD.b) / 2})`}>
        {yLabel}
      </text>
    </svg>
  );
}
