import { Model, SurfaceKind, traceQuadratic } from "./calculusMath";

export const KIND_LABEL: Record<SurfaceKind, string> = {
  circular: "Circular paraboloid",
  elliptic: "Elliptic paraboloid",
  cylinder: "Parabolic cylinder",
  saddle: "Hyperbolic paraboloid",
};

export const KIND_COLOR: Record<SurfaceKind, string> = {
  circular: "#0e7490",
  elliptic: "#4338ca",
  cylinder: "#b45309",
  saddle: "#be185d",
};

const tiny = (x: number) => Math.abs(x) < 0.005;

export function num(x: number, d = 2) {
  const v = tiny(x) && d <= 2 ? 0 : x;
  return v.toFixed(d);
}

function shifted(name: string, c: number) {
  if (tiny(c)) return name;
  return c > 0 ? `(${name} - ${num(c)})` : `(${name} + ${num(-c)})`;
}

function squared(name: string, c: number) {
  return tiny(c) ? `${name}^2` : `${shifted(name, c)}^2`;
}

function signed(x: number, d = 2) {
  return x < 0 ? `- ${num(-x, d)}` : `+ ${num(x, d)}`;
}

function axisAligned(theta: number): "none" | "w1" | "w2" {
  if (Math.abs(Math.sin(theta)) < 0.02) return "w1";
  if (Math.abs(Math.cos(theta)) < 0.02) return "w2";
  return "none";
}

// returns one or more TeX lines for the canonical form at the current data
export function canonicalLatex(m: Model): string[] {
  const [h, k] = m.center;
  const c = Math.cos(m.theta);
  const s = Math.sin(m.theta);
  const a = num(m.semiA);
  const b = num(m.semiB);

  if (m.kind === "cylinder") {
    const off = c * h + s * k;
    return [
      `z = ${num(m.lam1)}\\,\\left(${num(c)}\\,w_1 ${signed(s)}\\,w_2 ${signed(-off)}\\right)^2`,
      `\\lambda_2 = ${num(m.lam2, 3)} \\approx 0 \;\\Rightarrow\; \\text{no curvature along } \\langle ${num(-s)},\\, ${num(c)} \\rangle`,
    ];
  }

  if (m.kind === "circular") {
    return [`z = \\frac{${squared("w_1", h)}}{${a}^2} + \\frac{${squared("w_2", k)}}{${a}^2}`];
  }

  const align = axisAligned(m.theta);
  if (align !== "none") {
    const saddle = m.kind === "saddle";
    // coefficients of (w1-h)^2 and (w2-k)^2
    const q1 = align === "w1" ? (saddle ? -m.lam1 : m.lam1) : m.lam2;
    const q2 = align === "w1" ? m.lam2 : saddle ? -m.lam1 : m.lam1;
    const term = (q: number, name: string, ctr: number) =>
      `\\frac{${squared(name, ctr)}}{${num(1 / Math.sqrt(Math.abs(q) || 1e-3))}^2}`;
    const t1 = { q: q1, tex: term(q1, "w_1", h) };
    const t2 = { q: q2, tex: term(q2, "w_2", k) };
    if (!saddle) return [`z = ${t1.tex} + ${t2.tex}`];
    const pos = t1.q > 0 ? t1 : t2;
    const neg = t1.q > 0 ? t2 : t1;
    return [`z = ${pos.tex} - ${neg.tex}`];
  }

  const u1 = `u_1 = ${num(c)}\\,${shifted("w_1", h)} ${signed(s)}\\,${shifted("w_2", k)}`;
  const u2 = `u_2 = ${num(-s)}\\,${shifted("w_1", h)} ${signed(c)}\\,${shifted("w_2", k)}`;
  const main =
    m.kind === "saddle"
      ? `z = \\frac{u_2^2}{${b}^2} - \\frac{u_1^2}{${a}^2}`
      : `z = \\frac{u_1^2}{${a}^2} + \\frac{u_2^2}{${b}^2}`;
  return [main, u1, u2];
}

export function traceLatex(m: Model, axis: "x" | "y", k: number) {
  const { A, B, C } = traceQuadratic(m, axis, k);
  const v = axis === "x" ? "w_2" : "w_1";
  const parts: string[] = [];
  if (!tiny(A)) parts.push(`${num(A)}\\,${v}^2`);
  if (!tiny(B)) parts.push(`${parts.length ? (B < 0 ? "- " : "+ ") : B < 0 ? "-" : ""}${num(Math.abs(B))}\\,${v}`);
  const cs = parts.length ? `${C < 0 ? "- " : "+ "}${num(Math.abs(C))}` : num(C);
  parts.push(cs);
  const kind = tiny(A) ? (tiny(B) ? "constant (a horizontal line)" : "line") : A > 0 ? "parabola, opens up" : "parabola, opens down";
  return { tex: `z = ${parts.join(" ")}`, kind };
}

export function levelLatex(m: Model, level: number) {
  const [h, k] = m.center;
  const c = Math.cos(m.theta);
  const s = Math.sin(m.theta);
  const root = (x: number) => Math.sqrt(Math.max(x, 0));

  if (m.kind === "saddle") {
    if (Math.abs(level) < 0.02) return { tex: "u_2 = \\pm\\sqrt{\\lambda_1/\\lambda_2}\\,u_1", kind: "pair of crossing lines" };
    const pa = root(Math.abs(level) / m.lam1);
    const pb = root(Math.abs(level) / m.lam2);
    return level > 0
      ? { tex: `\\frac{u_2^2}{${num(pb)}^2} - \\frac{u_1^2}{${num(pa)}^2} = 1`, kind: "hyperbola" }
      : { tex: `\\frac{u_1^2}{${num(pa)}^2} - \\frac{u_2^2}{${num(pb)}^2} = 1`, kind: "hyperbola (conjugate)" };
  }
  if (level < -1e-9) return { tex: "\\text{empty set}", kind: "no points below the vertex" };
  if (level < 0.02) return { tex: `(w_1, w_2) = (${num(h)}, ${num(k)})`, kind: "single point (vertex)" };

  if (m.kind === "cylinder") {
    const off = c * h + s * k;
    const d = root(level / m.lam1);
    return {
      tex: `${num(c)}\\,w_1 ${signed(s)}\\,w_2 = ${num(off)} \\pm ${num(d)}`,
      kind: "pair of parallel lines",
    };
  }
  if (m.kind === "circular") {
    return { tex: `${squared("w_1", h)} + ${squared("w_2", k)} = ${num(level * m.semiA * m.semiA)}`, kind: "circle" };
  }
  return {
    tex: `\\frac{u_1^2}{${num(root(level / m.lam1))}^2} + \\frac{u_2^2}{${num(root(level / m.lam2))}^2} = 1`,
    kind: "ellipse",
  };
}

export const KIND_MEANING: Record<SurfaceKind, string> = {
  circular: "The loss curves equally in every direction. Steepest descent heads straight for the minimum.",
  elliptic: "The loss is steep in one direction and shallow in the other. Steepest descent can zig-zag across the steep direction.",
  cylinder: "The loss does not curve along one direction, so a whole line of weights gives the same minimum loss.",
  saddle: "The surface curves up in one direction and down in the other, so it has no minimum, only a saddle point.",
};
