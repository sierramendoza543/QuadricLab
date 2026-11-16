"use client";

import { FIT_LABEL, FitKind, fitEquation } from "@/lib/ml/landscape";
import { MathBundle } from "@/lib/ml/lesson";
import Paper from "./Paper";
import Tex from "../Tex";

const T = String.raw;
const f = (x: number, d = 3) => (Math.abs(x) < 0.5 * Math.pow(10, -d) ? 0 : x).toFixed(d);
// " + 1.500 w" or " - 0.200 w", so a negative number never appears as "+ -0.2"
const term = (x: number, v: string, d = 3) => ` ${x < 0 ? "-" : "+"} ${f(Math.abs(x), d)}${v ? String.raw`\,${v}` : ""}`;
const rowStyle = "border-b border-gray-200 last:border-0";

interface Props {
  b: MathBundle;
  target?: string;
}

function Heading({ n, title }: { n: number; title: string }) {
  return <h2 className="mb-3 text-2xl font-bold">{`${n}. ${title}`}</h2>;
}

const P = ({ children }: { children: React.ReactNode }) => <p className="my-3 text-[17px] leading-relaxed">{children}</p>;

export function MathLoss({ b }: Props) {
  const L = b.lesson;
  const [a, c] = L.start;
  const cls = b.task === "classification";
  return (
    <div>
      <Heading n={5} title="The loss function" />
      {cls ? (
        <>
          <P>
            To judge a classifier, compare the predicted probability <Tex tex={String.raw`\hat p_i`} /> with the actual label <Tex tex={"y_i"} /> (1 or 0). The <b>cross-entropy</b>, also called <b>log-loss</b>, charges <Tex tex={String.raw`-\ln\hat p`} /> when the label is 1 and <Tex tex={String.raw`-\ln(1-\hat p)`} /> when it is 0. A confident wrong answer costs a lot, and a confident right answer costs almost nothing. We call the average the <b>loss</b>, L.
          </P>
          <Paper lines={[T`L = -\frac{1}{N}\sum_{i=1}^{N}\Big[\,y_i\ln\hat p_i + (1-y_i)\ln(1-\hat p_i)\Big]`, T`\hat p_i = \sigma(z_i),\qquad z_i = b + w_1x_{i1}+w_2x_{i2}+\cdots`]} />
        </>
      ) : (
      <>
      <P>
        To judge a model, compare each prediction <Tex tex={String.raw`\hat y_i`} /> with the actual value <Tex tex={"y_i"} />. The <b>mean squared error</b> averages the squared differences. We call it the <b>loss</b>, L.
      </P>
      <Paper lines={[T`L = \frac{1}{N}\sum_{i=1}^{N}\left(y_i-\hat y_i\right)^2`]} />
      </>
      )}
      <P>
        Example. Try the weights <Tex tex={`w_1=${f(a)}`} /> (on {L.names[0]}) and <Tex tex={`w_2=${f(c)}`} /> (on {L.names[1]})
        {L.frozen ? ", with every other weight held at its fitted value" : ""}. For the first three rows:
      </P>
      <table className="mb-1 w-full max-w-xl border border-gray-300 text-left text-[16px]">
        <thead className="bg-gray-100">
          <tr>
            {(cls ? ["i", "y", "z", "p̂ = σ(z)", "loss of this row"] : ["i", "y", "ŷ", "y − ŷ", "(y − ŷ)²"]).map((h) => (
              <th key={h} className="border-b border-gray-300 px-3 py-1.5">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {L.rows.map((r, i) => (
            <tr key={i} className={rowStyle}>
              <td className="px-3 py-1">{i + 1}</td>
              <td>{f(r.y, cls ? 0 : 3)}</td>
              {cls ? (
                <>
                  <td>{f(r.z)}</td>
                  <td>{f(r.yhat)}</td>
                  <td>{f(r.miss2)}</td>
                </>
              ) : (
                <>
                  <td>{f(r.yhat)}</td>
                  <td>{f(r.miss)}</td>
                  <td>{f(r.miss2)}</td>
                </>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mb-1 text-sm text-gray-500">{cls ? "Each row's loss is −ln p̂ if y = 1 and −ln(1 − p̂) if y = 0." : "y is standardized, so 0 is the mean of the data."}</p>
      <Paper
        lines={[
          T`L &= \frac{1}{${L.n}}\left(${f(L.rows[0].miss2)} + ${f(L.rows[1].miss2)} + ${f(L.rows[2].miss2)} + \cdots\right)`,
          T`&= ${f(L.L0, 3)}`,
        ]}
      />
      <P>
        Other weights give other predictions and a different loss, so L is a function of the weights: <Tex tex={"L(w_1,\\,w_2)"} />. Fitting the model means finding the weights that make L as small as possible.
      </P>
    </div>
  );
}

export function MathSlope({ b }: Props) {
  const L = b.lesson;
  const cls = b.task === "classification";
  const [a, c] = L.start;
  const h = L.h;
  return (
    <div>
      <Heading n={7} title="Partial derivatives" />
      <P>
        To lower L we need to know how it changes when the weights change. The <b>partial derivative</b> <Tex tex={String.raw`\partial L/\partial w_1`} /> is the slope of the surface in the <Tex tex={"w_1"} /> direction, with <Tex tex={"w_2"} /> held constant.
      </P>
      <Paper lines={[T`\frac{\partial L}{\partial w_1} = \lim_{h\to 0}\frac{L(w_1+h,\,w_2)-L(w_1,\,w_2)}{h}`]} />
      <P>
        Evaluate it at the starting guess <Tex tex={`(w_1,w_2)=(${f(a)},\\,${f(c)})`} /> with <Tex tex={`h=${h}`} />:
      </P>
      <Paper
        lines={[
          T`\frac{\partial L}{\partial w_1} &\approx \frac{L(${f(a + h)},\,${f(c)})-L(${f(a)},\,${f(c)})}{${h}}`,
          T`&= \frac{${f(L.L1, 4)}-${f(L.L0, 4)}}{${h}}`,
          T`&= ${f(L.d1, 2)}`,
        ]}
      />
      <P>
        The same for <Tex tex={"w_2"} />, holding <Tex tex={"w_1"} /> constant:
      </P>
      <Paper
        lines={[
          T`\frac{\partial L}{\partial w_2} &\approx \frac{L(${f(a)},\,${f(c + h)})-L(${f(a)},\,${f(c)})}{${h}}`,
          T`&= \frac{${f(L.L2, 4)}-${f(L.L0, 4)}}{${h}}`,
          T`&= ${f(L.d2, 2)}`,
        ]}
      />
      <P>
        Collect the two partial derivatives into the <b>gradient</b>. (A much smaller <Tex tex={"h"} /> sharpens the estimate.)
      </P>
      <Paper lines={[T`\nabla L(${f(a)},\,${f(c)}) = \left\langle \frac{\partial L}{\partial w_1},\,\frac{\partial L}{\partial w_2}\right\rangle \approx \langle ${f(L.g1, 3)},\ ${f(L.g2, 3)}\rangle`]} />
      <P>
        The gradient points in the direction of steepest <i>increase</i>, so <Tex tex={String.raw`-\nabla L`} /> points toward steepest descent.{" "}
        {L.g1 < 0 ? "Since ∂L/∂w₁ is negative, increasing w₁ lowers the loss." : "Since ∂L/∂w₁ is positive, decreasing w₁ lowers the loss."}
      </P>
      <details className="border border-gray-300 p-3">
        <summary className="cursor-pointer font-bold">Check with the exact formula</summary>
        {cls ? (
          <>
            <p className="mt-2 text-[16px]">
              For one row, the chain rule gives <Tex tex={String.raw`\partial\ell/\partial z = \hat p - y`} />: the derivative of the sigmoid cancels against the derivative of the logarithm, which is why log-loss is paired with the sigmoid. Then <Tex tex={String.raw`\partial z/\partial w = x`} />, and averaging over all rows:
            </p>
            <Paper
              lines={[
                T`\frac{\partial L}{\partial w_1} &= \frac{1}{N}\sum_{i}(\hat p_i-y_i)\,x_{i1} = ${f(L.exact[0], 3)}`,
                T`\frac{\partial L}{\partial w_2} &= \frac{1}{N}\sum_{i}(\hat p_i-y_i)\,x_{i2} = ${f(L.exact[1], 3)}`,
              ]}
            />
          </>
        ) : (
          <>
            <p className="mt-2 text-[16px]">
              For one row, the chain rule gives the derivative of <Tex tex={"(y-\\hat y)^2"} /> as <Tex tex={"2(\\hat y-y)\\,x"} />, where <Tex tex={"x"} /> is the predictor that the weight multiplies. Averaging over all rows:
            </p>
            <Paper
              lines={[
                T`\frac{\partial L}{\partial w_1} &= \frac{2}{N}\sum_{i}(\hat y_i-y_i)\,x_{i1} = ${f(L.exact[0], 3)}`,
                T`\frac{\partial L}{\partial w_2} &= \frac{2}{N}\sum_{i}(\hat y_i-y_i)\,x_{i2} = ${f(L.exact[1], 3)}`,
              ]}
            />
          </>
        )}
        <p className="text-[16px]">These match the estimates above.</p>
      </details>
    </div>
  );
}

export function MathDescent({ b }: Props) {
  const L = b.lesson;
  const [a, c] = L.start;
  const shown = Array.from(new Set([0, 1, 2, 3, 4, 5, L.steps.length - 1])).filter((k) => k < L.steps.length);
  return (
    <div>
      <Heading n={8} title="Gradient descent" />
      <P>
        Since <Tex tex={String.raw`-\nabla L`} /> points downhill, move the weights a small step that way. The step size <Tex tex={String.raw`\alpha`} /> is called the <b>learning rate</b>.
      </P>
      <Paper lines={[T`\mathbf{w}_{\text{new}} = \mathbf{w} - \alpha\,\nabla L`]} />
      <P>
        With <Tex tex={`\\alpha=${L.alpha}`} /> and the gradient from the previous screen:
      </P>
      <Paper
        lines={[
          T`w_1 &= ${f(a)} - ${L.alpha}\,(${f(L.g1)}) = ${f(L.next[0])}`,
          T`w_2 &= ${f(c)} - ${L.alpha}\,(${f(L.g2)}) = ${f(L.next[1])}`,
          T`L(${f(L.next[0])},\,${f(L.next[1])}) &= ${f(L.Lnext, 4)} \;<\; ${f(L.L0, 4)} = L(${f(a)},\,${f(c)})`,
        ]}
      />
      <P>The loss went down. Repeating the step is gradient descent. The first iterations from this starting guess:</P>
      <table className="mb-3 w-full max-w-md border border-gray-300 text-left text-[16px]">
        <thead className="bg-gray-100">
          <tr>
            {["step", "w₁", "w₂", "L"].map((h) => (
              <th key={h} className="border-b border-gray-300 px-3 py-1.5">
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="tabular-nums">
          {shown.map((k) => (
            <tr key={k} className={rowStyle}>
              <td className="px-3 py-1">{L.steps[k].k}</td>
              <td>{f(L.steps[k].w1)}</td>
              <td>{f(L.steps[k].w2)}</td>
              <td>{f(L.steps[k].L, 4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <P>The loss keeps shrinking and the weights settle near the minimum. This is exactly what the “Fit model” button did, for all the weights at once.</P>
    </div>
  );
}

export function MathQuadratic({ b }: Props) {
  const L = b.lesson;
  const { A, B, C, D, E, F } = L.quad;
  const h11 = 2 * A;
  const h22 = 2 * C;
  const cls = b.task === "classification";
  return (
    <div>
      <Heading n={9} title={cls ? "The loss is close to a quadratic function" : "The loss is a quadratic function"} />
      {cls ? (
        <>
          <P>
            In Chapter 12, a <b>quadric surface</b> is the graph of a second-degree equation in x, y and z. For a regression the squared error is exactly a second-degree polynomial in the weights. Log-loss is not: the sigmoid is not linear, and far from the minimum the loss grows like a straight line, not a parabola.
          </P>
          <P>
            Near the minimum, though, any smooth function is well approximated by its <b>second-degree Taylor polynomial</b>, and that polynomial is a quadric. We fit it to the loss values in the middle of the surface:
          </P>
          <Paper lines={[T`L(w_1,w_2) \approx A\,w_1^2 + B\,w_1w_2 + C\,w_2^2 + D\,w_1 + E\,w_2 + F`]} />
          <p className="mb-3 text-[15px] text-gray-600">
            Quality of the approximation over that window: R² = {b.fit.r2.toFixed(3)}. The closer to 1, the more the loss behaves like a true quadric there.
          </p>
        </>
      ) : (
      <P>
        In Chapter 12, a <b>quadric surface</b> is the graph of a second-degree equation in x, y and z. Here the predictions <Tex tex={"\\hat y"} /> are linear in the weights, so the error <Tex tex={"y-\\hat y"} /> is linear too, and squaring it makes L a second-degree polynomial in <Tex tex={"w_1"} /> and <Tex tex={"w_2"} />:
      </P>
      )}
      {!cls && <Paper lines={[T`L(w_1,w_2) = A\,w_1^2 + B\,w_1w_2 + C\,w_2^2 + D\,w_1 + E\,w_2 + F`]} />}
      <P>For this model the coefficients are:</P>
      <Paper
        lines={[
          T`L(w_1,w_2) &= ${f(A)}\,w_1^2${term(B, "w_1w_2")}${term(C, "w_2^2")}`,
          T`&\quad ${term(D, "w_1")}${term(E, "w_2")}${term(F, "")}`,
        ]}
      />
      <P>
        Second partial derivatives measure how fast the slope changes, which is the curvature. Differentiating the quadratic twice:
      </P>
      <Paper
        lines={[
          T`L_{w_1w_1} &= 2A = ${f(h11)}`,
          T`L_{w_2w_2} &= 2C = ${f(h22)}`,
          T`L_{w_1w_2} &= B = ${f(B)}`,
        ]}
      />
      <P>{cls ? "In the quadratic approximation they do not depend on w₁ or w₂. Put them" : "They do not depend on w₁ or w₂, because L is quadratic. Put them"} in the <b>Hessian matrix</b>:</P>
      <Paper lines={[T`H = \begin{bmatrix} L_{w_1w_1} & L_{w_1w_2} \\ L_{w_1w_2} & L_{w_2w_2} \end{bmatrix} = \begin{bmatrix} ${f(h11)} & ${f(B)} \\ ${f(B)} & ${f(h22)} \end{bmatrix}`]} />
      {cls && L.hess && (
        <details className="mt-3 border border-gray-300 p-3">
          <summary className="cursor-pointer font-bold">Check with the exact formula</summary>
          <p className="mt-2 text-[16px]">
            Differentiating <Tex tex={String.raw`\partial L/\partial w=\frac1N\sum(\hat p-y)x`} /> once more uses <Tex tex={String.raw`\sigma'(z)=\hat p(1-\hat p)`} />, so at the fitted weights:
          </p>
          <Paper
            lines={[
              T`L_{w_1w_1} &= \frac{1}{N}\sum_i \hat p_i(1-\hat p_i)\,x_{i1}^2 = ${f(L.hess[0])}`,
              T`L_{w_1w_2} &= \frac{1}{N}\sum_i \hat p_i(1-\hat p_i)\,x_{i1}x_{i2} = ${f(L.hess[1])}`,
              T`L_{w_2w_2} &= \frac{1}{N}\sum_i \hat p_i(1-\hat p_i)\,x_{i2}^2 = ${f(L.hess[2])}`,
            ]}
          />
          <p className="text-[16px]">These are the exact second derivatives at the minimum. Compare them with 2A, B and 2C from the fit above. They agree when the surface is close to a parabola across the whole window, and differ where it is not, for example along a weight whose sigmoid has saturated.</p>
        </details>
      )}
    </div>
  );
}

const FORMS: { kind: FitKind; cond: string; name: string; eq: string }[] = [
  { kind: "circular", cond: "μ₁ ≈ μ₂ > 0", name: "Circular paraboloid", eq: T`z=\frac{x^2+y^2}{a^2}` },
  { kind: "elliptic", cond: "μ₁ > μ₂ > 0", name: "Elliptic paraboloid", eq: T`\frac{z}{c}=\frac{x^2}{a^2}+\frac{y^2}{b^2}` },
  { kind: "cylinder", cond: "μ₁ > 0, μ₂ = 0", name: "Parabolic cylinder", eq: T`z=\frac{x^2}{a^2}` },
  { kind: "saddle", cond: "μ₁ > 0 > μ₂", name: "Hyperbolic paraboloid", eq: T`\frac{z}{c}=\frac{x^2}{a^2}-\frac{y^2}{b^2}` },
];

export function MathClassify({ b }: Props) {
  const { fit, lesson: L } = b;
  const h11 = 2 * L.quad.A;
  const h22 = 2 * L.quad.C;
  const h12 = L.quad.B;
  const tr = h11 + h22;
  const det = h11 * h22 - h12 * h12;
  const root = Math.sqrt(Math.max(tr * tr - 4 * det, 0));
  const deg = (fit.angle * 180) / Math.PI;
  const flat = !Number.isFinite(fit.semiB);

  return (
    <div>
      <Heading n={10} title="Classifying the surface" />
      <P>
        The eigenvalues <Tex tex={"\\mu_1,\\mu_2"} /> of H are the curvatures along the surface&apos;s two principal axes. For a 2×2 matrix they come from the trace and determinant:
      </P>
      <Paper
        lines={[
          T`\operatorname{tr}H &= ${f(h11)} + ${f(h22)} = ${f(tr)}`,
          T`\det H &= (${f(h11)})(${f(h22)}) - (${f(h12)})^2 = ${f(det)}`,
          T`\mu &= \frac{\operatorname{tr}H \pm \sqrt{(\operatorname{tr}H)^2 - 4\det H}}{2}`,
          T`\mu_1 &= \frac{${f(tr)} + ${f(root)}}{2} = ${f(fit.mu1)}, \qquad \mu_2 = \frac{${f(tr)} - ${f(root)}}{2} = ${f(fit.mu2)}`,
        ]}
      />
      <P>The signs of the eigenvalues decide which quadric surface (Stewart §12.6) we have:</P>
      <table className="mb-3 w-full max-w-xl border border-gray-300 text-left text-[16px]">
        <tbody>
          {FORMS.map((m) => (
            <tr key={m.kind} className={`border-b border-gray-200 last:border-0 ${m.kind === fit.kind ? "bg-yellow-100 font-bold" : ""}`}>
              <td className="px-3 py-1.5">{m.cond}</td>
              <td className="px-3">{m.name}</td>
              <td className="px-3">
                <Tex tex={m.eq} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mb-3 text-[16px] text-gray-700">
        Here μ₂/μ₁ = {Number.isFinite(fit.mu2 / fit.mu1) ? f(fit.mu2 / fit.mu1) : "n/a"}. Because of rounding in real data, we treat a ratio above 0.85 as “equal” and a ratio below 0.05 as “zero”.
      </p>
      <P>
        Along each axis, <Tex tex={String.raw`L-z_0=\tfrac12\mu u^2`} />, so the standard form has <Tex tex={String.raw`a^2=2/\mu_1`} /> and <Tex tex={String.raw`b^2=2/\mu_2`} />:
      </P>
      <Paper
        lines={[
          T`a &= \sqrt{2/\mu_1} = \sqrt{2/${f(fit.mu1)}} = ${f(fit.semiA)}`,
          flat ? T`b &:\ \mu_2 \approx 0,\ \text{so } L \text{ does not depend on } u_2` : T`b &= \sqrt{2/\mu_2} = \sqrt{2/${f(fit.mu2)}} = ${f(fit.semiB)}`,
          T`${fitEquation(fit).replace(/ = /, " &= ")}`,
        ]}
      />
      <P>
        The two weights belong to <b>{L.names[0]}</b> and <b>{L.names[1]}</b>
        {L.rho !== 0 ? <> (correlation {f(L.rho, 2)} between those terms)</> : ""}. The ratio μ₁/μ₂ = {Number.isFinite(fit.condition) ? f(fit.condition, 1) : "∞"} says how much steeper the bowl is along one axis than the other: the larger it is, the more a descent path zig-zags.
      </P>
      <p className="mb-3 text-[16px] text-gray-700">
        {b.task === "classification"
          ? "A saddle (hyperbolic paraboloid) cannot appear here either. The Hessian of the log-loss is (1/N) Σ p̂(1 − p̂) x xᵀ. Every weight p̂(1 − p̂) is positive, so the loss is convex and neither eigenvalue is negative. One thing is new: if the two classes can be separated perfectly by a line, the curvature along the separating direction shrinks toward 0, the minimum runs off to infinity, and the surface looks like a trough that keeps getting flatter."
          : "A saddle (hyperbolic paraboloid) cannot appear here. The mean squared error of a model that is linear in its weights is convex, so neither eigenvalue is ever negative."} Saddles show up in min–max problems such as GANs, which the Quadric Explorer covers.
      </p>
      <p className="text-[16px] text-gray-700">
        Here <Tex tex={"u_1,u_2"} /> are the coordinates <Tex tex={"(w_1-w_1^*,\\ w_2-w_2^*)"} /> rotated by θ = {f(deg, 1)}°, where <Tex tex={`(w_1^*,w_2^*)=(${f(L.best[0])},\\,${f(L.best[1])})`} /> is the minimum.
        So the loss surface is a <b>{FIT_LABEL[fit.kind].toLowerCase()}</b>.
      </p>
    </div>
  );
}
