import { Model, Params, SurfaceKind, Vec2 } from "./calculusMath";

export type Kind = SurfaceKind;
export const KINDS: Kind[] = ["circular", "elliptic", "cylinder", "saddle"];

export interface Story {
  kind: Kind;
  name: string;
  stewart: string; // TeX of the standard form
  params: Params;
  alpha: number;
  start: Vec2;
  scenario: string;
  oneLine: string;
  dataTitle: string;
  cols: [string, string, string];
  dataNote: string;
  notice: string;
}

export const STORIES: Record<Kind, Story> = {
  circular: {
    kind: "circular",
    name: "Circular paraboloid",
    stewart: String.raw`z=\frac{x^2+y^2}{a^2}`,
    params: { ratio: 1, rho: 0, mode: "regression" },
    alpha: 0.2,
    start: [-2, 1.8],
    scenario: "Exam scores",
    oneLine: "Two predictors with the same spread and no correlation.",
    dataTitle: "Exam scores",
    cols: ["hours studied", "hours slept", "exam score"],
    dataNote: "Both predictors are standardized and uncorrelated, so neither one changes how much the other matters.",
    notice: "The loss curves equally in every direction, so each step points almost straight at the minimum.",
  },
  elliptic: {
    kind: "elliptic",
    name: "Elliptic paraboloid",
    stewart: String.raw`\frac{z}{c}=\frac{x^2}{a^2}+\frac{y^2}{b^2}`,
    params: { ratio: 3, rho: 0, mode: "regression" },
    alpha: 0.2,
    start: [-2, 1.8],
    scenario: "House prices",
    oneLine: "Two predictors measured on different scales.",
    dataTitle: "House prices",
    cols: ["square feet", "bedrooms", "price"],
    dataNote: "The first predictor varies 3 times as much as the second (values are centered but not rescaled). Square feet dominates, bedrooms barely moves.",
    notice: "The bowl is steep along w₁ and shallow along w₂. Each step overshoots across the steep direction, so the ball zig-zags while it crawls along the shallow one.",
  },
  cylinder: {
    kind: "cylinder",
    name: "Parabolic cylinder",
    stewart: String.raw`z=\frac{x^2}{a^2}`,
    params: { ratio: 1, rho: 1, mode: "regression" },
    alpha: 0.15,
    start: [-2, 1.8],
    scenario: "Temperature in °C and °F",
    oneLine: "Two predictors that carry exactly the same information.",
    dataTitle: "Energy use and temperature",
    cols: ["temperature (°C)", "temperature (°F)", "energy use"],
    dataNote: "°F is a straight-line function of °C, so after centering and scaling the two columns are identical. Only w₁ + w₂ matters, not each one separately.",
    notice: "The ball slides to the trough and stops. Every point on the trough fits equally well, and where it stops depends on where it started.",
  },
  saddle: {
    kind: "saddle",
    name: "Hyperbolic paraboloid",
    stewart: String.raw`\frac{z}{c}=\frac{x^2}{a^2}-\frac{y^2}{b^2}`,
    params: { ratio: 1.5, rho: 0, mode: "minimax" },
    alpha: 0.1,
    start: [-2, 1.8],
    scenario: "Adversarial training (GAN-style)",
    oneLine: "Two players pull the same payoff in opposite directions.",
    dataTitle: "A min–max game",
    cols: ["generator weight", "discriminator weight", "payoff"],
    dataNote: "This is not a regression. One weight tries to lower the payoff z and the other tries to raise it, and the payoff has the shape of a hyperbolic paraboloid.",
    notice: "Plain descent would slide off the saddle. Each player moves its own way (one descends, one ascends), and together they settle on the saddle point.",
  },
};

const r2 = (x: number) => Math.round(x * 100) / 100;
const n2 = (x: number) => (Math.abs(x) < 0.005 ? 0 : x).toFixed(2);

// the standard form z = f(x, y) with the model's own semi-axes, used for the build animation
export function standardForm(kind: Kind, m: Model) {
  const a = r2(m.semiA);
  const b = Number.isFinite(m.semiB) ? r2(m.semiB) : 0;
  const f = (x: number, y: number) => {
    switch (kind) {
      case "circular":
        return (x * x + y * y) / (a * a);
      case "elliptic":
        return (x * x) / (a * a) + (y * y) / (b * b);
      case "cylinder":
        return (x * x) / (a * a);
      default:
        return (x * x) / (a * a) - (y * y) / (b * b);
    }
  };
  return { a, b, f };
}

// traces of the standard form: which plane (x = k, y = k, z = k) and what curve results
export function standardTrace(kind: Kind, mode: "x" | "y" | "z", k: number, a: number, b: number): { tex: string; name: string } {
  const A2 = a * a;
  const B2 = b * b;
  const kk = n2(k);
  const T = String.raw;
  if (kind === "circular") {
    if (mode === "z") {
      if (k < -0.005) return { tex: T`\text{no points}`, name: "empty (z cannot be negative)" };
      if (k < 0.005) return { tex: T`x^2+y^2=0`, name: "a single point, the vertex" };
      return { tex: T`x^2+y^2=${n2(A2 * k)}`, name: `a circle of radius ${n2(Math.sqrt(A2 * k))}` };
    }
    const v = mode === "x" ? "y" : "x";
    return { tex: T`z=\frac{${v}^2}{${n2(A2)}}+${n2((k * k) / A2)}`, name: "a parabola, opening up" };
  }
  if (kind === "elliptic") {
    if (mode === "z") {
      if (k < -0.005) return { tex: T`\text{no points}`, name: "empty (z cannot be negative)" };
      if (k < 0.005) return { tex: T`\frac{x^2}{a^2}+\frac{y^2}{b^2}=0`, name: "a single point, the vertex" };
      return {
        tex: T`\frac{x^2}{${n2(A2 * k)}}+\frac{y^2}{${n2(B2 * k)}}=1`,
        name: `an ellipse with semi-axes ${n2(a * Math.sqrt(k))} and ${n2(b * Math.sqrt(k))}`,
      };
    }
    return mode === "x"
      ? { tex: T`z=\frac{y^2}{${n2(B2)}}+${n2((k * k) / A2)}`, name: "a parabola, opening up" }
      : { tex: T`z=\frac{x^2}{${n2(A2)}}+${n2((k * k) / B2)}`, name: "a parabola, opening up" };
  }
  if (kind === "cylinder") {
    if (mode === "z") {
      if (k < -0.005) return { tex: T`\text{no points}`, name: "empty (z cannot be negative)" };
      return { tex: T`x=\pm ${n2(a * Math.sqrt(Math.max(k, 0)))}`, name: k < 0.005 ? "one line (the trough floor)" : "two parallel lines" };
    }
    return mode === "x"
      ? { tex: T`z=${n2((k * k) / A2)}`, name: "a horizontal line (z is constant along y)" }
      : { tex: T`z=\frac{x^2}{${n2(A2)}}`, name: "a parabola, the same for every k" };
  }
  // hyperbolic paraboloid
  if (mode === "z") {
    if (Math.abs(k) < 0.005) return { tex: T`y=\pm\frac{${n2(b)}}{${n2(a)}}\,x`, name: "two lines crossing at the origin" };
    return k > 0
      ? { tex: T`\frac{x^2}{${n2(A2 * k)}}-\frac{y^2}{${n2(B2 * k)}}=1`, name: "a hyperbola opening along x" }
      : { tex: T`\frac{y^2}{${n2(B2 * -k)}}-\frac{x^2}{${n2(A2 * -k)}}=1`, name: "a hyperbola opening along y" };
  }
  return mode === "x"
    ? { tex: T`z=${n2((k * k) / A2)}-\frac{y^2}{${n2(B2)}}`, name: "a parabola, opening down" }
    : { tex: T`z=\frac{x^2}{${n2(A2)}}-${n2((k * k) / B2)}`, name: "a parabola, opening up" };
}

// the second-degree coefficients of the mean squared error, straight from the data
export function lossCoefficients(m: Model) {
  const { x1, x2, y, n } = m.data;
  let A = 0;
  let B = 0;
  let C = 0;
  let D = 0;
  let E = 0;
  let F = 0;
  for (let i = 0; i < n; i++) {
    A += x1[i] * x1[i];
    B += 2 * x1[i] * x2[i];
    C += x2[i] * x2[i];
    D += -2 * x1[i] * y[i];
    E += -2 * x2[i] * y[i];
    F += y[i] * y[i];
  }
  return { A: A / n, B: B / n, C: C / n, D: D / n, E: E / n, F: F / n };
}

// What each surface means in machine learning: where it comes from in the data and what it does to training.
export interface MlMeaning {
  heading: string;
  condition: string; // TeX: the eigenvalue condition for this surface
  why: string[];
  where: string[];
  training: string[];
  remedy: string;
}

export const ML_MEANING: Record<Kind, MlMeaning> = {
  circular: {
    heading: "Predictors with equal spread and no correlation",
    condition: String.raw`\mu_1=\mu_2>0`,
    why: [
      "The Hessian of the mean squared error is twice the covariance matrix of the predictors. Curvature of the loss in a direction equals the variance of the data in that direction.",
      "If both predictors have the same variance and are uncorrelated, the covariance matrix is a multiple of the identity. Both eigenvalues are equal, so the loss curves the same amount in every direction and the level curves are circles.",
    ],
    where: [
      "Predictors that were standardized (mean 0, variance 1) and that do not move together.",
      "Whitened data, where a transformation such as PCA has removed the correlation.",
    ],
    training: [
      "The condition number κ = μ₁/μ₂ is 1, the best case. The negative gradient points straight at the minimum from every starting point.",
      "With the step size α = 1/μ, gradient descent lands on the minimum in one step.",
    ],
    remedy:
      "This is the surface that standardizing and whitening try to produce. Preprocessing the inputs is an attempt to turn every other bowl on this list into this one.",
  },
  elliptic: {
    heading: "Predictors with unequal spread or correlation",
    condition: String.raw`\mu_1>\mu_2>0,\qquad \kappa=\mu_1/\mu_2>1`,
    why: [
      "Same rule as before: the Hessian is twice the covariance matrix, so the bowl takes the shape of the data cloud. When one predictor varies more than the other, the loss is steep along it and shallow along the other, and the level curves stretch into ellipses.",
      "Correlated predictors do the same thing along a tilted axis. The cross term B in the loss is what rotates the ellipse away from the w₁, w₂ axes.",
    ],
    where: [
      "Predictors on different scales, such as square feet (thousands) next to bedrooms (a few).",
      "Predictors that move together, such as height and weight.",
      "Near the end of training in a neural network, where the loss looks like an elongated bowl with very different curvatures in different directions.",
    ],
    training: [
      "The step size is capped by the steepest direction (α < 2/μ₁ or the weights diverge), but progress along the shallowest direction is proportional to μ₂. A large κ means a small α and a very slow crawl, with the ball zig-zagging across the steep walls.",
      "This is called ill-conditioning, and κ measures how bad it is.",
    ],
    remedy:
      "Standardize the predictors to shrink κ. Optimizers like Adam and momentum also help by scaling or smoothing each direction's step.",
  },
  cylinder: {
    heading: "Predictors that are linearly dependent",
    condition: String.raw`\mu_1>0,\qquad \mu_2=0`,
    why: [
      "If one predictor is a linear function of the others, the columns of the data matrix are dependent and the covariance matrix has an eigenvalue of 0. The loss does not curve at all along that direction, so a flat trough runs through the bowl.",
      "Every point on the trough has exactly the same loss. The minimum is a whole line of weights, not a single point, so the problem has no unique solution.",
    ],
    where: [
      "The same quantity in two units, like temperature in °C and °F.",
      "The dummy-variable trap: one 0/1 column for every category plus an intercept.",
      "More predictors than rows of data.",
    ],
    training: [
      "Gradient descent still converges, but only to one point on the trough, and which point depends on where it started. Two runs from different starts give different weights and identical predictions.",
      "Individual weights stop meaning anything. Only the combination along the curved direction is determined by the data.",
    ],
    remedy:
      "Drop the redundant column, or add ridge regularization. Ridge adds λ‖w‖² to the loss, which adds 2λ to every eigenvalue and turns the cylinder back into a bowl with one minimum.",
  },
  saddle: {
    heading: "Two objectives pulling in opposite directions",
    condition: String.raw`\text{Hessian eigenvalues of opposite sign: one }>0,\ \text{one }<0`,
    why: [
      "A mean squared error can never produce this surface. Its Hessian is a covariance matrix, which has no negative eigenvalues, so it can only curve upward.",
      "A saddle needs a direction that curves down. That happens when one player is maximizing the same quantity the other is minimizing, as with the generator and discriminator in a GAN.",
    ],
    where: [
      "Generative adversarial networks and other adversarial training.",
      "Robust optimization, where one side picks the weights and the other picks the worst-case input.",
      "Saddle points also appear in the loss of ordinary neural networks. In high dimensions they are far more common than local minima.",
    ],
    training: [
      "Plain gradient descent slides off a saddle along the downward-curving direction. Gradient descent–ascent has each player move its own way, one descending and one ascending.",
      "It can circle the saddle point instead of settling when the two weights are coupled strongly, which is why GAN training is hard to stabilize.",
    ],
    remedy:
      "Use smaller step sizes, different step sizes for the two players, or methods built for games, such as extragradient and optimistic updates.",
  },
};
