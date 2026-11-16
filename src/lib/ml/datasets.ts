import { gaussian, rng } from "./random";

// one column of the original file; a text column becomes several 0/1 columns
export interface Source {
  name: string;
  kind: "numeric" | "categorical";
  columns: string[]; // encoded column names in Table.columns
  levels?: string[];
  baseline?: string;
  dummies?: { column: string; level: string }[];
  mean?: number;
  min?: number;
  max?: number;
}

export interface Table {
  name: string;
  columns: string[];
  rows: number[][];
  rowMeaning?: string;
  notes?: Record<string, string>;
  sources?: Source[];
  preview?: { columns: string[]; rows: string[][] };
}

export function sourcesOf(t: Table): Source[] {
  if (t.sources) return t.sources;
  return t.columns.map((c, j) => {
    const col = t.rows.map((r) => r[j]);
    return { name: c, kind: "numeric" as const, columns: [c], mean: col.reduce((s, v) => s + v, 0) / col.length, min: Math.min(...col), max: Math.max(...col) };
  });
}

// columns that can be predicted: numeric columns, and the 0/1 column of a text column with two values
export function targetColumns(t: Table): number[] {
  const ok = new Set<string>();
  for (const s of sourcesOf(t)) {
    if (s.kind === "numeric") s.columns.forEach((c) => ok.add(c));
    else if (s.dummies?.length === 1) ok.add(s.dummies[0].column);
  }
  return t.columns.map((c, j) => (ok.has(c) ? j : -1)).filter((j) => j >= 0);
}

export interface ClassInfo {
  lo: number;
  hi: number;
  label: string; // what the model gives a probability for
  n1: number;
  n0: number;
}

// a column with exactly two distinct values is a classification target
export function classInfo(t: Table, target: number): ClassInfo | null {
  const vals = new Set<number>();
  for (const r of t.rows) {
    vals.add(r[target]);
    if (vals.size > 2) return null;
  }
  if (vals.size !== 2) return null;
  const [lo, hi] = [...vals].sort((a, b) => a - b);
  const col = t.columns[target];
  const src = sourcesOf(t).find((s) => s.columns.includes(col));
  const dummy = src?.dummies?.find((d) => d.column === col);
  const n1 = t.rows.filter((r) => r[target] === hi).length;
  return { lo, hi, label: dummy ? `${src!.name} = ${dummy.level}` : `${col} = ${hi}`, n1, n0: t.rows.length - n1 };
}

export interface BundledDataset {
  id: string;
  task: "regression" | "classification";
  title: string;
  blurb: string;
  lesson: string;
  size: string;
  build: () => Table;
}

const uniform = (r: () => number, lo: number, hi: number) => lo + (hi - lo) * r();
const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export const BUNDLED: BundledDataset[] = [
  {
    id: "housing",
    task: "regression",
    title: "House prices",
    blurb: "Sale price of 300 houses with floor area, bedrooms, age and distance to the city center.",
    lesson: "Columns on different scales",
    size: "300 rows · 5 columns",
    build: () => {
      const r = rng(11);
      const rows: number[][] = [];
      for (let i = 0; i < 300; i++) {
        const sqft = uniform(r, 600, 3500);
        const bed = clamp(Math.round(1 + sqft / 900 + gaussian(r) * 0.6), 1, 6);
        const age = uniform(r, 0, 60);
        const dist = uniform(r, 1, 30);
        const price = 50 + 0.12 * sqft + 8 * bed - 1.1 * age - 3.5 * dist + gaussian(r) * 20;
        rows.push([Math.round(sqft), bed, Math.round(age), Math.round(dist * 10) / 10, Math.round(price * 10) / 10]);
      }
      return {
        name: "House prices",
        columns: ["sqft", "bedrooms", "age_years", "km_to_center", "price_k"],
        rows,
        rowMeaning: "one house that got sold",
        notes: {
          sqft: "how big the house is, in square feet",
          bedrooms: "how many bedrooms it has",
          age_years: "how old the house is",
          km_to_center: "how far it is from the city center, in km",
          price_k: "what it sold for, in thousands of dollars",
        },
      };
    },
  },
  {
    id: "temperature",
    task: "regression",
    title: "Energy and temperature",
    blurb: "Daily energy use of a building against outdoor temperature, recorded in both Celsius and Fahrenheit.",
    lesson: "Two columns say the same thing",
    size: "200 rows · 3 columns",
    build: () => {
      const r = rng(23);
      const rows: number[][] = [];
      for (let i = 0; i < 200; i++) {
        const c = uniform(r, -5, 35);
        const e = 120 - 2.2 * c + 0.02 * c * c + gaussian(r) * 4;
        rows.push([c, 1.8 * c + 32, e]);
      }
      return {
        name: "Energy and temperature",
        columns: ["celsius", "fahrenheit", "energy_kwh"],
        rows,
        rowMeaning: "one day",
        notes: {
          celsius: "the outside temperature in Celsius",
          fahrenheit: "the exact same temperature, but in Fahrenheit",
          energy_kwh: "how much power the building used that day (kWh)",
        },
      };
    },
  },
  {
    id: "sine",
    task: "regression",
    title: "Sine wave",
    blurb: "A noisy sine wave: one predictor and a clearly nonlinear response.",
    lesson: "Needs a curve",
    size: "160 rows · 2 columns",
    build: () => {
      const r = rng(5);
      const rows: number[][] = [];
      for (let i = 0; i < 160; i++) {
        const x = uniform(r, -3, 3);
        rows.push([x, Math.sin(1.5 * x) + gaussian(r) * 0.1]);
      }
      return {
        name: "Sine wave",
        columns: ["x", "y"],
        rows,
        rowMeaning: "one measurement",
        notes: { x: "the number that goes in", y: "the number that comes out (a wave plus a little noise)" },
      };
    },
  },
  {
    id: "plants",
    task: "regression",
    title: "Plant growth",
    blurb: "Plant growth against sunlight and water. The two predictors interact.",
    lesson: "Clues team up",
    size: "250 rows · 3 columns",
    build: () => {
      const r = rng(41);
      const rows: number[][] = [];
      for (let i = 0; i < 250; i++) {
        const sun = uniform(r, 0, 10);
        const water = uniform(r, 0, 10);
        rows.push([sun, water, 3 + 0.12 * sun * water + gaussian(r) * 0.4]);
      }
      return {
        name: "Plant growth",
        columns: ["sunlight_hrs", "water_l", "growth_cm"],
        rows,
        rowMeaning: "one plant",
        notes: {
          sunlight_hrs: "hours of sun the plant gets per day",
          water_l: "liters of water it gets per day",
          growth_cm: "how much it grew, in cm",
        },
      };
    },
  },
  {
    id: "exam",
    task: "classification",
    title: "Exam pass or fail",
    blurb: "Whether 300 students passed, from hours studied and hours slept. A classification problem.",
    lesson: "Predict yes or no",
    size: "300 rows · 3 columns",
    build: () => {
      const r = rng(17);
      const rows: number[][] = [];
      for (let i = 0; i < 300; i++) {
        const study = uniform(r, 0, 10);
        const sleep = uniform(r, 4, 10);
        const logit = 1.1 * (study - 5) + 0.6 * (sleep - 7) + gaussian(r) * 0.8;
        rows.push([Math.round(study * 10) / 10, Math.round(sleep * 10) / 10, logit > 0 ? 1 : 0]);
      }
      return {
        name: "Exam pass or fail",
        columns: ["hours_studied", "hours_slept", "passed"],
        rows,
        rowMeaning: "one student",
        notes: {
          hours_studied: "hours spent studying for the exam",
          hours_slept: "hours of sleep the night before",
          passed: "1 if the student passed, 0 if not",
        },
      };
    },
  },
  {
    id: "loans",
    task: "classification",
    title: "Loan approval",
    blurb: "Whether 400 loan applications were approved, from income and savings. The two predictors move together.",
    lesson: "Correlated predictors",
    size: "400 rows · 3 columns",
    build: () => {
      const r = rng(29);
      const rows: number[][] = [];
      for (let i = 0; i < 400; i++) {
        const income = clamp(60 + gaussian(r) * 15, 20, 120);
        const savings = clamp(0.8 * income + gaussian(r) * 9, 0, 150);
        const logit = 0.09 * (income - 60) + 0.07 * (savings - 48) + gaussian(r) * 0.9;
        rows.push([Math.round(income * 10) / 10, Math.round(savings * 10) / 10, logit > 0 ? 1 : 0]);
      }
      return {
        name: "Loan approval",
        columns: ["income_k", "savings_k", "approved"],
        rows,
        rowMeaning: "one loan application",
        notes: {
          income_k: "yearly income, in thousands of dollars",
          savings_k: "savings, in thousands of dollars (people with higher income save more)",
          approved: "1 if the loan was approved, 0 if not",
        },
      };
    },
  },
  {
    id: "zone",
    task: "classification",
    title: "Target zone",
    blurb: "Points inside or outside a circle. No straight line separates them, so the model needs squared terms.",
    lesson: "Needs a curved boundary",
    size: "300 rows · 3 columns",
    build: () => {
      const r = rng(53);
      const rows: number[][] = [];
      for (let i = 0; i < 300; i++) {
        const x = uniform(r, -3, 3);
        const y = uniform(r, -3, 3);
        const inside = x * x + y * y + gaussian(r) * 0.8 < 4 ? 1 : 0;
        rows.push([Math.round(x * 100) / 100, Math.round(y * 100) / 100, inside]);
      }
      return {
        name: "Target zone",
        columns: ["x", "y", "inside"],
        rows,
        rowMeaning: "one point",
        notes: { x: "horizontal position", y: "vertical position", inside: "1 if the point is inside the target circle, 0 if outside" },
      };
    },
  },
];

export const DATA_SOURCES = [
  { name: "Kaggle Datasets", url: "https://www.kaggle.com/datasets", note: "Free login needed. Download a CSV, then drop it in here." },
  { name: "UCI ML Repository", url: "https://archive.ics.uci.edu/datasets", note: "Classic, clean datasets from research." },
  { name: "OpenML", url: "https://www.openml.org/search?type=data", note: "Thousands of tables you can download as CSV." },
  { name: "Hugging Face Datasets", url: "https://huggingface.co/datasets", note: "Filter by \"tabular\", then grab a CSV." },
  { name: "Google Dataset Search", url: "https://datasetsearch.research.google.com", note: "Like Google, but for datasets." },
  { name: "data.gov", url: "https://data.gov", note: "Open data from the US government." },
];
