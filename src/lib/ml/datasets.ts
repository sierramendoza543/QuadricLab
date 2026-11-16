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

// columns that can be predicted: the numeric ones from the original file
export function targetColumns(t: Table): number[] {
  const numeric = new Set(sourcesOf(t).filter((s) => s.kind === "numeric").flatMap((s) => s.columns));
  return t.columns.map((c, j) => (numeric.has(c) ? j : -1)).filter((j) => j >= 0);
}

export interface BundledDataset {
  id: string;
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
];

export const DATA_SOURCES = [
  { name: "Kaggle Datasets", url: "https://www.kaggle.com/datasets", note: "Free login needed. Download a CSV, then drop it in here." },
  { name: "UCI ML Repository", url: "https://archive.ics.uci.edu/datasets", note: "Classic, clean datasets from research." },
  { name: "OpenML", url: "https://www.openml.org/search?type=data", note: "Thousands of tables you can download as CSV." },
  { name: "Hugging Face Datasets", url: "https://huggingface.co/datasets", note: "Filter by \"tabular\", then grab a CSV." },
  { name: "Google Dataset Search", url: "https://datasetsearch.research.google.com", note: "Like Google, but for datasets." },
  { name: "data.gov", url: "https://data.gov", note: "Open data from the US government." },
];
