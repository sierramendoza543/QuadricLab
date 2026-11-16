import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#060913",
        trace: { z: "#f59e0b", x: "#10b981", y: "#ec4899" },
        vec: { normal: "#22d3ee", grad: "#a3e635", mom: "#fb923c" },
      },
      fontFamily: {
        sans: ["Calibri", "var(--font-carlito)", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
        mono: ["Calibri", "var(--font-carlito)", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
