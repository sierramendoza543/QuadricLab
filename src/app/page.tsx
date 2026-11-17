import Link from "next/link";
import { Box, FlaskConical, Github, Shapes } from "lucide-react";

const REPO = "https://github.com/sierramendoza543/QuadricLab";
const button = "flex items-center justify-center gap-3 border border-gray-400 bg-white px-6 py-3 text-lg hover:border-blue-700 hover:bg-blue-50";

export default function Landing() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-white px-6 py-16 text-center text-[#1a1a1a]">
      <h1 className="text-7xl font-bold tracking-tight sm:text-8xl">Quadric Lab</h1>
      <p className="mt-5 max-w-xl text-xl leading-relaxed text-gray-700">
        An interactive look at the quadric surfaces of Chapter 12, and how each one shows up as the loss surface of a machine learning model.
      </p>
      <p className="mt-6 text-lg">Made by Sierra Mendoza</p>
      <p className="text-gray-600">Made for Stewart / Larson calculus textbooks</p>

      <div className="mt-10 grid w-full max-w-md gap-3">
        <Link href="/lab" className={button}>
          <Box size={20} />
          3D visualization
        </Link>
        <Link href="/builder" className={button}>
          <FlaskConical size={20} />
          Build your own model
        </Link>
        <Link href="/explorer" className={button}>
          <Shapes size={20} />
          Quadric explorer
        </Link>
        <a href={REPO} target="_blank" rel="noopener noreferrer" className={button}>
          <Github size={20} />
          GitHub
        </a>
      </div>
    </main>
  );
}
