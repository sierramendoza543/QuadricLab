import Tex from "../Tex";

// each string is one line of working; use & to line up the equals signs
export default function Paper({ lines, small = false }: { lines: string[]; small?: boolean }) {
  const tex = String.raw`\begin{aligned}` + lines.join(String.raw` \\[0.6em] `) + String.raw`\end{aligned}`;
  return (
    <div className={`paper my-3 border-l-2 border-gray-300 pl-4 ${small ? "text-[14px]" : "text-[18px]"}`}>
      <Tex block left tex={tex} />
    </div>
  );
}
