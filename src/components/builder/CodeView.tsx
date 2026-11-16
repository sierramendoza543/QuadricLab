const KEYWORDS = new Set(["import", "from", "for", "in", "range", "as"]);

function highlight(line: string) {
  const parts: React.ReactNode[] = [];
  const re = /(#.*$)|("[^"]*")|(\b\d+(?:\.\d+)?\b)|(\b[A-Za-z_][A-Za-z_0-9]*\b)/g;
  let last = 0;
  let m: RegExpExecArray | null;
  let k = 0;
  while ((m = re.exec(line))) {
    if (m.index > last) parts.push(line.slice(last, m.index));
    const [tok, comment, str, num, word] = m;
    let cls = "";
    if (comment) cls = "text-gray-500";
    else if (str) cls = "text-green-800";
    else if (num) cls = "text-orange-700";
    else if (word && KEYWORDS.has(word)) cls = "text-purple-800 font-bold";
    parts.push(
      cls ? (
        <span key={k++} className={cls}>
          {tok}
        </span>
      ) : (
        tok
      )
    );
    last = m.index + tok.length;
  }
  if (last < line.length) parts.push(line.slice(last));
  return parts;
}

export default function CodeView({ code }: { code: string }) {
  return (
    <pre className="overflow-auto border border-gray-300 bg-gray-50 p-4 text-[14px] leading-relaxed text-gray-900">
      <code>
        {code.split("\n").map((l, i) => (
          <div key={i} className="min-h-[1.5em] whitespace-pre">
            {highlight(l)}
          </div>
        ))}
      </code>
    </pre>
  );
}
