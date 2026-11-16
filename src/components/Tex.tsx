"use client";

import katex from "katex";
import { useMemo } from "react";

interface TexProps {
  tex: string;
  block?: boolean;
  left?: boolean;
  className?: string;
}

export default function Tex({ tex, block = false, left = false, className }: TexProps) {
  const html = useMemo(
    () => katex.renderToString(tex, { displayMode: block, fleqn: left, throwOnError: false, strict: "ignore" }),
    [tex, block, left]
  );
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />;
}
