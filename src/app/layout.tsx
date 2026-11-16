import type { Metadata } from "next";
import { Carlito } from "next/font/google";
import "katex/dist/katex.min.css";
import "./globals.css";

const carlito = Carlito({
  subsets: ["latin"],
  weight: ["400", "700"],
  variable: "--font-carlito",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Quadric Lab",
  description: "Chapter 12 quadric surfaces and gradient descent on real regression loss landscapes.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={carlito.variable}>
      <body>{children}</body>
    </html>
  );
}
