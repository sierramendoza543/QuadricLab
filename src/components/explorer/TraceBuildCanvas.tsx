"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { Html, Line, OrbitControls } from "@react-three/drei";
import { useMemo, useRef } from "react";
import * as THREE from "three";

export type TraceMode = "x" | "y" | "z";

interface Props {
  f: (x: number, y: number) => number;
  D: number;
  mode: TraceMode;
  t: number; // 0..1, how far the cutting plane has swept
  className?: string;
}

const R = 3.2;
const K = 22; // number of traces drawn in total
const COLOR: Record<TraceMode, string> = { x: "#059669", y: "#db2777", z: "#d97706" };

function marching(f: (x: number, y: number) => number, level: number, D: number, n = 64): number[][] {
  const out: number[][] = [];
  const h = (2 * D) / n;
  const grid: number[][] = [];
  for (let i = 0; i <= n; i++) {
    grid.push([]);
    for (let j = 0; j <= n; j++) grid[i].push(f(-D + i * h, -D + j * h) - level);
  }
  const t = (a: number, b: number) => a / (a - b);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v00 = grid[i][j];
      const v10 = grid[i + 1][j];
      const v11 = grid[i + 1][j + 1];
      const v01 = grid[i][j + 1];
      const x0 = -D + i * h;
      const y0 = -D + j * h;
      const p: number[][] = [];
      if (v00 * v10 < 0) p.push([x0 + t(v00, v10) * h, y0]);
      if (v10 * v11 < 0) p.push([x0 + h, y0 + t(v10, v11) * h]);
      if (v01 * v11 < 0) p.push([x0 + t(v01, v11) * h, y0 + h]);
      if (v00 * v01 < 0) p.push([x0, y0 + t(v00, v01) * h]);
      if (p.length >= 2) out.push([p[0][0], p[0][1], p[1][0], p[1][1]]);
      if (p.length === 4) out.push([p[2][0], p[2][1], p[3][0], p[3][1]]);
    }
  }
  return out;
}

function Ghost({ f, D, s, shown }: { f: Props["f"]; D: number; s: number; shown: boolean }) {
  const N = 50;
  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(N * N * 3);
    const col = new Float32Array(N * N * 3);
    let hi = -Infinity;
    let lo = Infinity;
    for (let iy = 0; iy < N; iy++)
      for (let ix = 0; ix < N; ix++) {
        const x = -D + (2 * D * ix) / (N - 1);
        const y = -D + (2 * D * iy) / (N - 1);
        const z = f(x, y);
        hi = Math.max(hi, z);
        lo = Math.min(lo, z);
        const k = (iy * N + ix) * 3;
        pos[k] = (x / D) * R;
        pos[k + 1] = z * s;
        pos[k + 2] = -(y / D) * R;
      }
    const c = new THREE.Color();
    for (let i = 0; i < N * N; i++) {
      const tt = hi > lo ? (pos[i * 3 + 1] / s - lo) / (hi - lo) : 0;
      c.setHSL(0.6, 0.6, 0.86 - 0.5 * tt);
      col.set([c.r, c.g, c.b], i * 3);
    }
    const idx: number[] = [];
    for (let iy = 0; iy < N - 1; iy++)
      for (let ix = 0; ix < N - 1; ix++) {
        const a = iy * N + ix;
        idx.push(a, a + 1, a + N, a + 1, a + N + 1, a + N);
      }
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(col, 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return g;
  }, [f, D, s]);

  const mat = useRef<THREE.MeshStandardMaterial>(null);
  const level = useRef(0);
  useFrame(() => {
    if (!mat.current) return;
    level.current += ((shown ? 0.75 : 0) - level.current) * 0.08;
    mat.current.opacity = level.current;
    mat.current.visible = level.current > 0.01;
  });

  return (
    <mesh geometry={geom} renderOrder={1}>
      <meshStandardMaterial ref={mat} vertexColors transparent opacity={0} side={THREE.DoubleSide} roughness={0.85} depthWrite={false} />
    </mesh>
  );
}

function Scene({ f, D, mode, t }: Omit<Props, "className">) {
  const range = useMemo(() => {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i <= 40; i++)
      for (let j = 0; j <= 40; j++) {
        const z = f(-D + (2 * D * i) / 40, -D + (2 * D * j) / 40);
        lo = Math.min(lo, z);
        hi = Math.max(hi, z);
      }
    return { lo, hi };
  }, [f, D]);

  const s = 4.2 / Math.max(Math.abs(range.lo), Math.abs(range.hi), 1e-6);
  const sx = (x: number) => (x / D) * R;
  const sz = (y: number) => -(y / D) * R;
  const h = (z: number) => z * s;

  // x = k and y = k traces
  const lineTraces = useMemo(() => {
    if (mode === "z") return [] as THREE.Vector3[][];
    return Array.from({ length: K }, (_, i) => {
      const kv = -D + (2 * D * i) / (K - 1);
      return Array.from({ length: 61 }, (_, j) => {
        const u = -D + (2 * D * j) / 60;
        const [x, y] = mode === "x" ? [kv, u] : [u, kv];
        return new THREE.Vector3(sx(x), h(f(x, y)), sz(y));
      });
    });
  }, [f, D, mode, s]);

  const levels = useMemo(() => Array.from({ length: K }, (_, i) => range.lo + ((range.hi - range.lo) * i) / (K - 1)), [range]);
  const levelGeoms = useMemo(() => {
    if (mode !== "z") return [] as THREE.BufferGeometry[];
    return levels.map((L) => {
      const segs = marching(f, L, D);
      const arr = new Float32Array(segs.length * 6);
      segs.forEach(([x0, y0, x1, y1], i) => arr.set([sx(x0), h(L) + 0.02, sz(y0), sx(x1), h(L) + 0.02, sz(y1)], i * 6));
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
      return g;
    });
  }, [f, D, mode, levels, s]);

  const kcur = mode === "z" ? range.lo + (range.hi - range.lo) * t : -D + 2 * D * t;

  const currentLine = useMemo(() => {
    if (mode === "z") return null;
    return Array.from({ length: 61 }, (_, j) => {
      const u = -D + (2 * D * j) / 60;
      const [x, y] = mode === "x" ? [kcur, u] : [u, kcur];
      return new THREE.Vector3(sx(x), h(f(x, y)) + 0.03, sz(y));
    });
  }, [f, D, mode, kcur, s]);

  const currentLevel = useMemo(() => {
    if (mode !== "z") return null;
    const segs = marching(f, kcur, D);
    const arr = new Float32Array(segs.length * 6);
    segs.forEach(([x0, y0, x1, y1], i) => arr.set([sx(x0), h(kcur) + 0.04, sz(y0), sx(x1), h(kcur) + 0.04, sz(y1)], i * 6));
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return g;
  }, [f, D, mode, kcur, s]);

  const midY = h((range.lo + range.hi) / 2);
  const tall = Math.max(h(range.hi) - h(range.lo), 0.5);
  const color = COLOR[mode];
  const done = t >= 0.995;

  return (
    <>
      <color attach="background" args={["#ffffff"]} />
      <ambientLight intensity={1} />
      <directionalLight position={[6, 10, 4]} intensity={0.8} />
      <gridHelper args={[2 * R, 12, "#b8b8b8", "#dedede"]} position={[0, Math.min(h(range.lo), 0) - 0.05, 0]} />

      {mode !== "z" &&
        lineTraces.map((pts, i) => {
          const kv = -D + (2 * D * i) / (K - 1);
          return kv <= kcur - 1e-9 ? <Line key={i} points={pts} color={color} lineWidth={1.4} /> : null;
        })}
      {mode === "z" &&
        levelGeoms.map((g, i) =>
          levels[i] < kcur - 1e-9 ? (
            <lineSegments key={i} geometry={g}>
              <lineBasicMaterial color={color} />
            </lineSegments>
          ) : null
        )}

      {currentLine && <Line points={currentLine} color="#b00020" lineWidth={3.5} depthTest={false} />}
      {currentLevel && (
        <lineSegments geometry={currentLevel}>
          <lineBasicMaterial color="#b00020" />
        </lineSegments>
      )}

      {!done && mode === "x" && (
        <mesh position={[sx(kcur), midY, 0]} rotation={[0, Math.PI / 2, 0]}>
          <planeGeometry args={[2 * R, tall]} />
          <meshBasicMaterial color={color} transparent opacity={0.12} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}
      {!done && mode === "y" && (
        <mesh position={[0, midY, sz(kcur)]}>
          <planeGeometry args={[2 * R, tall]} />
          <meshBasicMaterial color={color} transparent opacity={0.12} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}
      {!done && mode === "z" && (
        <mesh position={[0, h(kcur), 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <planeGeometry args={[2 * R, 2 * R]} />
          <meshBasicMaterial color={color} transparent opacity={0.12} side={THREE.DoubleSide} depthWrite={false} />
        </mesh>
      )}

      <Ghost f={f} D={D} s={s} shown={done} />

      {(["x", "y", "z"] as const).map((ax) => (
        <Html
          key={ax}
          zIndexRange={[5, 0]}
          position={ax === "x" ? [R + 0.4, 0, 0] : ax === "y" ? [0, 0, -R - 0.4] : [-R, 4.9, R]}
          center
          style={{ pointerEvents: "none", color: "#222", fontSize: 15, fontStyle: "italic" }}
        >
          {ax}
        </Html>
      ))}
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} maxPolarAngle={Math.PI / 2 - 0.02} minDistance={6} maxDistance={26} target={[0, 1.6, 0]} />
    </>
  );
}

export default function TraceBuildCanvas({ className, ...rest }: Props) {
  return (
    <div className={className}>
      <Canvas camera={{ position: [7.5, 7, 9], fov: 40 }} dpr={[1, 2]} gl={{ antialias: true }}>
        <Scene {...rest} />
      </Canvas>
    </div>
  );
}
