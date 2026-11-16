"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { Html, Line, OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { Grid, QuadricFit, quadValue } from "@/lib/ml/landscape";
import { PathPoint } from "@/lib/ml/lesson";

const R = 3.5;

interface Props {
  grid: Grid;
  fit: QuadricFit | null;
  showFit: boolean;
  path: PathPoint[];
  replay: number | null;
  showStart: boolean;
  className?: string;
}

function percentile(a: Float64Array, p: number) {
  const s = Array.from(a).sort((x, y) => x - y);
  return s[Math.min(s.length - 1, Math.floor(p * s.length))];
}

function lineGrid(n: number, stride: number, h: (ix: number, iy: number) => number) {
  const out: number[] = [];
  const px = (i: number) => -R + (2 * R * i) / (n - 1);
  for (let iy = 0; iy < n; iy += stride)
    for (let ix = 0; ix < n - 1; ix++) out.push(px(ix), h(ix, iy), -px(iy), px(ix + 1), h(ix + 1, iy), -px(iy));
  for (let ix = 0; ix < n; ix += stride)
    for (let iy = 0; iy < n - 1; iy++) out.push(px(ix), h(ix, iy), -px(iy), px(ix), h(ix, iy + 1), -px(iy + 1));
  return new Float32Array(out);
}

interface Mapper {
  cap: number;
  h: (z: number) => number;
  sx: (a: number) => number;
  sz: (b: number) => number;
}

function Surface({ grid, m }: { grid: Grid; m: Mapper }) {
  const n = grid.n;
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(n * n * 3);
    for (let iy = 0; iy < n; iy++)
      for (let ix = 0; ix < n; ix++) {
        const k = (iy * n + ix) * 3;
        pos[k] = -R + (2 * R * ix) / (n - 1);
        pos[k + 2] = R - (2 * R * iy) / (n - 1);
      }
    const idx: number[] = [];
    for (let iy = 0; iy < n - 1; iy++)
      for (let ix = 0; ix < n - 1; ix++) {
        const a = iy * n + ix;
        idx.push(a, a + 1, a + n, a + 1, a + n + 1, a + n);
      }
    g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(n * n * 3), 3));
    g.setIndex(idx);
    return g;
  }, [n]);

  const lines = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(lineGrid(n, 2, () => 0), 3));
    return g;
  }, [n]);

  const target = useMemo(() => Float32Array.from(grid.z, (z) => m.h(z)), [grid, m]);
  const current = useRef<Float32Array | null>(null);
  const settled = useRef(false);
  useEffect(() => {
    settled.current = false;
  }, [target]);

  useFrame(() => {
    if (settled.current) return;
    if (!current.current || current.current.length !== target.length) current.current = new Float32Array(target.length);
    const cur = current.current;
    let drift = 0;
    for (let i = 0; i < cur.length; i++) {
      const d = target[i] - cur[i];
      drift = Math.max(drift, Math.abs(d));
      cur[i] += d * 0.2;
    }
    if (drift < 4e-4) {
      cur.set(target);
      settled.current = true;
    }
    const pos = geometry.getAttribute("position") as THREE.BufferAttribute;
    const col = geometry.getAttribute("color") as THREE.BufferAttribute;
    const top = Math.max(...cur, 1e-6);
    const c = new THREE.Color();
    for (let i = 0; i < cur.length; i++) {
      pos.setY(i, cur[i]);
      const t = Math.min(1, cur[i] / top);
      c.setHSL(0.6, 0.6, 0.86 - 0.5 * t);
      col.setXYZ(i, c.r, c.g, c.b);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    geometry.computeVertexNormals();
    (lines.getAttribute("position").array as Float32Array).set(lineGrid(n, 2, (ix, iy) => cur[iy * n + ix] + 0.012));
    lines.getAttribute("position").needsUpdate = true;
  });

  return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial vertexColors transparent opacity={0.82} side={THREE.DoubleSide} roughness={0.8} metalness={0} />
      </mesh>
      <lineSegments geometry={lines}>
        <lineBasicMaterial color="#1e3a8a" transparent opacity={0.3} />
      </lineSegments>
    </group>
  );
}

function FitOverlay({ grid, fit, m }: { grid: Grid; fit: QuadricFit; m: Mapper }) {
  const geom = useMemo(() => {
    const n = grid.n;
    const lo = grid.zMin - 0.05 * (m.cap - grid.zMin);
    const arr = lineGrid(n, 2, (ix, iy) => {
      const a = -grid.range + (2 * grid.range * ix) / (n - 1);
      const b = -grid.range + (2 * grid.range * iy) / (n - 1);
      return m.h(Math.min(m.cap, Math.max(lo, quadValue(fit, a, b)))) + 0.02;
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return g;
  }, [grid, fit, m]);
  return (
    <lineSegments geometry={geom}>
      <lineBasicMaterial color="#d9480f" />
    </lineSegments>
  );
}

function Dot({ position, color, size = 0.13 }: { position: THREE.Vector3; color: string; size?: number }) {
  return (
    <mesh position={position} renderOrder={10}>
      <sphereGeometry args={[size, 20, 20]} />
      <meshBasicMaterial color={color} depthTest={false} />
    </mesh>
  );
}

const tag: React.CSSProperties = { pointerEvents: "none", color: "#222", fontSize: 15, fontStyle: "italic", whiteSpace: "nowrap" };

function Scene({ grid, fit, showFit, path, replay, showStart }: Omit<Props, "className">) {
  const m = useMemo<Mapper>(() => {
    const cap = Math.max(percentile(grid.z, 0.97), grid.zMin + 1e-9);
    const scale = 4.5 / (cap - grid.zMin);
    return {
      cap,
      h: (z) => (Math.min(z, cap) - grid.zMin) * scale,
      sx: (a) => (a / grid.range) * R,
      sz: (b) => -(b / grid.range) * R,
    };
  }, [grid]);

  const pts = useMemo(() => path.map((p) => new THREE.Vector3(m.sx(p.alpha), m.h(p.z) + 0.07, m.sz(p.beta))), [path, m]);
  const center = useMemo(() => {
    const mid = (grid.n - 1) / 2;
    return new THREE.Vector3(0, m.h(grid.z[mid * grid.n + mid]) + 0.1, 0);
  }, [grid, m]);

  return (
    <>
      <color attach="background" args={["#ffffff"]} />
      <ambientLight intensity={1.0} />
      <directionalLight position={[6, 10, 4]} intensity={0.9} />
      <directionalLight position={[-6, 5, -6]} intensity={0.35} />
      <gridHelper args={[2 * R, 14, "#b8b8b8", "#dedede"]} position={[0, -0.05, 0]} />

      <Surface grid={grid} m={m} />
      {showFit && fit && <FitOverlay grid={grid} fit={fit} m={m} />}
      {pts.length > 1 && <Line points={pts} color="#111111" lineWidth={2.5} depthTest={false} renderOrder={9} />}
      <Dot position={center} color="#15803d" size={0.16} />
      {showStart && pts[0] && <Dot position={pts[0]} color="#ea580c" size={0.16} />}
      {replay !== null && pts[replay] && <Dot position={pts[replay]} color="#000000" size={0.11} />}

      <Html zIndexRange={[5, 0]} position={[R + 0.4, 0, 0]} center style={tag}>
        w₁
      </Html>
      <Html zIndexRange={[5, 0]} position={[0, 0, -R - 0.4]} center style={tag}>
        w₂
      </Html>
      <Html zIndexRange={[5, 0]} position={[-R, 4.9, R]} center style={tag}>
        L
      </Html>
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} maxPolarAngle={Math.PI / 2 - 0.02} minDistance={6} maxDistance={28} target={[0, 1.6, 0]} />
    </>
  );
}

export default function LandscapeCanvas({ className, ...rest }: Props) {
  return (
    <div className={className}>
      <Canvas camera={{ position: [7, 8, 8.5], fov: 40 }} dpr={[1, 2]} gl={{ antialias: true }}>
        <Scene {...rest} />
      </Canvas>
    </div>
  );
}
