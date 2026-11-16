"use client";

import { Canvas, ThreeEvent, useFrame } from "@react-three/fiber";
import { Html, Line, OrbitControls } from "@react-three/drei";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import {
  Agent,
  AgentKind,
  DOMAIN,
  Model,
  Vec2,
  gradient,
  levelSegments,
  surfaceValue,
  traceCurve,
  updateDirection,
} from "@/lib/calculusMath";
import { Slices, VectorToggles } from "@/lib/types";

const SEG = 80;
const GRID_LINES = 28;

const COLORS = {
  x: "#059669",
  y: "#db2777",
  z: "#d97706",
  normal: "#0e7490",
  grad: "#15803d",
  momentum: "#c2410c",
};

const toScene = (w1: number, w2: number, z: number, s: number): THREE.Vector3 => new THREE.Vector3(w1, z * s, -w2);

interface SurfaceProps {
  model: Model;
  zScale: number;
  onPick: (w: Vec2) => void;
}

function Surface({ model, zScale, onPick }: SurfaceProps) {
  const { geometry, baseX, baseZ } = useMemo(() => {
    const g = new THREE.PlaneGeometry(2 * DOMAIN, 2 * DOMAIN, SEG, SEG);
    g.rotateX(-Math.PI / 2);
    const pos = g.getAttribute("position");
    const bx = new Float32Array(pos.count);
    const bz = new Float32Array(pos.count);
    for (let i = 0; i < pos.count; i++) {
      bx[i] = pos.getX(i);
      bz[i] = pos.getZ(i);
    }
    g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(pos.count * 3), 3));
    return { geometry: g, baseX: bx, baseZ: bz };
  }, []);

  const gridGeometry = useMemo(() => {
    const segs = 2 * (GRID_LINES + 1) * SEG;
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(segs * 2 * 3), 3));
    return g;
  }, []);

  const target = useMemo(() => {
    const t = new Float32Array(baseX.length);
    for (let i = 0; i < t.length; i++) t[i] = surfaceValue(model, baseX[i], -baseZ[i]) * zScale;
    return t;
  }, [model, zScale, baseX, baseZ]);

  const current = useRef<Float32Array | null>(null);
  const settled = useRef(false);

  useEffect(() => {
    settled.current = false;
  }, [target]);

  useFrame(() => {
    if (settled.current) return;
    if (!current.current) current.current = Float32Array.from(target);
    const cur = current.current;
    let drift = 0;
    for (let i = 0; i < cur.length; i++) {
      const d = target[i] - cur[i];
      drift = Math.max(drift, Math.abs(d));
      cur[i] += d * 0.2;
    }
    if (drift < 5e-4) {
      cur.set(target);
      settled.current = true;
    }

    const pos = geometry.getAttribute("position") as THREE.BufferAttribute;
    const col = geometry.getAttribute("color") as THREE.BufferAttribute;
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < cur.length; i++) {
      if (cur[i] < lo) lo = cur[i];
      if (cur[i] > hi) hi = cur[i];
    }
    const span = Math.max(hi - lo, 1e-6);
    const c = new THREE.Color();
    for (let i = 0; i < cur.length; i++) {
      pos.setY(i, cur[i]);
      const t = Math.sqrt((cur[i] - lo) / span);
      c.setHSL(0.6, 0.6, 0.86 - 0.5 * t);
      col.setXYZ(i, c.r, c.g, c.b);
    }
    pos.needsUpdate = true;
    col.needsUpdate = true;
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
    geometry.computeBoundingBox();

    const gp = gridGeometry.getAttribute("position") as THREE.BufferAttribute;
    const stride = SEG + 1;
    let n = 0;
    const put = (idx: number) => {
      gp.setXYZ(n++, baseX[idx], cur[idx] + 0.012, baseZ[idx]);
    };
    for (let r = 0; r <= GRID_LINES; r++) {
      const j = Math.round((r * SEG) / GRID_LINES);
      for (let i = 0; i < SEG; i++) {
        put(j * stride + i);
        put(j * stride + i + 1);
      }
      const col2 = j;
      for (let i = 0; i < SEG; i++) {
        put(i * stride + col2);
        put((i + 1) * stride + col2);
      }
    }
    gp.needsUpdate = true;
  });

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    onPick([e.point.x, -e.point.z]);
  };

  return (
    <group>
      <mesh geometry={geometry} onClick={handleClick}>
        <meshStandardMaterial
          vertexColors
          transparent
          opacity={0.85}
          side={THREE.DoubleSide}
          roughness={0.65}
          metalness={0.1}
        />
      </mesh>
      <lineSegments geometry={gridGeometry} raycast={() => null}>
        <lineBasicMaterial color="#1e3a8a" transparent opacity={0.3} />
      </lineSegments>
    </group>
  );
}

interface PlaneProps {
  axis: "x" | "y" | "z";
  k: number;
  model: Model;
  zScale: number;
}

function SlicePlane({ axis, k, model, zScale }: PlaneProps) {
  const lo = model.zMin * zScale;
  const hi = model.zMax * zScale;
  const height = Math.max(hi - lo, 0.5);
  const mid = (hi + lo) / 2;
  const color = COLORS[axis];
  const size = 2 * DOMAIN;

  let position: [number, number, number];
  let rotation: [number, number, number] = [0, 0, 0];
  let dims: [number, number];
  if (axis === "x") {
    position = [k, mid, 0];
    rotation = [0, Math.PI / 2, 0];
    dims = [size, height];
  } else if (axis === "y") {
    position = [0, mid, -k];
    dims = [size, height];
  } else {
    position = [0, k * zScale, 0];
    rotation = [-Math.PI / 2, 0, 0];
    dims = [size, size];
  }

  return (
    <mesh position={position} rotation={rotation} raycast={() => null}>
      <planeGeometry args={dims} />
      <meshBasicMaterial color={color} transparent opacity={0.1} side={THREE.DoubleSide} depthWrite={false} />
    </mesh>
  );
}

function LevelCurve({ model, level, zScale }: { model: Model; level: number; zScale: number }) {
  const geometry = useMemo(() => {
    const segs = levelSegments(model, level);
    const arr = new Float32Array(segs.length * 6);
    segs.forEach(([a, b], i) => {
      const y = level * zScale + 0.03;
      arr.set([a[0], y, -a[1], b[0], y, -b[1]], i * 6);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return g;
  }, [model, level, zScale]);

  return (
    <lineSegments geometry={geometry} raycast={() => null}>
      <lineBasicMaterial color={COLORS.z} />
    </lineSegments>
  );
}

function AxisCurve({ model, axis, k, zScale }: { model: Model; axis: "x" | "y"; k: number; zScale: number }) {
  const pts = useMemo(
    () => traceCurve(model, axis, k).map(([a, b, z]) => toScene(a, b, z, zScale).add(new THREE.Vector3(0, 0.03, 0))),
    [model, axis, k, zScale]
  );
  return <Line points={pts} color={COLORS[axis]} lineWidth={3} raycast={() => null} />;
}

interface ArrowProps {
  origin: THREE.Vector3;
  dir: THREE.Vector3;
  color: string;
  length?: number;
}

function Arrow({ origin, dir, color, length = 1.3 }: ArrowProps) {
  const quat = useMemo(() => {
    const d = dir.clone().normalize();
    return new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
  }, [dir]);
  const shaft = length * 0.76;
  const head = length * 0.24;
  return (
    <group position={origin} quaternion={quat} renderOrder={20}>
      <mesh position={[0, shaft / 2, 0]} raycast={() => null}>
        <cylinderGeometry args={[0.03, 0.03, shaft, 10]} />
        <meshBasicMaterial color={color} depthTest={false} transparent />
      </mesh>
      <mesh position={[0, shaft + head / 2, 0]} raycast={() => null}>
        <coneGeometry args={[0.09, head, 14]} />
        <meshBasicMaterial color={color} depthTest={false} transparent />
      </mesh>
    </group>
  );
}

interface BallProps {
  agent: Agent;
  model: Model;
  zScale: number;
  color: string;
  focused: boolean;
  vectors: VectorToggles;
}

function Ball({ agent, model, zScale, color, focused, vectors }: BallProps) {
  const [w1, w2] = agent.pos;
  const z = surfaceValue(model, w1, w2);
  const origin = toScene(w1, w2, z, zScale);

  const path = useMemo(
    () => agent.path.map(([a, b]) => toScene(a, b, surfaceValue(model, a, b), zScale).add(new THREE.Vector3(0, 0.05, 0))),
    [agent.path, model, zScale]
  );

  const g = gradient(model, w1, w2);
  const normalDir = new THREE.Vector3(-zScale * g[0], 1, zScale * g[1]);
  const d = updateDirection(model, g);
  const slope = g[0] * d[0] + g[1] * d[1];
  const gradDir = new THREE.Vector3(d[0], zScale * slope, -d[1]);

  let momentumDir: THREE.Vector3 | null = null;
  if (agent.step) {
    const prev: Vec2 = [w1 - agent.step[0], w2 - agent.step[1]];
    const back = toScene(prev[0], prev[1], surfaceValue(model, prev[0], prev[1]), zScale);
    const v = origin.clone().sub(back);
    if (v.length() > 1e-6) momentumDir = v;
  }

  return (
    <group>
      {path.length > 1 && <Line points={path} color={color} lineWidth={2.5} transparent opacity={0.9} raycast={() => null} />}
      <group position={origin}>
        <mesh raycast={() => null}>
          <sphereGeometry args={[0.15, 24, 24]} />
          <meshStandardMaterial color={color} emissiveIntensity={0} />
        </mesh>
        <mesh raycast={() => null}>
          <sphereGeometry args={[0.32, 20, 20]} />
          <meshBasicMaterial color={color} transparent opacity={0.12} depthWrite={false} />
        </mesh>
      </group>
      {focused && vectors.normal && <Arrow origin={origin} dir={normalDir} color={COLORS.normal} />}
      {focused && vectors.grad && gradDir.length() > 1e-6 && <Arrow origin={origin} dir={gradDir} color={COLORS.grad} />}
      {focused && vectors.momentum && momentumDir && <Arrow origin={origin} dir={momentumDir} color={COLORS.momentum} />}
    </group>
  );
}

const BALL_COLOR: Record<AgentKind, string> = { gd: "#111111", momentum: COLORS.momentum };

interface SceneProps extends Omit<QuadricCanvasProps, "className"> {}

function Scene({ model, zScale, agents, focusId, vectors, slices, onPick }: SceneProps) {
  const floor = Math.min(model.zMin * zScale, 0) - 0.05;
  const top = model.zMax * zScale;
  const mid = Math.max(top, 1) / 3;

  return (
    <>
      <color attach="background" args={["#ffffff"]} />
      <fog attach="fog" args={["#ffffff", 26, 60]} />
      <ambientLight intensity={0.8} />
      <directionalLight position={[6, 10, 4]} intensity={1.1} />
      <directionalLight position={[-6, 4, -6]} intensity={0.4} color="#ffffff" />

      <gridHelper args={[2 * DOMAIN, 14, "#b8b8b8", "#dedede"]} position={[0, floor, 0]} />
      <Line
        points={[
          [-DOMAIN, floor, DOMAIN],
          [-DOMAIN, Math.max(top, 1), DOMAIN],
        ]}
        color="#999999"
        lineWidth={1.5}
      />
      <Html zIndexRange={[5, 0]} position={[DOMAIN + 0.35, floor, 0]} center style={{ pointerEvents: "none" }}>
        <span className="axis-tag">w₁</span>
      </Html>
      <Html zIndexRange={[5, 0]} position={[0, floor, -DOMAIN - 0.35]} center style={{ pointerEvents: "none" }}>
        <span className="axis-tag">w₂</span>
      </Html>
      <Html zIndexRange={[5, 0]} position={[-DOMAIN, Math.max(top, 1) + 0.25, DOMAIN]} center style={{ pointerEvents: "none" }}>
        <span className="axis-tag">z</span>
      </Html>

      <Surface model={model} zScale={zScale} onPick={onPick} />

      {slices.x.on && (
        <>
          <SlicePlane axis="x" k={slices.x.k} model={model} zScale={zScale} />
          <AxisCurve model={model} axis="x" k={slices.x.k} zScale={zScale} />
        </>
      )}
      {slices.y.on && (
        <>
          <SlicePlane axis="y" k={slices.y.k} model={model} zScale={zScale} />
          <AxisCurve model={model} axis="y" k={slices.y.k} zScale={zScale} />
        </>
      )}
      {slices.z.on && (
        <>
          <SlicePlane axis="z" k={slices.z.k} model={model} zScale={zScale} />
          <LevelCurve model={model} level={slices.z.k} zScale={zScale} />
        </>
      )}

      {agents.map((a) => (
        <Ball
          key={a.id}
          agent={a}
          model={model}
          zScale={zScale}
          color={BALL_COLOR[a.id]}
          focused={a.id === focusId}
          vectors={vectors}
        />
      ))}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        maxPolarAngle={Math.PI / 2 - 0.02}
        minDistance={5}
        maxDistance={26}
        target={[0, mid, 0]}
      />
    </>
  );
}

export interface QuadricCanvasProps {
  model: Model;
  zScale: number;
  agents: Agent[];
  focusId: AgentKind;
  vectors: VectorToggles;
  slices: Slices;
  onPick: (w: Vec2) => void;
  className?: string;
}

export default function QuadricCanvas(props: QuadricCanvasProps) {
  const { className, ...rest } = props;
  return (
    <div className={className}>
      <Canvas camera={{ position: [11, 8.5, 12.5], fov: 42 }} dpr={[1, 2]} gl={{ antialias: true }}>
        <Scene {...rest} />
      </Canvas>
    </div>
  );
}
