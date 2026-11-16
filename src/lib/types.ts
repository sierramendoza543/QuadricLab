export interface SliceState {
  on: boolean;
  k: number;
}

export interface Slices {
  x: SliceState;
  y: SliceState;
  z: SliceState;
}

export interface VectorToggles {
  normal: boolean;
  grad: boolean;
  momentum: boolean;
}

export interface Hyper {
  alpha: number;
  beta: number;
  speed: number;
}
