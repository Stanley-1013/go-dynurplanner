"""Analytic AABB -> occupancy-grid rasterization (closed-form, no simulator).

This is the observation-side bridge of the GO-DynURPlanner design (Loop-6
blueprint §systems): the policy observes occupancy grids, but during
training those grids are generated ANALYTICALLY from the parameterized
space's AABBs — pure index arithmetic, no renderer, no simulator, keeping
URPlanner's simulator-free training property. At deployment the same data
type comes from depth/nvblox voxelization.

The same function applied to obstacles advanced by v*dt produces the FREE
ground truth for the occupancy-forecasting auxiliary task (the keystone):
    now  : rasterize(obstacle boxes at t)
    label: rasterize(obstacle boxes at t + dt)

Semantics: a voxel is occupied iff its cell intersects any (expanded) box —
conservative occupancy, exact for AABBs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import AABB


@dataclass(frozen=True)
class GridSpec:
    lo: np.ndarray  # workspace lower corner, (3,)
    hi: np.ndarray  # workspace upper corner, (3,)
    n: int = 32  # voxels per axis

    @property
    def cell(self) -> np.ndarray:
        return (self.hi - self.lo) / self.n

    def index_range(self, box: AABB) -> tuple[np.ndarray, np.ndarray] | None:
        """Half-open voxel index range [i_lo, i_hi) intersecting `box`.

        A voxel [lo + i*cell, lo + (i+1)*cell) intersects the box iff
        i > (box.lo - lo)/cell - 1  and  i < (box.hi - lo)/cell.
        Closed under floating point via floor/ceil; None if disjoint from
        the workspace.
        """
        cell = self.cell
        i_lo = np.floor((box.lo - self.lo) / cell).astype(int)
        i_hi = np.ceil((box.hi - self.lo) / cell).astype(int)
        i_lo = np.maximum(i_lo, 0)
        i_hi = np.minimum(i_hi, self.n)
        if np.any(i_lo >= i_hi):
            return None
        return i_lo, i_hi


def rasterize(boxes: list[AABB], spec: GridSpec) -> np.ndarray:
    """Occupancy grid, shape (n, n, n), dtype float32 in {0, 1}."""
    grid = np.zeros((spec.n, spec.n, spec.n), dtype=np.float32)
    for box in boxes:
        rng = spec.index_range(box)
        if rng is None:
            continue
        (x0, y0, z0), (x1, y1, z1) = rng
        grid[x0:x1, y0:y1, z0:z1] = 1.0
    return grid


def rasterize_future(
    obstacles, spec: GridSpec, dt: float, margin: float = 0.0
) -> np.ndarray:
    """Forecasting label: occupancy after advancing each obstacle by v*dt.

    `obstacles`: iterable with .center, .half, .vel (godynur.scenes.Obstacle).
    Constant-velocity advance matches the interval model of continuous.py;
    the forecaster learns to predict this from grid history alone.
    """
    boxes = [
        AABB(
            ob.center + ob.vel * dt - ob.half - margin,
            ob.center + ob.vel * dt + ob.half + margin,
        )
        for ob in obstacles
    ]
    return rasterize(boxes, spec)
