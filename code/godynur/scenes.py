"""Dynamic scene generation for the Panda workspace.

Obstacles are AABBs translating at constant speed, reflecting off the
workspace bounds (keeps them in play; piecewise-constant velocity so the
constant-velocity-per-interval model of `continuous.py` is exact within a
control step, with reflections snapped to step boundaries).

Obstacle geometry mixes 'thin' plates and compact boxes: both occur in
human-robot collaboration (trays, tools, forearms are thin; boxes/bins are
compact). Thinness and speed are the two physical drivers of tunneling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .continuous import MovingBox
from .geometry import AABB

# Panda-reachable workspace shell (conservative box around the task region).
WS_LO = np.array([-0.75, -0.75, 0.05])
WS_HI = np.array([0.75, 0.75, 1.05])


@dataclass
class Obstacle:
    center: np.ndarray  # (3,)
    half: np.ndarray  # (3,)
    vel: np.ndarray  # (3,)

    def aabb(self, margin: float = 0.0) -> AABB:
        return AABB(self.center - self.half - margin, self.center + self.half + margin)

    def moving_box(self, margin: float = 0.0) -> MovingBox:
        return MovingBox(self.aabb(margin), self.vel.copy())


@dataclass
class DynamicScene:
    obstacles: list[Obstacle] = field(default_factory=list)

    def step(self, dt: float) -> None:
        """Advance obstacle centers by dt; reflect velocity at workspace bounds."""
        for ob in self.obstacles:
            ob.center = ob.center + ob.vel * dt
            for k in range(3):
                if ob.center[k] - ob.half[k] < WS_LO[k]:
                    ob.center[k] = WS_LO[k] + ob.half[k]
                    ob.vel[k] = abs(ob.vel[k])
                elif ob.center[k] + ob.half[k] > WS_HI[k]:
                    ob.center[k] = WS_HI[k] - ob.half[k]
                    ob.vel[k] = -abs(ob.vel[k])

    def moving_boxes(self, margin: float = 0.0) -> list[MovingBox]:
        return [ob.moving_box(margin) for ob in self.obstacles]

    def static_aabbs(self, margin: float = 0.0) -> list[AABB]:
        return [ob.aabb(margin) for ob in self.obstacles]


def sample_scene(
    rng: np.random.Generator,
    n_obstacles: int,
    speed: float,
    thin_fraction: float = 0.5,
    forbidden: list[AABB] | None = None,
) -> DynamicScene:
    """Sample a scene. `speed` is the obstacle |v| (m/s); direction random.

    `forbidden`: regions obstacles must not initially intersect (e.g. the
    arm's start configuration corridor), resampled up to 100 tries.
    """
    obstacles: list[Obstacle] = []
    for _ in range(n_obstacles):
        for _attempt in range(100):
            if rng.random() < thin_fraction:
                # Thin plate: one dimension 1-3 cm, others 10-25 cm.
                half = rng.uniform(0.10, 0.25, 3)
                half[rng.integers(0, 3)] = rng.uniform(0.005, 0.015)
            else:
                half = rng.uniform(0.04, 0.12, 3)
            center = rng.uniform(WS_LO + half, WS_HI - half)
            box = AABB(center - half, center + half)
            if forbidden and any(_aabb_overlap(box, f) for f in forbidden):
                continue
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            obstacles.append(Obstacle(center, half, speed * direction))
            break
    return DynamicScene(obstacles)


def _aabb_overlap(a: AABB, b: AABB) -> bool:
    return bool(np.all(a.lo <= b.hi) and np.all(b.lo <= a.hi))
