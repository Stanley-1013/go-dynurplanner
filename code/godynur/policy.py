"""Discretely-safe greedy goal-reaching policy.

This is NOT a contribution — it is the honest instantiation of the checking
protocol that standard DRL training loops use: propose candidate joint
steps, keep those whose RESULTING configuration is collision-free under a
per-timestep (discrete) check, execute the best. It exists to measure how
often that protocol tunnels through moving obstacles (M1 / Figure-1).

Candidate generation mirrors APE2's shape (a goal-directed base action plus
Gaussian exploration noise at several scales) without any learning.
"""

from __future__ import annotations

import numpy as np

from .geometry import AABB, segment_box_overlap
from .panda import DQ_MAX, PandaKinematics, Q_MAX, Q_MIN


def config_collides(kin: PandaKinematics, q: np.ndarray, boxes: list[AABB]) -> bool:
    """Discrete (instantaneous) collision check of configuration q."""
    for seg in kin.segments(q):
        for box in boxes:
            l, _, _ = segment_box_overlap(seg[0], seg[1], box)
            if l > 0.0:
                return True
    return False


class GreedyDiscretePolicy:
    """Greedy candidate-selection policy with discrete endpoint checking."""

    def __init__(
        self,
        kin: PandaKinematics,
        action_clip: float = 0.05,
        n_candidates: int = 24,
        noise_scales: tuple[float, ...] = (0.0, 0.3, 1.0),
        seed: int = 0,
    ):
        self.kin = kin
        self.action_clip = action_clip
        self.n_candidates = n_candidates
        self.noise_scales = noise_scales
        self.rng = np.random.default_rng(seed)

    def _ik_free_direction(self, q: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Finite-difference descent direction on |flange - goal| (IK-free,
        matching URPlanner's joint-increment action philosophy)."""
        base = np.linalg.norm(self.kin.flange(q) - goal)
        grad = np.zeros(7)
        h = 1e-4
        for i in range(7):
            qp = q.copy()
            qp[i] += h
            grad[i] = (np.linalg.norm(self.kin.flange(qp) - goal) - base) / h
        n = np.linalg.norm(grad)
        return -grad / n if n > 1e-9 else np.zeros(7)

    def act(
        self, q: np.ndarray, goal: np.ndarray, boxes: list[AABB]
    ) -> np.ndarray | None:
        """Return the greedy discretely-safe joint step, or None if every
        candidate's endpoint configuration is in collision (robot holds)."""
        direction = self._ik_free_direction(q, goal)
        base = direction * self.action_clip
        candidates = [base]
        per_scale = max(1, (self.n_candidates - 1) // len(self.noise_scales))
        for scale in self.noise_scales:
            for _ in range(per_scale):
                noise = self.rng.normal(0.0, scale * self.action_clip, 7)
                candidates.append(base + noise)
        best, best_cost = None, np.inf
        for dq in candidates:
            dq = np.clip(dq, -self.action_clip, self.action_clip)
            q_new = np.clip(q + dq, Q_MIN, Q_MAX)
            if config_collides(self.kin, q_new, boxes):
                continue  # discrete endpoint check: reject
            cost = np.linalg.norm(self.kin.flange(q_new) - goal)
            if cost < best_cost:
                best, best_cost = q_new - q, cost
        return best


__all__ = ["GreedyDiscretePolicy", "config_collides", "DQ_MAX"]
