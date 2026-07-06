"""APE2-Shield: URPlanner's candidate exploration + hybrid evaluation,
augmented with the continuous-time interval shield.

URPlanner (Eq.14-18): generate M x N + 1 action candidates around the
deterministic policy output (M noise scales, N samples each), score each by
the hybrid value V_HPE = eta * Q_LTR + (1 - eta) * R_IR where eta anneals
0 -> 1 with training progress (trust the critic late, the analytic
immediate reward early), execute the argmax.

Our addition (H4): before scoring, REJECT every candidate whose interval
first-contact time tau*(a) < dt on lemma-inflated obstacle boxes. Because
the candidate pool already exists, the shield costs one root computation
per candidate and needs no QP/reachability machinery. With the inflation of
`lemma_inflation` (chord error + obstacle acceleration + velocity noise),
an accepted candidate is provably interval-collision-free under bounded
model error.

Fallback ladder when NO candidate passes:
  1. scale down the best-scoring candidate (alpha in {0.5, 0.25, 0.125, 0})
     and accept the largest alpha that certifies — slowing down along the
     intended direction;
  2. if even alpha = 0 (holding still) is unsafe (an obstacle is running
     into the arm), no certified action exists in this family: execute the
     candidate that maximizes tau* (delay contact as long as possible) and
     record the event. This residual case is measurable and reported, not
     hidden.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .continuous import lemma_inflation
from .env import DynArmEnv


class APE2Shield:
    def __init__(
        self,
        env: DynArmEnv,
        base_fn: Callable[[np.ndarray], np.ndarray],
        q_fn: Callable[[np.ndarray, np.ndarray], float] | None,
        M: int = 2,
        N: int = 3,
        sigma_scales: tuple[float, ...] = (0.5, 1.5),  # x action_clip
        shield: bool = True,
        a_max: float = 2.0,  # obstacle acceleration bound (m/s^2)
        eps_v: float = 0.1,  # obstacle velocity estimation error (m/s)
        anneal_steps: int = 50_000,
        seed: int = 0,
    ):
        assert len(sigma_scales) == M, "one noise scale per M"
        self.env, self.base_fn, self.q_fn = env, base_fn, q_fn
        self.M, self.N = M, N
        self.sigma_scales = sigma_scales
        self.shield = shield
        self.a_max, self.eps_v = a_max, eps_v
        self.anneal_steps = anneal_steps
        self.rng = np.random.default_rng(seed)
        self.stats = {"steps": 0, "rejected": 0, "scaled": 0, "no_safe": 0}

    # ---- internals ---------------------------------------------------------

    def _inflation(self, dq: np.ndarray) -> float:
        eps_lin = self.env.kin.chord_error_bound(self.env.q, dq)
        return lemma_inflation(eps_lin, self.a_max, self.env.dt, self.eps_v)

    def _certified(self, a: np.ndarray) -> bool:
        return self.env.peek_tau_star(a, self._inflation(a)) is None

    def _candidates(self, s: np.ndarray) -> list[np.ndarray]:
        clip = self.env.action_bound[1]
        base = np.clip(self.base_fn(s), -clip, clip)
        cands = [base]
        for scale in self.sigma_scales:
            for _ in range(self.N):
                noise = self.rng.normal(0.0, scale * clip, self.env.action_dim)
                cands.append(np.clip(base + noise, -clip, clip))
        return cands

    def _scores(self, s: np.ndarray, cands: list[np.ndarray], step: int):
        eta = min(1.0, step / self.anneal_steps) if self.q_fn else 0.0
        out = []
        for a in cands:
            r_ir = self.env.peek_reward(a)
            q = self.q_fn(s, a) if self.q_fn else 0.0
            out.append(eta * q + (1.0 - eta) * r_ir)
        return out

    # ---- main --------------------------------------------------------------

    def act(self, s: np.ndarray, step: int) -> np.ndarray:
        self.stats["steps"] += 1
        cands = self._candidates(s)
        scores = self._scores(s, cands, step)
        order = np.argsort(scores)[::-1]

        if not self.shield:
            return cands[order[0]]

        for i in order:
            if self._certified(cands[i]):
                if i != order[0]:
                    self.stats["rejected"] += 1
                return cands[i]

        # No candidate certifies: scale the best one down along its direction.
        best = cands[order[0]]
        for alpha in (0.5, 0.25, 0.125, 0.0):
            a = alpha * best
            if self._certified(a):
                self.stats["scaled"] += 1
                return a

        # Even holding still is unsafe (obstacle running into the arm):
        # delay contact as long as possible, and count the event honestly.
        self.stats["no_safe"] += 1
        taus = [
            (self.env.peek_tau_star(a, self._inflation(a)) or np.inf, k)
        for k, a in enumerate(cands)]
        return cands[max(taus)[1]]
