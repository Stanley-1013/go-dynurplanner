"""DynArmEnv — dynamic-obstacle Panda environment, Morvan-interface style.

The lab's code lineage descends from Morvan Zhou's train-robot-arm-from-
scratch (Bilibili BV1nW411a7Qg, 2017): a hand-written analytic environment
with `step(action) -> (s, r, done)`, `reset() -> s`, class-level
state_dim/action_dim/action_bound, joint-increment actions, no simulator.
URPlanner's parameterized task space is that pattern scaled to 3D/7-DoF.
This env keeps the exact same interface so it drops into a Morvan/URPlanner
training stack unchanged, while upgrading the internals:

  - obstacles MOVE (godynur.scenes), speed-tiered for curriculum;
  - collision/termination accounting is CONTINUOUS-TIME EXACT (tau* via
    godynur.continuous) — to our knowledge the first manipulator RL env
    whose safety bookkeeping cannot tunnel;
  - reward is switchable: 'uoar' (URPlanner Eq.12, discrete) vs 'ct'
    (D-UOAR-CT: pose + current overlap + swept integral + continuous TTC)
    — the M3 ablation axis;
  - occupancy-grid observation (k-frame history) is available via
    `grid_history()` for the M2 encoder; the flat vector state stays
    Morvan-compatible for existing DDPG/TD3/APE2 code.

Every quantity is analytic (FK, overlap, swept integral, rasterization):
training needs no simulator, preserving the URPlanner property.
"""

from __future__ import annotations

import numpy as np

from . import kinodynamics, panda
from .continuous import MovingSegment, first_contact_time, swept_overlap_integral
from .geometry import AABB, segment_box_closest, segment_box_overlap, uoar
from .panda import DDQ_MAX, DQ_MAX, PandaKinematics, Q_MAX, Q_MIN
from .safety_qp import solve_safety_qp
from .scenes import WS_HI, WS_LO, DynamicScene, sample_scene
from .voxelizer import GridSpec, rasterize, rasterize_sdf


class DynArmEnv:
    dt = 0.05  # control interval (s)
    action_bound = [-0.05, 0.05]  # rad per step per joint (lemma budget)
    action_dim = 7
    goal_tol = 0.05  # m
    goal_dwell = 1  # reach = success (URPlanner phi_G semantics)
    max_steps = 100

    def __init__(
        self,
        speed: float = 0.5,
        n_obstacles: int = 3,
        reward_mode: str = "ct",  # 'uoar' | 'ct'
        obstacles_in_state: bool = True,
        a_r: float = 0.06,
        a_o: float = 0.05,
        k_frames: int = 3,
        grid_n: int = 32,
        zeta_c: float = 1.0,   # current-overlap weight (URPlanner zeta)
        zeta_s: float = 1.0,   # swept-integral weight (CT mode)
        zeta_t: float = 0.5,   # continuous-TTC weight (CT mode, bounded form)
        task: str = "random",  # 'random' | 'tabletop' (URPlanner-style)
        grid_mode: str = "binary",  # 'binary' | 'sdf' (advisor: SDF learns better)
        closest_point_in_state: bool = False,  # advisor: d, direction, point
        seed: int = 0,
        action_mode: str = "delta_q",  # 'delta_q' | 'velocity'
        dv_scale: np.ndarray | None = None,
        safety_n_steps: int = 2,
        brake_check_n_steps: int = 20,
        brake_derate: float = 0.85,
        lambda_intervention: float = 0.1,
        margin_bisection_iters: int = 4,
    ):
        assert reward_mode in ("uoar", "ct")
        assert task in ("random", "tabletop")
        assert grid_mode in ("binary", "sdf")
        assert action_mode in ("delta_q", "velocity")
        self.task = task
        self.grid_mode = grid_mode
        self.closest_point_in_state = closest_point_in_state
        self.action_mode = action_mode
        self.kin = PandaKinematics()
        self.speed = speed
        self.n_obstacles = n_obstacles
        self.reward_mode = reward_mode
        self.obstacles_in_state = obstacles_in_state
        self.a_r, self.a_o = a_r, a_o
        self.zeta_c, self.zeta_s, self.zeta_t = zeta_c, zeta_s, zeta_t
        self.k_frames = k_frames
        self.spec = GridSpec(lo=WS_LO, hi=WS_HI, n=grid_n)
        self.rng = np.random.default_rng(seed)
        if action_mode == "velocity":
            # Placeholder action scale for Phase 5 tuning: roughly the largest
            # one-step velocity change reachable from rest under the jerk cap.
            self.dv_scale = np.asarray(
                panda.DQ_MAX * self.dt if dv_scale is None else dv_scale,
                dtype=float,
            )
            assert self.dv_scale.shape == (self.action_dim,)
            self.safety_n_steps = safety_n_steps
            self.brake_check_n_steps = brake_check_n_steps
            self.brake_derate = brake_derate
            self.lambda_intervention = lambda_intervention
            # Default lowered from velocity_margin()'s own default of 10:
            # measured ~17ms/step at 10 iters (7 joints x 2 directions x 10
            # calls to braking_witness_jerk), unexpectedly QP-scale
            # overhead. 4 iters gives 1/16 of the search range's
            # resolution -- coarse but this only feeds an observation
            # signal, not a safety gate, so precision is not safety-
            # critical here. Revisit if the margin signal proves to help
            # learning and the extra cost is judged worth paying for
            # finer resolution.
            self.margin_bisection_iters = margin_bisection_iters
            self.action_bound = [-1.0, 1.0]
        # Fixed obstacle slots (zero-padded): curriculum can lower the LIVE
        # obstacle count without changing the network's input width.
        self.n_obstacles_max = n_obstacles
        # 7 q + 3 flange + 3 goal + 3 delta + 1 dwell flag
        self.state_dim = (
            17
            + (6 * n_obstacles if obstacles_in_state else 0)
            + (7 if closest_point_in_state else 0)
            # 7 v + 7 a + 7 margin+ + 7 margin- + 2 shield flags
            + (30 if action_mode == "velocity" else 0)
        )
        # Episode stats maintained for Figure-1-style instrumentation.
        self.last_tau_star: float | None = None
        self.last_discrete_missed: bool = False
        self.stats: dict[str, int] = {}

    # ---- Morvan interface -------------------------------------------------

    # Franka 'ready' pose (standard home configuration).
    Q_HOME = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.79])
    # Table-top goal region (URPlanner-style: goals above the table in
    # front of the robot).
    GOAL_LO = np.array([0.25, -0.35, 0.15])
    GOAL_HI = np.array([0.70, 0.35, 0.70])

    def set_difficulty(self, n_obstacles: int, speed: float) -> None:
        """Curriculum hook: takes effect at the next reset(). The live
        obstacle count may not exceed n_obstacles_max (state slots are
        fixed at construction; absent obstacles are zero-padded)."""
        assert n_obstacles <= self.n_obstacles_max
        self.n_obstacles = n_obstacles
        self.speed = speed

    def _sample_start_goal(self):
        if self.task == "tabletop":
            q = self.Q_HOME + self.rng.normal(0.0, 0.05, 7)
            q = np.clip(q, Q_MIN, Q_MAX)
            for _ in range(500):
                goal = self.kin.flange(Q_MIN + self.rng.random(7) * (Q_MAX - Q_MIN))
                if np.all(goal >= self.GOAL_LO) and np.all(goal <= self.GOAL_HI):
                    return q, goal
            # Fallback: nearest reachable sample to the region center.
            return q, self.kin.flange(Q_MIN + self.rng.random(7) * (Q_MAX - Q_MIN))
        for _ in range(200):
            q = Q_MIN + self.rng.random(7) * (Q_MAX - Q_MIN)
            if self.kin.flange(q)[2] > 0.15:
                break
        return q, self.kin.flange(Q_MIN + self.rng.random(7) * (Q_MAX - Q_MIN))

    def reset(self) -> np.ndarray:
        self.q, self.goal = self._sample_start_goal()
        if self.action_mode == "velocity":
            self.v = np.zeros(self.action_dim)
            self.a = np.zeros(self.action_dim)
            self._last_terminal_membership = False
            self._last_intervention_norm = 0.0
            self._last_velocity_margin_plus = np.zeros(self.action_dim)
            self._last_velocity_margin_minus = np.zeros(self.action_dim)
        margin = self.a_r + self.a_o
        for _ in range(100):
            self.scene: DynamicScene = sample_scene(
                self.rng, self.n_obstacles, self.speed
            )
            if not self._config_collides(self.q, margin + 0.05):
                break
        self.on_goal = 0
        self.t = 0
        self._grids = [self._rasterize_now()] * self.k_frames
        return self._state()

    def step(self, action: np.ndarray):
        if self.action_mode == "velocity":
            return self._step_velocity(action)

        dq = np.clip(np.asarray(action, float), *self.action_bound)
        q_new = np.clip(self.q + dq, Q_MIN, Q_MAX)
        dq = q_new - self.q

        # Exact continuous-time collision accounting over [t, t+dt].
        tau_star = self._first_contact(dq)
        self.last_tau_star = tau_star
        collided = tau_star is not None
        if collided:
            # Instrumentation: would the standard endpoint check have seen it?
            self.last_discrete_missed = not self._endpoint_overlaps(q_new)

        r = self._reward(dq)
        if collided:
            r -= 5.0  # terminal collision penalty (URPlanner-style)

        self.q = q_new
        self.scene.step(self.dt)
        self.t += 1
        self._grids = self._grids[1:] + [self._rasterize_now()]

        err = np.linalg.norm(self.kin.flange(self.q) - self.goal)
        if err < self.goal_tol:
            self.on_goal += 1
        else:
            self.on_goal = 0
        done = collided or self.on_goal >= self.goal_dwell or self.t >= self.max_steps
        return self._state(), float(r), bool(done)

    def _step_velocity(self, action: np.ndarray):
        (
            self._last_velocity_margin_plus,
            self._last_velocity_margin_minus,
        ) = self._velocity_margins(self.q, self.v, self.a)
        q_old = self.q.copy()
        v_old = self.v.copy()
        a_old = self.a.copy()
        u_t = np.clip(np.asarray(action, float), -1.0, 1.0)
        v_nom = self.v + self.dv_scale * u_t
        qp = solve_safety_qp(
            self.q,
            self.v,
            self.a,
            v_nom,
            h=self.dt,
            n_steps=self.safety_n_steps,
            q_min=Q_MIN,
            q_max=Q_MAX,
            v_min=-DQ_MAX,
            v_max=DQ_MAX,
            a_min=-panda.DDQ_MAX,
            a_max=panda.DDQ_MAX,
            j_min=-panda.DDDQ_MAX,
            j_max=panda.DDDQ_MAX,
            require_terminal_stop=False,
        )

        accepted = False
        if qp.certified and qp.jerk_sequence is not None:
            candidate_jerks = qp.jerk_sequence[:, 0]
            candidate = self._kinodynamic_update(candidate_jerks)
            if self._braking_witness_jerks(*candidate) is not None:
                q_new, v_new, a_new = candidate
                q_trajectory_end = q_new.copy()
                executed_jerks = candidate_jerks
                accepted = True

        if not accepted:
            # The environment is fully deterministic, so recomputing from the
            # actual state reproduces exactly what a cached multi-step witness
            # would prescribe there. It is also more robust: an accepted step
            # between fallbacks cannot leave us replaying a stale brake plan.
            brake_jerks = self._braking_witness_jerks(self.q, self.v, self.a)
            if brake_jerks is not None:
                q_new, v_new, a_new = self._kinodynamic_update(brake_jerks)
                q_trajectory_end = q_new.copy()
                executed_jerks = brake_jerks
                self.stats["shield_fallback"] = (
                    self.stats.get("shield_fallback", 0) + 1
                )
            else:
                # Least-committal last resort: preserve current acceleration
                # for one integration period, then defensively clip q and v.
                executed_jerks = np.zeros(self.action_dim)
                q_new, v_new, a_new = self._kinodynamic_update(executed_jerks)
                q_trajectory_end = q_new.copy()
                q_new = np.clip(q_new, Q_MIN, Q_MAX)
                v_new = np.clip(v_new, -DQ_MAX, DQ_MAX)
                self.stats["shield_emergency"] = (
                    self.stats.get("shield_emergency", 0) + 1
                )

        u_executed = (v_new - v_old) / self.dv_scale
        self._last_executed_action = np.clip(u_executed, -1.0, 1.0)
        eps_cubic = kinodynamics.cubic_linearization_bound(
            q_old, v_old, a_old, executed_jerks, self.dt, self.kin
        )
        self.q = np.clip(q_new, Q_MIN, Q_MAX)
        self.v = np.clip(v_new, -DQ_MAX, DQ_MAX)
        self.a = a_new
        self._last_terminal_membership = accepted
        self._last_intervention_norm = float(np.linalg.norm(self.v - v_nom))
        dq = self.q - q_old

        # Existing collision and reward conventions operate on (q, dq), so
        # temporarily restore q_t while evaluating the executed transition.
        self.q = q_old
        tau_star = self._first_contact(
            q_trajectory_end - q_old, inflation=eps_cubic
        )
        self.last_tau_star = tau_star
        collided = tau_star is not None
        if collided:
            self.last_discrete_missed = not self._endpoint_overlaps(q_old + dq)

        r = self._reward(dq)
        r -= self.lambda_intervention * self._last_intervention_norm**2
        if collided:
            r -= 5.0

        self.q = q_old + dq
        self.scene.step(self.dt)
        self.t += 1
        self._grids = self._grids[1:] + [self._rasterize_now()]

        err = np.linalg.norm(self.kin.flange(self.q) - self.goal)
        if err < self.goal_tol:
            self.on_goal += 1
        else:
            self.on_goal = 0
        done = collided or self.on_goal >= self.goal_dwell or self.t >= self.max_steps
        return self._state(), float(r), bool(done)

    def sample_action(self) -> np.ndarray:
        return self.rng.uniform(*self.action_bound, self.action_dim)

    def scene_params(self, pad_to: int | None = None) -> np.ndarray:
        """Obstacle snapshot as (pad_to, 9): [center(3), half(3), vel(3)] per
        slot; absent slots filled with a distant dummy (center=50) that
        rasterizes to nothing. The analytic-space replay trick: storing 9
        floats per obstacle instead of voxel grids makes grid-RL replay
        buffers ~4000x smaller; grids are re-rasterized lazily on sampling."""
        n = pad_to if pad_to is not None else self.n_obstacles_max
        out = np.full((n, 9), 50.0, dtype=np.float32)
        out[:, 3:6] = 0.01
        out[:, 6:9] = 0.0
        for i, ob in enumerate(self.scene.obstacles[:n]):
            out[i, 0:3] = ob.center
            out[i, 3:6] = ob.half
            out[i, 6:9] = ob.vel
        return out

    # ---- candidate peeking (no state mutation) ----------------------------
    # The analytic parameterized space makes candidate evaluation free: both
    # the immediate reward and the interval-collision verdict of a candidate
    # action are pure functions of (current state, action). This is what
    # APE2's candidate pool exploits (URPlanner) and what the shield needs.

    def peek_reward(self, action: np.ndarray) -> float:
        """Immediate reward the candidate would receive (excl. terminal
        collision penalty and goal bonus) — APE2's R_IR, analytically."""
        dq = np.clip(np.asarray(action, float), *self.action_bound)
        dq = np.clip(self.q + dq, Q_MIN, Q_MAX) - self.q
        return float(self._reward(dq))

    def peek_tau_star(
        self, action: np.ndarray, inflation: float = 0.0
    ) -> float | None:
        """First-contact time of the candidate over [0, dt], with obstacle
        boxes inflated by `inflation` (lemma margin) on top of a_r.
        None = certified interval-collision-free (under the inflation)."""
        dq = np.clip(np.asarray(action, float), *self.action_bound)
        dq = np.clip(self.q + dq, Q_MIN, Q_MAX) - self.q
        taus = []
        for seg in self._moving_segments(dq):
            seg_lo = np.minimum(seg.a0, seg.b0)
            seg_hi = np.maximum(seg.a0, seg.b0)
            for ob in self.scene.obstacles:
                if not self._pair_can_contact(
                    seg_lo, seg_hi, ob, self.a_r + inflation
                ):
                    continue
                t = first_contact_time(
                    seg, ob.moving_box(self.a_r + inflation), self.dt
                )
                if t is not None:
                    taus.append(t)
        return min(taus) if taus else None

    # ---- observations ------------------------------------------------------

    def grid_history(self) -> np.ndarray:
        """(k_frames, n, n, n) occupancy stack for the M2 encoder."""
        return np.stack(self._grids)

    def _state(self) -> np.ndarray:
        flange = self.kin.flange(self.q)
        q_norm = (self.q - Q_MIN) / (Q_MAX - Q_MIN) * 2.0 - 1.0
        parts = [
            q_norm,
            flange,
            self.goal,
            self.goal - flange,
            [1.0 if self.on_goal else 0.0],
        ]
        if self.obstacles_in_state:
            for ob in self.scene.obstacles:
                parts.append(ob.center - flange)
                parts.append(ob.vel)
            for _ in range(self.n_obstacles_max - len(self.scene.obstacles)):
                parts.append(np.zeros(6))
        if self.closest_point_in_state:
            d, p_arm, p_box = self.closest_obstacle_point()
            direction = (p_box - p_arm) / max(d, 1e-6) if d > 1e-6 else np.zeros(3)
            parts.append([d])
            parts.append(direction)
            parts.append(p_box - flange)
        if self.action_mode == "velocity":
            parts.append(self.v / DQ_MAX)
            parts.append(self.a / DDQ_MAX)
            parts.append(self._last_velocity_margin_plus / DQ_MAX)
            parts.append(self._last_velocity_margin_minus / DQ_MAX)
            parts.append([1.0 if self._last_terminal_membership else 0.0])
            parts.append([self._last_intervention_norm])
        return np.concatenate(parts).astype(np.float32)

    # ---- internals -----------------------------------------------------------

    def _kinodynamic_update(
        self, jerks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_states = np.array(
            [
                kinodynamics.discrete_update(
                    np.array([self.q[i], self.v[i], self.a[i]]),
                    jerks[i],
                    self.dt,
                )
                for i in range(self.action_dim)
            ]
        )
        return next_states[:, 0], next_states[:, 1], next_states[:, 2]

    def _braking_witness_jerks(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray
    ) -> np.ndarray | None:
        jerks = []
        for i in range(self.action_dim):
            jerk = kinodynamics.braking_witness_jerk(
                q[i],
                v[i],
                a[i],
                self.dt,
                self.brake_check_n_steps,
                Q_MIN[i],
                Q_MAX[i],
                -DQ_MAX[i],
                DQ_MAX[i],
                -self.brake_derate * panda.DDQ_MAX[i],
                self.brake_derate * panda.DDQ_MAX[i],
                -self.brake_derate * panda.DDDQ_MAX[i],
                self.brake_derate * panda.DDDQ_MAX[i],
            )
            if jerk is None:
                return None
            jerks.append(jerk)
        return np.asarray(jerks)

    def _velocity_margins(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        margins = []
        for direction in (1.0, -1.0):
            directional = []
            for i in range(self.action_dim):
                upper_bound = (
                    DQ_MAX[i] - v[i]
                    if direction > 0.0
                    else v[i] + DQ_MAX[i]
                )
                directional.append(
                    kinodynamics.velocity_margin(
                        q[i],
                        v[i],
                        a[i],
                        direction,
                        self.dt,
                        self.brake_check_n_steps,
                        Q_MIN[i],
                        Q_MAX[i],
                        -DQ_MAX[i],
                        DQ_MAX[i],
                        -self.brake_derate * panda.DDQ_MAX[i],
                        self.brake_derate * panda.DDQ_MAX[i],
                        -self.brake_derate * panda.DDDQ_MAX[i],
                        self.brake_derate * panda.DDDQ_MAX[i],
                        max(0.0, float(upper_bound)),
                        bisection_iters=self.margin_bisection_iters,
                    )
                )
            margins.append(np.asarray(directional))
        return margins[0], margins[1]

    def _rasterize_now(self) -> np.ndarray:
        boxes = self.scene.static_aabbs(margin=0.0)
        if self.grid_mode == "sdf":
            return rasterize_sdf(boxes, self.spec)
        return rasterize(boxes, self.spec)

    def closest_obstacle_point(self):
        """Global closest approach between any arm segment and any obstacle:
        (distance, point_on_arm, point_on_obstacle). Analytic (convex
        ternary search per pair). (0.5, zeros, zeros) when no obstacles."""
        best = (0.5, np.zeros(3), np.zeros(3))
        for seg in self.kin.segments(self.q):
            for box in self._boxes(0.0):
                d, p_arm, p_box = segment_box_closest(seg[0], seg[1], box)
                if d < best[0]:
                    best = (d, p_arm, p_box)
        return best

    def _boxes(self, margin: float) -> list[AABB]:
        return self.scene.static_aabbs(margin=margin)

    def _config_collides(self, q, margin) -> bool:
        for seg in self.kin.segments(q):
            for box in self._boxes(margin):
                l, _, _ = segment_box_overlap(seg[0], seg[1], box)
                if l > 0.0:
                    return True
        return False

    def _moving_segments(self, dq) -> list[MovingSegment]:
        s0 = self.kin.segments(self.q)
        s1 = self.kin.segments(self.q + dq)
        return [
            MovingSegment(
                a0=a[0], ua=(b[0] - a[0]) / self.dt,
                b0=a[1], ub=(b[1] - a[1]) / self.dt,
            )
            for a, b in zip(s0, s1)
        ]


    def _pair_can_contact(self, seg_lo, seg_hi, ob, margin: float) -> bool:
        """Conservative broad-phase: False only if the pair provably cannot
        touch within dt. Segment AABB vs obstacle box gap (L2 lower bound
        via per-axis gaps) compared against max closure = (|v_obs| +
        v_seg_max) * dt, where v_seg_max bounds link-point speed by total
        arm length * max joint speed... we use the action-clip kinematic
        bound: |dq|_1 * R_max / dt with R_max ~ 1.2 m, plus margin slack."""
        import numpy as _np
        lo = ob.center - ob.half - margin
        hi = ob.center + ob.half + margin
        gap = _np.maximum(lo - seg_hi, 0.0) + _np.maximum(seg_lo - hi, 0.0)
        dist_lb = float(_np.linalg.norm(gap))
        v_obs = float(_np.linalg.norm(ob.vel))
        v_seg = 1.2 * 7 * self.action_bound[1] / self.dt  # conservative
        return dist_lb <= (v_obs + v_seg) * self.dt + 1e-9

    def _first_contact(self, dq, inflation: float = 0.0) -> float | None:
        taus = []
        for seg in self._moving_segments(dq):
            seg_lo = np.minimum(seg.a0, seg.b0)
            seg_hi = np.maximum(seg.a0, seg.b0)
            for ob in self.scene.obstacles:
                margin = self.a_r + inflation
                if not self._pair_can_contact(seg_lo, seg_hi, ob, margin):
                    continue
                t = first_contact_time(seg, ob.moving_box(margin), self.dt)
                if t is not None:
                    taus.append(t)
        return min(taus) if taus else None

    def _endpoint_overlaps(self, q_new) -> bool:
        boxes_next = [
            AABB(
                ob.center + ob.vel * self.dt - ob.half - self.a_r,
                ob.center + ob.vel * self.dt + ob.half + self.a_r,
            )
            for ob in self.scene.obstacles
        ]
        for seg in self.kin.segments(q_new):
            for box in boxes_next:
                l, _, _ = segment_box_overlap(seg[0], seg[1], box)
                if l > 0.0:
                    return True
        return False

    def _reward(self, dq) -> float:
        flange = self.kin.flange(self.q + dq)
        err = np.linalg.norm(flange - self.goal)
        # URPlanner Eq.(11): r_pose = -(e_p + e_o) + phi_aux + phi_G.
        # Position-only for now (ASSUMPTIONS.md item 2). phi_aux's exact form
        # lives in their ref [9] (Eqs.12-13, not obtained); implemented as an
        # exponential proximity bonus — swap when the lab code arrives.
        phi_aux = 0.5 * np.exp(-err / 0.08)
        phi_g = 1.0 if err < self.goal_tol else 0.0
        r_pose = -err + phi_aux + phi_g
        margin = self.a_r + self.a_o
        segs_now = self.kin.segments(self.q + dq)
        r_current = uoar(segs_now, self._boxes(margin))  # <= 0
        if self.reward_mode == "uoar":
            return r_pose + self.zeta_c * r_current

        # D-UOAR-CT: swept integral + continuous TTC, both analytic.
        msegs = self._moving_segments(dq)
        total_len = sum(
            np.linalg.norm(s[1] - s[0]) for s in self.kin.segments(self.q)
        )
        swept = sum(
            swept_overlap_integral(seg, ob.moving_box(margin), self.dt)
            for seg in msegs
            for ob in self.scene.obstacles
        )
        r_swept = -swept / (total_len * self.dt)  # normalized like UOAR
        taus = [
            t
            for seg in msegs
            for ob in self.scene.obstacles
            if (t := first_contact_time(seg, ob.moving_box(margin), self.dt))
            is not None
        ]
        # Bounded TTC penalty in [-1, 0]: the unbounded -dt/(tau*+eps)
        # form blew up to ~-50 near contact and wrecked critic learning
        # (M3-curriculum: ct arms stuck at stage 1 with obstacle-phobia).
        r_ttc = -(1.0 - min(taus) / self.dt) if taus else 0.0
        return (
            r_pose
            + self.zeta_c * r_current
            + self.zeta_s * r_swept
            + self.zeta_t * r_ttc
        )
