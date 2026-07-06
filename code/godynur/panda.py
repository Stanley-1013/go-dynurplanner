"""Franka Emika Panda kinematics for the URPlanner parameterized space.

Conventions
-----------
URPlanner Eq.(4) uses Craig's *modified* DH:
    T_i^{i-1} = Rot_x(alpha_{i-1}) @ Trans_x(a_{i-1}) @ Rot_z(q_i) @ Trans_z(d_i)

The Panda modified-DH table below follows the Franka Emika official
control-parameters documentation (also shipped in roboticstoolbox-python).
Rows are (alpha_{i-1}, a_{i-1}, d_i); q_i is the joint variable. The last
row is the fixed flange transform (q = 0).

URPlanner link consolidation (paper §III-A): adjacent link pairs
(1,2), (3,4), (5,6), (7,8/flange) are consolidated, giving 4 line segments
between the 5 retained frame origins O0(base-top), O2, O4, O6, O_flange.
"""

from __future__ import annotations

import numpy as np

_PI_2 = np.pi / 2.0

# (alpha_{i-1}, a_{i-1}, d_i) for joints 1..7, then flange.
_MDH = np.array(
    [
        [0.0, 0.0, 0.333],
        [-_PI_2, 0.0, 0.0],
        [_PI_2, 0.0, 0.316],
        [_PI_2, 0.0825, 0.0],
        [-_PI_2, -0.0825, 0.384],
        [_PI_2, 0.0, 0.0],
        [_PI_2, 0.088, 0.0],
        [0.0, 0.0, 0.107],  # flange (fixed)
    ]
)

# Official joint position limits (rad) and velocity limits (rad/s), Panda.
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
DQ_MAX = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100])

# Consolidated segment definition: indices into the origin list
# [O0, O1, ..., O7, O_flange] (9 origins, index 0..8).
# URPlanner consolidates links (1,2),(3,4),(5,6),(7,flange):
_SEGMENT_ORIGIN_PAIRS = [(0, 2), (2, 4), (4, 6), (6, 8)]


def _mdh_transform(alpha: float, a: float, q: float, d: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    cq, sq = np.cos(q), np.sin(q)
    return np.array(
        [
            [cq, -sq, 0.0, a],
            [sq * ca, cq * ca, -sa, -d * sa],
            [sq * sa, cq * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


class PandaKinematics:
    """Forward kinematics and URPlanner segment model for the Panda arm."""

    n_joints = 7

    def joint_origins(self, q: np.ndarray) -> np.ndarray:
        """Return the 9 frame origins [O0..O7, O_flange] in the base frame.

        O0 is the origin of joint-1's frame (base frame origin per MDH,
        located at the base; the d_1 offset places O1 at shoulder height).
        """
        q = np.asarray(q, dtype=float)
        assert q.shape == (7,), "expects 7 joint angles"
        origins = np.zeros((9, 3))
        T = np.eye(4)
        origins[0] = T[:3, 3]
        for i in range(8):
            qi = q[i] if i < 7 else 0.0
            alpha, a, d = _MDH[i]
            T = T @ _mdh_transform(alpha, a, qi, d)
            origins[i + 1] = T[:3, 3]
        return origins

    def segments(self, q: np.ndarray) -> np.ndarray:
        """Return the 4 consolidated URPlanner line segments, shape (4, 2, 3)."""
        o = self.joint_origins(q)
        return np.array([[o[i], o[j]] for i, j in _SEGMENT_ORIGIN_PAIRS])

    def flange(self, q: np.ndarray) -> np.ndarray:
        return self.joint_origins(q)[8]

    def chord_error_bound(self, q: np.ndarray, dq: np.ndarray) -> float:
        """Conservative bound on chord (linear-interpolation) error over one step.

        For the joint-space linear path q(s) = q + s*dq, s in [0,1], any body
        point p satisfies the standard chord bound
            dev <= (1/8) * max_s |d^2 p / d s^2|.
        For a revolute chain, d2p/ds2 = sum_ij H_ij dq_i dq_j with
        H_ij = (z_i x z_j) x (p - o_j) + z_j x (z_i x (p - o_j)) for i <= j,
        so |H_ij| <= 2 * R_max(i,j), where R_k is the downstream arm length
        from joint k's origin (|p - o_k| <= R_k for p distal of k). Since
        R_max(i,j) = min(R_i, R_j) <= sqrt(R_i * R_j),

            dev <= (1/8) * 2 * (sum_i sqrt(R_i) |dq_i|)^2
                 = (1/4) * (sum_i sqrt(R_i) |dq_i|)^2.

        NOTE: the naive per-joint sum  sum_i R_i dq_i^2 / 8  (Loop-6 blueprint
        first draft) is NOT valid — it drops the i != j cross terms, and the
        empirical test caught it violating the bound. This corrected form is
        what the conservativeness lemma must cite.
        """
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        o = self.joint_origins(q)
        link_len = np.linalg.norm(np.diff(o, axis=0), axis=1)  # 8 lengths
        s = 0.0
        for i in range(7):
            downstream = float(np.sum(link_len[i:]))
            s += np.sqrt(downstream) * abs(dq[i])
        return 0.25 * s * s
