"""Static segment-vs-AABB overlap: URPlanner Eq.(6)-(10).

The obstacle is an axis-aligned bounding box (already expanded by the link
cylinder radius a_r, the safety offset a_o, and — for the conservative
continuous-time checker — the lemma inflation). The manipulator link is a
line segment. Overlap length is computed by the slab method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass(frozen=True)
class AABB:
    lo: np.ndarray  # (3,)
    hi: np.ndarray  # (3,)

    def expanded(self, margin: float) -> "AABB":
        m = float(margin)
        return AABB(self.lo - m, self.hi + m)

    def contains(self, p: np.ndarray) -> bool:
        return bool(np.all(p >= self.lo - _EPS) and np.all(p <= self.hi + _EPS))


def segment_box_overlap(
    p_s: np.ndarray, p_e: np.ndarray, box: AABB
) -> tuple[float, float, float]:
    """Overlap length of segment [p_s, p_e] with `box` (slab method).

    Returns (length, lam_enter, lam_exit); (0.0, 1.0, 0.0) if disjoint.
    lam parametrizes the segment: p(lam) = p_s + lam * (p_e - p_s), lam in [0,1].
    """
    p_s = np.asarray(p_s, dtype=float)
    p_e = np.asarray(p_e, dtype=float)
    d = p_e - p_s
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < _EPS:
            if p_s[k] < box.lo[k] or p_s[k] > box.hi[k]:
                return 0.0, 1.0, 0.0
        else:
            a = (box.lo[k] - p_s[k]) / d[k]
            b = (box.hi[k] - p_s[k]) / d[k]
            if a > b:
                a, b = b, a
            t0 = max(t0, a)
            t1 = min(t1, b)
            if t0 > t1:
                return 0.0, 1.0, 0.0
    return (t1 - t0) * float(np.linalg.norm(d)), t0, t1


def uoar(segments: np.ndarray, boxes: list[AABB]) -> float:
    """URPlanner Eq.(10): r_UOAR = - sum_j sum_g l_jg / sum_j L_j.

    `segments`: shape (J, 2, 3). Dense, minimum-distance-independent,
    non-positive; 0 iff no overlap anywhere.
    """
    segments = np.asarray(segments, dtype=float)
    total_len = float(np.sum(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)))
    if total_len < _EPS:
        return 0.0
    overlap = 0.0
    for seg in segments:
        for box in boxes:
            l, _, _ = segment_box_overlap(seg[0], seg[1], box)
            overlap += l
    return -overlap / total_len
