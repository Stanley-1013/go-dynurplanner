"""Exact continuous-time collision reasoning over a control interval.

Model (tau-linearization, Loop-3 Part 5 / Loop-6 blueprint):
  - Link segment endpoints move linearly over tau in [0, dt]:
        A(tau) = A0 + uA * tau,   B(tau) = B0 + uB * tau
    (uA, uB from forward kinematics at q and q + dq: chord approximation;
     the chord error is covered by the lemma inflation, see
     `lemma_inflation` and PandaKinematics.chord_error_bound).
  - Obstacle AABB translates with constant velocity v over the interval.

Working in the box frame (subtract the box motion), the segment endpoints
remain LINEAR in tau, so every slab coordinate
    lambda_face(tau) = (face_k - A_k(tau)) / D_k(tau)
is a ratio of two linear functions of tau. The boolean predicate
"segment(tau) intersects box" therefore switches value only at roots of a
finite, enumerable set of linear/quadratic polynomials (breakpoints):

  (i)   D_k(tau) = 0                                  (slab direction sign)
  (ii)  face_k - A_k(tau) = 0                          (lambda_face = 0)
  (iii) face_k - A_k(tau) - D_k(tau) = 0               (lambda_face = 1)
  (iv)  (f - A_k(tau)) * D_m(tau) - (g - A_m(tau)) * D_k(tau) = 0
        for faces f on axis k, g on axis m, k != m     (lambda ordering swap)

Between consecutive breakpoints the predicate is constant, so evaluating the
exact static slab test at one interior sample per sub-interval yields the
EXACT first-contact time tau* (up to floating point) — no time sampling, no
tunneling. The swept-overlap integral uses per-piece Gauss-Legendre
quadrature on the (piecewise-smooth) overlap length l(tau) * |D(tau)|.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import AABB, segment_box_overlap

_EPS = 1e-12
# 8-point Gauss-Legendre nodes/weights on [-1, 1].
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)


@dataclass(frozen=True)
class MovingSegment:
    a0: np.ndarray  # endpoint A at tau=0, (3,)
    ua: np.ndarray  # endpoint A velocity, (3,)
    b0: np.ndarray
    ub: np.ndarray


@dataclass(frozen=True)
class MovingBox:
    box: AABB
    v: np.ndarray  # translation velocity, (3,)


def _relative(seg: MovingSegment, mbox: MovingBox):
    """Reduce to a static box by subtracting the box motion."""
    return (
        np.asarray(seg.a0, float),
        np.asarray(seg.ua, float) - np.asarray(mbox.v, float),
        np.asarray(seg.b0, float),
        np.asarray(seg.ub, float) - np.asarray(mbox.v, float),
    )


def _quad_roots(c2: float, c1: float, c0: float) -> list[float]:
    """Real roots of c2 x^2 + c1 x + c0, degrading gracefully in degree."""
    if abs(c2) < _EPS:
        if abs(c1) < _EPS:
            return []
        return [-c0 / c1]
    disc = c1 * c1 - 4.0 * c2 * c0
    if disc < 0.0:
        return []
    s = np.sqrt(disc)
    return [(-c1 - s) / (2.0 * c2), (-c1 + s) / (2.0 * c2)]


def _breakpoints(seg: MovingSegment, mbox: MovingBox, dt: float) -> np.ndarray:
    a0, ua, b0, ub = _relative(seg, mbox)
    d0 = b0 - a0
    w = ub - ua
    lo, hi = mbox.box.lo, mbox.box.hi
    ts: list[float] = [0.0, dt]

    # Per-axis linear events (i)-(iii).
    for k in range(3):
        ts += _quad_roots(0.0, w[k], d0[k])  # D_k(tau) = 0
        for f in (lo[k], hi[k]):
            ts += _quad_roots(0.0, -ua[k], f - a0[k])  # lambda = 0
            ts += _quad_roots(0.0, -ua[k] - w[k], f - a0[k] - d0[k])  # lambda = 1

    # Cross-axis quadratic events (iv):
    # (f - A_k(tau)) * D_m(tau) = (g - A_m(tau)) * D_k(tau)
    # with A_k(tau) = a0k + uak*tau, D_k(tau) = d0k + wk*tau.
    for k in range(3):
        for m in range(k + 1, 3):
            for f in (lo[k], hi[k]):
                for g in (lo[m], hi[m]):
                    pk0, pk1 = f - a0[k], -ua[k]  # f - A_k = pk0 + pk1*tau
                    pm0, pm1 = g - a0[m], -ua[m]
                    c2 = pk1 * w[m] - pm1 * w[k]
                    c1 = pk1 * d0[m] + pk0 * w[m] - pm1 * d0[k] - pm0 * w[k]
                    c0 = pk0 * d0[m] - pm0 * d0[k]
                    ts += _quad_roots(c2, c1, c0)

    ts = [t for t in ts if 0.0 <= t <= dt]
    ts = np.unique(np.asarray(ts, dtype=float))
    return ts


def _intersects_at(seg: MovingSegment, mbox: MovingBox, tau: float) -> bool:
    a0, ua, b0, ub = _relative(seg, mbox)
    l, _, _ = segment_box_overlap(a0 + ua * tau, b0 + ub * tau, mbox.box)
    if l > 0.0:
        return True
    # Zero-length overlap can still mean touching/inside for degenerate
    # geometry; re-check via the slab feasibility directly.
    return _slab_feasible(a0 + ua * tau, b0 + ub * tau, mbox.box)


def _slab_feasible(p_s: np.ndarray, p_e: np.ndarray, box: AABB) -> bool:
    d = p_e - p_s
    t0, t1 = 0.0, 1.0
    for k in range(3):
        if abs(d[k]) < _EPS:
            if p_s[k] < box.lo[k] - _EPS or p_s[k] > box.hi[k] + _EPS:
                return False
        else:
            a = (box.lo[k] - p_s[k]) / d[k]
            b = (box.hi[k] - p_s[k]) / d[k]
            if a > b:
                a, b = b, a
            t0 = max(t0, a)
            t1 = min(t1, b)
            if t0 > t1 + _EPS:
                return False
    return True


def first_contact_time(
    seg: MovingSegment, mbox: MovingBox, dt: float
) -> float | None:
    """Exact first tau in [0, dt] at which the segment touches the box.

    Returns None if the interval is collision-free. Exactness: the contact
    predicate can only switch at enumerated breakpoints; we return the left
    endpoint of the first sub-interval whose interior intersects (or 0.0 if
    already in contact at tau=0).
    """
    if _intersects_at(seg, mbox, 0.0):
        return 0.0
    ts = _breakpoints(seg, mbox, dt)
    for i in range(len(ts) - 1):
        left, right = ts[i], ts[i + 1]
        if right - left < _EPS:
            continue
        mid = 0.5 * (left + right)
        if _intersects_at(seg, mbox, mid):
            return float(left)
    if _intersects_at(seg, mbox, dt):
        return float(ts[-2]) if len(ts) >= 2 else float(dt)
    return None


def interval_collision_free(
    segments: list[MovingSegment], mboxes: list[MovingBox], dt: float
) -> bool:
    """Shield predicate: True iff no segment touches any box within [0, dt]."""
    for seg in segments:
        for mbox in mboxes:
            if first_contact_time(seg, mbox, dt) is not None:
                return False
    return True


def _overlap_len_at(seg: MovingSegment, mbox: MovingBox, tau: float) -> float:
    a0, ua, b0, ub = _relative(seg, mbox)
    l, _, _ = segment_box_overlap(a0 + ua * tau, b0 + ub * tau, mbox.box)
    return l


def swept_overlap_integral(seg: MovingSegment, mbox: MovingBox, dt: float) -> float:
    """integral_0^dt l(tau) dtau — the continuous-time collision cost C_true
    for one segment-box pair (D-UOAR-CT numerator term).

    Piecewise-smooth between breakpoints; integrated per piece with 8-point
    Gauss-Legendre (machine-precision for these low-degree smooth pieces).
    """
    ts = _breakpoints(seg, mbox, dt)
    total = 0.0
    for i in range(len(ts) - 1):
        left, right = ts[i], ts[i + 1]
        h = right - left
        if h < _EPS:
            continue
        x = 0.5 * h * _GL_X + 0.5 * (left + right)
        y = np.array([_overlap_len_at(seg, mbox, t) for t in x])
        total += 0.5 * h * float(np.dot(_GL_W, y))
    return total


def lemma_inflation(
    eps_lin: float, a_max: float, dt: float, eps_v: float
) -> float:
    """Total conservative inflation (Loop-6 lemma):

        eps_total = eps_lin + a_max * dt^2 / 2 + eps_v * dt

    eps_lin  : chord/linearization bound (PandaKinematics.chord_error_bound)
    a_max    : obstacle acceleration bound (constant-velocity model error)
    eps_v    : obstacle velocity estimation error bound
    Inflating each box by eps_total makes `interval_collision_free` a
    provably conservative (no-false-negative) certificate for the true
    system under these bounded model errors.
    """
    return float(eps_lin) + 0.5 * float(a_max) * float(dt) ** 2 + float(eps_v) * float(dt)
