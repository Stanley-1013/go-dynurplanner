"""Single-joint jerk dynamics and continuous-interval limit checks.

The control input is constant jerk over one control period.  Consequently,
acceleration is linear, velocity is quadratic, and position is cubic in
``tau``.  Position and velocity extrema are found from their analytic
critical points; no time sampling is used.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

_EPS = 1e-12


class IntervalExtrema(NamedTuple):
    """True position and velocity extrema over one closed interval."""

    q_min: float
    q_max: float
    v_min: float
    v_max: float


def _positive_period(h: float) -> float:
    h = float(h)
    if h <= 0.0:
        raise ValueError("control period h must be positive")
    return h


def discrete_matrices(h: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact one-step matrices ``A`` and ``B`` for period ``h``."""
    h = _positive_period(h)
    A = np.array(
        [
            [1.0, h, 0.5 * h**2],
            [0.0, 1.0, h],
            [0.0, 0.0, 1.0],
        ]
    )
    B = np.array([h**3 / 6.0, 0.5 * h**2, h])
    return A, B


def discrete_update(state: np.ndarray, j: float, h: float) -> np.ndarray:
    """Advance one exact jerk-controlled step from state ``(q, v, a)``."""
    state = np.asarray(state, dtype=float)
    if state.shape != (3,):
        raise ValueError("state must have shape (3,) for (q, v, a)")
    A, B = discrete_matrices(h)
    return A @ state + B * float(j)


def acceleration(a0: float, j: float, tau: float) -> float:
    """Acceleration ``a(tau)`` under constant jerk ``j``."""
    return float(a0) + float(j) * float(tau)


def velocity(v0: float, a0: float, j: float, tau: float) -> float:
    """Velocity ``v(tau)`` under constant jerk ``j``."""
    tau = float(tau)
    return float(v0) + float(a0) * tau + 0.5 * float(j) * tau**2


def position(q0: float, v0: float, a0: float, j: float, tau: float) -> float:
    """Position ``q(tau)`` under constant jerk ``j``."""
    tau = float(tau)
    return (
        float(q0)
        + float(v0) * tau
        + 0.5 * float(a0) * tau**2
        + float(j) * tau**3 / 6.0
    )


def continuous_state(
    q0: float, v0: float, a0: float, j: float, tau: float
) -> np.ndarray:
    """Return ``(q(tau), v(tau), a(tau))`` for the continuous trajectory."""
    return np.array(
        [
            position(q0, v0, a0, j, tau),
            velocity(v0, a0, j, tau),
            acceleration(a0, j, tau),
        ]
    )


def _quadratic_roots(c2: float, c1: float, c0: float) -> list[float]:
    """Return all real roots, including degenerate linear cases."""
    scale = max(abs(c2), abs(c1), abs(c0), 1.0)
    if abs(c2) <= _EPS * scale:
        if abs(c1) <= _EPS * scale:
            return []
        return [-c0 / c1]

    disc = c1 * c1 - 4.0 * c2 * c0
    disc_tol = _EPS * max(c1 * c1 + abs(4.0 * c2 * c0), 1.0)
    if disc < -disc_tol:
        return []
    disc = max(disc, 0.0)
    root_disc = float(np.sqrt(disc))
    return [
        (-c1 - root_disc) / (2.0 * c2),
        (-c1 + root_disc) / (2.0 * c2),
    ]


def _closed_interval_time(tau: float, h: float) -> float | None:
    if tau < -_EPS or tau > h + _EPS:
        return None
    return float(np.clip(tau, 0.0, h))


def interval_extrema(
    q0: float, v0: float, a0: float, j: float, h: float
) -> IntervalExtrema:
    """Return exact extrema of ``q(tau)`` and ``v(tau)`` on ``[0, h]``.

    Velocity can have an interior extremum where ``a(tau) = 0``. Position
    can have interior extrema at the real roots of the quadratic
    ``v(tau) = 0``. Roots within floating-point tolerance of an endpoint are
    clamped to the closed interval before values are compared.
    """
    h = _positive_period(h)
    q0, v0, a0, j = map(float, (q0, v0, a0, j))

    v_times = [0.0, h]
    if abs(j) > _EPS:
        tau = _closed_interval_time(-a0 / j, h)
        if tau is not None:
            v_times.append(tau)

    q_times = [0.0, h]
    for root in _quadratic_roots(0.5 * j, a0, v0):
        tau = _closed_interval_time(root, h)
        if tau is not None:
            q_times.append(tau)

    q_values = [position(q0, v0, a0, j, tau) for tau in q_times]
    v_values = [velocity(v0, a0, j, tau) for tau in v_times]
    return IntervalExtrema(
        q_min=float(min(q_values)),
        q_max=float(max(q_values)),
        v_min=float(min(v_values)),
        v_max=float(max(v_values)),
    )


def _inside(value: float, lower: float, upper: float) -> bool:
    return lower - _EPS <= value <= upper + _EPS


def interval_within_limits(
    q0: float,
    v0: float,
    a0: float,
    j: float,
    h: float,
    q_min: float,
    q_max: float,
    v_min: float,
    v_max: float,
    a_min: float,
    a_max: float,
    j_min: float,
    j_max: float,
) -> bool:
    """Return whether the whole jerk-controlled interval obeys all limits."""
    limits = (q_min, q_max, v_min, v_max, a_min, a_max, j_min, j_max)
    q_min, q_max, v_min, v_max, a_min, a_max, j_min, j_max = map(
        float, limits
    )
    if q_min > q_max or v_min > v_max or a_min > a_max or j_min > j_max:
        raise ValueError("each lower limit must not exceed its upper limit")

    h = _positive_period(h)
    extrema = interval_extrema(q0, v0, a0, j, h)
    a1 = acceleration(a0, j, h)
    a_interval_min = min(float(a0), a1)
    a_interval_max = max(float(a0), a1)
    return bool(
        _inside(extrema.q_min, q_min, q_max)
        and _inside(extrema.q_max, q_min, q_max)
        and _inside(extrema.v_min, v_min, v_max)
        and _inside(extrema.v_max, v_min, v_max)
        and _inside(a_interval_min, a_min, a_max)
        and _inside(a_interval_max, a_min, a_max)
        and _inside(float(j), j_min, j_max)
    )


def braking_feasible(
    q0: float,
    v0: float,
    a0: float,
    h: float,
    n_steps: int,
    q_min: float,
    q_max: float,
    v_min: float,
    v_max: float,
    a_min: float,
    a_max: float,
    j_min: float,
    j_max: float,
) -> bool:
    """Return whether a conservative braking witness stops within ``N`` steps.

    This Phase-1 v1 is deliberately conservative, not the full horizon QP
    planned for Phase 2.  It drives acceleration toward the strongest value
    opposing the initial velocity, checking every continuous interval, and
    uses an exact one- or two-step terminal profile as soon as it can make both
    velocity and acceleration zero. The stopped state can then be padded with
    zero jerk through the rest of the horizon. ``False`` therefore means this
    fixed maximum-deceleration profile found no safe witness; a less
    restrictive jerk sequence may exist.
    """
    if isinstance(n_steps, bool) or int(n_steps) != n_steps or n_steps < 1:
        raise ValueError("n_steps must be a positive integer")
    n_steps = int(n_steps)
    h = _positive_period(h)
    q, v, a = map(float, (q0, v0, a0))
    bounds = tuple(
        map(float, (q_min, q_max, v_min, v_max, a_min, a_max, j_min, j_max))
    )
    q_min, q_max, v_min, v_max, a_min, a_max, j_min, j_max = bounds
    if q_min > q_max or v_min > v_max or a_min > a_max or j_min > j_max:
        raise ValueError("each lower limit must not exceed its upper limit")
    if not (
        _inside(q, q_min, q_max)
        and _inside(v, v_min, v_max)
        and _inside(a, a_min, a_max)
    ):
        return False

    if abs(v) <= _EPS and abs(a) <= _EPS:
        return _inside(0.0, j_min, j_max)

    direction_value = v if abs(v) > _EPS else a
    direction = 1.0 if direction_value >= 0.0 else -1.0
    target_acceleration = a_min if direction > 0.0 else a_max

    for step in range(n_steps):
        remaining = n_steps - step

        # First try to settle both v and a in one exact step. A stopped state
        # is a valid witness for all remaining steps when zero jerk is allowed.
        j_settle = -a / h
        settled = discrete_update(np.array([q, v, a]), j_settle, h)
        if (
            _inside(0.0, j_min, j_max)
            and _inside(j_settle, j_min, j_max)
            and abs(float(settled[1])) <= _EPS
            and abs(float(settled[2])) <= _EPS
            and interval_within_limits(q, v, a, j_settle, h, *bounds)
        ):
            return True

        # Two constant-jerk intervals have a closed-form solution for
        # v_{k+2}=a_{k+2}=0. This is the terminal S-curve release segment.
        if remaining >= 2 and _inside(0.0, j_min, j_max):
            j_first = -(v + 1.5 * h * a) / h**2
            j_second = -a / h - j_first
            if (
                _inside(j_first, j_min, j_max)
                and interval_within_limits(q, v, a, j_first, h, *bounds)
            ):
                first = discrete_update(np.array([q, v, a]), j_first, h)
                q1, v1, a1 = map(float, first)
                if (
                    _inside(j_second, j_min, j_max)
                    and interval_within_limits(
                        q1, v1, a1, j_second, h, *bounds
                    )
                ):
                    second = discrete_update(first, j_second, h)
                    stopped = (
                        abs(float(second[1])) <= _EPS
                        and abs(float(second[2])) <= _EPS
                    )
                    if stopped:
                        return True

        # Otherwise apply the largest admissible acceleration change toward
        # maximum braking, respecting the acceleration endpoint bound.
        j_brake = float(
            np.clip((target_acceleration - a) / h, j_min, j_max)
        )
        if not interval_within_limits(q, v, a, j_brake, h, *bounds):
            return False

        q, v, a = discrete_update(np.array([q, v, a]), j_brake, h)
        q, v, a = float(q), float(v), float(a)
        if direction * v < -_EPS:
            return False

    return abs(v) <= _EPS and abs(a) <= _EPS and _inside(0.0, j_min, j_max)
