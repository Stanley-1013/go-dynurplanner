"""Certified multi-joint jerk-horizon safety filter.

The optimizer uses smooth, affine collocation constraints.  A candidate is
returned only after the exact continuous-interval checker from
``kinodynamics`` certifies every joint and step against the original limits.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize

from .kinodynamics import (
    acceleration,
    discrete_update,
    interval_within_limits,
    position,
    velocity,
)

_TERMINAL_TOL = 1e-7


@dataclass(frozen=True)
class SafetyQPResult:
    """Outcome of a jerk-horizon safety-filter solve."""

    feasible: bool
    certified: bool
    jerk_sequence: np.ndarray | None
    v_exec: np.ndarray | None
    solve_time_s: float


def _positive_integer(name: str, value: int, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    invalid = (
        isinstance(value, (bool, np.bool_))
        or int(value) != value
        or value < minimum
    )
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def _vector(name: str, value: np.ndarray, size: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if size is not None and array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _affine_map(
    evaluate: Callable[[np.ndarray], np.ndarray], n_variables: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build an affine map by replaying zero and basis jerk sequences."""
    origin = np.zeros(n_variables)
    offset = evaluate(origin)
    matrix = np.empty((offset.size, n_variables))
    for column in range(n_variables):
        basis = np.zeros(n_variables)
        basis[column] = 1.0
        matrix[:, column] = evaluate(basis) - offset
    return offset, matrix


def _trajectory_quantities(
    jerk_flat: np.ndarray,
    q0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    h: float,
    n_steps: int,
    collocation_times: np.ndarray,
) -> np.ndarray:
    """Evaluate all quantities used by the QP, through Phase 1 helpers."""
    m = q0.size
    jerk_sequence = np.asarray(jerk_flat, dtype=float).reshape(m, n_steps)
    q_samples = np.empty((m, n_steps, collocation_times.size))
    v_samples = np.empty_like(q_samples)
    a_endpoints = np.empty((m, n_steps, 2))
    terminal_v = np.empty(m)
    terminal_a = np.empty(m)
    first_velocity = np.empty(m)

    for joint in range(m):
        state = np.array([q0[joint], v0[joint], a0[joint]], dtype=float)
        for step in range(n_steps):
            jerk = jerk_sequence[joint, step]
            for sample, tau in enumerate(collocation_times):
                q_samples[joint, step, sample] = position(*state, jerk, tau)
                v_samples[joint, step, sample] = velocity(
                    state[1], state[2], jerk, tau
                )
            a_endpoints[joint, step, 0] = acceleration(state[2], jerk, 0.0)
            a_endpoints[joint, step, 1] = acceleration(state[2], jerk, h)
            state = discrete_update(state, jerk, h)
            if step == 0:
                first_velocity[joint] = state[1]
        terminal_v[joint] = state[1]
        terminal_a[joint] = state[2]

    return np.concatenate(
        (
            q_samples.ravel(),
            v_samples.ravel(),
            a_endpoints.ravel(),
            terminal_v,
            terminal_a,
            first_velocity,
        )
    )


def _certify_candidate(
    jerk_sequence: np.ndarray,
    q0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    h: float,
    q_min: np.ndarray,
    q_max: np.ndarray,
    v_min: np.ndarray,
    v_max: np.ndarray,
    a_min: np.ndarray,
    a_max: np.ndarray,
    j_min: np.ndarray,
    j_max: np.ndarray,
) -> tuple[bool, np.ndarray | None]:
    """Replay and certify a candidate against the original hard limits."""
    m, n_steps = jerk_sequence.shape
    first_velocity = np.empty(m)
    for joint in range(m):
        state = np.array([q0[joint], v0[joint], a0[joint]], dtype=float)
        bounds = (
            q_min[joint],
            q_max[joint],
            v_min[joint],
            v_max[joint],
            a_min[joint],
            a_max[joint],
            j_min[joint],
            j_max[joint],
        )
        for step in range(n_steps):
            jerk = float(jerk_sequence[joint, step])
            if not interval_within_limits(*state, jerk, h, *bounds):
                return False, None
            state = discrete_update(state, jerk, h)
            if step == 0:
                first_velocity[joint] = state[1]
        if abs(float(state[1])) > _TERMINAL_TOL:
            return False, None
        if abs(float(state[2])) > _TERMINAL_TOL:
            return False, None
    return True, first_velocity


def solve_safety_qp(
    q0,
    v0,
    a0,
    v_nom,
    h,
    n_steps,
    q_min,
    q_max,
    v_min,
    v_max,
    a_min,
    a_max,
    j_min,
    j_max,
    weights=None,
    lambda_j=1e-3,
    collocation_points=5,
    margin_shrink_frac=0.02,
    max_retries=3,
) -> SafetyQPResult:
    """Solve and exactly certify an ``N``-step jerk sequence.

    SLSQP solves a linearly constrained quadratic objective using evenly
    spaced position/velocity collocation points.  If exact interval
    certification rejects its candidate, the position and velocity boxes are
    tightened on each retry.  No uncertified sequence is ever returned.
    """
    started = perf_counter()
    q0 = _vector("q0", q0)
    m = q0.size
    vectors = {
        "v0": _vector("v0", v0, m),
        "a0": _vector("a0", a0, m),
        "v_nom": _vector("v_nom", v_nom, m),
        "q_min": _vector("q_min", q_min, m),
        "q_max": _vector("q_max", q_max, m),
        "v_min": _vector("v_min", v_min, m),
        "v_max": _vector("v_max", v_max, m),
        "a_min": _vector("a_min", a_min, m),
        "a_max": _vector("a_max", a_max, m),
        "j_min": _vector("j_min", j_min, m),
        "j_max": _vector("j_max", j_max, m),
    }
    v0 = vectors["v0"]
    a0 = vectors["a0"]
    v_nom = vectors["v_nom"]
    q_min, q_max = vectors["q_min"], vectors["q_max"]
    v_min, v_max = vectors["v_min"], vectors["v_max"]
    a_min, a_max = vectors["a_min"], vectors["a_max"]
    j_min, j_max = vectors["j_min"], vectors["j_max"]

    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be positive and finite")
    n_steps = _positive_integer("n_steps", n_steps)
    collocation_points = _positive_integer(
        "collocation_points", collocation_points
    )
    if collocation_points < 2:
        raise ValueError("collocation_points must be at least 2")
    max_retries = _positive_integer("max_retries", max_retries, allow_zero=True)
    lambda_j = float(lambda_j)
    if not np.isfinite(lambda_j) or lambda_j < 0.0:
        raise ValueError("lambda_j must be non-negative and finite")
    margin_shrink_frac = float(margin_shrink_frac)
    if not np.isfinite(margin_shrink_frac) or margin_shrink_frac < 0.0:
        raise ValueError("margin_shrink_frac must be non-negative and finite")

    for name, lower, upper in (
        ("q", q_min, q_max),
        ("v", v_min, v_max),
        ("a", a_min, a_max),
        ("j", j_min, j_max),
    ):
        if np.any(lower > upper):
            raise ValueError(
                f"each {name} lower limit must not exceed its upper limit"
            )

    if weights is None:
        weights = np.ones(m)
    else:
        weights = _vector("weights", weights, m)
    if np.any(weights < 0.0):
        raise ValueError("weights must be non-negative")

    n_variables = m * n_steps
    collocation_times = np.linspace(0.0, h, collocation_points)
    q_count = m * n_steps * collocation_points
    v_count = q_count
    a_count = m * n_steps * 2
    terminal_count = 2 * m
    box_count = q_count + v_count + a_count

    def quantities(jerk_flat: np.ndarray) -> np.ndarray:
        return _trajectory_quantities(
            jerk_flat, q0, v0, a0, h, n_steps, collocation_times
        )

    affine_offset, affine_matrix = _affine_map(quantities, n_variables)
    box_offset = affine_offset[:box_count]
    box_matrix = affine_matrix[:box_count]
    terminal_offset = affine_offset[box_count : box_count + terminal_count]
    terminal_matrix = affine_matrix[box_count : box_count + terminal_count]
    v1_offset = affine_offset[-m:]
    v1_matrix = affine_matrix[-m:]

    jerk_lower = np.repeat(j_min, n_steps)
    jerk_upper = np.repeat(j_max, n_steps)
    optimizer_bounds = Bounds(jerk_lower, jerk_upper)
    terminal_constraint = LinearConstraint(
        terminal_matrix, -terminal_offset, -terminal_offset
    )

    def objective(jerk_flat: np.ndarray) -> float:
        velocity_error = v1_offset + v1_matrix @ jerk_flat - v_nom
        return float(
            np.dot(weights, velocity_error * velocity_error)
            + lambda_j * np.dot(jerk_flat, jerk_flat)
        )

    def objective_jacobian(jerk_flat: np.ndarray) -> np.ndarray:
        velocity_error = v1_offset + v1_matrix @ jerk_flat - v_nom
        tracking_gradient = 2.0 * (
            v1_matrix.T @ (weights * velocity_error)
        )
        return tracking_gradient + 2.0 * lambda_j * jerk_flat

    # A terminal least-squares solution is a more useful start than all-zero
    # jerk when the incoming velocity or acceleration is nonzero.
    initial, *_ = np.linalg.lstsq(
        terminal_matrix, -terminal_offset, rcond=None
    )
    initial = np.clip(initial, jerk_lower, jerk_upper)

    q_range = q_max - q_min
    v_range = v_max - v_min
    for attempt in range(max_retries + 1):
        q_margin = attempt * margin_shrink_frac * q_range
        v_margin = attempt * margin_shrink_frac * v_range
        tight_q_min, tight_q_max = q_min + q_margin, q_max - q_margin
        tight_v_min, tight_v_max = v_min + v_margin, v_max - v_margin
        if np.any(tight_q_min > tight_q_max) or np.any(tight_v_min > tight_v_max):
            continue

        q_lower = np.broadcast_to(
            tight_q_min[:, None, None], (m, n_steps, collocation_points)
        ).ravel()
        q_upper = np.broadcast_to(
            tight_q_max[:, None, None], (m, n_steps, collocation_points)
        ).ravel()
        v_lower = np.broadcast_to(
            tight_v_min[:, None, None], (m, n_steps, collocation_points)
        ).ravel()
        v_upper = np.broadcast_to(
            tight_v_max[:, None, None], (m, n_steps, collocation_points)
        ).ravel()
        a_lower = np.broadcast_to(
            a_min[:, None, None], (m, n_steps, 2)
        ).ravel()
        a_upper = np.broadcast_to(
            a_max[:, None, None], (m, n_steps, 2)
        ).ravel()
        box_lower = np.concatenate((q_lower, v_lower, a_lower))
        box_upper = np.concatenate((q_upper, v_upper, a_upper))
        box_constraint = LinearConstraint(
            box_matrix, box_lower - box_offset, box_upper - box_offset
        )

        solution = minimize(
            objective,
            initial,
            method="SLSQP",
            jac=objective_jacobian,
            bounds=optimizer_bounds,
            constraints=(box_constraint, terminal_constraint),
            options={"ftol": 1e-10, "maxiter": 300, "disp": False},
        )
        if np.all(np.isfinite(solution.x)):
            initial = np.clip(solution.x, jerk_lower, jerk_upper)
            candidate = initial.reshape(m, n_steps)
            certified, v_exec = _certify_candidate(
                candidate,
                q0,
                v0,
                a0,
                h,
                q_min,
                q_max,
                v_min,
                v_max,
                a_min,
                a_max,
                j_min,
                j_max,
            )
            if certified:
                return SafetyQPResult(
                    feasible=True,
                    certified=True,
                    jerk_sequence=candidate.copy(),
                    v_exec=v_exec.copy(),
                    solve_time_s=perf_counter() - started,
                )

    return SafetyQPResult(
        feasible=False,
        certified=False,
        jerk_sequence=None,
        v_exec=None,
        solve_time_s=perf_counter() - started,
    )
