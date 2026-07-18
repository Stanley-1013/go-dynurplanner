"""Differentiable box-only kinodynamic safety QP.

The CVXPY problem is compiled once for each fixed structural/physical
configuration.  Per-transition affine maps remain parameters because they
depend on the transition's own joint position, velocity, and acceleration.
Only the nominal first-step velocity is differentiated by the training code.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

from .safety_qp import _affine_map, _trajectory_quantities


@dataclass(frozen=True)
class DifferentiableQPLayer:
    """A compiled layer and the dimensions needed to assemble its inputs."""

    layer: CvxpyLayer
    m: int
    n_steps: int
    collocation_points: int
    n_variables: int
    box_count: int


_LAYER_CACHE: dict[tuple, DifferentiableQPLayer] = {}
_SOLVER_ARGS = {
    "solve_method": "Clarabel",
    "tol_gap_abs": 1e-7,
    "tol_gap_rel": 1e-7,
    "tol_feas": 1e-7,
}


def _finite_vector(name: str, value, m: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (m,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector with shape ({m},)")
    return array


def build_differentiable_qp_layer(
    m: int,
    n_steps: int,
    collocation_points: int,
    lambda_j: float,
    j_min,
    j_max,
    weights=None,
) -> DifferentiableQPLayer:
    """Return a cached, DPP-compliant box-only safety-QP layer.

    Jerk bounds, tracking weights, and ``lambda_j`` are constants in the
    compiled problem.  They are consequently part of the cache key, while
    every state-dependent affine coefficient is supplied on each call.
    """
    if isinstance(m, (bool, np.bool_)) or int(m) != m or m < 1:
        raise ValueError("m must be a positive integer")
    if (
        isinstance(n_steps, (bool, np.bool_))
        or int(n_steps) != n_steps
        or n_steps < 1
    ):
        raise ValueError("n_steps must be a positive integer")
    if (
        isinstance(collocation_points, (bool, np.bool_))
        or int(collocation_points) != collocation_points
        or collocation_points < 2
    ):
        raise ValueError("collocation_points must be an integer of at least 2")
    m = int(m)
    n_steps = int(n_steps)
    collocation_points = int(collocation_points)
    lambda_j = float(lambda_j)
    if not np.isfinite(lambda_j) or lambda_j < 0.0:
        raise ValueError("lambda_j must be non-negative and finite")

    j_min_array = _finite_vector("j_min", j_min, m)
    j_max_array = _finite_vector("j_max", j_max, m)
    if np.any(j_min_array > j_max_array):
        raise ValueError("each jerk lower limit must not exceed its upper limit")
    weights_array = (
        np.ones(m) if weights is None else _finite_vector("weights", weights, m)
    )
    if np.any(weights_array < 0.0):
        raise ValueError("weights must be non-negative")

    key = (
        m,
        n_steps,
        collocation_points,
        lambda_j,
        tuple(j_min_array),
        tuple(j_max_array),
        tuple(weights_array),
    )
    cached = _LAYER_CACHE.get(key)
    if cached is not None:
        return cached

    n_variables = m * n_steps
    q_count = m * n_steps * collocation_points
    v_count = q_count
    a_count = m * n_steps * 2
    box_count = q_count + v_count + a_count

    jerk = cp.Variable(n_variables)
    v_nom_param = cp.Parameter(m)
    box_matrix_param = cp.Parameter((box_count, n_variables))
    box_lower_param = cp.Parameter(box_count)
    box_upper_param = cp.Parameter(box_count)
    v1_matrix_param = cp.Parameter((m, n_variables))
    v1_offset_param = cp.Parameter(m)

    v1 = v1_offset_param + v1_matrix_param @ jerk
    objective = cp.Minimize(
        cp.sum(cp.multiply(weights_array, cp.square(v1 - v_nom_param)))
        + lambda_j * cp.sum_squares(jerk)
    )
    jerk_lower = np.repeat(j_min_array, n_steps)
    jerk_upper = np.repeat(j_max_array, n_steps)
    constraints = [
        box_lower_param <= box_matrix_param @ jerk,
        box_matrix_param @ jerk <= box_upper_param,
        cp.Constant(jerk_lower) <= jerk,
        jerk <= cp.Constant(jerk_upper),
    ]
    problem = cp.Problem(objective, constraints)
    if not problem.is_dpp():
        raise RuntimeError("differentiable safety QP is not DPP-compliant")

    layer = CvxpyLayer(
        problem,
        parameters=[
            v_nom_param,
            box_matrix_param,
            box_lower_param,
            box_upper_param,
            v1_matrix_param,
            v1_offset_param,
        ],
        variables=[jerk],
    )
    compiled = DifferentiableQPLayer(
        layer=layer,
        m=m,
        n_steps=n_steps,
        collocation_points=collocation_points,
        n_variables=n_variables,
        box_count=box_count,
    )
    _LAYER_CACHE[key] = compiled
    return compiled


def differentiable_v_exec(
    q_batch,
    v_batch,
    a_batch,
    v_nom_batch: torch.Tensor,
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
) -> torch.Tensor:
    """Solve a native batched safety QP and return differentiable ``v_1``.

    ``q_batch``, ``v_batch``, and ``a_batch`` are physical-state NumPy
    arrays and are intentionally outside autograd.  ``v_nom_batch`` is the
    sole differentiable input used by the actor update.
    """
    if not isinstance(v_nom_batch, torch.Tensor) or v_nom_batch.ndim != 2:
        raise ValueError("v_nom_batch must be a two-dimensional torch tensor")
    if not v_nom_batch.is_floating_point():
        raise ValueError("v_nom_batch must have a floating-point dtype")
    batch_size, m = v_nom_batch.shape
    if batch_size < 1 or m < 1:
        raise ValueError("v_nom_batch must have non-empty batch and joint axes")

    state_batches = {}
    for name, value in (
        ("q_batch", q_batch),
        ("v_batch", v_batch),
        ("a_batch", a_batch),
    ):
        array = np.asarray(value, dtype=float)
        if array.shape != (batch_size, m) or not np.all(np.isfinite(array)):
            raise ValueError(
                f"{name} must be finite with shape ({batch_size}, {m})"
            )
        state_batches[name] = array
    q_batch = state_batches["q_batch"]
    v_batch = state_batches["v_batch"]
    a_batch = state_batches["a_batch"]

    limits = {
        name: _finite_vector(name, value, m)
        for name, value in (
            ("q_min", q_min),
            ("q_max", q_max),
            ("v_min", v_min),
            ("v_max", v_max),
            ("a_min", a_min),
            ("a_max", a_max),
            ("j_min", j_min),
            ("j_max", j_max),
        )
    }
    for stem in ("q", "v", "a", "j"):
        if np.any(limits[f"{stem}_min"] > limits[f"{stem}_max"]):
            raise ValueError(
                f"each {stem} lower limit must not exceed its upper limit"
            )

    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("h must be positive and finite")
    compiled = build_differentiable_qp_layer(
        m,
        n_steps,
        collocation_points,
        lambda_j,
        limits["j_min"],
        limits["j_max"],
        weights=weights,
    )

    collocation_times = np.linspace(0.0, h, compiled.collocation_points)
    q_lower = np.broadcast_to(
        limits["q_min"][:, None, None],
        (m, compiled.n_steps, compiled.collocation_points),
    ).ravel()
    q_upper = np.broadcast_to(
        limits["q_max"][:, None, None],
        (m, compiled.n_steps, compiled.collocation_points),
    ).ravel()
    v_lower = np.broadcast_to(
        limits["v_min"][:, None, None],
        (m, compiled.n_steps, compiled.collocation_points),
    ).ravel()
    v_upper = np.broadcast_to(
        limits["v_max"][:, None, None],
        (m, compiled.n_steps, compiled.collocation_points),
    ).ravel()
    a_lower = np.broadcast_to(
        limits["a_min"][:, None, None], (m, compiled.n_steps, 2)
    ).ravel()
    a_upper = np.broadcast_to(
        limits["a_max"][:, None, None], (m, compiled.n_steps, 2)
    ).ravel()
    box_lower = np.concatenate((q_lower, v_lower, a_lower))
    box_upper = np.concatenate((q_upper, v_upper, a_upper))

    box_matrices = []
    shifted_lowers = []
    shifted_uppers = []
    v1_matrices = []
    v1_offsets = []
    for batch_index in range(batch_size):
        def quantities(jerk_flat, index=batch_index):
            return _trajectory_quantities(
                jerk_flat,
                q_batch[index],
                v_batch[index],
                a_batch[index],
                h,
                compiled.n_steps,
                collocation_times,
            )

        affine_offset, affine_matrix = _affine_map(
            quantities, compiled.n_variables
        )
        box_offset = affine_offset[: compiled.box_count]
        box_matrices.append(affine_matrix[: compiled.box_count])
        shifted_lowers.append(box_lower - box_offset)
        shifted_uppers.append(box_upper - box_offset)
        v1_offsets.append(affine_offset[-m:])
        v1_matrices.append(affine_matrix[-m:])

    tensor_options = {
        "dtype": v_nom_batch.dtype,
        "device": v_nom_batch.device,
    }
    box_matrix_tensor = torch.as_tensor(
        np.stack(box_matrices), **tensor_options
    )
    box_lower_tensor = torch.as_tensor(
        np.stack(shifted_lowers), **tensor_options
    )
    box_upper_tensor = torch.as_tensor(
        np.stack(shifted_uppers), **tensor_options
    )
    v1_matrix_tensor = torch.as_tensor(
        np.stack(v1_matrices), **tensor_options
    )
    v1_offset_tensor = torch.as_tensor(
        np.stack(v1_offsets), **tensor_options
    )

    # cvxpylayers interprets the leading axis as the native batch dimension
    # for every parameter here; no per-item solve loop occurs at this level.
    (jerk_solution,) = compiled.layer(
        v_nom_batch,
        box_matrix_tensor,
        box_lower_tensor,
        box_upper_tensor,
        v1_matrix_tensor,
        v1_offset_tensor,
        solver_args=_SOLVER_ARGS,
    )
    jerk_solution = jerk_solution.to(**tensor_options)
    return v1_offset_tensor + torch.bmm(
        v1_matrix_tensor, jerk_solution.unsqueeze(-1)
    ).squeeze(-1)
