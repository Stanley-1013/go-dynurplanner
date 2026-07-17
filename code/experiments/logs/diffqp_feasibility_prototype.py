"""Feasibility prototype: differentiable box-only safety QP via cvxpylayers.

Not wired into anything yet. Goal: (1) confirm cvxpylayers can express and
solve the SAME box-constrained QP structure as safety_qp.solve_safety_qp
(require_terminal_stop=False, the real-time live-control config), reusing
the existing verified _affine_map/_trajectory_quantities machinery for the
affine coefficients so there's no second, independently-fallible derivation
of the same math; (2) confirm the forward solution matches the existing
scipy-based solver closely; (3) confirm backward gradients w.r.t. v_nom are
finite and match a numerical finite-difference check.
"""
import sys

sys.path.insert(0, "/home/han/claude_project/papper-research/code")

import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from godynur.safety_qp import solve_safety_qp, _affine_map, _trajectory_quantities
from godynur.panda import Q_MIN, Q_MAX, DQ_MAX, DDQ_MAX, DDDQ_MAX

m = 7
h = 0.05
n_steps = 2
collocation_points = 5
lambda_j = 1e-3

q0 = (Q_MIN + Q_MAX) / 2.0
v0 = np.zeros(m)
a0 = np.zeros(m)
q_min, q_max = Q_MIN, Q_MAX
v_min, v_max = -DQ_MAX, DQ_MAX
a_min, a_max = -DDQ_MAX, DDQ_MAX
j_min, j_max = -DDDQ_MAX, DDDQ_MAX

n_variables = m * n_steps
collocation_times = np.linspace(0.0, h, collocation_points)
q_count = m * n_steps * collocation_points
v_count = q_count
a_count = m * n_steps * 2
box_count = q_count + v_count + a_count


def quantities(jerk_flat):
    return _trajectory_quantities(jerk_flat, q0, v0, a0, h, n_steps, collocation_times)


affine_offset, affine_matrix = _affine_map(quantities, n_variables)
box_offset = affine_offset[:box_count]
box_matrix = affine_matrix[:box_count]
v1_offset = affine_offset[-m:]
v1_matrix = affine_matrix[-m:]

q_lower = np.broadcast_to(q_min[:, None, None], (m, n_steps, collocation_points)).ravel()
q_upper = np.broadcast_to(q_max[:, None, None], (m, n_steps, collocation_points)).ravel()
v_lower = np.broadcast_to(v_min[:, None, None], (m, n_steps, collocation_points)).ravel()
v_upper = np.broadcast_to(v_max[:, None, None], (m, n_steps, collocation_points)).ravel()
a_lower = np.broadcast_to(a_min[:, None, None], (m, n_steps, 2)).ravel()
a_upper = np.broadcast_to(a_max[:, None, None], (m, n_steps, 2)).ravel()
box_lower = np.concatenate((q_lower, v_lower, a_lower))
box_upper = np.concatenate((q_upper, v_upper, a_upper))

jerk_lower = np.repeat(j_min, n_steps)
jerk_upper = np.repeat(j_max, n_steps)

# --- Build the CVXPY problem (v_nom is the differentiable Parameter) ---
jerk = cp.Variable(n_variables)
v_nom_param = cp.Parameter(m)
weights = np.ones(m)

v1 = v1_offset + v1_matrix @ jerk
tracking = cp.sum(cp.multiply(weights, cp.square(v1 - v_nom_param)))
reg = lambda_j * cp.sum_squares(jerk)
objective = cp.Minimize(tracking + reg)
constraints = [
    box_lower - box_offset <= box_matrix @ jerk,
    box_matrix @ jerk <= box_upper - box_offset,
    jerk_lower <= jerk,
    jerk <= jerk_upper,
]
problem = cp.Problem(objective, constraints)
assert problem.is_dpp(), "problem is not DPP -- cvxpylayers requires this"

layer = CvxpyLayer(problem, parameters=[v_nom_param], variables=[jerk])

# --- Forward check against the existing scipy solver ---
v_nom_np = 0.3 * DQ_MAX * np.array([1, -1, 1, -1, 1, -1, 1], dtype=float)

scipy_result = solve_safety_qp(
    q0, v0, a0, v_nom_np, h, n_steps,
    q_min, q_max, v_min, v_max, a_min, a_max, j_min, j_max,
    lambda_j=lambda_j, collocation_points=collocation_points,
    require_terminal_stop=False,
)
print("scipy certified:", scipy_result.certified, "v_exec:", scipy_result.v_exec)

v_nom_t = torch.tensor(v_nom_np, dtype=torch.float64, requires_grad=True)
(jerk_sol,) = layer(v_nom_t, solver_args={"solve_method": "ECOS"} if False else {})
v1_sol = torch.tensor(v1_offset) + torch.tensor(v1_matrix) @ jerk_sol
print("cvxpylayers v_exec:", v1_sol.detach().numpy())
if scipy_result.v_exec is not None:
    print("max abs diff:", np.max(np.abs(v1_sol.detach().numpy() - scipy_result.v_exec)))

# --- Backward gradient sanity: finite-difference check ---
loss = v1_sol.sum()
loss.backward()
analytic_grad = v_nom_t.grad.detach().numpy().copy()
print("analytic dL/dv_nom:", analytic_grad)

eps = 1e-4
fd_grad = np.zeros(m)
for i in range(m):
    bump = np.zeros(m)
    bump[i] = eps
    v_nom_t_plus = torch.tensor(v_nom_np + bump, dtype=torch.float64)
    (jerk_plus,) = layer(v_nom_t_plus)
    v1_plus = (torch.tensor(v1_offset) + torch.tensor(v1_matrix) @ jerk_plus).sum().item()
    v_nom_t_minus = torch.tensor(v_nom_np - bump, dtype=torch.float64)
    (jerk_minus,) = layer(v_nom_t_minus)
    v1_minus = (torch.tensor(v1_offset) + torch.tensor(v1_matrix) @ jerk_minus).sum().item()
    fd_grad[i] = (v1_plus - v1_minus) / (2 * eps)

print("finite-diff dL/dv_nom:", fd_grad)
print("max abs grad diff:", np.max(np.abs(analytic_grad - fd_grad)))
