from time import perf_counter

import numpy as np
import pytest
import torch

from godynur.differentiable_qp import differentiable_v_exec
from godynur.panda import DDQ_MAX, DDDQ_MAX, DQ_MAX, Q_MAX, Q_MIN
from godynur.safety_qp import solve_safety_qp


H = 0.1
N_STEPS = 2
COLLOCATION_POINTS = 5
LAMBDA_J = 1e-3
M = 3
LIMITS = (
    np.full(M, -2.0),
    np.full(M, 2.0),
    np.full(M, -1.0),
    np.full(M, 1.0),
    np.full(M, -2.0),
    np.full(M, 2.0),
    np.full(M, -20.0),
    np.full(M, 20.0),
)

CASES = [
    (
        np.array([0.0, 0.2, -0.3]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.10, -0.08, 0.05]),
    ),
    (
        np.array([0.8, -0.5, 0.2]),
        np.array([0.15, -0.10, 0.05]),
        np.array([0.20, -0.30, 0.10]),
        np.array([0.30, -0.25, 0.10]),
    ),
    (
        np.array([-1.1, 1.2, 0.4]),
        np.array([-0.25, 0.20, -0.05]),
        np.array([0.35, -0.20, 0.15]),
        np.array([-0.45, 0.35, 0.02]),
    ),
    (
        np.array([1.5, -1.4, -0.7]),
        np.array([0.30, -0.20, 0.10]),
        np.array([-0.10, 0.25, -0.30]),
        np.array([0.55, -0.40, 0.30]),
    ),
]


def _diff_solve(q, v, a, v_nom):
    v_nom_tensor = torch.as_tensor(v_nom, dtype=torch.float64).reshape(1, M)
    return differentiable_v_exec(
        q[None],
        v[None],
        a[None],
        v_nom_tensor,
        H,
        N_STEPS,
        *LIMITS,
        lambda_j=LAMBDA_J,
        collocation_points=COLLOCATION_POINTS,
    )[0]


@pytest.mark.parametrize("q,v,a,v_nom", CASES)
def test_forward_matches_certified_box_only_scipy_solution(q, v, a, v_nom):
    scipy_result = solve_safety_qp(
        q,
        v,
        a,
        v_nom,
        H,
        N_STEPS,
        *LIMITS,
        lambda_j=LAMBDA_J,
        collocation_points=COLLOCATION_POINTS,
        require_terminal_stop=False,
    )
    assert scipy_result.certified

    differentiable_result = _diff_solve(q, v, a, v_nom)

    assert np.allclose(
        differentiable_result.detach().numpy(), scipy_result.v_exec, atol=1e-4
    )


@pytest.mark.parametrize("case_index", [0, 2])
def test_gradient_matches_centered_finite_difference(case_index):
    q, v, a, v_nom = CASES[case_index]
    coefficients = torch.tensor([0.7, -1.1, 0.4], dtype=torch.float64)
    v_nom_tensor = torch.tensor(
        v_nom[None], dtype=torch.float64, requires_grad=True
    )
    v_exec = differentiable_v_exec(
        q[None],
        v[None],
        a[None],
        v_nom_tensor,
        H,
        N_STEPS,
        *LIMITS,
        lambda_j=LAMBDA_J,
        collocation_points=COLLOCATION_POINTS,
    )
    (v_exec[0] * coefficients).sum().backward()
    analytic = v_nom_tensor.grad[0].detach().numpy()

    epsilon = 1e-4
    finite_difference = np.empty(M)
    for joint in range(M):
        bump = np.zeros(M)
        bump[joint] = epsilon
        plus = (_diff_solve(q, v, a, v_nom + bump) * coefficients).sum().item()
        minus = (_diff_solve(q, v, a, v_nom - bump) * coefficients).sum().item()
        finite_difference[joint] = (plus - minus) / (2.0 * epsilon)

    assert np.all(np.isfinite(analytic))
    assert np.allclose(analytic, finite_difference, atol=1e-3, rtol=1e-3)


def test_native_batch_with_different_states_matches_individual_solves():
    q_batch = np.stack([case[0] for case in CASES])
    v_batch = np.stack([case[1] for case in CASES])
    a_batch = np.stack([case[2] for case in CASES])
    v_nom_batch = torch.tensor(
        np.stack([case[3] for case in CASES]), dtype=torch.float64
    )

    batched = differentiable_v_exec(
        q_batch,
        v_batch,
        a_batch,
        v_nom_batch,
        H,
        N_STEPS,
        *LIMITS,
        lambda_j=LAMBDA_J,
        collocation_points=COLLOCATION_POINTS,
    )
    individual = torch.stack(
        [_diff_solve(q, v, a, v_nom) for q, v, a, v_nom in CASES]
    )

    assert torch.allclose(batched, individual, atol=1e-5, rtol=1e-5)


def test_batch_64_forward_and_backward_timing_smoke():
    batch_size = 64
    dyn_h = 0.05
    midpoint = (Q_MIN + Q_MAX) / 2.0
    offsets = np.linspace(-0.02, 0.02, batch_size)[:, None]
    joint_pattern = np.linspace(-1.0, 1.0, Q_MIN.size)[None, :]
    q_batch = midpoint + offsets * joint_pattern
    v_batch = offsets * 2.0 * joint_pattern
    a_batch = offsets * 5.0 * joint_pattern
    v_nom_batch = torch.tensor(
        v_batch + 0.1 * np.sin(np.arange(batch_size)[:, None] + np.arange(7)),
        dtype=torch.float32,
        requires_grad=True,
    )

    # Warm the structural cache so this guards repeated training-call cost,
    # not one-time CVXPY construction and DPP compilation.
    differentiable_v_exec(
        q_batch[:1],
        v_batch[:1],
        a_batch[:1],
        v_nom_batch[:1],
        dyn_h,
        N_STEPS,
        Q_MIN,
        Q_MAX,
        -DQ_MAX,
        DQ_MAX,
        -DDQ_MAX,
        DDQ_MAX,
        -DDDQ_MAX,
        DDDQ_MAX,
    )

    started = perf_counter()
    result = differentiable_v_exec(
        q_batch,
        v_batch,
        a_batch,
        v_nom_batch,
        dyn_h,
        N_STEPS,
        Q_MIN,
        Q_MAX,
        -DQ_MAX,
        DQ_MAX,
        -DDQ_MAX,
        DDQ_MAX,
        -DDDQ_MAX,
        DDDQ_MAX,
    )
    result.square().mean().backward()
    elapsed = perf_counter() - started

    assert result.shape == (batch_size, Q_MIN.size)
    assert v_nom_batch.grad is not None
    assert torch.all(torch.isfinite(v_nom_batch.grad))
    assert elapsed < 2.0


@pytest.mark.parametrize(
    "obstacles_in_state,closest_point_in_state",
    [(True, False), (False, True)],
)
def test_m6_projection_extracts_physical_state_from_actual_layout(
    monkeypatch, obstacles_in_state, closest_point_in_state
):
    import experiments.m6_kinodynamic_shield as m6
    from godynur.env import DynArmEnv

    env = DynArmEnv(
        task="tabletop",
        n_obstacles=3,
        reward_mode="uoar",
        obstacles_in_state=obstacles_in_state,
        closest_point_in_state=closest_point_in_state,
        action_mode="velocity",
        seed=17,
    )
    env.reset()
    env.v = np.linspace(-0.2, 0.2, env.action_dim)
    env.a = np.linspace(0.3, -0.3, env.action_dim)
    state = torch.from_numpy(env._state()[None])
    raw_action = torch.from_numpy(
        np.linspace(-0.5, 0.5, env.action_dim, dtype=np.float32)[None]
    ).requires_grad_()
    captured = {}

    def fake_differentiable_v_exec(q, v, a, v_nom, *args, **kwargs):
        captured.update(q=q.copy(), v=v.copy(), a=a.copy())
        return v_nom

    monkeypatch.setattr(m6, "differentiable_v_exec", fake_differentiable_v_exec)
    projected = m6.build_diffqp_projection(env)(raw_action, state)

    assert np.allclose(captured["q"][0], env.q, atol=2e-7)
    assert np.allclose(captured["v"][0], env.v, atol=2e-7)
    assert np.allclose(captured["a"][0], env.a, atol=2e-7)
    assert torch.allclose(projected, raw_action, atol=2e-6)
