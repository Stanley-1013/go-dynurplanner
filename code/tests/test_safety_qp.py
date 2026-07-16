from time import perf_counter

import numpy as np

import godynur.safety_qp as safety_qp
from godynur.kinodynamics import discrete_update, interval_within_limits
from godynur.safety_qp import solve_safety_qp


def _replay_is_safe(result, q0, v0, a0, h, limits):
    if result.jerk_sequence is None:
        return True
    q_min, q_max, v_min, v_max, a_min, a_max, j_min, j_max = limits
    for joint in range(q0.size):
        state = np.array([q0[joint], v0[joint], a0[joint]], dtype=float)
        joint_limits = tuple(
            bound[joint]
            for bound in (q_min, q_max, v_min, v_max, a_min, a_max, j_min, j_max)
        )
        for jerk in result.jerk_sequence[joint]:
            if not interval_within_limits(*state, jerk, h, *joint_limits):
                return False
            state = discrete_update(state, jerk, h)
    return True


def test_feasible_multi_joint_solution_tracks_nominal_first_velocity():
    # Arrange
    m = 3
    q0 = np.array([0.0, 0.2, -0.3])
    v0 = np.zeros(m)
    a0 = np.zeros(m)
    v_nom = np.array([0.05, -0.04, 0.025])
    limits = (
        np.full(m, -2.0),
        np.full(m, 2.0),
        np.full(m, -2.0),
        np.full(m, 2.0),
        np.full(m, -3.0),
        np.full(m, 3.0),
        np.full(m, -20.0),
        np.full(m, 20.0),
    )

    # Act
    result = solve_safety_qp(
        q0, v0, a0, v_nom, 0.2, 5, *limits, lambda_j=1e-8
    )

    # Assert
    assert result.feasible
    assert result.certified
    assert result.jerk_sequence.shape == (m, 5)
    assert np.allclose(result.v_exec, v_nom, atol=5e-4)
    assert _replay_is_safe(result, q0, v0, a0, 0.2, limits)


def test_infeasible_near_position_limit_never_returns_unsafe_sequence():
    # Arrange: joint 0 cannot avoid crossing q_max during its first interval.
    q0 = np.array([0.99, 0.0])
    v0 = np.array([0.5, 0.0])
    a0 = np.zeros(2)
    v_nom = np.array([0.6, 0.1])
    limits = (
        np.array([-2.0, -2.0]),
        np.array([1.0, 2.0]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        np.array([-2.0, -2.0]),
        np.array([2.0, 2.0]),
        np.array([-20.0, -20.0]),
        np.array([20.0, 20.0]),
    )

    # Act
    result = solve_safety_qp(q0, v0, a0, v_nom, 0.1, 20, *limits)

    # Assert
    assert not result.feasible
    assert not result.certified
    assert _replay_is_safe(result, q0, v0, a0, 0.1, limits)


def test_seven_joint_solve_time_smoke_budget():
    # Arrange
    m = 7
    zeros = np.zeros(m)
    limits = (
        np.full(m, -2.0),
        np.full(m, 2.0),
        np.full(m, -2.0),
        np.full(m, 2.0),
        np.full(m, -3.0),
        np.full(m, 3.0),
        np.full(m, -20.0),
        np.full(m, 20.0),
    )

    # Act
    started = perf_counter()
    result = solve_safety_qp(
        zeros,
        zeros,
        zeros,
        np.linspace(-0.04, 0.04, m),
        0.1,
        6,
        *limits,
        lambda_j=1e-7,
    )
    elapsed = perf_counter() - started

    # Assert: smoke guard only; Phase 5 owns the real benchmark.
    assert result.feasible
    assert elapsed < 2.0
    assert result.solve_time_s <= elapsed


def test_failed_exact_certification_retries_with_tightened_margin(monkeypatch):
    # Arrange: force the first exact certification to model a raw collocation
    # miss, then require a second solve whose real candidate certifies.
    real_interval_check = safety_qp.interval_within_limits
    real_minimize = safety_qp.minimize
    exact_check_calls = 0
    solve_calls = 0
    box_lower_bounds = []

    def reject_first_exact_check(*args, **kwargs):
        nonlocal exact_check_calls
        exact_check_calls += 1
        if exact_check_calls == 1:
            return False
        return real_interval_check(*args, **kwargs)

    def count_solves(*args, **kwargs):
        nonlocal solve_calls
        solve_calls += 1
        box_lower_bounds.append(kwargs["constraints"][0].lb.copy())
        return real_minimize(*args, **kwargs)

    monkeypatch.setattr(safety_qp, "interval_within_limits", reject_first_exact_check)
    monkeypatch.setattr(safety_qp, "minimize", count_solves)
    q0 = np.array([0.9])
    zeros = np.zeros(1)
    limits = tuple(
        np.array([value])
        for value in (-1.0, 1.0, -1.0, 1.0, -2.0, 2.0, -20.0, 20.0)
    )

    # Act
    result = solve_safety_qp(
        q0,
        zeros,
        zeros,
        np.array([0.02]),
        0.1,
        5,
        *limits,
        lambda_j=1e-7,
        margin_shrink_frac=0.02,
        max_retries=2,
    )

    # Assert
    assert result.feasible and result.certified
    assert solve_calls == 2
    assert np.any(box_lower_bounds[1] > box_lower_bounds[0])
    assert exact_check_calls == 1 + result.jerk_sequence.shape[1]
    assert _replay_is_safe(result, q0, zeros, zeros, 0.1, limits)


def test_box_only_horizon_is_feasible_when_terminal_stop_is_not():
    # Arrange: one jerk interval cannot make both nonzero velocity and
    # acceleration exactly zero, but the current motion is safely in bounds.
    q0 = np.array([0.0])
    v0 = np.array([0.5])
    a0 = np.array([0.0])
    v_nom = np.array([0.6])
    limits = tuple(
        np.array([value])
        for value in (-2.0, 2.0, -2.0, 2.0, -2.0, 2.0, -20.0, 20.0)
    )

    # Act
    stopped = solve_safety_qp(q0, v0, a0, v_nom, 0.1, 1, *limits)
    box_only = solve_safety_qp(
        q0,
        v0,
        a0,
        v_nom,
        0.1,
        1,
        *limits,
        require_terminal_stop=False,
    )

    # Assert
    assert not stopped.feasible
    assert not stopped.certified
    assert box_only.feasible
    assert box_only.certified
    assert _replay_is_safe(box_only, q0, v0, a0, 0.1, limits)
