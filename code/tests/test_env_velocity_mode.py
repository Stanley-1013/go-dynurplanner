import numpy as np
import pytest

from godynur.env import DynArmEnv
from godynur.kinodynamics import cubic_linearization_bound
from godynur.panda import DDQ_MAX, DQ_MAX, Q_MAX, Q_MIN


@pytest.mark.parametrize(
    "obstacles_in_state,closest_point_in_state",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_velocity_reset_and_state_dim_add_exactly_nine_fields(
    obstacles_in_state, closest_point_in_state
):
    common = {
        "n_obstacles": 0,
        "obstacles_in_state": obstacles_in_state,
        "closest_point_in_state": closest_point_in_state,
        "seed": 10,
    }
    delta_env = DynArmEnv(**common)
    velocity_env = DynArmEnv(action_mode="velocity", **common)

    state = velocity_env.reset()

    assert np.array_equal(velocity_env.v, np.zeros(7))
    assert np.array_equal(velocity_env.a, np.zeros(7))
    assert velocity_env.state_dim == delta_env.state_dim + 9
    assert state.shape == (velocity_env.state_dim,)
    assert np.array_equal(state[-9:], np.zeros(9, dtype=np.float32))


def test_feasible_velocity_step_uses_normal_certified_path():
    env = DynArmEnv(
        action_mode="velocity",
        n_obstacles=0,
        obstacles_in_state=False,
        seed=11,
    )
    env.reset()
    env.q = (Q_MIN + Q_MAX) / 2.0
    v_nom = env.v + env.dv_scale * 0.1

    state, reward, done = env.step(np.full(7, 0.1))

    assert env.stats.get("shield_fallback", 0) == 0
    assert env.stats.get("shield_emergency", 0) == 0
    assert env._last_terminal_membership
    assert np.all(env.v > 0.0)
    assert np.linalg.norm(env.v - v_nom) < np.linalg.norm(v_nom)
    assert np.allclose(state[-9:-2], env.a / DDQ_MAX)
    assert state[-2] == 1.0
    assert np.isclose(state[-1], env._last_intervention_norm)
    assert np.isfinite(reward)
    assert isinstance(done, bool)


def test_near_limit_velocity_request_executes_fresh_braking_fallback():
    env = DynArmEnv(
        action_mode="velocity",
        n_obstacles=0,
        obstacles_in_state=False,
        seed=12,
    )
    env.reset()
    env.q = (Q_MIN + Q_MAX) / 2.0
    env.q[0] = Q_MAX[0] - 0.01
    env.v = np.zeros(7)
    env.v[0] = 0.05 * DQ_MAX[0]
    env.a = np.zeros(7)

    state, _, _ = env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert env.stats["shield_fallback"] == 1
    assert env.stats.get("shield_emergency", 0) == 0
    assert not env._last_terminal_membership
    assert state[-2] == 0.0
    assert np.all(env.q >= Q_MIN)
    assert np.all(env.q <= Q_MAX)


def test_numerical_edge_state_uses_nonthrowing_emergency_path():
    env = DynArmEnv(
        action_mode="velocity",
        n_obstacles=0,
        obstacles_in_state=False,
        seed=15,
    )
    env.reset()
    env.q = (Q_MIN + Q_MAX) / 2.0
    env.q[0] = Q_MAX[0] - 0.005
    env.v = np.zeros(7)
    env.v[0] = 0.05 * DQ_MAX[0]
    env.a = np.zeros(7)

    env.step(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    assert env.stats["shield_emergency"] == 1
    assert env.stats.get("shield_fallback", 0) == 0
    assert np.all(env.q >= Q_MIN)
    assert np.all(env.q <= Q_MAX)
    assert np.all(env.v >= -DQ_MAX)
    assert np.all(env.v <= DQ_MAX)


def test_velocity_reward_subtracts_intervention_penalty_only():
    common = {
        "action_mode": "velocity",
        "n_obstacles": 0,
        "obstacles_in_state": False,
        "seed": 13,
    }
    unpenalized = DynArmEnv(lambda_intervention=0.0, **common)
    penalized = DynArmEnv(lambda_intervention=0.1, **common)
    unpenalized.reset()
    penalized.reset()
    midpoint = (Q_MIN + Q_MAX) / 2.0
    unpenalized.q = midpoint.copy()
    penalized.q = midpoint.copy()
    action = np.full(7, 0.5)

    _, reward_without_penalty, _ = unpenalized.step(action)
    _, reward_with_penalty, _ = penalized.step(action)

    expected_penalty = 0.1 * penalized._last_intervention_norm**2
    assert np.isclose(reward_without_penalty - reward_with_penalty, expected_penalty)


def test_default_and_explicit_delta_q_modes_remain_identical():
    default_env = DynArmEnv(n_obstacles=0, obstacles_in_state=False, seed=14)
    explicit_env = DynArmEnv(
        action_mode="delta_q",
        n_obstacles=0,
        obstacles_in_state=False,
        seed=14,
    )
    default_state = default_env.reset()
    explicit_state = explicit_env.reset()
    action = np.linspace(-0.05, 0.05, 7)

    default_result = default_env.step(action)
    explicit_result = explicit_env.step(action)

    assert np.array_equal(default_state, explicit_state)
    assert np.array_equal(default_result[0], explicit_result[0])
    assert default_result[1:] == explicit_result[1:]
    assert "shield_fallback" not in default_env.stats
    assert "shield_emergency" not in default_env.stats
    assert "shield_fallback" not in explicit_env.stats
    assert "shield_emergency" not in explicit_env.stats
    assert not hasattr(default_env, "v")
    assert not hasattr(default_env, "a")


def test_velocity_collision_check_receives_cubic_linearization_inflation():
    env = DynArmEnv(
        action_mode="velocity",
        n_obstacles=0,
        obstacles_in_state=False,
        seed=16,
    )
    env.reset()
    env.q = (Q_MIN + Q_MAX) / 2.0
    q0, v0, a0 = env.q.copy(), env.v.copy(), env.a.copy()
    seen = {}

    def capture_first_contact(dq, inflation=0.0):
        seen["dq"] = np.asarray(dq).copy()
        seen["inflation"] = inflation
        return None

    env._first_contact = capture_first_contact
    env.step(np.full(7, 0.1))
    executed_jerk = (env.a - a0) / env.dt
    expected = cubic_linearization_bound(
        q0, v0, a0, executed_jerk, env.dt, env.kin
    )

    assert expected > 0.0
    assert seen["inflation"] == pytest.approx(expected)
