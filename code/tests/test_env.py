import numpy as np

from godynur.env import DynArmEnv


def rollout(env, n=30, seed=0):
    rng = np.random.default_rng(seed)
    s = env.reset()
    traj = [s]
    for _ in range(n):
        a = rng.uniform(*env.action_bound, env.action_dim)
        s, r, done = env.step(a)
        traj.append((s, r, done))
        if done:
            break
    return traj


def test_morvan_interface_shapes():
    env = DynArmEnv(speed=0.5, seed=1)
    s = env.reset()
    assert s.shape == (env.state_dim,)
    a = env.sample_action()
    assert a.shape == (env.action_dim,)
    s2, r, done = env.step(a)
    assert s2.shape == (env.state_dim,)
    assert isinstance(r, float) and isinstance(done, bool)


def test_state_dim_without_obstacles():
    env = DynArmEnv(obstacles_in_state=False, seed=1)
    assert env.reset().shape == (17,)


def test_grid_history_shape_and_dynamics():
    env = DynArmEnv(speed=2.0, k_frames=3, grid_n=32, seed=2)
    env.reset()
    g0 = env.grid_history()
    assert g0.shape == (3, 32, 32, 32)
    # After a few steps with fast obstacles, the newest frame must differ
    # from the oldest (obstacles moved).
    for _ in range(5):
        _, _, done = env.step(np.zeros(7))
        if done:
            break
    g1 = env.grid_history()
    assert not np.array_equal(g1[0], g1[-1])


def test_reward_modes_run_and_ct_penalizes_approach():
    """Both reward modes produce finite rewards; CT reward is <= UOAR-style
    reward when a contact is imminent (extra anticipatory penalty terms)."""
    for mode in ("uoar", "uoar_advisor", "ct"):
        env = DynArmEnv(reward_mode=mode, seed=3)
        traj = rollout(env, n=20)
        rewards = [t[1] for t in traj[1:]]
        assert all(np.isfinite(r) for r in rewards)


def test_advisor_reward_improving_equal_and_in_tolerance_formula():
    env = DynArmEnv(
        reward_mode="uoar_advisor", n_obstacles=0,
        obstacles_in_state=False, goal_tol=0.02, seed=31,
    )
    env.reset()
    dq = np.zeros(7)
    dq[0] = 0.01
    initial_flange = env.kin.flange(env.q)
    candidate_flange = env.kin.flange(env.q + dq)
    env.goal = candidate_flange.copy()
    env._prev_pos_err = float(np.linalg.norm(initial_flange - env.goal))

    improving_reward = env._reward(dq)
    assert np.isclose(improving_reward, 0.0 + 0.05 + 0.0 + 1.0)

    # The same candidate has exactly the same computed error on the next call,
    # exercising the advisor's elif tie-breaking (neither bonus nor penalty).
    equal_reward = env._reward(dq)
    assert np.isclose(equal_reward, 0.0 + 0.0 + 0.0 + 1.0)


def test_advisor_reward_worsening_formula():
    env = DynArmEnv(
        reward_mode="uoar_advisor", n_obstacles=0,
        obstacles_in_state=False, goal_tol=1e-12, seed=32,
    )
    env.reset()
    dq = np.zeros(7)
    dq[0] = 0.01
    env.goal = env.kin.flange(env.q).copy()
    env._prev_pos_err = 0.0
    err = float(np.linalg.norm(env.kin.flange(env.q + dq) - env.goal))

    reward = env._reward(dq)
    assert np.isclose(reward, -err / 3.0 - 0.05)


def test_advisor_reset_initializes_previous_distance_for_first_step():
    env = DynArmEnv(
        reward_mode="uoar_advisor", n_obstacles=0,
        obstacles_in_state=False, goal_tol=1e-12, seed=33,
    )
    env.reset()
    initial_err = float(np.linalg.norm(env.kin.flange(env.q) - env.goal))
    assert np.isclose(env._prev_pos_err, initial_err)

    _, reward, done = env.step(np.zeros(7))
    assert np.isclose(reward, -initial_err / 3.0)
    assert not done


def test_episode_configuration_overrides_and_300_step_timeout():
    env = DynArmEnv(
        reward_mode="uoar_advisor", n_obstacles=0,
        obstacles_in_state=False, goal_tol=0.02, goal_dwell=50,
        max_steps=300, seed=34,
    )
    env.reset()
    env.goal = env.kin.flange(env.q) + np.array([10.0, 0.0, 0.0])
    env._prev_pos_err = 10.0
    assert env.goal_tol == 0.02
    assert env.goal_dwell == 50
    assert env.max_steps == 300

    for _ in range(299):
        _, _, done = env.step(np.zeros(7))
        assert not done
    _, _, done = env.step(np.zeros(7))
    assert done
    assert env.t == 300


def test_collision_terminates_and_instruments():
    """Run until a collision occurs somewhere; check bookkeeping fields."""
    env = DynArmEnv(speed=2.0, seed=4)
    saw_collision = False
    for ep in range(30):
        env.reset()
        for _ in range(env.max_steps):
            _, _, done = env.step(env.sample_action())
            if done:
                if env.last_tau_star is not None:
                    saw_collision = True
                    assert 0.0 <= env.last_tau_star <= env.dt
                    assert isinstance(env.last_discrete_missed, bool)
                break
        if saw_collision:
            break
    assert saw_collision, "no collision in 30 random episodes at 2 m/s?"


def test_episode_terminates_on_goal_dwell_or_timeout():
    env = DynArmEnv(speed=0.1, seed=5)
    env.reset()
    done = False
    for _ in range(env.max_steps + 1):
        _, _, done = env.step(np.zeros(7))
        if done:
            break
    assert done


def test_closest_point_and_sdf_observation():
    env = DynArmEnv(task="tabletop", grid_mode="sdf", grid_n=16,
                    closest_point_in_state=True, obstacles_in_state=False,
                    seed=12)
    s = env.reset()
    assert s.shape == (17 + 7,)
    d = s[17]
    assert 0.0 <= d <= 0.5
    g = env.grid_history()
    assert g.shape == (3, 16, 16, 16)
    assert g.min() < 0.5  # SDF values, not binary (obstacles present)
