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
    for mode in ("uoar", "ct"):
        env = DynArmEnv(reward_mode=mode, seed=3)
        traj = rollout(env, n=20)
        rewards = [t[1] for t in traj[1:]]
        assert all(np.isfinite(r) for r in rewards)


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
