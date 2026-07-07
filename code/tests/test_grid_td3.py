import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skipped on numpy-only CI

from godynur.env import DynArmEnv  # noqa: E402
from godynur.grid_td3 import (  # noqa: E402
    GridReplayBuffer,
    GridTD3,
    params_to_grid,
    rasterize_batch,
)


def test_params_to_grid_matches_env_rasterization():
    env = DynArmEnv(task="tabletop", n_obstacles=3, seed=0)
    env.reset()
    from_params = params_to_grid(env.scene_params(3))
    from_env = env.grid_history()[-1]
    assert np.array_equal(from_params, from_env)


def test_dummy_slots_rasterize_to_nothing():
    params = np.full((3, 9), 50.0, dtype=np.float32)
    params[:, 3:6] = 0.01
    assert params_to_grid(params).sum() == 0.0


def test_buffer_roundtrip_and_mask():
    buf = GridReplayBuffer(vec_dim=17, action_dim=7, k_frames=3, n_slots=3,
                           capacity=64)
    h = np.zeros((3, 3, 9), np.float32)
    for i in range(10):
        buf.add(np.zeros(17), np.zeros(7), 0.1, np.zeros(17), 0.0,
                h, h, fut=h[0] if i % 2 == 0 else None, fut_valid=(i % 2 == 0))
    assert len(buf) == 10
    vec, a, r, vec2, d, hist, hist2, fut, fmask = buf.sample(8)
    assert hist.shape == (8, 3, 3, 9) and fut.shape == (8, 1, 3, 9)
    assert set(np.unique(fmask)).issubset({0.0, 1.0})


def test_act_and_learn_smoke():
    env = DynArmEnv(task="tabletop", n_obstacles=3, obstacles_in_state=False,
                    seed=1)
    vec = env.reset()
    agent = GridTD3(vec_dim=env.state_dim, action_dim=7,
                    action_scale=env.action_bound[1], device="cpu", seed=0)
    h = np.stack([env.scene_params(3)] * 3)
    a = agent.act(vec, h)
    assert a.shape == (7,) and np.all(np.abs(a) <= env.action_bound[1] + 1e-9)
    # Fill buffer past the learn threshold and take gradient steps.
    for _ in range(64 * 5 + 1):
        agent.buffer.add(vec, a, 0.0, vec, 0.0, h, h, fut=h[0], fut_valid=True)
    agent.learn(batch=64)
    agent.learn(batch=64)
    assert np.isfinite(agent.last_aux_loss)


def test_rasterize_batch_shape():
    params = np.full((2, 3, 3, 9), 50.0, dtype=np.float32)
    params[..., 3:6] = 0.01
    g = rasterize_batch(params)
    assert g.shape == (2, 3, 32, 32, 32) and g.sum() == 0.0
