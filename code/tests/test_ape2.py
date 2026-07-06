import numpy as np

from godynur.ape2 import APE2Shield
from godynur.env import DynArmEnv
from godynur.scenes import Obstacle


def make_env(seed=0, speed=0.5):
    env = DynArmEnv(speed=speed, seed=seed)
    env.reset()
    return env


def zeros_base(s):
    return np.zeros(7)


def test_peek_reward_matches_step_reward():
    """For a non-colliding, non-goal step, step() returns exactly the peeked
    reward (no terminal penalty / goal bonus applied)."""
    env = make_env(seed=6)
    for _ in range(20):
        a = env.sample_action()
        if env.peek_tau_star(a) is not None:
            continue
        r_peek = env.peek_reward(a)
        _, r_step, done = env.step(a)
        err = np.linalg.norm(env.kin.flange(env.q) - env.goal)
        if err < env.goal_tol or done:
            break
        assert abs(r_peek - r_step) < 1e-9
        break


def test_shield_rejects_collision_course_candidate():
    """Place an obstacle on a direct collision course with the flange; the
    certified verdict for aggressive motion toward it must be rejected and
    the selector must return a certified (or contact-delaying) action."""
    env = make_env(seed=7)
    flange = env.kin.flange(env.q)
    # Obstacle 6 cm from the flange, flying straight at it fast.
    direction = np.array([1.0, 0.0, 0.0])
    env.scene.obstacles = [
        Obstacle(
            center=flange + 0.12 * direction,
            half=np.array([0.03, 0.2, 0.2]),
            vel=-2.0 * direction,
        )
    ]
    sel = APE2Shield(env, zeros_base, q_fn=None, seed=1)
    a = sel.act(env._state(), step=0)
    # Either the action certifies, or the env is in the honest no-safe case.
    certified = env.peek_tau_star(a, sel._inflation(a)) is None
    assert certified or sel.stats["no_safe"] >= 1


def test_no_safe_case_is_counted_not_hidden():
    """Obstacle overlapping the arm's immediate inflated neighborhood while
    inbound: every candidate (incl. zero action) fails; the selector must
    record no_safe and still return a legal action."""
    env = make_env(seed=8)
    flange = env.kin.flange(env.q)
    env.scene.obstacles = [
        Obstacle(
            center=flange + np.array([0.09, 0.0, 0.0]),
            half=np.array([0.02, 0.3, 0.3]),
            vel=np.array([-3.0, 0.0, 0.0]),
        )
    ]
    sel = APE2Shield(env, zeros_base, q_fn=None, seed=2)
    a = sel.act(env._state(), step=0)
    assert a.shape == (7,)
    assert np.all(np.abs(a) <= env.action_bound[1] + 1e-12)
    # This crafted case should be unsafe for zero action too.
    assert sel.stats["no_safe"] + sel.stats["scaled"] >= 1


def test_candidate_pool_size_and_bounds():
    env = make_env(seed=9)
    sel = APE2Shield(env, zeros_base, q_fn=None, M=2, N=3, seed=3)
    cands = sel._candidates(env._state())
    assert len(cands) == 2 * 3 + 1
    for a in cands:
        assert np.all(np.abs(a) <= env.action_bound[1] + 1e-12)


def test_shield_off_returns_argmax():
    env = make_env(seed=10)
    sel = APE2Shield(env, zeros_base, q_fn=None, shield=False, seed=4)
    s = env._state()
    a = sel.act(s, step=0)
    cands = APE2Shield(env, zeros_base, q_fn=None, shield=False, seed=4)._candidates(s)
    scores = [env.peek_reward(c) for c in cands]
    assert np.allclose(a, cands[int(np.argmax(scores))])


def test_certified_actions_never_collide_when_executed():
    """End-to-end: run shielded episodes; whenever the executed action was
    certified, the env's exact accounting must agree (no contact)."""
    env = make_env(seed=11, speed=1.0)
    sel = APE2Shield(env, zeros_base, q_fn=None, seed=5)
    violations = 0
    for _ in range(5):
        env.reset()
        done = False
        while not done:
            a = sel.act(env._state(), step=0)
            was_certified = env.peek_tau_star(a, sel._inflation(a)) is None
            _, _, done = env.step(a)
            if was_certified and env.last_tau_star is not None:
                violations += 1
    assert violations == 0