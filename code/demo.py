"""One-minute demo of DynArmEnv for the advisor.

Run:  python3 demo.py

Shows: (1) the Morvan-style interface on a dynamic-obstacle episode;
(2) the continuous-time exact collision accounting catching contacts the
standard endpoint check would miss; (3) the uoar/ct reward-mode switch;
(4) the occupancy-grid observation for the encoder.
"""

import numpy as np

from godynur.env import DynArmEnv

print("=" * 64)
print("DynArmEnv demo — Morvan interface, upgraded internals")
print("=" * 64)

# ---- 1. Interface + collision instrumentation at two speed tiers --------
for speed in (0.5, 2.0):
    env = DynArmEnv(speed=speed, reward_mode="ct", seed=7)
    n_ep, n_coll, n_missed = 30, 0, 0
    for _ in range(n_ep):
        s = env.reset()  # Morvan: reset() -> s
        done = False
        while not done:
            a = env.sample_action()
            s, r, done = env.step(a)  # Morvan: step(a) -> (s, r, done)
        if env.last_tau_star is not None:
            n_coll += 1
            n_missed += int(env.last_discrete_missed)
    print(
        f"speed {speed} m/s | {n_ep} random episodes | "
        f"true collisions: {n_coll} | of those, endpoint check would MISS: "
        f"{n_missed}"
    )

# ---- 2. Reward-mode switch on the identical trajectory -------------------
# Hold the arm still while an obstacle approaches: the discrete UOAR stays
# flat until contact, the CT reward starts penalizing BEFORE contact
# (swept-overlap + continuous-TTC terms) — the anticipatory gradient.
print("-" * 64)
print("same scene, arm holding still, obstacle approaching — reward per step:")
for seed in range(11, 60):
    env_u = DynArmEnv(speed=1.0, reward_mode="uoar", seed=seed)
    env_c = DynArmEnv(speed=1.0, reward_mode="ct", seed=seed)
    s_u, s_c = env_u.reset(), env_c.reset()
    assert np.allclose(s_u, s_c), "same seed => same scene"
    a = np.zeros(7)
    rows, diverged = [], False
    for t in range(env_u.max_steps):
        _, r_u, d_u = env_u.step(a)
        _, r_c, d_c = env_c.step(a)
        rows.append((t, r_u, r_c, d_c))
        if r_c < r_u - 1e-6:
            diverged = True
        if d_u or d_c:
            break
    if diverged:
        first = next(i for i, (_, ru, rc, _) in enumerate(rows) if rc < ru - 1e-6)
        for t, r_u, r_c, done in rows[max(0, first - 2): first + 4]:
            tag = "  <- CT anticipatory penalty" if r_c < r_u - 1e-6 else ""
            end = "  [episode ends: collision]" if done else ""
            print(f"  t={t:>3}: r_uoar={r_u:8.4f}   r_ct={r_c:8.4f}{tag}{end}")
        break
else:
    print("  (no approach event found in scanned seeds)")

# ---- 3. Occupancy-grid observation ---------------------------------------
print("-" * 64)
g = env_c.grid_history()
print(
    f"grid_history(): shape {g.shape} (k frames x 32^3), "
    f"occupied voxels/frame: {[int(f.sum()) for f in g]}"
)
print("analytic rasterization: no simulator anywhere in this demo.")
