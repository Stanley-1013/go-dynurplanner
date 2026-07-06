"""M1 — Tunneling-rate experiment (the paper's Figure-1).

Protocol
--------
A greedy goal-reaching policy that uses the STANDARD discrete per-timestep
collision check (endpoint configuration vs obstacle positions, exactly what
DRL training loops do) runs in dynamic scenes across obstacle speed tiers.
Every executed step is audited two ways:

  1. Continuous ground truth: exact first-contact time tau* over the control
     interval (godynur.continuous, breakpoint method — no sampling).
  2. S-substep discrete checks, S in {1, 2, 4, 8, 16}: overlap tests at
     tau = j*dt/S with the robot linearly interpolated in joint space and
     obstacles advanced by v*tau. S=1 is the standard protocol.

A step "tunnels at S" if the continuous audit finds contact but the
S-substep check reports clear. Reported per speed tier:
  - step-level miss rate  : tunneled steps / steps with continuous contact
  - absolute step rate    : tunneled steps / all executed steps
  - episode-level rate    : episodes with >=1 missed contact / all episodes
  - success rate (context): goal reached within the step budget

Usage:  python3 experiments/m1_tunneling.py [--episodes 40] [--out results/m1]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godynur import MovingSegment, first_contact_time  # noqa: E402
from godynur.geometry import AABB, segment_box_overlap  # noqa: E402
from godynur.panda import PandaKinematics, Q_MAX, Q_MIN  # noqa: E402
from godynur.policy import GreedyDiscretePolicy, config_collides  # noqa: E402
from godynur.scenes import sample_scene  # noqa: E402

DT = 0.05  # control interval (s), 20 Hz decision rate
A_R = 0.06  # link cylinder radius folded into obstacle expansion (m)
MAX_STEPS = 100
N_OBSTACLES = 3
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0]
SUBSTEPS = [1, 2, 4, 8, 16]


def moving_segments(kin, q, dq, dt):
    s0 = kin.segments(q)
    s1 = kin.segments(q + dq)
    out = []
    for a, b in zip(s0, s1):
        out.append(
            MovingSegment(
                a0=a[0], ua=(b[0] - a[0]) / dt, b0=a[1], ub=(b[1] - a[1]) / dt
            )
        )
    return out


def continuous_contact(kin, q, dq, mboxes, dt):
    segs = moving_segments(kin, q, dq, dt)
    taus = [
        t
        for seg in segs
        for mb in mboxes
        if (t := first_contact_time(seg, mb, dt)) is not None
    ]
    return min(taus) if taus else None


def substep_caught(kin, q, dq, obstacles, dt, s):
    """Discrete check at tau = j*dt/s, j=1..s (j=s is the endpoint check)."""
    for j in range(1, s + 1):
        tau = j * dt / s
        q_j = q + dq * (tau / dt)
        boxes_j = [
            AABB(
                ob.center + ob.vel * tau - ob.half - A_R,
                ob.center + ob.vel * tau + ob.half + A_R,
            )
            for ob in obstacles
        ]
        for seg in kin.segments(q_j):
            for box in boxes_j:
                l, _, _ = segment_box_overlap(seg[0], seg[1], box)
                if l > 0.0:
                    return True
    return False


def run_episode(kin, policy, rng, speed):
    # Start configuration and goal (both sampled reachable).
    for _ in range(200):
        q = Q_MIN + rng.random(7) * (Q_MAX - Q_MIN)
        if kin.flange(q)[2] > 0.15:
            break
    goal = kin.flange(Q_MIN + rng.random(7) * (Q_MAX - Q_MIN))

    # Scene must not start in contact with the arm (spawn clearance 8 cm).
    for _ in range(100):
        scene = sample_scene(rng, N_OBSTACLES, speed)
        if not config_collides(kin, q, scene.static_aabbs(margin=A_R + 0.08)):
            break

    stats = {
        "steps": 0,
        "contact_steps": 0,
        "missed": {s: 0 for s in SUBSTEPS},
        "success": False,
    }
    for _ in range(MAX_STEPS):
        boxes_next = [
            AABB(
                ob.center + ob.vel * DT - ob.half - A_R,
                ob.center + ob.vel * DT + ob.half + A_R,
            )
            for ob in scene.obstacles
        ]
        dq = policy.act(q, goal, boxes_next)
        if dq is None:
            dq = np.zeros(7)  # hold; obstacles keep moving

        tau_star = continuous_contact(
            kin, q, dq, scene.moving_boxes(margin=A_R), DT
        )
        stats["steps"] += 1
        if tau_star is not None:
            # True (continuous-time) collision: record which discrete
            # protocols catch it, then terminate — collision is terminal.
            stats["contact_steps"] += 1
            for s in SUBSTEPS:
                if not substep_caught(kin, q, dq, scene.obstacles, DT, s):
                    stats["missed"][s] += 1
            break

        q = np.clip(q + dq, Q_MIN, Q_MAX)
        scene.step(DT)
        if np.linalg.norm(kin.flange(q) - goal) < 0.05:
            stats["success"] = True
            break
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", type=str, default="experiments/results/m1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    kin = PandaKinematics()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    t_start = time.time()
    for speed in SPEEDS:
        rng = np.random.default_rng(args.seed + int(speed * 1000))
        policy = GreedyDiscretePolicy(kin, seed=args.seed)
        agg = {
            "episodes": 0,
            "successes": 0,
            "steps": 0,
            "contact_steps": 0,
            "missed": {s: 0 for s in SUBSTEPS},
            "episodes_with_miss": {s: 0 for s in SUBSTEPS},
        }
        for _ in range(args.episodes):
            st = run_episode(kin, policy, rng, speed)
            agg["episodes"] += 1
            agg["successes"] += int(st["success"])
            agg["steps"] += st["steps"]
            agg["contact_steps"] += st["contact_steps"]
            for s in SUBSTEPS:
                agg["missed"][s] += st["missed"][s]
                agg["episodes_with_miss"][s] += int(st["missed"][s] > 0)
        results[speed] = agg
        m1 = agg["missed"][1]
        print(
            f"speed {speed:>4} m/s | steps {agg['steps']:>5} | contact "
            f"{agg['contact_steps']:>4} | missed@S=1 {m1:>4} | "
            f"success {agg['successes']}/{agg['episodes']} | "
            f"{time.time() - t_start:6.1f}s",
            flush=True,
        )

    with open(out_dir / "m1_results.json", "w") as f:
        json.dump(
            {
                "config": {
                    "dt": DT, "a_r": A_R, "max_steps": MAX_STEPS,
                    "n_obstacles": N_OBSTACLES, "episodes": args.episodes,
                    "speeds": SPEEDS, "substeps": SUBSTEPS, "seed": args.seed,
                },
                "results": {str(k): v for k, v in results.items()},
            },
            f,
            indent=2,
        )

    # ---- Figure 1 ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(SUBSTEPS)))
    for s, c in zip(SUBSTEPS, colors):
        miss_rate = [
            results[v]["missed"][s] / max(1, results[v]["contact_steps"])
            for v in SPEEDS
        ]
        axes[0].plot(SPEEDS, miss_rate, "o-", color=c, label=f"S={s} substeps")
    axes[0].set_xlabel("obstacle speed (m/s)")
    axes[0].set_ylabel("missed contacts / true contacts")
    axes[0].set_title("(a) Discrete checking misses true contacts\n(tunneling), by subsampling depth")
    axes[0].set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    for s, c in zip(SUBSTEPS, colors):
        ep_rate = [
            results[v]["episodes_with_miss"][s] / results[v]["episodes"]
            for v in SPEEDS
        ]
        axes[1].plot(SPEEDS, ep_rate, "o-", color=c, label=f"S={s}")
    succ = [results[v]["successes"] / results[v]["episodes"] for v in SPEEDS]
    axes[1].plot(SPEEDS, succ, "k--", alpha=0.5, label="success rate (context)")
    axes[1].set_xlabel("obstacle speed (m/s)")
    axes[1].set_ylabel("fraction of episodes")
    axes[1].set_title("(b) Episodes containing >=1 undetected contact")
    axes[1].set_xscale("log")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.suptitle(
        "Continuous-time audit of the standard discrete collision-check protocol "
        f"(greedy policy, {N_OBSTACLES} obstacles, dt={DT}s)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_tunneling.png", dpi=160)
    print(f"wrote {out_dir}/m1_results.json and figure1_tunneling.png")


if __name__ == "__main__":
    main()
