"""M1-hardened — paper-grade tunneling-rate experiment.

Extends m1_tunneling.py (kept frozen for provenance) with:
  - multiple seeds (default 3) x 100 episodes per speed tier
  - per-contact attribution: which obstacle made first contact, and whether
    it is 'thin' (min half-extent <= 2 cm) -> thickness stratification
  - Wilson 95% confidence intervals on all miss rates
  - wall-clock timing: exact continuous audit vs S-substep discrete checks

Usage: python3 experiments/m1_hardened.py [--episodes 100] [--seeds 3]
Output: experiments/results/m1_hardened/{m1h_results.json, figure1_hardened.png}
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

DT = 0.05
A_R = 0.06
MAX_STEPS = 100
N_OBSTACLES = 3
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0]
SUBSTEPS = [1, 2, 4, 8, 16]
THIN_THRESHOLD = 0.02  # min half-extent (m) separating thin plates from boxes


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval: returns (point, lo, hi)."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def moving_segments(kin, q, dq, dt):
    s0, s1 = kin.segments(q), kin.segments(q + dq)
    return [
        MovingSegment(a0=a[0], ua=(b[0] - a[0]) / dt, b0=a[1], ub=(b[1] - a[1]) / dt)
        for a, b in zip(s0, s1)
    ]


def continuous_audit(kin, q, dq, scene, dt):
    """Exact audit. Returns (tau*, index of first-contact obstacle) or (None, None)."""
    segs = moving_segments(kin, q, dq, dt)
    best_tau, best_ob = None, None
    for ob_idx, ob in enumerate(scene.obstacles):
        mb = ob.moving_box(margin=A_R)
        for seg in segs:
            t = first_contact_time(seg, mb, dt)
            if t is not None and (best_tau is None or t < best_tau):
                best_tau, best_ob = t, ob_idx
    return best_tau, best_ob


def substep_caught(kin, q, dq, obstacles, dt, s):
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


def run_episode(kin, policy, rng, speed, timing):
    for _ in range(200):
        q = Q_MIN + rng.random(7) * (Q_MAX - Q_MIN)
        if kin.flange(q)[2] > 0.15:
            break
    goal = kin.flange(Q_MIN + rng.random(7) * (Q_MAX - Q_MIN))
    for _ in range(100):
        scene = sample_scene(rng, N_OBSTACLES, speed)
        if not config_collides(kin, q, scene.static_aabbs(margin=A_R + 0.08)):
            break

    steps, contact = 0, None
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
            dq = np.zeros(7)

        t0 = time.perf_counter()
        tau_star, ob_idx = continuous_audit(kin, q, dq, scene, DT)
        timing["continuous"] += time.perf_counter() - t0
        timing["n_steps"] += 1
        steps += 1

        if tau_star is not None:
            ob = scene.obstacles[ob_idx]
            caught = {}
            for s in SUBSTEPS:
                t0 = time.perf_counter()
                caught[s] = substep_caught(kin, q, dq, scene.obstacles, DT, s)
                timing[f"S{s}"] += time.perf_counter() - t0
                timing[f"nS{s}"] += 1
            # Severity of the contact over the remainder of the interval:
            # max instantaneous overlap length (m of link inside obstacle)
            # and dwell fraction (share of the interval in contact). Fine
            # sampling is fine here — existence is already certified by the
            # exact checker; we are only measuring magnitude.
            segs = moving_segments(kin, q, dq, DT)
            l_max, dwell_hits, n_samp = 0.0, 0, 60
            for tau in np.linspace(0.0, DT, n_samp):
                l_tau = 0.0
                for ob2 in scene.obstacles:
                    box_t = AABB(
                        ob2.center + ob2.vel * tau - ob2.half - A_R,
                        ob2.center + ob2.vel * tau + ob2.half + A_R,
                    )
                    for seg in segs:
                        a = seg.a0 + seg.ua * tau
                        b = seg.b0 + seg.ub * tau
                        l, _, _ = segment_box_overlap(a, b, box_t)
                        l_tau += l
                if l_tau > 0:
                    dwell_hits += 1
                l_max = max(l_max, l_tau)
            contact = {
                "tau": float(tau_star),
                "thin": bool(np.min(ob.half) <= THIN_THRESHOLD),
                "caught": {str(s): bool(caught[s]) for s in SUBSTEPS},
                "l_max": float(l_max),
                "dwell": float(dwell_hits / n_samp),
            }
            break  # collision is terminal

        # Timing of substep checks on non-contact steps too (realistic cost).
        for s in SUBSTEPS:
            t0 = time.perf_counter()
            substep_caught(kin, q, dq, scene.obstacles, DT, s)
            timing[f"S{s}"] += time.perf_counter() - t0
            timing[f"nS{s}"] += 1

        q = np.clip(q + dq, Q_MIN, Q_MAX)
        scene.step(DT)
        if np.linalg.norm(kin.flange(q) - goal) < 0.05:
            return steps, contact, True
    return steps, contact, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=str, default="experiments/results/m1_hardened")
    args = ap.parse_args()

    kin = PandaKinematics()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    contacts = []  # flat list of contact events with full attribution
    per_tier = {v: {"episodes": 0, "successes": 0, "collisions": 0} for v in SPEEDS}
    timing = {"continuous": 0.0, "n_steps": 0}
    for s in SUBSTEPS:
        timing[f"S{s}"] = 0.0
        timing[f"nS{s}"] = 0

    t_start = time.time()
    for seed in range(args.seeds):
        for speed in SPEEDS:
            rng = np.random.default_rng(1000 * seed + int(speed * 1000))
            policy = GreedyDiscretePolicy(kin, seed=seed)
            for _ in range(args.episodes):
                _, contact, success = run_episode(kin, policy, rng, speed, timing)
                per_tier[speed]["episodes"] += 1
                per_tier[speed]["successes"] += int(success)
                if contact is not None:
                    per_tier[speed]["collisions"] += 1
                    contact.update({"speed": speed, "seed": seed})
                    contacts.append(contact)
        print(f"seed {seed} done at {time.time() - t_start:.0f}s", flush=True)

    # ---- aggregate ----
    def miss_stats(events, s):
        n = len(events)
        k = sum(1 for e in events if not e["caught"][str(s)])
        return {"missed": k, "n": n, "wilson": wilson(k, n)}

    agg = {"per_speed": {}, "per_thickness": {}, "pooled": {}}
    for v in SPEEDS:
        ev = [e for e in contacts if e["speed"] == v]
        agg["per_speed"][str(v)] = {
            "collisions": len(ev),
            "episodes": per_tier[v]["episodes"],
            "successes": per_tier[v]["successes"],
            "miss": {str(s): miss_stats(ev, s) for s in SUBSTEPS},
        }
    for label, pred in [
        ("thin", lambda e: e["thin"]),
        ("compact", lambda e: not e["thin"]),
    ]:
        ev = [e for e in contacts if pred(e)]
        agg["per_thickness"][label] = {
            "collisions": len(ev),
            "miss": {str(s): miss_stats(ev, s) for s in SUBSTEPS},
        }
    agg["pooled"] = {str(s): miss_stats(contacts, s) for s in SUBSTEPS}

    # Severity comparison: are the contacts MISSED by the standard check
    # milder than the caught ones? (Reviewer question: "do the missed
    # contacts matter?") Report distributional stats of max overlap length
    # and dwell fraction for missed-at-S=1 vs caught-at-S=1 contacts.
    def sev(events):
        if not events:
            return None
        lm = np.array([e["l_max"] for e in events])
        dw = np.array([e["dwell"] for e in events])
        return {
            "n": len(events),
            "l_max_cm": {
                "median": float(np.median(lm) * 100),
                "q25": float(np.percentile(lm, 25) * 100),
                "q75": float(np.percentile(lm, 75) * 100),
                "max": float(lm.max() * 100),
            },
            "dwell": {
                "median": float(np.median(dw)),
                "q75": float(np.percentile(dw, 75)),
            },
        }

    agg["severity"] = {
        "missed_at_S1": sev([e for e in contacts if not e["caught"]["1"]]),
        "caught_at_S1": sev([e for e in contacts if e["caught"]["1"]]),
    }
    agg["timing_ms_per_step"] = {
        "continuous": 1e3 * timing["continuous"] / max(1, timing["n_steps"]),
        **{
            f"S{s}": 1e3 * timing[f"S{s}"] / max(1, timing[f"nS{s}"])
            for s in SUBSTEPS
        },
    }
    agg["config"] = {
        "dt": DT, "a_r": A_R, "episodes_per_seed_tier": args.episodes,
        "seeds": args.seeds, "n_obstacles": N_OBSTACLES,
        "thin_threshold": THIN_THRESHOLD, "total_collisions": len(contacts),
    }

    with open(out_dir / "m1h_results.json", "w") as f:
        json.dump({"aggregate": agg, "contacts": contacts}, f, indent=2)

    # ---- figure ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) miss rate vs speed with Wilson CIs, selected S.
    for s, c in [(1, "#3b0f70"), (4, "#0f9d77"), (16, "#b5cf49")]:
        pts, los, his = [], [], []
        for v in SPEEDS:
            p, lo, hi = agg["per_speed"][str(v)]["miss"][str(s)]["wilson"]
            pts.append(p), los.append(p - lo), his.append(hi - p)
        axes[0].errorbar(
            SPEEDS, pts, yerr=[los, his], fmt="o-", color=c, capsize=3,
            label=f"S={s}",
        )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("obstacle speed (m/s)")
    axes[0].set_ylabel("missed / true contacts")
    axes[0].set_title("(a) Miss rate by speed (Wilson 95% CI)")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # (b) thickness stratification, pooled over speeds.
    x = np.arange(len(SUBSTEPS))
    w = 0.38
    for off, label, color in [(-w / 2, "thin", "#b23b2e"), (w / 2, "compact", "#2563b8")]:
        pts, los, his = [], [], []
        for s in SUBSTEPS:
            p, lo, hi = agg["per_thickness"][label]["miss"][str(s)]["wilson"]
            pts.append(p), los.append(p - lo), his.append(hi - p)
        n = agg["per_thickness"][label]["collisions"]
        axes[1].bar(
            x + off, pts, w, yerr=[los, his], capsize=3, color=color,
            label=f"{label} (n={n})",
        )
    axes[1].set_xticks(x, [f"S={s}" for s in SUBSTEPS])
    axes[1].set_ylabel("missed / true contacts")
    axes[1].set_title("(b) Miss rate by obstacle thickness (pooled)")
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis="y")

    # (c) wall-clock cost.
    labels = ["exact\n(continuous)"] + [f"S={s}" for s in SUBSTEPS]
    vals = [agg["timing_ms_per_step"]["continuous"]] + [
        agg["timing_ms_per_step"][f"S{s}"] for s in SUBSTEPS
    ]
    colors = ["#0f9d77"] + ["#54503f"] * len(SUBSTEPS)
    axes[2].bar(labels, vals, color=colors)
    axes[2].set_ylabel("mean per-step cost (ms)")
    axes[2].set_title("(c) Wall-clock: exact audit vs discrete substeps")
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"M1-hardened: {args.seeds} seeds x {args.episodes} eps x {len(SPEEDS)} speeds, "
        f"{len(contacts)} audited collisions",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "figure1_hardened.png", dpi=160)
    print(f"wrote {out_dir}/m1h_results.json and figure1_hardened.png")
    print(json.dumps(agg["pooled"], indent=1))
    print("timing ms/step:", {k: round(v, 3) for k, v in agg["timing_ms_per_step"].items()})


if __name__ == "__main__":
    main()
