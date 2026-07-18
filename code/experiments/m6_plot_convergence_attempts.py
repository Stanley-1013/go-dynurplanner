"""Plot success-rate trajectories across all kinodynamic_shield RL-convergence
attempts, for the Phase 5 final synthesis. Reads only already-committed (or
still-on-disk /tmp) result JSONs -- does not run any new experiments.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

RUNS = [
    ("0: baseline (no fixes)", ROOT / "experiments/results/m6_kinodynamic/m6_uoar_kinodynamic_shield_salt0.json", "0", "tab:gray"),
    ("1: +buffer fix (3000ep)", ROOT / "experiments/results/m6_kinodynamic_longrun/m6_uoar_kinodynamic_shield_salt0.json", "0", "tab:blue"),
    ("2: +v-obs fix (800ep)", ROOT / "experiments/results/m6_kinodynamic_vfix_probe/m6_uoar_kinodynamic_shield_salt0.json", "0", "tab:orange"),
    ("2: +v-obs fix (3000ep, decisive)", ROOT / "experiments/results/m6_kinodynamic_vfix_longrun/m6_uoar_kinodynamic_shield_salt0.json", "0", "tab:orange"),
    ("3: +margin-obs fix (800ep)", Path("/tmp/m6_margin_probe/m6_uoar_kinodynamic_shield_salt200.json"), "0", "tab:green"),
    ("4: +hyperparam sweep (800ep)", ROOT / "experiments/results/m6_kinodynamic_sweep1/m6_uoar_kinodynamic_shield_salt0.json", "0", "tab:red"),
    ("5: +STE actor grad (800ep)", ROOT / "experiments/results/m6_kinodynamic_ste/m6_uoar_kinodynamic_shield_salt0.json", "0", "tab:purple"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ax_succ, ax_interv = axes

for name, path, seed, color in RUNS:
    if not path.exists():
        print(f"SKIP (missing): {name} -> {path}")
        continue
    with open(path) as f:
        d = json.load(f)
    hist = d["results"]["kinodynamic_shield"][seed]
    eps = [e["episode"] for e in hist]
    succ = [e["success"] for e in hist]
    interv = [e.get("train_intervention_rate") for e in hist]
    linestyle = "--" if "3000ep" in name else "-"
    marker = "o" if "3000ep" not in name else None
    ax_succ.plot(eps, succ, label=name, color=color, linestyle=linestyle, marker=marker, alpha=0.85)
    if any(v is not None for v in interv):
        ax_interv.plot(eps, interv, label=name, color=color, linestyle=linestyle, marker=marker, alpha=0.85)

# 6th attempt: differentiable QP, single truncated point (no full JSON)
ax_succ.scatter([200], [0.0], color="black", marker="x", s=100, zorder=5,
                label="6: +diff. QP (ep200 only, truncated)")

ax_succ.axhline(0.38, color="black", linestyle=":", linewidth=1, alpha=0.6)
ax_succ.text(50, 0.39, "scripted (non-learning) policy: 38%", fontsize=8, alpha=0.7)
ax_succ.set_xlabel("training episode")
ax_succ.set_ylabel("eval success rate (30 episodes)")
ax_succ.set_title("Success rate across all 6 convergence attempts\n(all single-seed except attempt 0, n=5)")
ax_succ.set_ylim(-0.02, 0.45)
ax_succ.legend(fontsize=7, loc="upper right")
ax_succ.grid(alpha=0.3)

ax_interv.set_xlabel("training episode")
ax_interv.set_ylabel("shield intervention rate (train, block-local)")
ax_interv.set_title("Shield intervention rate over training\n(does NOT trend down in any attempt)")
ax_interv.set_ylim(0.15, 0.35)
ax_interv.legend(fontsize=7, loc="upper right")
ax_interv.grid(alpha=0.3)

fig.suptitle("GO-DynURPlanner Phase 5: kinodynamic_shield RL-convergence attempts (2026-07-18)", fontsize=11)
fig.tight_layout()

out_path = ROOT / "experiments/results/m6_kinodynamic/convergence_attempts.png"
fig.savefig(out_path, dpi=150)
print(f"saved: {out_path}")
