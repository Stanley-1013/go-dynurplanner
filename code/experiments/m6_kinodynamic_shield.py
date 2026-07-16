"""M6 — Phase-5a comparison of delta-q and kinodynamic safety mechanisms.

Arms (same TD3 hyperparameters, curriculum, task, and evaluation protocol):
  no_shield          : delta-q actions with plain TD3
  ape2_shield        : delta-q actions with the published APE2 shield
  kinodynamic_shield : velocity actions with safety enforced by DynArmEnv

This intentionally mirrors m4_shield.py's tabletop protocol.  The velocity
arm does not use APE2: its QP, terminal-membership check, fallback, and
emergency behavior are part of env.step().

Cheap Phase-5 metrics available today are logged here.  Standalone joint-limit
violation and inter-sample violation counters are deferred to a later Phase-5
pass because the current shield prevents those events but does not expose
separate counters for them.  Path length, jerk/energy cost, and per-QP timing
are likewise deferred; episode wall time is recorded now as a coarse solve-
overhead proxy without changing godynur instrumentation.

Usage: .venv/bin/python experiments/m6_kinodynamic_shield.py [--episodes 300]
       [--seeds 2] [--arms no_shield,ape2_shield,kinodynamic_shield]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godynur.ape2 import APE2Shield  # noqa: E402
from godynur.env import DynArmEnv  # noqa: E402
from godynur.td3 import TD3  # noqa: E402

WARMUP_STEPS = 1500
EVAL_EVERY = 200
EVAL_EPISODES = 30
STAGES = [(0, 0.0), (2, 0.10), (3, 0.25)]
ADVANCE_WINDOW = 100
ADVANCE_THRESHOLD = 0.7

ARMS = ("no_shield", "ape2_shield", "kinodynamic_shield")
# Stable disjoint ranges mean arms never share a raw seed value, including
# when a subset is launched separately.  --seed-salt follows m5_grid.py's
# deconfounding convention and shifts every range together.
ARM_SEED_STRIDE = 1_000_000


def make_selector(env, agent, seed: int) -> APE2Shield:
    def base_fn(s):
        return agent.act(s, explore=False)

    @torch.no_grad()
    def q_fn(s, a):
        st = torch.from_numpy(s[None].astype(np.float32))
        at = torch.from_numpy((a / agent.action_scale)[None].astype(np.float32))
        sa = torch.cat([st, at], 1)
        return float(torch.min(agent.q1(sa), agent.q2(sa)))

    return APE2Shield(
        env, base_fn, q_fn, shield=True, seed=seed, anneal_steps=150_000
    )


def shielded_actor_action(env, agent, sel, s):
    """M4's deployment-time certification ladder, unchanged."""
    a = agent.act(s, explore=False)
    if env.peek_tau_star(a, sel._inflation(a)) is None:
        return a
    cands = sel._candidates(s)
    certified = [
        c for c in cands
        if env.peek_tau_star(c, sel._inflation(c)) is None
    ]
    if certified:
        sel.stats["scaled"] += 1
        return max(certified, key=lambda c: env.peek_reward(c))
    for alpha in (0.5, 0.25, 0.125, 0.0):
        if env.peek_tau_star(alpha * a, sel._inflation(alpha * a)) is None:
            return alpha * a
    sel.stats["no_safe"] += 1
    cands = sel._candidates(s)
    taus = [
        (env.peek_tau_star(c, sel._inflation(c)) or np.inf, k)
        for k, c in enumerate(cands)
    ]
    return cands[max(taus)[1]]


def _kinodynamic_counts(env: DynArmEnv) -> dict[str, int]:
    return {
        "shield_fallback": int(env.stats.get("shield_fallback", 0)),
        "shield_emergency": int(env.stats.get("shield_emergency", 0)),
    }


def evaluate(env: DynArmEnv, agent: TD3, arm: str, seed: int):
    env.set_difficulty(*STAGES[-1])
    sel = (
        make_selector(env, agent, seed=90_000 + seed)
        if arm == "ape2_shield"
        else None
    )
    counts_before = _kinodynamic_counts(env)
    succ, coll, steps = 0, 0, 0
    success_steps: list[int] = []
    episode_wall_s: list[float] = []
    intervention_steps = 0
    intervention_norm_sum = 0.0

    for _ in range(EVAL_EPISODES):
        episode_t0 = time.perf_counter()
        s = env.reset()
        done = False
        episode_steps = 0
        while not done:
            if arm == "ape2_shield":
                a = shielded_actor_action(env, agent, sel, s)
            else:
                a = agent.act(s, explore=False)
            s, _, done = env.step(a)
            episode_steps += 1
            if arm == "kinodynamic_shield":
                intervention_steps += int(not env._last_terminal_membership)
                intervention_norm_sum += env._last_intervention_norm
        episode_wall_s.append(time.perf_counter() - episode_t0)
        steps += episode_steps
        if env.last_tau_star is not None:
            coll += 1
        elif env.on_goal >= env.goal_dwell:
            succ += 1
            success_steps.append(episode_steps)

    eval_stats: dict[str, int] = {}
    if sel is not None:
        eval_stats = dict(sel.stats)
    elif arm == "kinodynamic_shield":
        counts_after = _kinodynamic_counts(env)
        eval_stats = {
            key: counts_after[key] - counts_before[key]
            for key in counts_after
        }

    return {
        "success": succ / EVAL_EPISODES,
        "collision": coll / EVAL_EPISODES,
        "mean_steps": steps / EVAL_EPISODES,
        "mean_steps_to_success": (
            float(np.mean(success_steps)) if success_steps else None
        ),
        "eval_shield_stats": eval_stats,
        "eval_intervention_rate": (
            intervention_steps / steps if arm == "kinodynamic_shield" else None
        ),
        "eval_mean_intervention_norm": (
            intervention_norm_sum / steps
            if arm == "kinodynamic_shield"
            else None
        ),
        "eval_episode_wall_s": [round(value, 6) for value in episode_wall_s],
        "eval_mean_episode_wall_s": float(np.mean(episode_wall_s)),
    }


def run_one(
    arm: str,
    reward_mode: str,
    episodes: int,
    seed: int,
    out_dir: Path,
):
    action_mode = "velocity" if arm == "kinodynamic_shield" else "delta_q"
    env = DynArmEnv(
        task="tabletop",
        n_obstacles=3,
        reward_mode=reward_mode,
        seed=seed,
        action_mode=action_mode,
    )
    eval_env = DynArmEnv(
        task="tabletop",
        n_obstacles=3,
        reward_mode=reward_mode,
        seed=10_000 + seed,
        action_mode=action_mode,
    )
    agent = TD3(
        env.state_dim,
        env.action_dim,
        action_scale=env.action_bound[1],
        expl_noise=0.25,
        seed=seed,
    )
    sel = make_selector(env, agent, seed) if arm == "ape2_shield" else None
    stage_idx = 0
    env.set_difficulty(*STAGES[stage_idx])
    rolling = deque(maxlen=ADVANCE_WINDOW)
    history = []
    total_steps = 0
    t0 = time.perf_counter()

    block_steps = 0
    block_interventions = 0
    block_intervention_norm = 0.0
    block_episode_wall_s: list[float] = []

    for ep in range(1, episodes + 1):
        episode_t0 = time.perf_counter()
        s = env.reset()
        done = False
        while not done:
            if total_steps < WARMUP_STEPS:
                a = env.sample_action()
                if arm == "ape2_shield":
                    cand = a
                    a = None
                    for alpha in (1.0, 0.5, 0.25, 0.0):
                        if (
                            env.peek_tau_star(
                                alpha * cand, sel._inflation(alpha * cand)
                            )
                            is None
                        ):
                            a = alpha * cand
                            break
                    if a is None:
                        a = cand
            elif arm == "ape2_shield":
                a = sel.act(s, step=total_steps)
            else:
                a = agent.act(s, explore=True)

            s2, r, done = env.step(a)
            agent.buffer.add(s, a, r, s2, float(done))
            agent.learn()
            s = s2
            total_steps += 1
            block_steps += 1
            if arm == "kinodynamic_shield":
                block_interventions += int(not env._last_terminal_membership)
                block_intervention_norm += env._last_intervention_norm

        block_episode_wall_s.append(time.perf_counter() - episode_t0)
        rolling.append(1 if env.on_goal >= env.goal_dwell else 0)

        if (
            stage_idx < len(STAGES) - 1
            and len(rolling) == ADVANCE_WINDOW
            and np.mean(rolling) >= ADVANCE_THRESHOLD
        ):
            stage_idx += 1
            env.set_difficulty(*STAGES[stage_idx])
            rolling.clear()
            print(
                f"[{arm} seed{seed}] ep {ep}: ADVANCE to stage "
                f"{stage_idx} {STAGES[stage_idx]}",
                flush=True,
            )

        if ep % EVAL_EVERY == 0 or ep == episodes:
            ev = evaluate(eval_env, agent, arm, seed)
            ev.update(
                {
                    "episode": ep,
                    "stage": stage_idx,
                    "actual_seed": seed,
                    "train_rolling_succ": (
                        float(np.mean(rolling)) if rolling else 0.0
                    ),
                    "wall_s": round(time.perf_counter() - t0, 1),
                    "train_shield_stats": dict(sel.stats) if sel else {},
                    "train_env_stats": (
                        _kinodynamic_counts(env)
                        if arm == "kinodynamic_shield"
                        else {}
                    ),
                    # Block-local values make intervention-over-training trends
                    # directly comparable at every EVAL_EVERY checkpoint.
                    "train_intervention_rate": (
                        block_interventions / block_steps
                        if arm == "kinodynamic_shield"
                        else None
                    ),
                    "train_mean_intervention_norm": (
                        block_intervention_norm / block_steps
                        if arm == "kinodynamic_shield"
                        else None
                    ),
                    "train_episode_wall_s": [
                        round(value, 6) for value in block_episode_wall_s
                    ],
                    "train_mean_episode_wall_s": float(
                        np.mean(block_episode_wall_s)
                    ),
                }
            )
            history.append(ev)
            no_safe = sel.stats["no_safe"] if sel else 0
            intervention = ev["train_intervention_rate"]
            intervention_text = (
                f" | intervene {intervention:.3f}"
                if intervention is not None
                else ""
            )
            print(
                f"[{arm} seed{seed}] ep {ep:>4} | stage {stage_idx} | "
                f"succ {ev['success']:.2f} | true-coll {ev['collision']:.2f} "
                f"| no_safe {no_safe}{intervention_text} | {ev['wall_s']}s",
                flush=True,
            )
            block_steps = 0
            block_interventions = 0
            block_intervention_norm = 0.0
            block_episode_wall_s = []

    torch.save(
        {
            "actor": agent.actor.state_dict(),
            "q1": agent.q1.state_dict(),
            "q2": agent.q2.state_dict(),
        },
        out_dir / f"td3_{arm}_seed{seed}.pt",
    )
    return history


def _parse_arms(value: str) -> list[str]:
    arms = [arm.strip() for arm in value.split(",") if arm.strip()]
    invalid = [arm for arm in arms if arm not in ARMS]
    if not arms or invalid or len(set(arms)) != len(arms):
        choices = ",".join(ARMS)
        raise argparse.ArgumentTypeError(
            f"choose a unique comma-separated subset of {choices}"
        )
    return arms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument(
        "--seed-salt",
        type=int,
        default=0,
        help="added to disjoint per-arm seed ranges (M5 seed-confound fix)",
    )
    ap.add_argument(
        "--arms",
        type=_parse_arms,
        default=list(ARMS),
        help="comma-separated subset of no_shield,ape2_shield,kinodynamic_shield",
    )
    ap.add_argument("--reward-mode", type=str, default="uoar")
    ap.add_argument(
        "--out", type=str, default="experiments/results/m6_kinodynamic"
    )
    args = ap.parse_args()
    if args.episodes < 1 or args.seeds < 1:
        ap.error("--episodes and --seeds must be positive")
    if args.seeds >= ARM_SEED_STRIDE:
        ap.error(f"--seeds must be less than {ARM_SEED_STRIDE}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for arm in args.arms:
        results[arm] = {}
        arm_offset = ARMS.index(arm) * ARM_SEED_STRIDE
        for seed_index in range(args.seeds):
            actual_seed = args.seed_salt + arm_offset + seed_index
            results[arm][seed_index] = run_one(
                arm, args.reward_mode, args.episodes, actual_seed, out_dir
            )

    arm_tag = "-".join(args.arms)
    json_path = out_dir / f"m6_{args.reward_mode}_{arm_tag}_salt{args.seed_salt}.json"
    config = vars(args).copy()
    config["arms"] = args.arms
    with open(json_path, "w") as f:
        json.dump(
            {"config": config, "stages": STAGES, "results": results},
            f,
            indent=2,
        )

    for arm in results:
        fin = [results[arm][seed][-1] for seed in results[arm]]
        print(
            f"{arm:>19} final: succ "
            f"{np.mean([record['success'] for record in fin]):.2f} "
            f"| true-coll "
            f"{np.mean([record['collision'] for record in fin]):.2f}"
        )
    print(f"results: {json_path}")


if __name__ == "__main__":
    main()
