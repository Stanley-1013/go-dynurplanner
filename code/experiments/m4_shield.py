"""M4 — H4 experiment: APE2 candidate selection with vs without the
continuous-time interval shield, at equal training budget.

Both arms use the SAME TD3 learner and the SAME APE2 candidate pool /
hybrid evaluation (so exploration is identical in kind); the only
difference is whether candidates failing the certified interval check are
rejected (shield=on) or not (shield=off). Expected per blueprint H4:
shielded true-collision rate ~ 0 (up to the honest no-safe residual, which
is counted), with bounded success-rate cost.

Usage: .venv/bin/python experiments/m4_shield.py [--episodes 1500]
       [--speed 0.25] [--seeds 2] [--reward-mode ct]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godynur.ape2 import APE2Shield  # noqa: E402
from godynur.env import DynArmEnv  # noqa: E402
from godynur.td3 import TD3  # noqa: E402

WARMUP_STEPS = 1500
EVAL_EVERY = 100
EVAL_EPISODES = 30


def make_selector(env, agent, shield: bool, seed: int) -> APE2Shield:
    def base_fn(s):
        return agent.act(s, explore=False)

    @torch.no_grad()
    def q_fn(s, a):
        st = torch.from_numpy(s[None].astype(np.float32))
        at = torch.from_numpy((a / agent.action_scale)[None].astype(np.float32))
        sa = torch.cat([st, at], 1)
        return float(torch.min(agent.q1(sa), agent.q2(sa)))

    return APE2Shield(env, base_fn, q_fn, shield=shield, seed=seed)


def evaluate(env, agent, shield: bool, seed: int):
    sel = make_selector(env, agent, shield, seed=90_000 + seed)
    succ, coll, steps = 0, 0, 0
    for _ in range(EVAL_EPISODES):
        s = env.reset()
        done = False
        while not done:
            s, _, done = env.step(sel.act(s, step=10**9))  # eta=1: trust critic
            steps += 1
        if env.last_tau_star is not None:
            coll += 1
        elif env.on_goal >= env.goal_dwell:
            succ += 1
    return {
        "success": succ / EVAL_EPISODES,
        "collision": coll / EVAL_EPISODES,
        "mean_steps": steps / EVAL_EPISODES,
        "eval_shield_stats": dict(sel.stats),
    }


def run_one(shield: bool, reward_mode: str, speed: float, episodes: int, seed: int):
    env = DynArmEnv(speed=speed, reward_mode=reward_mode, seed=seed)
    eval_env = DynArmEnv(speed=speed, reward_mode=reward_mode, seed=10_000 + seed)
    agent = TD3(
        env.state_dim, env.action_dim, action_scale=env.action_bound[1], seed=seed
    )
    sel = make_selector(env, agent, shield, seed)
    history, total_steps, t0 = [], 0, time.time()
    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        while not done:
            if total_steps < WARMUP_STEPS:
                a = env.sample_action()
                if shield:  # shield the warmup too: safety during training
                    cand = a
                    a = None
                    for alpha in (1.0, 0.5, 0.25, 0.0):
                        if env.peek_tau_star(alpha * cand, sel._inflation(alpha * cand)) is None:
                            a = alpha * cand
                            break
                    if a is None:
                        a = cand  # honest residual; env counts the collision
            else:
                a = sel.act(s, step=total_steps)
            s2, r, done = env.step(a)
            agent.buffer.add(s, a, r, s2, float(done))
            agent.learn()
            s = s2
            total_steps += 1
        if ep % EVAL_EVERY == 0 or ep == episodes:
            ev = evaluate(eval_env, agent, shield, seed)
            ev.update(
                {
                    "episode": ep,
                    "wall_s": round(time.time() - t0, 1),
                    "train_shield_stats": dict(sel.stats),
                }
            )
            history.append(ev)
            tag = "shield" if shield else "noshield"
            print(
                f"[{tag} seed{seed}] ep {ep:>4} | succ {ev['success']:.2f} | "
                f"true-coll {ev['collision']:.2f} | no_safe {sel.stats['no_safe']} "
                f"| {ev['wall_s']}s",
                flush=True,
            )
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1500)
    ap.add_argument("--speed", type=float, default=0.25)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--reward-mode", type=str, default="ct")
    ap.add_argument("--out", type=str, default="experiments/results/m4_shield")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for shield in (True, False):
        key = "shield" if shield else "noshield"
        results[key] = {}
        for seed in range(args.seeds):
            results[key][seed] = run_one(
                shield, args.reward_mode, args.speed, args.episodes, seed
            )
    with open(out_dir / f"m4_speed{args.speed}_{args.reward_mode}.json", "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)

    for key in results:
        fin = [results[key][s][-1] for s in results[key]]
        print(
            f"{key:>9} final: succ {np.mean([f['success'] for f in fin]):.2f} "
            f"| true-coll {np.mean([f['collision'] for f in fin]):.2f}"
        )


if __name__ == "__main__":
    main()
