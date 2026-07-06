"""M3-step-1 — first H3 experiment: discrete UOAR vs continuous-time reward.

Identical TD3, identical budget, identical seeds; the ONLY difference is
DynArmEnv's reward_mode ('uoar' = URPlanner Eq.12 discrete accounting,
'ct' = D-UOAR-CT with swept integral + continuous TTC). Because the env's
termination/collision bookkeeping is continuous-time exact regardless of
reward mode, the comparison is clean: does correcting the OBJECTIVE to the
continuous-time collision cost reduce TRUE collision rate at equal budget?

Also logged per run (free instrumentation from the env):
  - tunneled collisions: true collisions the endpoint check would miss —
    i.e. collisions an ordinary discrete env would silently ignore.

Usage: .venv/bin/python experiments/m3_baseline.py [--episodes 600]
       [--speed 0.5] [--seeds 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godynur.env import DynArmEnv  # noqa: E402
from godynur.td3 import TD3  # noqa: E402

WARMUP_STEPS = 1500  # uniform-random action steps before policy takes over
EVAL_EVERY = 100  # episodes
EVAL_EPISODES = 30


def evaluate(env: DynArmEnv, agent: TD3):
    succ, coll, tunneled, steps = 0, 0, 0, 0
    for _ in range(EVAL_EPISODES):
        s = env.reset()
        done = False
        while not done:
            s, _, done = env.step(agent.act(s, explore=False))
            steps += 1
        if env.last_tau_star is not None:
            coll += 1
            tunneled += int(env.last_discrete_missed)
        elif env.on_goal >= env.goal_dwell:
            succ += 1
    return {
        "success": succ / EVAL_EPISODES,
        "collision": coll / EVAL_EPISODES,
        "tunneled": tunneled,
        "mean_steps": steps / EVAL_EPISODES,
    }


def run_one(reward_mode: str, speed: float, episodes: int, seed: int):
    env = DynArmEnv(speed=speed, reward_mode=reward_mode, seed=seed)
    eval_env = DynArmEnv(speed=speed, reward_mode=reward_mode, seed=10_000 + seed)
    agent = TD3(
        env.state_dim, env.action_dim, action_scale=env.action_bound[1], seed=seed
    )
    history, total_steps, t0 = [], 0, time.time()
    for ep in range(1, episodes + 1):
        s = env.reset()
        done, ep_r = False, 0.0
        while not done:
            if total_steps < WARMUP_STEPS:
                a = env.sample_action()
            else:
                a = agent.act(s, explore=True)
            s2, r, done = env.step(a)
            agent.buffer.add(s, a, r, s2, float(done))
            agent.learn()
            s, ep_r = s2, ep_r + r
            total_steps += 1
        if ep % EVAL_EVERY == 0 or ep == episodes:
            ev = evaluate(eval_env, agent)
            ev.update({"episode": ep, "wall_s": round(time.time() - t0, 1)})
            history.append(ev)
            print(
                f"[{reward_mode} seed{seed}] ep {ep:>4} | succ {ev['success']:.2f} "
                f"| true-coll {ev['collision']:.2f} | tunneled {ev['tunneled']} "
                f"| {ev['wall_s']}s",
                flush=True,
            )
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--speed", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=str, default="experiments/results/m3_baseline")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ("uoar", "ct"):
        results[mode] = {}
        for seed in range(args.seeds):
            results[mode][seed] = run_one(mode, args.speed, args.episodes, seed)
    with open(out_dir / f"m3_speed{args.speed}.json", "w") as f:
        json.dump({"config": vars(args), "results": results}, f, indent=2)
    print(f"wrote {out_dir}/m3_speed{args.speed}.json")

    # Final-eval summary across seeds.
    for mode in ("uoar", "ct"):
        fin = [results[mode][s][-1] for s in results[mode]]
        print(
            f"{mode:>5} final: succ {np.mean([f['success'] for f in fin]):.2f} "
            f"| true-coll {np.mean([f['collision'] for f in fin]):.2f}"
        )


if __name__ == "__main__":
    main()
