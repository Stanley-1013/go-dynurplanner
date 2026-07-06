"""M3-v2 — H3 experiment with the blueprint's curriculum (tabletop task).

The flat protocol (m3_baseline.py) showed plain TD3 cannot learn the fully
random task: episodes die in collision before goal signal accumulates —
an honest negative result motivating curriculum (and later ED2). This
version implements the blueprint §5 protocol:

  Stage 0: 0 obstacles                (learn reaching)
  Stage 1: 2 obstacles @ 0.10 m/s     (learn slow avoidance)
  Stage 2: 3 obstacles @ 0.25 m/s     (target difficulty)

Auto-advance when rolling train success over the last ADVANCE_WINDOW
episodes >= ADVANCE_THRESHOLD. State width is fixed (zero-padded obstacle
slots), so the same TD3 network learns across stages. The uoar-vs-ct
comparison (H3) is evaluated in the FINAL stage's dynamic scenes.

Usage: .venv/bin/python experiments/m3_curriculum.py [--episodes 3000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godynur.env import DynArmEnv  # noqa: E402
from godynur.td3 import TD3  # noqa: E402

WARMUP_STEPS = 1500
EVAL_EVERY = 200
EVAL_EPISODES = 30
STAGES = [(0, 0.0), (2, 0.10), (3, 0.25)]
ADVANCE_WINDOW = 100
ADVANCE_THRESHOLD = 0.7


def evaluate(env: DynArmEnv, agent: TD3, stage: tuple[int, float]):
    env.set_difficulty(*stage)
    succ, coll, tunneled = 0, 0, 0
    for _ in range(EVAL_EPISODES):
        s = env.reset()
        done = False
        while not done:
            s, _, done = env.step(agent.act(s, explore=False))
        if env.last_tau_star is not None:
            coll += 1
            tunneled += int(env.last_discrete_missed)
        elif env.on_goal >= env.goal_dwell:
            succ += 1
    return {
        "success": succ / EVAL_EPISODES,
        "collision": coll / EVAL_EPISODES,
        "tunneled": tunneled,
    }


def run_one(reward_mode: str, episodes: int, seed: int):
    env = DynArmEnv(
        task="tabletop", n_obstacles=3, reward_mode=reward_mode, seed=seed
    )
    eval_env = DynArmEnv(
        task="tabletop", n_obstacles=3, reward_mode=reward_mode, seed=10_000 + seed
    )
    agent = TD3(
        env.state_dim, env.action_dim, action_scale=env.action_bound[1], seed=seed
    )
    stage_idx = 0
    env.set_difficulty(*STAGES[stage_idx])
    rolling = deque(maxlen=ADVANCE_WINDOW)
    history, total_steps, t0 = [], 0, time.time()

    for ep in range(1, episodes + 1):
        s = env.reset()
        done = False
        while not done:
            a = (
                env.sample_action()
                if total_steps < WARMUP_STEPS
                else agent.act(s, explore=True)
            )
            s2, r, done = env.step(a)
            agent.buffer.add(s, a, r, s2, float(done))
            agent.learn()
            s = s2
            total_steps += 1
        rolling.append(1 if env.on_goal >= env.goal_dwell else 0)

        # Curriculum advance.
        if (
            stage_idx < len(STAGES) - 1
            and len(rolling) == ADVANCE_WINDOW
            and np.mean(rolling) >= ADVANCE_THRESHOLD
        ):
            stage_idx += 1
            env.set_difficulty(*STAGES[stage_idx])
            rolling.clear()
            print(
                f"[{reward_mode} seed{seed}] ep {ep}: ADVANCE to stage "
                f"{stage_idx} {STAGES[stage_idx]}",
                flush=True,
            )

        if ep % EVAL_EVERY == 0 or ep == episodes:
            ev_final = evaluate(eval_env, agent, STAGES[-1])
            ev_cur = (
                ev_final
                if stage_idx == len(STAGES) - 1
                else evaluate(eval_env, agent, STAGES[stage_idx])
            )
            rec = {
                "episode": ep,
                "stage": stage_idx,
                "train_rolling_succ": float(np.mean(rolling)) if rolling else 0.0,
                "eval_current_stage": ev_cur,
                "eval_final_stage": ev_final,
                "wall_s": round(time.time() - t0, 1),
            }
            history.append(rec)
            print(
                f"[{reward_mode} seed{seed}] ep {ep:>4} | stage {stage_idx} | "
                f"roll {rec['train_rolling_succ']:.2f} | final-stage succ "
                f"{ev_final['success']:.2f} coll {ev_final['collision']:.2f} | "
                f"{rec['wall_s']}s",
                flush=True,
            )
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--out", type=str, default="experiments/results/m3_curriculum")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ("uoar", "ct"):
        results[mode] = {}
        for seed in range(args.seeds):
            results[mode][seed] = run_one(mode, args.episodes, seed)
    with open(out_dir / "m3c_results.json", "w") as f:
        json.dump({"config": vars(args), "stages": STAGES, "results": results}, f, indent=2)

    for mode in results:
        fin = [results[mode][s][-1]["eval_final_stage"] for s in results[mode]]
        print(
            f"{mode:>5} final-stage: succ "
            f"{np.mean([f['success'] for f in fin]):.2f} | true-coll "
            f"{np.mean([f['collision'] for f in fin]):.2f}"
        )


if __name__ == "__main__":
    main()
