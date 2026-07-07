"""M5 — H2 in-RL: occupancy-grid observation with vs without the jointly
trained forecasting auxiliary task (the keystone's final validation).

Arms (identical GridTD3, identical curriculum, identical budget):
  aux    : lambda_aux = 1.0 (jointly trained forecasting head)
  noaux  : lambda_aux = 0.0 (encoder shaped by critic loss only)
Reference: M3-curriculum uoar VECTOR arm (obstacles as explicit
rel-pos+vel): succ 0.65 / true-coll 0.25 — the object-centric upper line.

Grid arms use obstacles_in_state=False: the policy's only obstacle channel
is the grid latent (deployment-realistic). The forecast label is the true
occupancy at t + HORIZON, recorded during rollout (delayed finalization).

Usage: [.venv-gpu|.venv]/bin/python experiments/m5_grid.py [--episodes 3000]
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
from godynur.grid_td3 import GridTD3  # noqa: E402

WARMUP_STEPS = 1500
EVAL_EVERY = 200
EVAL_EPISODES = 30
STAGES = [(0, 0.0), (2, 0.10), (3, 0.25)]
ADVANCE_WINDOW = 100
ADVANCE_THRESHOLD = 0.7
K_FRAMES = 3
HORIZON = 4  # forecast label at t + 4*dt = 0.2 s (kill-test condition)
N_SLOTS = 3


def hist_params(env, frames):
    return np.stack(frames[-K_FRAMES:])


def evaluate(env, agent):
    env.set_difficulty(*STAGES[-1])
    succ, coll = 0, 0
    for _ in range(EVAL_EPISODES):
        vec = env.reset()
        frames = [env.scene_params(N_SLOTS)] * K_FRAMES
        done = False
        while not done:
            a = agent.act(vec, hist_params(env, frames), explore=False)
            vec, _, done = env.step(a)
            frames.append(env.scene_params(N_SLOTS))
        if env.last_tau_star is not None:
            coll += 1
        elif env.on_goal >= env.goal_dwell:
            succ += 1
    return {"success": succ / EVAL_EPISODES, "collision": coll / EVAL_EPISODES}


def run_one(lambda_aux: float, episodes: int, seed: int, device: str | None,
            learn_every: int = 2):
    env = DynArmEnv(
        task="tabletop", n_obstacles=N_SLOTS, reward_mode="uoar",
        obstacles_in_state=False, seed=seed,
    )
    eval_env = DynArmEnv(
        task="tabletop", n_obstacles=N_SLOTS, reward_mode="uoar",
        obstacles_in_state=False, seed=10_000 + seed,
    )
    agent = GridTD3(
        vec_dim=env.state_dim, action_dim=env.action_dim,
        action_scale=env.action_bound[1], k_frames=K_FRAMES, n_slots=N_SLOTS,
        lambda_aux=lambda_aux, device=device, seed=seed,
    )
    tag = f"aux{lambda_aux:g}"
    stage_idx = 0
    env.set_difficulty(*STAGES[stage_idx])
    rolling = deque(maxlen=ADVANCE_WINDOW)
    history, total_steps, t0 = [], 0, time.time()

    for ep in range(1, episodes + 1):
        vec = env.reset()
        frames = [env.scene_params(N_SLOTS)] * K_FRAMES
        pending = []  # (vec, a, r, vec2, done, hist, hist2, step_idx)
        done, step_idx = False, 0
        while not done:
            h = hist_params(env, frames)
            if total_steps < WARMUP_STEPS:
                a = env.sample_action()
            else:
                a = agent.act(vec, h, explore=True)
            vec2, r, done = env.step(a)
            frames.append(env.scene_params(N_SLOTS))
            h2 = hist_params(env, frames)
            pending.append([vec, a, r, vec2, float(done), h, h2, step_idx])
            # Finalize the transition whose forecast label is now observable.
            if len(pending) > HORIZON:
                tr = pending.pop(0)
                agent.buffer.add(*tr[:7], fut=frames[-1], fut_valid=True)
            if total_steps % learn_every == 0:
                agent.learn()
            vec = vec2
            total_steps += 1
            step_idx += 1
        for tr in pending:  # episode ended before t + HORIZON: mask the label
            agent.buffer.add(*tr[:7], fut=None, fut_valid=False)
        rolling.append(1 if env.on_goal >= env.goal_dwell else 0)

        if (
            stage_idx < len(STAGES) - 1
            and len(rolling) == ADVANCE_WINDOW
            and np.mean(rolling) >= ADVANCE_THRESHOLD
        ):
            stage_idx += 1
            env.set_difficulty(*STAGES[stage_idx])
            rolling.clear()
            print(f"[{tag} seed{seed}] ep {ep}: ADVANCE to {STAGES[stage_idx]}", flush=True)

        if ep % EVAL_EVERY == 0 or ep == episodes:
            ev = evaluate(eval_env, agent)
            ev.update(
                {
                    "episode": ep, "stage": stage_idx,
                    "train_rolling_succ": float(np.mean(rolling)) if rolling else 0.0,
                    "aux_loss": agent.last_aux_loss,
                    "wall_s": round(time.time() - t0, 1),
                }
            )
            history.append(ev)
            print(
                f"[{tag} seed{seed}] ep {ep:>4} | stage {stage_idx} | roll "
                f"{ev['train_rolling_succ']:.2f} | final succ {ev['success']:.2f} "
                f"coll {ev['collision']:.2f} | aux {ev['aux_loss']:.4f} | "
                f"{ev['wall_s']}s",
                flush=True,
            )
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--arms", type=str, default="1.0,0.0")  # lambda_aux values
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--learn-every", type=int, default=2)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--out", type=str, default="experiments/results/m5_grid")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for lam in [float(x) for x in args.arms.split(",")]:
        key = f"aux{lam:g}"
        results[key] = {}
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            results[key][seed] = run_one(lam, args.episodes, seed, args.device,
                                         args.learn_every)
    tag = f"{args.arms.replace(',','-')}_s{args.seed_offset}"
    with open(out_dir / f"m5_{tag}.json", "w") as f:
        json.dump({"config": vars(args), "stages": STAGES, "results": results}, f, indent=2)
    for key in results:
        fin = [results[key][s][-1] for s in results[key]]
        print(
            f"{key:>7} final: succ {np.mean([f['success'] for f in fin]):.2f} "
            f"| true-coll {np.mean([f['collision'] for f in fin]):.2f}"
        )


if __name__ == "__main__":
    main()
