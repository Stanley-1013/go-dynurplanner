"""M4 — H4 experiment: APE2 candidate selection with vs without the
continuous-time interval shield, at equal training budget.

Both arms use the SAME TD3 learner and the SAME APE2 candidate pool /
hybrid evaluation (so exploration is identical in kind); the only
difference is whether candidates failing the certified interval check are
rejected (shield=on) or not (shield=off). Expected per blueprint H4:
shielded true-collision rate ~ 0 (up to the honest no-safe residual, which
is counted), with bounded success-rate cost.

Ported onto the blueprint §5 curriculum protocol (see m3_curriculum.py):
tabletop task, STAGES [(0,0.0),(2,0.10),(3,0.25)], rolling-success
auto-advance on the TRAIN env only, fixed-width state via
env.set_difficulty. Evaluation always runs in the FINAL stage's dynamic
scenes, mirroring m3_curriculum.evaluate.

Usage: .venv/bin/python experiments/m4_shield.py [--episodes 3000]
       [--seeds 2] [--reward-mode uoar]
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


def make_selector(env, agent, shield: bool, seed: int) -> APE2Shield:
    def base_fn(s):
        return agent.act(s, explore=False)

    @torch.no_grad()
    def q_fn(s, a):
        st = torch.from_numpy(s[None].astype(np.float32))
        at = torch.from_numpy((a / agent.action_scale)[None].astype(np.float32))
        sa = torch.cat([st, at], 1)
        return float(torch.min(agent.q1(sa), agent.q2(sa)))

    return APE2Shield(env, base_fn, q_fn, shield=shield, seed=seed,
                      anneal_steps=150_000)


def shielded_actor_action(env, agent, sel, s):
    """Deployment semantics: actor action passed through the certification
    ladder (scale-down along its direction; last resort = max-tau* candidate
    from the pool around it)."""
    a = agent.act(s, explore=False)
    if env.peek_tau_star(a, sel._inflation(a)) is None:
        return a
    # Active dodge: search the candidate pool around the actor action for a
    # CERTIFIED alternative (still a hard guarantee — just more directions
    # than the passive slow-down ladder, which traps cornered states).
    cands = sel._candidates(s)
    certified = [c for c in cands if env.peek_tau_star(c, sel._inflation(c)) is None]
    if certified:
        sel.stats["scaled"] += 1  # reuse counter: intervention happened
        best = max(certified, key=lambda c: env.peek_reward(c))
        return best
    for alpha in (0.5, 0.25, 0.125, 0.0):
        if env.peek_tau_star(alpha * a, sel._inflation(alpha * a)) is None:
            return alpha * a
    sel.stats["no_safe"] += 1
    cands = sel._candidates(s)
    taus = [(env.peek_tau_star(c, sel._inflation(c)) or np.inf, k)
            for k, c in enumerate(cands)]
    return cands[max(taus)[1]]


def evaluate(env, agent, shield: bool, seed: int):
    env.set_difficulty(*STAGES[-1])
    sel = make_selector(env, agent, shield, seed=90_000 + seed)
    succ, coll, steps = 0, 0, 0
    for _ in range(EVAL_EPISODES):
        s = env.reset()
        done = False
        while not done:
            a = (
                shielded_actor_action(env, agent, sel, s)
                if shield
                else agent.act(s, explore=False)
            )
            s, _, done = env.step(a)
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


def run_one(shield: bool, reward_mode: str, episodes: int, seed: int):
    env = DynArmEnv(
        task="tabletop", n_obstacles=3, reward_mode=reward_mode, seed=seed
    )
    eval_env = DynArmEnv(
        task="tabletop", n_obstacles=3, reward_mode=reward_mode, seed=10_000 + seed
    )
    agent = TD3(
        env.state_dim, env.action_dim, action_scale=env.action_bound[1],
        expl_noise=0.25, seed=seed,
    )
    sel = make_selector(env, agent, shield, seed)
    stage_idx = 0
    env.set_difficulty(*STAGES[stage_idx])
    rolling = deque(maxlen=ADVANCE_WINDOW)
    history, total_steps, t0 = [], 0, time.time()
    tag = "shield" if shield else "noshield"

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
        rolling.append(1 if env.on_goal >= env.goal_dwell else 0)

        # Curriculum advance (train env only).
        if (
            stage_idx < len(STAGES) - 1
            and len(rolling) == ADVANCE_WINDOW
            and np.mean(rolling) >= ADVANCE_THRESHOLD
        ):
            stage_idx += 1
            env.set_difficulty(*STAGES[stage_idx])
            rolling.clear()
            print(
                f"[{tag} seed{seed}] ep {ep}: ADVANCE to stage "
                f"{stage_idx} {STAGES[stage_idx]}",
                flush=True,
            )

        if ep % EVAL_EVERY == 0 or ep == episodes:
            ev = evaluate(eval_env, agent, shield, seed)
            ev.update(
                {
                    "episode": ep,
                    "stage": stage_idx,
                    "train_rolling_succ": float(np.mean(rolling)) if rolling else 0.0,
                    "wall_s": round(time.time() - t0, 1),
                    "train_shield_stats": dict(sel.stats),
                }
            )
            history.append(ev)
            print(
                f"[{tag} seed{seed}] ep {ep:>4} | stage {stage_idx} | "
                f"succ {ev['success']:.2f} | true-coll {ev['collision']:.2f} "
                f"| no_safe {sel.stats['no_safe']} | {ev['wall_s']}s",
                flush=True,
            )
    torch.save(
        {"actor": agent.actor.state_dict(), "q1": agent.q1.state_dict(),
         "q2": agent.q2.state_dict()},
        Path("experiments/results/m4_shield") / f"td3_{tag}_seed{seed}.pt",
    )
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=3000)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--reward-mode", type=str, default="uoar")
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
                shield, args.reward_mode, args.episodes, seed
            )
    with open(out_dir / f"m4_{args.reward_mode}.json", "w") as f:
        json.dump(
            {"config": vars(args), "stages": STAGES, "results": results}, f, indent=2
        )

    for key in results:
        fin = [results[key][s][-1] for s in results[key]]
        print(
            f"{key:>9} final: succ {np.mean([f['success'] for f in fin]):.2f} "
            f"| true-coll {np.mean([f['collision'] for f in fin]):.2f}"
        )


if __name__ == "__main__":
    main()
