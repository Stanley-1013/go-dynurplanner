"""Stage-0 physical-reachability diagnostic for the velocity-action arm.

This is deliberately standalone experiment code: it does not alter or bypass
DynArmEnv's kinodynamic safety shield.  A scripted controller first solves a
position-only IK problem for the sampled Cartesian goal, then applies a
moderate, constant velocity-increment command along that joint-space path.
The command is normalized by ``dv_scale`` so all joints make synchronized
progress despite the Panda's two velocity-limit groups.

The same stage-0 task stream is also evaluated with the saved 3000-episode
TD3 actor.  Results are printed as JSON for reproducibility.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godynur.env import DynArmEnv  # noqa: E402
from godynur.panda import DQ_MAX, Q_MAX, Q_MIN  # noqa: E402
from godynur.td3 import TD3  # noqa: E402


TRAIN_SEED = 2_000_000
EVAL_SEED = 10_000 + TRAIN_SEED
SNAPSHOT_STEPS = (1, 10, 25, 50, 75, 100)
SCRIPTED_AMPLITUDE = 0.2
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "results/m6_kinodynamic_longrun/td3_kinodynamic_shield_seed2000000.pt"
)


def make_stage0_env() -> DynArmEnv:
    """Match M6's velocity arm construction, then select curriculum stage 0."""
    env = DynArmEnv(
        task="tabletop",
        n_obstacles=3,
        reward_mode="uoar",
        seed=EVAL_SEED,
        action_mode="velocity",
    )
    env.set_difficulty(0, 0.0)
    return env


def ik_target(env: DynArmEnv) -> tuple[np.ndarray, float]:
    """Find a nearby joint configuration realizing the sampled flange goal."""
    result = least_squares(
        lambda q: env.kin.flange(q) - env.goal,
        env.q.copy(),
        bounds=(Q_MIN, Q_MAX),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=500,
    )
    residual = float(np.linalg.norm(env.kin.flange(result.x) - env.goal))
    return result.x, residual


def scripted_action(env: DynArmEnv, q_start: np.ndarray, q_target: np.ndarray) -> np.ndarray:
    """Moderate bang-bang-style push along a straight joint-space IK path."""
    scaled_direction = (q_target - q_start) / env.dv_scale
    peak = float(np.max(np.abs(scaled_direction)))
    if peak < 1e-12:
        return np.zeros(env.action_dim)
    return SCRIPTED_AMPLITUDE * scaled_direction / peak


def termination(env: DynArmEnv) -> str:
    if env.last_tau_star is not None:
        return "collision"
    if env.on_goal >= env.goal_dwell:
        return "success"
    return "timeout"


def summarize(name: str, episodes: list[dict]) -> dict:
    n = len(episodes)
    counts = {
        cause: sum(ep["termination"] == cause for ep in episodes)
        for cause in ("success", "collision", "timeout")
    }
    velocities = np.concatenate(
        [np.asarray(ep["velocity_norms"], dtype=float) for ep in episodes]
    )
    velocity_fractions = np.concatenate(
        [np.asarray(ep["velocity_fractions"], dtype=float) for ep in episodes]
    )
    steps = np.asarray([ep["steps"] for ep in episodes], dtype=float)
    result = {
        "policy": name,
        "episodes": n,
        "termination_counts": counts,
        "termination_fractions": {key: value / n for key, value in counts.items()},
        "steps_mean": float(np.mean(steps)),
        "steps_median": float(np.median(steps)),
        "steps_by_termination": {
            cause: {
                "mean": float(np.mean([ep["steps"] for ep in episodes if ep["termination"] == cause])),
                "median": float(np.median([ep["steps"] for ep in episodes if ep["termination"] == cause])),
            }
            for cause in counts
            if counts[cause]
        },
        "velocity_norm_rad_s": {
            "mean": float(np.mean(velocities)),
            "median": float(np.median(velocities)),
            "p90": float(np.percentile(velocities, 90)),
            "max": float(np.max(velocities)),
        },
        # RMS fraction across joints: 1.0 means every joint is at its DQ_MAX.
        "velocity_dqmax_rms_fraction": {
            "mean": float(np.mean(velocity_fractions)),
            "median": float(np.median(velocity_fractions)),
            "p90": float(np.percentile(velocity_fractions, 90)),
            "max": float(np.max(velocity_fractions)),
        },
        "sample_episode_velocity_snapshots": [
            {
                "episode": ep["episode"],
                "termination": ep["termination"],
                "steps": ep["steps"],
                "norm_rad_s_by_step": ep["snapshots"],
            }
            for ep in episodes[:5]
        ],
        "shield_intervention_rate": (
            sum(ep["interventions"] for ep in episodes)
            / sum(ep["steps"] for ep in episodes)
        ),
    }
    return result


def rollout_scripted(n_episodes: int) -> tuple[dict, list[dict]]:
    env = make_stage0_env()
    episodes = []
    ik_residuals = []
    for episode in range(n_episodes):
        env.last_tau_star = None
        env.reset()
        q_start = env.q.copy()
        q_target, residual = ik_target(env)
        ik_residuals.append(residual)
        action = scripted_action(env, q_start, q_target)
        done = False
        norms = []
        fractions = []
        snapshots = {}
        interventions = 0
        while not done:
            _, _, done = env.step(action)
            norm = float(np.linalg.norm(env.v))
            norms.append(norm)
            fractions.append(float(np.linalg.norm(env.v / DQ_MAX) / np.sqrt(env.action_dim)))
            interventions += int(not env._last_terminal_membership)
            if env.t in SNAPSHOT_STEPS:
                snapshots[str(env.t)] = norm
        episodes.append(
            {
                "episode": episode,
                "termination": termination(env),
                "steps": env.t,
                "velocity_norms": norms,
                "velocity_fractions": fractions,
                "snapshots": snapshots,
                "interventions": interventions,
            }
        )
    summary = summarize("scripted_ik_path", episodes)
    summary["scripted_amplitude"] = SCRIPTED_AMPLITUDE
    summary["ik_residual_m"] = {
        "median": float(np.median(ik_residuals)),
        "max": float(np.max(ik_residuals)),
    }
    return summary, episodes


def load_actor(env: DynArmEnv, checkpoint: Path) -> TD3:
    agent = TD3(
        env.state_dim,
        env.action_dim,
        action_scale=env.action_bound[1],
        expl_noise=0.25,
        seed=TRAIN_SEED,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    agent.actor.load_state_dict(saved["actor"])
    return agent


def rollout_trained(n_episodes: int, checkpoint: Path) -> tuple[dict, list[dict]]:
    env = make_stage0_env()
    agent = load_actor(env, checkpoint)
    episodes = []
    for episode in range(n_episodes):
        env.last_tau_star = None
        state = env.reset()
        done = False
        norms = []
        fractions = []
        snapshots = {}
        interventions = 0
        while not done:
            action = agent.act(state, explore=False)
            state, _, done = env.step(action)
            norm = float(np.linalg.norm(env.v))
            norms.append(norm)
            fractions.append(float(np.linalg.norm(env.v / DQ_MAX) / np.sqrt(env.action_dim)))
            interventions += int(not env._last_terminal_membership)
            if env.t in SNAPSHOT_STEPS:
                snapshots[str(env.t)] = norm
        episodes.append(
            {
                "episode": episode,
                "termination": termination(env),
                "steps": env.t,
                "velocity_norms": norms,
                "velocity_fractions": fractions,
                "snapshots": snapshots,
                "interventions": interventions,
            }
        )
    return summarize("trained_td3_3000ep", episodes), episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripted-episodes", type=int, default=100)
    parser.add_argument("--trained-episodes", type=int, default=50)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    if args.scripted_episodes < 1 or args.trained_episodes < 1:
        parser.error("episode counts must be positive")

    scripted, _ = rollout_scripted(args.scripted_episodes)
    trained, _ = rollout_trained(args.trained_episodes, args.checkpoint)
    output = {
        "config": {
            "task": "tabletop",
            "curriculum_stage": 0,
            "n_obstacles": 0,
            "obstacle_speed": 0.0,
            "action_mode": "velocity",
            "reward_mode": "uoar",
            "max_steps": DynArmEnv.max_steps,
            "eval_seed": EVAL_SEED,
            "checkpoint": str(args.checkpoint),
            "dq_max": DQ_MAX.tolist(),
        },
        "scripted": scripted,
        "trained": trained,
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
