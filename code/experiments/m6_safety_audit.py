"""Phase 5j: evaluation-only audit of kinodynamic safety guarantees.

The six committed checkpoints predate later velocity-observation additions.
Their actor input width is recovered from the saved weights, and the current
environment observation is projected back to the matching historical layout.
This changes only what the frozen actor sees; the current environment and
shield execute every audited transition.

Usage: PYTHONPATH=. python experiments/m6_safety_audit.py [--episodes 30]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from godynur.env import DynArmEnv  # noqa: E402
from godynur.td3 import TD3  # noqa: E402


RESULT_ROOT = Path(__file__).resolve().parent / "results"
RUNS = (
    ("800ep", RESULT_ROOT / "m6_kinodynamic"),
    ("3000ep", RESULT_ROOT / "m6_kinodynamic_longrun"),
)
AUDIT_KEYS = (
    "endpoint_violation",
    "inter_sample_violation",
    "shield_emergency",
    "shield_fallback",
)


@dataclass(frozen=True)
class CheckpointSpec:
    label: str
    path: Path
    seed: int
    stage_idx: int
    stage: tuple[int, float]


class CheckpointObservationEnv(DynArmEnv):
    """DynArmEnv with the observation layout used by a saved actor."""

    def __init__(self, checkpoint_state_dim: int, **kwargs):
        self._checkpoint_state_dim = int(checkpoint_state_dim)
        super().__init__(**kwargs)
        self._current_state_dim = self.state_dim
        self.state_dim = self._checkpoint_state_dim

    def _state(self) -> np.ndarray:
        state = super()._state()
        if state.size == self._checkpoint_state_dim:
            return state

        # Current velocity suffix: v(7), a(7), margin+(7), margin-(7),
        # terminal-membership(1), intervention-norm(1). Historical saved
        # actors used either a+flags (44 values total here) or v+a+flags.
        base_dim = state.size - 30
        base = state[:base_dim]
        velocity = state[base_dim : base_dim + 7]
        acceleration = state[base_dim + 7 : base_dim + 14]
        flags = state[-2:]
        layouts = {
            base_dim + 9: np.concatenate((base, acceleration, flags)),
            base_dim + 16: np.concatenate(
                ((base, velocity, acceleration, flags))
            ),
        }
        projected = layouts.get(self._checkpoint_state_dim)
        if projected is None:
            raise ValueError(
                "unsupported checkpoint observation width "
                f"{self._checkpoint_state_dim}; current width is {state.size}"
            )
        return projected.astype(np.float32, copy=False)


def _seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)\.pt$", path.name)
    if match is None:
        raise ValueError(f"cannot recover seed from {path}")
    return int(match.group(1))


def _run_metadata(run_dir: Path, seed: int) -> tuple[int, tuple[int, float]]:
    json_path = run_dir / "m6_uoar_kinodynamic_shield_salt0.json"
    payload = json.loads(json_path.read_text())
    histories = payload["results"]["kinodynamic_shield"]
    for history in histories.values():
        if int(history[-1]["actual_seed"]) == seed:
            stage_idx = int(history[-1]["stage"])
            n_obstacles, speed = payload["stages"][stage_idx]
            return stage_idx, (int(n_obstacles), float(speed))
    raise ValueError(f"seed {seed} is absent from {json_path}")


def discover_checkpoints() -> list[CheckpointSpec]:
    specs = []
    for run_label, run_dir in RUNS:
        paths = sorted(run_dir.glob("td3_kinodynamic_shield_seed*.pt"))
        for path in paths:
            seed = _seed_from_path(path)
            stage_idx, stage = _run_metadata(run_dir, seed)
            specs.append(
                CheckpointSpec(
                    label=f"{run_label}/seed{seed}",
                    path=path,
                    seed=seed,
                    stage_idx=stage_idx,
                    stage=stage,
                )
            )
    return specs


def _checkpoint_dimensions(actor_state: dict[str, torch.Tensor]) -> tuple[int, int]:
    linear_weights = [
        value for key, value in actor_state.items()
        if key.endswith(".weight") and value.ndim == 2
    ]
    if not linear_weights:
        raise ValueError("actor state contains no linear weights")
    return int(linear_weights[0].shape[1]), int(linear_weights[-1].shape[0])


def audit_checkpoint(spec: CheckpointSpec, episodes: int) -> dict[str, object]:
    checkpoint = torch.load(spec.path, map_location="cpu", weights_only=True)
    actor_state = checkpoint["actor"]
    state_dim, action_dim = _checkpoint_dimensions(actor_state)
    env = CheckpointObservationEnv(
        checkpoint_state_dim=state_dim,
        task="tabletop",
        n_obstacles=3,
        reward_mode="uoar",
        seed=10_000 + spec.seed,
        action_mode="velocity",
    )
    if env.action_dim != action_dim:
        raise ValueError(
            f"{spec.path}: actor action width {action_dim} != env {env.action_dim}"
        )
    agent = TD3(
        env.state_dim,
        env.action_dim,
        action_scale=env.action_bound[1],
        expl_noise=0.25,
        seed=spec.seed,
    )
    agent.actor.load_state_dict(actor_state)
    agent.actor.eval()
    env.set_difficulty(*spec.stage)

    steps = 0
    counts_before = {key: int(env.stats.get(key, 0)) for key in AUDIT_KEYS}
    with torch.no_grad():
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = agent.act(state, explore=False)
                state, _, done = env.step(action)
                steps += 1
    counts = {
        key: int(env.stats.get(key, 0)) - counts_before[key]
        for key in AUDIT_KEYS
    }
    return {
        "checkpoint": spec.label,
        "path": str(spec.path),
        "episodes": episodes,
        "steps": steps,
        "stage_idx": spec.stage_idx,
        "stage": list(spec.stage),
        "checkpoint_state_dim": state_dim,
        "counts": counts,
        "rates": {key: counts[key] / steps for key in AUDIT_KEYS},
    }


def _print_table(results: list[dict[str, object]]) -> None:
    print(
        "| checkpoint | stage | steps | endpoint violation | "
        "inter-sample violation | emergency | fallback |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        rates = result["rates"]
        print(
            f"| {result['checkpoint']} | {result['stage_idx']} | "
            f"{result['steps']} | {rates['endpoint_violation']:.6e} | "
            f"{rates['inter_sample_violation']:.6e} | "
            f"{rates['shield_emergency']:.6e} | "
            f"{rates['shield_fallback']:.6e} |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    specs = discover_checkpoints()
    if len(specs) != 6:
        raise RuntimeError(f"expected 6 checkpoints, found {len(specs)}")
    results = []
    for spec in specs:
        print(
            f"auditing {spec.label} at reached stage {spec.stage_idx} "
            f"{spec.stage} ...",
            flush=True,
        )
        result = audit_checkpoint(spec, args.episodes)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    print()
    _print_table(results)
    if any(
        result["counts"][key] > 0
        for result in results
        for key in ("endpoint_violation", "inter_sample_violation")
    ):
        print("WARNING: the safety audit found a certified-bound violation.")


if __name__ == "__main__":
    main()
