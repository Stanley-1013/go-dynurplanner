"""M2-pre — keystone kill test: can occupancy forecasting extract motion?

The Loop-6 keystone claims a forecasting auxiliary task forces obstacle
MOTION information into the encoder latent. This experiment tests exactly
that mechanism, standalone, before any RL is built on it:

  train : k-frame grid history -> predict grid at t + dt   (labels free via
          analytic rasterization of the parameterized space)
  eval  : IoU of predicted vs true future occupancy, stratified by obstacle
          speed, against two controls:
            (1) persistence baseline  — copy the last observed frame
            (2) k=1 forecaster        — velocity-blind by construction

Kill criterion: if the k=3 forecaster does not beat BOTH controls at
moderate/high speeds, grid history does not carry recoverable motion signal
at this resolution, and the keystone (blueprint H2 mechanism) dies here —
cheaply, before M2's RL integration.

Usage: .venv/bin/python experiments/m2_forecast_feasibility.py [--steps 3000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from godynur.forecast import OccupancyForecaster, forecast_loss, occupancy_iou  # noqa: E402
from godynur.scenes import WS_HI, WS_LO, sample_scene  # noqa: E402
from godynur.voxelizer import GridSpec, rasterize, rasterize_future  # noqa: E402

DT = 0.05
K_FRAMES = 3
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0]
SPEC = GridSpec(lo=WS_LO, hi=WS_HI, n=32)


def make_sample(rng: np.random.Generator, speed: float, horizon: int):
    """One (history, future) pair. `horizon` = forecast distance in DT steps.

    v2: the v1 kill test forecast a single DT (50 ms) ahead — at 0.5 m/s an
    obstacle moves half a voxel per step, so persistence is near-optimal BY
    PHYSICS and the test measured the wrong thing. The auxiliary task should
    forecast at the anticipation horizon relevant for avoidance (~0.2 s).
    """
    scene = sample_scene(rng, n_obstacles=3, speed=speed)
    # Warm up the scene so histories are not aligned to spawn time.
    for _ in range(rng.integers(0, 10)):
        scene.step(DT)
    frames = []
    for _ in range(K_FRAMES):
        frames.append(rasterize(scene.static_aabbs(), SPEC))
        scene.step(DT)
    # Scene now sits one DT after the last frame; advance the remaining
    # (horizon - 1) steps so the label is `horizon` DTs after the last frame.
    for _ in range(horizon - 1):
        scene.step(DT)
    future = rasterize(scene.static_aabbs(), SPEC)
    return np.stack(frames), future


def batch(rng, batch_size, speed_choices, horizon):
    xs, ys, last = [], [], []
    for _ in range(batch_size):
        v = float(rng.choice(speed_choices))
        h, f = make_sample(rng, v, horizon)
        xs.append(h)
        ys.append(f)
        last.append(h[-1])
    return (
        torch.from_numpy(np.stack(xs)),
        torch.from_numpy(np.stack(ys)),
        torch.from_numpy(np.stack(last)),
    )


def train(model, rng, steps, batch_size, lr, k_frames, log_prefix, horizon):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for it in range(steps):
        x, y, _ = batch(rng, batch_size, SPEEDS, horizon)
        if k_frames == 1:
            x = x[:, -1:].contiguous()
        _, logits = model(x)
        loss = forecast_loss(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (it + 1) % max(1, steps // 10) == 0:
            print(
                f"{log_prefix} step {it + 1}/{steps} loss {loss.item():.4f} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    return model


@torch.no_grad()
def evaluate(model_k3, model_k1, rng, n_eval_per_speed, horizon):
    out = {}
    for v in SPEEDS:
        ious = {"k3": [], "k1": [], "persist": []}
        for _ in range(n_eval_per_speed):
            x, y, last = batch(rng, 1, [v], horizon)
            _, lg3 = model_k3(x)
            ious["k3"].append(occupancy_iou(lg3, y).item())
            _, lg1 = model_k1(x[:, -1:].contiguous())
            ious["k1"].append(occupancy_iou(lg1, y).item())
            inter = (last * y).sum()
            union = ((last + y) > 0).float().sum()
            ious["persist"].append((inter / union).item() if union > 0 else 1.0)
        out[str(v)] = {k: float(np.mean(vv)) for k, vv in ious.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-n", type=int, default=100)
    ap.add_argument("--horizon", type=int, default=4,
                    help="forecast distance in DT steps (4 = 0.2 s)")
    ap.add_argument("--out", type=str, default="experiments/results/m2_forecast")
    args = ap.parse_args()

    torch.manual_seed(0)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("training k=3 forecaster...", flush=True)
    m3 = train(
        OccupancyForecaster(k_frames=K_FRAMES),
        np.random.default_rng(0), args.steps, args.batch, args.lr, 3, "k3",
        args.horizon,
    )
    print("training k=1 (velocity-blind) control...", flush=True)
    m1 = train(
        OccupancyForecaster(k_frames=1),
        np.random.default_rng(1), args.steps, args.batch, args.lr, 1, "k1",
        args.horizon,
    )

    print("evaluating...", flush=True)
    results = evaluate(m3, m1, np.random.default_rng(42), args.eval_n, args.horizon)
    for v in SPEEDS:
        r = results[str(v)]
        print(
            f"speed {v:>4}: IoU k3={r['k3']:.3f}  k1={r['k1']:.3f}  "
            f"persist={r['persist']:.3f}  "
            f"{'PASS' if r['k3'] > max(r['k1'], r['persist']) else 'FAIL'}",
            flush=True,
        )

    torch.save(m3.state_dict(), out_dir / "forecaster_k3.pt")
    with open(out_dir / "m2_forecast_results.json", "w") as f:
        json.dump(
            {"config": vars(args) | {"dt": DT, "k_frames": K_FRAMES}, "iou": results},
            f,
            indent=2,
        )
    print(f"wrote {out_dir}/m2_forecast_results.json")


if __name__ == "__main__":
    main()
