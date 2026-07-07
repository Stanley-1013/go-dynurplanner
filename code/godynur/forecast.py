"""Occupancy encoder + forecasting auxiliary head (the Loop-6 keystone).

A small 3D-CNN encodes k stacked occupancy frames into (i) a latent vector
z_t for the future policy/critic and (ii) a spatial bottleneck from which a
transposed-conv decoder predicts the occupancy grid at t + dt. Predicting
the future is only possible by extracting motion from the frame history, so
the auxiliary loss forces velocity information into z_t — the structural
cure for the grid's velocity-blindness (Hart et al. 2112.12465).

Training labels are free: `voxelizer.rasterize_future` on the parameterized
space (no simulator, no sensor, no annotation).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OccupancyForecaster(nn.Module):
    """U-Net-style encoder-decoder over (B, k, N, N, N) occupancy stacks.

    v2 (post kill-test-v1): the v1 bottleneck decoder could not even
    reproduce the current frame (IoU 0.42 vs persistence 0.92 at low speed),
    confounding reconstruction capacity with motion prediction. v2 adds
    (i) U-Net skip connections so identity is cheaply representable, and
    (ii) the last input frame concatenated at full resolution before the
    output conv — the model starts from persistence and learns the delta.

    forward() returns (z, future_logits):
      z             : (B, latent_dim)   — policy/critic feature
      future_logits : (B, N, N, N)      — occupancy logits at t + horizon
    """

    def __init__(self, k_frames: int = 3, latent_dim: int = 256, n: int = 32):
        super().__init__()
        assert n % 8 == 0, "spatial dims must survive three stride-2 stages"
        self.n = n
        m = n // 8
        c1, c2, c3 = 16, 32, 64
        self.enc1 = nn.Sequential(
            nn.Conv3d(k_frames, c1, 3, stride=2, padding=1), nn.SiLU()
        )  # 32 -> 16
        self.enc2 = nn.Sequential(
            nn.Conv3d(c1, c2, 3, stride=2, padding=1), nn.SiLU()
        )  # 16 -> 8
        self.enc3 = nn.Sequential(
            nn.Conv3d(c2, c3, 3, stride=2, padding=1), nn.SiLU()
        )  # 8 -> 4
        self.to_latent = nn.Sequential(
            nn.Flatten(), nn.Linear(c3 * m * m * m, latent_dim), nn.SiLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(c3, c2, 4, stride=2, padding=1), nn.SiLU()
        )  # 4 -> 8
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(c2 + c2, c1, 4, stride=2, padding=1), nn.SiLU()
        )  # 8 -> 16
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(c1 + c1, c1, 4, stride=2, padding=1), nn.SiLU()
        )  # 16 -> 32
        # Final conv sees decoder features + the raw last frame (identity path).
        self.out = nn.Conv3d(c1 + 1, 1, 3, padding=1)

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        last = frames[:, -1:].contiguous()
        e1 = self.enc1(frames)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        z = self.to_latent(e3)
        d3 = self.dec3(e3)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        future_logits = self.out(torch.cat([d1, last], dim=1)).squeeze(1)
        return z, future_logits


def forecast_loss(
    logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 20.0
) -> torch.Tensor:
    """BCE with positive re-weighting (occupancy is sparse, ~2-5%)."""
    return nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=torch.tensor(pos_weight, device=logits.device),
    )


def forecast_loss_sdf(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Regression loss for SDF-mode grids (advisor directive): Huber on the
    signed distance values — BCE semantics do not apply to SDF."""
    return nn.functional.huber_loss(pred, target, delta=0.1)


@torch.no_grad()
def occupancy_iou(
    logits: torch.Tensor, target: torch.Tensor, thresh: float = 0.5
) -> torch.Tensor:
    """Mean IoU of thresholded prediction vs target, per batch element."""
    pred = (torch.sigmoid(logits) > thresh).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    return torch.where(union > 0, inter / union, torch.ones_like(union))
