"""GridTD3 — TD3 whose critic/actor consume [proprioceptive vector; grid
latent z], with the occupancy-forecasting auxiliary task trained JOINTLY.

This is the M2-full keystone integration (blueprint §systems + kill-test
conditions): the encoder (OccupancyForecaster trunk) is updated by
critic loss + lambda_aux * forecast BCE; the actor sees z.detach() (SAC-AE
/ DrQ convention). A target encoder (EMA) provides z for TD targets.

Replay stores obstacle-parameter snapshots (9 floats per obstacle slot per
frame) instead of voxel grids — ~4000x smaller — and re-rasterizes lazily
on sampling via the analytic voxelizer (0.074 ms/grid). The forecast label
is the TRUE occupancy at t + horizon (recorded h steps later during
rollout, wall reflections included), masked out when the episode ended
before t + horizon.

Joint-vs-frozen matters: the closest prior work (arXiv:2508.20457) freezes
its occupancy predictor before RL; our H2 claim is specifically about the
jointly-trained auxiliary signal. m5_grid.py ablates aux on/off.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .forecast import OccupancyForecaster, forecast_loss
from .geometry import AABB
from .scenes import WS_HI, WS_LO
from .td3 import mlp
from .voxelizer import GridSpec, rasterize

_SPEC = GridSpec(lo=WS_LO, hi=WS_HI, n=32)
_DUMMY_THRESHOLD = 10.0  # slots with |center| beyond this rasterize to nothing


def params_to_grid(frame_params: np.ndarray) -> np.ndarray:
    """(n_slots, 9) obstacle params -> (32, 32, 32) occupancy grid."""
    boxes = []
    for row in frame_params:
        c, h = row[0:3], row[3:6]
        if np.abs(c).max() > _DUMMY_THRESHOLD:
            continue
        boxes.append(AABB(c - h, c + h))
    return rasterize(boxes, _SPEC)


def rasterize_batch(params: np.ndarray) -> np.ndarray:
    """(B, k, n_slots, 9) -> (B, k, 32, 32, 32) float32."""
    B, k = params.shape[:2]
    out = np.empty((B, k, _SPEC.n, _SPEC.n, _SPEC.n), dtype=np.float32)
    for b in range(B):
        for j in range(k):
            out[b, j] = params_to_grid(params[b, j])
    return out


class GridReplayBuffer:
    """Stores vector transitions + obstacle-parameter snapshots."""

    def __init__(self, vec_dim, action_dim, k_frames, n_slots, capacity=100_000):
        self.vec = np.zeros((capacity, vec_dim), np.float32)
        self.a = np.zeros((capacity, action_dim), np.float32)
        self.r = np.zeros((capacity, 1), np.float32)
        self.vec2 = np.zeros((capacity, vec_dim), np.float32)
        self.d = np.zeros((capacity, 1), np.float32)
        self.hist = np.zeros((capacity, k_frames, n_slots, 9), np.float32)
        self.hist2 = np.zeros((capacity, k_frames, n_slots, 9), np.float32)
        self.fut = np.zeros((capacity, 1, n_slots, 9), np.float32)
        self.fut_mask = np.zeros((capacity, 1), np.float32)
        self.capacity, self.idx, self.full = capacity, 0, False

    def add(self, vec, a, r, vec2, d, hist, hist2, fut, fut_valid):
        i = self.idx
        self.vec[i], self.a[i], self.r[i] = vec, a, r
        self.vec2[i], self.d[i] = vec2, d
        self.hist[i], self.hist2[i] = hist, hist2
        if fut is not None:
            self.fut[i, 0] = fut
        self.fut_mask[i] = float(fut_valid)
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch):
        n = len(self)
        j = np.random.randint(0, n, batch)
        return (
            self.vec[j], self.a[j], self.r[j], self.vec2[j], self.d[j],
            self.hist[j], self.hist2[j], self.fut[j], self.fut_mask[j],
        )


class GridTD3:
    def __init__(
        self,
        vec_dim: int,
        action_dim: int,
        action_scale: float,
        k_frames: int = 3,
        n_slots: int = 3,
        latent_dim: int = 256,
        lambda_aux: float = 1.0,  # 0 => the no-aux ablation arm (H2 control)
        gamma: float = 0.98,
        tau: float = 0.01,
        lr: float = 1e-3,
        policy_delay: int = 2,
        expl_noise: float = 0.25,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        hidden: int = 256,
        device: str | None = None,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.action_scale = action_scale
        self.gamma, self.tau, self.policy_delay = gamma, tau, policy_delay
        self.expl_noise, self.target_noise = expl_noise, target_noise
        self.noise_clip, self.lambda_aux = noise_clip, lambda_aux

        d = self.device
        self.enc = OccupancyForecaster(k_frames, latent_dim).to(d)
        self.enc_t = OccupancyForecaster(k_frames, latent_dim).to(d)
        self.enc_t.load_state_dict(self.enc.state_dict())
        sd = vec_dim + latent_dim
        self.actor = mlp([sd, hidden, hidden, action_dim], nn.Tanh()).to(d)
        self.actor_t = mlp([sd, hidden, hidden, action_dim], nn.Tanh()).to(d)
        self.q1 = mlp([sd + action_dim, hidden, hidden, 1]).to(d)
        self.q2 = mlp([sd + action_dim, hidden, hidden, 1]).to(d)
        self.q1_t = mlp([sd + action_dim, hidden, hidden, 1]).to(d)
        self.q2_t = mlp([sd + action_dim, hidden, hidden, 1]).to(d)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(
            list(self.q1.parameters())
            + list(self.q2.parameters())
            + list(self.enc.parameters()),
            lr=lr,
        )
        self.buffer = GridReplayBuffer(vec_dim, action_dim, k_frames, n_slots)
        self.it = 0
        self.last_aux_loss = float("nan")

    @torch.no_grad()
    def act(self, vec: np.ndarray, hist_params: np.ndarray, explore=True):
        g = torch.from_numpy(rasterize_batch(hist_params[None])).to(self.device)
        z, _ = self.enc(g)
        v = torch.from_numpy(vec[None].astype(np.float32)).to(self.device)
        a = self.actor(torch.cat([v, z], 1))[0].cpu().numpy()
        if explore:
            a = a + np.random.normal(0, self.expl_noise, a.shape)
        return np.clip(a, -1, 1) * self.action_scale

    def learn(self, batch: int = 64):
        if len(self.buffer) < 5 * batch:
            return
        vec, a, r, vec2, dn, hist, hist2, fut, fmask = self.buffer.sample(batch)
        d = self.device
        t = lambda x: torch.from_numpy(x).to(d)  # noqa: E731
        vec, a, r, vec2, dn, fmask = map(t, (vec, a, r, vec2, dn, fmask))
        a = a / self.action_scale
        g = t(rasterize_batch(hist))
        g2 = t(rasterize_batch(hist2))
        fut_grid = t(rasterize_batch(fut))[:, 0]

        z, logits = self.enc(g)
        with torch.no_grad():
            z2, _ = self.enc_t(g2)
            noise = (torch.randn_like(a) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            s2 = torch.cat([vec2, z2], 1)
            a2 = (self.actor_t(s2) + noise).clamp(-1, 1)
            q_t = torch.min(
                self.q1_t(torch.cat([s2, a2], 1)), self.q2_t(torch.cat([s2, a2], 1))
            )
            y = r + self.gamma * (1 - dn) * q_t
        s = torch.cat([vec, z], 1)
        sa = torch.cat([s, a], 1)
        loss = nn.functional.mse_loss(self.q1(sa), y) + nn.functional.mse_loss(
            self.q2(sa), y
        )
        if self.lambda_aux > 0 and fmask.sum() > 0:
            per_vox = nn.functional.binary_cross_entropy_with_logits(
                logits, fut_grid, pos_weight=torch.tensor(20.0, device=d),
                reduction="none",
            ).mean(dim=(1, 2, 3), keepdim=True)
            aux = (per_vox * fmask).sum() / fmask.sum()
            self.last_aux_loss = float(aux)
            loss = loss + self.lambda_aux * aux
        self.opt_c.zero_grad()
        loss.backward()
        self.opt_c.step()

        self.it += 1
        if self.it % self.policy_delay == 0:
            s_det = torch.cat([vec, z.detach()], 1)
            loss_a = -self.q1(torch.cat([s_det, self.actor(s_det)], 1)).mean()
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()
            for net, tgt in [
                (self.actor, self.actor_t), (self.q1, self.q1_t),
                (self.q2, self.q2_t), (self.enc, self.enc_t),
            ]:
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
