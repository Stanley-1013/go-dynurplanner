"""Compact TD3 for DynArmEnv baselines (vector state).

Deliberately standard (Fujimoto et al. 2018): twin critics, delayed policy
updates, target policy smoothing. This is the plain-RL reference point for
the H3 experiment (reward-mode comparison) and later the backbone that
APE2's candidate exploration / hybrid evaluation plugs into, mirroring how
URPlanner layers APE2 on deterministic-PG algorithms.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def mlp(sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if out_act is not None:
        layers.append(out_act)
    return nn.Sequential(*layers)


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=200_000):
        self.s = np.zeros((capacity, state_dim), np.float32)
        self.a = np.zeros((capacity, action_dim), np.float32)
        self.r = np.zeros((capacity, 1), np.float32)
        self.s2 = np.zeros((capacity, state_dim), np.float32)
        self.d = np.zeros((capacity, 1), np.float32)
        self.capacity, self.idx, self.full = capacity, 0, False

    def add(self, s, a, r, s2, d):
        i = self.idx
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.idx = (i + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch):
        n = self.capacity if self.full else self.idx
        j = np.random.randint(0, n, batch)
        t = torch.from_numpy
        return t(self.s[j]), t(self.a[j]), t(self.r[j]), t(self.s2[j]), t(self.d[j])

    def __len__(self):
        return self.capacity if self.full else self.idx


class TD3:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_scale: float,
        gamma: float = 0.98,  # URPlanner Table I: xi = 0.98
        tau: float = 0.01,  # URPlanner Table I: soft update 0.01
        lr: float = 1e-3,  # URPlanner Table I
        policy_delay: int = 2,
        expl_noise: float = 0.15,
        target_noise: float = 0.2,
        noise_clip: float = 0.5,
        hidden: int = 256,
        seed: int = 0,
        use_action_ste: bool = False,
    ):
        torch.manual_seed(seed)
        self.action_scale = action_scale
        self.gamma, self.tau, self.policy_delay = gamma, tau, policy_delay
        self.expl_noise, self.target_noise = expl_noise, target_noise
        self.noise_clip = noise_clip
        # Straight-through actor update (default off, byte-identical to
        # standard DPG otherwise): when a hard action filter/shield sits
        # inside env.step() and the replay buffer stores the ACTUALLY
        # EXECUTED action (not the raw policy proposal), the standard
        # actor loss -Q(s, actor(s)) queries Q at raw actions the critic
        # may have rarely or never seen executed in filter-heavy regions
        # (extrapolation/OOD-action error, cf. Fujimoto et al. BCQ 2019).
        # With this on, the actor loss is evaluated at the buffer's
        # on-distribution executed action in the forward pass, while the
        # gradient still flows to the raw actor output in the backward
        # pass (classic straight-through estimator) -- the actor gets a
        # locally-correct gradient computed where the critic is actually
        # accurate, instead of extrapolating.
        self.use_action_ste = use_action_ste

        self.actor = mlp([state_dim, hidden, hidden, action_dim], nn.Tanh())
        self.actor_t = mlp([state_dim, hidden, hidden, action_dim], nn.Tanh())
        self.q1 = mlp([state_dim + action_dim, hidden, hidden, 1])
        self.q2 = mlp([state_dim + action_dim, hidden, hidden, 1])
        self.q1_t = mlp([state_dim + action_dim, hidden, hidden, 1])
        self.q2_t = mlp([state_dim + action_dim, hidden, hidden, 1])
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.q1_t.load_state_dict(self.q1.state_dict())
        self.q2_t.load_state_dict(self.q2.state_dict())
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.buffer = ReplayBuffer(state_dim, action_dim)
        self.it = 0

    @torch.no_grad()
    def act(self, s: np.ndarray, explore: bool = True) -> np.ndarray:
        a = self.actor(torch.from_numpy(s[None].astype(np.float32)))[0].numpy()
        if explore:
            a = a + np.random.normal(0, self.expl_noise, a.shape)
        return np.clip(a, -1, 1) * self.action_scale

    def learn(self, batch: int = 64):  # URPlanner Table I: B = 64
        if len(self.buffer) < 5 * batch:
            return
        s, a, r, s2, d = self.buffer.sample(batch)
        a = a / self.action_scale  # store env-scale, learn in [-1,1]

        with torch.no_grad():
            noise = (torch.randn_like(a) * self.target_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            a2 = (self.actor_t(s2) + noise).clamp(-1, 1)
            q_t = torch.min(
                self.q1_t(torch.cat([s2, a2], 1)), self.q2_t(torch.cat([s2, a2], 1))
            )
            y = r + self.gamma * (1 - d) * q_t
        sa = torch.cat([s, a], 1)
        loss_c = nn.functional.mse_loss(self.q1(sa), y) + nn.functional.mse_loss(
            self.q2(sa), y
        )
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

        self.it += 1
        if self.it % self.policy_delay == 0:
            raw_action = self.actor(s)
            if self.use_action_ste:
                # Forward value is exactly the buffered executed action
                # (on-distribution for the critic); gradient flows to
                # raw_action unchanged (the detached term contributes 0).
                actor_action = raw_action + (a - raw_action).detach()
            else:
                actor_action = raw_action
            loss_a = -self.q1(torch.cat([s, actor_action], 1)).mean()
            self.opt_a.zero_grad()
            loss_a.backward()
            self.opt_a.step()
            for net, tgt in [
                (self.actor, self.actor_t), (self.q1, self.q1_t), (self.q2, self.q2_t)
            ]:
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.data.mul_(1 - self.tau).add_(self.tau * p.data)
