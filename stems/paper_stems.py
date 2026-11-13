from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from stems.baselines import (_episode_arrays, _gae, _MLP, _ObsNorm, _PPOActor,
                             RunningNormalizer)
from stems.encoder import STEncoder

__all__ = ["PaperSTEMS"]


class PaperSTEMS:

    def __init__(self, obs_dim: int, action_dim: int, num_buildings: int,
                 adjacency: np.ndarray,
                 shield: Optional[Any] = None,
                 spatial_dim: int = 64, gcn_layers: int = 3,
                 temporal_dim: int = 32, num_heads: int = 4,
                 window_size: int = 24, repr_dim: int = 64,
                 hidden_dim: int = 128, lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_eps: Optional[float] = None,
                 epochs: int = 1, minibatch_size: int = 256,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5, scale_returns: bool = True,
                 device: str = "cpu") -> None:
        self.B = int(num_buildings)
        self.obs_dim, self.action_dim = int(obs_dim), int(action_dim)
        self.window_size = int(window_size)
        self.gamma, self.lam = float(gamma), float(gae_lambda)
        self.clip_eps = None if clip_eps is None else float(clip_eps)
        self.epochs, self.minibatch_size = int(epochs), int(minibatch_size)
        self.value_coef, self.entropy_coef = float(value_coef), float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.scale_returns = bool(scale_returns)
        self.device = torch.device(device)
        self.shield = shield

        adj = torch.as_tensor(np.asarray(adjacency, dtype=np.float32))
        self.register_adjacency(adj)

        self.encoder = STEncoder(
            obs_dim=self.obs_dim, spatial_dim=spatial_dim, temporal_dim=temporal_dim,
            output_dim=repr_dim, gcn_num_layers=gcn_layers, num_heads=num_heads,
            window_size=self.window_size).to(self.device)
        self.actors = nn.ModuleList(
            [_PPOActor(repr_dim, hidden_dim, self.action_dim)
             for _ in range(self.B)]).to(self.device)
        self.critics = nn.ModuleList(
            [_MLP(repr_dim, hidden_dim, 1) for _ in range(self.B)]).to(self.device)

        self.actor_opt = optim.Adam(self.actors.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critics.parameters(), lr=lr)
        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=lr)

        self.obs_norm = _ObsNorm(self.obs_dim).to(self.device)
        self.return_normalizer = RunningNormalizer(1).to(self.device)
        self._last_pre_tanh = np.zeros((self.B, self.action_dim), dtype=np.float32)
        self._last_log_probs = np.zeros(self.B, dtype=np.float32)
        self._last_nominal_actions: Optional[np.ndarray] = None

    def register_adjacency(self, adj: torch.Tensor) -> None:
        self.adj = adj.to(getattr(self, "device", "cpu"))

    def select_action(self, obs_list, history=None, explore: bool = True) -> np.ndarray:
        obs = torch.as_tensor(np.asarray(obs_list, dtype=np.float32),
                              device=self.device)
        hist = (torch.as_tensor(np.asarray(history, dtype=np.float32),
                                device=self.device)
                if history is not None else obs.unsqueeze(1).expand(
                    -1, self.window_size, -1).contiguous())
        with torch.no_grad():
            r = self._represent(obs.unsqueeze(0), hist.unsqueeze(0))[0]
            raw = np.zeros((self.B, self.action_dim), dtype=np.float32)
            z = np.zeros_like(raw)
            for i in range(self.B):
                dist = self.actors[i].distribution(r[i:i + 1])
                zi = dist.sample() if explore else dist.mean
                z[i] = zi.squeeze(0).cpu().numpy()
                raw[i] = torch.tanh(zi).squeeze(0).cpu().numpy()

        self._last_nominal_actions = raw.copy()
        safe = (raw if self.shield is None
                else np.asarray(self.shield.project(raw, list(obs_list)),
                                dtype=np.float32))

        z_safe = np.arctanh(np.clip(safe, -1.0 + 1e-6, 1.0 - 1e-6)).astype(np.float32)
        with torch.no_grad():
            zt = torch.as_tensor(z_safe, device=self.device)
            for i in range(self.B):
                self._last_log_probs[i] = float(
                    self.actors[i].distribution(r[i:i + 1])
                    .log_prob(zt[i:i + 1]).sum())
        self._last_pre_tanh = z_safe
        return safe.astype(np.float32)

    def _represent(self, obs_nb: torch.Tensor, hist_nb: torch.Tensor) -> torch.Tensor:
        return self.encoder.batch_forward(self.obs_norm(obs_nb), self.adj,
                                          self.obs_norm(hist_nb))

    def _windows(self, obs_nb: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        return obs_nb[rows].transpose(1, 2).contiguous()

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}
        arr = _episode_arrays(batch)
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        raw_obs, raw_next = t(arr["obs"]), t(arr["next_obs"])
        rewards, done = t(arr["rewards"]), t(arr["dones"])
        N = raw_obs.shape[0]

        if batch.get("pre_tanh") is not None and batch.get("behaviour_log_probs") is not None:
            z_all, old_logp = t(batch["pre_tanh"]), t(batch["behaviour_log_probs"])
        else:
            z_all = torch.atanh(t(arr["actions"]).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
            old_logp = None

        idx = (torch.arange(N, device=self.device).unsqueeze(1)
               - torch.arange(self.window_size - 1, -1, -1, device=self.device))
        self._win_idx = idx.clamp_min(0)

        with torch.no_grad():
            if self.scale_returns:
                ret, running = torch.zeros_like(rewards), torch.zeros(self.B, device=self.device)
                for k in reversed(range(N)):
                    running = rewards[k] + self.gamma * (1.0 - done[k]) * running
                    ret[k] = running
                self.return_normalizer.update(ret.reshape(-1, 1))
                scale = float(self.return_normalizer.var.sqrt().clamp_min(1e-6))
            else:
                scale = 1.0
            r_all = self._represent(raw_obs, self._windows(raw_obs, self._win_idx))
            r_next = self._represent(raw_next, self._windows(raw_next, self._win_idx))
            values = torch.stack([self.critics[b](r_all[:, b]).squeeze(-1)
                                  for b in range(self.B)], 1)
            next_values = torch.stack([self.critics[b](r_next[:, b]).squeeze(-1)
                                       for b in range(self.B)], 1)
            adv = _gae(rewards / scale, values, next_values, done, self.gamma, self.lam)
            returns = adv + values
            adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)
            if old_logp is None:
                old_logp = torch.stack(
                    [self.actors[b].distribution(r_all[:, b]).log_prob(z_all[:, b]).sum(-1)
                     for b in range(self.B)], 1)

        mb = max(2, min(self.minibatch_size, N))
        stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "n": 0.0}
        for _ in range(self.epochs):
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb):
                sel = perm[start:start + mb]
                if sel.numel() < 2:
                    continue
                r_mb = self._represent(raw_obs[sel],
                                       self._windows(raw_obs, self._win_idx[sel]))
                policy_loss = r_mb.new_zeros(())
                value_loss = r_mb.new_zeros(())
                entropy = r_mb.new_zeros(())
                for b in range(self.B):
                    dist = self.actors[b].distribution(r_mb[:, b])
                    logp = dist.log_prob(z_all[sel, b]).sum(-1)
                    a_b = adv[sel, b]
                    if self.clip_eps is None:
                        policy_loss = policy_loss - (logp * a_b).mean() / self.B
                    else:
                        ratio = (logp - old_logp[sel, b]).clamp(-20.0, 20.0).exp()
                        policy_loss = policy_loss - torch.min(
                            ratio * a_b,
                            ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * a_b
                        ).mean() / self.B
                    v = self.critics[b](r_mb[:, b]).squeeze(-1)
                    value_loss = value_loss + ((v - returns[sel, b]) ** 2).mean() / self.B
                    entropy = entropy + dist.entropy().sum(-1).mean() / self.B

                total = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.actor_opt.zero_grad(); self.critic_opt.zero_grad()
                self.encoder_opt.zero_grad()
                total.backward()
                for mod in (self.actors, self.critics, self.encoder):
                    nn.utils.clip_grad_norm_(mod.parameters(), self.max_grad_norm)
                self.actor_opt.step(); self.critic_opt.step(); self.encoder_opt.step()
                stats["policy"] += float(policy_loss.detach())
                stats["value"] += float(value_loss.detach())
                stats["entropy"] += float(entropy.detach())
                stats["n"] += 1.0

        n = max(stats.pop("n"), 1.0)
        self.obs_norm(raw_obs)
        return {k: v / n for k, v in stats.items()}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({"encoder": self.encoder.state_dict(),
                    "actors": self.actors.state_dict(),
                    "critics": self.critics.state_dict(),
                    "obs_norm": self.obs_norm.state_dict(),
                    "adj": self.adj.cpu()},
                   os.path.join(path, "paper_stems.pt"))

    def load(self, path: str) -> None:
        d = torch.load(os.path.join(path, "paper_stems.pt"),
                       map_location=self.device, weights_only=False)
        self.encoder.load_state_dict(d["encoder"])
        self.actors.load_state_dict(d["actors"])
        self.critics.load_state_dict(d["critics"])
        self.obs_norm.load_state_dict(d["obs_norm"])
        self.register_adjacency(d["adj"])
