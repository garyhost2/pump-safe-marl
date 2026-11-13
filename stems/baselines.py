from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from stems.environment import thermostat_step
from stems.observations import obs_indices
from stems.utils import RunningNormalizer

__all__ = ["RuleBasedAgent", "SingleAgentSAC", "IndependentSAC", "DMAPPOAgent", "MPCAgent", "MADDPGAgent",
           "MARLISAAgent", "MADCQAgent", "MetaEMSAgent"]

_IDX_HOUR, _IDX_PRICE, _IDX_SOC_ELEC, _IDX_T_IN, _IDX_LOAD, _IDX_SOLAR = obs_indices(
    "hour", "electricity_pricing", "electrical_storage_soc",
    "indoor_dry_bulb_temperature", "non_shiftable_load", "solar_generation")
_IDX_T_COOL, _IDX_T_HEAT, _IDX_NET = obs_indices(
    "indoor_dry_bulb_temperature_cooling_set_point",
    "indoor_dry_bulb_temperature_heating_set_point", "net_electricity_consumption")
_IDX_SOC_DHW, _IDX_DHW_DEMAND = obs_indices("dhw_storage_soc", "dhw_demand")


class RuleBasedAgent:
    CHARGE_HOURS = range(11, 17)
    PEAK_HOURS = range(17, 22)
    CHARGE_ACTION = 0.5
    DHW_CHARGE_ACTION = 0.3
    DHW_DISCHARGE_ACTION = -0.5

    def __init__(self, num_buildings: int = 3, hvac_control: str = "power",
                 battery_nominal_power: Optional[np.ndarray] = None,
                 has_hvac: bool = True) -> None:
        if hvac_control not in ("power", "setpoint"):
            raise ValueError(f"hvac_control must be 'power' or 'setpoint', got {hvac_control!r}")
        self.B = num_buildings
        self.hvac_control = hvac_control
        self.has_hvac = bool(has_hvac)
        self.p_batt = (None if battery_nominal_power is None
                       else np.asarray(battery_nominal_power, dtype=np.float32).reshape(-1))
        self.reset()

    def reset(self) -> None:
        self._u = np.zeros(self.B, dtype=np.float32)

    def notify_executed(self, executed: np.ndarray, hvac_idx: int = 2) -> None:
        if self.hvac_control == "power":
            self._u = np.asarray(executed, dtype=np.float32)[:, hvac_idx].copy()

    def _hvac_actions(self, obs_list: List[np.ndarray]) -> np.ndarray:
        if self.hvac_control == "setpoint" or not self.has_hvac:
            return np.zeros(self.B, dtype=np.float32)
        if any(len(o) <= _IDX_T_HEAT for o in obs_list):
            raise ValueError(
                "RuleBasedAgent's thermostat needs the heating set point, which is only "
                "observed with STEMSEnvironment(heat_pump=True).")
        col = lambda idx: np.array([o[idx] for o in obs_list], dtype=np.float32)
        self._u = thermostat_step(self._u, col(_IDX_T_IN), col(_IDX_T_HEAT), col(_IDX_T_COOL),
                                  np.zeros(self.B, dtype=np.float32))
        return self._u

    def select_action(
        self,
        obs_list: List[np.ndarray],
        history: Optional[np.ndarray] = None,
        explore: bool = False,
    ) -> np.ndarray:
        if self.p_batt is None:
            raise ValueError("RuleBasedAgent needs battery_nominal_power "
                             "(env.battery_info()['nominal_power']) for its "
                             "load-following discharge")
        actions = np.zeros((self.B, 3), dtype=np.float32)
        hvac = self._hvac_actions(obs_list)

        for i, obs in enumerate(obs_list):
            hour = int(round(float(obs[_IDX_HOUR])))
            if hour in self.CHARGE_HOURS:
                elec_action, dhw_action = self.CHARGE_ACTION, self.DHW_CHARGE_ACTION
            elif hour in self.PEAK_HOURS:
                residual = max(float(obs[_IDX_LOAD]) - float(obs[_IDX_SOLAR]), 0.0)
                elec_action = -min(residual / max(float(self.p_batt[i]), 1e-6), 1.0)
                dhw_action = self.DHW_DISCHARGE_ACTION
            else:
                elec_action, dhw_action = 0.0, 0.0
            actions[i] = [dhw_action, elec_action, hvac[i]]

        return actions

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


def _presquash_penalty(mean: "torch.Tensor", log_std: "torch.Tensor",
                       weight: float) -> "torch.Tensor":
    if weight <= 0.0:
        return torch.zeros((), device=mean.device, dtype=mean.dtype)
    return weight * (mean.pow(2).mean() + log_std.pow(2).mean())


class _MLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SquashedGaussian(nn.Module):

    LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.trunk(x)
        return self.mean_head(feat), torch.clamp(
            self.log_std_head(feat), self.LOG_STD_MIN, self.LOG_STD_MAX)

    def sample(self, x: torch.Tensor, deterministic: bool = False
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        action, log_prob, _, _ = self.sample_with_params(x, deterministic)
        return action, log_prob

    def sample_with_params(self, x: torch.Tensor, deterministic: bool = False
                           ) -> Tuple[torch.Tensor, torch.Tensor,
                                      torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(x)
        if deterministic:
            return (torch.tanh(mean),
                    torch.zeros(mean.shape[:-1], device=mean.device), mean, log_std)
        normal = torch.distributions.Normal(mean, log_std.exp())
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = (normal.log_prob(z)
                    - torch.log(1.0 - action.pow(2) + 1e-6)).sum(dim=-1)
        return action, log_prob, mean, log_std


class _FlatReplay:

    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = int(capacity)
        self._rows: List[Tuple[np.ndarray, ...]] = []
        self._cursor = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        row = (np.asarray(obs, dtype=np.float32), np.asarray(action, dtype=np.float32),
               np.asarray(reward, dtype=np.float32), np.asarray(next_obs, dtype=np.float32),
               np.float32(done))
        if len(self._rows) < self.capacity:
            self._rows.append(row)
        else:
            self._rows[self._cursor] = row
            self._cursor = (self._cursor + 1) % self.capacity

    def __len__(self) -> int:
        return len(self._rows)

    def sample(self, batch_size: int, device) -> Tuple[torch.Tensor, ...]:
        idx = random.sample(range(len(self._rows)), min(batch_size, len(self._rows)))
        cols = list(zip(*[self._rows[i] for i in idx]))
        to = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32, device=device)
        return (to(cols[0]), to(cols[1]), to(cols[2]), to(cols[3]),
                torch.as_tensor(np.array(cols[4]), dtype=torch.float32, device=device))


def _episode_arrays(batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
    obs = np.asarray(batch["obs"], dtype=np.float32)
    next_obs = np.asarray(batch["next_obs"], dtype=np.float32)
    actions = np.asarray(batch.get("raw_actions", batch["actions"]), dtype=np.float32)
    rewards = np.asarray(batch["rewards"], dtype=np.float32)
    dones = np.asarray(batch["dones"], dtype=np.float32)
    return {"obs": obs, "next_obs": next_obs, "actions": actions,
            "rewards": rewards, "dones": dones}


class _ObsNorm(nn.Module):

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = RunningNormalizer(int(dim))

    def fit(self, x: torch.Tensor) -> None:
        self.norm.update(x.reshape(-1, x.shape[-1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if int(self.norm.count) == 0:
            return x
        return self.norm(x)


def _soft_update(online: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(online.parameters(), target.parameters()):
            tp.mul_(1.0 - tau).add_(tau * p)


class _NoLearn:

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None


class SingleAgentSAC:

    def __init__(self, obs_dim: int, action_dim: int, num_buildings: int = 3,
                 hidden_dim: int = 256, lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, batch_size: int = 256,
                 updates_per_step: float = 1.0, start_steps: int = 256,
                 buffer_capacity: int = 200_000, autotune_alpha: bool = True,
                 alpha_entropy: float = 0.2, policy_reg: float = 0.0,
                 scale_returns: bool = False, device: str = "cpu") -> None:
        self.B, self.obs_dim, self.action_dim = int(num_buildings), int(obs_dim), int(action_dim)
        self.policy_reg = float(policy_reg)
        self.scale_returns = bool(scale_returns)
        self.return_normalizer = RunningNormalizer(1).to(torch.device(device))
        self._reward_scale = 1.0
        self.gamma, self.tau = float(gamma), float(tau)
        self.batch_size, self.start_steps = int(batch_size), int(start_steps)
        self.updates_per_step = float(updates_per_step)
        self.device = torch.device(device)

        joint_obs, joint_act = self.B * self.obs_dim, self.B * self.action_dim
        self.policy = _SquashedGaussian(joint_obs, hidden_dim, joint_act).to(self.device)
        self.q1 = _MLP(joint_obs + joint_act, hidden_dim, 1).to(self.device)
        self.q2 = _MLP(joint_obs + joint_act, hidden_dim, 1).to(self.device)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.autotune = bool(autotune_alpha)
        self.target_entropy = -float(joint_act)
        self.log_alpha = torch.tensor(float(np.log(alpha_entropy)), requires_grad=self.autotune,
                                      device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr) if self.autotune else None
        self.buffer = _FlatReplay(buffer_capacity)
        self.obs_norm = _ObsNorm(self.obs_dim).to(self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def _norm(self, flat: torch.Tensor) -> torch.Tensor:
        return self.obs_norm(flat.view(*flat.shape[:-1], self.B, self.obs_dim)
                             ).reshape(flat.shape)

    def select_action(self, obs_list, history=None, explore: bool = True) -> np.ndarray:
        x = torch.as_tensor(np.concatenate([np.asarray(o, dtype=np.float32) for o in obs_list]),
                            dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.policy.sample(self._norm(x), deterministic=not explore)
        return a.squeeze(0).cpu().numpy().reshape(self.B, self.action_dim).astype(np.float32)

    def _ingest(self, b: Dict[str, np.ndarray]) -> None:
        N = b["obs"].shape[0]
        for n in range(N):
            self.buffer.add(b["obs"][n].reshape(-1), b["actions"][n].reshape(-1),
                            float(b["rewards"][n].sum()), b["next_obs"][n].reshape(-1),
                            float(b["dones"][n]))

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}
        return self._update_from_arrays(_episode_arrays(batch))

    def _update_from_arrays(self, arr: Dict[str, np.ndarray]) -> Dict[str, float]:
        self.obs_norm.fit(torch.as_tensor(arr["obs"], device=self.device))
        self._ingest(arr)
        if self.scale_returns:
            r = torch.as_tensor(np.asarray(arr["rewards"], dtype=np.float32),
                                device=self.device)
            d = torch.as_tensor(np.asarray(arr["dones"], dtype=np.float32),
                                device=self.device)
            ret = torch.zeros_like(r)
            running = torch.zeros(r.shape[1], device=self.device)
            for k in reversed(range(r.shape[0])):
                running = r[k] + self.gamma * (1.0 - d[k]) * running
                ret[k] = running
            self.return_normalizer.update(ret.reshape(-1, 1))
            self._reward_scale = float(self.return_normalizer.var.sqrt().clamp_min(1e-6))
        if len(self.buffer) < self.start_steps:
            return {"gradient_steps": 0, "buffer": len(self.buffer)}
        steps = max(1, int(self.updates_per_step * arr["obs"].shape[0]))
        totals = {"critic_loss": 0.0, "actor_loss": 0.0, "alpha": 0.0}
        for _ in range(steps):
            o, a, r, o2, done = self.buffer.sample(self.batch_size, self.device)
            o, o2 = self._norm(o), self._norm(o2)
            with torch.no_grad():
                a2, logp2 = self.policy.sample(o2)
                q_next = torch.min(self.q1_target(torch.cat([o2, a2], -1)).squeeze(-1),
                                   self.q2_target(torch.cat([o2, a2], -1)).squeeze(-1))
                target = (r / self._reward_scale
                          + self.gamma * (1.0 - done) * (q_next - self.alpha * logp2))
            q_loss = (F.mse_loss(self.q1(torch.cat([o, a], -1)).squeeze(-1), target)
                      + F.mse_loss(self.q2(torch.cat([o, a], -1)).squeeze(-1), target))
            self.q_opt.zero_grad(); q_loss.backward(); self.q_opt.step()

            a_new, logp, mean_new, log_std_new = self.policy.sample_with_params(o)
            q_new = torch.min(self.q1(torch.cat([o, a_new], -1)).squeeze(-1),
                              self.q2(torch.cat([o, a_new], -1)).squeeze(-1))
            policy_loss = ((self.alpha * logp - q_new).mean()
                           + _presquash_penalty(mean_new, log_std_new, self.policy_reg))
            self.policy_opt.zero_grad(); policy_loss.backward(); self.policy_opt.step()

            if self.autotune:
                alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
                self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()

            _soft_update(self.q1, self.q1_target, self.tau)
            _soft_update(self.q2, self.q2_target, self.tau)
            totals["critic_loss"] += float(q_loss.detach())
            totals["actor_loss"] += float(policy_loss.detach())
            totals["alpha"] += float(self.alpha)
        return {**{k: v / steps for k, v in totals.items()},
                "gradient_steps": steps, "buffer": len(self.buffer)}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({"policy": self.policy.state_dict(), "q1": self.q1.state_dict(),
                    "q2": self.q2.state_dict(), "log_alpha": self.log_alpha.detach(),
                    "obs_norm": self.obs_norm.state_dict()},
                   os.path.join(path, "sac.pt"))

    def load(self, path: str) -> None:
        d = torch.load(os.path.join(path, "sac.pt"), map_location=self.device)
        self.policy.load_state_dict(d["policy"])
        self.q1.load_state_dict(d["q1"]); self.q2.load_state_dict(d["q2"])
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))


class _PPOActor(nn.Module):

    LOG_STD_MIN, LOG_STD_MAX = -4.0, 1.0

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def distribution(self, x: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_head(self.trunk(x))
        std = torch.clamp(self.log_std, self.LOG_STD_MIN, self.LOG_STD_MAX).exp()
        return torch.distributions.Normal(mean, std.expand_as(mean))


def _gae(rewards: torch.Tensor, values: torch.Tensor, next_values: torch.Tensor,
         episode_end: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        delta = rewards[t] + gamma * (1.0 - episode_end[t]) * next_values[t] - values[t]
        last = delta + gamma * lam * (1.0 - episode_end[t]) * last
        adv[t] = last
    return adv


class DMAPPOAgent:

    def __init__(self, obs_dim: int, action_dim: int, num_buildings: int = 3,
                 hidden_dim: int = 128, lr: float = 3e-4, gamma: float = 0.99,
                 gae_lambda: float = 0.95, clip_eps: float = 0.2, ppo_epochs: int = 10,
                 minibatch_size: int = 64, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, max_grad_norm: float = 0.5,
                 target_kl: float = 0.02, scale_returns: bool = True,
                 device: str = "cpu") -> None:
        self.B, self.obs_dim, self.action_dim = int(num_buildings), int(obs_dim), int(action_dim)
        self.gamma, self.lam = float(gamma), float(gae_lambda)
        self.clip_eps, self.epochs = float(clip_eps), int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.value_coef, self.entropy_coef = float(value_coef), float(entropy_coef)
        self.max_grad_norm, self.target_kl = float(max_grad_norm), float(target_kl)
        self.scale_returns = bool(scale_returns)
        self.device = torch.device(device)
        self.return_normalizer = RunningNormalizer(1).to(self.device)
        self.actors = nn.ModuleList([_PPOActor(obs_dim, hidden_dim, action_dim)
                                     for _ in range(self.B)]).to(self.device)
        self.critics = nn.ModuleList([_MLP(obs_dim, hidden_dim, 1)
                                      for _ in range(self.B)]).to(self.device)
        self.actor_opts = [optim.Adam(self.actors[i].parameters(), lr=lr)
                           for i in range(self.B)]
        self.critic_opts = [optim.Adam(self.critics[i].parameters(), lr=lr)
                            for i in range(self.B)]
        self._last_log_probs = np.zeros(self.B, dtype=np.float32)
        self._last_pre_tanh = np.zeros((self.B, self.action_dim), dtype=np.float32)
        self.obs_norm = _ObsNorm(self.obs_dim).to(self.device)

    def select_action(self, obs_list, history=None, explore: bool = True) -> np.ndarray:
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        pre_tanh = np.zeros((self.B, self.action_dim), dtype=np.float32)
        log_probs = np.zeros(self.B, dtype=np.float32)
        with torch.no_grad():
            for i, obs in enumerate(obs_list):
                x = self.obs_norm(torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                                  device=self.device).unsqueeze(0))
                dist = self.actors[i].distribution(x)
                z = dist.sample() if explore else dist.mean
                log_probs[i] = float(dist.log_prob(z).sum())
                pre_tanh[i] = z.squeeze(0).cpu().numpy()
                actions[i] = torch.tanh(z).squeeze(0).cpu().numpy()
        self._last_pre_tanh, self._last_log_probs = pre_tanh, log_probs
        return actions

    def _behaviour(self, batch: Dict[str, Any], arr: Dict[str, np.ndarray]):
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        if batch.get("pre_tanh") is not None and batch.get("behaviour_log_probs") is not None:
            return t(batch["pre_tanh"]), t(batch["behaviour_log_probs"]), False
        z = torch.atanh(t(arr["actions"]).clamp(-1.0 + 1e-6, 1.0 - 1e-6))
        with torch.no_grad():
            obs_n = self.obs_norm(t(arr["obs"]))
            logp = torch.stack([self.actors[b].distribution(obs_n[:, b])
                                .log_prob(z[:, b]).sum(-1) for b in range(self.B)], 1)
        return z, logp, True

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}
        arr = _episode_arrays(batch)
        t = lambda x: torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        raw_obs = t(arr["obs"])
        z_all, old_logp, fell_back = self._behaviour(batch, arr)
        obs, next_obs = self.obs_norm(raw_obs), self.obs_norm(t(arr["next_obs"]))
        rewards, done = t(arr["rewards"]), t(arr["dones"])
        N = obs.shape[0]

        with torch.no_grad():
            if self.scale_returns:
                ret = torch.zeros_like(rewards)
                running = torch.zeros(self.B, device=self.device)
                for k in reversed(range(N)):
                    running = rewards[k] + self.gamma * (1.0 - done[k]) * running
                    ret[k] = running
                self.return_normalizer.update(ret.reshape(-1, 1))
                scale = float(self.return_normalizer.var.sqrt().clamp_min(1e-6))
            else:
                scale = 1.0
            values = torch.stack([self.critics[b](obs[:, b]).squeeze(-1) for b in range(self.B)], 1)
            next_values = torch.stack([self.critics[b](next_obs[:, b]).squeeze(-1)
                                       for b in range(self.B)], 1)
            adv = _gae(rewards / scale, values, next_values, done, self.gamma, self.lam)
            returns = adv + values
            adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)

        mb = max(2, min(self.minibatch_size, N))
        stats = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                 "clip_frac": 0.0}
        n_updates, stopped = 0, False
        for _ in range(self.epochs):
            if stopped:
                break
            perm = torch.randperm(N, device=self.device)
            for start in range(0, N, mb):
                idx = perm[start:start + mb]
                if idx.numel() < 2:
                    continue
                kl_epoch = 0.0
                for b in range(self.B):
                    dist = self.actors[b].distribution(obs[idx, b])
                    logp = dist.log_prob(z_all[idx, b]).sum(-1)
                    log_ratio = logp - old_logp[idx, b]
                    ratio = log_ratio.clamp(-20.0, 20.0).exp()
                    a_b = adv[idx, b]
                    policy_loss = -torch.min(
                        ratio * a_b,
                        ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * a_b).mean()
                    v_pred = self.critics[b](obs[idx, b]).squeeze(-1)
                    v_clipped = values[idx, b] + (v_pred - values[idx, b]).clamp(
                        -self.clip_eps, self.clip_eps)
                    value_loss = torch.max((v_pred - returns[idx, b]) ** 2,
                                           (v_clipped - returns[idx, b]) ** 2).mean()
                    entropy = dist.entropy().sum(-1).mean()
                    actor_loss = policy_loss - self.entropy_coef * entropy
                    self.actor_opts[b].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[b].parameters(), self.max_grad_norm)
                    self.actor_opts[b].step()
                    self.critic_opts[b].zero_grad()
                    (self.value_coef * value_loss).backward()
                    nn.utils.clip_grad_norm_(self.critics[b].parameters(), self.max_grad_norm)
                    self.critic_opts[b].step()
                    with torch.no_grad():
                        kl = float(((ratio - 1.0) - log_ratio).mean())
                    kl_epoch += kl
                    stats["policy"] += float(policy_loss.detach()) / self.B
                    stats["value"] += float(value_loss.detach()) / self.B
                    stats["entropy"] += float(entropy.detach()) / self.B
                    stats["approx_kl"] += kl / self.B
                    stats["clip_frac"] += float(
                        ((ratio - 1.0).abs() > self.clip_eps).float().mean()) / self.B
                n_updates += 1
                if self.target_kl and kl_epoch / self.B > 1.5 * self.target_kl:
                    stopped = True
                    break
        with torch.no_grad():
            self.obs_norm.fit(raw_obs)
        k = max(n_updates, 1)
        return {**{key: v / k for key, v in stats.items()}, "gradient_steps": n_updates,
                "stopped_early": stopped, "recomputed_behaviour_logp": fell_back,
                "reward_scale": scale}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({"actors": self.actors.state_dict(),
                    "critics": self.critics.state_dict(),
                    "obs_norm": self.obs_norm.state_dict()},
                   os.path.join(path, "dmappo.pt"))

    def load(self, path: str) -> None:
        d = torch.load(os.path.join(path, "dmappo.pt"), map_location=self.device)
        self.actors.load_state_dict(d["actors"])
        self.critics.load_state_dict(d["critics"])
        if "obs_norm" in d:
            self.obs_norm.load_state_dict(d["obs_norm"])


def MPCAgent(*args, **kwargs):
    raise NotImplementedError(
        "MPCAgent has been replaced by stems.mpc.StorageMPC, which optimises over the "
        "exact BatteryModel and TankModel inverses with the real price series, the "
        "district cap and a forecast that is either causal (arm 'mpc') or perfect "
        "(arm 'mpc-oracle'). Build it through experiments.controllers.build_controller, "
        "or construct StorageMPC directly. See CHANGELOG.md step 2.")


class _DDPGActor(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MADDPGAgent:

    def __init__(self, obs_dim: int, action_dim: int, num_buildings: int = 3,
                 hidden_dim: int = 128, lr: float = 1e-3, gamma: float = 0.99,
                 tau: float = 0.01, noise_std: float = 0.2, noise_decay: float = 0.995,
                 noise_min: float = 0.02, batch_size: int = 256,
                 updates_per_step: float = 1.0, start_steps: int = 256,
                 buffer_capacity: int = 200_000, policy_reg: float = 0.0,
                 device: str = "cpu") -> None:
        self.B, self.obs_dim, self.action_dim = int(num_buildings), int(obs_dim), int(action_dim)
        self.policy_reg = float(policy_reg)
        self.gamma, self.tau = float(gamma), float(tau)
        self.noise_std, self.noise_decay = float(noise_std), float(noise_decay)
        self.noise_min = float(noise_min)
        self.batch_size, self.start_steps = int(batch_size), int(start_steps)
        self.updates_per_step = float(updates_per_step)
        self.device = torch.device(device)
        joint = self.B * (self.obs_dim + self.action_dim)
        self.actors = nn.ModuleList([_DDPGActor(obs_dim, hidden_dim, action_dim)
                                     for _ in range(self.B)]).to(self.device)
        self.critics = nn.ModuleList([_MLP(joint, hidden_dim, 1)
                                      for _ in range(self.B)]).to(self.device)
        self.actors_target = copy.deepcopy(self.actors)
        self.critics_target = copy.deepcopy(self.critics)
        for p in list(self.actors_target.parameters()) + list(self.critics_target.parameters()):
            p.requires_grad_(False)
        self.actor_opts = [optim.Adam(self.actors[i].parameters(), lr=lr) for i in range(self.B)]
        self.critic_opts = [optim.Adam(self.critics[i].parameters(), lr=lr) for i in range(self.B)]
        self.buffer = _FlatReplay(buffer_capacity)
        self._rng = np.random.default_rng(0)
        self.obs_norm = _ObsNorm(self.obs_dim).to(self.device)

    def _norm(self, flat: torch.Tensor) -> torch.Tensor:
        return self.obs_norm(flat.view(*flat.shape[:-1], self.B, self.obs_dim)
                             ).reshape(flat.shape)

    def select_action(self, obs_list, history=None, explore: bool = True) -> np.ndarray:
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        with torch.no_grad():
            for i, obs in enumerate(obs_list):
                x = self.obs_norm(torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                                  device=self.device).unsqueeze(0))
                actions[i] = self.actors[i](x).squeeze(0).cpu().numpy()
        if explore:
            actions += self._rng.normal(0.0, self.noise_std, size=actions.shape)
            self.noise_std = max(self.noise_min, self.noise_std * self.noise_decay)
        return np.clip(actions, -1.0, 1.0).astype(np.float32)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}
        arr = _episode_arrays(batch)
        N = arr["obs"].shape[0]
        self.obs_norm.fit(torch.as_tensor(arr["obs"], device=self.device))
        for n in range(N):
            self.buffer.add(arr["obs"][n].reshape(-1), arr["actions"][n].reshape(-1),
                            arr["rewards"][n], arr["next_obs"][n].reshape(-1),
                            float(arr["dones"][n]))
        if len(self.buffer) < self.start_steps:
            return {"gradient_steps": 0, "buffer": len(self.buffer)}
        steps = max(1, int(self.updates_per_step * N))
        totals = {"critic_loss": 0.0, "actor_loss": 0.0}
        for _ in range(steps):
            o, a, r, o2, done = self.buffer.sample(self.batch_size, self.device)
            o, o2 = self._norm(o), self._norm(o2)
            o_b = o.view(-1, self.B, self.obs_dim)
            o2_b = o2.view(-1, self.B, self.obs_dim)
            with torch.no_grad():
                a2 = torch.cat([self.actors_target[i](o2_b[:, i]) for i in range(self.B)], -1)
                joint_next = torch.cat([o2, a2], -1)
            joint = torch.cat([o, a], -1)
            for i in range(self.B):
                with torch.no_grad():
                    target = (r[:, i] + self.gamma * (1.0 - done)
                              * self.critics_target[i](joint_next).squeeze(-1))
                critic_loss = F.mse_loss(self.critics[i](joint).squeeze(-1), target)
                self.critic_opts[i].zero_grad(); critic_loss.backward()
                self.critic_opts[i].step()
                a_new = a.view(-1, self.B, self.action_dim).clone()
                own = self.actors[i](o_b[:, i])
                a_new[:, i] = own
                actor_loss = -self.critics[i](
                    torch.cat([o, a_new.reshape(a.shape)], -1)).mean()
                if self.policy_reg > 0.0:
                    actor_loss = actor_loss + self.policy_reg * own.pow(2).mean()
                self.actor_opts[i].zero_grad(); actor_loss.backward()
                self.actor_opts[i].step()
                totals["critic_loss"] += float(critic_loss.detach()) / self.B
                totals["actor_loss"] += float(actor_loss.detach()) / self.B
            _soft_update(self.actors, self.actors_target, self.tau)
            _soft_update(self.critics, self.critics_target, self.tau)
        return {**{k: v / steps for k, v in totals.items()},
                "gradient_steps": steps, "noise_std": self.noise_std,
                "buffer": len(self.buffer)}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({"actors": self.actors.state_dict(),
                    "critics": self.critics.state_dict(),
                    "obs_norm": self.obs_norm.state_dict()},
                   os.path.join(path, "maddpg.pt"))

    def load(self, path: str) -> None:
        d = torch.load(os.path.join(path, "maddpg.pt"), map_location=self.device)
        self.actors.load_state_dict(d["actors"]); self.critics.load_state_dict(d["critics"])
        if "obs_norm" in d:
            self.obs_norm.load_state_dict(d["obs_norm"])
        self.actors_target = copy.deepcopy(self.actors)
        self.critics_target = copy.deepcopy(self.critics)


class MARLISAAgent:

    def __init__(self, obs_dim: int, action_dim: int, num_buildings: int = 3,
                 hidden_dim: int = 256, lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, batch_size: int = 256,
                 updates_per_step: float = 1.0, start_steps: int = 256,
                 buffer_capacity: int = 200_000, alpha_entropy: float = 0.2,
                 autotune_alpha: bool = True, share_consumption: bool = True,
                 policy_reg: float = 0.0, device: str = "cpu") -> None:
        self.B, self.obs_dim, self.action_dim = int(num_buildings), int(obs_dim), int(action_dim)
        self.policy_reg = float(policy_reg)
        self.gamma, self.tau = float(gamma), float(tau)
        self.batch_size, self.start_steps = int(batch_size), int(start_steps)
        self.updates_per_step = float(updates_per_step)
        self.share_consumption = bool(share_consumption)
        self.device = torch.device(device)
        self.shared_dim = self.B if self.share_consumption else 0
        self.aug_dim = [self.obs_dim + i * self.action_dim + self.shared_dim
                        for i in range(self.B)]
        self.policies = nn.ModuleList([
            _SquashedGaussian(self.aug_dim[i], hidden_dim, action_dim)
            for i in range(self.B)]).to(self.device)
        self.q1 = nn.ModuleList([_MLP(self.aug_dim[i] + action_dim, hidden_dim, 1)
                                 for i in range(self.B)]).to(self.device)
        self.q2 = nn.ModuleList([_MLP(self.aug_dim[i] + action_dim, hidden_dim, 1)
                                 for i in range(self.B)]).to(self.device)
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        for p in list(self.q1_target.parameters()) + list(self.q2_target.parameters()):
            p.requires_grad_(False)
        self.policy_opts = [optim.Adam(self.policies[i].parameters(), lr=lr)
                            for i in range(self.B)]
        self.q_opts = [optim.Adam(list(self.q1[i].parameters()) + list(self.q2[i].parameters()),
                                  lr=lr) for i in range(self.B)]
        self.autotune = bool(autotune_alpha)
        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.tensor(float(np.log(alpha_entropy)),
                                      requires_grad=self.autotune, device=self.device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr) if self.autotune else None
        self.buffers = [_FlatReplay(buffer_capacity) for _ in range(self.B)]
        self.obs_norms = nn.ModuleList([_ObsNorm(self.aug_dim[i])
                                        for i in range(self.B)]).to(self.device)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    def augment(self, obs_list: Sequence[np.ndarray], actions: np.ndarray,
                building: int) -> np.ndarray:
        parts = [np.asarray(obs_list[building], dtype=np.float32)]
        if building > 0:
            parts.append(np.asarray(actions, dtype=np.float32)[:building].reshape(-1))
        if self.share_consumption:
            parts.append(np.array([float(o[_IDX_NET]) for o in obs_list], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def select_action(self, obs_list, history=None, explore: bool = True) -> np.ndarray:
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        with torch.no_grad():
            for i in range(self.B):
                aug = self.augment(obs_list, actions, i)
                x = self.obs_norms[i](
                    torch.as_tensor(aug, device=self.device).unsqueeze(0))
                a, _ = self.policies[i].sample(x, deterministic=not explore)
                actions[i] = a.squeeze(0).cpu().numpy()
        return actions

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}
        arr = _episode_arrays(batch)
        N = arr["obs"].shape[0]
        fit = [[] for _ in range(self.B)]
        for n in range(N):
            nxt = min(n + 1, N - 1)
            for i in range(self.B):
                aug = self.augment(arr["obs"][n], arr["actions"][n], i)
                aug_next = self.augment(arr["next_obs"][n], arr["actions"][nxt], i)
                self.buffers[i].add(aug, arr["actions"][n, i], float(arr["rewards"][n, i]),
                                    aug_next, float(arr["dones"][n]))
                fit[i].append(aug)
        for i in range(self.B):
            self.obs_norms[i].fit(torch.as_tensor(np.stack(fit[i]), device=self.device))
        if len(self.buffers[0]) < self.start_steps:
            return {"gradient_steps": 0, "buffer": len(self.buffers[0])}
        steps = max(1, int(self.updates_per_step * N))
        totals = {"critic_loss": 0.0, "actor_loss": 0.0}
        for _ in range(steps):
            for i in range(self.B):
                o, a, r, o2, done = self.buffers[i].sample(self.batch_size, self.device)
                o, o2 = self.obs_norms[i](o), self.obs_norms[i](o2)
                with torch.no_grad():
                    a2, logp2 = self.policies[i].sample(o2)
                    q_next = torch.min(self.q1_target[i](torch.cat([o2, a2], -1)).squeeze(-1),
                                       self.q2_target[i](torch.cat([o2, a2], -1)).squeeze(-1))
                    target = r + self.gamma * (1.0 - done) * (q_next - self.alpha * logp2)
                q_loss = (F.mse_loss(self.q1[i](torch.cat([o, a], -1)).squeeze(-1), target)
                          + F.mse_loss(self.q2[i](torch.cat([o, a], -1)).squeeze(-1), target))
                self.q_opts[i].zero_grad(); q_loss.backward(); self.q_opts[i].step()
                a_new, logp, mean_new, log_std_new = \
                    self.policies[i].sample_with_params(o)
                q_new = torch.min(self.q1[i](torch.cat([o, a_new], -1)).squeeze(-1),
                                  self.q2[i](torch.cat([o, a_new], -1)).squeeze(-1))
                policy_loss = ((self.alpha * logp - q_new).mean()
                               + _presquash_penalty(mean_new, log_std_new,
                                                    self.policy_reg))
                self.policy_opts[i].zero_grad(); policy_loss.backward()
                self.policy_opts[i].step()
                if self.autotune:
                    alpha_loss = -(self.log_alpha
                                   * (logp.detach() + self.target_entropy)).mean()
                    self.alpha_opt.zero_grad(); alpha_loss.backward(); self.alpha_opt.step()
                _soft_update(self.q1[i], self.q1_target[i], self.tau)
                _soft_update(self.q2[i], self.q2_target[i], self.tau)
                totals["critic_loss"] += float(q_loss.detach()) / self.B
                totals["actor_loss"] += float(policy_loss.detach()) / self.B
        return {**{k: v / steps for k, v in totals.items()},
                "gradient_steps": steps, "alpha": float(self.alpha),
                "buffer": len(self.buffers[0])}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({"policies": self.policies.state_dict(), "q1": self.q1.state_dict(),
                    "q2": self.q2.state_dict(), "log_alpha": self.log_alpha.detach(),
                    "obs_norms": self.obs_norms.state_dict()},
                   os.path.join(path, "marlisa.pt"))

    def load(self, path: str) -> None:
        d = torch.load(os.path.join(path, "marlisa.pt"), map_location=self.device)
        self.policies.load_state_dict(d["policies"])
        self.q1.load_state_dict(d["q1"]); self.q2.load_state_dict(d["q2"])
        self.q1_target, self.q2_target = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        if "obs_norms" in d:
            self.obs_norms.load_state_dict(d["obs_norms"])
        with torch.no_grad():
            self.log_alpha.copy_(d["log_alpha"].to(self.device))


class _BranchingQ(nn.Module):

    def __init__(self, obs_dim: int, hidden_dim: int, action_dim: int, n_bins: int) -> None:
        super().__init__()
        self.action_dim, self.n_bins = int(action_dim), int(n_bins)
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.heads = nn.Linear(hidden_dim, self.action_dim * self.n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.heads(self.trunk(x)).view(*x.shape[:-1], self.action_dim, self.n_bins)


class MADCQAgent:

    def __init__(self, obs_dim: int, action_dim: int, num_buildings: int = 3,
                 hidden_dim: int = 128, n_bins: int = 11, lr: float = 3e-4,
                 gamma: float = 0.99, tau: float = 0.005, batch_size: int = 256,
                 updates_per_step: float = 1.0, start_steps: int = 256,
                 buffer_capacity: int = 200_000, eps_start: float = 1.0,
                 eps_end: float = 0.05, eps_decay_steps: int = 20_000,
                 soc_min: float = 0.1, soc_max: float = 0.9,
                 battery_model=None, elec_idx: int = 1, device: str = "cpu") -> None:
        if battery_model is None:
            raise ValueError(
                "MADCQAgent needs the exact BatteryModel to build its safe action set "
                "(pass STEMSEnvironment.battery_model()). Constrained Q-learning without "
                "a plant model is just Q-learning.")
        self.B, self.obs_dim, self.action_dim = int(num_buildings), int(obs_dim), int(action_dim)
        self.n_bins = int(n_bins)
        self.gamma, self.tau = float(gamma), float(tau)
        self.batch_size, self.start_steps = int(batch_size), int(start_steps)
        self.updates_per_step = float(updates_per_step)
        self.eps_start, self.eps_end = float(eps_start), float(eps_end)
        self.eps_decay_steps = max(1, int(eps_decay_steps))
        self.soc_min, self.soc_max = float(soc_min), float(soc_max)
        self.battery = battery_model
        self.elec_idx = int(elec_idx)
        self.device = torch.device(device)
        self.bins = np.linspace(-1.0, 1.0, self.n_bins).astype(np.float32)
        self.q = nn.ModuleList([_BranchingQ(obs_dim, hidden_dim, action_dim, self.n_bins)
                                for _ in range(self.B)]).to(self.device)
        self.q_target = copy.deepcopy(self.q)
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.opts = [optim.Adam(self.q[i].parameters(), lr=lr) for i in range(self.B)]
        self.buffers = [_FlatReplay(buffer_capacity) for _ in range(self.B)]
        self._steps = 0
        self._rng = np.random.default_rng(0)
        self._plant_cache: Dict[Tuple[int, int], Any] = {}
        self.obs_norm = _ObsNorm(self.obs_dim).to(self.device)

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def safe_bin_mask(self, soc: np.ndarray) -> np.ndarray:
        lo, hi = self.battery.safe_interval(
            np.asarray(soc, dtype=np.float64),
            np.full(self.B, self.soc_min), np.full(self.B, self.soc_max))
        mask = (self.bins[None, :] >= lo[:, None] - 1e-6) & \
               (self.bins[None, :] <= hi[:, None] + 1e-6)
        empty = ~mask.any(axis=1)
        if empty.any():
            mid = 0.5 * (lo + hi)
            nearest = np.abs(self.bins[None, :] - mid[:, None]).argmin(axis=1)
            mask[empty, nearest[empty]] = True
        return mask

    def _plant_slice(self, building: int, n: int):
        from stems.battery import BatteryModel

        key = (int(building), int(n))
        cached = self._plant_cache.get(key)
        if cached is not None:
            return cached
        b, src = int(building), self.battery
        model = BatteryModel(
            capacity=np.repeat(src.capacity[b], n),
            nominal_power=np.repeat(src.nominal_power[b], n),
            loss=np.repeat(src.loss[b], n),
            eta_curves=[np.stack([src._eta_x[b], src._eta_y[b]])] * n,
            power_curves=[np.stack([src._pow_x[b], src._pow_y[b]])] * n,
            hours_per_step=src.dt)
        self._plant_cache[key] = model
        return model

    def _batch_bin_mask(self, building: int, soc: np.ndarray) -> np.ndarray:
        n = int(soc.shape[0])
        model = self._plant_slice(building, n)
        lo, hi = model.safe_interval(soc, np.full(n, self.soc_min),
                                     np.full(n, self.soc_max))
        m = (self.bins[None, :] >= lo[:, None] - 1e-6) & \
            (self.bins[None, :] <= hi[:, None] + 1e-6)
        empty = ~m.any(axis=1)
        if empty.any():
            mid = 0.5 * (lo + hi)
            nearest = np.abs(self.bins[None, :] - mid[:, None]).argmin(axis=1)
            m[empty, nearest[empty]] = True
        return m

    def _masked_q(self, q_vals: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = q_vals.clone()
        out[..., self.elec_idx, :] = out[..., self.elec_idx, :].masked_fill(~mask, -1e9)
        return out

    def select_action(self, obs_list, history=None, explore: bool = True) -> np.ndarray:
        soc = np.array([float(o[_IDX_SOC_ELEC]) for o in obs_list], dtype=np.float64)
        mask_np = self.safe_bin_mask(soc)
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        eps = self.epsilon if explore else 0.0
        with torch.no_grad():
            for i, obs in enumerate(obs_list):
                x = self.obs_norm(torch.as_tensor(np.asarray(obs, dtype=np.float32),
                                                  device=self.device).unsqueeze(0))
                mask = torch.as_tensor(mask_np[i], device=self.device).unsqueeze(0)
                idx = self._masked_q(self.q[i](x), mask).argmax(dim=-1).squeeze(0).cpu().numpy()
                if eps > 0.0:
                    for j in range(self.action_dim):
                        if self._rng.random() < eps:
                            allowed = (np.flatnonzero(mask_np[i]) if j == self.elec_idx
                                       else np.arange(self.n_bins))
                            idx[j] = int(self._rng.choice(allowed))
                actions[i] = self.bins[idx]
        if explore:
            self._steps += 1
        return actions

    def _to_bins(self, action: np.ndarray) -> np.ndarray:
        return np.abs(self.bins[None, :] - np.asarray(action)[:, None]).argmin(axis=1)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}
        arr = _episode_arrays(batch)
        N = arr["obs"].shape[0]
        self.obs_norm.fit(torch.as_tensor(arr["obs"], device=self.device))
        for n in range(N):
            for i in range(self.B):
                self.buffers[i].add(arr["obs"][n, i],
                                    self._to_bins(arr["actions"][n, i]).astype(np.float32),
                                    float(arr["rewards"][n, i]), arr["next_obs"][n, i],
                                    float(arr["dones"][n]))
        if len(self.buffers[0]) < self.start_steps:
            return {"gradient_steps": 0, "buffer": len(self.buffers[0])}
        steps = max(1, int(self.updates_per_step * N))
        total = 0.0
        for _ in range(steps):
            for i in range(self.B):
                o, a_idx, r, o2, done = self.buffers[i].sample(self.batch_size, self.device)
                soc2 = o2[:, _IDX_SOC_ELEC].detach().cpu().numpy().astype(np.float64)
                m = self._batch_bin_mask(i, soc2)
                mask = torch.as_tensor(m, device=self.device)
                o, o2 = self.obs_norm(o), self.obs_norm(o2)
                with torch.no_grad():
                    sel = self._masked_q(self.q[i](o2), mask).argmax(dim=-1, keepdim=True)
                    q_next = self._masked_q(self.q_target[i](o2), mask).gather(
                        -1, sel).squeeze(-1).sum(-1)
                    target = r + self.gamma * (1.0 - done) * q_next
                q_pred = self.q[i](o).gather(-1, a_idx.long().unsqueeze(-1)).squeeze(-1).sum(-1)
                loss = F.smooth_l1_loss(q_pred, target)
                self.opts[i].zero_grad(); loss.backward(); self.opts[i].step()
                _soft_update(self.q[i], self.q_target[i], self.tau)
                total += float(loss.detach()) / self.B
        return {"critic_loss": total / steps, "gradient_steps": steps,
                "epsilon": self.epsilon, "buffer": len(self.buffers[0])}

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save({"q": self.q.state_dict(), "steps": self._steps,
                    "obs_norm": self.obs_norm.state_dict()},
                   os.path.join(path, "madcq.pt"))

    def load(self, path: str) -> None:
        d = torch.load(os.path.join(path, "madcq.pt"), map_location=self.device)
        self.q.load_state_dict(d["q"])
        if "obs_norm" in d:
            self.obs_norm.load_state_dict(d["obs_norm"])
        self.q_target = copy.deepcopy(self.q)
        self._steps = int(d.get("steps", 0))


class MetaEMSAgent:

    def __init__(self, obs_dim: int, action_dim: int, num_buildings: int = 3,
                 hidden_dim: int = 256, lr_inner: float = 3e-4, lr_outer: float = 0.3,
                 inner_steps: int = 8, gamma: float = 0.99, tau: float = 0.005,
                 batch_size: int = 128, start_steps: int = 256,
                 buffer_capacity: int = 200_000, alpha_entropy: float = 0.2,
                 policy_reg: float = 0.0, device: str = "cpu") -> None:
        self.B, self.obs_dim, self.action_dim = int(num_buildings), int(obs_dim), int(action_dim)
        self.inner_steps = max(1, int(inner_steps))
        self.epsilon_outer = float(lr_outer)
        self.start_steps = int(start_steps)
        self.device = torch.device(device)
        self._base = SingleAgentSAC(
            obs_dim=obs_dim, action_dim=action_dim, num_buildings=1,
            hidden_dim=hidden_dim, lr=lr_inner, gamma=gamma, tau=tau,
            batch_size=batch_size, updates_per_step=0.0, start_steps=0,
            buffer_capacity=buffer_capacity, autotune_alpha=False,
            alpha_entropy=alpha_entropy, policy_reg=policy_reg, device=device)
        self.buffers = [_FlatReplay(buffer_capacity) for _ in range(self.B)]
        self._total = 0
        self.obs_norm = self._base.obs_norm

    def _modules(self):
        return (self._base.policy, self._base.q1, self._base.q2)

    def _snapshot(self):
        return [copy.deepcopy(m.state_dict()) for m in self._modules()]

    def _restore(self, snap) -> None:
        for m, s in zip(self._modules(), snap):
            m.load_state_dict(s)

    def select_action(self, obs_list, history=None, explore: bool = True) -> np.ndarray:
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            actions[i] = self._base.select_action([obs], None, explore)[0]
        return actions

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}
        arr = _episode_arrays(batch)
        N = arr["obs"].shape[0]
        self.obs_norm.fit(torch.as_tensor(arr["obs"], device=self.device))
        for n in range(N):
            for i in range(self.B):
                self.buffers[i].add(arr["obs"][n, i], arr["actions"][n, i],
                                    float(arr["rewards"][n, i]), arr["next_obs"][n, i],
                                    float(arr["dones"][n]))
        self._total += N
        if len(self.buffers[0]) < self.start_steps:
            return {"gradient_steps": 0, "buffer": len(self.buffers[0])}

        theta = self._snapshot()
        adapted: List[List[Dict[str, torch.Tensor]]] = []
        losses = 0.0
        for i in range(self.B):
            self._restore(theta)
            losses += self._adapt_on(self.buffers[i], self.inner_steps)
            adapted.append(self._snapshot())
        with torch.no_grad():
            for m_i, module in enumerate(self._modules()):
                state = module.state_dict()
                for key, start in theta[m_i].items():
                    if not torch.is_floating_point(start):
                        state[key] = start.clone()
                        continue
                    mean = torch.stack([a[m_i][key].float() for a in adapted]).mean(0)
                    state[key] = start.float() + self.epsilon_outer * (mean - start.float())
                module.load_state_dict(state)
        self._base.q1_target = copy.deepcopy(self._base.q1)
        self._base.q2_target = copy.deepcopy(self._base.q2)
        for p in list(self._base.q1_target.parameters()) + list(self._base.q2_target.parameters()):
            p.requires_grad_(False)
        return {"critic_loss": losses / self.B, "gradient_steps": self.B * self.inner_steps,
                "tasks": self.B, "buffer": len(self.buffers[0])}

    def _adapt_on(self, buffer: _FlatReplay, steps: int) -> float:
        sac, total = self._base, 0.0
        sac.policy_opt = optim.Adam(sac.policy.parameters(),
                                    lr=sac.policy_opt.param_groups[0]["lr"])
        sac.q_opt = optim.Adam(list(sac.q1.parameters()) + list(sac.q2.parameters()),
                               lr=sac.q_opt.param_groups[0]["lr"])
        for _ in range(steps):
            o, a, r, o2, done = buffer.sample(sac.batch_size, sac.device)
            o, o2 = sac._norm(o), sac._norm(o2)
            with torch.no_grad():
                a2, logp2 = sac.policy.sample(o2)
                q_next = torch.min(sac.q1_target(torch.cat([o2, a2], -1)).squeeze(-1),
                                   sac.q2_target(torch.cat([o2, a2], -1)).squeeze(-1))
                target = r + sac.gamma * (1.0 - done) * (q_next - sac.alpha * logp2)
            q_loss = (F.mse_loss(sac.q1(torch.cat([o, a], -1)).squeeze(-1), target)
                      + F.mse_loss(sac.q2(torch.cat([o, a], -1)).squeeze(-1), target))
            sac.q_opt.zero_grad(); q_loss.backward(); sac.q_opt.step()
            a_new, logp, mean_new, log_std_new = sac.policy.sample_with_params(o)
            q_new = torch.min(sac.q1(torch.cat([o, a_new], -1)).squeeze(-1),
                              sac.q2(torch.cat([o, a_new], -1)).squeeze(-1))
            policy_loss = ((sac.alpha * logp - q_new).mean()
                           + _presquash_penalty(mean_new, log_std_new, sac.policy_reg))
            sac.policy_opt.zero_grad(); policy_loss.backward(); sac.policy_opt.step()
            _soft_update(sac.q1, sac.q1_target, sac.tau)
            _soft_update(sac.q2, sac.q2_target, sac.tau)
            total += float(q_loss.detach())
        return total / max(steps, 1)

    def adapt(self, building: int, steps: Optional[int] = None) -> float:
        return self._adapt_on(self.buffers[int(building)],
                              self.inner_steps if steps is None else int(steps))

    def save(self, path: str) -> None:
        self._base.save(path)

    def load(self, path: str) -> None:
        self._base.load(path)


class IndependentSAC:

    def __init__(self, obs_dim: int, action_dim: int, num_buildings: int = 3,
                 **kwargs: Any) -> None:
        self.B, self.obs_dim, self.action_dim = int(num_buildings), int(obs_dim), int(action_dim)
        self.agents = [SingleAgentSAC(obs_dim, action_dim, num_buildings=1, **kwargs)
                       for _ in range(self.B)]

    def select_action(self, obs_list, history=None, explore: bool = True) -> np.ndarray:
        return np.stack([self.agents[b].select_action([obs_list[b]], explore=explore)[0]
                         for b in range(self.B)]).astype(np.float32)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        if len(batch["obs"]) == 0:
            return {}
        arr = _episode_arrays(batch)
        out: Dict[str, float] = {}
        for b, agent in enumerate(self.agents):
            sub = {"obs": arr["obs"][:, b:b + 1, :],
                   "next_obs": arr["next_obs"][:, b:b + 1, :],
                   "actions": arr["actions"][:, b:b + 1, :],
                   "rewards": arr["rewards"][:, b:b + 1],
                   "dones": arr["dones"]}
            stats = agent._update_from_arrays(sub)
            for k, v in stats.items():
                out[k] = out.get(k, 0.0) + float(v) / self.B
        return out

    def save(self, path: str) -> None:
        for b, agent in enumerate(self.agents):
            agent.save(os.path.join(path, f"building{b}"))

    def load(self, path: str) -> None:
        for b, agent in enumerate(self.agents):
            agent.load(os.path.join(path, f"building{b}"))
