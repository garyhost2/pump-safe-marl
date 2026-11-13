from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stems.dhw_v2_rl.common import ObsBuilder, to_env_action

torch.set_num_threads(1)


class RunningMeanStd:
    def __init__(self, shape, eps: float = 1e-4):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, np.float64).reshape(-1, *self.mean.shape) if self.mean.shape else np.asarray(x, np.float64).ravel()
        bm, bv, n = x.mean(axis=0), x.var(axis=0), x.shape[0]
        d = bm - self.mean
        tot = self.count + n
        self.mean = self.mean + d * n / tot
        self.var = (self.var * self.count + bv * n + d ** 2 * self.count * n / tot) / tot
        self.count = tot

    def norm(self, x, clip: float = 10.0):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip).astype(np.float32)

    def state(self):
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": float(self.count)}

    def load(self, s):
        self.mean, self.var, self.count = np.array(s["mean"]), np.array(s["var"]), float(s["count"])


def mlp(i, o, h=(64, 64), act=nn.Tanh):
    layers, d = [], i
    for k in h:
        layers += [nn.Linear(d, k), act()]
        d = k
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def episode_summary(env) -> Dict[str, float]:
    S = env.summary()
    B = len(S["houses"])
    out = {"cost": float(np.sum(S["cost"])), "reward": float(np.sum(S["reward"])),
           "elec_kwh": float(np.sum(S["elec_kwh"])), "viol_intra_hour": float(np.sum(S["hours_below_floor"])),
           "weeks_missed": int(S["weeks_full"]) * B - int(np.sum(S["weeks_completed"])),
           "sh_degree_hours": float(np.sum(S["sh_degree_hours_below_min"])),
           "starts": float(np.sum(S["starts"]))}
    if "intervention_rate" in S:
        out["intervention_rate"] = float(np.mean(S["intervention_rate"]))
        out["mean_abs_correction_el_kwh"] = float(np.mean(S["mean_abs_correction_el_kwh"]))
    return out


def _inv_softplus(y):
    return float(np.log(np.expm1(y)))


INIT_BETA = ((1.5, 3.0), (3.0, 3.0), (1.1, 6.0))
INIT_SAC_MEAN = (0.2, 0.5, 0.1)


class BetaActor(nn.Module):
    def __init__(self, d, n_act=3, h=(64, 64), init=INIT_BETA):
        super().__init__()
        self.net = mlp(d, 2 * n_act, h)
        self.n = n_act
        last = self.net[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)
            for j, (al, be) in enumerate(init):
                last.bias[j] = _inv_softplus(al - 1.0)
                last.bias[n_act + j] = _inv_softplus(be - 1.0)

    def dist(self, x):
        o = self.net(x)
        a = 1.0 + F.softplus(o[..., :self.n])
        b = 1.0 + F.softplus(o[..., self.n:])
        return torch.distributions.Beta(a, b)

    def mode(self, x):
        d = self.dist(x)
        a, b = d.concentration1, d.concentration0
        return torch.where((a > 1) & (b > 1), (a - 1) / (a + b - 2), torch.where(a <= b, torch.zeros_like(a), torch.ones_like(a)))


class PPO:
    def __init__(self, obs_dim: int, B: int, *, seed: int = 0, critic: str = "local", lr: float = 3e-4,
                 gamma: float = 0.995, lam: float = 0.95, clip: float = 0.2, epochs: int = 10,
                 minibatch: int = 512, rollout: int = 546, ent_coef: float = 0.0, vf_coef: float = 0.5,
                 max_grad: float = 0.5, hidden=(64, 64), reward_scale: float = 1.0, total_updates: int = 160,
                 action_map: str = "need_relative"):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.B, self.d = B, obs_dim
        self.critic_kind = critic
        self.actor = BetaActor(obs_dim, 3, hidden)
        cin = obs_dim * B + B if critic == "central" else obs_dim
        self.critic = mlp(cin, 1, hidden)
        self.opt = torch.optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr, eps=1e-5)
        self.lr0 = lr
        self.hp = dict(gamma=gamma, lam=lam, clip=clip, epochs=epochs, minibatch=minibatch, rollout=rollout,
                       ent_coef=ent_coef, vf_coef=vf_coef, max_grad=max_grad, hidden=list(hidden),
                       reward_scale=reward_scale, critic=critic, lr=lr, total_updates=total_updates,
                       action_map=action_map)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.ret_rms = RunningMeanStd(())
        self.ret = np.zeros(B)
        self.n_updates = 0

    def _cin(self, Xn: torch.Tensor) -> torch.Tensor:
        if self.critic_kind != "central":
            return Xn
        sh = Xn.shape
        flat = Xn.reshape(*sh[:-2], 1, sh[-2] * sh[-1]).expand(*sh[:-2], sh[-2], sh[-2] * sh[-1])
        eye = torch.eye(self.B).expand(*sh[:-2], self.B, self.B)
        return torch.cat([flat, eye], -1)

    def act(self, X: np.ndarray, deterministic: bool = False):
        Xn = torch.as_tensor(self.obs_rms.norm(X))
        with torch.no_grad():
            if deterministic:
                return self.actor.mode(Xn).numpy()
            d = self.actor.dist(Xn)
            a = d.sample().clamp(1e-6, 1 - 1e-6)
            return a.numpy(), d.log_prob(a).sum(-1).numpy(), self.critic(self._cin(Xn)).squeeze(-1).numpy()

    def env_action(self, X):
        return to_env_action(self.act(X, deterministic=True), X, self.hp.get("action_map", "direct"))

    def value(self, X):
        with torch.no_grad():
            return self.critic(self._cin(torch.as_tensor(self.obs_rms.norm(X)))).squeeze(-1).numpy()

    def scale_reward(self, r: np.ndarray) -> np.ndarray:
        r = r * self.hp["reward_scale"]
        self.ret = self.ret * self.hp["gamma"] + r
        self.ret_rms.update(self.ret)
        return r / np.sqrt(self.ret_rms.var + 1e-8)

    def update(self, Xs, A, LP, V, R, last_v, done_last):
        hp = self.hp
        T = len(R)
        adv = np.zeros_like(R)
        g = 0.0
        for t in range(T - 1, -1, -1):
            nv = (0.0 if done_last else last_v) if t == T - 1 else V[t + 1]
            delta = R[t] + hp["gamma"] * nv - V[t]
            g = delta + hp["gamma"] * hp["lam"] * g
            adv[t] = g
        ret = adv + V
        frac = max(0.0, 1.0 - self.n_updates / max(hp["total_updates"], 1))
        for pg in self.opt.param_groups:
            pg["lr"] = self.lr0 * max(frac, 0.05)
        Xn = torch.as_tensor(self.obs_rms.norm(Xs))
        At, LPt = torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(LP, dtype=torch.float32)
        ADV, RET = torch.as_tensor(adv, dtype=torch.float32), torch.as_tensor(ret, dtype=torch.float32)
        n = T * self.B
        stats = []
        for _ in range(hp["epochs"]):
            perm = self.rng.permutation(T)
            mb_t = max(1, hp["minibatch"] // self.B)
            for s in range(0, T, mb_t):
                idx = torch.as_tensor(perm[s:s + mb_t])
                x, a, lp0 = Xn[idx], At[idx], LPt[idx]
                ad = ADV[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                d = self.actor.dist(x)
                lp = d.log_prob(a).sum(-1)
                ratio = torch.exp(lp - lp0)
                l_pi = -torch.min(ratio * ad, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * ad).mean()
                v = self.critic(self._cin(x)).squeeze(-1)
                l_v = F.mse_loss(v, RET[idx])
                ent = d.entropy().sum(-1).mean()
                loss = l_pi + hp["vf_coef"] * l_v - hp["ent_coef"] * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), hp["max_grad"])
                self.opt.step()
                stats.append((float(l_pi), float(l_v), float(ent), float((ratio - 1).abs().mean())))
        self.n_updates += 1
        s = np.mean(stats, axis=0)
        return {"l_pi": s[0], "l_v": s[1], "entropy": s[2], "ratio_dev": s[3], "n": n}

    def train_episode(self, env, extra=None) -> Dict:
        ob = ObsBuilder(env)
        obs, _ = env.reset()
        ob.reset()
        X = ob.build(obs)
        buf = {k: [] for k in ("X", "A", "LP", "V", "R")}
        ustats = []
        t0 = time.time()
        while True:
            a, lp, v = self.act(X)
            obs, r, term, _, info = env.step(to_env_action(a, X, self.hp["action_map"]))
            ob.after_step(info)
            buf["X"].append(X); buf["A"].append(a); buf["LP"].append(lp); buf["V"].append(v)
            r = np.asarray(r, float) - (extra(info) if extra is not None else 0.0)
            buf["R"].append(self.scale_reward(r))
            Xn = ob.build(obs) if not term else X
            if len(buf["R"]) >= self.hp["rollout"] or term:
                Xs = np.stack(buf["X"])
                self.obs_rms.update(Xs.reshape(-1, self.d))
                last_v = self.value(Xn) if not term else np.zeros(self.B)
                ustats.append(self.update(Xs, np.stack(buf["A"]), np.stack(buf["LP"]), np.stack(buf["V"]),
                                          np.stack(buf["R"]), last_v, term))
                buf = {k: [] for k in buf}
            if term:
                break
            X = Xn
        self.ret[:] = 0.0
        out = episode_summary(env)
        out.update({k: float(np.mean([u[k] for u in ustats])) for k in ("l_pi", "l_v", "entropy", "ratio_dev")})
        out["wall_s"] = time.time() - t0
        return out

    def state_dict(self):
        return {"kind": "ppo", "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "obs_rms": self.obs_rms.state(), "hp": self.hp, "d": self.d, "B": self.B}

    @classmethod
    def from_state(cls, s):
        hp = dict(s["hp"])
        p = cls(s["d"], s["B"], critic=hp["critic"], hidden=tuple(hp["hidden"]), action_map=hp.get("action_map", "direct"))
        p.actor.load_state_dict(s["actor"]); p.critic.load_state_dict(s["critic"])
        p.obs_rms.load(s["obs_rms"])
        return p


class SquashedActor(nn.Module):
    def __init__(self, d, n_act=3, h=(128, 128), init=INIT_SAC_MEAN):
        super().__init__()
        self.net = mlp(d, 2 * n_act, h, act=nn.ReLU)
        self.n = n_act
        last = self.net[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)
            for j, m in enumerate(init):
                last.bias[j] = float(np.arctanh(2 * m - 1))
                last.bias[n_act + j] = -0.5

    def forward(self, x, deterministic=False):
        o = self.net(x)
        mu, ls = o[..., :self.n], o[..., self.n:].clamp(-5.0, 2.0)
        if deterministic:
            return 0.5 * (torch.tanh(mu) + 1.0), None
        std = ls.exp()
        z = mu + std * torch.randn_like(mu)
        y = torch.tanh(z)
        lp = (-0.5 * ((z - mu) / std) ** 2 - ls - 0.5 * np.log(2 * np.pi)).sum(-1)
        lp = lp - torch.log(1 - y ** 2 + 1e-6).sum(-1) - self.n * np.log(0.5)
        return 0.5 * (y + 1.0), lp


class SAC:
    def __init__(self, obs_dim: int, B: int, *, seed: int = 0, lr: float = 3e-4, gamma: float = 0.995,
                 tau: float = 0.005, batch: int = 256, buffer: int = 400_000, warmup: int = 4000,
                 updates_per_step: float = 1.0, hidden=(128, 128), reward_scale: float = 1.0,
                 target_entropy: Optional[float] = None, obs_norm_steps: int = 4368,
                 action_map: str = "need_relative"):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.B, self.d = B, obs_dim
        self.actor = SquashedActor(obs_dim, 3, hidden)
        self.q1, self.q2 = mlp(obs_dim + 3, 1, hidden, nn.ReLU), mlp(obs_dim + 3, 1, hidden, nn.ReLU)
        self.q1t, self.q2t = mlp(obs_dim + 3, 1, hidden, nn.ReLU), mlp(obs_dim + 3, 1, hidden, nn.ReLU)
        self.q1t.load_state_dict(self.q1.state_dict()); self.q2t.load_state_dict(self.q2.state_dict())
        self.log_alpha = torch.tensor(np.log(0.1), requires_grad=True)
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_q = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.opt_al = torch.optim.Adam([self.log_alpha], lr=lr)
        self.tgt_ent = -3.0 if target_entropy is None else float(target_entropy)
        self.hp = dict(lr=lr, gamma=gamma, tau=tau, batch=batch, buffer=buffer, warmup=warmup,
                       updates_per_step=updates_per_step, hidden=list(hidden), reward_scale=reward_scale,
                       target_entropy=self.tgt_ent, obs_norm_steps=obs_norm_steps, action_map=action_map)
        self.obs_rms = RunningMeanStd((obs_dim,))
        self.frozen = False
        n = buffer
        self.bX = np.zeros((n, obs_dim), np.float32); self.bX2 = np.zeros((n, obs_dim), np.float32)
        self.bA = np.zeros((n, 3), np.float32); self.bR = np.zeros(n, np.float32); self.bD = np.zeros(n, np.float32)
        self.ptr, self.size, self.steps = 0, 0, 0
        self._acc = 0.0

    def act(self, X, deterministic=False):
        with torch.no_grad():
            a, _ = self.actor(torch.as_tensor(self.obs_rms.norm(X)), deterministic)
        return a.numpy()

    def env_action(self, X):
        return to_env_action(self.act(X, deterministic=True), X, self.hp.get("action_map", "direct"))

    def _store(self, X, A, R, X2, D):
        n = len(R)
        i = (self.ptr + np.arange(n)) % self.hp["buffer"]
        self.bX[i], self.bA[i], self.bR[i], self.bX2[i], self.bD[i] = X, A, R, X2, D
        self.ptr = (self.ptr + n) % self.hp["buffer"]
        self.size = min(self.size + n, self.hp["buffer"])

    def _update(self):
        hp = self.hp
        i = self.rng.integers(0, self.size, hp["batch"])
        x = torch.as_tensor(self.obs_rms.norm(self.bX[i])); x2 = torch.as_tensor(self.obs_rms.norm(self.bX2[i]))
        a, r, d = torch.as_tensor(self.bA[i]), torch.as_tensor(self.bR[i]), torch.as_tensor(self.bD[i])
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            a2, lp2 = self.actor(x2)
            xa2 = torch.cat([x2, a2], -1)
            qt = torch.min(self.q1t(xa2), self.q2t(xa2)).squeeze(-1) - alpha * lp2
            y = r + hp["gamma"] * (1 - d) * qt
        xa = torch.cat([x, a], -1)
        lq = F.mse_loss(self.q1(xa).squeeze(-1), y) + F.mse_loss(self.q2(xa).squeeze(-1), y)
        self.opt_q.zero_grad(); lq.backward(); self.opt_q.step()
        an, lp = self.actor(x)
        xan = torch.cat([x, an], -1)
        la = (alpha * lp - torch.min(self.q1(xan), self.q2(xan)).squeeze(-1)).mean()
        self.opt_a.zero_grad(); la.backward(); self.opt_a.step()
        lal = -(self.log_alpha * (lp.detach() + self.tgt_ent)).mean()
        self.opt_al.zero_grad(); lal.backward(); self.opt_al.step()
        with torch.no_grad():
            for q, qt_ in ((self.q1, self.q1t), (self.q2, self.q2t)):
                for p, pt in zip(q.parameters(), qt_.parameters()):
                    pt.mul_(1 - hp["tau"]).add_(hp["tau"] * p)
        return float(lq), float(la), float(alpha), float(-lp.mean())

    def train_episode(self, env, extra=None) -> Dict:
        ob = ObsBuilder(env)
        obs, _ = env.reset()
        ob.reset()
        X = ob.build(obs)
        st = []
        t0 = time.time()
        while True:
            if not self.frozen:
                self.obs_rms.update(X)
                if self.steps >= self.hp["obs_norm_steps"]:
                    self.frozen = True
            a = self.act(X)
            obs, r, term, _, info = env.step(to_env_action(a, X, self.hp["action_map"]))
            ob.after_step(info)
            X2 = ob.build(obs) if not term else X
            r = np.asarray(r, float) - (extra(info) if extra is not None else 0.0)
            self._store(X, a, r * self.hp["reward_scale"], X2,
                        np.full(self.B, 1.0 if term else 0.0))
            self.steps += 1
            if self.size >= self.hp["warmup"]:
                self._acc += self.hp["updates_per_step"]
                while self._acc >= 1.0:
                    st.append(self._update())
                    self._acc -= 1.0
            if term:
                break
            X = X2
        out = episode_summary(env)
        if st:
            s = np.mean(st, axis=0)
            out.update(l_q=s[0], l_pi=s[1], alpha=s[2], entropy=s[3])
        out["wall_s"] = time.time() - t0
        return out

    def state_dict(self):
        return {"kind": "sac", "actor": self.actor.state_dict(), "obs_rms": self.obs_rms.state(), "hp": self.hp,
                "d": self.d, "B": self.B}

    @classmethod
    def from_state(cls, s):
        hp = dict(s["hp"])
        p = cls(s["d"], s["B"], hidden=tuple(hp["hidden"]), buffer=8, action_map=hp.get("action_map", "direct"))
        p.actor.load_state_dict(s["actor"])
        p.obs_rms.load(s["obs_rms"])
        p.size = 10 ** 9
        return p


def load_policy(path):
    s = torch.load(path, map_location="cpu", weights_only=False)
    return (PPO if s["kind"] == "ppo" else SAC).from_state(s)
