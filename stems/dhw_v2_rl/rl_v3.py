from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stems.dhw_v2_rl.learners import RunningMeanStd, mlp
from stems.dhw_v2_rl.mpc_v3 import PreheatRule, need_frac

torch.set_num_threads(1)

PREHEAT_BASE = {"k_cheap": 4, "k_dear": 8, "t_hi": 23.0}
BAND = (20.0, 23.0)
_PRICE_REF = {}


def train_price_ref(site="karlstad", years=(2022, 2023, 2024)):
    k = (site, tuple(years))
    if k not in _PRICE_REF:
        from stems.dhw_v2_rl.common import load_site
        d, _ = load_site(site, list(years))
        _PRICE_REF[k] = float(np.mean(d.price))
    return _PRICE_REF[k]


class ObsV3:

    def __init__(self, env, price_ref: Optional[float] = None):
        self.env = env
        self.pref = price_ref if price_ref is not None else train_price_ref()
        self.recent = np.zeros((3, env.B))
        self.base_state = (21.0, np.zeros(env.B))

    def after_step(self, rec):
        self.recent = np.roll(self.recent, 1, axis=0)
        self.recent[0] = np.asarray(rec["draw_kwh"], float)

    def build(self) -> np.ndarray:
        e = self.env
        B, t, d = e.B, e.t, e.data
        if t >= d.T:
            return np.zeros((B, 101), np.float32)
        idx = np.minimum(np.arange(t, t + 24), d.T - 1)
        hod, dow = d.hour_of_day[t], d.day_of_week[t]
        fc = e.filter.forecaster if e.filter is not None else e.shadow_fc
        dr = np.stack([fc.forecast(int(hod) + k) for k in range(24)], axis=1)
        ctx = e.plant.context()
        tin = np.array([getattr(b, "t_in", 21.0) for b in e.plant.buildings])
        thm = np.array([getattr(b, "theta_m", getattr(b, "t_in", 21.0)) for b in e.plant.buildings])
        q = np.asarray(ctx["q_max_s"], float)
        need = np.where(q > 0, np.asarray(ctx["sh_need_set_kw"], float) / np.where(q > 0, q, 1.0), 0.0)
        since = np.where(e.last_hold >= 0, t - e.last_hold, t + 168)
        p24 = d.price[idx]
        sp, bon = self.base_state
        cols = [np.tile([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24), np.sin(2 * np.pi * dow / 7),
                         np.cos(2 * np.pi * dow / 7)], (B, 1)),
                np.tile(p24 / self.pref, (B, 1)), np.tile(d.t_out_c[idx] / 10.0, (B, 1)), dr, self.recent.T,
                e.plant.T / 60.0, np.stack([tin - 21.0, thm - 21.0], 1),
                np.stack([np.asarray(ctx["t_src_s"], float) / 10.0, np.asarray(ctx["t_src_d"], float) / 10.0], 1),
                np.stack([q / 12.0, np.clip(need, 0, 2)], 1),
                np.stack([np.full(B, e.week_left[t] / 168.0), e.week_done.astype(float), np.minimum(since / 168.0, 2.0)], 1),
                np.stack([np.full(B, sp - 21.0), np.asarray(bon, float), np.full(B, float(p24[0] <= 0.95 * p24.mean()))], 1)]
        return np.concatenate(cols, axis=1).astype(np.float32)


class SetpointLayer:

    CHEAP, DEAR = (50.0, 47.0, 53.0), (30.0, 47.0, 40.0)

    def __init__(self, B: int, mode: str = "residual", preheat: Optional[Dict] = None):
        self.B, self.mode = B, mode
        self.rule = PreheatRule(B, **(preheat or PREHEAT_BASE))
        self.on = np.zeros(B, dtype=bool)
        self.pa_on = np.zeros(B, dtype=bool)

    def reset(self):
        self.on[:] = False
        self.pa_on[:] = False

    def _hyst(self, on, T, on_b, on_t, off_b):
        T_top, T_bot = T[:, 0], T[:, -1]
        return np.where(on, T_bot < off_b, (T_bot < on_b) | (T_top < on_t))

    def baseline(self, env):
        p = env.price_path(24)
        ctx = env.plant.context()
        cop = np.asarray(ctx["r_d"]) / np.asarray(ctx["p_d"])
        c = p[:, None] / cop[None, :]
        cheap = c[0] <= 0.95 * c.mean(axis=0)
        thr = [np.where(cheap, a, b) for a, b in zip(self.CHEAP, self.DEAR)]
        elapsed = 168 - env.hours_left_week()
        dis = (~env.week_done) & (elapsed >= 120.0) & cheap
        return thr, dis, self.rule.setpoint(env)

    def base_obs(self, env):
        thr, dis, sp = self.baseline(env)
        self.pa_on = self._hyst(self.pa_on, env.plant.T, *thr)
        return sp, (self.pa_on | dis).astype(float)

    def act(self, env, delta) -> Dict:
        delta = np.clip(np.asarray(delta, float).reshape(self.B, 2), -1.0, 1.0)
        thr, dis, sp = self.baseline(env)
        if self.mode == "residual":
            s = 8.0 * delta[:, 0]
            off_b = np.minimum(thr[2] + s, 54.0)
            on_b = np.minimum(thr[0] + s, off_b - 2.0)
            target = np.clip(sp + 1.5 * delta[:, 1], *BAND)
        else:
            off_b = np.minimum(46.0 + 8.0 * delta[:, 0], 54.0)
            on_b = off_b - 8.0
            target = np.clip(21.5 + 1.5 * delta[:, 1], *BAND)
        self.on = self._hyst(self.on, env.plant.T, on_b, thr[1], off_b)
        u_d = np.where(self.on | dis, 1.0, 0.0)
        u_s = np.minimum(need_frac(env, target), 1.0 - u_d)
        return {"u_d": u_d, "u_s": u_s, "dis": dis}


class GaussPolicy(nn.Module):
    def __init__(self, d, n=2, h=(128, 128), log_std0=np.log(0.3), zero_init=True):
        super().__init__()
        self.mu = mlp(d, n, h)
        if zero_init:
            with torch.no_grad():
                self.mu[-1].weight.zero_()
                self.mu[-1].bias.zero_()
        self.log_std = nn.Parameter(torch.full((n,), float(log_std0)))

    def dist(self, x):
        return torch.distributions.Normal(self.mu(x), self.log_std.exp().expand(x.shape[:-1] + (self.log_std.shape[0],)))


class PPOv3:
    def __init__(self, d, B, *, seed=0, lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, epochs=10, minibatch=512,
                 rollout=546, vf_coef=0.5, max_grad=0.5, reward_scale=1.0, total_updates=1000,
                 zero_init=True, bc_coef=0.0, bc_frac=0.3, hidden=(128, 128)):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.d, self.B = d, B
        self.pi = GaussPolicy(d, 2, hidden, zero_init=zero_init)
        self.v = mlp(d, 1, hidden)
        self.opt = torch.optim.Adam(list(self.pi.parameters()) + list(self.v.parameters()), lr=lr, eps=1e-5)
        self.hp = dict(lr=lr, gamma=gamma, lam=lam, clip=clip, epochs=epochs, minibatch=minibatch, rollout=rollout,
                       vf_coef=vf_coef, max_grad=max_grad, reward_scale=reward_scale, total_updates=total_updates,
                       zero_init=zero_init, bc_coef=bc_coef, bc_frac=bc_frac, hidden=list(hidden))
        self.obs_rms = RunningMeanStd((d,))
        self.freeze_obs = False
        self.ret_rms = RunningMeanStd(())
        self.ret = np.zeros(B)
        self.n_updates = 0
        self.bc_pi = None

    def norm(self, X):
        return torch.as_tensor(self.obs_rms.norm(X))

    def act(self, X, deterministic=False):
        x = self.norm(X)
        with torch.no_grad():
            d = self.pi.dist(x)
            if deterministic:
                return d.mean.numpy()
            a = d.sample()
            return a.numpy(), d.log_prob(a).sum(-1).numpy(), self.v(x).squeeze(-1).numpy()

    def fit_bc(self, X, Y, epochs=30, lr=1e-3, batch=1024):
        import copy
        self.obs_rms.update(X)
        self.freeze_obs = True
        Xn = self.norm(X)
        Yt = torch.as_tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(self.pi.mu.parameters(), lr=lr)
        n = len(Xn)
        hist = []
        for ep in range(epochs):
            perm = self.rng.permutation(n)
            tot = 0.0
            for s in range(0, n, batch):
                i = torch.as_tensor(perm[s:s + batch])
                loss = F.mse_loss(self.pi.mu(Xn[i]), Yt[i])
                opt.zero_grad(); loss.backward(); opt.step()
                tot += float(loss) * len(i)
            hist.append(tot / n)
        self.bc_pi = copy.deepcopy(self.pi.mu).eval()
        for p in self.bc_pi.parameters():
            p.requires_grad_(False)
        return hist

    def scale_reward(self, r):
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
            g = R[t] + hp["gamma"] * nv - V[t] + hp["gamma"] * hp["lam"] * g
            adv[t] = g
        ret = adv + V
        frac = self.n_updates / max(hp["total_updates"], 1)
        for pg in self.opt.param_groups:
            pg["lr"] = hp["lr"] * max(1.0 - frac, 0.05)
        bc_w = hp["bc_coef"] * max(0.0, 1.0 - frac / hp["bc_frac"]) if self.bc_pi is not None else 0.0
        Xn = self.norm(Xs)
        At, LPt = torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(LP, dtype=torch.float32)
        ADV, RET = torch.as_tensor(adv, dtype=torch.float32), torch.as_tensor(ret, dtype=torch.float32)
        st = []
        mb_t = max(1, hp["minibatch"] // self.B)
        for _ in range(hp["epochs"]):
            perm = self.rng.permutation(T)
            for s in range(0, T, mb_t):
                i = torch.as_tensor(perm[s:s + mb_t])
                x = Xn[i]
                ad = ADV[i]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                dist = self.pi.dist(x)
                ratio = torch.exp(dist.log_prob(At[i]).sum(-1) - LPt[i])
                l_pi = -torch.min(ratio * ad, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * ad).mean()
                l_v = F.mse_loss(self.v(x).squeeze(-1), RET[i])
                loss = l_pi + hp["vf_coef"] * l_v
                if bc_w > 0:
                    loss = loss + bc_w * F.mse_loss(dist.mean, self.bc_pi(x))
                self.opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(self.pi.parameters()) + list(self.v.parameters()), hp["max_grad"])
                self.opt.step()
                st.append((float(l_pi), float(l_v)))
        self.n_updates += 1
        return np.mean(st, axis=0), bc_w

    def state_dict(self):
        return {"pi": self.pi.state_dict(), "v": self.v.state_dict(), "obs_rms": self.obs_rms.state(), "hp": self.hp,
                "d": self.d, "B": self.B}


def run_v3(env, pol, layer, *, train=False, extra=None, price_ref=None):
    from stems.dhw_v2_rl.common import run_episode
    ob = ObsV3(env, price_ref)
    layer.reset()
    if not train:
        def act(e, _X):
            ob.base_state = layer.base_obs(e)
            X = ob.build()
            delta = pol.act(X, deterministic=True) if pol is not None else np.zeros((e.B, 2))
            return layer.act(e, delta)
        return run_episode(env, act, update=lambda r: ob.after_step(r))
    obs, _ = env.reset()
    buf = {k: [] for k in ("X", "A", "LP", "V", "R")}
    hp = pol.hp
    stats = []
    t0 = time.time()
    while True:
        ob.base_state = layer.base_obs(env)
        X = ob.build()
        if not pol.freeze_obs:
            pol.obs_rms.update(X)
        a, lp, v = pol.act(X)
        cmd = layer.act(env, a)
        obs, r, term, _, info = env.step_command(cmd)
        ob.after_step(info["record"])
        r = np.asarray(r, float) - (extra(info) if extra is not None else 0.0)
        buf["X"].append(X); buf["A"].append(a); buf["LP"].append(lp); buf["V"].append(v); buf["R"].append(pol.scale_reward(r))
        if len(buf["R"]) >= hp["rollout"] or term:
            if term:
                last_v = np.zeros(env.B)
            else:
                ob.base_state = layer.base_obs_peek(env) if hasattr(layer, "base_obs_peek") else ob.base_state
                with torch.no_grad():
                    last_v = pol.v(pol.norm(ob.build())).squeeze(-1).numpy()
            stats.append(pol.update(np.stack(buf["X"]), np.stack(buf["A"]), np.stack(buf["LP"]), np.stack(buf["V"]),
                                    np.stack(buf["R"]), last_v, term))
            buf = {k: [] for k in buf}
        if term:
            break
    pol.ret[:] = 0.0
    return stats, time.time() - t0
