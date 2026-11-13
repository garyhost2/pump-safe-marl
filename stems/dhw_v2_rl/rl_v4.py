from __future__ import annotations

import copy
import time
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stems.dhw_v2_rl.learners import RunningMeanStd, mlp
from stems.dhw_v2_rl.mpc_v3 import need_frac
from stems.dhw_v2_rl.rl_v3 import GaussPolicy, ObsV3, SetpointLayer, PREHEAT_BASE

torch.set_num_threads(1)
TRAIN_YEARS = (2022, 2023)
D_OBS = 101


def pref():
    from stems.dhw_v2_rl.rl_v3 import train_price_ref
    return train_price_ref("karlstad", TRAIN_YEARS)
SP_SCALE = 3.0
BAND = (20.0, 23.0)


class LayerV4(SetpointLayer):
    def __init__(self, B: int, preheat: Optional[Dict] = None):
        super().__init__(B, "residual", preheat)
        self.last = None

    def base_obs(self, env):
        thr, dis, sp = self.baseline(env)
        self.pa_on = self._hyst(self.pa_on, env.plant.T, *thr)
        u_base = (self.pa_on | dis).astype(float)
        self.last = (sp, u_base, dis)
        return sp, u_base

    def act(self, env, delta) -> Dict:
        delta = np.clip(np.asarray(delta, float).reshape(self.B, 2), -1.0, 1.0)
        sp, u_base, dis = self.last
        u_d = np.where(dis, 1.0, np.clip(u_base + delta[:, 0], 0.0, 1.0))
        target = np.clip(sp + SP_SCALE * delta[:, 1], *BAND)
        u_s = np.minimum(need_frac(env, target), 1.0 - u_d)
        return {"u_d": u_d, "u_s": u_s, "dis": dis}

    def label(self, mpc_cmd, plan, dhw=True):
        sp, u_base, dis = self.last
        plan = np.asarray(plan, float)
        ok = np.isfinite(plan)
        ds = np.where(ok, np.clip((np.where(ok, plan, sp) - sp) / SP_SCALE, -1.0, 1.0), 0.0)
        dd = np.clip(np.asarray(mpc_cmd["u_d"], float) - u_base, -1.0, 1.0) if dhw else np.zeros(self.B)
        dd = np.where(ok & ~dis, dd, 0.0)
        return np.stack([dd, ds], 1), ok


class PolicyV4:

    def __init__(self, seed=0, hidden=(128, 128), log_std0=np.log(0.3)):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.pi = GaussPolicy(D_OBS, 2, hidden, log_std0=log_std0, zero_init=True)
        self.obs_rms = RunningMeanStd((D_OBS,))
        self.hidden = list(hidden)

    def norm(self, X):
        return torch.as_tensor(self.obs_rms.norm(X))

    def mean(self, X):
        with torch.no_grad():
            return self.pi.mu(self.norm(X)).numpy()

    def fit_bc(self, X, Y, steps=6000, lr=1e-3, batch=1024, W=None):
        self.obs_rms = RunningMeanStd((D_OBS,))
        self.obs_rms.update(X)
        Xn = self.norm(X)
        Yt = torch.as_tensor(Y, dtype=torch.float32)
        opt = torch.optim.Adam(self.pi.mu.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        n, hist = len(Xn), []
        for s in range(steps):
            i = torch.as_tensor(self.rng.integers(0, n, batch))
            loss = F.mse_loss(self.pi.mu(Xn[i]), Yt[i])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if s % 500 == 0 or s == steps - 1:
                hist.append(float(loss))
        with torch.no_grad():
            P = torch.cat([self.pi.mu(Xn[j:j + 65536]) for j in range(0, n, 65536)])
            per = ((P - Yt) ** 2).mean(0).numpy()
        return {"loss_hist": hist, "mse_per_dim": per.tolist(), "target_sd": Y.std(0).tolist(), "n": int(n)}

    def state_dict(self):
        return {"pi": self.pi.state_dict(), "obs_rms": self.obs_rms.state(), "hidden": self.hidden}

    @classmethod
    def load(cls, path_or_sd, seed=0):
        sd = torch.load(path_or_sd, weights_only=False) if isinstance(path_or_sd, (str, bytes)) or hasattr(path_or_sd, "open") else path_or_sd
        p = cls(seed, tuple(sd.get("hidden", (128, 128))))
        p.pi.load_state_dict(sd["pi"])
        p.obs_rms.load(sd["obs_rms"])
        return p


def train_extra(info):
    from experiments.dhw_v2_rl.train_v3 import extra_v3
    return extra_v3(info)


def rollout(env, layer, policy: Optional[PolicyV4] = None, *, expert=None, expert_exec=False, beta=0.0, noise=0.0,
            rng=None, record=False, dhw_label=True, extra: Optional[Callable] = None):
    from stems.dhw_v2_rl.common import run_episode
    rng = rng if rng is not None else np.random.default_rng(0)
    ob = ObsV3(env, pref())
    layer.reset()
    if expert is not None and hasattr(expert, "reset"):
        expert.reset()
    B = env.B
    D = {k: [] for k in ("X", "A", "Y", "ok", "R")}
    orig = env.step_command
    rbuf = {}

    def sc(cmd):
        out = orig(cmd)
        rbuf["r"] = np.asarray(out[1], float)
        return out

    env.step_command = sc

    def act(e, _X):
        ob.base_state = layer.base_obs(e)
        X = ob.build()
        y = ok = None
        if expert is not None:
            cmd = expert.act(e)
            y, ok = layer.label(cmd, expert.plan_t_in, dhw=dhw_label)
        if expert_exec:
            a = y
        else:
            a = policy.mean(X) if policy is not None else np.zeros((B, 2))
            if noise > 0:
                a = a + noise * rng.standard_normal(a.shape)
            a = np.clip(a, -1.0, 1.0)
            if beta > 0 and y is not None:
                m = rng.random(B) < beta
                a = np.where(m[:, None], y, a)
        if record:
            D["X"].append(X.astype(np.float16)); D["A"].append(a.astype(np.float32))
            D["Y"].append((y if y is not None else np.full((B, 2), np.nan)).astype(np.float32))
            D["ok"].append(ok if ok is not None else np.zeros(B, bool))
        return layer.act(e, a)

    def upd(rec):
        ob.after_step(rec)
        if record:
            r = rbuf["r"] - (extra({"record": rec}) if extra is not None else 0.0)
            D["R"].append(r.astype(np.float32))

    S = run_episode(env, act, update=upd)
    env.step_command = orig
    data = {k: np.stack(v) for k, v in D.items()} if record else None
    return S, data


class AnchoredPPO:

    def __init__(self, init: PolicyV4, B, *, seed=0, lr_pi=3e-5, lr_v=3e-4, gamma=0.99, lam=0.95, clip=0.1, epochs=5,
                 minibatch=512, rollout=546, max_grad=0.5, reward_scale=1.0, total_updates=600, log_std0=np.log(0.1),
                 anchor_w0=10.0, anchor_w1=1.0, critic_warmup=20):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.B = B
        self.pol = copy.deepcopy(init)
        with torch.no_grad():
            self.pol.pi.log_std.fill_(float(log_std0))
        self.anchor = copy.deepcopy(init.pi.mu).eval()
        for p in self.anchor.parameters():
            p.requires_grad_(False)
        self.v = mlp(D_OBS, 1, tuple(init.hidden))
        self.opt_pi = torch.optim.Adam(self.pol.pi.parameters(), lr=lr_pi, eps=1e-5)
        self.opt_v = torch.optim.Adam(self.v.parameters(), lr=lr_v, eps=1e-5)
        self.hp = dict(lr_pi=lr_pi, lr_v=lr_v, gamma=gamma, lam=lam, clip=clip, epochs=epochs, minibatch=minibatch,
                       rollout=rollout, max_grad=max_grad, reward_scale=reward_scale, total_updates=total_updates,
                       log_std0=float(log_std0), anchor_w0=anchor_w0, anchor_w1=anchor_w1, critic_warmup=critic_warmup)
        self.ret_rms = RunningMeanStd(())
        self.ret = np.zeros(B)
        self.n_updates = 0

    def act(self, X):
        x = self.pol.norm(X)
        with torch.no_grad():
            d = self.pol.pi.dist(x)
            a = d.sample()
            return a.numpy(), d.log_prob(a).sum(-1).numpy(), self.v(x).squeeze(-1).numpy()

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
        frac = min(self.n_updates / max(hp["total_updates"], 1), 1.0)
        w = hp["anchor_w0"] + (hp["anchor_w1"] - hp["anchor_w0"]) * frac
        warm = self.n_updates < hp["critic_warmup"]
        Xn = self.pol.norm(Xs)
        At, LPt = torch.as_tensor(A, dtype=torch.float32), torch.as_tensor(LP, dtype=torch.float32)
        ADV, RET = torch.as_tensor(adv, dtype=torch.float32), torch.as_tensor(ret, dtype=torch.float32)
        mb_t = max(1, hp["minibatch"] // self.B)
        st = []
        for _ in range(hp["epochs"]):
            perm = self.rng.permutation(T)
            for s in range(0, T, mb_t):
                i = torch.as_tensor(perm[s:s + mb_t])
                x = Xn[i]
                l_v = F.mse_loss(self.v(x).squeeze(-1), RET[i])
                self.opt_v.zero_grad(); l_v.backward()
                nn.utils.clip_grad_norm_(self.v.parameters(), hp["max_grad"]); self.opt_v.step()
                if warm:
                    st.append((0.0, float(l_v), 0.0)); continue
                ad = ADV[i]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                dist = self.pol.pi.dist(x)
                ratio = torch.exp(dist.log_prob(At[i]).sum(-1) - LPt[i])
                l_pi = -torch.min(ratio * ad, ratio.clamp(1 - hp["clip"], 1 + hp["clip"]) * ad).mean()
                var = dist.scale.detach() ** 2
                l_a = (((dist.mean - self.anchor(x)) ** 2) / (2 * var)).sum(-1).mean()
                loss = l_pi + w * l_a / 100.0
                self.opt_pi.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.pol.pi.parameters(), hp["max_grad"]); self.opt_pi.step()
                st.append((float(l_pi), float(l_v), float(l_a)))
        self.n_updates += 1
        return np.mean(st, axis=0), w


def ppo_episode(env, ag: AnchoredPPO, layer, extra):
    from stems.dhw_v2_rl.common import run_episode
    ob = ObsV3(env, pref())
    layer.reset()
    env.reset()
    buf = {k: [] for k in ("X", "A", "LP", "V", "R")}
    stats = []
    while True:
        ob.base_state = layer.base_obs(env)
        X = ob.build()
        a, lp, v = ag.act(X)
        cmd = layer.act(env, a)
        _, r, term, _, info = env.step_command(cmd)
        ob.after_step(info["record"])
        r = np.asarray(r, float) - extra(info)
        buf["X"].append(X); buf["A"].append(a); buf["LP"].append(lp); buf["V"].append(v); buf["R"].append(ag.scale_reward(r))
        if len(buf["R"]) >= ag.hp["rollout"] or term:
            if term:
                last_v = np.zeros(env.B)
            else:
                with torch.no_grad():
                    last_v = ag.v(ag.pol.norm(X)).squeeze(-1).numpy()
            stats.append(ag.update(np.stack(buf["X"]), np.stack(buf["A"]), np.stack(buf["LP"]), np.stack(buf["V"]),
                                   np.stack(buf["R"]), last_v, term))
            buf = {k: [] for k in buf}
        if term:
            break
    ag.ret[:] = 0.0
    return stats


class Replay:

    def __init__(self):
        self.S, self.A, self.R, self.S2, self.Dn = [], [], [], [], []
        self._cat = None

    def add_episode(self, X, A, R):
        T, B, _ = X.shape
        X = X.astype(np.float32)
        Xn = np.concatenate([X[1:], X[-1:]], 0)
        dn = np.zeros((T, B), np.float32); dn[-1] = 1.0
        self.S.append(X.reshape(-1, D_OBS)); self.S2.append(Xn.reshape(-1, D_OBS))
        self.A.append(A.reshape(-1, 2).astype(np.float32)); self.R.append(R.reshape(-1).astype(np.float32))
        self.Dn.append(dn.reshape(-1))
        self._cat = None

    def cat(self):
        if self._cat is None:
            self._cat = tuple(torch.as_tensor(np.concatenate(v)) for v in (self.S, self.A, self.R, self.S2, self.Dn))
        return self._cat

    def __len__(self):
        return sum(len(r) for r in self.R)


class IQL:
    def __init__(self, init: PolicyV4, *, seed=0, gamma=0.99, tau=0.7, beta=3.0, lr=3e-4, polyak=0.005, hidden=(256, 256),
                 reward_scale=1.0, batch=512, adv_clip=100.0):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.pol = copy.deepcopy(init)
        self.rms = init.obs_rms
        self.q1, self.q2 = mlp(D_OBS + 2, 1, hidden, nn.ReLU), mlp(D_OBS + 2, 1, hidden, nn.ReLU)
        self.q1t, self.q2t = copy.deepcopy(self.q1), copy.deepcopy(self.q2)
        self.v = mlp(D_OBS, 1, hidden, nn.ReLU)
        self.oq = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        self.ov = torch.optim.Adam(self.v.parameters(), lr=lr)
        self.op = torch.optim.Adam(self.pol.pi.parameters(), lr=lr)
        self.hp = dict(gamma=gamma, tau=tau, beta=beta, lr=lr, polyak=polyak, hidden=list(hidden), reward_scale=reward_scale,
                       batch=batch, adv_clip=adv_clip)
        self.mu_r = torch.as_tensor(self.rms.mean, dtype=torch.float32)
        self.sd_r = torch.as_tensor(np.sqrt(self.rms.var + 1e-8), dtype=torch.float32)

    def n(self, S):
        return ((S - self.mu_r) / self.sd_r).clamp(-10, 10)

    def step(self, rep: Replay):
        hp = self.hp
        S, A, R, S2, Dn = rep.cat()
        i = torch.as_tensor(self.rng.integers(0, len(R), hp["batch"]))
        s, a, r, s2, d = self.n(S[i]), A[i], R[i] * hp["reward_scale"], self.n(S2[i]), Dn[i]
        sa = torch.cat([s, a], 1)
        with torch.no_grad():
            qt = torch.min(self.q1t(sa), self.q2t(sa)).squeeze(-1)
            v2 = self.v(s2).squeeze(-1)
        v = self.v(s).squeeze(-1)
        u = qt - v
        lv = (torch.abs(hp["tau"] - (u < 0).float()) * u ** 2).mean()
        self.ov.zero_grad(); lv.backward(); self.ov.step()
        tgt = r + hp["gamma"] * (1 - d) * v2
        lq = F.mse_loss(self.q1(sa).squeeze(-1), tgt) + F.mse_loss(self.q2(sa).squeeze(-1), tgt)
        self.oq.zero_grad(); lq.backward(); self.oq.step()
        with torch.no_grad():
            w = torch.exp(hp["beta"] * u.detach()).clamp(max=hp["adv_clip"])
        dist = self.pol.pi.dist(s)
        lp = -(w * dist.log_prob(a).sum(-1)).mean()
        self.op.zero_grad(); lp.backward(); self.op.step()
        with torch.no_grad():
            for q, qt_ in ((self.q1, self.q1t), (self.q2, self.q2t)):
                for p, pt in zip(q.parameters(), qt_.parameters()):
                    pt.mul_(1 - hp["polyak"]).add_(hp["polyak"] * p)
        return float(lv), float(lq), float(lp), float(u.std()), float(w.mean())
