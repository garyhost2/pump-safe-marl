from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from stems.dhw_v2.controllers import sh_fraction
from stems.dhw_v2 import tank as tk

CV_KWH_PER_L_K = 4.186 * 0.983 / 3600.0


class MPCController:
    name = "mpc"

    def __init__(self, B: int, *, horizon: int = 24, rho: float = 0.5, forecast: str = "causal",
                 request_after_h: float = 120.0, cheap_margin: float = 0.05):
        self.B, self.H, self.rho, self.forecast = B, int(horizon), float(rho), forecast
        self.request_after_h, self.cheap_margin = request_after_h, cheap_margin
        self.lp_fail = 0
        self._A = None

    def reset(self):
        self.lp_fail = 0

    def _matrices(self, B, H):
        n = 3 * H
        rows, cols, vals = [], [], []
        for b in range(B):
            o = b * n
            for k in range(H):
                r = b * H + k
                rows += [r, r]; cols += [o + H + k, o + k]; vals += [1.0, -1.0]
                if k > 0:
                    rows.append(r); cols.append(o + H + k - 1); vals.append(-1.0)
        Aeq = coo_matrix((vals, (rows, cols)), shape=(B * H, B * n)).tocsr()
        rows, cols, vals = [], [], []
        for b in range(B):
            o = b * n
            for k in range(H):
                r = b * H + k
                rows += [r, r]; cols += [o + H + k, o + 2 * H + k]; vals += [-1.0, -1.0]
        Aub = coo_matrix((vals, (rows, cols)), shape=(B * H, B * n)).tocsr()
        return Aeq, Aub

    def act(self, env) -> Dict:
        pl = env.plant
        ctx = pl.context()
        B, H = self.B, self.H
        t = env.t
        T = pl.T
        vol = pl.tank["node_vol"]
        vol = np.broadcast_to(np.asarray(vol, float).reshape(-1, 1) if np.ndim(vol) else vol, T.shape)
        tc = np.asarray(ctx["t_cold"], float)
        Cn = vol * CV_KWH_PER_L_K
        Ctot = Cn.sum(axis=1)
        E0 = np.sum(Cn * (T - tc[:, None]), axis=1)
        ua = np.asarray(pl.tank["ua"], float)
        t_room = np.asarray(pl.tank["t_room"], float)
        loss = np.maximum(ua * (T.mean(axis=1) - t_room), 0.0) / 1000.0
        r_d = np.asarray(ctx["r_d"], float)
        cop = r_d / np.maximum(np.asarray(ctx["p_d"], float), 1e-6)
        price = env.price_path(H)
        hour = int(env.data.hour_of_day[t])
        if self.forecast == "perfect":
            idx = np.minimum(np.arange(t, t + H), env.data.T - 1)
            d = np.stack([pl.draws[i].sum(axis=0) for i in idx])
        else:
            fc = env.filter.forecaster if env.filter is not None else env.shadow_fc
            m = env.filter.margin.m if env.filter is not None else env.shadow_m.m
            d = np.stack([fc.forecast(hour + k) for k in range(H)]) + m[None, :]
        Emin = self.rho * Ctot * (45.0 - tc)
        Emax = np.maximum(Ctot * (55.0 - tc), E0)
        if self._A is None or self._A[0] != (B, H):
            self._A = ((B, H),) + self._matrices(B, H)
        _, Aeq, Aub = self._A
        n = 3 * H
        c = np.zeros(B * n)
        beq = np.zeros(B * H)
        bub = np.zeros(B * H)
        bounds = []
        M = 100.0 * float(np.max(np.abs(price)) + 1e-3)
        for b in range(B):
            o = b * n
            pc = price / cop[b]
            c[o:o + H] = pc
            c[o + 2 * H:o + 3 * H] = M
            c[o + 2 * H - 1] = -float(np.mean(pc))
            beq[b * H:(b + 1) * H] = -(d[:, b] + loss[b])
            beq[b * H] += E0[b]
            bub[b * H:(b + 1) * H] = -Emin[b]
            bounds += [(0.0, r_d[b])] * H + [(None, Emax[b])] * H + [(0.0, None)] * H
        res = linprog(c, A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if res.status == 0:
            q0 = res.x.reshape(B, n)[:, 0]
            u_d = np.clip(q0 / np.maximum(r_d, 1e-9), 0.0, 1.0)
        else:
            self.lp_fail += 1
            u_d = np.where(E0 < Emin, 1.0, 0.0)
        p24 = env.price_path(24)
        cheap = p24[0] <= (1.0 - self.cheap_margin) * p24.mean()
        elapsed = 168 - env.hours_left_week()
        dis = (~env.week_done) & (elapsed >= self.request_after_h) & cheap
        u_d = np.where(dis, 1.0, u_d)
        return {"u_d": u_d, "u_s": sh_fraction(ctx, u_d), "dis": dis}

    def update(self, rec):
        pass
