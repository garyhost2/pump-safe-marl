from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from stems.dhw_v2.building import DemandBuilding, RC5R1C, heating_curve_c
from stems.dhw_v2.controllers import PriceAwareController, sh_fraction

CV_KWH_PER_L_K = 4.186 * 0.983 / 3600.0


def affine_5r1c(bl: RC5R1C):
    save = bl.theta_m

    def S(th, phi, te, g):
        bl.theta_m = th
        a, m = bl._solve(phi, te, g)
        return np.array([a, m], float)

    base = S(0.0, 0.0, 0.0, 0.0)
    out = {"base": base, "th": S(1.0, 0.0, 0.0, 0.0) - base, "phi": (S(0.0, 1000.0, 0.0, 0.0) - base) / 1000.0,
           "te": S(0.0, 0.0, 1.0, 0.0) - base, "ghi": (S(0.0, 0.0, 0.0, 100.0) - base) / 100.0}
    bl.theta_m = save
    return out


class MPCBandController:
    name = "mpc_band"

    def __init__(self, B: int, *, horizon: int = 24, rho: float = 0.9, forecast: str = "causal",
                 band: Tuple[float, float] = (20.0, 23.0), w_band: float = 5.0,
                 request_after_h: float = 120.0, cheap_margin: float = 0.05):
        self.B, self.H, self.rho, self.forecast = B, int(horizon), float(rho), forecast
        self.lo, self.hi = float(band[0]), float(band[1])
        self.w_band = float(w_band)
        self.request_after_h, self.cheap_margin = request_after_h, cheap_margin
        self.lp_fail = 0
        self._aff = None
        self.plan_t_in = None

    def reset(self):
        self.lp_fail = 0
        self._aff = None

    def _setup(self, env):
        self._aff = []
        for bl in env.plant.buildings:
            if isinstance(bl, RC5R1C):
                self._aff.append(("5r1c", affine_5r1c(bl)))
            elif isinstance(bl, DemandBuilding):
                self._aff.append(("demand", None))
            else:
                raise NotImplementedError(type(bl))

    def act(self, env) -> Dict:
        if self._aff is None:
            self._setup(env)
        pl = env.plant
        ctx = pl.context()
        B, H, t = self.B, self.H, env.t
        T = pl.T
        data = env.data
        fx = 1.0 / 10.28 if data.currency.upper() == "USD" else 1.0
        wb = self.w_band * fx
        idx = np.minimum(np.arange(t, t + H), data.T - 1)
        price = data.price[idx]
        t_e = data.t_out_c[idx]
        ghi = np.maximum(data.ghi_w_m2[idx], 0.0)
        vol = np.asarray(pl.tank["node_vol"], float).reshape(-1, 1) * np.ones_like(T)
        tc = np.asarray(ctx["t_cold"], float)
        Cn = vol * CV_KWH_PER_L_K
        Ctot = Cn.sum(axis=1)
        E0 = np.sum(Cn * (T - tc[:, None]), axis=1)
        loss = np.maximum(np.asarray(pl.tank["ua"], float) * (T.mean(axis=1) - np.asarray(pl.tank["t_room"], float)), 0.0) / 1000.0
        r_d = np.asarray(ctx["r_d"], float)
        cop_d = r_d / np.maximum(np.asarray(ctx["p_d"], float), 1e-6)
        hour = int(data.hour_of_day[t])
        if self.forecast == "perfect":
            d = np.stack([pl.draws[i].sum(axis=0) for i in idx])
        else:
            fc = env.filter.forecaster if env.filter is not None else env.shadow_fc
            m = env.filter.margin.m if env.filter is not None else env.shadow_m.m
            d = np.stack([fc.forecast(hour + k) for k in range(H)]) + m[None, :]
        Emin = self.rho * Ctot * (45.0 - tc)
        Emax = np.maximum(Ctot * (55.0 - tc), E0)
        t_src = np.asarray(ctx["t_src_s"], float)
        hp = pl.hp
        nv = 7 * H
        rows, cols, vals, beq_rows = [], [], [], []
        A_eq_r, A_eq_c, A_eq_v, b_eq = [], [], [], []
        A_ub_r, A_ub_c, A_ub_v, b_ub = [], [], [], []
        cvec = np.zeros(B * nv)
        bounds = []
        neq = nub = 0
        qmax_now = np.asarray(ctx["q_max_s"], float)
        for b in range(B):
            o = b * nv
            iq, iE, isE, iqs, ith, ilo, ihi = (o + j * H for j in range(7))
            kind, aff = self._aff[b]
            bl = pl.buildings[b]
            ts = heating_curve_c(t_e, bl.curve)
            on = np.isfinite(ts)
            tsv = np.where(on, ts, 35.0)
            qmax = np.where(on, np.asarray(hp.q_max(np.full(H, t_src[b]), tsv), float), 0.0)
            qmax[0] = qmax_now[b]
            kap = np.where(on, np.asarray(hp.chord(np.full(H, t_src[b]), tsv)[1], float), 0.0)
            pdc = price / cop_d[b]
            psc = price * kap
            cvec[iq:iq + H] = pdc
            cvec[isE:isE + H] = 100.0 * float(np.max(np.abs(price)) + 1e-3)
            cvec[iE + H - 1] -= float(np.mean(pdc))
            cvec[iqs:iqs + H] = psc
            cvec[ilo:ilo + H] = wb
            cvec[ihi:ihi + H] = wb
            mval = float(np.mean(psc[on])) if on.any() else float(np.mean(price) / 3.0)
            for k in range(H):
                A_eq_r += [neq, neq]; A_eq_c += [iE + k, iq + k]; A_eq_v += [1.0, -1.0]
                if k > 0:
                    A_eq_r.append(neq); A_eq_c.append(iE + k - 1); A_eq_v.append(-1.0)
                b_eq.append(-(d[k, b] + loss[b]) + (E0[b] if k == 0 else 0.0)); neq += 1
                A_ub_r += [nub, nub]; A_ub_c += [iE + k, isE + k]; A_ub_v += [-1.0, -1.0]; b_ub.append(-Emin[b]); nub += 1
                if qmax[k] > 1e-9:
                    A_ub_r += [nub, nub]; A_ub_c += [iq + k, iqs + k]; A_ub_v += [1.0 / max(r_d[b], 1e-9), 1.0 / qmax[k]]
                    b_ub.append(1.0); nub += 1
            if kind == "5r1c":
                a_th, b_th = aff["th"][1], aff["phi"][1] * 1000.0
                d_th, e_th = aff["th"][0], aff["phi"][0] * 1000.0
                c_k = aff["base"][1] + aff["te"][1] * t_e + aff["ghi"][1] * ghi
                f_k = aff["base"][0] + aff["te"][0] * t_e + aff["ghi"][0] * ghi
                th0 = float(bl.theta_m)
                for k in range(H):
                    A_eq_r += [neq, neq]; A_eq_c += [ith + k, iqs + k]; A_eq_v += [1.0, -b_th]
                    rhs = c_k[k]
                    if k > 0:
                        A_eq_r.append(neq); A_eq_c.append(ith + k - 1); A_eq_v.append(-a_th)
                    else:
                        rhs += a_th * th0
                    b_eq.append(rhs); neq += 1
                    base_c = f_k[k] + (d_th * th0 if k == 0 else 0.0)
                    r_ = [iqs + k] + ([ith + k - 1] if k > 0 else [])
                    v_ = [e_th] + ([d_th] if k > 0 else [])
                    A_ub_r += [nub] * (len(r_) + 1); A_ub_c += r_ + [ilo + k]; A_ub_v += [-x for x in v_] + [-1.0]
                    b_ub.append(-self.lo + base_c); nub += 1
                    A_ub_r += [nub] * (len(r_) + 1); A_ub_c += r_ + [ihi + k]; A_ub_v += v_ + [-1.0]
                    b_ub.append(self.hi - base_c); nub += 1
                Cm_kwh = bl.C_m / 3.6e6
                cvec[ith + H - 1] -= mval * Cm_kwh
                th_bounds = [(None, None)] * H
            else:
                dem = np.asarray(bl.demand, float)[np.minimum(bl.k + np.arange(H), len(bl.demand) - 1)]
                D0 = float(bl.debt)
                Ce = bl.C_EFF_KWH_K
                for k in range(H):
                    A_ub_r += [nub, nub]; A_ub_c += [ith + k, iqs + k]; A_ub_v += [-1.0, -1.0]
                    rhs = -dem[k] - (D0 if k == 0 else 0.0)
                    if k > 0:
                        A_ub_r.append(nub); A_ub_c.append(ith + k - 1); A_ub_v.append(1.0)
                    b_ub.append(rhs); nub += 1
                    A_ub_r += [nub, nub]; A_ub_c += [ith + k, ilo + k]; A_ub_v += [1.0 / Ce, -1.0]
                    b_ub.append(bl.t_set - self.lo); nub += 1
                cvec[ith + H - 1] += mval
                th_bounds = [(0.0, None)] * H
            bounds += ([(0.0, r_d[b])] * H + [(None, Emax[b])] * H + [(0.0, None)] * H
                       + [(0.0, float(q)) for q in qmax] + th_bounds + [(0.0, None)] * H
                       + ([(0.0, None)] * H if kind == "5r1c" else [(0.0, 0.0)] * H))
        Aeq = coo_matrix((A_eq_v, (A_eq_r, A_eq_c)), shape=(neq, B * nv)).tocsr()
        Aub = coo_matrix((A_ub_v, (A_ub_r, A_ub_c)), shape=(nub, B * nv)).tocsr()
        res = linprog(cvec, A_ub=Aub, b_ub=np.array(b_ub), A_eq=Aeq, b_eq=np.array(b_eq), bounds=bounds, method="highs")
        self.plan_t_in = np.full(B, np.nan)
        if res.status == 0:
            x = res.x.reshape(B, nv)
            q0, qs0 = x[:, 0], x[:, 3 * H]
            u_d = np.clip(q0 / np.maximum(r_d, 1e-9), 0.0, 1.0)
            u_s = np.where(qmax_now > 1e-9, np.clip(qs0 / np.maximum(qmax_now, 1e-9), 0.0, 1.0), 0.0)
            for b in range(B):
                kind, aff = self._aff[b]
                bl = pl.buildings[b]
                if kind == "5r1c":
                    self.plan_t_in[b] = (aff["base"][0] + aff["te"][0] * t_e[0] + aff["ghi"][0] * ghi[0]
                                         + aff["th"][0] * bl.theta_m + aff["phi"][0] * 1000.0 * qs0[b])
                else:
                    self.plan_t_in[b] = bl.t_set - x[b, 4 * H] / bl.C_EFF_KWH_K
        else:
            self.lp_fail += 1
            u_d = np.where(E0 < Emin, 1.0, 0.0)
            u_s = sh_fraction(ctx, u_d)
        p24 = env.price_path(24)
        cheap = p24[0] <= (1.0 - self.cheap_margin) * p24.mean()
        P = getattr(env, "period_h", 168)
        elapsed = P - env.hours_left_week()
        dis = (~env.week_done) & (elapsed >= self.request_after_h * P / 168.0) & cheap
        u_d = np.where(dis, 1.0, u_d)
        u_s = np.minimum(u_s, 1.0 - u_d)
        return {"u_d": u_d, "u_s": u_s, "dis": dis}

    def update(self, rec):
        pass


def need_frac(env, target):
    pl = env.plant
    ctx = pl.context()
    t = env.t
    t_e, ghi = float(env.data.t_out_c[t]), float(env.data.ghi_w_m2[t])
    q = np.asarray(ctx["q_max_s"], float)
    need = np.array([bl.need_w(float(tg), t_e, ghi) / 1000.0 if bl is not None else 0.0
                     for bl, tg in zip(pl.buildings, np.broadcast_to(target, (env.B,)))])
    return np.where(q > 0, np.clip(need / np.where(q > 0, q, 1.0), 0.0, 1.0), 0.0)


class PreheatRule:
    name = "preheat"

    def __init__(self, B: int, *, k_cheap: int = 6, k_dear: int = 6, t_hi: float = 22.0, t_lo: float = 20.0,
                 t_mid: Optional[float] = None):
        self.B, self.k_cheap, self.k_dear, self.t_hi, self.t_lo, self.t_mid = B, k_cheap, k_dear, t_hi, t_lo, t_mid
        self.pa = PriceAwareController(B)

    def reset(self):
        self.pa.reset()

    def setpoint(self, env):
        p = env.price_path(24)
        r = int(np.sum(p < p[0]))
        mid = self.t_mid if self.t_mid is not None else 21.0
        if r < self.k_cheap:
            return self.t_hi
        if r >= 24 - self.k_dear:
            return self.t_lo
        return mid

    def act(self, env) -> Dict:
        cmd = self.pa.act(env)
        sp = self.setpoint(env)
        u_s = np.minimum(need_frac(env, sp), 1.0 - np.asarray(cmd["u_d"], float))
        return {"u_d": cmd["u_d"], "u_s": u_s, "dis": cmd["dis"]}

    def update(self, rec):
        self.pa.update(rec)
