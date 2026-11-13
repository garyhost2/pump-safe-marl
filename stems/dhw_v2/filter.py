from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from stems.dhw_v2 import tank as tk
from stems.dhw_v2.data import intra_hour_weights
from stems.dhw_v2.plant import simulate_hour

__all__ = ["FilterConfig", "DrawForecaster", "MarginTracker", "FilterModel", "JointFilter",
           "project_polygon", "solve_joint_qp", "week_calendar"]


class DrawForecaster:

    def __init__(self, B: int, period: int = 24):
        self.sum = np.zeros((period, B))
        self.n = np.zeros(period)
        self.tot = np.zeros(B)
        self.ntot = 0
        self.period = period

    def forecast(self, hour: int) -> np.ndarray:
        h = hour % self.period
        if self.n[h] > 0:
            return self.sum[h] / self.n[h]
        if self.ntot > 0:
            return self.tot / self.ntot
        return np.zeros_like(self.tot)

    def update(self, hour: int, draw_kwh: np.ndarray) -> None:
        h = hour % self.period
        self.sum[h] += draw_kwh
        self.n[h] += 1
        self.tot += draw_kwh
        self.ntot += 1


class MarginTracker:

    def __init__(self, B: int, alpha: float, eta: float, m0: float = 0.0):
        self.alpha, self.eta = float(alpha), float(eta)
        self.m = np.full(B, float(m0))
        self.n = 0
        self.exceed = np.zeros(B)
        self.e_lo = np.full(B, min(0.0, float(m0)))
        self.e_hi = np.full(B, max(0.0, float(m0)))

    def update(self, e: np.ndarray) -> None:
        e = np.asarray(e, dtype=np.float64)
        hit = e > self.m
        self.exceed += hit
        self.n += 1
        self.e_lo = np.minimum(self.e_lo, e)
        self.e_hi = np.maximum(self.e_hi, e)
        self.m = self.m + self.eta * (hit.astype(float) - self.alpha)

    def upper_bound(self) -> np.ndarray:
        return self.alpha + (self.e_hi - self.e_lo + self.eta) / (self.eta * max(self.n, 1))

    def coverage_bound(self) -> np.ndarray:
        return (self.e_hi - self.e_lo + self.eta) / (self.eta * max(self.n, 1))


def week_calendar(hour_of_day: np.ndarray, day_of_week: np.ndarray):
    start = (np.asarray(day_of_week) == 0) & (np.asarray(hour_of_day) == 0)
    wid = np.cumsum(start) - (1 if start[0] else 0)
    wid = wid - wid[0]
    T = len(wid)
    left = np.zeros(T, dtype=int)
    nxt = T
    for t in range(T - 1, -1, -1):
        if t + 1 < T and start[t + 1]:
            nxt = t + 1
        left[t] = nxt - t
    return wid, left


def project_polygon(p: np.ndarray, A: np.ndarray, b: np.ndarray, tol: float = 1e-10):
    p = np.asarray(p, dtype=np.float64)
    if np.all(A @ p <= b + tol):
        return p.copy()
    best, bd = None, np.inf
    m = len(b)
    for i in range(m):
        a = A[i]
        nn = a @ a
        if nn <= 0:
            continue
        x = p - (a @ p - b[i]) / nn * a
        if np.all(A @ x <= b + tol):
            d = np.sum((x - p) ** 2)
            if d < bd:
                best, bd = x, d
    for i in range(m):
        for j in range(i + 1, m):
            M = np.array([A[i], A[j]])
            det = np.linalg.det(M)
            if abs(det) < 1e-14:
                continue
            x = np.linalg.solve(M, np.array([b[i], b[j]]))
            if np.all(A @ x <= b + tol):
                d = np.sum((x - p) ** 2)
                if d < bd:
                    best, bd = x, d
    if best is None:
        raise ValueError("empty polygon")
    return best


def _house_polygon(lo_d, lo_s, a_d, a_s, cap_rhs):
    A = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 1.0]]
    b = [1.0, 1.0, -lo_d, -lo_s, 1.0]
    if np.isfinite(cap_rhs):
        A.append([a_d, a_s])
        b.append(cap_rhs)
    return np.array(A), np.array(b)


def solve_joint_qp(prop: np.ndarray, lo: np.ndarray, coef: np.ndarray, const: np.ndarray,
                   cap_house: np.ndarray, cap_district: Optional[float]) -> np.ndarray:
    B = prop.shape[0]
    rhs = cap_house - const

    def solve(lam):
        out = np.zeros((B, 2))
        for i in range(B):
            A, b = _house_polygon(lo[i, 0], lo[i, 1], coef[i, 0], coef[i, 1], rhs[i])
            out[i] = project_polygon(prop[i] - 0.5 * lam * coef[i], A, b)
        return out

    x = solve(0.0)
    if cap_district is None or not np.isfinite(cap_district):
        return x
    g = lambda xx: float(np.sum(coef * xx) + np.sum(const))
    if g(x) <= cap_district + 1e-12:
        return x
    lo_l, hi_l = 0.0, 1.0
    while g(solve(hi_l)) > cap_district and hi_l < 1e8:
        hi_l *= 4.0
    for _ in range(60):
        mid = 0.5 * (lo_l + hi_l)
        if g(solve(mid)) > cap_district:
            lo_l = mid
        else:
            hi_l = mid
    return solve(hi_l)


def _min_slack_lp(lo_c, lo_dis, lo_s, coef, const, cap_house, cap_district, w):
    from scipy.optimize import linprog
    B = len(lo_c)
    nv = 5 * B
    cvec = np.zeros(nv)
    A, b = [], []
    for i in range(B):
        o = 5 * i
        cvec[o + 2], cvec[o + 3], cvec[o + 4] = w
        for (j, lb, k) in ((0, lo_c[i], 2), (0, lo_dis[i], 3), (1, lo_s[i], 4)):
            row = np.zeros(nv)
            row[o + j] = -1.0
            row[o + k] = -1.0
            A.append(row)
            b.append(-lb)
        row = np.zeros(nv)
        row[o], row[o + 1] = 1.0, 1.0
        A.append(row)
        b.append(1.0)
        if np.isfinite(cap_house[i]):
            row = np.zeros(nv)
            row[o], row[o + 1] = coef[i]
            A.append(row)
            b.append(cap_house[i] - const[i])
    if cap_district is not None and np.isfinite(cap_district):
        row = np.zeros(nv)
        for i in range(B):
            row[5 * i], row[5 * i + 1] = coef[i]
        A.append(row)
        b.append(cap_district - float(np.sum(const)))
    bounds = []
    for i in range(B):
        bounds += [(0.0, 1.0), (0.0, 1.0), (0.0, None), (0.0, None), (0.0, None)]
    r = linprog(cvec, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(f"min-slack LP failed: {r.message} (hard set empty?)")
    x = r.x.reshape(B, 5)
    return x[:, 2], x[:, 3], x[:, 4]


class FilterModel:

    def __init__(self, tank_arrays: Dict, hp, *, kind: str = "oracle", meta: Optional[Dict] = None):
        self.tank = dict(tank_arrays)
        self.hp = hp
        self.kind = kind
        self.meta = meta or {}

    @property
    def N(self) -> int:
        return int(self.tank["nodes"])

    def state(self, T_plant: np.ndarray) -> np.ndarray:
        return tk.aggregate_nodes(T_plant, self.N)

    def rates(self, ctx: Dict, dis: np.ndarray, t_s_n: float, t_s_x: float):
        if self.kind == "oracle":
            r = np.where(dis, ctx["r_dis"], ctx["r_d"])
            p = np.where(dis, ctx["p_dis"], ctx["p_d"])
            return r, p
        t_src = np.where(dis, ctx["t_src_dis"], ctx["t_src_d"])
        t_s = np.where(dis, t_s_x, t_s_n)
        f = np.where(dis, float(ctx.get("dis_speed_frac", 1.0)), float(ctx.get("dhw_speed_frac", 1.0)))
        r = f * self.hp.q_max(t_src, t_s)
        return r, self.hp.p_el(r, t_src, t_s)

    def sh_coefficients(self, ctx: Dict):
        if self.kind == "oracle":
            return ctx["q_max_s"], ctx["chord_kappa"] * ctx["q_max_s"], ctx["chord_p0"]
        ts = np.where(ctx["sh_on"], ctx["t_sup_s"], 35.0)
        q = np.where(ctx["sh_on"], self.hp.q_max(ctx["t_src_s"], ts), 0.0)
        p0, kap = self.hp.chord(ctx["t_src_s"], ts)
        return q, np.where(ctx["sh_on"], kap * q, 0.0), np.where(ctx["sh_on"], p0, 0.0)

    hold_rule = "bottom20"
    hold_steps = None

    def sim(self, T, draws, u_d, t_s, r_d, p_d, t_cold, dt_h, t_dis_c=60.0):
        extra = T.ndim - 2
        tank = {k: (v.reshape(v.shape + (1,) * extra) if isinstance(v, np.ndarray) else v)
                for k, v in self.tank.items()}
        return simulate_hour(T, tank, draws, dt_h, u_d=u_d, r_d=r_d, p_d=p_d, t_s=t_s,
                             t_cold=t_cold, t_dis_c=t_dis_c, hold_rule=self.hold_rule,
                             hold_steps=self.hold_steps)

    def hold_metric(self, T):
        from stems.dhw_v2.plant import bottom_weights
        if self.hold_rule == "bottom20":
            return T @ bottom_weights(T.shape[-1], 0.2)
        if self.hold_rule == "whole_mean":
            return T.mean(axis=-1)
        return T.min(axis=-1)


class HourOfDayQuantile:

    def __init__(self, B: int, beta: float, period: int = 24, method: str = "conformal"):
        self.beta = float(beta)
        self.period = period
        self.method = method
        self.samples = [[] for _ in range(period)]
        self.pool: List[np.ndarray] = []
        self.B = B

    @property
    def alpha(self):
        return self.beta

    def quantile(self, hour: int) -> np.ndarray:
        h = hour % self.period
        xs = self.samples[h]
        if self.method == "conformal":
            n = len(xs)
            k = int(np.ceil((n + 1) * (1.0 - self.beta)))
            if n == 0 or k > n:
                return np.full(self.B, np.inf)
            return np.sort(np.stack(xs), axis=0)[k - 1]
        xs = xs if len(xs) >= 5 else self.pool
        if not xs:
            return np.zeros(self.B)
        return np.quantile(np.stack(xs), 1.0 - self.beta, axis=0)

    def update(self, hour: int, draw_kwh: np.ndarray) -> None:
        x = np.asarray(draw_kwh, dtype=np.float64).copy()
        self.samples[hour % self.period].append(x)
        self.pool.append(x)


@dataclass
class FilterConfig:
    floor_c: float = 45.0
    t_dis_c: float = 60.0
    t_supply_dhw_c: float = 55.0
    t_supply_dis_c: float = 65.0
    alpha: float = 0.05
    eta_kwh: float = 0.05
    m0_kwh: float = 0.0
    forecast: str = "hod_mean"
    use_margin: bool = True
    cert_shape: str = "uniform"
    cap_house_kw: Optional[float] = None
    cap_district_kw: Optional[float] = None
    w_comfort: float = 4.0
    w_dis: float = 2.0
    w_sh: float = 1.0
    sh_comfort: bool = True
    grid: int = 17
    reserve_h: int = 24
    brine_band_k: float = 3.0
    lookahead: bool = False
    disinfection: bool = True
    trigger: str = "robust"
    beta: float = 0.02
    retry_h: int = 2
    trigger_horizon_h: int = 72
    jit_reserve_h: int = 2
    hold_rule: str = "bottom20"
    hold_min: float = 60.0


class JointFilter:
    def __init__(self, model: FilterModel, cfg: FilterConfig, B: int, S: int, dt_h: float,
                 week_id: np.ndarray, hours_left: np.ndarray):
        self.model, self.cfg = model, cfg
        self.model.hold_rule = cfg.hold_rule
        self.model.hold_steps = max(1, min(S, int(np.ceil(cfg.hold_min / (dt_h * 60.0) - 1e-9))))
        self.B, self.S, self.dt_h = B, S, dt_h
        self.week_id, self.hours_left = week_id, hours_left
        self.forecaster = DrawForecaster(B)
        self.margin = MarginTracker(B, cfg.alpha, cfg.eta_kwh, cfg.m0_kwh)
        self.q_dis = HourOfDayQuantile(B, cfg.beta, method="conformal") if cfg.trigger == "robust" \
            else HourOfDayQuantile(B, 0.10, method="interp")
        self.shape = intra_hour_weights(S, cfg.cert_shape)
        self.done = np.zeros(B, dtype=bool)
        self.cur_week = 0
        self.monotonicity_failures = 0
        self.trigger_fired_infeasible = 0
        self._last = None

    def _sim(self, X, D, u, t_s, r, p, t_cold):
        extra = X.ndim - 2
        col = lambda a: np.asarray(a, dtype=np.float64).reshape((self.B,) + (1,) * extra)
        draws = self.shape.reshape((-1,) + (1,) * (extra + 1)) * col(D)[None]
        return self.model.sim(X, draws, col(u) if np.ndim(u) == 1 else u, col(t_s), col(r), col(p),
                              col(t_cold), self.dt_h, self.cfg.t_dis_c)

    def _prepare_reserve(self, ctx, hour, m, t_cold, u_max):
        cfg, B, H = self.cfg, self.B, self.cfg.reserve_h
        t_s = np.full(B, cfg.t_supply_dhw_c)
        src = np.asarray(ctx["t_src_d"], float) - cfg.brine_band_k
        f = float(ctx.get("dhw_speed_frac", 1.0))
        r = f * self.model.hp.q_max(src, t_s)
        p = self.model.hp.p_el(r, src, t_s)
        Dq = np.stack([np.maximum(self.forecaster.forecast(hour + 1 + k) + (m if cfg.use_margin else 0.0), 0.0)
                       for k in range(H)])
        R = {"Dq": Dq, "r": r, "p": p, "t_s": t_s, "u": u_max, "t_cold": t_cold}
        full = np.full((B, H, self.model.N), cfg.t_supply_dhw_c)
        F = np.ones((B, H + 1), dtype=bool)
        R["F"] = F
        tank = {k: (v.reshape(v.shape + (1,)) if isinstance(v, np.ndarray) else v) for k, v in self.model.tank.items()}
        col = lambda a: np.asarray(a, dtype=np.float64)[:, None]
        draws = self.shape[:, None, None] * Dq.T[None, :, :]
        res = simulate_hour(full, tank, draws, self.dt_h, u_d=col(u_max), r_d=col(r), p_d=col(p),
                            t_s=col(t_s), t_cold=col(t_cold), t_dis_c=cfg.t_dis_c)
        ok = res["top_min"] >= cfg.floor_c - 1e-9
        end_full = np.all(res["T_end"] >= cfg.t_supply_dhw_c - 1e-7, axis=-1)
        for k in range(H - 1, -1, -1):
            tail = F[:, k + 1].copy()
            if not end_full[:, k].all():
                tail = np.where(end_full[:, k], tail, self._tail(res["T_end"][:, k, :], k + 1, R))
            F[:, k] = ok[:, k] & tail
        return R

    def _tail(self, X, k0, R):
        cfg = self.cfg
        H = cfg.reserve_h
        shp = X.shape[:-1]
        extra = X.ndim - 2
        colF = lambda j: R["F"][:, j].reshape((self.B,) + (1,) * extra)
        ok = np.ones(shp, dtype=bool)
        done = np.all(X >= cfg.t_supply_dhw_c - 1e-7, axis=-1)
        ok = np.where(done, np.broadcast_to(colF(min(k0, H)), shp), ok)
        for j in range(k0, H):
            if done.all():
                break
            res = self._sim(X, R["Dq"][j], R["u"], R["t_s"], R["r"], R["p"], R["t_cold"])
            ok = np.where(done, ok, ok & (res["top_min"] >= cfg.floor_c - 1e-9))
            X = res["T_end"]
            full = np.all(X >= cfg.t_supply_dhw_c - 1e-7, axis=-1)
            newly = full & ~done
            ok = np.where(newly, ok & np.broadcast_to(colF(j + 1), shp), ok)
            done = done | full | ~ok
        return ok

    def _lower_bound(self, T, D, t_s, r_d, p_d, t_cold, cond: str, tail=None):
        cfg = self.cfg
        K = cfg.grid
        from stems.dhw_v2.plant import HOLD_KEYS
        thr = cfg.floor_c if cond == "top" else cfg.t_dis_c
        key = "top_min" if cond == "top" else HOLD_KEYS[cfg.hold_rule]

        def ok(u):
            r = self._sim(np.broadcast_to(T[:, None, :], (self.B, u.shape[1], T.shape[-1])), D, u,
                          t_s, r_d, p_d, t_cold)
            now = r[key] >= thr - 1e-9 if cond == "top" else r["hold"]
            if tail is None:
                return now, now
            both = now.copy()
            if now.any():
                both = now & tail(r["T_end"])
            return now, both

        def first(good):
            infeasible = ~good[:, -1]
            j = np.where(good.any(axis=1), np.argmax(good, axis=1), K - 1)
            return j, infeasible

        g = np.linspace(0.0, 1.0, K)
        now0, both0 = ok(np.zeros((self.B, 1)))
        if both0.all():
            return np.zeros(self.B), np.zeros(self.B, bool), np.zeros(self.B, bool)
        u = np.broadcast_to(g, (self.B, K))
        now, both = ok(u)
        j_n, inf_n = first(now)
        j_b, inf_b = first(both)
        use_b = ~inf_b
        j = np.where(use_b, j_b, j_n)
        infeasible = inf_n
        res_inf = (inf_b & ~inf_n) if tail is not None else np.zeros(self.B, bool)
        lo = np.where(j > 0, g[np.maximum(j - 1, 0)], 0.0)
        hi = g[j]
        need = (j > 0) & ~infeasible
        l = np.where(infeasible, 1.0, hi)
        if np.any(need):
            u2 = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, K)[None, :]
            now2, both2 = ok(u2)
            sel = np.where(use_b[:, None], both2, now2)
            sel[:, -1] = True
            j2 = np.argmax(sel, axis=1)
            l = np.where(need, u2[np.arange(self.B), j2], l)
        l = np.where(j == 0, 0.0, l)
        return l, infeasible, res_inf

    def _draw_path(self, hour, n, m):
        tr = self.cfg.trigger
        if tr in ("robust", "quantile"):
            return [self.q_dis.quantile(hour + k) for k in range(n)]
        if tr == "recourse":
            return [self.forecaster.forecast(hour + k) for k in range(n)]
        return [np.maximum(self.forecaster.forecast(hour + k) + m, 0.0) for k in range(n)]

    def _cycle_hours(self, X, path, r, p, u_max, t_cold):
        from stems.dhw_v2.plant import HOLD_KEYS
        cfg = self.cfg
        key = HOLD_KEYS[cfg.hold_rule]
        Hc = np.full(self.B, np.inf)
        t_s = np.full(self.B, cfg.t_supply_dis_c)
        found = np.zeros(self.B, bool)
        dead = np.zeros(self.B, bool)
        for k in range(min(cfg.trigger_horizon_h, len(path))):
            b = np.asarray(path[k], dtype=np.float64)
            dead |= ~np.isfinite(b) & ~found
            res = self._sim(X, np.where(np.isfinite(b), b, 0.0), u_max, t_s, r, p, t_cold)
            hit = res["hold"] & ~found & ~dead
            Hc = np.where(hit, k + 1, Hc)
            found |= hit
            X = res["T_end"]
            if (found | dead).all():
                break
        return Hc

    def _trigger(self, t, ctx, T, hour, m, t_cold, u_max_dis):
        cfg, B = self.cfg, self.B
        owed = ~self.done
        left = int(self.hours_left[t])
        must = np.zeros(B, bool)
        Hc = np.zeros(B)
        tinf = np.zeros(B, bool)
        if cfg.trigger == "none" or not owed.any():
            return must, Hc, tinf
        if cfg.trigger != "robust" and left > cfg.trigger_horizon_h + cfg.jit_reserve_h + 1:
            return must, Hc, tinf
        ones = np.ones(B, dtype=bool)
        r_x, p_x = self.model.rates(ctx, ones, cfg.t_supply_dhw_c, cfg.t_supply_dis_c)
        if cfg.trigger == "robust":
            src = np.asarray(ctx["t_src_dis"], float) - cfg.brine_band_k
            r_x = float(ctx.get("dis_speed_frac", 1.0)) * self.model.hp.q_max(src, cfg.t_supply_dis_c)
            p_x = self.model.hp.p_el(r_x, src, cfg.t_supply_dis_c)
        n = min(left + 1, cfg.trigger_horizon_h + 1)
        path = self._draw_path(hour, n + 1, m)
        if cfg.trigger == "robust":
            b0 = np.asarray(path[0], dtype=np.float64)
            x1 = self._sim(T, np.where(np.isfinite(b0), b0, 0.0), np.zeros(B),
                           np.full(B, cfg.t_supply_dhw_c), r_x, p_x, t_cold)["T_end"]
            Hc = self._cycle_hours(x1, path[1:], r_x, p_x, u_max_dis, t_cold)
            Hc = np.where(np.isfinite(b0), Hc, np.inf)
            must = owed & (left - 1 < Hc + cfg.retry_h)
            tinf = owed & ~np.isfinite(Hc)
        else:
            Hc = self._cycle_hours(T, path, r_x, p_x, u_max_dis, t_cold)
            must = owed & (left <= Hc + cfg.jit_reserve_h)
            tinf = owed & ~np.isfinite(Hc)
        return must, Hc, tinf

    def project(self, t: int, ctx: Dict, proposal: Dict[str, np.ndarray], draw_true: Optional[np.ndarray] = None):
        cfg, B = self.cfg, self.B
        if self.week_id[t] != self.cur_week:
            self.cur_week = int(self.week_id[t])
            self.done[:] = False
        T = self.model.state(ctx["T_nodes"])
        hour = int(ctx["hour"])
        d_hat = self.forecaster.forecast(hour) if cfg.forecast == "hod_mean" else np.asarray(draw_true, float)
        m = self.margin.m.copy() if (cfg.use_margin and cfg.forecast == "hod_mean") else np.zeros(B)
        D_cert = np.maximum(d_hat + m, 0.0)
        t_cold = np.asarray(ctx["t_cold"], float)
        p_d_prop = np.clip(np.asarray(proposal["u_d"], float), 0.0, 1.0)
        p_s_prop = np.clip(np.asarray(proposal["u_s"], float), 0.0, 1.0)
        dis_prop = np.asarray(proposal.get("dis", np.zeros(B, bool)), dtype=bool)
        q_s, a_s, p0_s = self.model.sh_coefficients(ctx)
        const = p0_s + float(ctx["p_standby"])
        cap_h = np.full(B, np.inf if cfg.cap_house_kw is None else float(cfg.cap_house_kw))

        def u_cap(p):
            return np.clip(np.where(np.isfinite(cap_h), (cap_h - const) / np.maximum(p, 1e-9), 1.0), 0.0, 1.0)

        owed = (~self.done) & cfg.disinfection
        ones = np.ones(B, dtype=bool)
        r_x0, p_x0 = self.model.rates(ctx, ones, cfg.t_supply_dhw_c, cfg.t_supply_dis_c)
        if cfg.disinfection:
            must, H_need, trig_inf = self._trigger(t, ctx, T, hour, m, t_cold, u_cap(p_x0))
            must &= owed
        else:
            must, H_need, trig_inf = np.zeros(B, bool), np.zeros(B, int), np.zeros(B, bool)
        dis = dis_prop | must
        t_s = np.where(dis, cfg.t_supply_dis_c, cfg.t_supply_dhw_c)
        r_d, p_d = self.model.rates(ctx, dis, cfg.t_supply_dhw_c, cfg.t_supply_dis_c)
        tail = None
        D_next = np.maximum(self.forecaster.forecast(hour + 1) + (m if cfg.use_margin else 0.0), 0.0)
        if cfg.reserve_h > 0:
            R = self._prepare_reserve(ctx, hour, m, t_cold, u_cap(p_d))
            tail = lambda X: self._tail(X, 0, R)
            D_next = R["Dq"][0]
        elif cfg.lookahead:
            nb = np.zeros(B, dtype=bool)
            r_n, p_n = self.model.rates(ctx, nb, cfg.t_supply_dhw_c, cfg.t_supply_dis_c)
            D_next = np.maximum(self.forecaster.forecast(hour + 1) + (m if cfg.use_margin else 0.0), 0.0)
            tail = lambda X: self._sim(X, D_next, np.ones(B), np.full(B, cfg.t_supply_dhw_c), r_n, p_n,
                                       t_cold)["top_min"] >= cfg.floor_c - 1e-9
        l_c, inf_c, res_inf = self._lower_bound(T, D_cert, t_s, r_d, p_d, t_cold, "top", tail)
        l_dis = np.zeros(B)
        inf_h = np.zeros(B, dtype=bool)
        hold_ready = self.model.hold_metric(T) >= cfg.t_dis_c - 1e-9
        if cfg.disinfection:
            hold_now = dis_prop & ~must & owed & hold_ready
            if hold_now.any():
                lh, ih, _ = self._lower_bound(T, D_cert, t_s, r_d, p_d, t_cold, "hold")
                l_dis = np.where(hold_now, lh, 0.0)
                inf_h = hold_now & ih
            l_dis = np.where(must, 1.0, l_dis)
        need_min = np.asarray(ctx["sh_need_min_kw"], float)
        l_s = np.where(q_s > 0, np.minimum(need_min / np.where(q_s > 0, q_s, 1.0), 1.0), 0.0) \
            if cfg.sh_comfort else np.zeros(B)
        inf_s = (q_s > 0) & (need_min > q_s) if cfg.sh_comfort else np.zeros(B, bool)
        coef = np.stack([p_d, a_s], axis=1)
        capD = cfg.cap_district_kw
        lo = np.stack([np.maximum(l_c, l_dis), l_s], axis=1)
        s_c = np.zeros(B); s_dis = np.zeros(B); s_sh = np.zeros(B)
        corner_ok = (lo.sum(axis=1) <= 1.0 + 1e-12) & (np.sum(coef * lo, axis=1) + const <= cap_h + 1e-12)
        if capD is not None:
            corner_ok = corner_ok & (np.sum(coef * lo) + np.sum(const) <= capD + 1e-12)
        if not corner_ok.all():
            s_c, s_dis, s_sh = _min_slack_lp(l_c, l_dis, l_s, coef, const, cap_h, capD,
                                             (cfg.w_comfort, cfg.w_dis, cfg.w_sh))
            lo = np.stack([np.maximum(np.maximum(l_c - s_c, l_dis - s_dis), 0.0),
                           np.maximum(l_s - s_sh, 0.0)], axis=1) - 1e-9
            lo = np.maximum(lo, 0.0)
        prop = np.stack([p_d_prop, p_s_prop], axis=1)
        x = np.clip(solve_joint_qp(prop, lo, coef, const, cap_h, capD), 0.0, 1.0)
        chk = (x[:, 0] > l_c + 1e-9) & (l_c > 0) & ~inf_c
        if chk.any():
            r = self._sim(T, D_cert, x[:, 0], t_s, r_d, p_d, t_cold)
            okx = r["top_min"] >= cfg.floor_c - 1e-9
            if tail is not None and not res_inf[chk].all():
                okx = okx & (tail(r["T_end"]) | res_inf)
            bad = chk & ~okx
            self.monotonicity_failures += int(bad.sum())
        power_limited = (u_cap(p_d) < 1.0 - 1e-9) | (s_c > 1e-9) | ((capD is not None) & (x[:, 0] < l_c - 1e-6))
        dx = np.abs(x - prop)
        info = {
            "t": t, "u_d_prop": p_d_prop, "u_s_prop": p_s_prop, "dis_prop": dis_prop,
            "u_d": x[:, 0], "u_s": x[:, 1], "dis": dis, "intervened": (dx.max(axis=1) > 1e-9) | (dis != dis_prop),
            "l_c": l_c, "l_dis": l_dis, "l_s": l_s, "s_c": s_c, "s_dis": s_dis, "s_sh": s_sh,
            "infeasible_c": inf_c, "reserve_infeasible": res_inf, "infeasible_hold": inf_h,
            "infeasible_s": inf_s, "power_limited": power_limited,
            "lookahead_infeasible": res_inf,
            "d_hat": d_hat, "m": m, "D_cert": D_cert, "D_next_cert": D_next, "must": must, "H_need": H_need,
            "trigger_infeasible": trig_inf, "hours_left": int(self.hours_left[t]), "t_s": t_s,
            "elec_bound_kw": np.sum(coef * x, axis=1) + const,
            "delta_el_kwh": np.abs(x[:, 0] - p_d_prop) * p_d + np.abs(x[:, 1] - p_s_prop) * a_s,
        }
        self._last = info
        return {"u_d": x[:, 0], "u_s": x[:, 1], "dis": dis}, info

    def observe(self, t: int, hour: int, D: np.ndarray, rec: Dict[str, np.ndarray]) -> Dict:
        info = self._last or {}
        d_hat = info.get("d_hat", self.forecaster.forecast(hour))
        m = info.get("m", self.margin.m.copy())
        delta = D - d_hat
        self.done |= np.asarray(rec["hold"], dtype=bool)
        if self.cfg.forecast == "hod_mean":
            self.margin.update(delta)
        self.forecaster.update(hour, D)
        self.q_dis.update(hour, D)
        return {"delta": delta, "m": m, "bound": np.maximum(delta - m, 0.0)}

    def describe(self) -> Dict:
        return {"cfg": asdict(self.cfg), "model": self.model.kind, "model_meta": self.model.meta,
                "model_nodes": self.model.N}
