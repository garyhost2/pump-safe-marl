from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from stems.dhw_v2 import tank as tk

__all__ = ["IdentifiedMap", "identify", "LogRecorder"]


class IdentifiedMap:
    name = "identified_linear"

    def __init__(self, g, h, *, q_min_frac=0.27, t_supply_max=70.0, p_standby_kw=0.017,
                 env_src=(0.0, 10.0), env_sup=(24.0, 55.0)):
        self.g = np.asarray(g, float)
        self.h = np.asarray(h, float)
        self.q_min_frac = q_min_frac
        self.t_supply_max = t_supply_max
        self.p_standby_kw = p_standby_kw
        self.env_src, self.env_sup = env_src, env_sup
        self.p_el_max = np.inf
        self.k = {}

    def q_max(self, t_src, t_sup):
        t_src, t_sup = np.broadcast_arrays(np.asarray(t_src, float), np.asarray(t_sup, float))
        q = self.g[0] + self.g[1] * t_src + self.g[2] * t_sup
        return np.where(t_sup > self.t_supply_max, 0.0, np.maximum(q, 0.0))

    def q_min(self, t_src, t_sup=55.0):
        return self.q_min_frac * self.q_max(t_src, t_sup)

    def _eir(self, t_src, t_sup):
        return np.maximum(self.h[0] + self.h[1] * (np.asarray(t_sup, float) - np.asarray(t_src, float)), 0.05)

    def p_el(self, q, t_src, t_sup):
        q = np.asarray(q, float)
        return np.where(q > 0, q * self._eir(t_src, t_sup), 0.0)

    def cop(self, q, t_src, t_sup):
        return 1.0 / self._eir(t_src, t_sup) * np.ones_like(np.asarray(q, float))

    def chord(self, t_src, t_sup):
        e = self._eir(t_src, t_sup)
        return np.zeros_like(e), e * (1.0 + 1e-9)

    def flags(self, t_src, t_sup):
        t_src, t_sup = np.broadcast_arrays(np.asarray(t_src, float), np.asarray(t_sup, float))
        return {"extrap_supply": (t_sup > self.env_sup[1]) | (t_sup < self.env_sup[0]),
                "extrap_source": (t_src < self.env_src[0]) | (t_src > self.env_src[1])}

    def describe(self):
        return {"name": self.name, "g": self.g.tolist(), "h": self.h.tolist()}


class LogRecorder:

    def __init__(self, n_nodes: int, *, sigma_t: float = 0.2, sigma_p: float = 0.01, seed: int = 0):
        self.n = n_nodes
        self.sigma_t, self.sigma_p = sigma_t, sigma_p
        self.rng = np.random.default_rng(seed)
        self.rows: List[Dict[str, np.ndarray]] = []

    def record(self, T_start, T_end, ctx, rec, t_s_used, volume_l):
        nz = lambda x: x + self.rng.normal(0.0, self.sigma_t, np.shape(x))
        self.rows.append({
            "T0": nz(tk.aggregate_nodes(T_start, self.n)), "T1": nz(tk.aggregate_nodes(T_end, self.n)),
            "served": np.asarray(rec["served_kwh"], float),
            "elec_d": np.asarray(rec["elec_dhw_kwh"], float) * (1 + self.rng.normal(0, self.sigma_p, np.shape(rec["elec_dhw_kwh"]))),
            "time_d": np.asarray(rec["dhw_time_h"], float), "t_src": np.asarray(ctx["t_src_d"], float),
            "t_src_dis": np.asarray(ctx["t_src_dis"], float),
            "t_s": np.asarray(t_s_used, float), "vol": np.asarray(volume_l, float),
            "frac": np.where(np.asarray(t_s_used, float) > 60.0, float(ctx.get("dis_speed_frac", 1.0)),
                             float(ctx.get("dhw_speed_frac", 1.0))) * np.ones(np.shape(t_s_used))})


def identify(log: LogRecorder, *, t_room_c: float = 20.0, house: Optional[int] = None) -> Dict[str, object]:
    rows = log.rows
    if not rows:
        raise ValueError("empty log")
    sel = (lambda a: a) if house is None else (lambda a: a[house:house + 1])
    T0 = np.concatenate([sel(r["T0"]) for r in rows])
    T1 = np.concatenate([sel(r["T1"]) for r in rows])
    served = np.concatenate([sel(r["served"]) for r in rows])
    elec = np.concatenate([sel(r["elec_d"]) for r in rows])
    td = np.concatenate([sel(r["time_d"]) for r in rows])
    tsrc = np.concatenate([np.where(sel(r["t_s"]) > 60.0, sel(r["t_src_dis"]), sel(r["t_src"])) for r in rows])
    tsup = np.concatenate([sel(r["t_s"]) for r in rows])
    frac = np.concatenate([sel(r.get("frac", np.ones_like(r["t_s"]))) for r in rows])
    vol_total = np.concatenate([sel(r["vol"]) for r in rows])
    E0 = (T0.mean(axis=1)) * tk.CV * vol_total
    E1 = (T1.mean(axis=1)) * tk.CV * vol_total
    dE = E1 - E0
    dT = 0.5 * (T0.mean(axis=1) + T1.mean(axis=1)) - t_room_c
    idle = td <= 1e-9
    y = -dE[idle] - served[idle]
    x = dT[idle]
    ua_kw = float(np.sum(x * y) / max(np.sum(x * x), 1e-12))
    Q = dE + served + ua_kw * dT
    run = td > 0.12
    q = Q[run] / td[run] / frac[run]
    A = np.c_[np.ones(run.sum()), tsrc[run], tsup[run]]
    if np.ptp(tsup[run]) < 1.0:
        g01, *_ = np.linalg.lstsq(A[:, :2], q, rcond=None)
        g = np.r_[g01, 0.0]
    else:
        g, *_ = np.linalg.lstsq(A, q, rcond=None)
    eir = elec[run] / np.maximum(Q[run], 1e-9)
    Ah = np.c_[np.ones(run.sum()), tsup[run] - tsrc[run]]
    hcoef, *_ = np.linalg.lstsq(Ah, eir, rcond=None)
    return {"ua_w_k": ua_kw * 1000.0, "g": g.tolist(), "h": hcoef.tolist(),
            "n_idle_hours": int(idle.sum()), "n_dhw_hours": int(run.sum()),
            "supply_levels_c": sorted(set(np.round(tsup[run], 1).tolist())),
            "rms_rate_kw": float(np.sqrt(np.mean((A @ g - q) ** 2))) if run.any() else float("nan"),
            "rms_eir": float(np.sqrt(np.mean((Ah @ hcoef - eir) ** 2))) if run.any() else float("nan")}
