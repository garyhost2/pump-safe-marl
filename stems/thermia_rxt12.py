from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

DATA_PATH = Path(__file__).resolve().parent / "data" / "thermia_rxt12_keymark.json"


def load_data(path: Path = DATA_PATH) -> dict:
    with open(path) as fh:
        return json.load(fh)


_FULL_LOAD_CONDITIONS = ("TOL=Tdesign",)
_MIN_LOAD_CONDITIONS = ("D(+12)",)


def _arrays(points: Sequence[dict]):
    ts = np.array([p["t_source_c"] for p in points], float)
    tw = np.array([p["t_supply_c"] for p in points], float)
    q = np.array([p["q_kw"] for p in points], float)
    cop = np.array([p["cop"] for p in points], float)
    w = np.array([p["source"] == "water" for p in points], float)
    full = np.array([any(k in p["condition"] for k in _FULL_LOAD_CONDITIONS) for p in points])
    mn = np.array([any(k in p["condition"] for k in _MIN_LOAD_CONDITIONS) for p in points])
    return ts, tw, q, cop, w, full, mn


def _model(theta, ts, tw, q, w):
    dw, a0, a1, a2, b0, b1, b2, b3, b4, b5 = theta
    tse = ts + dw * w
    lift = tw - tse
    qmax = a0 + a1 * tse + a2 * tw
    s = q / qmax
    eir = b0 + b1 * lift + b2 * lift ** 2 + b3 * q + b4 / q + b5 * s ** 3
    return 1.0 / eir, qmax


_X0 = np.array([3.0, 12.47, 0.287, -0.0375, -0.01, 1e-3, 1e-4, 0.01, 0.3, 0.01])
THETA_NAMES = ("dw", "a0", "a1", "a2", "b0", "b1", "b2", "b3", "b4", "b5")


def fit(points: Optional[Sequence[dict]] = None, subset: Optional[np.ndarray] = None) -> Dict[str, float]:
    from scipy.optimize import least_squares

    pts = load_data()["operating_points"] if points is None else list(points)
    ts, tw, q, cop, w, full, mn = _arrays(pts)
    tr = np.arange(len(pts)) if subset is None else np.asarray(subset)
    trf = np.intersect1d(np.flatnonzero(full), tr)

    def resid(th):
        c, _ = _model(th, ts[tr], tw[tr], q[tr], w[tr])
        _, qm = _model(th, ts[trf], tw[trf], q[trf], w[trf])
        return np.r_[(c - cop[tr]) / cop[tr], (qm - q[trf]) / q[trf]]

    th = least_squares(resid, _X0, max_nfev=100000, xtol=1e-12, ftol=1e-12, gtol=1e-12).x
    out = {k: float(v) for k, v in zip(THETA_NAMES, th)}
    trm = np.intersect1d(np.flatnonzero(mn), tr)
    tse = ts[trm] + out["dw"] * w[trm]
    c1, c0 = np.polyfit(tse, q[trm], 1)
    out["c0"], out["c1"] = float(c0), float(c1)
    cdh = [p["cdh"] for i, p in enumerate(pts) if mn[i] and i in set(tr.tolist())]
    out["cdh"] = float(np.mean(cdh))
    return out


def fit_report(points: Optional[Sequence[dict]] = None) -> dict:
    pts = load_data()["operating_points"] if points is None else list(points)
    ts, tw, q, cop, w, _, _ = _arrays(pts)
    n = len(pts)

    def pred(th):
        theta = [th[k] for k in THETA_NAMES]
        return _model(theta, ts, tw, q, w)[0]

    e_in = (pred(fit(pts)) - cop) / cop
    e_loo = np.array([(pred(fit(pts, np.delete(np.arange(n), i)))[i] - cop[i]) / cop[i] for i in range(n)])
    brine = w == 0
    return {
        "n_points": n,
        "in_sample_rms": float(np.sqrt(np.mean(e_in ** 2))), "in_sample_max": float(np.abs(e_in).max()),
        "loo_rms": float(np.sqrt(np.mean(e_loo ** 2))), "loo_max": float(np.abs(e_loo).max()),
        "brine_loo_rms": float(np.sqrt(np.mean(e_loo[brine] ** 2))),
        "brine_loo_max": float(np.abs(e_loo[brine]).max()),
        "per_point": [dict(source=p["source"], t_supply_c=p["t_supply_c"], q_kw=p["q_kw"], cop_cert=p["cop"],
                           err_in=float(a), err_loo=float(b)) for p, a, b in zip(pts, e_in, e_loo)],
    }


@dataclass
class Operation:
    q_request_kw: float
    q_compressor_kw: float
    q_aux_kw: float
    p_compressor_kw: float
    p_aux_kw: float
    p_standby_kw: float
    cop: float
    duty: float
    starts: int
    aux_step_kw: float
    extrapolated: bool
    regime: str

    @property
    def p_total_kw(self) -> float:
        return self.p_compressor_kw + self.p_aux_kw + self.p_standby_kw

    @property
    def q_total_kw(self) -> float:
        return self.q_compressor_kw + self.q_aux_kw


class CalibraRXT12:

    def __init__(self, coefficients: Optional[Dict[str, float]] = None, data: Optional[dict] = None):
        self.data = load_data() if data is None else data
        self.k = fit(self.data["operating_points"]) if coefficients is None else dict(coefficients)
        lim = self.data["limits"]
        self.t_supply_max = float(lim["t_supply_max_c"])
        self.t_supply_min = float(lim["t_supply_min_c"])
        self.p_el_max = float(lim["compressor_max_el_kw_400v"]) + float(lim["circulation_pumps_rated_kw"])
        self.aux_steps = tuple(float(x) for x in lim["aux_heater_steps_kw_400v"])
        self.p_standby = float(self.data["standby_w"]["PSB"]) / 1000.0
        env = self.data["certified_envelope"]
        self.env_src = tuple(env["t_source_c"])
        self.env_sup = tuple(env["t_supply_c"])

    def _theta(self):
        return [self.k[n] for n in THETA_NAMES]

    def cop(self, q_kw, t_source, t_supply):
        c, _ = _model(self._theta(), np.asarray(t_source, float), np.asarray(t_supply, float),
                      np.asarray(q_kw, float), 0.0)
        return c

    def q_max_map(self, t_source, t_supply):
        _, qm = _model(self._theta(), np.asarray(t_source, float), np.asarray(t_supply, float), 1.0, 0.0)
        return qm

    def q_min(self, t_source):
        return self.k["c0"] + self.k["c1"] * np.asarray(t_source, float)

    def q_max(self, t_source: float, t_supply: float) -> float:
        if t_supply > self.t_supply_max:
            return 0.0
        qm = float(self.q_max_map(t_source, t_supply))
        if qm / float(self.cop(qm, t_source, t_supply)) <= self.p_el_max:
            return qm
        lo, hi = float(self.q_min(t_source)), qm
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if mid / float(self.cop(mid, t_source, t_supply)) > self.p_el_max:
                hi = mid
            else:
                lo = mid
        return lo

    def extrapolated(self, t_source: float, t_supply: float) -> bool:
        return not (self.env_src[0] <= t_source <= self.env_src[1] and self.env_sup[0] <= t_supply <= self.env_sup[1])

    def operate(self, q_request_kw: float, t_source: float, t_supply: float, dt_h: float = 1.0,
                was_running: bool = False) -> Operation:
        q_req = max(0.0, float(q_request_kw))
        ext = self.extrapolated(t_source, t_supply)
        if q_req <= 0.0:
            return Operation(0.0, 0.0, 0.0, 0.0, 0.0, self.p_standby, float("nan"), 0.0, 0, 0.0, ext, "off")
        qmax = self.q_max(t_source, t_supply)
        if qmax <= 0.0:
            step = self._aux_step(q_req)
            q_aux = min(q_req, self.aux_steps[-1])
            return Operation(q_req, 0.0, q_aux, 0.0, q_aux, self.p_standby, float("nan"), 0.0, 0, step, True, "above_limit")
        qmin = float(self.q_min(t_source))
        if q_req < qmin:
            cr = q_req / qmin
            cdh = self.k["cdh"]
            c = float(self.cop(qmin, t_source, t_supply)) * cr / (cdh * cr + (1.0 - cdh))
            return Operation(q_req, q_req, 0.0, q_req / c, 0.0, 0.0, c, cr, 1, 0.0, ext, "cycling")
        starts = 0 if was_running else 1
        if q_req <= qmax:
            c = float(self.cop(q_req, t_source, t_supply))
            return Operation(q_req, q_req, 0.0, q_req / c, 0.0, 0.0, c, 1.0, starts, 0.0, ext, "modulating")
        c = float(self.cop(qmax, t_source, t_supply))
        deficit = q_req - qmax
        q_aux = min(deficit, self.aux_steps[-1])
        return Operation(q_req, qmax, q_aux, qmax / c, q_aux, 0.0, c, 1.0, starts, self._aux_step(deficit), ext, "full+aux")

    def _aux_step(self, deficit: float) -> float:
        for s in self.aux_steps:
            if deficit <= s + 1e-12:
                return s
        return self.aux_steps[-1]
