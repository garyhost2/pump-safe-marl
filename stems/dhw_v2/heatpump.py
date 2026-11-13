from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from stems.thermia_rxt12 import CalibraRXT12

__all__ = ["RXT12Map", "HplibMap", "make_heatpump"]


def _f(x):
    return np.asarray(x, dtype=np.float64)


class RXT12Map:

    name = "rxt12"

    def __init__(self, coefficients: Optional[Dict[str, float]] = None, *,
                 p_el_max_kw: Optional[float] = None):
        self.unit = CalibraRXT12(coefficients)
        self.k = dict(self.unit.k)
        self.t_supply_max = self.unit.t_supply_max
        self.p_el_max = float(self.unit.p_el_max if p_el_max_kw is None else p_el_max_kw)
        self.p_standby_kw = float(self.unit.p_standby)
        self.env_src = self.unit.env_src
        self.env_sup = self.unit.env_sup
        self.provenance = {
            "map": "stems/thermia_rxt12.py fit to KEYMARK 012-C700403 (21 points)",
            "p_el_max_kw": "datasheet compressor 4.7 kW + circulation pumps 0.2 kW",
            "p_standby_kw": "KEYMARK PSB 17 W",
            "extrapolated": "supply > 55 degC (no certified point), source outside 0-10 degC"}

    def _eir(self, q, t_src, t_sup):
        k = self.k
        lift = t_sup - t_src
        qmax_map = k["a0"] + k["a1"] * t_src + k["a2"] * t_sup
        s = q / qmax_map
        return (k["b0"] + k["b1"] * lift + k["b2"] * lift ** 2 + k["b3"] * q
                + k["b4"] / q + k["b5"] * s ** 3)

    def q_min(self, t_src, t_sup=None):
        return self.k["c0"] + self.k["c1"] * _f(t_src)

    def q_max(self, t_src, t_sup):
        t_src, t_sup = np.broadcast_arrays(_f(t_src), _f(t_sup))
        k = self.k
        qm = k["a0"] + k["a1"] * t_src + k["a2"] * t_sup
        over = qm * self._eir(qm, t_src, t_sup) > self.p_el_max
        lo = np.minimum(self.q_min(t_src), qm)
        hi = qm.copy()
        if np.any(over):
            for _ in range(50):
                mid = 0.5 * (lo + hi)
                bad = mid * self._eir(mid, t_src, t_sup) > self.p_el_max
                hi = np.where(bad, mid, hi)
                lo = np.where(bad, lo, mid)
            qm = np.where(over, lo, qm)
        return np.where(t_sup > self.t_supply_max, 0.0, qm)

    def p_el(self, q, t_src, t_sup):
        q, t_src, t_sup = np.broadcast_arrays(_f(q), _f(t_src), _f(t_sup))
        qmin = self.q_min(t_src)
        q_safe = np.maximum(q, qmin)
        p_mod = q_safe * self._eir(q_safe, t_src, t_sup)
        p_qmin = qmin * self._eir(qmin, t_src, t_sup)
        cdh = self.k["cdh"]
        p_cyc = p_qmin * (cdh * q / qmin + 1.0 - cdh)
        return np.where(q <= 0.0, 0.0, np.where(q < qmin, p_cyc, p_mod))

    def cop(self, q, t_src, t_sup):
        q = _f(q)
        p = self.p_el(q, t_src, t_sup)
        return np.where(p > 0.0, q / np.where(p > 0.0, p, 1.0), np.nan)

    def chord(self, t_src, t_sup):
        t_src, t_sup = np.broadcast_arrays(_f(t_src), _f(t_sup))
        qmin = self.q_min(t_src)
        qmax = self.q_max(t_src, t_sup)
        p_qmin = qmin * self._eir(qmin, t_src, t_sup)
        cdh = self.k["cdh"]
        p0 = (1.0 - cdh) * p_qmin
        pq = self.p_el(qmax, t_src, t_sup)
        safe = np.where(qmax > 0.0, qmax, 1.0)
        kappa = np.maximum(cdh * p_qmin / qmin, (pq - p0) / safe)
        kappa = np.where(qmax > 0.0, kappa, 0.0) * (1.0 + 1e-9)
        return p0, kappa

    def flags(self, t_src, t_sup):
        t_src, t_sup = np.broadcast_arrays(_f(t_src), _f(t_sup))
        return {"extrap_supply": (t_sup > self.env_sup[1]) | (t_sup < self.env_sup[0]),
                "extrap_source": (t_src < self.env_src[0]) | (t_src > self.env_src[1])}

    def describe(self) -> dict:
        return {"name": self.name, "coefficients": self.k, "p_el_max_kw": self.p_el_max,
                "t_supply_max_c": self.t_supply_max, "certified_source_c": list(self.env_src),
                "certified_supply_c": list(self.env_sup), "provenance": self.provenance}


class HplibMap:

    name = "hplib_calibra_eco_8"

    def __init__(self, spec=None):
        from stems.heatpump_hplib import calibra_eco_8
        self.spec = calibra_eco_8() if spec is None else spec
        self.t_supply_max = float(self.spec.t_sink_max_c)
        self.p_el_max = np.inf
        self.p_standby_kw = 0.0
        self.k = {}

    def q_max(self, t_src, t_sup):
        t_src, t_sup = np.broadcast_arrays(_f(t_src), _f(t_sup))
        c = _f(self.spec.capacity_kw(t_src, t_sup))
        return np.where(t_sup > self.t_supply_max, 0.0, np.maximum(c, 0.0))

    def q_min(self, t_src, t_sup=None):
        t_src = _f(t_src)
        return _f(self.spec.min_output_kw(t_src, 55.0 if t_sup is None else t_sup)) * np.ones_like(t_src)

    def p_el(self, q, t_src, t_sup):
        q, t_src, t_sup = np.broadcast_arrays(_f(q), _f(t_src), _f(t_sup))
        cop = _f(self.spec.cop(t_src, t_sup))
        return np.where(q > 0.0, q / cop, 0.0)

    def cop(self, q, t_src, t_sup):
        return _f(self.spec.cop(t_src, t_sup)) * np.ones_like(_f(q))

    def chord(self, t_src, t_sup):
        cop = _f(self.spec.cop(t_src, t_sup))
        return np.zeros_like(cop), (1.0 / cop) * (1.0 + 1e-9)

    def flags(self, t_src, t_sup):
        t_src, t_sup = np.broadcast_arrays(_f(t_src), _f(t_sup))
        z = np.ones(t_src.shape, dtype=bool)
        return {"extrap_supply": z, "extrap_source": z}

    def describe(self) -> dict:
        return {"name": self.name, "note": "v1 plant, wrong unit, before/after only",
                "spec": self.spec.describe() if hasattr(self.spec, "describe") else str(self.spec)}


def make_heatpump(name: str = "rxt12", **kw):
    if name == "rxt12":
        return RXT12Map(**kw)
    if name in ("hplib", "hplib_calibra_eco_8"):
        return HplibMap(**kw)
    raise ValueError(f"unknown heat pump {name!r}")
