from __future__ import annotations

from typing import Dict, Optional

import numpy as np

__all__ = ["DEFAULT_SE_SFH", "heating_curve_c", "RC5R1C", "RC1R1C", "DemandBuilding",
           "make_building"]

DEFAULT_SE_SFH: Dict[str, object] = {
    "model": "5R1C",
    "source": "PLACEHOLDER: sized for 10 kW design load at -18 degC; replace by TABULA SE data",
    "A_f_m2": 140.0, "H_ve_W_K": 60.0, "H_tr_w_W_K": 40.0, "H_tr_op_W_K": 156.4,
    "C_m_J_K": 165000.0 * 140.0, "solar_aperture_m2": 3.0, "internal_gains_W": 400.0,
    "T_set_c": 21.0, "T_comfort_min_c": 20.0,
    "heating_curve": {"curve_c": 40.0, "T_supply_min_c": 20.0, "T_supply_max_c": 55.0,
                      "seasonal_stop_c": 17.0},
}


def heating_curve_c(t_out_c, curve: Optional[Dict[str, float]] = None):
    c = {"curve_c": 40.0, "T_supply_min_c": 20.0, "T_supply_max_c": 55.0, "seasonal_stop_c": 17.0}
    if curve:
        c.update(curve)
    t = np.asarray(t_out_c, dtype=np.float64)
    ts = 20.0 + (c["curve_c"] - 20.0) * (20.0 - t) / 20.0
    ts = np.clip(ts, c["T_supply_min_c"], c["T_supply_max_c"])
    return np.where(t >= c["seasonal_stop_c"], np.nan, ts)


def _gains(p):
    if "internal_gains_W" in p:
        return float(p["internal_gains_W"])
    return float(p.get("internal_gains_W_m2", 3.0)) * float(p["A_f_m2"])


class RC5R1C:
    def __init__(self, p: Dict[str, object], t_init_c: Optional[float] = None):
        self.p = dict(p)
        A_f = float(p["A_f_m2"])
        self.A_m = float(p.get("A_m_m2", 2.5 * A_f))
        self.A_t = float(p.get("A_t_m2", 4.5 * A_f))
        self.H_ve = float(p["H_ve_W_K"])
        self.H_w = float(p["H_tr_w_W_K"])
        H_op = float(p["H_tr_op_W_K"])
        self.C_m = float(p["C_m_J_K"])
        self.H_is = 3.45 * self.A_t
        self.H_ms = 9.1 * self.A_m
        self.H_em = 1.0 / (1.0 / H_op - 1.0 / self.H_ms)
        self.H_1 = 1.0 / (1.0 / self.H_ve + 1.0 / self.H_is)
        self.H_2 = self.H_1 + self.H_w
        self.H_3 = 1.0 / (1.0 / self.H_2 + 1.0 / self.H_ms)
        self.aperture = float(p.get("solar_aperture_m2", 0.0))
        self.g_int = _gains(p)
        self.t_set = float(p.get("T_set_c", 21.0))
        self.t_min = float(p.get("T_comfort_min_c", self.t_set - 1.0))
        self.curve = dict(p.get("heating_curve", {}))
        self.reset(t_init_c)

    def reset(self, t_init_c=None):
        t0 = self.t_set if t_init_c is None else float(t_init_c)
        self.theta_m = t0
        self.t_in = t0

    def _solve(self, phi_hc, t_e, ghi, dt_s=3600.0):
        phi_int = self.g_int
        phi_sol = self.aperture * max(float(ghi), 0.0)
        phi_ia = 0.5 * phi_int
        phi_m = self.A_m / self.A_t * (0.5 * phi_int + phi_sol)
        phi_st = (1.0 - self.A_m / self.A_t - self.H_w / (9.1 * self.A_t)) * (0.5 * phi_int + phi_sol)
        t_sup = t_e
        phi_mtot = (phi_m + self.H_em * t_e + self.H_3 / self.H_2 *
                    (phi_st + self.H_w * t_e + self.H_1 * ((phi_ia + phi_hc) / self.H_ve + t_sup)))
        c = self.C_m / dt_s
        tm_new = (self.theta_m * (c - 0.5 * (self.H_3 + self.H_em)) + phi_mtot) / (c + 0.5 * (self.H_3 + self.H_em))
        tm = 0.5 * (tm_new + self.theta_m)
        ts = ((self.H_ms * tm + phi_st + self.H_w * t_e + self.H_1 * (t_sup + (phi_ia + phi_hc) / self.H_ve))
              / (self.H_ms + self.H_w + self.H_1))
        t_air = (self.H_is * ts + self.H_ve * t_sup + phi_ia + phi_hc) / (self.H_is + self.H_ve)
        return t_air, tm_new

    def need_w(self, t_target, t_e, ghi):
        a0, _ = self._solve(0.0, t_e, ghi)
        a1, _ = self._solve(1000.0, t_e, ghi)
        return max(0.0, (t_target - a0) / (a1 - a0) * 1000.0)

    def step(self, phi_w, t_e, ghi):
        self.t_in, self.theta_m = self._solve(float(phi_w), t_e, ghi)
        return self.t_in

    def state(self):
        return {"t_in": self.t_in, "theta_m": self.theta_m}


class RC1R1C:
    def __init__(self, p: Dict[str, object], t_init_c: Optional[float] = None):
        self.p = dict(p)
        self.H = float(p["H_W_K"])
        self.C = float(p["C_J_K"])
        self.aperture = float(p.get("solar_aperture_m2", 0.0))
        self.g_int = _gains(p)
        self.t_set = float(p.get("T_set_c", 21.0))
        self.t_min = float(p.get("T_comfort_min_c", self.t_set - 1.0))
        self.curve = dict(p.get("heating_curve", {}))
        self.reset(t_init_c)

    def reset(self, t_init_c=None):
        self.t_in = self.t_set if t_init_c is None else float(t_init_c)

    def _end(self, phi, t_e, ghi, dt_s=3600.0):
        g = self.g_int + self.aperture * max(float(ghi), 0.0) + phi
        teq = t_e + g / self.H
        return teq + (self.t_in - teq) * np.exp(-self.H * dt_s / self.C)

    def need_w(self, t_target, t_e, ghi):
        a0 = self._end(0.0, t_e, ghi)
        a1 = self._end(1000.0, t_e, ghi)
        return max(0.0, (t_target - a0) / (a1 - a0) * 1000.0)

    def step(self, phi_w, t_e, ghi):
        self.t_in = float(self._end(float(phi_w), t_e, ghi))
        return self.t_in

    def state(self):
        return {"t_in": self.t_in}


class DemandBuilding:

    C_EFF_KWH_K = 5.0

    def __init__(self, p: Dict[str, object]):
        self.p = dict(p)
        self.demand = np.asarray(p["heating_demand_kwh"], dtype=np.float64)
        self.t_set = float(p.get("T_set_c", 21.0))
        self.t_min = float(p.get("T_comfort_min_c", self.t_set - 1.0))
        self.curve = dict(p.get("heating_curve", {}))
        self.k = 0
        self.reset()

    def reset(self, t_init_c=None):
        self.debt = 0.0
        self.k = 0
        self.t_in = self.t_set

    def need_w(self, t_target, t_e, ghi):
        d = self.demand[min(self.k, len(self.demand) - 1)] + self.debt
        slack = max(0.0, (self.t_set - t_target)) * self.C_EFF_KWH_K
        return max(0.0, d - slack) * 1000.0

    def step(self, phi_w, t_e, ghi):
        d = self.demand[min(self.k, len(self.demand) - 1)] + self.debt
        self.debt = max(0.0, d - float(phi_w) / 1000.0)
        self.k += 1
        self.t_in = self.t_set - self.debt / self.C_EFF_KWH_K
        return self.t_in

    def state(self):
        return {"t_in": self.t_in, "debt_kwh": self.debt}


def make_building(p: Optional[Dict[str, object]]):
    p = DEFAULT_SE_SFH if p is None else p
    m = str(p.get("model", "5R1C")).upper()
    if m == "5R1C":
        return RC5R1C(p)
    if m == "1R1C":
        return RC1R1C(p)
    if m == "DEMAND":
        return DemandBuilding(p)
    raise ValueError(f"unknown building model {p.get('model')!r}")
