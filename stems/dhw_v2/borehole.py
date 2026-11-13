from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional

import numpy as np

__all__ = ["BoreholeSpec", "Borehole", "BrineCorrelation", "make_brine_model", "spinup_sequence",
           "reference_extraction_kw"]


@dataclass
class BoreholeSpec:
    H_m: float = 200.0
    D_m: float = 4.0
    r_b_m: float = 0.0575
    k_s_w_mk: float = 3.5
    rho_c_s_j_m3k: float = 2.16e6
    t_ground_c: float = 8.0
    R_b_mk_w: float = 0.08
    cp_f_j_kgk: float = 4180.0
    m_flow_kg_s: Optional[float] = None
    years: int = 2
    spinup_years: int = 0

    @property
    def alpha(self) -> float:
        return self.k_s_w_mk / self.rho_c_s_j_m3k


def _design_flow(hp, cp):
    q = float(hp.q_max(0.0, 35.0))
    p = float(hp.p_el(q, 0.0, 35.0))
    return (q - p) * 1000.0 / (3.0 * cp)


class Borehole:

    def __init__(self, spec: BoreholeSpec, hp=None):
        import pygfunction as gt
        spec = replace(spec)
        self.spec = spec
        if spec.m_flow_kg_s is None:
            if hp is None:
                from stems.dhw_v2.heatpump import RXT12Map
                hp = RXT12Map()
            spec.m_flow_kg_s = _design_flow(hp, spec.cp_f_j_kgk)
        self.dt = 3600.0
        tmax = (spec.years + spec.spinup_years) * 8784 * 3600.0
        self._la = gt.load_aggregation.ClaessonJaved(self.dt, tmax)
        times = self._la.get_times_for_simulation()
        b = gt.boreholes.Borehole(spec.H_m, spec.D_m, spec.r_b_m, 0.0, 0.0)
        g = gt.gfunction.gFunction([b], spec.alpha, time=times, boundary_condition="UHTR",
                                   method="similarities")
        self.g = np.asarray(g.gFunc, dtype=np.float64)
        self.times = np.asarray(times, dtype=np.float64)
        self._la.initialize(self.g / (2.0 * np.pi * spec.k_s_w_mk))
        self.k = 0
        self.t_wall = float(spec.t_ground_c)

    def wall_temperature(self) -> float:
        return self.t_wall

    def inlet(self, q_ext_kw, t_wall=None):
        s = self.spec
        tb = self.t_wall if t_wall is None else t_wall
        q = np.asarray(q_ext_kw, dtype=np.float64) * 1000.0
        return tb - q / s.H_m * s.R_b_mk_w + q / (2.0 * s.m_flow_kg_s * s.cp_f_j_kgk)

    def advance(self, mean_q_ext_kw: float) -> float:
        self.k += 1
        self._la.next_time_step(self.k * self.dt)
        self._la.set_current_load(np.array([float(mean_q_ext_kw) * 1000.0 / self.spec.H_m]))
        dT = float(np.asarray(self._la.temporal_superposition()).ravel()[0])
        self.t_wall = self.spec.t_ground_c - dT
        return self.t_wall

    def spin_up(self, q_ext_kw) -> float:
        for q in np.asarray(q_ext_kw, dtype=np.float64):
            self.advance(float(max(q, 0.0)))
        self.spun_hours = self.k
        return self.t_wall

    def describe(self) -> Dict[str, object]:
        d = asdict(self.spec)
        d.update(model="pygfunction g-function (UHTR, similarities) + Claesson-Javed aggregation",
                 alpha_m2_s=self.spec.alpha, spun_hours=int(getattr(self, "spun_hours", 0)))
        return d


def spinup_sequence(profile, offset: int, n: int) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    return profile[np.arange(int(offset) - int(n), int(offset)) % len(profile)]


def reference_extraction_kw(data, house, hp, *, t_supply_dhw_c: float = 55.0, dhw_speed_frac: float = 1.0,
                            store_loss_kw: float = 0.0, src_drop_k: float = 3.0, aux: bool = True) -> np.ndarray:
    from stems.dhw_v2.building import heating_curve_c, make_building
    T = data.T
    bl = make_building(house.building)
    t_src = float(data.t_ground_c) - src_drop_k
    ts = np.asarray(heating_curve_c(data.t_out_c, bl.curve), dtype=np.float64)
    on = np.isfinite(ts)
    tsv = np.where(on, ts, 35.0)
    qmax = np.where(on, hp.q_max(np.full(T, t_src), tsv), 0.0)
    heat = np.zeros(T)
    for t in range(T):
        te, g = float(data.t_out_c[t]), float(data.ghi_w_m2[t])
        q = 0.0
        top = 0.0
        if on[t]:
            q = min(bl.need_w(bl.t_set, te, g) / 1000.0, float(qmax[t]))
            if aux:
                top = min(max(bl.need_w(bl.t_min, te, g) / 1000.0 - q, 0.0), 9.0)
        bl.step((q + top) * 1000.0, te, g)
        heat[t] = q
    p_s = np.where(heat > 0, hp.p_el(heat, t_src, tsv), 0.0)
    k = 60 // house.draw_dt_min
    d = np.asarray(house.draw_kwh[: T * k]).reshape(T, k).sum(axis=1) + store_loss_kw
    qd = dhw_speed_frac * float(hp.q_max(t_src, t_supply_dhw_c))
    cop_d = qd / float(hp.p_el(qd, t_src, t_supply_dhw_c))
    return np.maximum(heat - p_s, 0.0) + d * (1.0 - 1.0 / cop_d)


class BrineCorrelation:

    def __init__(self, t_out_c: np.ndarray):
        from stems.heatpump_hplib import brine_temperature_c
        t = np.asarray(t_out_c, dtype=np.float64)
        c = np.cumsum(np.r_[0.0, t])
        idx = np.arange(1, len(t) + 1)
        lo = np.maximum(idx - 24, 0)
        trailing = (c[idx] - c[lo]) / (idx - lo)
        self.series = brine_temperature_c(trailing)
        self.k = 0

    def wall_temperature(self) -> float:
        return float(self.series[min(self.k, len(self.series) - 1)])

    def inlet(self, q_ext_kw, t_wall=None):
        tb = self.wall_temperature() if t_wall is None else t_wall
        return np.full(np.shape(q_ext_kw), tb, dtype=np.float64) if np.ndim(q_ext_kw) else tb

    def advance(self, mean_q_ext_kw: float) -> float:
        self.k += 1
        return self.wall_temperature()

    def describe(self) -> Dict[str, object]:
        return {"model": "hplib WP-Monitor correlation on trailing 24 h mean T_out (v1)"}


def make_brine_model(kind: str, *, t_out_c=None, spec: Optional[BoreholeSpec] = None, hp=None):
    if kind == "borehole":
        return Borehole(spec or BoreholeSpec(), hp=hp)
    if kind == "correlation":
        return BrineCorrelation(t_out_c)
    raise ValueError(kind)
