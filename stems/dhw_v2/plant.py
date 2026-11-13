from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from stems.dhw_v2 import tank as tk
from stems.dhw_v2.borehole import Borehole, BoreholeSpec, make_brine_model, reference_extraction_kw, spinup_sequence
from stems.dhw_v2.calibration import DHW_SPEED_FRAC
from stems.dhw_v2.building import heating_curve_c, make_building
from stems.dhw_v2.heatpump import make_heatpump

__all__ = ["PlantConfig", "simulate_hour", "tank_arrays", "DistrictPlant"]


@dataclass
class PlantConfig:
    dt_min: int = 5
    t_supply_dhw_c: float = 55.0
    t_supply_dis_c: float = 65.0
    t_floor_c: float = 45.0
    t_use_c: float = 40.0
    t_dis_c: float = 60.0
    hold_rule: str = "bottom20"
    hold_min: float = 60.0
    profile: str = "uniform"
    brine: str = "borehole"
    heatpump: str = "rxt12"
    space_heating: bool = True
    t_init_c: float = 55.0
    dhw_speed_frac: float = DHW_SPEED_FRAC
    dis_speed_frac: Optional[float] = None
    aux_heater: str = "auto_min"
    aux_steps_kw: tuple = (3.0, 6.0, 9.0)

    def __post_init__(self):
        if self.dt_min not in (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60) or 60 % self.dt_min:
            raise ValueError("dt_min must divide 60")
        if self.aux_heater not in ("off", "auto_min", "auto_set"):
            raise ValueError("aux_heater must be 'off', 'auto_min' or 'auto_set'")
        self.aux_steps_kw = tuple(sorted(float(x) for x in self.aux_steps_kw))


def bottom_weights(N: int, frac: float = 0.2) -> np.ndarray:
    edges = np.linspace(0.0, 1.0, N + 1)
    lo = 1.0 - frac
    w = np.clip(np.minimum(edges[1:], 1.0) - np.maximum(edges[:-1], lo), 0.0, None)
    return w / w.sum()


HOLD_KEYS = {"bottom20": "bot_min", "whole_mean": "mean_min", "all_nodes": "nodes_min"}


def tank_arrays(tanks: Sequence[tk.TankSpec2]) -> Dict[str, np.ndarray]:
    n = {t.nodes for t in tanks}
    lm = {t.loss_model for t in tanks}
    if len(n) != 1 or len(lm) != 1:
        raise ValueError("all houses of one plant need the same node count and loss model")
    f = lambda xs: np.array([float(x) for x in xs])
    return {"node_vol": f(t.node_volume_l for t in tanks), "ua": f(t.ua_w_k for t in tanks),
            "t_room": f(t.t_room_c for t in tanks), "cl_loss": f(t.citylearn_loss_per_h for t in tanks),
            "t_use": float(tanks[0].t_use_c), "loss_model": tanks[0].loss_model,
            "nodes": tanks[0].nodes, "sidx": tk.sensor_index(tanks[0]),
            "w_bot": tuple(bottom_weights(tanks[0].nodes, 0.2).tolist())}


def simulate_hour(T0, tank: Dict, draws, dt_h: float, *, u_d, r_d, p_d, t_s, t_cold,
                  thermo: Optional[Dict] = None, sh: Optional[Dict] = None, trace: bool = False,
                  t_dis_c: float = 60.0, t_floor_c: float = 45.0, comp_on0=None,
                  hold_rule: str = "bottom20", hold_steps: Optional[int] = None,
                  series: bool = False) -> Dict[str, np.ndarray]:
    T0 = np.asarray(T0, dtype=np.float64)
    S = draws.shape[0]
    bshape = np.broadcast_shapes(T0.shape[:-1], np.shape(u_d), np.shape(t_s), np.shape(r_d),
                                 np.shape(draws)[1:])
    T = np.broadcast_to(T0, bshape + T0.shape[-1:]).copy()
    z = lambda: np.zeros(bshape)
    h = np.broadcast_to(tank["node_vol"], bshape)
    acc = {k: z() for k in ("served", "unmet", "loss", "heat_d", "elec_d", "time_d", "heat_s",
                            "elec_s", "time_s", "time_comp", "starts", "heat_leg", "time_below")}
    top_min = np.full(bshape, np.inf)
    v40_min = np.full(bshape, np.inf)
    nmin = np.full(bshape, np.inf)
    botmin = np.full(bshape, np.inf)
    meanmin = np.full(bshape, np.inf)
    deliv = np.full(bshape, np.inf)
    hold_key = HOLD_KEYS[hold_rule]
    hold_need = S if hold_steps is None else max(1, min(int(hold_steps), S))
    hold_run = np.zeros(bshape, dtype=np.int64)
    hold_best = np.zeros(bshape, dtype=np.int64)
    ser = {k: [] for k in ("top", "sensor", "bot", "mean", "nmin")} if series else None
    w_bot = np.asarray(tank["w_bot"], dtype=np.float64)
    Nn = T.shape[-1]
    u_d = np.broadcast_to(np.asarray(u_d, dtype=np.float64), bshape)
    r_d = np.broadcast_to(np.asarray(r_d, dtype=np.float64), bshape)
    p_d = np.broadcast_to(np.asarray(p_d, dtype=np.float64), bshape)
    t_s = np.broadcast_to(np.asarray(t_s, dtype=np.float64), bshape)
    tc = np.broadcast_to(np.asarray(t_cold, dtype=np.float64), bshape)
    c0 = comp_on0 if comp_on0 is not None else (thermo.get("comp_on", False) if thermo is not None else False)
    comp_prev = np.broadcast_to(np.asarray(c0, dtype=bool), bshape).copy()
    if thermo is not None:
        on = np.broadcast_to(np.asarray(thermo["on"], dtype=bool), bshape).copy()
        armed = np.broadcast_to(np.asarray(thermo.get("leg_armed", False), dtype=bool), bshape).copy()
        leg = np.broadcast_to(np.asarray(thermo.get("leg_active", False), dtype=bool), bshape).copy()
        leg_started = np.zeros(bshape, dtype=bool)
        leg_done = np.zeros(bshape, dtype=bool)
        sidx = int(tank["sidx"])
    if sh is not None:
        E_rem = np.broadcast_to(np.asarray(sh["E"], dtype=np.float64), bshape).copy()
        q_max_s = np.broadcast_to(np.asarray(sh["q_max"], dtype=np.float64), bshape)
    tr_top = [] if trace else None
    tr_nodes = [] if trace else None
    for s in range(S):
        T, lost = tk.loss(T, tank["ua"], tank["t_room"], dt_h, h, model=tank["loss_model"],
                          citylearn_loss_per_h=tank["cl_loss"], t_cold_c=tc)
        T_pre = T
        T, served, unmet, vol = tk.draw(T, h, draws[s], tc, tank["t_use"])
        xdeep = np.ceil(vol / np.where(h > 0, h, 1.0) - 1e-12).astype(np.int64) - 1
        tdeep = np.take_along_axis(T_pre, np.clip(xdeep, 0, Nn - 1)[..., None], -1)[..., 0]
        deliv = np.where(served > 1e-12, np.minimum(deliv, tdeep), deliv)
        acc["served"] += served
        acc["unmet"] += unmet
        acc["loss"] += lost
        top_min = np.minimum(top_min, T[..., 0])
        acc["time_below"] += np.where(T[..., 0] < t_floor_c - 1e-9, dt_h, 0.0)
        cur = {"nodes_min": T.min(axis=-1), "bot_min": T @ w_bot, "mean_min": T.mean(axis=-1)}
        nmin = np.minimum(nmin, cur["nodes_min"])
        botmin = np.minimum(botmin, cur["bot_min"])
        meanmin = np.minimum(meanmin, cur["mean_min"])
        hold_run = np.where(cur[hold_key] >= t_dis_c - 1e-9, hold_run + 1, 0)
        hold_best = np.maximum(hold_best, hold_run)
        if series:
            ser["top"].append(T[..., 0].copy())
            ser["sensor"].append(T[..., int(tank["sidx"])].copy())
            ser["bot"].append(cur["bot_min"])
            ser["mean"].append(cur["mean_min"])
            ser["nmin"].append(cur["nodes_min"])
        v40_min = np.minimum(v40_min, tk.v40_l(T, h, tc, tank["t_use"]))
        if thermo is None:
            tau = np.clip(u_d * 1.0 - s * dt_h, 0.0, dt_h)
            sup, rate, pw = t_s, r_d, p_d
        else:
            sensor = T[..., sidx]
            start = (~on) & (sensor < thermo["start_c"])
            new_leg = start & armed
            leg = leg | new_leg
            leg_started |= new_leg
            armed = armed & ~new_leg
            on = on | start | leg
            sup = np.where(leg, thermo["t_s_leg"], t_s)
            rate = np.where(leg, thermo["r_leg"], r_d)
            pw = np.where(leg, thermo["p_leg"], p_d)
            tau = np.where(on, dt_h, 0.0)
        a = tk.accept_max(T, h, sup)
        Q = np.minimum(rate * tau, a)
        T, Q, _ = tk.charge(T, h, Q, sup)
        tau_act = np.where(rate > 0.0, Q / np.where(rate > 0.0, rate, 1.0), 0.0)
        acc["heat_d"] += Q
        acc["elec_d"] += pw * tau_act
        acc["time_d"] += tau_act
        if thermo is not None:
            acc["heat_leg"] += np.where(leg, Q, 0.0)
            full = (a - Q) <= 1e-9
            stop_n = (~leg) & on & ((T[..., sidx] >= thermo["stop_c"]) | full)
            bot = T @ w_bot
            stop_l = leg & ((bot >= thermo["leg_target_c"]) | full)
            leg_done |= stop_l & (bot >= thermo["leg_target_c"] - 1e-9)
            on = on & ~(stop_n | stop_l)
            leg = leg & ~stop_l
        if sh is not None:
            avail = np.maximum(dt_h - tau_act, 0.0)
            if thermo is None:
                left = np.maximum((S - s) * dt_h - np.maximum(u_d * 1.0 - s * dt_h, 0.0), avail)
            else:
                left = np.maximum((S - s) * dt_h - tau_act, 0.0)
            rate_s = np.where(left > 1e-12, np.minimum(q_max_s, E_rem / np.where(left > 1e-12, left, 1.0)), 0.0)
            rate_s = np.where(E_rem > 1e-12, rate_s, 0.0)
            e = rate_s * avail
            E_rem = E_rem - e
            t_src_s = sh["t_wall"] - sh["coef"] * np.maximum(rate_s - sh["p_guess"](rate_s), 0.0)
            acc["elec_s"] += sh["hp"].p_el(rate_s, t_src_s, sh["t_sup"]) * avail
            acc["heat_s"] += e
            acc["time_s"] += np.where(rate_s > 0, avail, 0.0)
            comp_now = (tau_act > 0) | (rate_s * avail > 0)
            acc["time_comp"] += tau_act + np.where(rate_s > 0, avail, 0.0)
        else:
            comp_now = tau_act > 0
            acc["time_comp"] += tau_act
        acc["starts"] += (comp_now & ~comp_prev)
        comp_prev = comp_now
        if trace:
            tr_top.append(T[..., 0].copy())
            tr_nodes.append(T.copy())
    out = dict(acc)
    out.update(T_end=T, top_min=top_min, v40_min=v40_min, nodes_min=nmin, bot_min=botmin,
               mean_min=meanmin, delivered_min=deliv, hold=hold_best >= hold_need,
               hold_whole=meanmin >= t_dis_c - 1e-9, hold_all=nmin >= t_dis_c - 1e-9,
               hold_bot=botmin >= t_dis_c - 1e-9, comp_on_end=comp_prev)
    if thermo is not None:
        out.update(on_end=on, leg_armed_end=armed, leg_active_end=leg, leg_started=leg_started,
                   leg_done=leg_done)
    if sh is not None:
        out["sh_shortfall"] = np.maximum(E_rem, 0.0)
    if trace:
        out["trace_top"] = np.stack(tr_top)
        out["trace_nodes"] = np.stack(tr_nodes)
    if series:
        out["series"] = {k: np.stack(v) for k, v in ser.items()}
    return out


_SPINUP_CACHE: Dict[tuple, tuple] = {}


class HouseDataNoSH:

    def __init__(self, h):
        self.draw_kwh, self.draw_dt_min = h.draw_kwh, h.draw_dt_min
        self.building = {"model": "demand", "heating_demand_kwh": np.zeros(1), "T_set_c": 21.0,
                         "heating_curve": {"seasonal_stop_c": -99.0}}


class DistrictPlant:

    def __init__(self, data, *, tanks: Union[tk.TankSpec2, Sequence[tk.TankSpec2], None] = None,
                 cfg: Optional[PlantConfig] = None, borehole: Optional[BoreholeSpec] = None,
                 hp=None):
        self.data = data
        self.cfg = cfg or PlantConfig()
        B = data.B
        if tanks is None:
            tanks = tk.rxt12_tank(2)
        self.tanks = [tanks] * B if isinstance(tanks, tk.TankSpec2) else list(tanks)
        self.tank = tank_arrays(self.tanks)
        self.hp = hp if hp is not None else make_heatpump(self.cfg.heatpump)
        self.borehole_spec = borehole or BoreholeSpec(t_ground_c=data.t_ground_c)
        self.dt_h = self.cfg.dt_min / 60.0
        self.S = 60 // self.cfg.dt_min
        self.hold_steps = max(1, min(self.S, int(np.ceil(self.cfg.hold_min / self.cfg.dt_min - 1e-9))))
        self.draws = data.draws_substep(self.cfg.dt_min, self.cfg.profile)
        self.reset()

    def reset(self, t_init_c: Optional[float] = None):
        B, N = self.data.B, self.tank["nodes"]
        t0 = self.cfg.t_init_c if t_init_c is None else t_init_c
        self.T = np.full((B, N), float(t0))
        self.t = 0
        self.comp_on = np.zeros(B, dtype=bool)
        self.thermo_on = np.zeros(B, dtype=bool)
        self.leg_active = np.zeros(B, dtype=bool)
        self.brine = [make_brine_model(self.cfg.brine, t_out_c=self.data.t_out_c,
                                       spec=self.borehole_spec, hp=self.hp) for _ in range(B)]
        self.buildings = [make_building(h.building) for h in self.data.houses] \
            if self.cfg.space_heating else [None] * B
        self._ctx = None
        self.spinup_info = None
        if self.cfg.brine == "borehole" and self.borehole_spec.spinup_years > 0:
            self._spin_up()

    def _spin_up(self):
        import copy
        d = self.data
        par = d.parent if d.parent is not None else d
        spec = self.borehole_spec
        n = spec.spinup_years * 8760
        c = self.cfg
        t0, wall, mean_q = [], [], []
        ids = {h.id: h for h in par.houses}
        for i, h in enumerate(d.houses):
            ph = ids.get(h.id, h)
            key = (id(par), par.name, par.T, h.id, d.parent_offset, tuple(sorted(asdict(spec).items())),
                   c.t_supply_dhw_c, c.dhw_speed_frac, c.aux_heater, c.space_heating, self.tanks[i].ua_w_k)
            hit = _SPINUP_CACHE.get(key)
            if hit is None:
                loss = self.tanks[i].ua_w_k * (50.0 - self.tanks[i].t_room_c) / 1000.0
                if c.space_heating:
                    prof = reference_extraction_kw(par, ph, self.hp, t_supply_dhw_c=c.t_supply_dhw_c,
                                                   dhw_speed_frac=c.dhw_speed_frac, store_loss_kw=loss,
                                                   aux=c.aux_heater != "off")
                else:
                    hh = HouseDataNoSH(ph)
                    prof = reference_extraction_kw(par, hh, self.hp, t_supply_dhw_c=c.t_supply_dhw_c,
                                                   dhw_speed_frac=c.dhw_speed_frac, store_loss_kw=loss, aux=False)
                b = self.brine[i]
                b.spin_up(spinup_sequence(prof, d.parent_offset, n))
                if len(_SPINUP_CACHE) >= 64:
                    _SPINUP_CACHE.pop(next(iter(_SPINUP_CACHE)))
                hit = _SPINUP_CACHE[key] = (copy.deepcopy(b), float(prof.mean()), par)
            self.brine[i] = copy.deepcopy(hit[0])
            wall.append(float(self.brine[i].wall_temperature()))
            mean_q.append(hit[1])
        self.spinup_info = {"years": int(spec.spinup_years), "profile_hours": int(par.T),
                            "short_profile": bool(par.T < 8760), "t_ground_c": float(spec.t_ground_c),
                            "t_wall_start_c": wall, "reference_mean_extraction_kw": mean_q,
                            "method": "cyclic reference extraction of the parent series, ending at the run start"}

    @property
    def B(self) -> int:
        return self.data.B

    def _brine_coef(self, b):
        if hasattr(b, "spec") and hasattr(b.spec, "H_m"):
            s = b.spec
            return s.R_b_mk_w / s.H_m * 1000.0 - 1000.0 / (2.0 * s.m_flow_kg_s * s.cp_f_j_kgk)
        return 0.0

    def _full_rate(self, t_wall, coef, t_sup, frac: float = 1.0):
        t_src = np.array(t_wall, dtype=np.float64)
        for _ in range(3):
            q = frac * self.hp.q_max(t_src, t_sup)
            p = self.hp.p_el(q, t_src, t_sup)
            t_src = t_wall - coef * np.maximum(q - p, 0.0)
        q = frac * self.hp.q_max(t_src, t_sup)
        return q, self.hp.p_el(q, t_src, t_sup), t_src

    def context(self) -> Dict[str, np.ndarray]:
        if self._ctx is not None and self._ctx["t"] == self.t:
            return self._ctx
        c = self.cfg
        t = self.t
        d = self.data
        B = self.B
        t_wall = np.array([b.wall_temperature() for b in self.brine])
        coef = np.array([self._brine_coef(b) for b in self.brine])
        r_n, p_n, src_n = self._full_rate(t_wall, coef, np.full(B, c.t_supply_dhw_c), c.dhw_speed_frac)
        fx = c.dhw_speed_frac if c.dis_speed_frac is None else c.dis_speed_frac
        r_x, p_x, src_x = self._full_rate(t_wall, coef, np.full(B, c.t_supply_dis_c), fx)
        t_e, ghi = float(d.t_out_c[t]), float(d.ghi_w_m2[t])
        t_sup_s = np.zeros(B)
        need_set = np.zeros(B)
        need_min = np.zeros(B)
        t_in = np.zeros(B)
        for i, bl in enumerate(self.buildings):
            if bl is None:
                t_sup_s[i] = np.nan
                continue
            ts = float(heating_curve_c(t_e, bl.curve))
            t_sup_s[i] = ts
            t_in[i] = bl.t_in
            if np.isfinite(ts):
                need_set[i] = bl.need_w(bl.t_set, t_e, ghi) / 1000.0
                need_min[i] = bl.need_w(bl.t_min, t_e, ghi) / 1000.0
        sh_on = np.isfinite(t_sup_s)
        ts_eval = np.where(sh_on, t_sup_s, 35.0)
        q_s, p_s, src_s = self._full_rate(t_wall, coef, ts_eval)
        q_s = np.where(sh_on, q_s, 0.0)
        p0, kap = self.hp.chord(src_s, ts_eval)
        f_n = self.hp.flags(src_n, c.t_supply_dhw_c)
        f_x = self.hp.flags(src_x, c.t_supply_dis_c)
        f_s = self.hp.flags(src_s, ts_eval)
        self._ctx = {
            "t": t, "T_nodes": self.T.copy(), "t_wall": t_wall, "brine_coef": coef,
            "r_d": r_n, "p_d": p_n, "t_src_d": src_n, "r_dis": r_x, "p_dis": p_x, "t_src_dis": src_x,
            "t_sup_s": t_sup_s, "sh_on": sh_on, "q_max_s": q_s, "p_s_full": np.where(sh_on, p_s, 0.0),
            "t_src_s": src_s, "chord_p0": np.where(sh_on, p0, 0.0), "chord_kappa": np.where(sh_on, kap, 0.0),
            "sh_need_set_kw": need_set, "sh_need_min_kw": need_min, "t_in": t_in,
            "t_out": t_e, "ghi": ghi, "price": float(d.price[t]), "hour": int(d.hour_of_day[t]),
            "dow": int(d.day_of_week[t]), "t_cold": np.array([h.t_cold_c[t] for h in d.houses]),
            "p_standby": float(self.hp.p_standby_kw), "dhw_speed_frac": float(c.dhw_speed_frac),
            "dis_speed_frac": float(c.dhw_speed_frac if c.dis_speed_frac is None else c.dis_speed_frac),
            "extrap_supply_dis": f_x["extrap_supply"], "extrap_supply_d": f_n["extrap_supply"],
            "extrap_source_d": f_n["extrap_source"],
            "extrap_supply_s": f_s["extrap_supply"] & sh_on, "extrap_source_s": f_s["extrap_source"] & sh_on,
        }
        return self._ctx

    def step(self, u_d=None, u_s=None, dis=None, thermostat: Optional[Dict] = None,
             trace: bool = False, cap_house_kw: Optional[float] = None,
             cap_district_kw: Optional[float] = None) -> Dict[str, np.ndarray]:
        c = self.cfg
        ctx = self.context()
        B = self.B
        u_d = np.zeros(B) if u_d is None else np.clip(np.asarray(u_d, dtype=np.float64), 0.0, 1.0)
        u_s = np.zeros(B) if u_s is None else np.clip(np.asarray(u_s, dtype=np.float64), 0.0, 1.0)
        dis = np.zeros(B, dtype=bool) if dis is None else np.asarray(dis, dtype=bool)
        t_s = np.where(dis, c.t_supply_dis_c, c.t_supply_dhw_c)
        r_d = np.where(dis, ctx["r_dis"], ctx["r_d"])
        p_d = np.where(dis, ctx["p_dis"], ctx["p_d"])
        th = None
        if thermostat is not None:
            th = dict(thermostat)
            th.update(on=self.thermo_on, leg_active=self.leg_active, comp_on=self.comp_on,
                      t_s_leg=c.t_supply_dis_c, r_leg=ctx["r_dis"], p_leg=ctx["p_dis"])
            th.setdefault("leg_target_c", c.t_dis_c)
            th.setdefault("leg_armed", np.zeros(B, dtype=bool))
        sh = None
        if c.space_heating:
            hp = self.hp
            t_sup = np.where(ctx["sh_on"], ctx["t_sup_s"], 35.0)
            t_src_full = ctx["t_src_s"]
            sh = {"E": u_s * ctx["q_max_s"] * 1.0, "q_max": ctx["q_max_s"], "t_wall": ctx["t_wall"],
                  "coef": ctx["brine_coef"], "t_sup": t_sup, "hp": hp,
                  "p_guess": lambda r: hp.p_el(r, t_src_full, t_sup)}
        res = simulate_hour(self.T, self.tank, self.draws[self.t], self.dt_h, u_d=u_d, r_d=r_d,
                            p_d=p_d, t_s=t_s, t_cold=ctx["t_cold"], thermo=th, sh=sh, trace=trace,
                            t_dis_c=c.t_dis_c, t_floor_c=c.t_floor_c, comp_on0=self.comp_on,
                            hold_rule=c.hold_rule, hold_steps=self.hold_steps, series=True)
        self.T = res["T_end"]
        self.comp_on = res["comp_on_end"]
        if th is not None:
            self.thermo_on = res["on_end"]
            self.leg_active = res["leg_active_end"]
        p_sb = ctx["p_standby"] * np.maximum(1.0 - res["time_comp"], 0.0)
        q_aux = np.zeros(B)
        aux_step = np.zeros(B)
        if c.space_heating and c.aux_heater != "off":
            need = ctx["sh_need_min_kw"] if c.aux_heater == "auto_min" else ctx["sh_need_set_kw"]
            deficit = np.where(ctx["sh_on"], need * 1.0 - res["heat_s"], 0.0)
            q_aux = np.clip(np.where(deficit > 1e-6, deficit, 0.0), 0.0, c.aux_steps_kw[-1])
            base = res["elec_d"] + res["elec_s"] + p_sb
            if cap_house_kw is not None:
                q_aux = np.minimum(q_aux, np.maximum(float(cap_house_kw) - base, 0.0))
            if cap_district_kw is not None and q_aux.sum() > 0:
                room = max(float(cap_district_kw) - float(base.sum()), 0.0)
                q_aux = q_aux * min(1.0, room / float(q_aux.sum()))
            steps = np.asarray(c.aux_steps_kw)
            aux_step = np.where(q_aux > 0, steps[np.minimum(np.searchsorted(steps, q_aux - 1e-9), len(steps) - 1)], 0.0)
        elec = res["elec_d"] + res["elec_s"] + p_sb + q_aux
        q_ext = res["heat_d"] + res["heat_s"] - res["elec_d"] - res["elec_s"]
        for i, b in enumerate(self.brine):
            b.advance(float(max(q_ext[i], 0.0)))
        t_in = np.zeros(B)
        for i, bl in enumerate(self.buildings):
            if bl is not None:
                t_in[i] = bl.step((res["heat_s"][i] + q_aux[i]) * 1000.0, ctx["t_out"], ctx["ghi"])
        heat_leg = res.get("heat_leg", np.zeros(B))
        dis_used = (dis & (res["time_d"] > 0)) | (heat_leg > 0)
        rec = {
            "t": self.t, "elec_kwh": elec, "elec_dhw_kwh": res["elec_d"], "elec_sh_kwh": res["elec_s"],
            "elec_standby_kwh": p_sb, "elec_aux_kwh": q_aux, "heat_aux_kwh": q_aux, "aux_step_kw": aux_step,
            "cost": elec * ctx["price"], "heat_dhw_kwh": res["heat_d"],
            "heat_sh_kwh": res["heat_s"], "served_kwh": res["served"], "unmet_kwh": res["unmet"],
            "loss_kwh": res["loss"], "top_min_c": res["top_min"], "top_end_c": res["T_end"][:, 0],
            "nodes_end_min_c": res["T_end"].min(axis=1), "draw_kwh": self.draws[self.t].sum(axis=0),
            "v40_min_l": res["v40_min"], "nodes_min_c": res["nodes_min"], "hold": res["hold"],
            "bot20_min_c": res["bot_min"], "mean_min_c": res["mean_min"],
            "hold_bottom20": res["hold_bot"], "hold_whole_mean": res["hold_whole"], "hold_all_nodes": res["hold_all"],
            "delivered_min_c": np.where(np.isfinite(res["delivered_min"]), res["delivered_min"], np.nan),
            "delivered_below_floor": res["delivered_min"] < c.t_floor_c - 1e-9,
            "dhw_time_h": res["time_d"], "sh_time_h": res["time_s"], "starts": res["starts"],
            "t_src_d": ctx["t_src_d"], "t_wall": ctx["t_wall"], "t_in": t_in,
            "sh_shortfall_kwh": res.get("sh_shortfall", np.zeros(B)), "q_ext_kwh": q_ext,
            "u_d_exec": res["time_d"],
            "u_s_exec": np.where(ctx["q_max_s"] > 0, res["heat_s"] / np.where(ctx["q_max_s"] > 0, ctx["q_max_s"], 1.0), 0.0),
            "dis_supply_used": dis_used,
            "extrap_supply": ((res["time_d"] > 0) & np.where(dis_used, ctx["extrap_supply_dis"], ctx["extrap_supply_d"]))
                             | (ctx["extrap_supply_s"] & (res["time_s"] > 0)),
            "extrap_source": (ctx["extrap_source_d"] & (res["time_d"] > 0)) | (ctx["extrap_source_s"] & (res["time_s"] > 0)),
            "below_floor": res["top_min"] < c.t_floor_c - 1e-9, "time_below_floor_h": res["time_below"],
        }
        rec["series"] = res["series"]
        if th is not None:
            rec.update(leg_started=res["leg_started"], leg_done=res["leg_done"])
        if trace:
            rec["trace_top"] = res["trace_top"]
            rec["trace_nodes"] = res["trace_nodes"]
        self.t += 1
        self._ctx = None
        return rec

    def describe(self) -> Dict[str, object]:
        return {"cfg": asdict(self.cfg), "tank": [asdict(t) for t in self.tanks[:1]],
                "heatpump": self.hp.describe(), "brine": self.brine[0].describe(),
                "borehole_spinup": self.spinup_info,
                "buildings": [h.building.get("source") if h.building else None for h in self.data.houses]}
