from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np

from stems.dhw_v2 import tank as tk
from stems.dhw_v2.data import DistrictData, intra_hour_weights
from stems.dhw_v2.filter import DrawForecaster, FilterConfig, FilterModel, JointFilter, MarginTracker
from stems.dhw_v2.hygiene import HygieneScorer, ThermiaRuleScorer, period_calendar
from stems.dhw_v2.identify import IdentifiedMap
from stems.dhw_v2.plant import DistrictPlant, PlantConfig, simulate_hour, tank_arrays

__all__ = ["EnvConfig", "DHWDistrictEnv", "OBS_NAMES"]


def _rf_failures(A):
    if "f_infeasible_c" not in A:
        return None
    stuck = ((A["f_infeasible_c"][1:] > 0) | (A["f_s_c"][1:] > 1e-9)) & (A["inherited"][1:] == 0)
    prev_ok = (A["delta"][:-1] <= A["m"][:-1] + 1e-12) & (A["declared_infeasible"][:-1] == 0)
    return (stuck & prev_ok).sum(axis=0).astype(float).tolist()


def _max_drop(x, H):
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(x.shape[1])
    for k in range(1, H + 1):
        out = np.maximum(out, (x[:-k] - x[k:]).max(axis=0)) if len(x) > k else out
    return out.tolist()

OBS_NAMES = [
    ("hour_sin", "sin(2 pi hour/24), local time"), ("hour_cos", "cos(2 pi hour/24)"),
    ("dow_sin", "sin(2 pi day_of_week/7)"), ("dow_cos", "cos(2 pi day_of_week/7)"),
    ("t_out", "outdoor temperature / 10 degC"), ("ghi", "GHI / 1000 W/m2"),
    ("price_0", "price now / mean price of the data"),
    ("price_1", "price t+1 / mean (day-ahead, known)"), ("price_2", "price t+2 / mean"),
    ("price_3", "price t+3 / mean"),
    ("t_top", "top node / 60 degC"), ("t_sensor", "tank sensor node / 60 degC"),
    ("t_bottom", "bottom node / 60 degC"), ("v40", "V40 mixed 40 degC litres / 300"),
    ("t_brine", "brine inlet at full DHW speed / 10 degC"), ("t_in", "indoor - 21 K"),
    ("sh_need", "heat to reach set point / q_max_s (0 when no SH)"),
    ("q_max_s", "full SH capacity / 12 kW"), ("d_hat", "causal draw forecast this hour, kWh"),
    ("margin", "calibrated margin m, kWh (0 when the filter is off)"),
    ("week_left", "hours left in the calendar week / 168"),
    ("week_done", "1 if this week's disinfection hold is complete"),
    ("since_hold", "hours since last completed hold / 168 (capped at 2)"),
    ("comp_last", "compressor run time last hour, h"),
]


@dataclass
class EnvConfig:
    plant: PlantConfig = field(default_factory=PlantConfig)
    tank: str = "rxt12"
    tank_nodes: int = 10
    tank_calibration: str = "pes_minus_psb_cop"
    sensor_height: float = 0.6
    filter: Optional[FilterConfig] = None
    filter_model: str = "oracle"
    identified: Optional[Dict] = None
    w_comfort: float = 1.0
    w_sh: float = 0.5
    w_dis: float = 5.0
    k_correction: float = 1.0
    hours: Optional[int] = None
    compute_dtol: bool = True
    keep_log: bool = True
    period_h: int = 168
    hygiene: bool = True
    regime: Optional[str] = None
    thermia_sensor_weights: tuple = (0.5, 0.5)


class DHWDistrictEnv:
    def __init__(self, data: DistrictData, cfg: Optional[EnvConfig] = None):
        self.cfg = cfg or EnvConfig()
        self.data = data if self.cfg.hours is None else data.subset(hours=self.cfg.hours)
        self.agents = [h.id for h in self.data.houses]
        self.B = self.data.B
        self.observation_names = [n for n, _ in OBS_NAMES]
        self.action_low = np.zeros(3, np.float32)
        self.action_high = np.ones(3, np.float32)
        self.week_id, self.week_left = period_calendar(self.data.hour_of_day, self.data.day_of_week,
                                                        self.cfg.period_h)
        self._tanks = self._make_tanks()
        self.reset()

    def _make_tanks(self):
        c = self.cfg
        if c.tank == "rxt12":
            return [tk.rxt12_tank(c.tank_nodes, c.tank_calibration, sensor_height=c.sensor_height)] * self.B
        if c.tank == "citylearn":
            out = []
            for h in self.data.houses:
                if not h.tank_alt:
                    raise ValueError(f"house {h.id} has no CityLearn tank (tank_alt)")
                out.append(tk.citylearn_tank(h.tank_alt["capacity_kwh"], h.tank_alt["loss_coefficient"],
                                             c.tank_nodes, sensor_height=c.sensor_height))
            return out
        raise ValueError(c.tank)

    def _make_filter(self):
        c = self.cfg
        if c.filter is None:
            return None
        if c.filter_model == "oracle":
            model = FilterModel(self.plant.tank, self.plant.hp, kind="oracle")
        elif c.filter_model == "identified":
            idf = c.identified
            if idf is None:
                raise ValueError("filter_model='identified' needs cfg.identified")
            n = int(idf.get("nodes", 2))
            specs = [tk.TankSpec2(volume_l=t.volume_l, nodes=n, ua_w_k=float(idf["ua_w_k"]),
                                  t_room_c=t.t_room_c, t_use_c=t.t_use_c, name="identified")
                     for t in self._tanks]
            model = FilterModel(tank_arrays(specs), IdentifiedMap(idf["g"], idf["h"]),
                                kind="identified", meta={k: idf[k] for k in idf if k not in ("g", "h")}
                                | {"g": list(idf["g"]), "h": list(idf["h"])})
        else:
            raise ValueError(c.filter_model)
        from dataclasses import replace as _rep
        fcfg = _rep(c.filter, hold_rule=c.plant.hold_rule, t_dis_c=c.plant.t_dis_c, floor_c=c.plant.t_floor_c,
                    hold_min=c.plant.hold_min, t_supply_dhw_c=c.plant.t_supply_dhw_c,
                    t_supply_dis_c=c.plant.t_supply_dis_c,
                    disinfection=c.filter.disinfection and c.hygiene)
        return JointFilter(model, fcfg, self.B, self.plant.S, self.plant.dt_h,
                           self.week_id, self.week_left)

    def reset(self, seed: Optional[int] = None):
        self.plant = DistrictPlant(self.data, tanks=self._tanks, cfg=self.cfg.plant)
        self.filter = self._make_filter()
        a = self.cfg.filter.alpha if self.cfg.filter else 0.05
        e = self.cfg.filter.eta_kwh if self.cfg.filter else 0.05
        self.shadow_fc = DrawForecaster(self.B)
        self.shadow_m = MarginTracker(self.B, a, e)
        self.week_done = np.zeros(self.B, dtype=bool) if self.cfg.hygiene else np.ones(self.B, dtype=bool)
        self.hygiene_score = HygieneScorer(self.B, self.cfg.plant.dt_min, self.data.hour_of_day,
                                           self.data.day_of_week)
        self.thermia_score = ThermiaRuleScorer(self.B, self.cfg.plant.dt_min,
                                               weights=self.cfg.thermia_sensor_weights,
                                               sensor_height=self.cfg.sensor_height)
        self.week_done_alt = {k: np.zeros(self.B, dtype=bool) for k in ("hold_whole_mean", "hold_all_nodes", "hold_bottom20")}
        self.weeks_completed_alt = {k: np.zeros(self.B, dtype=int) for k in self.week_done_alt}
        self._prev_finfo = None
        self.weeks_full = 0
        self.weeks_completed = np.zeros(self.B, dtype=int)
        self.last_hold = np.full(self.B, -1)
        self.max_gap = np.zeros(self.B)
        self.comp_last = np.zeros(self.B)
        self.log: Dict[str, List[np.ndarray]] = {}
        self.t0_wall = time.time()
        self.cur_week = 0
        self.week_hours = 0
        self.mean_price = float(np.mean(self.data.price)) or 1.0
        return self.observe(), {"agents": self.agents}

    @property
    def t(self) -> int:
        return self.plant.t

    def draw_forecast(self) -> np.ndarray:
        if self.filter is not None and self.cfg.filter.forecast == "hod_mean":
            return self.filter.forecaster.forecast(int(self.data.hour_of_day[self.t]))
        return self.shadow_fc.forecast(int(self.data.hour_of_day[self.t]))

    def cert_shape(self) -> np.ndarray:
        shape = self.cfg.filter.cert_shape if self.cfg.filter else self.cfg.plant.profile
        return intra_hour_weights(self.plant.S, shape)

    def price_path(self, H: int) -> np.ndarray:
        p = self.data.price
        idx = np.minimum(np.arange(self.t, self.t + H), len(p) - 1)
        return p[idx]

    def hours_left_week(self) -> int:
        return int(self.week_left[self.t])

    @property
    def period_h(self) -> int:
        return int(self.cfg.period_h)

    def observe(self) -> np.ndarray:
        t = min(self.t, self.data.T - 1)
        ctx = self.plant.context() if self.t < self.data.T else None
        B = self.B
        hod, dow = self.data.hour_of_day[t], self.data.day_of_week[t]
        T = self.plant.T
        h = self.plant.tank["node_vol"]
        tc = np.array([hh.t_cold_c[t] for hh in self.data.houses])
        pp = self.price_path(4) / self.mean_price
        q_s = ctx["q_max_s"] if ctx is not None else np.zeros(B)
        need = ctx["sh_need_set_kw"] if ctx is not None else np.zeros(B)
        m = self.filter.margin.m if self.filter is not None else np.zeros(B)
        since = np.where(self.last_hold >= 0, self.t - self.last_hold, self.t + 168)
        cols = [np.full(B, np.sin(2 * np.pi * hod / 24)), np.full(B, np.cos(2 * np.pi * hod / 24)),
                np.full(B, np.sin(2 * np.pi * dow / 7)), np.full(B, np.cos(2 * np.pi * dow / 7)),
                np.full(B, self.data.t_out_c[t] / 10.0), np.full(B, self.data.ghi_w_m2[t] / 1000.0),
                np.full(B, pp[0]), np.full(B, pp[1]), np.full(B, pp[2]), np.full(B, pp[3]),
                T[:, 0] / 60.0, T[:, self.plant.tank["sidx"]] / 60.0, T[:, -1] / 60.0,
                tk.v40_l(T, h, tc) / 300.0,
                (ctx["t_src_d"] if ctx is not None else np.zeros(B)) / 10.0,
                (ctx["t_in"] if ctx is not None else np.zeros(B)) - 21.0,
                np.where(q_s > 0, need / np.where(q_s > 0, q_s, 1.0), 0.0), q_s / 12.0,
                self.draw_forecast(), m, np.full(B, self.week_left[t] / 168.0),
                self.week_done.astype(float), np.minimum(since / 168.0, 2.0), self.comp_last]
        return np.stack(cols, axis=1).astype(np.float32)

    def step(self, action):
        if isinstance(action, dict):
            action = np.stack([np.asarray(action[a], float) for a in self.agents])
        a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1.0).reshape(self.B, 3)
        return self.step_command({"u_d": a[:, 0], "u_s": a[:, 1], "dis": a[:, 2] >= 0.5})

    def _d_tol(self, T_pre, ctx, u_d, dis, draws, mask):
        pl = self.plant
        idx = np.flatnonzero(mask)
        K = 33
        floor = pl.cfg.t_floor_c
        t_s = np.where(dis, pl.cfg.t_supply_dis_c, pl.cfg.t_supply_dhw_c)[idx]
        r_d = np.where(dis, ctx["r_dis"], ctx["r_d"])[idx]
        p_d = np.where(dis, ctx["p_dis"], ctx["p_d"])[idx]
        tank = {k: (v[idx] if isinstance(v, np.ndarray) else v) for k, v in pl.tank.items()}
        tank = {k: (v[:, None] if isinstance(v, np.ndarray) else v) for k, v in tank.items()}
        dr = draws[:, idx]
        tc = ctx["t_cold"][idx]

        def ok(lam):
            r = simulate_hour(T_pre[idx][:, None, :], tank, dr[:, :, None] * lam[None], pl.dt_h,
                              u_d=u_d[idx][:, None], r_d=r_d[:, None], p_d=p_d[:, None],
                              t_s=t_s[:, None], t_cold=tc[:, None], t_floor_c=floor)
            return r["top_min"] >= floor - 1e-9

        g = np.broadcast_to(np.linspace(0.0, 1.0, K), (len(idx), K))
        good = ok(g)
        j = np.where(good.all(axis=1), K - 1, np.argmin(good, axis=1) - 1)
        lo = np.where(j >= 0, g[0][np.maximum(j, 0)], 0.0)
        hi = g[0][np.minimum(j + 1, K - 1)]
        g2 = lo[:, None] + (hi - lo)[:, None] * np.linspace(0.0, 1.0, K)[None, :]
        good2 = ok(g2)
        j2 = np.where(good2.all(axis=1), K - 1, np.argmin(good2, axis=1) - 1)
        lam = np.where(j2 >= 0, g2[np.arange(len(idx)), np.maximum(j2, 0)], 0.0)
        lam = np.where(good[:, 0], lam, 0.0)
        out = np.full(self.B, np.nan)
        out[idx] = lam * draws[:, idx].sum(axis=0)
        return out

    def step_command(self, cmd: Dict):
        c = self.cfg
        pl = self.plant
        t = pl.t
        B = self.B
        ctx = pl.context()
        T_pre = pl.T.copy()
        draws = pl.draws[t]
        D = draws.sum(axis=0)
        hour = int(self.data.hour_of_day[t])
        thermostat = cmd.get("thermostat")
        prop = {"u_d": np.asarray(cmd.get("u_d", np.zeros(B)), float),
                "u_s": np.asarray(cmd.get("u_s", np.zeros(B)), float),
                "dis": np.asarray(cmd.get("dis", np.zeros(B, bool)), bool)}
        finfo = None
        if self.filter is not None:
            if thermostat is not None:
                raise ValueError("native thermostat commands are unfiltered; use mode='hourly'")
            exe, finfo = self.filter.project(t, ctx, prop, draw_true=D)
        else:
            exe = prop
        fc = c.filter if self.filter is not None else None
        rec = pl.step(u_d=exe["u_d"], u_s=exe["u_s"], dis=exe["dis"], thermostat=thermostat,
                      cap_house_kw=fc.cap_house_kw if fc else None,
                      cap_district_kw=fc.cap_district_kw if fc else None)
        if self.filter is not None:
            ob = self.filter.observe(t, hour, D, rec)
        else:
            d_hat = self.shadow_fc.forecast(hour)
            ob = {"delta": D - d_hat, "m": self.shadow_m.m.copy()}
            ob["bound"] = np.maximum(ob["delta"] - ob["m"], 0.0)
            self.shadow_m.update(ob["delta"])
            self.shadow_fc.update(hour, D)
        viol = np.asarray(rec["below_floor"], bool)
        v = np.zeros(B)
        if self.filter is not None and c.compute_dtol and viol.any():
            dtol = self._d_tol(T_pre, ctx, np.asarray(exe["u_d"], float), np.asarray(exe["dis"], bool), draws, viol)
            v = np.where(viol, np.maximum(D - dtol, 0.0), 0.0)
        infeasible = np.zeros(B, bool) if finfo is None else (finfo["infeasible_c"] | (finfo["s_c"] > 1e-9))
        explained_fc = ob["delta"] > ob["m"] + 1e-12
        pf = self._prev_finfo
        if pf is not None and "D_next_cert" in pf:
            explained_prev = D > pf["D_next_cert"] + 1e-12
            announced = pf["reserve_infeasible"] | pf["infeasible_c"] | (pf["s_c"] > 1e-9)
        else:
            explained_prev = np.zeros(B, bool)
            announced = np.zeros(B, bool)
        inherited = T_pre[:, 0] < pl.cfg.t_floor_c - 1e-9
        unexplained = viol & ~explained_fc & ~explained_prev & ~announced & ~infeasible & ~inherited
        self._prev_finfo = finfo
        self.hygiene_score.update(t, rec)
        self.thermia_score.update(t, rec)
        hold = np.asarray(rec["hold"], bool)
        first = hold & ~self.week_done
        gap_now = np.where(self.last_hold >= 0, t - self.last_hold, t + 1)
        self.max_gap = np.where(first, np.maximum(self.max_gap, gap_now), self.max_gap)
        self.last_hold = np.where(hold, t, self.last_hold)
        self.week_done |= hold
        for k in self.week_done_alt:
            self.week_done_alt[k] |= np.asarray(rec[k], bool)
        self.week_hours += 1
        missed = np.zeros(B)
        end_week = (t + 1 >= self.data.T) or (self.week_id[t + 1] != self.week_id[t])
        if end_week:
            if self.week_hours >= c.period_h and c.hygiene:
                self.weeks_full += 1
                self.weeks_completed += self.week_done.astype(int)
                for k in self.week_done_alt:
                    self.weeks_completed_alt[k] += self.week_done_alt[k].astype(int)
                missed = (~self.week_done).astype(float)
            self.week_done = np.zeros(B, bool) if c.hygiene else np.ones(B, bool)
            for k in self.week_done_alt:
                self.week_done_alt[k] = np.zeros(B, bool)
            self.week_hours = 0
        t_min = np.array([getattr(b, "t_min", 20.0) if b is not None else 20.0 for b in pl.buildings])
        sh_dev = np.maximum(t_min - rec["t_in"], 0.0) if pl.cfg.space_heating else np.zeros(B)
        corr = finfo["delta_el_kwh"] if finfo is not None else np.zeros(B)
        price = ctx["price"]
        rew = -(rec["cost"] + c.w_comfort * rec["time_below_floor_h"] + c.w_sh * sh_dev
                + c.w_dis * missed + c.k_correction * price * corr)
        self.comp_last = rec["dhw_time_h"] + rec["sh_time_h"]
        info = {"t": t, "record": rec, "proposed": prop, "executed": exe, "filter": finfo,
                "delta": ob["delta"], "m": ob["m"], "bound": ob["bound"], "violation": viol, "v": v,
                "declared_infeasible": infeasible, "unexplained": unexplained, "inherited": inherited,
                "announced_prev": announced, "explained_prev": explained_prev, "missed_week": missed,
                "reward": rew}
        if c.keep_log:
            self._append(info, rec, finfo, sh_dev)
        term = pl.t >= self.data.T
        obs = self.observe() if not term else np.zeros((B, len(OBS_NAMES)), np.float32)
        return obs, rew, term, False, info

    def _append(self, info, rec, finfo, sh_dev):
        L = self.log
        put = lambda k, x: L.setdefault(k, []).append(np.asarray(x, dtype=np.float32))
        for k in ("elec_kwh", "elec_dhw_kwh", "elec_sh_kwh", "elec_standby_kwh", "elec_aux_kwh", "aux_step_kw",
                  "cost", "heat_dhw_kwh",
                  "heat_sh_kwh", "served_kwh", "unmet_kwh", "loss_kwh", "top_min_c", "v40_min_l",
                  "nodes_min_c", "hold", "dhw_time_h", "sh_time_h", "starts", "t_src_d", "t_wall", "t_in",
                  "sh_shortfall_kwh", "dis_supply_used", "extrap_supply", "extrap_source", "below_floor",
                  "time_below_floor_h", "draw_kwh", "hold_bottom20", "hold_whole_mean", "hold_all_nodes",
                  "bot20_min_c", "delivered_below_floor"):
            put(k, rec[k])
        if "leg_started" in rec:
            put("leg_started", rec["leg_started"])
            put("leg_done", rec["leg_done"])
        put("sh_dev_kh", sh_dev)
        for k in ("delta", "m", "bound", "v", "declared_infeasible", "unexplained", "inherited", "reward",
                  "announced_prev", "explained_prev"):
            put(k, info[k])
        put("u_d_prop", info["proposed"]["u_d"])
        put("u_s_prop", info["proposed"]["u_s"])
        put("u_d", info["executed"]["u_d"])
        put("u_s", info["executed"]["u_s"])
        put("dis", info["executed"]["dis"])
        if finfo is not None:
            for k in ("intervened", "l_c", "l_dis", "l_s", "s_c", "s_dis", "s_sh", "infeasible_c",
                      "infeasible_hold", "infeasible_s", "reserve_infeasible", "power_limited", "trigger_infeasible", "D_cert", "d_hat", "must", "delta_el_kwh",
                      "elec_bound_kw"):
                put("f_" + k, finfo[k])

    def arrays(self) -> Dict[str, np.ndarray]:
        return {k: np.stack(v) for k, v in self.log.items()}

    def summary(self) -> Dict[str, object]:
        A = self.arrays()
        Tn = A["elec_kwh"].shape[0]
        s = lambda k: A[k].sum(axis=0).astype(float).tolist()
        out = {
            "hours": int(Tn), "houses": self.agents, "wall_s": time.time() - self.t0_wall,
            "elec_kwh": s("elec_kwh"), "elec_dhw_kwh": s("elec_dhw_kwh"), "elec_sh_kwh": s("elec_sh_kwh"),
            "elec_standby_kwh": s("elec_standby_kwh"), "elec_aux_kwh": s("elec_aux_kwh"),
            "aux_hours": (A["elec_aux_kwh"] > 0).sum(axis=0).astype(float).tolist(),
            "aux_step_max_kw": A["aux_step_kw"].max(axis=0).astype(float).tolist(),
            "aux_heater": self.cfg.plant.aux_heater, "borehole_spinup": self.plant.spinup_info,
            "cost": s("cost"), "heat_dhw_kwh": s("heat_dhw_kwh"),
            "heat_sh_kwh": s("heat_sh_kwh"), "served_kwh": s("served_kwh"), "unmet_kwh": s("unmet_kwh"),
            "loss_kwh": s("loss_kwh"), "draw_kwh": s("draw_kwh"),
            "hours_below_floor": s("below_floor"), "time_below_floor_h": s("time_below_floor_h"),
            "top_min_min_c": A["top_min_c"].min(axis=0).astype(float).tolist(),
            "v40_min_min_l": A["v40_min_l"].min(axis=0).astype(float).tolist(),
            "weeks_full": int(self.weeks_full), "weeks_completed": self.weeks_completed.tolist(),
            "hold_rule": self.cfg.plant.hold_rule,
            "regime": {"name": self.cfg.regime, "period_h": self.cfg.period_h, "hygiene": self.cfg.hygiene,
                       "t_dis_c": self.cfg.plant.t_dis_c, "t_supply_dis_c": self.cfg.plant.t_supply_dis_c,
                       "hold_rule": self.cfg.plant.hold_rule, "hold_min": self.cfg.plant.hold_min,
                       "note": "weeks_* count the regime's periods (days when period_h = 24)"},
            "hygiene": self.hygiene_score.summary(), "thermia_rule": self.thermia_score.summary(),
            "weeks_completed_bottom20": self.weeks_completed_alt["hold_bottom20"].tolist(),
            "weeks_completed_whole_mean": self.weeks_completed_alt["hold_whole_mean"].tolist(),
            "weeks_completed_all_nodes": self.weeks_completed_alt["hold_all_nodes"].tolist(),
            "hours_delivered_below_floor": s("delivered_below_floor"),
            "max_gap_between_holds_h": self.max_gap.tolist(),
            "hold_hours": s("hold"), "dis_supply_hours": s("dis_supply_used"),
            "elec_dhw_dis_kwh": (A["elec_dhw_kwh"] * A["dis_supply_used"]).sum(axis=0).astype(float).tolist(),
            "hours_extrap_supply": s("extrap_supply"), "hours_extrap_source": s("extrap_source"),
            "elec_at_extrap_supply_kwh": (A["elec_kwh"] * A["extrap_supply"]).sum(axis=0).astype(float).tolist(),
            "sh_degree_hours_below_min": s("sh_dev_kh"), "sh_shortfall_kwh": s("sh_shortfall_kwh"),
            "starts": s("starts"), "t_src_d_min_c": A["t_src_d"].min(axis=0).astype(float).tolist(),
            "t_src_d_mean_c": A["t_src_d"].mean(axis=0).astype(float).tolist(),
            "t_in_min_c": A["t_in"].min(axis=0).astype(float).tolist(),
            "violation_hours": s("below_floor"),
            "brine_max_24h_drop_k": _max_drop(A["t_src_d"], 24),
            "violations_inherited": ((A["below_floor"] > 0) & (A["inherited"] > 0)).sum(axis=0).astype(float).tolist(),
            "recursive_feasibility_failures": _rf_failures(A),
            "violations_new": ((A["below_floor"] > 0) & (A["inherited"] == 0)).sum(axis=0).astype(float).tolist(),
            "exceed_freq_delta_gt_m": (A["delta"] > A["m"] + 1e-12).mean(axis=0).astype(float).tolist(),
            "reward": s("reward"),
        }
        if "leg_started" in A:
            out["thermia_programs_started"] = s("leg_started")
            out["thermia_programs_completed"] = s("leg_done")
        if "f_intervened" in A:
            viol = A["below_floor"] > 0
            inf = A["declared_infeasible"] > 0
            expl = (A["delta"] > A["m"] + 1e-12)
            chk = viol & ~inf
            holds = (A["v"] <= A["bound"] + 1e-6)
            out.update({
                "interventions": s("f_intervened"),
                "intervention_rate": A["f_intervened"].mean(axis=0).astype(float).tolist(),
                "mean_abs_correction_el_kwh": A["f_delta_el_kwh"].mean(axis=0).astype(float).tolist(),
                "slack_comfort_sum": s("f_s_c"), "slack_dis_sum": s("f_s_dis"), "slack_sh_sum": s("f_s_sh"),
                "declared_infeasible_comfort_hours": s("f_infeasible_c"),
                "declared_infeasible_hold_hours": s("f_infeasible_hold"),
                "declared_infeasible_sh_hours": s("f_infeasible_s"),
                "violations_explained_by_forecast": (viol & expl).sum(axis=0).astype(float).tolist(),
                "violations_declared_infeasible": (viol & inf).sum(axis=0).astype(float).tolist(),
                "violations_unexplained": s("unexplained"),
                "violations_announced_prev_hour": ((A["below_floor"] > 0) & (A["announced_prev"] > 0)).sum(axis=0).astype(float).tolist(),
                "violations_explained_by_plan_exceedance": ((A["below_floor"] > 0) & (A["explained_prev"] > 0)).sum(axis=0).astype(float).tolist(),
                "bound_checked_steps": chk.sum(axis=0).astype(float).tolist(),
                "bound_holds_steps": (chk & holds).sum(axis=0).astype(float).tolist(),
                "max_bound_excess_kwh": np.where(chk, A["v"] - A["bound"], -np.inf).max(axis=0).astype(float).tolist(),
                "margin_final_kwh": A["m"][-1].astype(float).tolist(),
                "margin_mean_kwh": A["m"].mean(axis=0).astype(float).tolist(),
                "monotonicity_failures": int(self.filter.monotonicity_failures),
                "jit_forced_hours": s("f_must"),
                "reserve_infeasible_hours": s("f_reserve_infeasible"),
                "power_limited_hours": s("f_power_limited"),
                "trigger_infeasible_hours": s("f_trigger_infeasible"),
                "violations_power_limited": ((A["below_floor"] > 0) & (A["f_power_limited"] > 0)).sum(axis=0).astype(float).tolist(),
                "elec_bound_max_kw": A["f_elec_bound_kw"].max(axis=0).astype(float).tolist(),
                "elec_max_kwh_per_h": A["elec_kwh"].max(axis=0).astype(float).tolist(),
            })
            fc = self.cfg.filter
            if fc.cap_house_kw is not None:
                out["cap_house_kw"] = fc.cap_house_kw
                out["hours_over_house_cap"] = (A["elec_kwh"] > fc.cap_house_kw + 1e-9).sum(axis=0).astype(float).tolist()
            if fc.cap_district_kw is not None:
                out["cap_district_kw"] = fc.cap_district_kw
                out["hours_over_district_cap"] = int((A["elec_kwh"].sum(axis=1) > fc.cap_district_kw + 1e-9).sum())
            if self.filter is not None:
                out["coverage_bound"] = self.filter.margin.coverage_bound().astype(float).tolist()
                out["exceed_freq_upper_bound"] = self.filter.margin.upper_bound().astype(float).tolist()
                out["filter"] = self.filter.describe()
        return out

    def spec(self) -> Dict[str, object]:
        return {"agents": self.agents, "obs": OBS_NAMES, "action": ["u_d", "u_s", "u_dis"],
                "action_low": self.action_low.tolist(), "action_high": self.action_high.tolist(),
                "plant": self.plant.describe(), "env": {k: v for k, v in asdict(self.cfg).items()
                                                        if k not in ("identified",)}}
