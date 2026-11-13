from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from stems.dhw_v2 import tank as tk
from stems.dhw_v2.plant import simulate_hour

__all__ = ["ThermiaLogic", "ThermiaWeekly", "ReactiveController", "PriceAwareController", "NullController",
           "sh_fraction"]


def sh_fraction(ctx, u_d):
    q = np.asarray(ctx["q_max_s"], float)
    need = np.asarray(ctx["sh_need_set_kw"], float)
    u = np.where(q > 0, need / np.where(q > 0, q, 1.0), 0.0)
    return np.clip(np.minimum(u, 1.0 - np.asarray(u_d, float)), 0.0, 1.0)


class ThermiaLogic:
    name = "thermia"

    def __init__(self, B: int, *, start_c: float = 47.0, stop_c: float = 53.0,
                 interval_h: int = 168, target_c: float = 60.0, mode: str = "native"):
        self.B = B
        self.start_c, self.stop_c = start_c, stop_c
        self.interval_h, self.target_c = interval_h, target_c
        self.mode = mode
        self.reset()

    def reset(self):
        self.since = np.full(self.B, float(self.interval_h))
        self.on = np.zeros(self.B, dtype=bool)
        self.leg_active = np.zeros(self.B, dtype=bool)
        self.programs = np.zeros(self.B, dtype=int)

    def thermostat(self, armed):
        return {"start_c": self.start_c, "stop_c": self.stop_c, "leg_armed": armed,
                "leg_target_c": self.target_c}

    def act(self, env) -> Dict:
        ctx = env.plant.context()
        armed = self.since >= self.interval_h
        if self.mode == "native":
            return {"thermostat": self.thermostat(armed), "u_s": sh_fraction(ctx, np.zeros(self.B))}
        pl = env.plant
        d_hat = env.draw_forecast()
        draws = env.cert_shape()[:, None] * d_hat[None, :]
        th = dict(self.thermostat(armed))
        th.update(on=self.on, leg_active=self.leg_active, t_s_leg=pl.cfg.t_supply_dis_c,
                  r_leg=ctx["r_dis"], p_leg=ctx["p_dis"])
        r = simulate_hour(pl.T, pl.tank, draws, pl.dt_h, u_d=0.0, r_d=ctx["r_d"], p_d=ctx["p_d"],
                          t_s=pl.cfg.t_supply_dhw_c, t_cold=ctx["t_cold"], thermo=th,
                          t_dis_c=pl.cfg.t_dis_c)
        u_d = r["time_d"]
        dis = r["leg_started"] | self.leg_active
        self.on = r["on_end"]
        self._pending = (r["leg_started"], r["leg_done"], r["leg_active_end"])
        return {"u_d": u_d, "u_s": sh_fraction(ctx, u_d), "dis": dis}

    def update(self, rec: Dict) -> None:
        if self.mode == "native":
            done = np.asarray(rec.get("leg_done", np.zeros(self.B, bool)), bool)
            started = np.asarray(rec.get("leg_started", np.zeros(self.B, bool)), bool)
        else:
            started = self._pending[0]
            self.leg_active = self.leg_active | started
            done = self.leg_active & (np.asarray(rec["nodes_end_min_c"]) >= self.target_c - 1e-9)
            self.leg_active = self.leg_active & ~done
            self.on = self.on & ~done
        self.programs += started.astype(int)
        self.since = np.where(done, 0.0, self.since + 1.0)


class ReactiveController:
    name = "reactive"

    def __init__(self, B: int, *, on_bottom_c=40.0, on_top_c=47.0, off_bottom_c=50.0,
                 timer_dow: Optional[int] = 6, timer_hour: int = 2):
        self.B = B
        self.on_bottom_c, self.on_top_c, self.off_bottom_c = on_bottom_c, on_top_c, off_bottom_c
        self.timer_dow, self.timer_hour = timer_dow, timer_hour
        self.reset()

    def reset(self):
        self.on = np.zeros(self.B, dtype=bool)
        self.owed = np.zeros(self.B, dtype=bool)

    def act(self, env) -> Dict:
        ctx = env.plant.context()
        T = env.plant.T
        T_top, T_bot = T[:, 0], T[:, -1]
        self.on = np.where(self.on, T_bot < self.off_bottom_c,
                           (T_bot < self.on_bottom_c) | (T_top < self.on_top_c))
        daily = getattr(env, "period_h", 168) == 24
        if self.timer_dow is not None and (daily or ctx["dow"] == self.timer_dow) and ctx["hour"] == self.timer_hour:
            self.owed |= ~env.week_done
        self.owed &= ~env.week_done
        u_d = np.where(self.on | self.owed, 1.0, 0.0)
        return {"u_d": u_d, "u_s": sh_fraction(ctx, u_d), "dis": self.owed.copy()}

    def update(self, rec: Dict) -> None:
        self.owed &= ~np.asarray(rec["hold"], bool)


class ThermiaWeekly(ReactiveController):
    name = "thermia_weekly"

    def __init__(self, B: int, *, dow: int = 6, hour: int = 2, target_c: float = 60.0, limit_h: int = 7,
                 end_sensor: str = "top", weights=(0.5, 0.5), **kw):
        if end_sensor not in ("top", "middle", "weighted"):
            raise ValueError("end_sensor must be 'top', 'middle' or 'weighted'")
        self.dow, self.hour, self.target_c, self.limit_h = int(dow), int(hour), float(target_c), int(limit_h)
        self.end_sensor, self.weights = end_sensor, tuple(float(w) for w in weights)
        super().__init__(B, timer_dow=None, **kw)

    def reset(self):
        super().reset()
        self.prog = np.zeros(self.B, dtype=bool)
        self.prog_h = np.zeros(self.B, dtype=int)
        self.programs = np.zeros(self.B, dtype=int)
        self.done = np.zeros(self.B, dtype=int)
        self.alarms = np.zeros(self.B, dtype=int)

    def sensor(self, env):
        T = env.plant.T
        top, mid = T[:, 0], T[:, int(env.plant.tank["sidx"])]
        if self.end_sensor == "top":
            return top
        if self.end_sensor == "middle":
            return mid
        return self.weights[0] * top + self.weights[1] * mid

    def act(self, env) -> Dict:
        ctx = env.plant.context()
        T = env.plant.T
        T_top, T_bot = T[:, 0], T[:, -1]
        self.on = np.where(self.on, T_bot < self.off_bottom_c,
                           (T_bot < self.on_bottom_c) | (T_top < self.on_top_c))
        if ctx["dow"] == self.dow and ctx["hour"] == self.hour:
            start = ~self.prog
            self.programs += start
            self.prog |= start
            self.prog_h = np.where(start, 0, self.prog_h)
        reached = self.prog & (self.sensor(env) >= self.target_c - 1e-9)
        self.done += reached
        self.prog &= ~reached
        late = self.prog & (self.prog_h >= self.limit_h)
        self.alarms += late
        self.prog &= ~late
        self.prog_h = np.where(self.prog, self.prog_h + 1, self.prog_h)
        cfg = env.plant.cfg
        t_s = np.where(self.prog, cfg.t_supply_dis_c, cfg.t_supply_dhw_c)
        rate = np.where(self.prog, ctx["r_dis"], ctx["r_d"])
        need = tk.accept_max(T, env.plant.tank["node_vol"], t_s) / np.maximum(rate, 1e-9)
        u_d = np.where(self.on | self.prog, np.clip(need, 0.0, 1.0), 0.0)
        return {"u_d": u_d, "u_s": sh_fraction(ctx, u_d), "dis": self.prog.copy()}

    def update(self, rec: Dict) -> None:
        pass


class PriceAwareController:
    name = "price_aware"

    def __init__(self, B: int, *, cheap_margin=0.05, request_after_h=120.0,
                 cheap=(50.0, 47.0, 53.0), dear=(30.0, 47.0, 40.0)):
        self.B = B
        self.cheap_margin, self.request_after_h = float(cheap_margin), float(request_after_h)
        self.cheap_thr, self.dear_thr = cheap, dear
        self.reset()

    def reset(self):
        self.on = np.zeros(self.B, dtype=bool)

    def act(self, env) -> Dict:
        ctx = env.plant.context()
        T = env.plant.T
        price = env.price_path(24)
        cop = np.asarray(ctx["r_d"]) / np.asarray(ctx["p_d"])
        c = price[:, None] / cop[None, :]
        cheap = c[0] <= (1.0 - self.cheap_margin) * c.mean(axis=0)
        on_b = np.where(cheap, self.cheap_thr[0], self.dear_thr[0])
        on_t = np.where(cheap, self.cheap_thr[1], self.dear_thr[1])
        off_b = np.where(cheap, self.cheap_thr[2], self.dear_thr[2])
        T_top, T_bot = T[:, 0], T[:, -1]
        self.on = np.where(self.on, T_bot < off_b, (T_bot < on_b) | (T_top < on_t))
        P = getattr(env, "period_h", 168)
        elapsed = P - env.hours_left_week()
        dis = (~env.week_done) & (elapsed >= self.request_after_h * P / 168.0) & cheap
        u_d = np.where(self.on | dis, 1.0, 0.0)
        return {"u_d": u_d, "u_s": sh_fraction(ctx, u_d), "dis": dis}

    def update(self, rec: Dict) -> None:
        pass


class NullController:
    name = "null"

    def __init__(self, B: int):
        self.B = B

    def reset(self):
        pass

    def act(self, env) -> Dict:
        ctx = env.plant.context()
        z = np.zeros(self.B)
        return {"u_d": z, "u_s": sh_fraction(ctx, z), "dis": np.zeros(self.B, bool)}

    def update(self, rec: Dict) -> None:
        pass
