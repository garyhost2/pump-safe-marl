from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stems.dhw_barrier import (FLOOR_UNREACHABLE, PHASE_DISINFECT, PHASE_NAMES,
                               PHASE_NORMAL, DHWStoreModel, PhaseBands, ProjectionResult,
                               StoreStep, _col)
from stems.dhw_physics import charge_step, loss_and_draw, request_scale_kwh, tank_arrays
from stems.dhw_plant import DHWPlant, TankSpec
from stems.heatpump_hplib import HplibSpec, columns, spec_arrays_of

COMFORT_C = (45.0, 55.0)
DISINFECT_C = 60.0
SUPPLY_NORMAL_C = 55.0
SUPPLY_DISINFECT_C = 65.0


def hold_met(post_draw_node_soc, soc_dis: float, tol: float = 1e-9) -> np.ndarray:
    s = np.asarray(post_draw_node_soc, dtype=np.float64)
    return np.all(s + tol >= soc_dis, axis=-1)


@dataclass
class StratifiedStep(StoreStep):
    post_draw_node_soc: Optional[np.ndarray] = None
    electricity_kwh: Optional[np.ndarray] = None


class StratifiedStoreModel(DHWStoreModel):

    def __init__(self, tanks: Sequence[TankSpec], units: Sequence[HplibSpec], *,
                 allow_element: bool = True) -> None:
        tanks, units = list(tanks), list(units)
        if not tanks or len(tanks) != len(units):
            raise ValueError("need one HplibSpec per TankSpec, and at least one store")
        if any(t.charging != "stratified" for t in tanks):
            raise ValueError("StratifiedStoreModel models stratified tanks only")
        nodes = {int(t.nodes) for t in tanks}
        if len(nodes) != 1:
            raise ValueError("every store in one model must have the same number of nodes")
        self.tanks, self.units = tanks, units
        self.B, self.n = len(tanks), nodes.pop()
        self.caps = np.stack([t.node_capacities_kwh for t in tanks]).astype(np.float64)
        self.allow_element = bool(allow_element)
        self.tank_arr = tank_arrays(tanks)
        self.tank_col = tank_arrays(tanks, column=True)
        hp = spec_arrays_of(units)
        if not self.allow_element:
            hp["element_kw"] = np.zeros(self.B)
        self.hp_arr = hp
        self.hp_col = columns(hp)
        self.t_cold = self.tank_arr["t_cold"]
        self.span = self.tank_arr["span"]
        self.sink = np.zeros(self.B, dtype=np.int64)
        self._rows = np.arange(self.B)
        self._sink_caps = self.caps[:, 0]
        self.t_supply_c = np.full(self.B, SUPPLY_NORMAL_C)
        self._inputs = None
        self.last_projection = None

    @classmethod
    def from_plants(cls, plants: Sequence[DHWPlant], *, allow_element: bool = True):
        return cls([p.tank for p in plants], [p.unit.spec for p in plants],
                   allow_element=allow_element)

    def soc_of_temperature(self, t_c) -> np.ndarray:
        return (np.asarray(t_c, dtype=np.float64) - self.t_cold) / self.span

    def _supply(self, t_supply_c=None) -> np.ndarray:
        t = self.t_supply_c if t_supply_c is None else t_supply_c
        return np.broadcast_to(np.asarray(t, dtype=np.float64), (self.B,)).astype(np.float64)

    def request_scale(self, t_source_c, t_supply_c=None) -> np.ndarray:
        ts = np.broadcast_to(np.asarray(t_source_c, dtype=np.float64), (self.B,))
        return request_scale_kwh(ts, self._supply(t_supply_c), self.hp_arr, self.allow_element)

    def request_kwh(self, action, t_source_c=None, t_supply_c=None) -> np.ndarray:
        if t_source_c is None:
            raise ValueError("a stratified request depends on the operating point: pass "
                             "t_source_c (and t_supply_c or set model.t_supply_c)")
        u = np.broadcast_to(np.asarray(action, dtype=np.float64), (self.B,))
        return u * self.request_scale(t_source_c, t_supply_c)


    def _after_draw(self, energy_kwh, draw_kwh):
        E = np.array(energy_kwh, dtype=np.float64).reshape(self.B, self.n)
        d = np.maximum(np.broadcast_to(np.asarray(draw_kwh, dtype=np.float64),
                                       (self.B,)).astype(np.float64), 0.0)
        E_d, served, unmet, _loss = loss_and_draw(E, d, self.tank_arr)
        return E_d, d, served, unmet

    def _charge(self, E_d: np.ndarray, u: np.ndarray, t_source: np.ndarray) -> Dict:
        sup = self._supply()[:, None]
        ts = np.asarray(t_source, dtype=np.float64).reshape(self.B, 1)
        scale = request_scale_kwh(ts, sup, self.hp_col, self.allow_element)
        req = np.maximum(u, 0.0) * scale
        out = charge_step(E_d[:, None, :], req, ts, sup, self.tank_col, self.hp_col)
        E_next = np.broadcast_to(out["E_next"], (self.B, u.shape[1], self.n))
        return {"E_next": E_next, "pre_loss": E_next,
                "post_charge": E_next[:, :, 0] / self.caps[:, None, 0],
                "delivered": np.broadcast_to(out["heat"], u.shape),
                "accepted": np.broadcast_to(out["heat"], u.shape),
                "spill": np.zeros(u.shape), "cap_kw": np.broadcast_to(out["capacity_kw"],
                                                                     u.shape),
                "limit": np.broadcast_to(np.minimum(scale, out["accept"]), u.shape),
                "electricity": np.broadcast_to(out["electricity"], u.shape)}

    def step(self, energy_kwh, action, draw_kwh, t_source_c, t_supply_c=None) -> StratifiedStep:
        if t_supply_c is not None:
            old = self.t_supply_c
            self.t_supply_c = self._supply(t_supply_c)
        try:
            E_d, d, served, unmet = self._after_draw(energy_kwh, draw_kwh)
            u = _col(np.broadcast_to(np.asarray(action, dtype=np.float64), (self.B,)))
            o = self._charge(E_d, u, self._t_source(t_source_c))
        finally:
            if t_supply_c is not None:
                self.t_supply_c = old
        E = np.array(o["E_next"][:, 0, :])
        return StratifiedStep(
            energy_kwh=E, node_soc=E / self.caps, pre_loss_node_soc=E / self.caps,
            post_charge_sink_soc=o["post_charge"][:, 0], served_kwh=served,
            unmet_kwh=unmet, delivered_kwh=np.array(o["delivered"][:, 0]),
            accepted_kwh=np.array(o["accepted"][:, 0]), spill_kwh=np.zeros(self.B),
            capacity_kw=np.array(o["cap_kw"][:, 0]),
            charge_limit_kwh=np.array(o["limit"][:, 0]),
            post_draw_node_soc=E_d / self.caps,
            electricity_kwh=np.array(o["electricity"][:, 0]))


class ForecastBundle:

    def __init__(self, price: np.ndarray, t_out: np.ndarray, source_fn, *, weather: bool,
                 seed: Optional[int], horizon: int = 24, noise_sigma_per_h: float = 0.0324,
                 window: int = 24) -> None:
        self.price = np.asarray(price, dtype=np.float64)
        self.t_out = np.asarray(t_out, dtype=np.float64)
        self.source_fn = source_fn
        self.weather = bool(weather)
        self.H = int(horizon)
        self.window = int(window)
        T = self.t_out.shape[0]
        if self.weather:
            rng = np.random.default_rng(seed)
            sig = noise_sigma_per_h * np.arange(1, self.H + 1)
            self.noise = rng.standard_normal((T, self.H)) * sig[None, :]
        else:
            self.noise = None
        csum = np.concatenate([[0.0], np.cumsum(self.t_out)])
        idx = np.arange(T)
        lo = np.maximum(idx - self.window + 1, 0)
        self.t_avg_actual = (csum[idx + 1] - csum[lo]) / (idx + 1 - lo)
        self.source_actual = np.asarray(self.source_fn(self.t_avg_actual), dtype=np.float64)

    def source_now(self, t: int) -> float:
        return float(self.source_actual[t])

    def source_path(self, t: int, H: Optional[int] = None) -> np.ndarray:
        H = self.H if H is None else int(H)
        T = self.t_out.shape[0]
        if not self.weather:
            return np.full(H, self.source_actual[t])
        out = np.empty(H)
        out[0] = self.source_actual[t]
        hist_lo = max(t - self.window + 1, 0)
        hist = list(self.t_out[hist_lo:t + 1])
        for j in range(1, H):
            k = min(t + j, T - 1)
            lead = min(j, self.H) - 1
            hist.append(float(self.t_out[k] + self.noise[t, lead]))
            w = hist[-self.window:]
            out[j] = float(self.source_fn(np.mean(w)))
        return out

    def price_path(self, t: int, H: Optional[int] = None) -> np.ndarray:
        H = self.H if H is None else int(H)
        T = self.price.shape[0]
        return self.price[np.minimum(t + np.arange(H), T - 1)]


class HourOfDayQuantileForecast:

    def __init__(self, B: int, q: float = 0.9, period: int = 24, start_index: int = 0,
                 max_days: int = 400) -> None:
        self.B, self.q, self.period = int(B), float(q), int(period)
        self.buf = np.zeros((self.period, max_days, self.B))
        self.cnt = np.zeros(self.period, dtype=np.int64)
        self.cache = np.zeros((self.period, self.B))
        self.all: List[np.ndarray] = []
        self.overall = np.zeros(self.B)
        self.t = int(start_index)

    def forecast(self, horizon: int) -> np.ndarray:
        horizon = max(int(horizon), 1)
        hrs = (self.t + np.arange(horizon)) % self.period
        seen = self.cnt[hrs] > 0
        return np.where(seen[:, None], self.cache[hrs], self.overall[None, :])

    def update(self, draw_kwh) -> None:
        d = np.broadcast_to(np.asarray(draw_kwh, dtype=np.float64), (self.B,))
        h = self.t % self.period
        i = self.cnt[h]
        if i >= self.buf.shape[1]:
            self.buf = np.concatenate([self.buf, np.zeros_like(self.buf)], axis=1)
        self.buf[h, i] = d
        self.cnt[h] += 1
        self.cache[h] = np.quantile(self.buf[h, : i + 1], self.q, axis=0)
        self.all.append(d.copy())
        if len(self.all) % 24 == 0 or len(self.all) < 24:
            self.overall = np.quantile(np.array(self.all), self.q, axis=0)
        self.t += 1


TRIGGERS = ("timer", "jit", "recourse")


class DisinfectionBarrier:

    def __init__(self, model: StratifiedStoreModel, bands: PhaseBands, *,
                 trigger: str = "recourse", use_filter: bool = True,
                 period_steps: float = 168.0, safety_stock_kwh=0.0, jit_reserve_kwh=0.0,
                 timer_offset: int = 2, guard_includes_floor: bool = True,
                 on_empty: str = "floor_first", a_max: float = 1.0,
                 t_supply_normal_c: float = SUPPLY_NORMAL_C,
                 t_supply_disinfect_c: float = SUPPLY_DISINFECT_C,
                 t_disinfect_c: float = DISINFECT_C, plan_horizon: Optional[int] = None):
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {TRIGGERS}")
        if bands.n != model.n:
            raise ValueError("bands and model disagree on the number of nodes")
        B = model.B
        self.model, self.bands = model, bands
        self.trigger, self.use_filter = trigger, bool(use_filter)
        self.period = float(period_steps)
        self.stock = np.broadcast_to(np.asarray(safety_stock_kwh, float), (B,)).astype(float)
        self.reserve = np.broadcast_to(np.asarray(jit_reserve_kwh, float), (B,)).astype(float)
        self.timer_offset = int(timer_offset)
        self.guard_floor = bool(guard_includes_floor)
        self.on_empty = on_empty
        self.a_max = float(a_max)
        self.t_sup_n, self.t_sup_d = float(t_supply_normal_c), float(t_supply_disinfect_c)
        self.soc_dis = (float(t_disinfect_c) - model.t_cold) / model.span
        self.plan_horizon = plan_horizon
        self.reset()

    def reset(self) -> None:
        B = self.model.B
        self.q = np.full(B, PHASE_NORMAL, dtype=np.int64)
        self.k = np.zeros(B)
        self.t = 0
        self.owed = np.ones(B, dtype=bool)
        self.pending_exit = np.zeros(B, dtype=bool)
        self.timer_due = np.zeros(B, dtype=bool)
        self.completions = np.zeros(B, dtype=np.int64)
        self.hold_events = np.zeros(B, dtype=np.int64)
        self.missed_windows = np.zeros(B, dtype=np.int64)
        self.completion_steps: List[Tuple[int, int]] = []
        self.hold_steps: List[Tuple[int, int]] = []
        self.refusals = np.zeros(B, dtype=np.int64)
        self.refusals_above_ceiling = np.zeros(B, dtype=np.int64)
        self.refusals_below_floor = np.zeros(B, dtype=np.int64)
        self.empty_steps = np.zeros(B, dtype=np.int64)
        self.empty_by_reason = {"floor_unreachable": 0, "ceiling_unavoidable": 0, "conflict": 0}
        self.forced_steps = np.zeros(B, dtype=np.int64)
        self.modified_steps = np.zeros(B, dtype=np.int64)
        self.phase_steps = np.zeros((B, 2), dtype=np.int64)
        self.max_nonempty_residual = 0.0


    def steps_left(self) -> np.ndarray:
        return np.maximum(self.period - self.k, 0.0)

    def _jit_must(self, E, t_source_now) -> np.ndarray:
        m = self.model
        soc = np.asarray(E, float) / m.caps
        gap = (m.caps * np.maximum(self.soc_dis[:, None] - soc, 0.0)).sum(axis=1)
        gross = m.request_scale(t_source_now, np.full(m.B, self.t_sup_d))
        loss = (m.tank_arr["loss"] * m.caps.sum(axis=1) * self.soc_dis)
        rate = np.maximum(gross - self.reserve - loss, 1e-6)
        steps = np.ceil(gap / rate) + 1.0
        return self.steps_left() <= steps

    def _recourse_steps(self, E, draw_now, source_path, draw_forecast) -> np.ndarray:
        m = self.model
        L = self.steps_left()
        H = int(np.max(L)) if self.plan_horizon is None else int(min(np.max(L),
                                                                     self.plan_horizon))
        H = max(H, 1)
        steps = np.full(m.B, np.inf)
        En = np.asarray(E, float).copy()
        full = np.full(m.B, self.a_max)
        sup = np.full(m.B, self.t_sup_d)
        for j in range(H):
            d = (np.asarray(draw_now, float) + self.stock if j == 0
                 else draw_forecast[min(j, draw_forecast.shape[0] - 1)])
            ts = source_path[min(j, len(source_path) - 1)]
            st = m.step(En, full, d, ts, t_supply_c=sup)
            done = hold_met(st.post_draw_node_soc, self.soc_dis[:, None]) & ~np.isfinite(steps)
            steps = np.where(done, j + 1.0, steps)
            En = st.energy_kwh
            if np.all(np.isfinite(steps) | (j + 1 >= L)):
                break
        return np.where(np.isfinite(steps), steps, L + 1.0)

    def deadline(self, E, draw_now, t_source_now, source_path=None,
                 draw_forecast=None) -> np.ndarray:
        if self.trigger == "timer":
            if int(self.t % int(self.period)) == self.timer_offset:
                self.timer_due[:] = True
            return self.timer_due & self.owed
        if self.trigger == "jit":
            return self._jit_must(E, t_source_now) & self.owed
        if not self.owed.any():
            return np.zeros(self.model.B, dtype=bool)
        if source_path is None:
            source_path = np.full(int(max(np.max(self.steps_left()), 1)), t_source_now)
        steps = self._recourse_steps(E, draw_now, source_path, draw_forecast)
        return (self.steps_left() <= steps) & self.owed


    def _feasible(self, node_soc, q):
        lo, hi = self.bands.lo_hi(q)
        s = np.asarray(node_soc, float).reshape(self.model.B, self.model.n)
        return (s <= hi).all(axis=1), (s >= lo).all(axis=1)

    def _guard(self, desired, node_soc) -> None:
        change = desired != self.q
        if not change.any():
            return
        ceil_from, floor_from = self._feasible(node_soc, self.q)
        ceil_to, floor_to = self._feasible(node_soc, desired)
        ceil_ok = ceil_from & ceil_to
        floor_ok = (floor_from & floor_to) if self.guard_floor else np.ones_like(ceil_ok)
        ok = ceil_ok & floor_ok
        refuse = change & ~ok
        self.refusals += refuse
        self.refusals_above_ceiling += refuse & ~ceil_ok
        self.refusals_below_floor += refuse & ceil_ok & ~floor_ok
        self.q = np.where(change & ok, desired, self.q)

    @property
    def in_d_owed(self) -> np.ndarray:
        return (self.q == PHASE_DISINFECT) & ~self.pending_exit & self.owed

    def supply(self) -> np.ndarray:
        return np.where((self.q == PHASE_DISINFECT) & ~self.pending_exit & self.owed,
                        self.t_sup_d, self.t_sup_n)


    def decide(self, u_policy, q_request, E, draw_barrier, t_source_now, *,
               source_path=None, draw_forecast=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        m = self.model
        E = np.asarray(E, float)
        must = self.deadline(E, draw_barrier, t_source_now, source_path, draw_forecast)
        want_q = (self.q.copy() if q_request is None
                  else np.asarray(q_request, dtype=np.int64).copy())
        want_q = np.where(self.pending_exit & ~must, PHASE_NORMAL, want_q)
        want_q = np.where(must, PHASE_DISINFECT, want_q)
        if self.use_filter:
            self._guard(want_q, E / m.caps)
        else:
            self.q = want_q
        t_sup = self.supply()
        a_min = np.where(must, self.a_max, 0.0)
        self.forced_steps += must
        want = np.maximum(np.clip(np.broadcast_to(np.asarray(u_policy, float), (m.B,)),
                                  0.0, self.a_max), a_min)
        info: Dict[str, Any] = {"must": must, "t_supply": t_sup, "empty": np.zeros(m.B, bool)}
        if not self.use_filter:
            return want, t_sup, info
        m.t_supply_c = t_sup
        lo, hi = self.bands.lo_hi(self.q)
        res = m.project(E, want, lo, hi, draw_barrier, t_source_now, self.a_max,
                        self.on_empty, need_interval=False)
        self.modified_steps += np.abs(res.action - want) > 1e-12
        self.max_nonempty_residual = max(self.max_nonempty_residual,
                                         res.residual_when_nonempty)
        for b in np.where(res.empty)[0]:
            self.empty_steps[b] += 1
            if res.conflict[b]:
                self.empty_by_reason["conflict"] += 1
            elif (res.status[b] == FLOOR_UNREACHABLE).any():
                self.empty_by_reason["floor_unreachable"] += 1
            else:
                self.empty_by_reason["ceiling_unavoidable"] += 1
        info["empty"] = res.empty
        return res.action, t_sup, info

    def observe(self, post_draw_node_soc) -> Dict[str, np.ndarray]:
        B = self.model.B
        self.phase_steps[np.arange(B), self.q] += 1
        held = hold_met(post_draw_node_soc, self.soc_dis[:, None])
        done = held & self.owed
        self.hold_events += held
        for b in np.where(held)[0]:
            self.hold_steps.append((self.t, int(b)))
        for b in np.where(done)[0]:
            self.completion_steps.append((self.t, int(b)))
        self.completions += done
        self.owed = self.owed & ~done
        self.k = self.k + 1.0
        end = self.k >= self.period
        expired = end & self.owed
        self.missed_windows += expired
        self.k = np.where(end, 0.0, self.k)
        self.owed = np.where(end, True, self.owed)
        self.timer_due = self.timer_due & ~done & ~end
        self.pending_exit = (self.pending_exit | done) & (self.q == PHASE_DISINFECT)
        if not self.use_filter:
            self.q = np.where(self.pending_exit, PHASE_NORMAL, self.q)
            self.pending_exit[:] = False
        self.t += 1
        return {"completed": done, "held": held, "expired": expired}

    def report(self) -> Dict[str, Any]:
        return {"trigger": self.trigger, "filter": self.use_filter,
                "completions": self.completions.tolist(),
                "missed_windows": self.missed_windows.tolist(),
                "hold_events": self.hold_events.tolist(),
                "refusals": self.refusals.tolist(),
                "refusals_above_ceiling": self.refusals_above_ceiling.tolist(),
                "refusals_below_floor": self.refusals_below_floor.tolist(),
                "empty_steps": self.empty_steps.tolist(),
                "empty_by_reason": dict(self.empty_by_reason),
                "forced_steps": self.forced_steps.tolist(),
                "modified_steps": self.modified_steps.tolist(),
                "steps_in_N": self.phase_steps[:, 0].tolist(),
                "steps_in_D": self.phase_steps[:, 1].tolist(),
                "max_band_residual_on_nonempty_steps": self.max_nonempty_residual}


class HysteresisState:
    def __init__(self, B: int) -> None:
        self.on = np.zeros(B, dtype=bool)


class ReactiveController:

    name = "reactive"

    def __init__(self, B: int, *, on_bottom_c=40.0, on_top_c=47.0, off_bottom_c=50.0):
        self.s = HysteresisState(B)
        self.on_bottom_c, self.on_top_c, self.off_bottom_c = on_bottom_c, on_top_c, off_bottom_c

    def act(self, T_top, T_bot, *, owed_d, **_):
        on = self.s.on
        on = np.where(on, T_bot < self.off_bottom_c,
                      (T_bot < self.on_bottom_c) | (T_top < self.on_top_c))
        self.s.on = on
        u = np.where(on | owed_d, 1.0, 0.0)
        return u, None


class PriceAwareController:

    name = "price_aware"

    def __init__(self, B: int, cop_fn, *, cheap_margin=0.05, request_after_h=120.0,
                 cheap=(50.0, 47.0, 53.0), dear=(30.0, 47.0, 40.0)):
        self.s = HysteresisState(B)
        self.cop_fn = cop_fn
        self.cheap_margin = float(cheap_margin)
        self.request_after_h = float(request_after_h)
        self.cheap_thr, self.dear_thr = cheap, dear

    def is_cheap(self, price_path, source_path) -> bool:
        c = np.asarray(price_path, float) / np.asarray(self.cop_fn(source_path), float)
        return bool(c[0] <= (1.0 - self.cheap_margin) * float(np.mean(c)))

    def act(self, T_top, T_bot, *, owed_d, k, price_path, source_path, window_owed=True, **_):
        cheap = self.is_cheap(price_path, source_path)
        on_b, on_t, off_b = self.cheap_thr if cheap else self.dear_thr
        on = self.s.on
        on = np.where(on, T_bot < off_b, (T_bot < on_b) | (T_top < on_t))
        self.s.on = on
        u = np.where(on | owed_d, 1.0, 0.0)
        q = np.where(owed_d | ((np.asarray(k) >= self.request_after_h) & cheap
                              & np.asarray(window_owed, dtype=bool)),
                     PHASE_DISINFECT, PHASE_NORMAL)
        return u, q
