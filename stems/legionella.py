from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.flexibility import FlexibleLoad

__all__ = [
    "LegionellaSpec",
    "ShadowTank",
    "LegionellaCycleBarrier",
    "LegionellaStack",
    "HOURS_PER_WEEK",
]

HOURS_PER_WEEK = 168.0

_UNSOURCED = "UNSOURCED"


@dataclass
class LegionellaSpec:

    capacity_kwh: np.ndarray
    heat_source_kw: np.ndarray
    t_cold_c: float
    t_rated_c: float
    t_normal_c: float
    t_legionella_c: float
    cop_legionella: float
    element_efficiency: float
    t_heat_pump_max_c: float
    loss_coefficient: np.ndarray = 0.0
    draw_margin_kwh: np.ndarray = 0.0
    period_hours: float = HOURS_PER_WEEK
    dt_hours: float = 1.0
    provenance: Dict[str, str] = field(default_factory=dict)

    OPEN_QUESTIONS: ClassVar[Tuple[str, ...]] = (
        "heat_source_kw", "t_cold_c", "t_rated_c", "t_normal_c", "t_legionella_c",
        "cop_legionella", "element_efficiency", "t_heat_pump_max_c")

    def __post_init__(self) -> None:
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        self.capacity_kwh = f(self.capacity_kwh)
        self.B = self.capacity_kwh.size
        self.heat_source_kw = np.broadcast_to(f(self.heat_source_kw), (self.B,)).copy()
        self.loss_coefficient = np.broadcast_to(f(self.loss_coefficient),
                                                (self.B,)).copy()
        self.draw_margin_kwh = np.broadcast_to(f(self.draw_margin_kwh),
                                               (self.B,)).copy()
        if self.t_rated_c <= self.t_cold_c:
            raise ValueError(
                f"t_rated_c ({self.t_rated_c}) must exceed t_cold_c ({self.t_cold_c}): "
                "the rated capacity is the energy between the inlet and the rated "
                "temperature, and a non-positive span makes every state of charge "
                "meaningless.")
        if self.t_legionella_c <= self.t_normal_c:
            raise ValueError(
                f"t_legionella_c ({self.t_legionella_c}) must exceed t_normal_c "
                f"({self.t_normal_c}); a disinfection set point at or below the normal "
                "one is satisfied by doing nothing, and the constraint would be vacuous.")
        if not np.all(self.heat_source_kw > 0.0):
            raise ValueError("heat_source_kw must be positive: a store with no source "
                             "can never reach its required level.")
        if np.any(self.draw_margin_kwh < 0.0):
            raise ValueError("draw_margin_kwh must be non-negative")
        if np.any(self.draw_margin_kwh >= self.heat_source_kw * self.dt_hours):
            raise ValueError(
                "draw_margin_kwh is at or above the source's output per step, so the "
                "reserved draw alone consumes everything the source can deliver and "
                "the cycle can never progress. Either the source is undersized for "
                "this draw or the reserve is wrong.")
        if self.period_hours < self.dt_hours:
            raise ValueError(f"period_hours ({self.period_hours}) is shorter than one "
                             f"step ({self.dt_hours})")
        self.provenance = {k: str(v) for k, v in dict(self.provenance).items()}


    def soc_of_temperature(self, t_c: float) -> float:
        return float(np.clip((t_c - self.t_cold_c) / (self.t_rated_c - self.t_cold_c),
                             0.0, 1.0))

    @property
    def soc_legionella(self) -> float:
        return self.soc_of_temperature(self.t_legionella_c)

    @property
    def soc_normal(self) -> float:
        return self.soc_of_temperature(self.t_normal_c)

    @property
    def soc_heat_pump_max(self) -> float:
        return self.soc_of_temperature(self.t_heat_pump_max_c)

    @property
    def rate(self) -> np.ndarray:
        return (self.heat_source_kw * self.dt_hours
                / np.maximum(self.capacity_kwh, 1e-9))

    @property
    def net_rate(self) -> np.ndarray:
        gross = (self.heat_source_kw * self.dt_hours - self.draw_margin_kwh)
        return np.maximum(gross / np.maximum(self.capacity_kwh, 1e-9)
                          - self.loss_coefficient * self.soc_legionella, 1e-6)

    @property
    def period_steps(self) -> float:
        return float(self.period_hours / self.dt_hours)

    @property
    def unsourced(self) -> List[str]:
        out = [k for k in self.OPEN_QUESTIONS
               if self.provenance.get(k, _UNSOURCED) == _UNSOURCED]
        if (np.any(self.draw_margin_kwh > 0.0)
                and self.provenance.get("draw_margin_kwh", _UNSOURCED) == _UNSOURCED):
            out.append("draw_margin_kwh")
        return out

    def summary(self) -> Dict[str, Any]:
        return {"capacity_kwh": [round(float(c), 4) for c in self.capacity_kwh],
                "heat_source_kw": [round(float(p), 4) for p in self.heat_source_kw],
                "t_cold_c": self.t_cold_c, "t_rated_c": self.t_rated_c,
                "t_normal_c": self.t_normal_c, "t_legionella_c": self.t_legionella_c,
                "soc_normal": round(self.soc_normal, 4),
                "soc_legionella": round(self.soc_legionella, 4),
                "soc_heat_pump_max": round(self.soc_heat_pump_max, 4),
                "cop_legionella": self.cop_legionella,
                "element_efficiency": self.element_efficiency,
                "period_hours": self.period_hours,
                "rate_soc_per_step": [round(float(r), 5) for r in self.rate],
                "net_rate_soc_per_step": [round(float(r), 5) for r in self.net_rate],
                "draw_margin_kwh": [round(float(d), 4) for d in self.draw_margin_kwh],
                "hours_to_disinfect_from_normal": [
                    round(float((self.soc_legionella - self.soc_normal) / r), 3)
                    for r in self.net_rate],
                "unsourced_parameters": self.unsourced,
                "provenance": dict(self.provenance)}


class ShadowTank:

    def __init__(self, spec: LegionellaSpec, soc0: Optional[np.ndarray] = None) -> None:
        self.spec = spec
        self.B = spec.B
        self.reset(soc0)

    def reset(self, soc0: Optional[np.ndarray] = None) -> None:
        start = (np.full(self.B, self.spec.soc_normal) if soc0 is None
                 else np.clip(np.asarray(soc0, dtype=np.float64).reshape(-1), 0.0, 1.0))
        self.E = start * self.spec.capacity_kwh
        self.steps = 0
        self.unmet_kwh = np.zeros(self.B)
        self.demand_kwh = np.zeros(self.B)
        self.spill_kwh = np.zeros(self.B)
        self.heat_kwh = np.zeros(self.B)
        self.electricity_kwh = np.zeros(self.B)
        self.unmet_events = np.zeros(self.B, dtype=np.int64)
        self.draw_events = np.zeros(self.B, dtype=np.int64)

    @property
    def soc(self) -> np.ndarray:
        return self.E / np.maximum(self.spec.capacity_kwh, 1e-9)

    def heat_for_action(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), 0.0, 1.0)
        return a * self.spec.heat_source_kw * self.spec.dt_hours

    def electricity_for_heat(self, heat_kwh: np.ndarray,
                             soc_before: np.ndarray) -> np.ndarray:
        heat = np.maximum(np.asarray(heat_kwh, dtype=np.float64).reshape(-1), 0.0)
        cap = self.spec.capacity_kwh
        ceiling = self.spec.soc_heat_pump_max * cap
        e0 = np.clip(np.asarray(soc_before, dtype=np.float64).reshape(-1), 0.0, 1.0) * cap
        by_pump = np.clip(ceiling - e0, 0.0, heat)
        by_element = heat - by_pump
        return (by_pump / max(self.spec.cop_legionella, 1e-6)
                + by_element / max(self.spec.element_efficiency, 1e-6))

    def step(self, heat_requested_kwh: np.ndarray,
             draw_kwh: np.ndarray) -> Dict[str, np.ndarray]:
        s = self.spec
        draw = np.maximum(np.asarray(draw_kwh, dtype=np.float64).reshape(-1), 0.0)
        soc_before = self.soc.copy()
        e_avail = np.maximum(self.E * (1.0 - s.loss_coefficient), 0.0)
        heat_in = np.clip(np.asarray(heat_requested_kwh, dtype=np.float64).reshape(-1),
                          0.0, s.heat_source_kw * s.dt_hours)
        served = np.minimum(draw, e_avail + heat_in)
        unmet = draw - served
        raw = e_avail + heat_in - served
        e_next = np.minimum(raw, s.capacity_kwh)
        spill = raw - e_next
        electricity = self.electricity_for_heat(heat_in, soc_before)

        self.E = e_next
        self.steps += 1
        self.unmet_kwh += unmet
        self.demand_kwh += draw
        self.spill_kwh += spill
        self.heat_kwh += heat_in
        self.electricity_kwh += electricity
        self.unmet_events += (unmet > 1e-9).astype(np.int64)
        self.draw_events += (draw > 1e-9).astype(np.int64)
        return {"soc": self.soc.copy(), "unmet_kwh": unmet, "served_kwh": served,
                "heat_in_kwh": heat_in, "spill_kwh": spill,
                "electricity_kwh": electricity}

    def report(self) -> Dict[str, float]:
        demand = float(self.demand_kwh.sum())
        draws = int(self.draw_events.sum())
        return {"unmet_kwh": float(self.unmet_kwh.sum()),
                "demand_kwh": demand,
                "unmet_share_of_demand": (float(self.unmet_kwh.sum()) / demand
                                          if demand > 0 else 0.0),
                "unmet_draws": int(self.unmet_events.sum()),
                "draws": draws,
                "unmet_share_of_draws": (float(self.unmet_events.sum()) / draws
                                         if draws else 0.0),
                "heat_kwh": float(self.heat_kwh.sum()),
                "electricity_kwh": float(self.electricity_kwh.sum()),
                "spill_kwh": float(self.spill_kwh.sum()),
                "steps": int(self.steps)}


class LegionellaCycleBarrier(DeadlineStorageBarrier):

    TRIGGERS = ("jit", "recourse")

    def __init__(self, spec: LegionellaSpec, tank: ShadowTank, action_index: int,
                 action_bound: Optional[np.ndarray] = None, margin: float = 0.0,
                 name: str = "legionella", trigger: str = "jit",
                 forecaster: Optional[Any] = None, safety_stock_kwh: Any = 0.0) -> None:
        if trigger not in self.TRIGGERS:
            raise ValueError(f"trigger must be one of {self.TRIGGERS}, got {trigger!r}")
        if trigger == "recourse" and forecaster is None:
            raise ValueError("the recourse trigger plans against a draw forecast; pass "
                             "forecaster= (e.g. stems.dhw_barrier.HourOfDayDrawForecast)")
        self.trigger = trigger
        self.forecaster = forecaster
        self.safety_stock_kwh = np.broadcast_to(
            np.asarray(safety_stock_kwh, dtype=np.float64), (spec.B,)).astype(np.float64)
        if np.any(self.safety_stock_kwh < 0.0):
            raise ValueError("safety_stock_kwh must be non-negative")
        self.spec = spec
        self.tank = tank
        self.B = spec.B
        self.steps_since_cycle = np.zeros(self.B, dtype=np.float64)
        self.cycles_completed = np.zeros(self.B, dtype=np.int64)
        self.missed_windows = np.zeros(self.B, dtype=np.int64)
        bound = np.ones(self.B) if action_bound is None else action_bound
        super().__init__(
            rate=spec.net_rate,
            action_bound=bound,
            action_index=int(action_index),
            capacity=spec.capacity_kwh,
            efficiency=np.ones(self.B),
            requirement_fn=self._requirement,
            soc_fn=lambda _obs: self.tank.soc.astype(np.float32),
            margin=margin,
            soc_cap=1.0,
            name=name,
        )

    def current_rate(self, obs_list: Sequence[np.ndarray]) -> np.ndarray:
        return self.rate

    def _recourse_steps(self, steps_to_deadline: np.ndarray) -> np.ndarray:
        s = self.spec
        L = np.asarray(steps_to_deadline, dtype=np.float64)
        H = max(int(np.max(L)) if L.size else 0, 1)
        fc = np.asarray(self.forecaster.forecast(H), dtype=np.float64)
        E = self.tank.E.astype(np.float64).copy()
        cap = np.maximum(s.capacity_kwh, 1e-9)
        heat = s.heat_source_kw * s.dt_hours
        steps = np.full(self.B, np.inf)
        for j in range(H):
            d = fc[min(j, fc.shape[0] - 1)] + (self.safety_stock_kwh if j == 0 else 0.0)
            e_avail = np.maximum(E * (1.0 - s.loss_coefficient), 0.0)
            served = np.minimum(d, e_avail + heat)
            E = np.minimum(e_avail + heat - served, s.capacity_kwh)
            done = (E / cap + 1e-9 >= s.soc_legionella) & ~np.isfinite(steps)
            steps = np.where(done, float(j + 1), steps)
            if np.all(np.isfinite(steps) | (j + 1 >= L)):
                break
        return np.where(np.isfinite(steps), steps, L + 1.0)

    def urgency(self, obs_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        u = super().urgency(obs_list)
        if self.trigger == "jit":
            return u
        L = u["steps_to_deadline"].astype(np.float64)
        steps = self._recourse_steps(L)
        u["steps_needed"] = steps.astype(np.float32)
        u["slack"] = (L - steps).astype(np.float32)
        return u

    def _forced_action(self, u: Dict[str, np.ndarray],
                       obs_list: List[np.ndarray]) -> np.ndarray:
        if self.trigger == "jit":
            return super()._forced_action(u, obs_list)
        s = self.spec
        e_avail = np.maximum(self.tank.E * (1.0 - s.loss_coefficient), 0.0)
        d = np.asarray(self.forecaster.forecast(1), dtype=np.float64)[0] + self.safety_stock_kwh
        heat = s.soc_legionella * s.capacity_kwh - e_avail + d
        a = np.clip(heat / np.maximum(s.heat_source_kw * s.dt_hours, 1e-9), 0.0, 1.0)
        return a.astype(np.float32)

    def _requirement(self, obs_list: Sequence[np.ndarray]) -> DeadlineRequirement:
        left = np.maximum(self.spec.period_steps - self.steps_since_cycle, 0.0)
        return DeadlineRequirement(
            soc=np.full(self.B, self.spec.soc_legionella),
            steps_to_deadline=left,
            active=np.ones(self.B, dtype=bool))

    def observe(self, soc: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        level = self.tank.soc if soc is None else np.asarray(soc, dtype=np.float64)
        done = level + 1e-9 >= self.spec.soc_legionella
        self.steps_since_cycle += 1.0
        expired = (~done) & (self.steps_since_cycle >= self.spec.period_steps)
        self.missed_windows += expired.astype(np.int64)
        self.cycles_completed += done.astype(np.int64)
        self.steps_since_cycle = np.where(done | expired, 0.0, self.steps_since_cycle)
        return {"completed": done, "expired": expired,
                "steps_since_cycle": self.steps_since_cycle.copy()}

    def report(self) -> Dict[str, Any]:
        return {"trigger": self.trigger,
                "safety_stock_kwh": [round(float(x), 4) for x in self.safety_stock_kwh],
                "cycles_completed": [int(c) for c in self.cycles_completed],
                "missed_windows": [int(m) for m in self.missed_windows],
                "steps_since_cycle": [float(s) for s in self.steps_since_cycle],
                "soc_legionella": self.spec.soc_legionella,
                "period_steps": self.spec.period_steps}


@dataclass
class LegionellaStack:

    spec: LegionellaSpec
    tank: ShadowTank
    barrier: LegionellaCycleBarrier
    load: FlexibleLoad

    @classmethod
    def build(cls, spec: LegionellaSpec, action_index: int,
              action_bound: Optional[np.ndarray] = None,
              soc0: Optional[np.ndarray] = None, *, trigger: str = "jit",
              forecaster: Optional[Any] = None,
              safety_stock_kwh: Any = 0.0) -> "LegionellaStack":
        tank = ShadowTank(spec, soc0=soc0)
        barrier = LegionellaCycleBarrier(spec, tank, action_index,
                                         action_bound=action_bound, trigger=trigger,
                                         forecaster=forecaster,
                                         safety_stock_kwh=safety_stock_kwh)
        load = FlexibleLoad(
            name="legionella", kind="legionella", action_index=action_index,
            barrier=barrier, capacity_kwh=spec.capacity_kwh,
            power_kw=spec.heat_source_kw,
            action_bound=float(np.max(barrier.action_bound)),
            notes=("weekly disinfection deadline on the hot-water store; unsourced "
                   f"parameters: {spec.unsourced or 'none'}"))
        return cls(spec=spec, tank=tank, barrier=barrier, load=load)

    def observe(self, action: np.ndarray, draw_kwh: np.ndarray) -> Dict[str, np.ndarray]:
        out = self.tank.step(self.tank.heat_for_action(action), draw_kwh)
        out.update(self.barrier.observe())
        if self.barrier.forecaster is not None:
            self.barrier.forecaster.update(draw_kwh)
        return out

    def report(self) -> Dict[str, Any]:
        return {"service": self.tank.report(), "cycle": self.barrier.report(),
                "spec": self.spec.summary()}
