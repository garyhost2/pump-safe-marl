from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Tuple

import numpy as np

_T0_K = 273.15

_COP_MIN = 1.0
_COP_MAX = 8.0


@dataclass
class ThermiaSpec:

    power_min_kw: float
    power_max_kw: float
    eta_ref: float
    t_sink_ref_c: float
    eta_slope_per_k: float
    capacity_ref_kw: float
    t_source_ref_c: float
    capacity_slope_per_k_sink: float
    capacity_slope_per_k_source: float
    t_sink_max_c: float
    element_kw: float
    element_efficiency: float
    cycling_penalty_kwh: float
    dt_hours: float = 1.0
    provenance: Dict[str, str] = field(default_factory=dict)

    OPEN_QUESTIONS: ClassVar[Tuple[str, ...]] = (
        "power_min_kw", "power_max_kw", "eta_ref", "t_sink_ref_c", "eta_slope_per_k",
        "capacity_ref_kw", "t_source_ref_c", "capacity_slope_per_k_sink",
        "capacity_slope_per_k_source", "t_sink_max_c", "element_kw",
        "element_efficiency", "cycling_penalty_kwh")

    def __post_init__(self) -> None:
        if not 0.0 <= self.power_min_kw < self.power_max_kw:
            raise ValueError(
                f"need 0 <= power_min_kw ({self.power_min_kw}) < power_max_kw "
                f"({self.power_max_kw}); an inverter with no range is a fixed-speed unit "
                f"and should be declared as one")
        if not 0.0 < self.eta_ref <= 1.0:
            raise ValueError(f"eta_ref ({self.eta_ref}) is a Carnot efficiency and must "
                             f"lie in (0, 1]")
        if self.eta_slope_per_k > 0.0:
            raise ValueError(
                f"eta_slope_per_k ({self.eta_slope_per_k}) is positive, which says the "
                f"unit becomes more efficient as the sink gets hotter; no "
                f"vapour-compression cycle does that")
        if self.capacity_slope_per_k_sink > 0.0:
            raise ValueError(
                f"capacity_slope_per_k_sink ({self.capacity_slope_per_k_sink}) is "
                f"positive, which says capacity rises with sink temperature. If the "
                f"intent was no derating, pass 0.0 and record that choice in provenance, "
                f"because it is the assumption the review asked us not to make silently")
        if self.capacity_slope_per_k_source < 0.0:
            raise ValueError(f"capacity_slope_per_k_source "
                             f"({self.capacity_slope_per_k_source}) should be >= 0")
        if self.capacity_ref_kw <= 0.0:
            raise ValueError("capacity_ref_kw must be positive")
        if self.element_kw < 0.0 or not 0.0 < self.element_efficiency <= 1.0:
            raise ValueError("element_kw must be >= 0 and element_efficiency in (0, 1]")
        if self.cycling_penalty_kwh < 0.0:
            raise ValueError("cycling_penalty_kwh must be >= 0")
        if self.dt_hours <= 0.0:
            raise ValueError("dt_hours must be positive")
        self.provenance = {k: str(v) for k, v in dict(self.provenance).items()}

    @property
    def unsourced(self) -> Tuple[str, ...]:
        return tuple(k for k in self.OPEN_QUESTIONS if k not in self.provenance)


    def carnot_efficiency(self, t_sink_c: np.ndarray | float) -> np.ndarray:
        eta = (self.eta_ref
               + self.eta_slope_per_k * (np.asarray(t_sink_c, dtype=np.float64)
                                         - self.t_sink_ref_c))
        return np.clip(eta, 1e-3, 1.0)

    def cop(self, t_source_c: np.ndarray | float,
            t_sink_c: np.ndarray | float) -> np.ndarray:
        return np.clip(self.cop_raw(t_source_c, t_sink_c), _COP_MIN, _COP_MAX)

    def cop_raw(self, t_source_c: np.ndarray | float,
                t_sink_c: np.ndarray | float) -> np.ndarray:
        t_sink_k = np.asarray(t_sink_c, dtype=np.float64) + _T0_K
        t_source_k = np.asarray(t_source_c, dtype=np.float64) + _T0_K
        lift = np.maximum(t_sink_k - t_source_k, 1e-6)
        return self.carnot_efficiency(t_sink_c) * t_sink_k / lift

    def capacity_kw(self, t_source_c: np.ndarray | float,
                    t_sink_c: np.ndarray | float) -> np.ndarray:
        t_sink = np.asarray(t_sink_c, dtype=np.float64)
        t_source = np.asarray(t_source_c, dtype=np.float64)
        cap = (self.capacity_ref_kw
               + self.capacity_slope_per_k_sink * (t_sink - self.t_sink_ref_c)
               + self.capacity_slope_per_k_source * (t_source - self.t_source_ref_c))
        cap = np.clip(cap, 0.0, self.power_max_kw)
        return np.where(t_sink >= self.t_sink_max_c, 0.0, cap)


@dataclass
class Delivery:

    thermal_kwh: float
    electricity_kwh: float
    compressor_thermal_kwh: float
    element_thermal_kwh: float
    duty: float
    cycling: bool
    starts: int
    cop_used: float
    capacity_kw: float
    cop_clamped: bool
    unmet_thermal_kwh: float


class ThermiaUnit:

    def __init__(self, spec: ThermiaSpec) -> None:
        self.spec = spec
        self.total_starts = 0
        self.total_cycling_steps = 0
        self._was_on = False

    def reset(self) -> None:
        self.total_starts = 0
        self.total_cycling_steps = 0
        self._was_on = False

    def deliver(self, request_thermal_kwh: float, t_source_c: float,
                t_sink_c: float, *, allow_element: bool = True) -> Delivery:
        s = self.spec
        dt = s.dt_hours
        request = max(float(request_thermal_kwh), 0.0)

        cap_kw = float(s.capacity_kw(t_source_c, t_sink_c))
        cop_raw = float(s.cop_raw(t_source_c, t_sink_c))
        cop = float(np.clip(cop_raw, _COP_MIN, _COP_MAX))
        clamped = not (_COP_MIN <= cop_raw <= _COP_MAX)

        comp_max_kwh = cap_kw * dt
        comp_min_kwh = min(s.power_min_kw, cap_kw) * dt

        comp_kwh = min(request, comp_max_kwh)
        duty, cycling, starts = 0.0, False, 0

        if comp_kwh <= 0.0 or comp_max_kwh <= 0.0:
            comp_kwh = 0.0
            on = False
        elif comp_min_kwh > 0.0 and comp_kwh < comp_min_kwh:
            duty = comp_kwh / comp_min_kwh
            cycling = True
            on = True
        else:
            duty = comp_kwh / comp_max_kwh if comp_max_kwh > 0.0 else 0.0
            on = True

        if on and not self._was_on:
            starts = 1
        if cycling:
            starts = max(starts, 1)
            self.total_cycling_steps += 1
        self._was_on = on
        self.total_starts += starts

        comp_elec = comp_kwh / cop if comp_kwh > 0.0 else 0.0

        remaining = request - comp_kwh
        elem_kwh = 0.0
        if allow_element and remaining > 1e-12 and s.element_kw > 0.0:
            elem_kwh = min(remaining, s.element_kw * dt)
        elem_elec = elem_kwh / s.element_efficiency if elem_kwh > 0.0 else 0.0

        electricity = comp_elec + elem_elec + starts * s.cycling_penalty_kwh
        delivered = comp_kwh + elem_kwh
        return Delivery(thermal_kwh=delivered, electricity_kwh=electricity,
                        compressor_thermal_kwh=comp_kwh, element_thermal_kwh=elem_kwh,
                        duty=float(duty), cycling=cycling, starts=starts,
                        cop_used=cop, capacity_kw=cap_kw, cop_clamped=clamped,
                        unmet_thermal_kwh=max(request - delivered, 0.0))

    def max_thermal_kwh(self, t_source_c: float, t_sink_c: float, *,
                        allow_element: bool = True) -> float:
        s = self.spec
        out = float(s.capacity_kw(t_source_c, t_sink_c)) * s.dt_hours
        if allow_element:
            out += s.element_kw * s.dt_hours
        return out

    def describe(self) -> Dict[str, object]:
        return {"kind": "ThermiaUnit",
                "spec": {k: getattr(self.spec, k) for k in ThermiaSpec.OPEN_QUESTIONS},
                "dt_hours": self.spec.dt_hours,
                "provenance": dict(self.spec.provenance),
                "unsourced": list(self.spec.unsourced),
                "total_starts": self.total_starts,
                "total_cycling_steps": self.total_cycling_steps}
