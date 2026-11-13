from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Optional, Tuple

import numpy as np

HPLIB_GENERIC: Dict[int, Dict[str, Tuple[float, float, float, float]]] = {
    1: {"p_el": (65.24998301955087, 0.011364929455468386, 0.04700671542496862,
                 -65.29932853643015),
        "cop": (46.37462900126993, -0.08756667775339223, 7.0454352113504,
                -46.22057938564581)},
    2: {"p_el": (-0.013036429478276419, 0.013036428662668095, 0.06982106812617543,
                 -0.03415413177391993),
        "cop": (0.09856460505867312, -0.09856460590283955, 8.12241287944408,
                -0.00010931425657465474)},
}

HPLIB_SOURCE = ("hplib 1.9 (PyPI wheel hplib-1.9-py3-none-any.whl), hplib_database.csv, "
                "rows Model=='Generic'; Tjaden, Hoops, Roesken (2021) "
                "doi:10.5281/zenodo.5521597")

HPLIB_BRINE_T_AMB_C = -7.0

_COP_FLOOR = 1.0


def brine_temperature_c(t_avg_day_c, clip=(-15.0, 25.0)):
    t = np.clip(np.asarray(t_avg_day_c, dtype=np.float64), clip[0], clip[1])
    return -0.0003 * t ** 3 + 0.0086 * t ** 2 + 0.3047 * t + 5.0647


@dataclass
class HplibSpec:

    group: int
    p_el_ref_kw: float
    min_modulation: float
    t_sink_max_c: float
    element_kw: float = 0.0
    element_efficiency: float = 1.0
    cycling_penalty_kwh: float = 0.0
    dt_hours: float = 1.0
    provenance: Dict[str, str] = field(default_factory=dict)

    KIND: ClassVar[str] = "hplib"

    def __post_init__(self) -> None:
        if self.group not in HPLIB_GENERIC:
            raise ValueError(f"group must be one of {sorted(HPLIB_GENERIC)}")
        if self.p_el_ref_kw <= 0.0:
            raise ValueError("p_el_ref_kw must be positive")
        if not 0.0 <= self.min_modulation < 1.0:
            raise ValueError("min_modulation is a fraction in [0, 1)")
        if self.element_kw < 0.0 or not 0.0 < self.element_efficiency <= 1.0:
            raise ValueError("element_kw must be >= 0 and element_efficiency in (0, 1]")
        if self.cycling_penalty_kwh < 0.0 or self.dt_hours <= 0.0:
            raise ValueError("cycling_penalty_kwh >= 0 and dt_hours > 0 required")
        self.provenance = {k: str(v) for k, v in dict(self.provenance).items()}


    def _t_amb(self, t_source):
        if self.group == 2:
            return np.full_like(np.asarray(t_source, dtype=np.float64), HPLIB_BRINE_T_AMB_C)
        return np.asarray(t_source, dtype=np.float64)

    def cop_raw(self, t_source_c, t_sink_c) -> np.ndarray:
        p1, p2, p3, p4 = HPLIB_GENERIC[self.group]["cop"]
        ts = np.asarray(t_source_c, dtype=np.float64)
        return p1 * ts + p2 * np.asarray(t_sink_c, dtype=np.float64) + p3 + p4 * self._t_amb(ts)

    def cop(self, t_source_c, t_sink_c) -> np.ndarray:
        return np.maximum(self.cop_raw(t_source_c, t_sink_c), _COP_FLOOR)

    def p_el_kw(self, t_source_c, t_sink_c) -> np.ndarray:
        p1, p2, p3, p4 = HPLIB_GENERIC[self.group]["p_el"]
        ts = np.asarray(t_source_c, dtype=np.float64)
        return self.p_el_ref_kw * (p1 * ts + p2 * np.asarray(t_sink_c, dtype=np.float64)
                                   + p3 + p4 * self._t_amb(ts))

    def capacity_kw(self, t_source_c, t_sink_c) -> np.ndarray:
        t_sink = np.asarray(t_sink_c, dtype=np.float64)
        cap = np.maximum(self.p_el_kw(t_source_c, t_sink) * self.cop(t_source_c, t_sink), 0.0)
        return np.where(t_sink > self.t_sink_max_c, 0.0, cap)

    def min_output_kw(self, t_source_c, t_sink_c) -> np.ndarray:
        return self.min_modulation * self.capacity_kw(t_source_c, t_sink_c)

    @classmethod
    def scaled_to(cls, *, group: int, p_th_max_kw: float, t_source_ref_c: float,
                  t_sink_ref_c: float, **kw) -> "HplibSpec":
        unit = cls(group=group, p_el_ref_kw=1.0, min_modulation=kw.pop("min_modulation"),
                   t_sink_max_c=kw.pop("t_sink_max_c"), **kw)
        per_kw = float(unit.p_el_kw(t_source_ref_c, t_sink_ref_c)
                       * unit.cop(t_source_ref_c, t_sink_ref_c))
        unit.p_el_ref_kw = float(p_th_max_kw) / per_kw
        return unit

    def describe(self) -> Dict[str, object]:
        return {"kind": self.KIND, "group": self.group, "p_el_ref_kw": self.p_el_ref_kw,
                "min_modulation": self.min_modulation, "t_sink_max_c": self.t_sink_max_c,
                "element_kw": self.element_kw, "element_efficiency": self.element_efficiency,
                "cycling_penalty_kwh": self.cycling_penalty_kwh,
                "coefficients": HPLIB_GENERIC[self.group], "source": HPLIB_SOURCE,
                "provenance": dict(self.provenance)}


CALIBRA_ECO_8 = dict(
    group=2, p_th_max_kw=8.0, t_source_ref_c=0.0, t_sink_ref_c=35.0, min_modulation=0.25,
    t_sink_max_c=65.0, element_kw=0.0, element_efficiency=1.0, cycling_penalty_kwh=0.0,
    provenance={
        "shape": HPLIB_SOURCE + ", group 2 (brine/water, regulated), T_amb=-7 degC",
        "p_th_max_kw": "Thermia Calibra Eco 8 datasheet: heating capacity 2-8 kW; "
                       "assumed quoted at B0/W35",
        "min_modulation": "same datasheet: 2 kW of 8 kW = 0.25",
        "t_sink_max_c": "Calibra Eco brochure: up to 65 degC supply line",
        "element_kw": "chosen: no resistive heat for hot water (compressor reaches 65 degC)",
        "cycling_penalty_kwh": "chosen: starts counted, not priced"})


def calibra_eco_8(**over) -> HplibSpec:
    kw = dict(CALIBRA_ECO_8)
    kw["provenance"] = dict(kw["provenance"])
    kw.update(over)
    return HplibSpec.scaled_to(**kw)


@dataclass
class HPDelivery:

    heat_kwh: np.ndarray
    compressor_kwh: np.ndarray
    element_kwh: np.ndarray
    electricity_kwh: np.ndarray
    capacity_kw: np.ndarray
    cop: np.ndarray
    on: np.ndarray
    cycling: np.ndarray


def deliver(spec_arrays: Dict[str, np.ndarray], request_kwh, accept_kwh, t_source_c,
            t_sink_c) -> HPDelivery:
    s = spec_arrays
    dt = s["dt"]
    want = np.minimum(np.maximum(np.asarray(request_kwh, dtype=np.float64), 0.0),
                      np.maximum(np.asarray(accept_kwh, dtype=np.float64), 0.0))
    t_src = np.asarray(t_source_c, dtype=np.float64)
    t_snk = np.asarray(t_sink_c, dtype=np.float64)
    cop_g = s["cop_p"]
    elp = s["el_p"]
    t_amb = np.where(s["group"] == 2, HPLIB_BRINE_T_AMB_C, t_src)
    cop = np.maximum(cop_g[..., 0] * t_src + cop_g[..., 1] * t_snk + cop_g[..., 2]
                     + cop_g[..., 3] * t_amb, _COP_FLOOR)
    p_el = s["p_el_ref"] * (elp[..., 0] * t_src + elp[..., 1] * t_snk + elp[..., 2]
                            + elp[..., 3] * t_amb)
    cap = np.where(t_snk > s["t_sink_max"], 0.0, np.maximum(p_el * cop, 0.0))
    comp_max = cap * dt
    comp = np.minimum(want, comp_max)
    comp = np.where(comp_max > 0.0, comp, 0.0)
    rest = want - comp
    elem = np.where((rest > 1e-12) & (s["element_kw"] > 0.0),
                    np.minimum(rest, s["element_kw"] * dt), 0.0)
    on = comp > 1e-12
    cycling = on & (comp < s["min_mod"] * comp_max)
    elec = np.where(on, comp / cop, 0.0) + np.where(elem > 0.0, elem / s["element_eff"], 0.0)
    return HPDelivery(heat_kwh=comp + elem, compressor_kwh=comp, element_kwh=elem,
                      electricity_kwh=elec, capacity_kw=cap, cop=cop, on=on, cycling=cycling)


def spec_arrays_of(specs) -> Dict[str, np.ndarray]:
    f = lambda xs: np.array([float(x) for x in xs], dtype=np.float64)
    return {"group": np.array([int(s.group) for s in specs]),
            "cop_p": np.array([HPLIB_GENERIC[s.group]["cop"] for s in specs], dtype=np.float64),
            "el_p": np.array([HPLIB_GENERIC[s.group]["p_el"] for s in specs], dtype=np.float64),
            "p_el_ref": f(s.p_el_ref_kw for s in specs),
            "min_mod": f(s.min_modulation for s in specs),
            "t_sink_max": f(s.t_sink_max_c for s in specs),
            "element_kw": f(s.element_kw for s in specs),
            "element_eff": f(s.element_efficiency for s in specs),
            "penalty": f(s.cycling_penalty_kwh for s in specs),
            "dt": f(s.dt_hours for s in specs)}


def columns(spec_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {}
    for k, v in spec_arrays.items():
        out[k] = v[:, None, :] if v.ndim == 2 else v[:, None]
    return out


class HplibUnit:

    def __init__(self, spec: HplibSpec) -> None:
        self.spec = spec
        self._arr = spec_arrays_of([spec])
        self.total_starts = 0
        self.total_cycling_steps = 0
        self._was_on = False

    def reset(self) -> None:
        self.total_starts = 0
        self.total_cycling_steps = 0
        self._was_on = False

    def account(self, on: bool, cycling: bool) -> int:
        starts = 1 if (on and not self._was_on) or cycling else 0
        self.total_cycling_steps += int(bool(cycling))
        self._was_on = bool(on)
        self.total_starts += starts
        return starts

    def deliver(self, request_thermal_kwh: float, t_source_c: float, t_sink_c: float, *,
                allow_element: bool = True):
        from stems.heatpump_thermia import Delivery
        arr = dict(self._arr)
        if not allow_element:
            arr["element_kw"] = np.zeros(1)
        d = deliver(arr, np.array([request_thermal_kwh]), np.array([np.inf]),
                    np.array([t_source_c]), np.array([t_sink_c]))
        on, cyc = bool(d.on[0]), bool(d.cycling[0])
        starts = self.account(on, cyc)
        elec = float(d.electricity_kwh[0]) + starts * self.spec.cycling_penalty_kwh
        heat = float(d.heat_kwh[0])
        cop_raw = float(self.spec.cop_raw(t_source_c, t_sink_c))
        return Delivery(thermal_kwh=heat, electricity_kwh=elec,
                        compressor_thermal_kwh=float(d.compressor_kwh[0]),
                        element_thermal_kwh=float(d.element_kwh[0]),
                        duty=float(d.compressor_kwh[0] / max(float(d.capacity_kw[0])
                                                              * self.spec.dt_hours, 1e-12)),
                        cycling=cyc, starts=starts, cop_used=float(d.cop[0]),
                        capacity_kw=float(d.capacity_kw[0]), cop_clamped=cop_raw < _COP_FLOOR,
                        unmet_thermal_kwh=max(float(request_thermal_kwh) - heat, 0.0))

    def max_thermal_kwh(self, t_source_c: float, t_sink_c: float, *,
                        allow_element: bool = True) -> float:
        out = float(self.spec.capacity_kw(t_source_c, t_sink_c)) * self.spec.dt_hours
        if allow_element:
            out += self.spec.element_kw * self.spec.dt_hours
        return out

    def describe(self) -> Dict[str, object]:
        d = self.spec.describe()
        d.update({"total_starts": self.total_starts,
                  "total_cycling_steps": self.total_cycling_steps})
        return d
