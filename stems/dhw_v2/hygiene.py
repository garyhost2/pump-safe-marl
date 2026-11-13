from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Sequence, Union

import numpy as np

__all__ = ["HygieneSpec", "REGIMES", "SCORED", "get_regime", "regime_overrides", "period_calendar",
           "HygieneScorer", "ThermiaRuleScorer"]

_VOLUME_SERIES = {"bottom20": "bot", "whole": "nmin", "whole_mean": "mean"}
_VOLUME_HOLD_RULE = {"bottom20": "bottom20", "whole": "all_nodes", "whole_mean": "whole_mean"}


@dataclass(frozen=True)
class HygieneSpec:
    name: str
    label: str
    t_c: Optional[float] = None
    hold_min: float = 0.0
    volume: str = "bottom20"
    period_h: Optional[int] = None
    t_supply_c: float = 65.0
    tol_k: float = 0.0
    t_tap_min_c: float = 50.0
    source: str = ""

    def __post_init__(self):
        if self.volume not in _VOLUME_SERIES:
            raise ValueError(f"volume must be one of {sorted(_VOLUME_SERIES)}")
        if self.t_c is not None and self.period_h not in (24, 168):
            raise ValueError("period_h must be 24 or 168 for a regime with a hold")

    @property
    def threshold_c(self) -> Optional[float]:
        return None if self.t_c is None else self.t_c - self.tol_k

    def hold_steps(self, dt_min: float) -> int:
        return max(1, int(np.ceil(self.hold_min / dt_min - 1e-9)))


REGIMES: Dict[str, HygieneSpec] = {s.name: s for s in (
    HygieneSpec("none", "no periodic hold; >= 50 degC at the tap (BBR)",
                source="Boverket BFS 2024:8 ch. 8 par. 6: >= 50 degC at the tap, 60 degC only recommended"),
    HygieneSpec("weekly_bottom20_60_1h", "weekly, bottom 20 % at 60 degC for 1 h", 60.0, 60.0, "bottom20", 168, 65.0,
                source="manufacturer convention (NZ G12/AS2 3.5.1(c) reading); not required by any code "
                       "for a single-family house"),
    HygieneSpec("daily_whole_60_1h", "daily, whole tank at 60 degC for 1 h", 60.0, 60.0, "whole", 24, 65.0,
                source="HSE HSG274 Part 2 calorifier style (workplaces and landlords, not owner-occupiers)"),
    HygieneSpec("w551_70_3min", "whole tank at 70 degC for 3 min (weekly period ASSUMED)", 70.0, 3.0, "whole", 168, 70.0,
                source="DVGW W 551 remedial flush (>= 70 degC, 3 min at every tap), read through secondary "
                       "sources; W 551 gives no period"),
)}

SCORED = tuple(REGIMES.values()) + (replace(REGIMES["w551_70_3min"], name="w551_70_3min_tol1K", tol_k=1.0,
                                            label="as w551_70_3min with a 1 K tolerance (>= 69 degC)"),)


def get_regime(regime: Union[str, HygieneSpec], tol_k: Optional[float] = None) -> HygieneSpec:
    spec = regime if isinstance(regime, HygieneSpec) else REGIMES[regime]
    return spec if tol_k is None else replace(spec, tol_k=float(tol_k))


def regime_overrides(regime: Union[str, HygieneSpec], tol_k: Optional[float] = None) -> Dict[str, Dict]:
    s = get_regime(regime, tol_k)
    if s.t_c is None:
        return {"plant": {}, "filter": {"disinfection": False},
                "env": {"hygiene": False, "regime": s.name}}
    return {"plant": {"t_dis_c": s.threshold_c, "t_supply_dis_c": s.t_supply_c,
                      "hold_rule": _VOLUME_HOLD_RULE[s.volume], "hold_min": s.hold_min},
            "filter": {},
            "env": {"period_h": int(s.period_h), "hygiene": True,
                    "regime": s.name + (f"~tol{s.tol_k:g}K" if s.tol_k else "")}}


def period_calendar(hour_of_day, day_of_week, period_h: int = 168):
    hod, dow = np.asarray(hour_of_day), np.asarray(day_of_week)
    if period_h == 168:
        start = (dow == 0) & (hod == 0)
    elif period_h == 24:
        start = hod == 0
    else:
        raise ValueError("period_h must be 24 or 168")
    pid = np.cumsum(start) - (1 if start[0] else 0)
    pid = pid - pid[0]
    T = len(pid)
    left = np.zeros(T, dtype=int)
    nxt = T
    for t in range(T - 1, -1, -1):
        if t + 1 < T and start[t + 1]:
            nxt = t + 1
        left[t] = nxt - t
    return pid, left


class HygieneScorer:

    def __init__(self, B: int, dt_min: float, hour_of_day, day_of_week,
                 specs: Sequence[HygieneSpec] = SCORED):
        self.B, self.dt_min = int(B), float(dt_min)
        self.specs = [s for s in specs]
        self.T = len(hour_of_day)
        self.st = {}
        for s in self.specs:
            if s.t_c is None:
                continue
            pid, _ = period_calendar(hour_of_day, day_of_week, s.period_h)
            self.st[s.name] = {"pid": pid, "run": np.zeros(B, int), "need": s.hold_steps(dt_min),
                               "done": np.zeros(B, bool), "hours_in_period": 0, "periods_full": 0,
                               "periods_completed": np.zeros(B, int), "completions": np.zeros(B, int),
                               "last": np.full(B, np.nan), "max_gap_h": np.zeros(B),
                               "steps_at_or_above": np.zeros(B, int)}
        self.draw_hours = np.zeros(B, int)
        self.tap_below = np.zeros(B, int)
        self.tap_min = np.full(B, np.inf)
        self.hours = 0

    def update(self, t: int, rec: Dict) -> None:
        ser = rec["series"]
        S = next(iter(ser.values())).shape[0]
        for s in self.specs:
            if s.t_c is None:
                continue
            st = self.st[s.name]
            ok = np.asarray(ser[_VOLUME_SERIES[s.volume]]) >= s.threshold_c - 1e-9
            for k in range(S):
                st["run"] = np.where(ok[k], st["run"] + 1, 0)
                ev = st["run"] == st["need"]
                if ev.any():
                    now = t + (k + 1) / S
                    gap = np.where(np.isnan(st["last"]), now, now - st["last"])
                    st["max_gap_h"] = np.where(ev, np.maximum(st["max_gap_h"], gap), st["max_gap_h"])
                    st["last"] = np.where(ev, now, st["last"])
                    st["completions"] += ev
                    st["done"] |= ev
            st["steps_at_or_above"] += ok.sum(axis=0)
            st["hours_in_period"] += 1
            if t + 1 >= self.T or st["pid"][t + 1] != st["pid"][t]:
                if st["hours_in_period"] >= s.period_h:
                    st["periods_full"] += 1
                    st["periods_completed"] += st["done"]
                st["done"] = np.zeros(self.B, bool)
                st["hours_in_period"] = 0
        drawn = np.asarray(rec["served_kwh"]) > 1e-12
        deliv = np.asarray(rec["delivered_min_c"], dtype=np.float64)
        tap = self.specs[0].t_tap_min_c
        self.draw_hours += drawn
        self.tap_below += drawn & (deliv < tap - 1e-9)
        self.tap_min = np.where(drawn, np.fmin(self.tap_min, deliv), self.tap_min)
        self.hours += 1

    def summary(self) -> Dict[str, object]:
        out = {"_note": "OUR criteria, scored on post-draw sub-step states; a miss here is not a Thermia failure",
               "dt_min": self.dt_min, "hours": self.hours}
        for s in self.specs:
            d = {"label": s.label, "t_c": s.t_c, "hold_min": s.hold_min, "volume": s.volume,
                 "period_h": s.period_h, "tol_k": s.tol_k, "source": s.source}
            if s.t_c is not None:
                st = self.st[s.name]
                full = int(st["periods_full"])
                d.update(periods_full=full, periods_completed=st["periods_completed"].tolist(),
                         periods_missed=(full - st["periods_completed"]).tolist(),
                         holds_completed=st["completions"].tolist(),
                         max_gap_h=np.where(st["completions"] > 0, st["max_gap_h"], np.nan).tolist(),
                         hours_at_or_above=(st["steps_at_or_above"] * self.dt_min / 60.0).tolist())
            out[s.name] = d
        out["tap_50c"] = {"t_tap_min_c": self.specs[0].t_tap_min_c, "draw_hours": self.draw_hours.tolist(),
                          "draw_hours_delivered_below": self.tap_below.tolist(),
                          "min_delivered_c": np.where(np.isfinite(self.tap_min), self.tap_min, np.nan).tolist(),
                          "note": "delivered = coldest store water a sub-step's draw reached, before any mixing "
                                  "valve or pipe loss"}
        return out


class ThermiaRuleScorer:

    SENSORS = ("top", "middle", "weighted")

    def __init__(self, B: int, dt_min: float, *, target_c: float = 60.0, limit_h: float = 7.0,
                 weights: Sequence[float] = (0.5, 0.5), sensor_height: Optional[float] = None):
        self.B, self.dt_min = int(B), float(dt_min)
        self.target_c, self.limit_h = float(target_c), float(limit_h)
        w = np.asarray(weights, dtype=np.float64)
        self.weights = (w / w.sum()).tolist()
        self.sensor_height = sensor_height
        self.limit_steps = int(round(limit_h * 60.0 / dt_min))
        self.prev_used = np.zeros(B, bool)
        z = lambda dtype=int: np.zeros(B, dtype)
        self.st = {k: {"open": z(bool), "elapsed": z(), "programs": z(), "done": z(), "alarms": z(),
                       "at_target_at_start": z(), "sum_min": z(float), "max_min": z(float)} for k in self.SENSORS}

    def update(self, t: int, rec: Dict) -> None:
        used = np.asarray(rec["dis_supply_used"], bool)
        edge = used & ~self.prev_used
        self.prev_used = used
        ser = rec["series"]
        top, mid = np.asarray(ser["top"]), np.asarray(ser["sensor"])
        x = {"top": top, "middle": mid, "weighted": self.weights[0] * top + self.weights[1] * mid}
        S = top.shape[0]
        for k in self.SENSORS:
            st = self.st[k]
            start = edge & ~st["open"]
            st["programs"] += start
            st["open"] |= start
            st["elapsed"] = np.where(start, 0, st["elapsed"])
            st["at_target_at_start"] += start & (x[k][0] >= self.target_c - 1e-9)
            if not st["open"].any():
                continue
            for s in range(S):
                hit = st["open"] & (x[k][s] >= self.target_c - 1e-9)
                minutes = st["elapsed"] * self.dt_min
                st["done"] += hit
                st["sum_min"] += np.where(hit, minutes, 0.0)
                st["max_min"] = np.where(hit, np.maximum(st["max_min"], minutes), st["max_min"])
                st["open"] &= ~hit
                st["elapsed"] = np.where(st["open"], st["elapsed"] + 1, st["elapsed"])
                late = st["open"] & (st["elapsed"] >= self.limit_steps)
                st["alarms"] += late
                st["open"] &= ~late

    def summary(self) -> Dict[str, object]:
        out = {"_note": "Thermia's own rule (Manal Rahal, 2026-11-02): done when the ending sensor reaches the "
                        "target, no hold, within the limit; alarm otherwise. The ending sensor is NOT known: "
                        "all three candidates are reported. Weights of 'weighted' are an assumption.",
               "target_c": self.target_c, "limit_h": self.limit_h, "weights_top_middle": self.weights,
               "middle_sensor_height": self.sensor_height}
        for k in self.SENSORS:
            st = self.st[k]
            n = np.maximum(st["done"], 1)
            out[k] = {"programs": st["programs"].tolist(), "done_within_limit": st["done"].tolist(),
                      "alarms": st["alarms"].tolist(), "open_at_end": st["open"].astype(int).tolist(),
                      "at_target_at_start": st["at_target_at_start"].tolist(),
                      "mean_minutes_to_target": np.where(st["done"] > 0, st["sum_min"] / n, np.nan).tolist(),
                      "max_minutes_to_target": np.where(st["done"] > 0, st["max_min"], np.nan).tolist()}
        return out
