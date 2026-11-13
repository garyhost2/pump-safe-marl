from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from stems.dhw_v2.data import DistrictData, HouseData
from stems.dhw_v2.env import DHWDistrictEnv, EnvConfig
from stems.dhw_v2.filter import FilterConfig
from stems.dhw_v2.plant import PlantConfig

TX_MEAN_PRICE = 0.27313734444571297
BASE_WEIGHTS = {"w_comfort": 1.0, "w_sh": 0.5, "w_dis": 5.0}


def slice_hours(data: DistrictData, start: int, stop: int) -> DistrictData:
    stop = min(int(stop), data.T)
    hs = []
    for h in data.houses:
        k = 60 // h.draw_dt_min
        b = dict(h.building) if h.building else None
        if b is not None and "heating_demand_kwh" in b:
            b["heating_demand_kwh"] = np.asarray(b["heating_demand_kwh"])[start:stop]
        hs.append(HouseData(h.id, h.draw_kwh[start * k:stop * k], h.draw_dt_min, h.t_cold_c[start:stop], b,
                            h.tank_alt, dict(h.meta)))
    meta = dict(data.meta)
    meta["slice_hours"] = [int(start), int(stop)]
    return DistrictData(data.name, data.source, data.time_utc[start:stop], data.hour_of_day[start:stop],
                        data.day_of_week[start:stop], data.t_out_c[start:stop], data.ghi_w_m2[start:stop],
                        data.price[start:stop], data.currency, data.t_ground_c, hs, meta,
                        parent=data.parent if data.parent is not None else data,
                        parent_offset=data.parent_offset + int(start))


def texas_split():
    from stems.dhw_v2.data import load_texas
    d = load_texas()
    cut = 26 * 168
    assert d.day_of_week[cut] == 0 and d.hour_of_day[cut] == 0
    return slice_hours(d, 0, cut), slice_hours(d, cut, d.T), {"dataset": "texas", "train_hours": [0, cut],
                                                                "eval_hours": [cut, d.T]}


def package_split(path: str):
    from stems.dhw_v2.data import load_package
    d = load_package(path)
    years = d.time_utc.astype("datetime64[Y]").astype(int) + 1970
    last = int(years.max())
    t0 = int(np.flatnonzero(years == last)[0])
    while not (d.day_of_week[t0] == 0 and d.hour_of_day[t0] == 0):
        t0 += 1
    return slice_hours(d, 0, t0), slice_hours(d, t0, d.T), {
        "dataset": "sweden", "package": str(path), "years": sorted(set(int(y) for y in years)),
        "train_hours": [0, t0], "eval_hours": [t0, d.T], "eval_year": last}


@dataclass
class TrackConfig:
    tank_nodes: Optional[int] = None
    dt_min: int = 5
    plant_kwargs: Dict = field(default_factory=dict)
    filter_kwargs: Dict = field(default_factory=dict)
    env_kwargs: Dict = field(default_factory=dict)

    def plant(self) -> PlantConfig:
        kw = dict(dt_min=self.dt_min)
        kw.update(self.plant_kwargs)
        return PlantConfig(**kw)

    def filter(self) -> FilterConfig:
        return FilterConfig(**self.filter_kwargs)


SEK_PER_USD = 10.28


@dataclass
class RewardSpec:
    w_comfort_h: float = 20.0
    w_start: float = 0.1
    w_dis: float = 50.0
    w_sh: float = 5.0

    def fx(self, data: DistrictData) -> float:
        return 1.0 / SEK_PER_USD if data.currency.upper() == "USD" else 1.0

    def env_weights(self, data: DistrictData) -> Dict[str, float]:
        f = self.fx(data)
        return {"w_comfort": 0.0, "w_sh": self.w_sh * f, "w_dis": self.w_dis * f}

    def extra(self, data: DistrictData):
        f = self.fx(data)
        wc, ws = self.w_comfort_h * f, self.w_start * f

        def pen(info):
            rec = info["record"]
            return wc * np.asarray(rec["below_floor"], float) + ws * np.asarray(rec["starts"], float)
        return pen

    def objective(self, m: Dict[str, float], data_currency: str) -> float:
        f = 1.0 / SEK_PER_USD if data_currency.upper() == "USD" else 1.0
        return float(m["cost"] + f * (self.w_comfort_h * m["viol_intra_hour"] + self.w_start * m["starts"]
                                      + self.w_dis * m["weeks_missed"] + self.w_sh * m["sh_degree_hours"]))


def weights_for(data: DistrictData) -> Dict[str, float]:
    s = float(np.mean(data.price)) / TX_MEAN_PRICE
    return {k: v * s for k, v in BASE_WEIGHTS.items()}


def make_env(data: DistrictData, track: TrackConfig, *, filtered: bool, k_correction: float,
             weights: Optional[Dict[str, float]] = None, keep_log: bool = True) -> DHWDistrictEnv:
    w = weights or weights_for(data)
    kw = dict(plant=track.plant(), filter=track.filter() if filtered else None, filter_model="oracle",
              k_correction=float(k_correction), keep_log=keep_log, **w)
    if track.tank_nodes is not None:
        kw["tank_nodes"] = int(track.tank_nodes)
    kw.update(track.env_kwargs)
    return DHWDistrictEnv(data, EnvConfig(**kw))


ARMS = {
    "a": {"filtered": True, "k": 1.0, "label": "behind filter + correction penalty (k=1)"},
    "b": {"filtered": True, "k": 0.0, "label": "behind filter, no penalty"},
    "c": {"filtered": False, "k": 0.0, "label": "unshielded training, filter at deployment"},
}


from stems.dhw_v2.env import OBS_NAMES as _OBS_NAMES
SH_NEED_COL = [n for n, _ in _OBS_NAMES].index("sh_need")


def to_env_action(a: np.ndarray, X: np.ndarray, mode: str = "need_relative") -> np.ndarray:
    a = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0).copy()
    n = np.clip(np.asarray(X[:, SH_NEED_COL], dtype=np.float64), 0.0, None)
    if mode == "need_relative":
        a[:, 1] = np.clip(2.0 * a[:, 1] * n, 0.0, 1.0)
    elif mode.startswith("need_band"):
        a[:, 1] = np.clip((0.75 + 0.5 * a[:, 1]) * n, 0.0, 1.0)
    if mode.endswith("_nodis"):
        a[:, 2] = 0.0
    return a


class ObsBuilder:

    def __init__(self, env: DHWDistrictEnv):
        self.env = env
        self.N = env.cfg.tank_nodes
        self.prev = np.zeros((env.B, 2))

    @property
    def dim(self) -> int:
        return 24 + 11 + self.N + 2

    def reset(self):
        self.prev = np.zeros((self.env.B, 2))

    def _forecaster(self):
        e = self.env
        if e.filter is not None and e.cfg.filter.forecast == "hod_mean":
            return e.filter.forecaster
        return e.shadow_fc

    def build(self, obs: np.ndarray) -> np.ndarray:
        e = self.env
        B = e.B
        if e.t >= e.data.T:
            return np.zeros((B, self.dim), np.float32)
        p = e.price_path(24) / e.mean_price
        rank = float(np.mean(p < p[0]))
        tmin = float(np.argmin(p)) / 24.0
        fc = self._forecaster()
        hour = int(e.data.hour_of_day[e.t])
        d1, d2, d3 = (fc.forecast(hour + k) for k in (1, 2, 3))
        d4 = sum(fc.forecast(hour + k) for k in range(4, 13))
        ctx = e.plant.context()
        cop = np.asarray(ctx["r_d"]) / np.maximum(np.asarray(ctx["p_d"]), 1e-6)
        extra = [np.full(B, p.mean()), np.full(B, p.min()), np.full(B, p.max()), np.full(B, rank), np.full(B, tmin),
                 d1, d2, d3, d4, cop / 4.0, np.asarray(ctx["p_d"]) / 5.0]
        X = np.concatenate([obs, np.stack(extra, 1), e.plant.T / 60.0, self.prev], axis=1)
        return X.astype(np.float32)

    def after_step(self, info):
        ex = info["executed"]
        self.prev = np.stack([np.asarray(ex["u_d"], float), np.asarray(ex["u_s"], float)], 1)


BANDS = ((20.0, 23.0), (20.0, 22.0), (21.0, 22.0))


def run_episode(env: DHWDistrictEnv, act: Callable, *, update: Optional[Callable] = None) -> Dict:
    ob = ObsBuilder(env)
    obs, _ = env.reset()
    ob.reset()
    X = ob.build(obs)
    keys = ("heat_dhw", "heat_dhw_filter", "elec_dhw", "elec_dhw_filter", "cost_dhw", "dis_hours",
            "dis_forced", "dis_prop_hours", "end_viol", "intra_viol") + tuple(
            f"band_{lo:g}_{hi:g}_{w}" for lo, hi in BANDS for w in ("below", "above"))
    acc = {k: np.zeros(env.B) for k in keys}
    t0 = time.time()
    while True:
        a = act(env, X)
        if isinstance(a, dict):
            obs, r, term, _, info = env.step_command(a)
        else:
            obs, r, term, _, info = env.step(a)
        rec = info["record"]
        pr, ex = info["proposed"], info["executed"]
        ud_p = np.clip(np.asarray(pr["u_d"], float), 0, 1)
        ud_e = np.asarray(ex["u_d"], float)
        extra = np.where(ud_e > 1e-9, np.maximum(ud_e - ud_p, 0.0) / np.maximum(ud_e, 1e-9), 0.0)
        acc["heat_dhw"] += rec["heat_dhw_kwh"]
        acc["heat_dhw_filter"] += rec["heat_dhw_kwh"] * extra
        acc["elec_dhw"] += rec["elec_dhw_kwh"]
        acc["elec_dhw_filter"] += rec["elec_dhw_kwh"] * extra
        acc["cost_dhw"] += rec["elec_dhw_kwh"] * float(env.data.price[info["t"]])
        dsu = np.asarray(rec["dis_supply_used"], bool)
        dprop = np.asarray(pr.get("dis", np.zeros(env.B, bool)), bool)
        acc["dis_hours"] += dsu
        acc["dis_forced"] += dsu & ~dprop
        acc["dis_prop_hours"] += dprop
        acc["end_viol"] += np.asarray(rec["top_end_c"]) < env.cfg.plant.t_floor_c - 1e-9
        acc["intra_viol"] += np.asarray(rec["below_floor"], bool)
        tin = np.asarray(rec["t_in"], float)
        for lo, hi in BANDS:
            acc[f"band_{lo:g}_{hi:g}_below"] += np.maximum(lo - tin, 0.0)
            acc[f"band_{lo:g}_{hi:g}_above"] += np.maximum(tin - hi, 0.0)
        if update is not None:
            update(rec)
        ob.after_step(info)
        if term:
            break
        X = ob.build(obs)
    S = env.summary()
    S["wall_episode_s"] = time.time() - t0
    S["track"] = {k: v.astype(float).tolist() for k, v in acc.items()}
    return S


def metrics(S: Dict) -> Dict[str, float]:
    tr = S["track"]
    sm = lambda x: float(np.sum(x))
    B = len(S["houses"])
    out = {
        "cost": sm(S["cost"]),
        "objective": -sm(S["reward"]),
        "elec_kwh": sm(S["elec_kwh"]), "elec_dhw_kwh": sm(S["elec_dhw_kwh"]),
        "elec_sh_kwh": sm(S["elec_sh_kwh"]), "elec_aux_kwh": sm(S.get("elec_aux_kwh", 0.0)),
        "aux_hours": sm(S.get("aux_hours", 0.0)), "t_in_min_c": float(np.min(S["t_in_min_c"])),
        "cost_dhw": sm(tr["cost_dhw"]),
        "viol_end_of_hour": sm(tr["end_viol"]), "viol_intra_hour": sm(tr["intra_viol"]),
        "time_below_floor_h": sm(S["time_below_floor_h"]),
        "weeks_total": int(S["weeks_full"]) * B,
        "weeks_missed": int(S["weeks_full"]) * B - int(np.sum(S["weeks_completed"])),
        "starts": sm(S["starts"]), "sh_degree_hours": sm(S["sh_degree_hours_below_min"]),
        "heat_dhw_kwh": sm(tr["heat_dhw"]),
        "filter_share_dhw_heat": sm(tr["heat_dhw_filter"]) / max(sm(tr["heat_dhw"]), 1e-9),
        "filter_share_dhw_elec": sm(tr["elec_dhw_filter"]) / max(sm(tr["elec_dhw"]), 1e-9),
        "dis_supply_hours": sm(tr["dis_hours"]),
        "filter_share_disinfection": sm(tr["dis_forced"]) / max(sm(tr["dis_hours"]), 1e-9),
        "hours_extrap_supply": sm(S["hours_extrap_supply"]),
        "tank_nodes": int(S.get("tank_nodes", 0)),
    }
    for lo, hi in BANDS:
        for w in ("below", "above"):
            k = f"band_{lo:g}_{hi:g}_{w}"
            if k in tr:
                out[k + "_kh"] = sm(tr[k])
    if "intervention_rate" in S:
        out["intervention_rate"] = float(np.mean(S["intervention_rate"]))
        out["violations_unexplained"] = sm(S["violations_unexplained"])
        out["declared_infeasible_comfort_hours"] = sm(S["declared_infeasible_comfort_hours"])
        for k in ("violations_power_limited", "recursive_feasibility_failures", "jit_forced_hours",
                  "violations_explained_by_forecast", "violations_declared_infeasible"):
            if S.get(k) is not None:
                out[k] = sm(S[k])
    else:
        out["intervention_rate"] = float("nan")
    return out


def dumps(o) -> str:
    return json.dumps(o, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))


from stems.paths import SWEDEN_PACKAGES as _SWEDEN_PACKAGES
SWEDEN_PACKAGES = str(_SWEDEN_PACKAGES)


def concat_years(parts):
    import numpy as _np
    d0 = parts[0]
    hs = []
    for j, h in enumerate(d0.houses):
        assert all(p.houses[j].id == h.id and p.houses[j].draw_dt_min == h.draw_dt_min for p in parts)
        b = dict(h.building) if h.building else None
        if b is not None and "heating_demand_kwh" in b:
            b["heating_demand_kwh"] = _np.concatenate([_np.asarray(p.houses[j].building["heating_demand_kwh"]) for p in parts])
        hs.append(HouseData(h.id, _np.concatenate([p.houses[j].draw_kwh for p in parts]), h.draw_dt_min,
                            _np.concatenate([p.houses[j].t_cold_c for p in parts]), b, h.tank_alt, dict(h.meta)))
    cat = lambda k: _np.concatenate([getattr(p, k) for p in parts])
    meta = dict(d0.meta)
    meta["parts"] = [p.name for p in parts]
    return DistrictData("+".join(p.name for p in parts), d0.source, cat("time_utc"), cat("hour_of_day"),
                        cat("day_of_week"), cat("t_out_c"), cat("ghi_w_m2"), cat("price"), d0.currency,
                        d0.t_ground_c, hs, meta)


def load_site(site: str, years, root: str = SWEDEN_PACKAGES, price_area: Optional[str] = None):
    from pathlib import Path
    from stems.dhw_v2.data import load_package
    parts, bh = [], None
    for y in years:
        p = Path(root) / f"{site}_{y}"
        parts.append(load_package(p))
        man = json.loads((p / "manifest.json").read_text())
        bh = man.get("borehole")
    d = concat_years(parts) if len(parts) > 1 else parts[0]
    if price_area is not None:
        from stems.dhw_v2.data import apply_price_area
        d = apply_price_area(d, price_area)
    d.meta["borehole"] = bh
    return d, bh


def window_starts(data: DistrictData, hours: int):
    idx = np.flatnonzero((np.asarray(data.day_of_week) == 0) & (np.asarray(data.hour_of_day) == 0))
    return idx[idx + hours <= data.T]


def make_env_v2(data: DistrictData, track: TrackConfig, spec: "RewardSpec", *, filtered: bool,
                k_correction: float, borehole: Optional[Dict] = None, price_ref: Optional[float] = None,
                keep_log: bool = True):
    import functools
    import stems.dhw_v2.env as envmod
    from stems.dhw_v2.borehole import BoreholeSpec
    w = spec.env_weights(data)
    bspec = BoreholeSpec(t_ground_c=data.t_ground_c, **borehole) if borehole else None

    class _Env(DHWDistrictEnv):
        def reset(self, seed=None):
            orig = envmod.DistrictPlant
            if bspec is not None:
                envmod.DistrictPlant = functools.partial(orig, borehole=bspec)
            try:
                out = super().reset(seed)
            finally:
                envmod.DistrictPlant = orig
            if price_ref is not None:
                self.mean_price = float(price_ref)
                out = (self.observe(), out[1])
            return out

    kw = dict(plant=track.plant(), filter=track.filter() if filtered else None, filter_model="oracle",
              k_correction=float(k_correction), keep_log=keep_log, **w)
    if track.tank_nodes is not None:
        kw["tank_nodes"] = int(track.tank_nodes)
    kw.update(track.env_kwargs)
    env = _Env(data, EnvConfig(**kw))
    env.borehole_spec_used = dict(borehole) if borehole else None
    return env
