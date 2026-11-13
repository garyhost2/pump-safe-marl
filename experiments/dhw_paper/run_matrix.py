#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stems.dhw_barrier import PHASE_DISINFECT, HourOfDayDrawForecast, PhaseBands
from stems.dhw_plant import DHWPlant, TankSpec
from stems.dhw_stratified import (COMFORT_C, DISINFECT_C, SUPPLY_DISINFECT_C,
                                  SUPPLY_NORMAL_C, DisinfectionBarrier, ForecastBundle,
                                  HourOfDayQuantileForecast, PriceAwareController,
                                  ReactiveController, StratifiedStoreModel)
from stems.heatpump_hplib import HplibUnit, brine_temperature_c, calibra_eco_8

SERIES = REPO / "results" / "dhw_paper" / "series_8759.npz"
T_COLD, T_RATED, T_MAX, T_USE = 10.0, 60.0, 65.0, 40.0
INITIAL_SOC = 0.9
PERIOD = 168
RESERVE_Q = 0.9
FLAT_PRICE_NOTE = "flat = annual time-average of the dynamic tariff"

_DATA: Dict[str, Any] = {}


def load(path=SERIES) -> Dict[str, Any]:
    z = np.load(path)
    cols = [str(c) for c in z["cols"]]
    a = z["data"]
    c = lambda n: a[:, :, cols.index(n)]
    price = c("electricity_pricing")[:, 0].astype(np.float64)
    return {"draw": c("dhw_demand").astype(np.float64),
            "t_out": c("outdoor_dry_bulb_temperature")[:, 0].astype(np.float64),
            "price_dynamic": price, "price_flat": np.full_like(price, price.mean()),
            "hour": c("hour")[:, 0], "capacity": np.asarray(z["capacity"], float),
            "loss": np.asarray(z["loss_coefficient"], float), "source": str(z["source"])}


def unit_specs(plant: str, data) -> List:
    B = data["draw"].shape[1]
    if plant == "calibra8":
        return [calibra_eco_8() for _ in range(B)]
    if plant.startswith("rule"):
        f = float(plant[len("rule"):])
        p = data["draw"].mean(axis=0) * 24.0 / 4.0 * f
        return [calibra_eco_8(p_th_max_kw=float(max(x, 0.3))) for x in p]
    raise ValueError(plant)


def build(cfg, data):
    B = data["draw"].shape[1]
    tanks = [TankSpec(capacity_kwh=float(data["capacity"][i]), t_cold_c=T_COLD,
                      t_rated_c=T_RATED, loss_coefficient=float(data["loss"][i]), nodes=2,
                      top_fraction=0.5, node_conduction=float(cfg.get("kappa", 0.0)),
                      charging="stratified", t_max_c=T_MAX, t_use_c=T_USE,
                      provenance={"capacity_kwh": "schema dhw_storage.capacity",
                                  "loss_coefficient": "schema dhw_storage"})
             for i in range(B)]
    specs = unit_specs(cfg["plant"], data)
    plants = [DHWPlant(tanks[i], HplibUnit(specs[i]), initial_soc=INITIAL_SOC)
              for i in range(B)]
    model = StratifiedStoreModel(tanks, specs, allow_element=True)
    bands = PhaseBands.from_temperatures(tanks[0], floor_c=COMFORT_C[0],
                                         ceiling_c=COMFORT_C[1], disinfect_c=T_MAX,
                                         bottom="ceiling")
    return tanks, specs, plants, model, bands


def run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = _DATA
    t0 = time.time()
    draw = data["draw"][: cfg.get("steps", data["draw"].shape[0])]
    T, B = draw.shape
    t_out = data["t_out"][:T]
    price = data["price_dynamic" if cfg["price"] == "dynamic" else "price_flat"][:T]
    tanks, specs, plants, model, bands = build(cfg, data)
    span = T_RATED - T_COLD
    fb = ForecastBundle(price, t_out, brine_temperature_c, weather=cfg["weather"] == "on",
                        seed=cfg.get("seed"), horizon=24)
    fc = HourOfDayDrawForecast(B)
    fq = HourOfDayQuantileForecast(B, q=RESERVE_Q)
    reserve = np.quantile(draw, RESERVE_Q, axis=0)
    trig = cfg["trigger"]
    btrig = {"none": "timer", "recourse_q": "recourse"}.get(trig, trig)
    barrier = DisinfectionBarrier(
        model, bands, trigger=btrig,
        use_filter=bool(cfg["filter"]), period_steps=PERIOD,
        safety_stock_kwh=reserve if trig == "recourse" else 0.0,
        jit_reserve_kwh=reserve if trig == "jit" else 0.0,
        t_supply_normal_c=SUPPLY_NORMAL_C, t_supply_disinfect_c=SUPPLY_DISINFECT_C,
        t_disinfect_c=DISINFECT_C)
    if trig == "none":
        barrier.timer_offset = 10 ** 9
    cop_fn = lambda src: specs[0].cop(src, SUPPLY_NORMAL_C)
    if cfg["controller"] == "reactive":
        ctl = ReactiveController(B)
    else:
        ctl = PriceAwareController(B, cop_fn)
    lo_c, hi_c = COMFORT_C
    acc = {k: np.zeros(B) for k in ("elec", "cost", "unmet", "served", "heat", "loss",
                                    "below_steps", "above_steps", "below_dh", "above_dh",
                                    "above_steps_nd", "elec_d", "below_post_draw")}
    windows = np.zeros((T // PERIOD, B), dtype=bool)
    bottom_peak = np.zeros((T // PERIOD + 1, B))
    max_fwd = 0.0
    viol_after_filter = 0
    unmet_steps = 0
    cop_heat = 0.0
    for t in range(T):
        E = model.energy_of(plants)
        Tn = T_COLD + E / model.caps * span
        d = draw[t]
        src = fb.source_now(t)
        need_path = cfg["controller"] == "price_aware" or trig in ("recourse", "recourse_q")
        spath = fb.source_path(t) if need_path else None
        ppath = fb.price_path(t)
        dfc = (fc.forecast(PERIOD) if trig == "recourse" else
               fq.forecast(PERIOD) if trig == "recourse_q" else None)
        if cfg["barrier_draw"] == "known":
            d_bar = d
        else:
            d_bar = (fq if trig == "recourse_q" else fc).forecast(1)[0]
        u_pol, q_req = ctl.act(Tn[:, 0], Tn[:, -1], owed_d=barrier.in_d_owed, k=barrier.k,
                               price_path=ppath, source_path=spath,
                               window_owed=barrier.owed.copy())
        if trig == "none":
            q_req = None
        u, t_sup, info = barrier.decide(u_pol, q_req, E, d_bar, src, source_path=spath,
                                        draw_forecast=dfc)
        req = model.request_kwh(u, src, t_sup)
        pred = model.step(E, u, d, src, t_supply_c=t_sup)
        steps = [p.step(float(d[i]), float(req[i]), src, t_supply_c=float(t_sup[i]))
                 for i, p in enumerate(plants)]
        E2 = model.energy_of(plants)
        max_fwd = max(max_fwd, float(np.max(np.abs(E2 - pred.energy_kwh))))
        e = np.array([s.electricity_kwh for s in steps])
        acc["elec"] += e
        acc["cost"] += e * price[t]
        acc["elec_d"] += e * (barrier.q == PHASE_DISINFECT)
        acc["unmet"] += [s.unmet_kwh for s in steps]
        acc["served"] += [s.served_kwh for s in steps]
        acc["heat"] += [s.charge_thermal_kwh for s in steps]
        acc["loss"] += [s.loss_kwh for s in steps]
        unmet_steps += int(sum(s.unmet_kwh > 1e-9 for s in steps))
        cop_heat += sum(s.charge_thermal_kwh for s in steps)
        top = np.array([s.top_temperature_c for s in steps])
        below, above = top < lo_c - 1e-9, top > hi_c + 1e-9
        acc["below_steps"] += below
        acc["above_steps"] += above
        acc["above_steps_nd"] += above & (barrier.q != PHASE_DISINFECT)
        acc["below_dh"] += np.maximum(lo_c - top, 0.0)
        acc["above_dh"] += np.maximum(top - hi_c, 0.0)
        pdn = np.array([s.post_draw_node_soc for s in steps])
        acc["below_post_draw"] += (T_COLD + pdn[:, 0] * span) < lo_c - 1e-9
        if cfg["filter"] and cfg["barrier_draw"] == "known":
            lo, hi = bands.lo_hi(barrier.q)
            s2 = E2 / model.caps
            bad = ((s2 < lo - 1e-12) | (s2 > hi + 1e-12)).any(axis=1) & ~info["empty"]
            viol_after_filter += int(bad.sum())
        w = t // PERIOD
        bottom_peak[w] = np.maximum(bottom_peak[w], T_COLD + pdn[:, -1] * span)
        ob = barrier.observe(pdn)
        if w < windows.shape[0]:
            windows[w] |= ob["completed"]
        fc.update(d)
        fq.update(d)
    rep = barrier.report()
    gaps_max, gaps_over = [], 0
    for b in range(B):
        ts = np.array(sorted(t for t, bb in barrier.hold_steps if bb == b), dtype=float)
        g = np.diff(np.concatenate([[-1.0], ts])) if ts.size else np.array([float(T)])
        gaps_max.append(float(g.max()))
        gaps_over += int((g > PERIOD).sum())
    starts = [p.unit.total_starts for p in plants]
    cyc = [p.unit.total_cycling_steps for p in plants]
    n_win = windows.size
    completed = int(windows.sum())
    out = dict(cfg)
    out.update({
        "steps": T, "buildings": B, "calendar_windows": n_win,
        "windows_completed": completed, "windows_missed": n_win - completed,
        "completion_rate": completed / n_win if n_win else None,
        "barrier_missed_windows": int(sum(rep["missed_windows"])),
        "hold_met_steps": int(sum(rep["hold_events"])),
        "max_gap_between_hold_steps_h": float(max(gaps_max)),
        "gaps_over_168h": gaps_over,
        "comfort_below_steps": int(acc["below_steps"].sum()),
        "comfort_above_steps": int(acc["above_steps"].sum()),
        "comfort_above_steps_outside_D": int(acc["above_steps_nd"].sum()),
        "comfort_below_degree_h": float(acc["below_dh"].sum()),
        "comfort_above_degree_h": float(acc["above_dh"].sum()),
        "top_below_45_after_draw_steps": int(acc["below_post_draw"].sum()),
        "unmet_kwh": float(acc["unmet"].sum()), "unmet_draw_steps": unmet_steps,
        "served_kwh": float(acc["served"].sum()), "demand_kwh": float(draw.sum()),
        "electricity_kwh": float(acc["elec"].sum()), "cost_usd": float(acc["cost"].sum()),
        "electricity_in_D_kwh": float(acc["elec_d"].sum()),
        "heat_kwh": float(acc["heat"].sum()), "standing_loss_kwh": float(acc["loss"].sum()),
        "seasonal_cop": float(acc["heat"].sum() / max(acc["elec"].sum(), 1e-12)),
        "starts": int(sum(starts)), "cycling_steps": int(sum(cyc)),
        "per_house_electricity_kwh": [round(float(x), 3) for x in acc["elec"]],
        "per_house_windows_missed": [int(x) for x in (windows.shape[0] - windows.sum(0))],
        "mean_window_bottom_peak_c": float(bottom_peak[: T // PERIOD].mean()),
        "min_window_bottom_peak_c": float(bottom_peak[: T // PERIOD].min()),
        "refusals": int(sum(rep["refusals"])), "empty_intersections": int(sum(rep["empty_steps"])),
        "empty_by_reason": rep["empty_by_reason"], "forced_steps": int(sum(rep["forced_steps"])),
        "modified_steps": int(sum(rep["modified_steps"])),
        "steps_in_D": int(sum(rep["steps_in_D"])),
        "band_violations_unreported": viol_after_filter if (cfg["filter"] and
                                                           cfg["barrier_draw"] == "known") else None,
        "max_band_residual_on_nonempty_steps": rep["max_band_residual_on_nonempty_steps"],
        "model_vs_plant_max_abs_kwh": max_fwd,
        "unit_p_el_ref_kw": [round(s.p_el_ref_kw, 5) for s in specs],
        "seconds": round(time.time() - t0, 1)})
    return out


def key(cfg) -> str:
    return "|".join(f"{k}={cfg.get(k)}" for k in ("plant", "controller", "trigger", "filter",
                                                  "barrier_draw", "price", "weather", "seed",
                                                  "kappa", "steps"))


def grid(name: str) -> List[Dict[str, Any]]:
    cfgs: List[Dict[str, Any]] = []
    seeds = [0, 1, 2, 3, 4]

    def add(**kw):
        base = dict(plant="calibra8", filter=False, barrier_draw="known", kappa=0.0,
                    steps=8759, seed=None)
        base.update(kw)
        if base["weather"] == "on" and base["seed"] is None:
            for s in seeds:
                cfgs.append(dict(base, seed=s))
        else:
            cfgs.append(base)

    if name in ("main", "smoke"):
        for price in ("dynamic", "flat"):
            add(controller="reactive", trigger="none", price=price, weather="off")
            add(controller="reactive", trigger="timer", price=price, weather="off")
            for flt in (False, True):
                add(controller="reactive", trigger="jit", price=price, weather="off", filter=flt)
                for trig in ("recourse", "recourse_q"):
                    for weather in ("off", "on"):
                        add(controller="reactive", trigger=trig, price=price,
                            weather=weather, filter=flt)
            for weather in ("off", "on"):
                add(controller="price_aware", trigger="none", price=price, weather=weather)
                for trig in ("jit", "recourse", "recourse_q"):
                    for flt in (False, True):
                        add(controller="price_aware", trigger=trig, price=price,
                            weather=weather, filter=flt)
            for ctl in ("reactive", "price_aware"):
                for trig in ("jit", "recourse", "recourse_q"):
                    add(controller=ctl, trigger=trig, price=price, weather="off", filter=True,
                        barrier_draw="forecast")
    if name == "base_filter":
        for price in ("dynamic", "flat"):
            add(controller="reactive", trigger="none", price=price, weather="off", filter=True)
            for weather in ("off", "on"):
                add(controller="price_aware", trigger="none", price=price, weather=weather,
                    filter=True)
        for plant in ("rule1", "rule2"):
            for ctl in ("reactive", "price_aware"):
                add(plant=plant, controller=ctl, trigger="none", price="dynamic",
                    weather="off", filter=True)
    if name == "power_limited":
        for plant in ("rule1", "rule2"):
            for ctl in ("reactive", "price_aware"):
                add(plant=plant, controller=ctl, trigger="none", price="dynamic", weather="off")
                for trig in (["timer"] if ctl == "reactive" else []) + ["jit", "recourse",
                                                                        "recourse_q"]:
                    for flt in ((False, True) if trig != "timer" else (False,)):
                        add(plant=plant, controller=ctl, trigger=trig, price="dynamic",
                            weather="off", filter=flt)
    if name == "smoke":
        cfgs = [dict(c, steps=336) for c in cfgs if c.get("seed") in (None, 0)]
    seen, out = set(), []
    for c in cfgs:
        if key(c) not in seen:
            seen.add(key(c))
            out.append(c)
    return out


def _init(data) -> None:
    _DATA.update(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="main")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    data = load()
    _init(data)
    cfgs = grid(args.grid)
    out = Path(args.out) if args.out else REPO / "results" / "dhw_paper" / f"{args.grid}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(key(json.loads(line)))
    todo = [c for c in cfgs if key(c) not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(cfgs)} configs, {len(done)} done, {len(todo)} to run -> {out}", flush=True)
    with mp.get_context("fork").Pool(args.workers, initializer=_init, initargs=(data,)) as pool:
        for r in pool.imap_unordered(run_one, todo, chunksize=1):
            with out.open("a") as fh:
                fh.write(json.dumps(r) + "\n")
            print(f"{r['controller']:>11} {r['trigger']:>8} filt={int(r['filter'])} "
                  f"{r['barrier_draw']:>8} {r['price']:>7} w={r['weather']:>3} s={r['seed']} "
                  f"missed={r['windows_missed']:>3} below={r['comfort_below_steps']:>5} "
                  f"unmet={r['unmet_kwh']:8.2f} elec={r['electricity_kwh']:8.1f} "
                  f"cost={r['cost_usd']:8.2f} t={r['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
