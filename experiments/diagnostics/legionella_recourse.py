#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.legionella_demo import SCHEMA, TEMPERATURES, demand_series
from stems.dhw_barrier import HourOfDayDrawForecast
from stems.legionella import HOURS_PER_WEEK, LegionellaSpec, LegionellaStack

SIZINGS = (0.25, 0.5, 1.0, 2.0)
RESERVES = (0.5, 0.75, 0.9, 0.95, 1.0)
TRIGGERS = ("jit", "recourse")


def run(spec: LegionellaSpec, demand: np.ndarray, cycle_on: bool, trigger: str = "jit",
        safety_stock: Any = 0.0) -> Dict[str, Any]:
    forecaster = HourOfDayDrawForecast(spec.B) if trigger == "recourse" else None
    stack = LegionellaStack.build(spec, action_index=0, trigger=trigger,
                                  forecaster=forecaster, safety_stock_kwh=safety_stock)
    B = spec.B
    per_step = spec.heat_source_kw * spec.dt_hours
    forced = np.zeros(B, dtype=np.int64)
    events = np.zeros(B, dtype=np.int64)
    prev = np.zeros(B, dtype=bool)
    for t in range(demand.shape[0]):
        soc = stack.tank.soc
        need_kwh = np.maximum(spec.soc_normal - soc, 0.0) * spec.capacity_kwh + demand[t]
        thermostat = np.clip(need_kwh / np.maximum(per_step, 1e-9), 0.0, 1.0)
        actions = np.zeros((B, 1), dtype=np.float32)
        actions[:, 0] = thermostat.astype(np.float32)
        if cycle_on:
            before = actions[:, 0].copy()
            actions = stack.load.project(actions, [np.zeros(40, dtype=np.float32)] * B)
            forced += (actions[:, 0] > before).astype(np.int64)
        done = np.asarray(stack.observe(actions[:, 0], demand[t])["completed"], dtype=bool)
        events += (done & ~prev).astype(np.int64)
        prev = done
    rep = stack.report()
    rep["forced_steps"] = forced.tolist()
    rep["completion_events"] = events.tolist()
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=336)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from stems.environment import STEMSEnvironment

    kw = {"env_kwargs": {"episode_time_steps": [(0, args.steps)]}} if args.steps < 8759 else {}
    env = STEMSEnvironment(schema=SCHEMA, seed=0, heat_pump=True, **kw)
    capacity = np.asarray(env.dhw_info()["capacity"], dtype=np.float64)
    loss = np.asarray(env.dhw_info()["loss_coefficient"], dtype=np.float64)
    demand = demand_series(env, args.steps)
    mean_daily = demand.mean(axis=0) * 24.0
    base_p_hp = mean_daily / 4.0
    quant = {q: np.quantile(demand, q, axis=0) for q in RESERVES}

    def make(m: float, margin: np.ndarray) -> LegionellaSpec:
        spec = LegionellaSpec(capacity_kwh=capacity, heat_source_kw=base_p_hp * m,
                              loss_coefficient=loss, draw_margin_kwh=margin,
                              period_hours=HOURS_PER_WEEK, **TEMPERATURES)
        spec.provenance["capacity_kwh"] = f"schema {SCHEMA}, dhw_storage.capacity"
        spec.provenance["loss_coefficient"] = f"schema {SCHEMA}, dhw_storage"
        return spec

    rows: List[Dict[str, Any]] = []
    for m in SIZINGS:
        r = run(make(m, np.zeros(env.num_buildings)), demand, cycle_on=False)
        rows.append({"trigger": "none", "sizing": m, "reserve_q": None, **_summ(r)})
        for q in RESERVES:
            for trig in TRIGGERS:
                base = {"trigger": trig, "sizing": m, "reserve_q": q,
                        "reserve_kwh": [round(float(x), 4) for x in quant[q]]}
                try:
                    if trig == "jit":
                        spec = make(m, quant[q])
                        r = run(spec, demand, True, "jit")
                    else:
                        spec = make(m, np.zeros(env.num_buildings))
                        r = run(spec, demand, True, "recourse", safety_stock=quant[q])
                except ValueError as exc:
                    rows.append({**base, "refused": str(exc)})
                    continue
                rows.append({**base, **_summ(r)})

    verdict = {}
    for trig in TRIGGERS:
        for q in RESERVES:
            seq = [next((x for x in rows if x["trigger"] == trig and x["sizing"] == m
                         and x["reserve_q"] == q), None) for m in SIZINGS]
            ok = [x for x in seq if x is not None and "refused" not in x]
            pick = lambda key: [None if (x is None or "refused" in x) else x[key] for x in seq]
            missed = [x["missed"] for x in ok]
            events = [x["completion_events"] for x in ok]
            verdict[f"{trig}_q{q}"] = {
                "missed_by_sizing": pick("missed"),
                "events_by_sizing": pick("completion_events"),
                "raw_completions_by_sizing": pick("completed"),
                "event_rate_by_sizing": pick("event_rate"),
                "monotone_missed": all(a >= b for a, b in zip(missed, missed[1:])),
                "monotone_events": all(a <= b for a, b in zip(events, events[1:])),
                "refused_sizings": [m for m, x in zip(SIZINGS, seq)
                                    if x is None or "refused" in x]}

    out = Path(args.out) if args.out else (REPO / "experiments" / "diagnostics" /
                                           "legionella" / f"recourse_{args.steps}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "schema": SCHEMA, "steps": int(demand.shape[0]), "buildings": int(demand.shape[1]),
        "mean_daily_demand_kwh": [round(float(d), 4) for d in mean_daily],
        "temperatures_are_assumptions_of_this_diagnostic": TEMPERATURES,
        "reserve_quantiles_are_over_the_whole_run": True,
        "forecast": "HourOfDayDrawForecast (causal, expanding hour-of-day mean)",
        "runs": rows, "monotone_in_source_size": verdict}, indent=2) + "\n")

    print(f"{demand.shape[0]} steps x {demand.shape[1]} buildings")
    head = (f"{'trigger':>9}{'q':>6}{'sizing':>8}{'events':>8}{'raw':>6}{'missed':>8}"
            f"{'rate':>7}{'unmet kWh':>11}{'elec kWh':>10}")
    print(head)
    print("-" * len(head))
    for x in rows:
        q = "-" if x["reserve_q"] is None else x["reserve_q"]
        lead = f"{x['trigger']:>9}{q:>6}{x['sizing']:>8}"
        if "refused" in x:
            print(lead + "   REFUSED (reserve >= source output per step)")
            continue
        print(lead + f"{x['completion_events']:>8}{x['completed']:>6}{x['missed']:>8}"
              f"{x['event_rate']:>7.3f}{x['unmet_kwh']:>11.3f}{x['electricity_kwh']:>10.2f}")
    print("\nmissed windows by sizing (refused sizings shown as None):")
    for k, v in verdict.items():
        print(f"  {k:>16}: missed {v['missed_by_sizing']} monotone={v['monotone_missed']}"
              f" | events {v['events_by_sizing']} monotone={v['monotone_events']}")
    print(f"written to {out}")


def _summ(r: Dict[str, Any]) -> Dict[str, Any]:
    c, s = r["cycle"], r["service"]
    done, missed = int(sum(c["cycles_completed"])), int(sum(c["missed_windows"]))
    ev = int(sum(r.get("completion_events", c["cycles_completed"])))
    return {"completed": done, "missed": missed, "completion_events": ev,
            "completion_rate": done / (done + missed) if done + missed else float("nan"),
            "event_rate": ev / (ev + missed) if ev + missed else float("nan"),
            "events_per_building": r.get("completion_events"),
            "completed_per_building": c["cycles_completed"],
            "missed_per_building": c["missed_windows"],
            "unmet_kwh": s["unmet_kwh"], "unmet_draws": s["unmet_draws"],
            "electricity_kwh": s["electricity_kwh"], "spill_kwh": s["spill_kwh"],
            "forced_steps": int(sum(r.get("forced_steps", [0])))}


if __name__ == "__main__":
    main()
