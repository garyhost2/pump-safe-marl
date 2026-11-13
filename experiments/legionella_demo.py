#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stems.environment import STEMSEnvironment
from stems.legionella import HOURS_PER_WEEK, LegionellaSpec, LegionellaStack, ShadowTank
from stems.observations import obs_index

OUT = REPO / "experiments" / "diagnostics" / "legionella" / "demo.json"
SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
STEPS = 336
IDX_DHW_DEMAND = obs_index("dhw_demand")

TEMPERATURES = {"t_cold_c": 10.0, "t_rated_c": 60.0, "t_normal_c": 50.0,
                "t_legionella_c": 60.0, "t_heat_pump_max_c": 55.0,
                "cop_legionella": 2.5, "element_efficiency": 1.0}

SIZING_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)

RESERVE_QUANTILES = (0.5, 0.9, 1.0)


def demand_series(env: STEMSEnvironment, steps: int) -> np.ndarray:
    obs, _ = env.reset()
    rows = [np.array([float(o[IDX_DHW_DEMAND]) for o in obs])]
    zero = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    for _ in range(steps - 1):
        obs, _r, term, trunc, _i = env.step(zero)
        rows.append(np.array([float(o[IDX_DHW_DEMAND]) for o in obs]))
        if term or trunc:
            break
    return np.stack(rows)


def run(spec: LegionellaSpec, demand: np.ndarray, cycle_on: bool) -> Dict[str, Any]:
    stack = LegionellaStack.build(spec, action_index=0)
    B = spec.B
    per_step = spec.heat_source_kw * spec.dt_hours
    for t in range(demand.shape[0]):
        soc = stack.tank.soc
        need_kwh = np.maximum(spec.soc_normal - soc, 0.0) * spec.capacity_kwh + demand[t]
        thermostat = np.clip(need_kwh / np.maximum(per_step, 1e-9), 0.0, 1.0)
        actions = np.zeros((B, 1), dtype=np.float32)
        actions[:, 0] = thermostat.astype(np.float32)
        if cycle_on:
            actions = stack.load.project(actions, [np.zeros(40, dtype=np.float32)] * B)
        stack.observe(actions[:, 0], demand[t])
    report = stack.report()
    report["cycle_enforced"] = cycle_on
    return report


def main() -> None:
    env = STEMSEnvironment(schema=SCHEMA, seed=0, heat_pump=True,
                           env_kwargs={"episode_time_steps": [(0, STEPS)]})
    capacity = np.asarray(env.dhw_info()["capacity"], dtype=np.float64)
    demand = demand_series(env, STEPS)
    mean_daily = demand.mean(axis=0) * 24.0
    base_p_hp = mean_daily / 4.0

    loss = np.asarray(env.dhw_info()["loss_coefficient"], dtype=np.float64)

    def make(multiplier: float, quantile: float) -> LegionellaSpec:
        spec = LegionellaSpec(
            capacity_kwh=capacity, heat_source_kw=base_p_hp * multiplier,
            loss_coefficient=loss,
            draw_margin_kwh=np.quantile(demand, quantile, axis=0),
            period_hours=HOURS_PER_WEEK, **TEMPERATURES)
        spec.provenance["capacity_kwh"] = f"schema {SCHEMA}, dhw_storage.capacity"
        spec.provenance["loss_coefficient"] = f"schema {SCHEMA}, dhw_storage"
        return spec

    rows: List[Dict[str, Any]] = []
    for m in SIZING_MULTIPLIERS:
        spec = make(m, 0.5)
        for cycle_on in (False, True):
            r = run(spec, demand, cycle_on)
            r.update(sweep="sizing", sizing_multiplier=m, reserve_quantile=0.5,
                     heat_source_kw=[round(float(p), 4) for p in spec.heat_source_kw],
                     draw_margin_kwh=[round(float(d), 4) for d in spec.draw_margin_kwh])
            rows.append(r)
    for q in RESERVE_QUANTILES:
        try:
            spec = make(1.0, q)
        except ValueError as exc:
            rows.append({"sweep": "reserve", "sizing_multiplier": 1.0,
                         "reserve_quantile": q, "cycle_enforced": True,
                         "refused": str(exc),
                         "service": {"unmet_kwh": float("nan"),
                                     "electricity_kwh": float("nan")},
                         "cycle": {"cycles_completed": [], "missed_windows": []}})
            continue
        r = run(spec, demand, True)
        r.update(sweep="reserve", sizing_multiplier=1.0, reserve_quantile=q,
                 heat_source_kw=[round(float(p), 4) for p in spec.heat_source_kw],
                 draw_margin_kwh=[round(float(d), 4) for d in spec.draw_margin_kwh])
        rows.append(r)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"schema": SCHEMA, "steps": int(demand.shape[0]),
         "buildings": int(demand.shape[1]),
         "mean_daily_demand_kwh": [round(float(d), 4) for d in mean_daily],
         "sizing_rule": "mean daily demand / 4 h (docs/dhw_legionella_design.md)",
         "temperatures_are_assumptions_of_this_diagnostic": TEMPERATURES,
         "runs": rows}, indent=2) + "\n")

    print(f"{demand.shape[0]} steps, {demand.shape[1]} buildings, "
          f"total draw {demand.sum():.1f} kWh, "
          f"{demand.shape[1] * demand.shape[0] / HOURS_PER_WEEK:.0f} building-windows")
    print(f"mean daily demand per building, kWh: "
          f"{[round(float(d), 2) for d in mean_daily]}")
    head = (f"{'sweep':>9}{'P_hp x':>8}{'reserve q':>11}{'cycle':>7}{'unmet kWh':>11}"
            f"{'done':>6}{'missed':>8}{'elec kWh':>10}")
    for label in ("sizing", "reserve"):
        print("\n" + head)
        print("-" * len(head))
        for r in (x for x in rows if x["sweep"] == label):
            s, c = r["service"], r["cycle"]
            lead = (f"{r['sweep']:>9}{r['sizing_multiplier']:>8}"
                    f"{r['reserve_quantile']:>11}{str(r['cycle_enforced']):>7}")
            if "refused" in r:
                print(lead + "   REFUSED: reserve >= source output per step")
                continue
            print(lead + f"{s['unmet_kwh']:>11.3f}{sum(c['cycles_completed']):>6}"
                  f"{sum(c['missed_windows']):>8}{s['electricity_kwh']:>10.2f}")
    print(f"\nunsourced parameters carried by every row: "
          f"{make(1.0, 0.5).unsourced}")
    print(f"written to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
