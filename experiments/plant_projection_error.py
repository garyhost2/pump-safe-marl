#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stems.environment import STEMSEnvironment
from stems.observations import obs_index

OUT = REPO / "experiments" / "diagnostics" / "plant_projection" / "projection_error.json"

TX_SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
EV_SCHEMA = "citylearn_schemas/tx_travis_8b_ev/schema.json"
WEEK = {"episode_time_steps": [(0, 167)]}

IDX_SOC_ELEC = obs_index("electrical_storage_soc")
IDX_SOC_DHW = obs_index("dhw_storage_soc")


def _stats(err: np.ndarray) -> Dict[str, float]:
    err = np.asarray(err, dtype=np.float64).reshape(-1)
    return {"max": float(err.max()), "p95": float(np.percentile(err, 95)),
            "mean": float(err.mean()), "n": int(err.size)}


def _inverse_sweep(forward, inverse, soc: np.ndarray, fractions=(0.0, 0.25, 0.5, 0.75, 1.0),
                   **kw) -> np.ndarray:
    lo = forward(soc, -np.ones_like(soc), **kw)
    hi = forward(soc, np.ones_like(soc), **kw)
    out = []
    for f in fractions:
        target = lo + f * (hi - lo)
        reached = forward(soc, inverse(soc, target, **kw), **kw)
        out.append(np.abs(reached - target))
    return np.concatenate(out)


def battery_error() -> Dict[str, Any]:
    env = STEMSEnvironment(schema=TX_SCHEMA, seed=0, heat_pump=True, env_kwargs=WEEK)
    model = env.battery_model()
    e = env.electrical_storage_action_index
    floor = np.array([1.0 - b.electrical_storage.depth_of_discharge
                      for b in env._env.buildings])
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    fwd, inv, kept, total, done = [], [], 0, 0, False
    while not done:
        soc = np.array([o[IDX_SOC_ELEC] for o in obs], dtype=np.float64)
        a = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        a[:, e] = rng.uniform(-1.0, 1.0, env.num_buildings)
        inv.append(_inverse_sweep(model.next_soc, model.action_for_soc, soc))
        out = env.step(a)
        obs, done = out[0], out[2] or out[3]
        executed = env.executed_actions[:, e].astype(np.float64)
        predicted = model.next_soc(soc, executed)
        real = np.array([o[IDX_SOC_ELEC] for o in obs], dtype=np.float64)
        free = (real > floor + 0.02) & (real < 0.95)
        total += free.size
        kept += int(free.sum())
        fwd.append(np.abs(predicted - real)[free])
    return {"model": "BatteryModel", "unit_forward": "state of charge (dimensionless)",
            "unit_inverse": "state of charge (dimensionless)",
            "forward": _stats(np.concatenate(fwd)), "inverse": _stats(np.concatenate(inv)),
            "unsaturated_share": round(kept / total, 4), "schema": TX_SCHEMA,
            "window_steps": 168}


def tank_error() -> Dict[str, Any]:
    env = STEMSEnvironment(schema=EV_SCHEMA, seed=0, heat_pump=True,
                           env_kwargs={"episode_time_steps": [(0, 239)]},
                           hvac_control="setpoint")
    tank = env.dhw_tank_model()
    d = env.dhw_action_index
    buildings = env._env.buildings
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    fwd, inv, done = [], [], False
    while not done:
        soc = np.array([o[IDX_SOC_DHW] for o in obs], dtype=np.float64)
        ts = env._env.time_step
        demand = np.array([b.dhw_demand[ts] for b in buildings], dtype=np.float64)
        a = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        a[:, d] = rng.uniform(-0.6, 0.6, env.num_buildings)
        predicted = tank.drawn_kwh(soc, a[:, d].astype(np.float64), demand)
        inv.append(_inverse_sweep(tank.next_soc, tank.action_for_soc, soc, demand=demand))
        out = env.step(a)
        obs, done = out[0], out[2] or out[3]
        actual = np.array([b.dhw_device.electricity_consumption[ts] for b in buildings],
                          dtype=np.float64)
        fwd.append(np.abs(actual - demand / tank.heater_efficiency - predicted))
    return {"model": "TankModel", "unit_forward": "heater electricity, kWh per step",
            "unit_inverse": "state of charge (dimensionless)",
            "forward": _stats(np.concatenate(fwd)), "inverse": _stats(np.concatenate(inv)),
            "unsaturated_share": 1.0, "schema": EV_SCHEMA, "window_steps": 240}


def ev_error() -> Dict[str, Any]:
    env = STEMSEnvironment(schema=EV_SCHEMA, seed=0, heat_pump=True,
                           env_kwargs={"episode_time_steps": [(0, 239)]},
                           hvac_control="setpoint")
    cl = env._env
    model = env.ev_fleet_model()
    layout, e = env.ev_obs_layout()[0], env.ev_action_indices()[0]
    vehicles = {ev.name: ev for ev in cl.electric_vehicles}
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    fwd, draw_err, inv, dead, done = [], [], [], [], False
    while not done:
        t = cl.time_step
        conn = np.array([o[layout["connected_state"]] for o in obs]) > 0.5
        soc = np.array([o[layout["soc"]] for o in obs], dtype=np.float64)
        a = np.where(conn, rng.choice([0.0, 0.05, 0.3, 0.7, 1.0], env.num_buildings), 0.0)
        actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        actions[:, e] = a
        predicted = model.next_soc(soc, a)
        draw = model.draw_kw(soc, a)
        top = model.draw_kw(soc, np.ones(model.B))
        live = conn & model.has_ev & (top > 1e-9)
        for f in (0.0, 0.25, 0.5, 0.75, 1.0):
            want = f * top
            got = model.draw_kw(soc, model.action_for_draw(soc, want))
            err = np.abs(got - want)
            attainable = (want <= 1e-9) | (want >= model.p_min - 1e-9)
            inv.append(err[live & attainable])
            dead.append(err[live & ~attainable])
        out = env.step(actions)
        obs, done = out[0], out[2] or out[3]
        for i in np.flatnonzero(conn & model.has_ev):
            charger = cl.buildings[i].electric_vehicle_chargers[0]
            name = str(np.asarray(charger.charger_simulation.electric_vehicle_id)[t])
            fwd.append(abs(predicted[i] - vehicles[name].battery.soc[t]))
            draw_err.append(abs(draw[i] - env.ev_draw_kwh[i]))
    return {"model": "EVFleetModel", "unit_forward": "state of charge (dimensionless)",
            "unit_inverse": "charger draw, kW",
            "forward": _stats(np.asarray(fwd)), "inverse": _stats(np.concatenate(inv)),
            "draw_kw": _stats(np.asarray(draw_err)),
            "dead_band_miss_kw": _stats(np.concatenate(dead)),
            "p_min_kw": sorted({float(x) for x in model.p_min[model.has_ev]}),
            "unsaturated_share": 1.0, "schema": EV_SCHEMA, "window_steps": 240}


def main() -> None:
    rows = [battery_error(), tank_error(), ev_error()]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    head = f"{'model':<14}{'forward max':>14}{'inverse max':>14}{'n':>9}  unit (forward)"
    print(head)
    print("-" * len(head))
    for r in rows:
        print(f"{r['model']:<14}{r['forward']['max']:>14.3e}{r['inverse']['max']:>14.3e}"
              f"{r['forward']['n']:>9}  {r['unit_forward']}")
    print(f"\nwritten to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
