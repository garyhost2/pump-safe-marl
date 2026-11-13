#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig
from stems.environment import STEMSEnvironment
from stems.flexibility import FlexibilityPortfolio
from stems.protocols import Environment, PlantProvider, missing_capabilities
from stems.replay import CSVReplayEnvironment

OUT = REPO / "experiments" / "diagnostics" / "replay_site"
SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
STEPS = 72
T0 = datetime(2026, 1, 1, 0, 0)


def export(env: STEMSEnvironment, steps: int, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    names = env.obs_names
    rng = np.random.default_rng(0)
    obs, _ = env.reset()
    rows: List[List[np.ndarray]] = [[o.copy() for o in obs]]
    for _ in range(steps - 1):
        a = rng.uniform(-0.4, 0.4, (env.num_buildings, env.action_dim)).astype(np.float32)
        obs, _r, term, trunc, _i = env.step(a)
        rows.append([o.copy() for o in obs])
        if term or trunc:
            break

    batt = env.battery_info()
    cl = env._env.buildings
    dhw = env.dhw_info()
    buildings = []
    for b in range(env.num_buildings):
        name = f"b{b}"
        with (root / f"{name}.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp"] + names)
            for t, step in enumerate(rows):
                w.writerow([(T0 + timedelta(hours=t)).isoformat()]
                           + [f"{float(v):.10g}" for v in step[b]])
        buildings.append({
            "name": name, "csv": f"{name}.csv",
            "battery": {"capacity_kwh": float(batt["capacity"][b]),
                        "nominal_power_kw": float(batt["nominal_power"][b]),
                        "loss_coefficient": float(cl[b].electrical_storage.loss_coefficient)},
            "dhw": {"capacity_kwh": float(dhw["capacity"][b]),
                    "heater_power_kw": float(dhw["nominal_power"][b]),
                    "heater_efficiency": float(dhw["efficiency"][b]),
                    "storage_efficiency": float(cl[b].dhw_storage.round_trip_efficiency),
                    "loss_coefficient": float(dhw["loss_coefficient"][b])},
        })
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(
        {"dt_hours": float(env._env.seconds_per_time_step) / 3600.0,
         "heat_pump": True, "action_names": list(env.action_names),
         "exported_from": SCHEMA, "steps": len(rows),
         "buildings": buildings}, indent=2) + "\n")
    return manifest


def shield_for(env: Any, battery_model: Any = None) -> CBFShield:
    cfg = CBFConfig(P_grid_max=300.0, P_building_max=80.0)
    return CBFShield(cfg, num_buildings=env.num_buildings,
                     battery_model=battery_model or env.battery_model(),
                     nominal_power=env.battery_info()["nominal_power"],
                     elec_idx=env.electrical_storage_action_index,
                     safety_cfg=SafetyConfig(anticipatory=False, robust_margins=False))


def main() -> None:
    sim = STEMSEnvironment(schema=SCHEMA, seed=0, heat_pump=True,
                           env_kwargs={"episode_time_steps": [(0, STEPS)]})
    manifest = export(sim, STEPS, OUT)
    replay = CSVReplayEnvironment(manifest)

    print("protocol conformance")
    for env, label in ((sim, "STEMSEnvironment"), (replay, "CSVReplayEnvironment")):
        gaps = missing_capabilities(env)
        print(f"  {label:<22} Environment={isinstance(env, Environment)} "
              f"PlantProvider={isinstance(env, PlantProvider)} "
              f"missing={sorted(gaps) or 'nothing'}")
    assert sim.obs_names == replay.obs_names, "observation layouts differ"

    rng = np.random.default_rng(0)
    sim2 = STEMSEnvironment(schema=SCHEMA, seed=0, heat_pump=True,
                            env_kwargs={"episode_time_steps": [(0, STEPS)]})
    s_sim = shield_for(sim2)
    s_rep = shield_for(replay)
    s_rep_fitted = shield_for(replay, battery_model=sim2.battery_model())
    pf_sim = FlexibilityPortfolio.from_environment(sim2, s_sim.cfg)
    pf_rep = FlexibilityPortfolio.from_environment(replay, s_rep.cfg)
    assert [l.name for l in pf_sim.loads] == [l.name for l in pf_rep.loads]

    o_sim, _ = sim2.reset()
    o_rep, _ = replay.reset()
    worst_obs = worst_act = worst_act_fitted = 0.0
    n = 0
    for _ in range(STEPS - 1):
        a = rng.uniform(-0.4, 0.4, (sim2.num_buildings, sim2.action_dim)).astype(np.float32)
        worst_obs = max(worst_obs, float(np.abs(np.stack(o_sim) - np.stack(o_rep)).max()))
        ref = s_sim.project(a.copy(), o_sim)
        worst_act = max(worst_act, float(np.abs(ref - s_rep.project(a.copy(), o_rep)).max()))
        worst_act_fitted = max(worst_act_fitted,
                               float(np.abs(ref - s_rep_fitted.project(a.copy(), o_rep)).max()))
        o_sim = sim2.step(a)[0]
        o_rep = replay.step(a)[0]
        n += 1

    print(f"\nreplayed {n} steps over {sim2.num_buildings} buildings")
    print(f"  max |observation_sim - observation_replay|                 = {worst_obs:.3e}")
    print(f"  max |action_sim - action_replay|, assumed flat curves      = {worst_act:.3e}")
    print(f"  max |action_sim - action_replay|, simulator's own curves   = {worst_act_fitted:.3e}")
    print("  (the first residual is the plant assumption, not the interface:"
          " the second isolates it)")
    print(f"\nreplay adapter's declared limitations:")
    for note in replay.plant_notes:
        print(f"  - {note}")
    print(f"\nexport written to {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
