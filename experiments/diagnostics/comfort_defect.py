from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

WATCH = ("stems_discomfort_proportion", "stems_safety_violations_with_comfort",
         "stems_safety_violations_soc_equipment", "stems_cost", "discomfort_rate",
         "discomfort_degree_hours")


def one_run(arm_name: str, scenario, config, patched: bool, seed: int,
            hvac_control: str = "power") -> Dict[str, Any]:
    from stems.environment import STEMSEnvironment
    from stems.utils import set_seed
    from experiments.controllers import ARMS, build_controller
    from experiments.runner import evaluate

    set_seed(seed)
    env = STEMSEnvironment(schema=scenario.schema_path(), seed=seed,
                           heat_pump=scenario.heat_pump,
                           env_kwargs=scenario.env_kwargs("eval"),
                           hvac_control=hvac_control,
                           allow_missing_obs=scenario.allow_missing_obs,
                           patch_endogenous_obs=patched)
    controller = build_controller(ARMS[arm_name], env, config)
    out = evaluate(controller, env, config, max_steps=10 ** 9)
    kpis = out["kpis"]
    return {k: (None if kpis.get(k) is None else float(kpis[k])) for k in WATCH}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["rbc+calibrated"])
    ap.add_argument("--season", default="year")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-cap", type=float, default=2000.0)
    ap.add_argument("--building-cap", type=float, default=500.0)
    ap.add_argument("--hvac-control", default="power", choices=("power", "setpoint"))
    ap.add_argument("--out", default="results/comfort_defect")
    args = ap.parse_args()

    from experiments.runner import make_config
    from experiments.scenario import Scenario

    scenario = Scenario(season=args.season, grid_cap_kw=args.grid_cap,
                        building_cap_kw=args.building_cap)
    config = make_config(scenario)

    rows: List[Dict[str, Any]] = []
    for arm in args.arms:
        for patched in (True, False):
            r = one_run(arm, scenario, config, patched, args.seed,
                        hvac_control=args.hvac_control)
            r.update({"arm": arm, "patched": patched})
            rows.append(r)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "comfort_defect.json"), "w") as fh:
        json.dump(rows, fh, indent=2)

    hdr = f"{'arm':<20}{'simulator':<12}" + "".join(f"{k.replace('stems_',''):>34}"
                                                    for k in WATCH[:2])
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = "".join("{:>34.6f}".format(r[k]) if r[k] is not None else f"{'--':>34}"
                        for k in WATCH[:2])
        print(f"{r['arm']:<20}{'patched' if r['patched'] else 'unpatched':<12}{cells}")

    print()
    for arm in args.arms:
        pair = {r["patched"]: r for r in rows if r["arm"] == arm}
        a, b = pair[True]["stems_discomfort_proportion"], pair[False]["stems_discomfort_proportion"]
        if a is None or b is None:
            continue
        verdict = ("identical -- the indoor-temperature defect does not reach this "
                   "metric in this scenario"
                   if a == b else
                   f"differ by {abs(a - b):.6f} -- the comfort column depends on the fix")
        print(f"{arm}: patched {a:.6f} vs unpatched {b:.6f}; {verdict}")
    print(f"\nwrote {args.out}/comfort_defect.json")


if __name__ == "__main__":
    main()
