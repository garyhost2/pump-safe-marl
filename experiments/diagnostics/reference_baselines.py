from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

NON_LEARNING = {"citylearn-baseline", "citylearn-rbc", "citylearn-rbc-optimized"}


def build_agent(name: str, env):
    from citylearn.agents.base import BaselineAgent
    from citylearn.agents.marlisa import MARLISA
    from citylearn.agents.rbc import BasicRBC, OptimizedRBC
    from citylearn.agents.sac import SAC

    return {"citylearn-baseline": BaselineAgent,
            "citylearn-rbc": BasicRBC,
            "citylearn-rbc-optimized": OptimizedRBC,
            "citylearn-sac": SAC,
            "citylearn-marlisa": MARLISA}[name](env=env)


def episode_series(env) -> Dict[str, np.ndarray]:
    nets, socs, prices, carbons, t_ins = [], [], [], [], []
    for b in env.buildings:
        net = np.asarray(b.net_electricity_consumption, dtype=np.float64)
        soc = np.asarray(b.electrical_storage.soc, dtype=np.float64)
        n = len(net)
        if len(soc) == n + 1:
            soc = soc[1:]
        tracker = getattr(env, "episode_tracker", None)
        lo = int(getattr(tracker, "episode_start_time_step", 0) or 0)
        price = np.asarray(b.pricing.electricity_pricing, dtype=np.float64)[lo:lo + n]
        carbon = np.asarray(b.carbon_intensity.carbon_intensity, dtype=np.float64)[lo:lo + n]
        t_in = np.asarray(b.energy_simulation.indoor_dry_bulb_temperature,
                          dtype=np.float64)[lo:lo + n]
        if not (len(soc) == len(price) == len(carbon) == len(t_in) == n):
            raise RuntimeError(
                f"series lengths disagree for {b.name}: net {n}, soc {len(soc)}, "
                f"price {len(price)}, carbon {len(carbon)}, t_in {len(t_in)}")
        nets.append(net); socs.append(soc); prices.append(price)
        carbons.append(carbon); t_ins.append(t_in)
    return {k: np.stack(v, axis=1) for k, v in
            (("net", nets), ("soc", socs), ("price", prices),
             ("carbon", carbons), ("t_in", t_ins))}


def one_run(name: str, schema: str, env_kwargs: Dict[str, Any], episodes: int,
            seed: int, soc_min: float, soc_max: float,
            p_building_max: float) -> Dict[str, Any]:
    from citylearn.citylearn import CityLearnEnv
    from stems.stems_table import stems_table_metrics
    from stems.utils import set_seed

    set_seed(seed)
    t0 = time.time()
    env = CityLearnEnv(schema=schema, central_agent=False, random_seed=seed, **env_kwargs)
    active = [a for b in env.action_space for a in []] or env.buildings[0].active_actions
    agent = build_agent(name, env)
    budget = 1 if name in NON_LEARNING else episodes
    agent.learn(episodes=budget, deterministic_finish=name not in NON_LEARNING)
    s = episode_series(env)
    kpis = stems_table_metrics(s["net"], s["price"], s["carbon"], s["t_in"], s["soc"],
                               soc_min=soc_min, soc_max=soc_max,
                               p_building_max=p_building_max)
    return {"agent": name, "seed": seed, "episodes": budget,
            "active_actions": list(active),
            "steps": int(s["net"].shape[0]), "buildings": int(s["net"].shape[1]),
            "seconds": round(time.time() - t0, 1),
            "citylearn_version": __import__("citylearn").__version__,
            "eval": {k: (None if v is None else float(v)) for k, v in kpis.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+",
                    default=["citylearn-baseline", "citylearn-rbc",
                             "citylearn-sac", "citylearn-marlisa"])
    ap.add_argument("--episodes", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--season", default="year")
    ap.add_argument("--soc-min", type=float, default=0.1)
    ap.add_argument("--soc-max", type=float, default=0.9)
    ap.add_argument("--building-cap", type=float, default=500.0)
    ap.add_argument("--inactive-actions", nargs="*", default=None,
                    help="actions to switch off, e.g. cooling_or_heating_device. "
                         "CityLearn's BasicRBC drives the heat pump at 40-80 per cent "
                         "of nominal power by hour of day with no reference to indoor "
                         "temperature, so whether that action is active changes what "
                         "the standard normaliser means.")
    ap.add_argument("--tag", default="", help="suffix for the output filenames")
    ap.add_argument("--out", default="results/reference_baselines")
    args = ap.parse_args()

    from experiments.scenario import Scenario

    scenario = Scenario(season=args.season, building_cap_kw=args.building_cap)
    schema = scenario.schema_path()
    env_kwargs = {k: v for k, v in scenario.env_kwargs("eval").items()
                  if k in ("simulation_start_time_step", "simulation_end_time_step",
                           "episode_time_steps")}
    if args.inactive_actions:
        env_kwargs["inactive_actions"] = list(args.inactive_actions)
    os.makedirs(args.out, exist_ok=True)

    for name in args.agents:
        seeds = [args.seeds[0]] if name in NON_LEARNING else args.seeds
        for seed in seeds:
            path = os.path.join(
                args.out, f"{name}__seed{seed}__ep{args.episodes}{args.tag}.json")
            if os.path.exists(path):
                print(f"skip {name} seed {seed} (exists)")
                continue
            rec = one_run(name, schema, env_kwargs, args.episodes, seed,
                          args.soc_min, args.soc_max, args.building_cap)
            with open(path, "w") as fh:
                json.dump(rec, fh, indent=2)
            print(f"{name:<24} seed {seed}  cost {rec['eval']['stems_cost']:>12.1f}  "
                  f"{rec['seconds']:>7.1f}s")


if __name__ == "__main__":
    main()
