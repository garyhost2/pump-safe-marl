from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

HVAC_ACTION = "cooling_or_heating_device"

CACHE = os.path.join(os.getcwd(), ".citylearn_cache", "citylearn")


def resolve_schema(name: str) -> str:
    if os.path.sep in name or name.endswith(".json"):
        return name
    for version in sorted(os.listdir(CACHE)) if os.path.isdir(CACHE) else []:
        path = os.path.join(CACHE, version, "datasets", name, "schema.json")
        if os.path.exists(path):
            return path
    return name


def _cost(env) -> Dict[str, float]:
    nets, prices = [], []
    tracker = getattr(env, "episode_tracker", None)
    lo = int(getattr(tracker, "episode_start_time_step", 0) or 0)
    for b in env.buildings:
        net = np.asarray(b.net_electricity_consumption, dtype=np.float64)
        nets.append(net)
        prices.append(np.asarray(b.pricing.electricity_pricing,
                                 dtype=np.float64)[lo:lo + len(net)])
    net = np.stack(nets, axis=1)
    price = np.stack(prices, axis=1)
    district = net.sum(axis=1)
    return {"cost": float((np.maximum(net, 0.0) * price).sum()),
            "consumption": float(np.maximum(net, 0.0).sum()),
            "peak": float(np.maximum(district, 0.0).max()),
            "mean_price": float(price.mean()),
            "steps": int(net.shape[0]), "buildings": int(net.shape[1])}


def one_dataset(name: str, drop_hvac: bool) -> Dict[str, Any]:
    from citylearn.agents.base import BaselineAgent
    from citylearn.agents.rbc import BasicRBC
    from citylearn.citylearn import CityLearnEnv

    t0 = time.time()
    kw: Dict[str, Any] = {"central_agent": False}
    if drop_hvac:
        kw["inactive_actions"] = [HVAC_ACTION]
    schema = resolve_schema(name)
    probe = CityLearnEnv(schema=schema, central_agent=False)
    active = list(probe.buildings[0].active_actions)
    if drop_hvac and HVAC_ACTION not in active:
        return {"dataset": name, "drop_hvac": drop_hvac, "skipped":
                "no HVAC action to drop"}

    out: Dict[str, Any] = {"dataset": name, "drop_hvac": drop_hvac,
                           "schema": schema, "from_cache": schema != name,
                           "active_actions_before": active}
    for label, Agent in (("idle", BaselineAgent), ("rbc", BasicRBC)):
        env = CityLearnEnv(schema=schema, **kw)
        Agent(env=env).learn(episodes=1)
        out[label] = _cost(env)
        out["active_actions"] = list(env.buildings[0].active_actions)
    out["idle_over_rbc"] = (out["idle"]["cost"] / out["rbc"]["cost"]
                            if out["rbc"]["cost"] else None)
    out["idle_over_rbc_consumption"] = (out["idle"]["consumption"] / out["rbc"]["consumption"]
                                        if out["rbc"]["consumption"] else None)
    out["idle_over_rbc_peak"] = (out["idle"]["peak"] / out["rbc"]["peak"]
                                 if out["rbc"]["peak"] else None)
    out["seconds"] = round(time.time() - t0, 1)
    return out


def _task(args):
    name, drop_hvac = args
    try:
        return one_dataset(name, drop_hvac)
    except Exception as exc:
        return {"dataset": name, "drop_hvac": drop_hvac, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-800:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/rbc_normaliser_sweep")
    args = ap.parse_args()

    from citylearn.data import DataSet

    names = args.datasets or DataSet().get_dataset_names()
    os.makedirs(args.out, exist_ok=True)
    def done(n: str, d: bool) -> bool:
        path = os.path.join(args.out, f"{n}__{'storage_only' if d else 'as_shipped'}.json")
        if not os.path.exists(path):
            return False
        try:
            return "error" not in json.load(open(path))
        except Exception:
            return False

    jobs = [(n, d) for n in names for d in (False, True) if not done(n, d)]
    print(f"{len(jobs)} runs over {len(names)} datasets")

    from multiprocessing import get_context

    with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
        for rec in pool.imap_unordered(_task, jobs):
            tag = "storage_only" if rec["drop_hvac"] else "as_shipped"
            with open(os.path.join(args.out, f"{rec['dataset']}__{tag}.json"), "w") as fh:
                json.dump(rec, fh, indent=2)
            if "error" in rec:
                print(f"{rec['dataset'][:46]:<48}{tag:<14}ERROR {rec['error'][:50]}")
            elif "skipped" in rec:
                print(f"{rec['dataset'][:46]:<48}{tag:<14}skipped")
            else:
                r = rec.get("idle_over_rbc")
                rc = rec.get("idle_over_rbc_consumption")
                cost_txt = "no tariff" if r is None else f"{r:.3f}"
                cons_txt = "n/a" if rc is None else f"{rc:.3f}"
                print(f"{rec['dataset'][:46]:<48}{tag:<14}"
                      f"idle/rbc cost {cost_txt:>9}  kWh {cons_txt:>6}  "
                      f"({rec['idle']['buildings']}b, {rec['idle']['steps']}t)")


if __name__ == "__main__":
    main()
