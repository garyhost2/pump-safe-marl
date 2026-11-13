#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.controllers import ARMS
from experiments.runner import code_fingerprint, run_one
from experiments.scenario import SEASONS, TX_SCHEMA, Scenario


def build_grid(args) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    reward_overrides = {}
    for item in getattr(args, "reward", []) or []:
        if "=" not in item:
            raise SystemExit(f"--reward expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        reward_overrides[k.strip()] = float(v)

    for season in args.seasons:
        for subset in args.subsets:
            scenario = Scenario(schema=args.schema, season=season,
                                subset_seed=None if subset == "ref" else int(subset),
                                n_buildings=args.buildings, days=args.days,
                                hvac_control=args.hvac_control,
                                heat_pump=not args.no_heat_pump,
                                allow_missing_obs=args.allow_missing_obs,
                                grid_cap_kw=args.grid_cap,
                                building_cap_kw=args.building_cap)
            for arm_name in args.arms:
                seeds = args.seeds if ARMS[arm_name].learns else [0]
                for seed in seeds:
                    out = Path(args.out) / scenario.key / f"{arm_name}__seed{seed}.json"
                    specs.append({"scenario": asdict(scenario), "arm": arm_name,
                                  "seed": seed, "episodes": args.episodes,
                                  "learner": {"share_parameters": not args.per_building_actors,
                                              **({"policy_reg": args.policy_reg}
                                                 if args.policy_reg is not None else {}),
                                              **({"metaems_inner_steps":
                                                  args.metaems_inner_steps}
                                                 if args.metaems_inner_steps is not None
                                                 else {}),
                                              **({"device": args.device}
                                                 if args.device else {}),
                                              **({"stems_clip_eps": args.stems_clip}
                                                 if args.stems_clip is not None else {}),
                                              **({"sac_scale_returns": True}
                                                 if args.sac_scale_returns else {}),
                                              **({"stems_epochs": args.stems_epochs}
                                                 if args.stems_epochs is not None else {}),
                                              **({"reward": reward_overrides} if reward_overrides else {})},
                                  "out": str(out)})
    return specs


def already_done(spec: Dict[str, Any], fingerprint: str) -> bool:
    path = Path(spec["out"])
    path = path if path.is_absolute() else REPO / path
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    meta = record.get("meta", {})
    return (record.get("status") == "ok"
            and (meta.get("code") or {}).get("fingerprint") == fingerprint
            and (meta.get("episodes") == spec["episodes"]
                 or not ARMS[spec["arm"]].learns))


def main() -> None:
    ap = argparse.ArgumentParser(description="Policy vs safety-layer ablation grid")
    ap.add_argument("--seasons", nargs="+", default=["winter", "summer"],
                    choices=SEASONS)
    ap.add_argument("--subsets", nargs="+", default=["ref", "1", "2"],
                    help="'ref' = the schema's own buildings; integers = sampled subsets")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--days", type=int, default=28,
                    help="length of each training and evaluation window")
    ap.add_argument("--buildings", type=int, default=8)
    ap.add_argument("--grid-cap", type=float, default=Scenario.grid_cap_kw,
                    help="District import cap, kW. Defaults to the Scenario value, "
                         "which is 80%% of the measured uncontrolled peak. The old "
                         "300 kW could not bind: 0 of 8759 hours.")
    ap.add_argument("--building-cap", type=float, default=Scenario.building_cap_kw,
                    help="Per-building cap, kW. Defaults to the Scenario value.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--schema", default=TX_SCHEMA,
                    help="schema to run on; the housing-and-EV schema adds the fleet "
                         "shield to every shielded arm")
    ap.add_argument("--hvac-control", choices=["setpoint", "power"], default="setpoint",
                    help="HVAC action semantics: set-point offset (default) or power fraction")
    ap.add_argument("--no-heat-pump", action="store_true",
                    help="the schema has no heat pump the controller drives "
                         "(datasets without thermal dynamics)")
    ap.add_argument("--allow-missing-obs", action="store_true",
                    help="zero-fill observations the dataset does not have")
    ap.add_argument("--per-building-actors", action="store_true",
                    help="one actor/critic per building (the paper's layout) instead of one "
                         "shared by all buildings")
    ap.add_argument("--out", default="results/ablation_v1")
    ap.add_argument("--sac-scale-returns", action="store_true",
                    help="divide the reward by a running standard deviation of the "
                         "discounted return in SingleAgentSAC's TD target. Its critic "
                         "otherwise converges to |Q| ~ 7e4 with an action-dependence "
                         "span of 4.6, so the actor ascends a near-constant surface "
                         "and saturates. Off by default.")
    ap.add_argument("--stems-clip", type=float, default=None, metavar="EPS",
                    help="trust region for the paper-stems arm. Their Algorithm 2 "
                         "line 11 is vanilla advantage policy gradient, which is the "
                         "default and which DEGRADES on this task. 0.2 switches to the "
                         "standard clipped PPO surrogate. Their paper states no "
                         "surrogate form for any method, so this is our choice for an "
                         "underspecified line, not a reading of their text.")
    ap.add_argument("--stems-epochs", type=int, default=None, metavar="K",
                    help="passes over the episode buffer per paper-stems update "
                         "(default 1, which is on-policy correct for vanilla PG).")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="torch device for the baseline learners. Default cpu. The "
                         "CityLearn simulator stays on the CPU either way, so the "
                         "speedup is bounded by the ~87 s per episode it costs.")
    ap.add_argument("--metaems-inner-steps", type=int, default=None, metavar="K",
                    help="MetaEMS Reptile inner-loop steps per task per episode. The "
                         "agent takes num_buildings*K gradient steps per episode; the "
                         "default 8 gives 64 against 8759 for the other learners. Use "
                         "~8759/num_buildings for a matched budget.")
    ap.add_argument("--policy-reg", type=float, default=None, metavar="W",
                    help="pre-squash policy regularisation weight for the actor-critic "
                         "baselines (SAC, MARLISA, MetaEMS, and the L2-on-output "
                         "analogue for MADDPG). The reference implementations use 1e-3; "
                         "ours omitted it, and results/action_profile shows SAC "
                         "commanding a saturated action on 97.4 per cent of "
                         "building-steps as a result. Default leaves it at 0.0.")
    ap.add_argument("--reward", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override reward weights, e.g. --reward lambda_indoor=0.1 xi=0.6. "
                         "Runs with different weights optimise different objectives and "
                         "must go to different --out directories; the record filename "
                         "carries only the arm name.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    specs = build_grid(args)
    fingerprint = code_fingerprint()["fingerprint"]
    todo = [s for s in specs if args.force or not already_done(s, fingerprint)]
    print(f"[ablation] {len(specs)} runs in grid, {len(todo)} to do, workers={args.workers}")
    if args.dry_run:
        for s in todo:
            print("   ", s["out"])
        return

    for scenario_json in {json.dumps(s["scenario"], sort_keys=True) for s in todo}:
        Scenario(**json.loads(scenario_json)).schema_path()

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    Path(args.out).mkdir(parents=True, exist_ok=True)

    t0, finished = time.time(), 0
    with get_context("spawn").Pool(processes=max(1, args.workers), maxtasksperchild=1) as pool:
        for res in pool.imap_unordered(run_one, todo):
            finished += 1
            print(f"[ablation] {finished}/{len(todo)} {res['status']:5s} "
                  f"verified={res['verified']} {res['seconds']:8.1f}s  {res['out']}",
                  flush=True)
    print(f"[ablation] finished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
