#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def controller(name, B, rho):
    from stems.dhw_v2.controllers import NullController, PriceAwareController, ReactiveController, ThermiaLogic
    from stems.dhw_v2_rl.mpc import MPCController
    return {"reactive": lambda: ReactiveController(B), "price_aware": lambda: PriceAwareController(B),
            "thermia": lambda: ThermiaLogic(B, mode="hourly"), "null": lambda: NullController(B),
            "mpc": lambda: MPCController(B, rho=rho, forecast="causal"),
            "mpc_pf": lambda: MPCController(B, rho=rho, forecast="perfect")}[name]()


def job(args):
    name, rho, setname, filtered, out, track_js = args
    from experiments.dhw_v2_rl.train_v2 import eval_sets
    from stems.dhw_v2_rl.common import RewardSpec, TrackConfig, dumps, make_env_v2, metrics, run_episode
    tag = f"{name}{'_rho' + str(rho) if name.startswith('mpc') else ''}__{'filtered' if filtered else 'raw'}"
    p = Path(out) / "baselines" / setname.replace(":", "_")
    f = p / f"{tag}.json"
    if f.exists():
        return setname, tag, "skip"
    (data, bh), = eval_sets([setname]).values()
    track = TrackConfig(**(json.loads(track_js) if track_js else {}))
    env = make_env_v2(data, track, RewardSpec(), filtered=filtered, k_correction=0.0, borehole=bh)
    ctl = controller(name, env.B, rho)
    if hasattr(ctl, "reset"):
        ctl.reset()
    t0 = time.time()
    S = run_episode(env, lambda e, X: ctl.act(e), update=ctl.update)
    S["tank_nodes"] = env.cfg.tank_nodes
    m = metrics(S)
    m["wall_s"] = time.time() - t0
    if hasattr(ctl, "lp_fail"):
        m["mpc_lp_fail_hours"] = ctl.lp_fail
    p.mkdir(parents=True, exist_ok=True)
    f.write_text(dumps({"controller": name, "rho": rho, "set": setname, "filtered": filtered, "currency": data.currency,
                        "borehole": bh, "metrics": m, "summary": {k: S[k] for k in S if k != "filter"}}))
    return setname, tag, round(m["cost"], 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["karlstad:2025", "lulea:2025"])
    ap.add_argument("--controllers", nargs="+", default=["reactive", "price_aware", "thermia", "null", "mpc", "mpc_pf"])
    ap.add_argument("--rho", type=float, nargs="+", default=[0.8])
    ap.add_argument("--modes", nargs="+", default=["filtered", "raw"])
    ap.add_argument("--out", default=".scratch/fair")
    ap.add_argument("--track", default="")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    jobs = []
    for s in a.sets:
        for c in a.controllers:
            for r in (a.rho if c.startswith("mpc") else [None]):
                for mode in a.modes:
                    jobs.append((c, r, s, mode == "filtered", a.out, a.track))
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for res in pool.imap_unordered(job, jobs):
            print(*res, flush=True)


if __name__ == "__main__":
    main()
