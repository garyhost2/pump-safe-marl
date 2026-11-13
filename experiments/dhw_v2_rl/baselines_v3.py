#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_SPECS = [
    {"name": "reactive", "ctl": "reactive"}, {"name": "price_aware", "ctl": "price_aware"},
    {"name": "thermia", "ctl": "thermia"}, {"name": "null", "ctl": "null"},
    {"name": "mpc_setpoint", "ctl": "mpc", "kw": {"rho": 0.9}},
    {"name": "mpc_band_20_23", "ctl": "mpc_band", "kw": {"band": [20, 23]}},
    {"name": "mpc_band_20_22", "ctl": "mpc_band", "kw": {"band": [20, 22]}},
    {"name": "mpc_band_21_22", "ctl": "mpc_band", "kw": {"band": [21, 22]}},
    {"name": "pf_band_20_23", "ctl": "mpc_band", "kw": {"band": [20, 23], "horizon": 72, "forecast": "perfect"}},
    {"name": "pf_band_20_22", "ctl": "mpc_band", "kw": {"band": [20, 22], "horizon": 72, "forecast": "perfect"}},
    {"name": "pf_band_21_22", "ctl": "mpc_band", "kw": {"band": [21, 22], "horizon": 72, "forecast": "perfect"}},
]


def make_ctl(ctl, B, kw):
    from stems.dhw_v2.controllers import NullController, PriceAwareController, ReactiveController, ThermiaLogic
    from stems.dhw_v2_rl.mpc import MPCController
    from stems.dhw_v2_rl.mpc_v3 import MPCBandController, PreheatRule
    kw = dict(kw or {})
    if "band" in kw:
        kw["band"] = tuple(kw["band"])
    return {"reactive": lambda: ReactiveController(B), "price_aware": lambda: PriceAwareController(B),
            "thermia": lambda: ThermiaLogic(B, mode="hourly"), "null": lambda: NullController(B),
            "mpc": lambda: MPCController(B, **kw), "mpc_band": lambda: MPCBandController(B, **kw),
            "preheat": lambda: PreheatRule(B, **kw)}[ctl]()


def job(args):
    spec, setname, out, bc_out = args
    from experiments.dhw_v2_rl.train_v2 import eval_sets
    from stems.dhw_v2_rl.common import RewardSpec, TrackConfig, dumps, make_env_v2, metrics, run_episode
    p = Path(out) / "baselines" / setname.replace(":", "_")
    f = p / f"{spec['name']}.json"
    if f.exists():
        return setname, spec["name"], "skip"
    (data, bh), = eval_sets([setname]).values()
    env = make_env_v2(data, TrackConfig(), RewardSpec(), filtered=True, k_correction=0.0, borehole=bh)
    ctl = make_ctl(spec["ctl"], env.B, spec.get("kw"))
    if hasattr(ctl, "reset"):
        ctl.reset()
    rec = None
    if bc_out:
        from stems.dhw_v2_rl.rl_v3 import ObsV3, PREHEAT_BASE
        from stems.dhw_v2_rl.mpc_v3 import PreheatRule
        base = PreheatRule(env.B, **PREHEAT_BASE)
        ob = ObsV3(env, price_ref=None)
        rec = {"X": [], "sp": [], "plan": [], "u_d": [], "base_on": []}

    def act(e, X):
        if rec is not None:
            sp = base.setpoint(e)
            bcmd = base.act(e)
            ob.base_state = (sp, np.asarray(bcmd["u_d"], float))
            rec["X"].append(ob.build())
        cmd = ctl.act(e)
        if rec is not None:
            rec["sp"].append(sp)
            rec["plan"].append(np.asarray(ctl.plan_t_in, float).copy())
            rec["u_d"].append(np.asarray(cmd["u_d"], float))
            rec["base_on"].append(np.asarray(bcmd["u_d"], float))
        return cmd

    def upd(r):
        ctl.update(r)
        if rec is not None:
            base.update(r)
            ob.after_step(r)

    t0 = time.time()
    S = run_episode(env, act, update=upd)
    m = metrics(S)
    m["wall_s"] = time.time() - t0
    if hasattr(ctl, "lp_fail"):
        m["mpc_lp_fail_hours"] = ctl.lp_fail
    p.mkdir(parents=True, exist_ok=True)
    f.write_text(dumps({"name": spec["name"], "ctl": spec["ctl"], "kw": spec.get("kw"), "set": setname,
                        "currency": data.currency, "metrics": m}))
    if rec is not None:
        bp = Path(bc_out)
        bp.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bp / f"bc_{setname.replace(':', '_')}_{spec['name']}.npz",
                            X=np.array(rec["X"], np.float32), sp=np.array(rec["sp"], np.float32),
                            plan=np.array(rec["plan"], np.float32), u_d=np.array(rec["u_d"], np.float32),
                            base_on=np.array(rec["base_on"], np.float32))
    return setname, spec["name"], round(m["cost"], 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["karlstad:2025", "lulea:2025", "texas"])
    ap.add_argument("--specs", default="", help="JSON list of specs (default: DEFAULT_SPECS)")
    ap.add_argument("--out", default=".scratch/v3")
    ap.add_argument("--bc-out", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    specs = json.loads(a.specs) if a.specs else DEFAULT_SPECS
    jobs = [(sp, s, a.out, a.bc_out) for s in a.sets for sp in specs]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for r in pool.imap_unordered(job, jobs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
