#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MAX_WORKERS = 15


def make_ctl(ctl, B, kw):
    from experiments.dhw_v2_rl.baselines_v3 import make_ctl as base
    if ctl == "thermia_weekly":
        from stems.dhw_v2.controllers import ThermiaWeekly
        return ThermiaWeekly(B, **(kw or {}))
    return base(ctl, B, kw)


def run_one(spec):
    from experiments.dhw_v2_rl.train_v2 import eval_sets
    from stems.dhw_v2.data import DistrictData
    from stems.dhw_v2_rl.common import RewardSpec, TrackConfig, dumps, make_env_v2, metrics, run_episode, slice_hours, window_starts
    out = Path(spec["out"]) / f"{spec['name']}.json"
    if out.exists():
        return spec["name"], "skip"
    (data, bh), = eval_sets([spec["set"]]).values()
    if spec.get("houses"):
        idx = [i for i, h in enumerate(data.houses) if h.id in spec["houses"]]
        data = data.subset(houses=idx)
    hours = spec.get("hours")
    if hours:
        st = spec.get("start", "monday")
        start = int(window_starts(data, hours)[0]) if st == "monday" else int(st)
        data = slice_hours(data, start, start + hours)
    env_kw = {}
    plant_kw = dict(spec.get("plant") or {})
    filt_kw = dict(spec.get("filter") or {})
    if spec.get("regime"):
        from stems.dhw_v2.hygiene import regime_overrides
        ov = regime_overrides(spec["regime"], spec.get("regime_tol_k"))
        plant_kw = {**ov["plant"], **plant_kw}
        filt_kw = {**ov["filter"], **filt_kw}
        env_kw.update(ov["env"])
    bh = dict(bh or {})
    if spec.get("spinup_years"):
        bh["spinup_years"] = int(spec["spinup_years"])
    track = TrackConfig(plant_kwargs=plant_kw, filter_kwargs=filt_kw, env_kwargs=env_kw)
    env = make_env_v2(data, track, RewardSpec(), filtered=spec.get("filtered", True), k_correction=0.0,
                      borehole=bh or None)
    ctl = make_ctl(spec["ctl"], env.B, spec.get("kw"))
    t0 = time.time()
    S = run_episode(env, lambda e, X: ctl.act(e), update=ctl.update)
    m = metrics(S)
    m["wall_s"] = time.time() - t0
    if hasattr(ctl, "lp_fail"):
        m["mpc_lp_fail_hours"] = ctl.lp_fail
    keep = ("houses", "hours", "elec_kwh", "elec_aux_kwh", "aux_hours", "aux_step_max_kw", "aux_heater", "cost",
            "sh_degree_hours_below_min", "sh_shortfall_kwh", "t_in_min_c", "t_src_d_min_c", "t_src_d_mean_c",
            "hours_below_floor", "weeks_full", "weeks_completed", "hygiene", "thermia_rule", "regime",
            "borehole_spinup", "hours_extrap_supply", "elec_dhw_kwh", "elec_dhw_dis_kwh", "dis_supply_hours")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps({"NOT_QUOTABLE": "pre-fix smoke run (HW2, HW14 open)", "spec": {k: v for k, v in spec.items() if k != "out"},
                          "set": spec["set"], "currency": data.currency, "price_area": data.meta.get("price_area"),
                          "window_utc": [str(data.time_utc[0]), str(data.time_utc[-1])], "metrics": m,
                          "summary": {k: S[k] for k in keep if k in S}}))
    return spec["name"], round(m["cost"], 1), round(m["wall_s"], 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True, help="JSON list of specs, or @file.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    specs = json.loads(Path(a.specs[1:]).read_text() if a.specs.startswith("@") else a.specs)
    for s in specs:
        s["out"] = a.out
    n = max(1, min(a.workers, MAX_WORKERS, len(specs)))
    with mp.get_context("spawn").Pool(n) as pool:
        for r in pool.imap_unordered(run_one, specs):
            print(*r, flush=True)


if __name__ == "__main__":
    main()
