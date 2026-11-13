#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.diagnostics.multinode_guarantees import STOCKS, _bounds, _env
from stems.battery import BatteryModel
from stems.config import CBFConfig, SafetyConfig


def replicate(bm: BatteryModel, g: int) -> BatteryModel:
    rep = lambda x: np.repeat(np.asarray(x), g, axis=0)
    eta = [np.stack([x, y]) for x, y in zip(rep(bm._eta_x), rep(bm._eta_y))]
    pw = [np.stack([x, y]) for x, y in zip(rep(bm._pow_x), rep(bm._pow_y))]
    return BatteryModel(rep(bm.capacity), rep(bm.nominal_power), rep(bm.loss), eta, pw,
                        hours_per_step=bm.dt,
                        depth_of_discharge=rep(bm.depth_of_discharge))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=2001)
    ap.add_argument("--out", default=str(REPO / "experiments" / "diagnostics" / "dhw_band"
                                         / "battery_crossing.json"))
    args = ap.parse_args()
    cfg, saf = CBFConfig(), SafetyConfig()
    grid = np.linspace(0.0, 1.0, args.grid)
    rows = []
    for name, seed in STOCKS:
        env = _env(seed)
        bm, info = env.battery_model(), env.battery_info()
        lo_e, hi_e = _bounds(info["soc_rate"].astype(float), cfg, saf)
        big = replicate(bm, args.grid)
        soc = np.tile(grid, bm.B)
        for label, lo, hi in (("enforced", np.repeat(lo_e, args.grid),
                               np.repeat(hi_e, args.grid)),
                              ("reported_0.1_0.9", np.full(soc.size, cfg.SOC_min),
                               np.full(soc.size, cfg.SOC_max))):
            mid = big.safe_interval(soc, lo, hi)
            crossed = big.last_crossed.copy()
            flo = big.safe_interval(soc, lo, hi, on_crossed="floor")
            cei = big.safe_interval(soc, lo, hi, on_crossed="ceiling")
            diff = max(float(np.max(np.abs(mid[0] - flo[0]))),
                       float(np.max(np.abs(mid[1] - flo[1]))),
                       float(np.max(np.abs(mid[0] - cei[0]))),
                       float(np.max(np.abs(mid[1] - cei[1]))))
            rows.append({"stock": name, "bounds": label, "devices": int(bm.B),
                         "grid_points": int(args.grid), "evaluations": int(soc.size),
                         "crossed": int(crossed.sum()),
                         "max_abs_change_from_flag": diff})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = sum(r["crossed"] for r in rows)
    out.write_text(json.dumps({"rows": rows, "crossed_total": total,
                               "evaluations_total": sum(r["evaluations"] for r in rows)},
                              indent=2) + "\n")
    for r in rows:
        print(f"{r['stock']:>10} {r['bounds']:>17}: {r['crossed']} crossed of "
              f"{r['evaluations']}, max change from the flag {r['max_abs_change_from_flag']:.3g}")
    print(f"total crossed {total}; written to {out}")


if __name__ == "__main__":
    main()
