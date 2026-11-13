from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stems.stems_table import NORMALISED_KEYS

METRICS = ("stems_cost", "stems_emission", "stems_avg_daily_peak",
           "stems_electricity_consumption", "stems_ramping_rate",
           "stems_discomfort_proportion", "stems_safety_violations_soc_equipment",
           "stems_soc_violations_pct")
BASELINE_ARM = "rbc"
DEFAULT_LAMBDA = 0.4
DEFAULT_XI = 0.0


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _stdev(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def read_dir(path: str) -> List[Dict[str, Any]]:
    out = []
    for f in glob.glob(os.path.join(path, "*", "*.json")):
        rec = json.load(open(f))
        if rec.get("status") not in (None, "ok"):
            continue
        ev = rec.get("eval") or {}
        if not any(k in ev for k in METRICS):
            continue
        meta = rec.get("meta") or {}
        arm = meta.get("arm") or {}
        scen = meta.get("scenario") or {}
        out.append({
            "arm": arm.get("name") if isinstance(arm, dict) else arm,
            "subset": ("ref" if scen.get("subset_seed") in (None, "")
                       else str(scen.get("subset_seed"))),
            "seed": meta.get("seed"),
            "episodes": meta.get("episodes"),
            "reward_override": (meta.get("learner") or {}).get("reward") or {},
            **{k: (float(ev[k]) if ev.get(k) is not None else float("nan"))
               for k in METRICS},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/overnight")
    ap.add_argument("--baseline-dir", default="results/t1_repro_ep15")
    ap.add_argument("--extra", nargs="*", default=["results/t1_repro_ep15",
                                                   "results/t1_repro_ep50"],
                    help="directories whose runs used the default weights")
    ap.add_argument("--out", default="results/frontier.csv")
    args = ap.parse_args()

    bases: Dict[str, Dict[str, float]] = {}
    baseline_dirs = [args.baseline_dir, args.root] + list(args.extra)
    baseline_dirs += sorted(glob.glob(os.path.join(args.root, "*")))
    for d in baseline_dirs:
        for r in (read_dir(d) if os.path.isdir(d) else []):
            if r["arm"] == BASELINE_ARM:
                bases.setdefault(r["subset"], {}).setdefault("_recs", []).append(r)
    for sub, holder in list(bases.items()):
        recs = holder.pop("_recs")
        bases[sub] = {k: _mean([r[k] for r in recs]) for k in METRICS}
    if "ref" not in bases:
        raise SystemExit(f"no '{BASELINE_ARM}' baseline for the reference stock under "
                         f"{args.baseline_dir}; the ratio columns cannot be normalised "
                         "the way STEMS normalise them")

    groups: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    dirs = sorted(glob.glob(os.path.join(args.root, "*"))) + list(args.extra)
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for r in read_dir(d):
            if r["arm"] not in ("rl+calibrated", "rl+calibrated+bc"):
                continue
            ov = r["reward_override"]
            lam = float(ov.get("lambda_indoor", DEFAULT_LAMBDA))
            xi = float(ov.get("xi", DEFAULT_XI))
            m = re.search(r"lam([0-9.]+)", os.path.basename(d))
            if m and abs(float(m.group(1)) - lam) > 1e-9:
                raise SystemExit(
                    f"{d} is named for lambda_indoor={m.group(1)} but its records were "
                    f"run with {lam}; refusing to plot a mislabelled point")
            groups[(r["arm"], lam, xi, r["episodes"], r["subset"])].append(r)

    rows = []
    for (arm, lam, xi, eps, subset), rs in sorted(groups.items(), key=lambda kv: kv[0][1:]):
        row: Dict[str, Any] = {"arm": arm, "lambda_indoor": lam, "xi": xi,
                               "episodes": eps, "subset": subset, "seeds": len(rs)}
        base = bases.get(subset)
        if base is None:
            print(f"  [skip] {arm} subset={subset}: no rule baseline for this stock")
            continue
        for k in METRICS:
            vals = [r[k] for r in rs]
            m = _mean(vals)
            short = k.replace("stems_", "")
            if k in NORMALISED_KEYS:
                denom = base.get(k, float("nan"))
                row[short] = m / denom if denom else float("nan")
                row[f"{short}_abs"] = m
            else:
                row[short] = m
            row[f"{short}_sd"] = _stdev(vals)
        rows.append(row)

    if not rows:
        raise SystemExit(f"no frontier runs found under {args.root}")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fields = list(rows[0])
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    hdr = (f"{'arm':<18}{'sub':>5}{'lambda':>8}{'xi':>6}{'eps':>5}{'n':>3}"
           f"{'cost':>9}{'discomf':>9}{'peak':>8}{'ramping':>9}{'soc viol%':>10}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['arm']:<18}{r['subset']:>5}{r['lambda_indoor']:>8.2f}{r['xi']:>6.2f}"
              f"{r['episodes']:>5}{r['seeds']:>3}"
              f"{r['cost']:>9.3f}{r['discomfort_proportion']:>9.3f}"
              f"{r['avg_daily_peak']:>8.3f}{r['ramping_rate']:>9.3f}"
              f"{r['soc_violations_pct']:>10.3f}")
    print(f"\nnormalised to '{BASELINE_ARM}' in {args.baseline_dir}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
