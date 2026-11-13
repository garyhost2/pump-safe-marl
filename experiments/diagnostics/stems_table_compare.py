from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stems.stems_table import NORMALISED_KEYS

PUBLISHED: Dict[str, Dict[str, float]] = {
    "Rule-Based": dict(cost=1.000, emission=1.000, peak=1.000, consumption=1.000,
                       ramping=1.000, discomfort=0.130, violations=0.351),
    "MPC": dict(cost=0.872, emission=0.914, peak=0.983, consumption=0.975,
                ramping=0.981, discomfort=0.654, violations=0.330),
    "Single-Agent SAC": dict(cost=0.824, emission=0.867, peak=0.856, consumption=0.813,
                             ramping=0.925, discomfort=0.485, violations=0.223),
    "MADDPG": dict(cost=0.795, emission=0.817, peak=0.854, consumption=0.856,
                   ramping=0.936, discomfort=0.357, violations=0.197),
    "MetaEMS": dict(cost=0.836, emission=0.804, peak=0.884, consumption=0.835,
                    ramping=0.907, discomfort=0.396, violations=0.231),
    "MARLISA": dict(cost=0.847, emission=0.875, peak=0.826, consumption=0.838,
                    ramping=0.956, discomfort=0.237, violations=0.214),
    "MADCQ": dict(cost=0.814, emission=0.837, peak=0.862, consumption=0.823,
                  ramping=0.906, discomfort=0.325, violations=0.155),
    "D-MAPPO": dict(cost=0.803, emission=0.834, peak=0.841, consumption=0.816,
                    ramping=0.912, discomfort=0.152, violations=0.198),
    "STEMS": dict(cost=0.792, emission=0.824, peak=0.821, consumption=0.805,
                  ramping=0.883, discomfort=0.132, violations=0.056),
}


ARM_TO_PUBLISHED = {
    "rbc": "Rule-Based",
    "mpc": "MPC",
    "sac": "Single-Agent SAC",
    "maddpg": "MADDPG",
    "metaems": "MetaEMS",
    "marlisa": "MARLISA",
    "madcq": "MADCQ",
    "dmappo": "D-MAPPO",
    "mpc-unshielded": "MPC",
    "sac-unshielded": "Single-Agent SAC",
    "maddpg-unshielded": "MADDPG",
    "metaems-unshielded": "MetaEMS",
    "marlisa-unshielded": "MARLISA",
    "madcq-unshielded": "MADCQ",
    "dmappo-unshielded": "D-MAPPO",
}

COLUMNS = [
    ("cost", "stems_cost"),
    ("emission", "stems_emission"),
    ("peak", "stems_avg_daily_peak"),
    ("consumption", "stems_electricity_consumption"),
    ("ramping", "stems_ramping_rate"),
    ("discomfort", "stems_discomfort_proportion"),
    ("violations_soc_equipment", "stems_safety_violations_soc_equipment"),
    ("violations_with_comfort", "stems_safety_violations_with_comfort"),
    ("soc_violations", "stems_soc_violations_pct"),
]

BASELINE_ARM = "rbc"


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def _stdev(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def load(results_dir) -> Dict[str, Dict[str, List[float]]]:
    dirs = [results_dir] if isinstance(results_dir, str) else list(results_dir)
    by_arm: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    scenarios: set = set()
    found = 0
    for root, _, files in ((r, d, f) for one in dirs for r, d, f in os.walk(one)):
        for fn in files:
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(root, fn)) as fh:
                rec = json.load(fh)
            if rec.get("status") not in (None, "ok"):
                continue
            meta_arm = (rec.get("meta") or {}).get("arm")
            if isinstance(meta_arm, dict):
                arm = meta_arm.get("name") or fn.split("__")[0]
            else:
                arm = meta_arm or fn.split("__")[0]
            ev = rec.get("eval") or {}
            if not any(k in ev for _, k in COLUMNS):
                continue
            found += 1
            scenarios.add(os.path.basename(root))
            for _, key in COLUMNS:
                if ev.get(key) is not None:
                    by_arm[arm][key].append(float(ev[key]))
    if not found:
        raise SystemExit(f"no scored records with STEMS metrics under {dirs}. "
                         "Were these runs produced before stems/stems_table.py was wired in?")
    if len(scenarios) > 1:
        raise SystemExit(
            "refusing to pool results measured on different scenarios: "
            + ", ".join(sorted(scenarios))
            + ". A ratio is only meaningful when numerator and denominator "
              "controlled the same buildings over the same window.")
    return by_arm


def build_rows(by_arm: Dict[str, Dict[str, List[float]]],
               baseline: str = BASELINE_ARM) -> List[Dict[str, Any]]:
    if baseline not in by_arm:
        raise SystemExit(f"the normalisation baseline '{baseline}' is not in these "
                         f"results; STEMS normalises to rule-based and so must we. "
                         f"Arms present: {sorted(by_arm)}")
    base = {k: _mean(v) for k, v in by_arm[baseline].items()}
    rows: List[Dict[str, Any]] = []
    for arm in sorted(by_arm):
        row: Dict[str, Any] = {"source": "measured", "row": arm,
                               "seeds": len(next(iter(by_arm[arm].values()), []))}
        for label, key in COLUMNS:
            vals = by_arm[arm].get(key, [])
            m = _mean(vals)
            if key in NORMALISED_KEYS:
                denom = base.get(key, float("nan"))
                row[label] = m / denom if denom else float("nan")
                row[f"{label}_abs"] = m
            else:
                row[label] = m
            row[f"{label}_sd"] = _stdev(vals)
        rows.append(row)
    for name, pub in PUBLISHED.items():
        rows.append({"source": "published(arXiv:2510.14112)", "row": name, "seeds": 5,
                     "cost": pub["cost"], "emission": pub["emission"],
                     "peak": pub["peak"], "consumption": pub["consumption"],
                     "ramping": pub["ramping"], "discomfort": pub["discomfort"],
                     "violations_soc_equipment": pub["violations"] * 100.0,
                     "violations_with_comfort": None, "soc_violations": None})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, nargs="+",
                    help="one or more results directories, pooled. Pass the "
                         "unshielded grid together with the grid holding the rule "
                         "it is normalised to; they must share a scenario.")
    ap.add_argument("--out", default="results/stems_table_compare.csv")
    ap.add_argument("--baseline", default=BASELINE_ARM,
                    help="arm the five ratio columns are normalised to")
    args = ap.parse_args()

    rows = build_rows(load(args.results), baseline=args.baseline)
    fields = ["source", "row", "seeds"]
    for label, key in COLUMNS:
        fields.append(label)
        if key in NORMALISED_KEYS:
            fields.append(f"{label}_abs")
        fields.append(f"{label}_sd")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    head = f"{'source':<28}{'row':<24}" + "".join(f"{lbl:>12}" for lbl, _ in COLUMNS[:7])
    print(head)
    print("-" * len(head))
    for r in rows:
        cells = "".join(
            ("{:>12}".format("--") if r.get(lbl) is None or
             (isinstance(r.get(lbl), float) and math.isnan(r[lbl]))
             else "{:>12.3f}".format(r[lbl]))
            for lbl, _ in COLUMNS[:7])
        print(f"{r['source']:<28}{r['row']:<24}{cells}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
