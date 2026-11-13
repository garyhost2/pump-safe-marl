from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

COMFORT_LOW_C, COMFORT_HIGH_C = 20.0, 26.0
TEMP_COLUMN = "indoor_dry_bulb_temperature"
PUBLISHED_RULE, PUBLISHED_STEMS = 0.130, 0.132


def per_building_rate(schema_path: str, root: str, horizon: int) -> Dict[str, float]:
    schema = json.load(open(schema_path))
    rates: Dict[str, float] = {}
    for name, b in schema["buildings"].items():
        t = pd.read_csv(os.path.join(root, b["energy_simulation"]))[TEMP_COLUMN].to_numpy()
        t = t[:horizon]
        rates[name] = float(((t < COMFORT_LOW_C) | (t > COMFORT_HIGH_C)).mean())
    return rates


def active_buildings(schema_path: str) -> List[str]:
    schema = json.load(open(schema_path))
    return [n for n, b in schema["buildings"].items() if b.get("include", True)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", default="citylearn_schemas/tx_travis_8b/schema.json")
    ap.add_argument("--root", default=None,
                    help="dataset directory; defaults to the cached Travis County copy")
    ap.add_argument("--horizon", type=int, default=8759)
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--measured", default="results/comfort_defect/comfort_defect.json")
    ap.add_argument("--out", default="results/comfort_is_the_dataset")
    args = ap.parse_args()

    root = args.root
    if root is None:
        from stems.environment import STEMSEnvironment
        root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), ".citylearn_cache", "citylearn", "v2.6.0b1",
            "datasets", "tx_travis_county_neighborhood")
    rates = per_building_rate(args.schema, root, args.horizon)
    ours = active_buildings(args.schema)
    subset_rate = float(np.mean([rates[n] for n in ours]))

    vals = np.array(list(rates.values()))
    rng = np.random.default_rng(0)
    draws = np.array([vals[rng.choice(len(vals), len(ours), replace=False)].mean()
                      for _ in range(args.draws)])

    report = {
        "n_buildings_in_stock": len(vals),
        "subset_size": len(ours),
        "subset_dataset_rate": subset_rate,
        "stock_pooled_rate": float(vals.mean()),
        "stock_min": float(vals.min()), "stock_max": float(vals.max()),
        "best_subset_rate": float(np.sort(vals)[:len(ours)].mean()),
        "worst_subset_rate": float(np.sort(vals)[-len(ours):].mean()),
        "resample_mean": float(draws.mean()),
        "resample_p5": float(np.percentile(draws, 5)),
        "resample_p95": float(np.percentile(draws, 95)),
        "published_rule_percentile": float((draws <= PUBLISHED_RULE).mean()),
        "published_rule": PUBLISHED_RULE, "published_stems": PUBLISHED_STEMS,
        "per_building_rate": rates,
    }

    if os.path.exists(args.measured):
        meas = {bool(r["patched"]): float(r["stems_discomfort_proportion"])
                for r in json.load(open(args.measured))}
        report["measured_patched"] = meas.get(True)
        report["measured_unpatched"] = meas.get(False)
        if meas.get(False) is not None:
            report["unpatched_equals_dataset"] = bool(
                abs(meas[False] - subset_rate) < 1e-9)
            report["unpatched_minus_dataset"] = meas[False] - subset_rate

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "comfort_is_the_dataset.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    pd.DataFrame({"building": list(rates), "dataset_discomfort": list(rates.values())}) \
        .sort_values("dataset_discomfort") \
        .to_csv(os.path.join(args.out, "per_building_discomfort.csv"), index=False)

    print(f"dataset rate for the {len(ours)} buildings in the schema : {subset_rate:.9f}")
    if report.get("measured_unpatched") is not None:
        print(f"unpatched rollout reports                          : "
              f"{report['measured_unpatched']:.9f}")
        print(f"  identical: {report['unpatched_equals_dataset']}  "
              f"(difference {report['unpatched_minus_dataset']:+.2e})")
    if report.get("measured_patched") is not None:
        print(f"patched rollout reports                            : "
              f"{report['measured_patched']:.9f}")
    print(f"\n{len(vals)}-building stock: pooled {vals.mean():.4f}, "
          f"range [{vals.min():.4f}, {vals.max():.4f}]")
    print(f"  best {len(ours)} buildings  {report['best_subset_rate']:.4f}   "
          f"worst {len(ours)}  {report['worst_subset_rate']:.4f}")
    print(f"  random {len(ours)}-building subsets: mean {draws.mean():.4f}, "
          f"5-95 pct [{np.percentile(draws, 5):.4f}, {np.percentile(draws, 95):.4f}]")
    print(f"  the published Rule-Based figure of {PUBLISHED_RULE} sits at the "
          f"{100 * report['published_rule_percentile']:.1f}th percentile of that "
          f"distribution")
    print(f"\nwrote {args.out}/")


if __name__ == "__main__":
    main()
