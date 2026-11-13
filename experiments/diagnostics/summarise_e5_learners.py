#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

COLS = [("cost", "cost", "{:.0f}"),
        ("total viol", "safety_violation_rate", "{:.4g}"),
        ("soc", "soc_violation_rate", "{:.4g}"),
        ("cap", "grid_violation_rate", "{:.4g}"),
        ("peak kW", "peak_import_kw", "{:.1f}"),
        ("EV missed", "ev_missed_departure_rate", "{:.4f}"),
        ("EV short kWh", "ev_energy_shortfall_kwh", "{:.0f}"),
        ("discomfort", "discomfort_rate", "{:.4f}")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/e5_learners")
    a = ap.parse_args()

    by_cap = collections.defaultdict(lambda: collections.defaultdict(list))
    for p in sorted(glob.glob(os.path.join(a.dir, "*", "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        meta = r["meta"]
        arm = meta["arm"]
        arm = arm["name"] if isinstance(arm, dict) else arm
        by_cap[meta["scenario"]["grid_cap_kw"]][arm].append(r["eval"])

    for cap in sorted(by_cap, reverse=True):
        arms = by_cap[cap]
        n = sum(len(v) for v in arms.values())
        print(f"\n=== district cap {cap} kW — {n} records, {len(arms)} arms")
        print(f"{'arm':<16}{'seeds':>6}" + "".join(f"{c[0]:>14}" for c in COLS))
        for arm in sorted(arms):
            rows = arms[arm]
            cells = ""
            for _, key, spec in COLS:
                vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
                cells += f"{(spec.format(float(np.mean(vals))) if vals else '--'):>14}"
            print(f"{arm:<16}{len(rows):>6}" + cells)


if __name__ == "__main__":
    main()