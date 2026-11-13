#!/usr/bin/env python3
from __future__ import annotations

import collections
import glob
import json
import sys


def fmt(v, spec="g"):
    return "--" if v is None else format(v, spec)


def mean(rows, key):
    vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
    return sum(vals) / len(vals) if vals else None


def main(paths):
    for path in paths:
        blob = json.load(open(path + "/rules.json", encoding="utf-8"))
        rows, meta = blob["results"], blob["meta"]
        print(f"\n=== {path}  cap {meta['caps']}  {len(rows)} runs")
        if rows:
            print("    available keys:", ", ".join(sorted(rows[0])))
        agg = collections.defaultdict(list)
        for r in rows:
            agg[(r.get("rule"), r.get("forecast"))].append(r)
        print(f"{'rule':<14}{'forecast':<9}{'infeas rate':>12}{'scored h':>9}"
              f"{'cap exceed':>11}{'exceed rate':>12}{'missed rate':>12}"
              f"{'unserved kWh':>13}{'peak kW':>9}")
        for (rule, fc), rs in sorted(agg.items(), key=lambda kv: (str(kv[0][1]), str(kv[0][0]))):
            ir, sh = mean(rs, "infeasible_hour_rate"), mean(rs, "infeasible_scored_hours")
            print(f"{str(rule):<14}{str(fc):<9}{fmt(ir, '.4g'):>12}{fmt(sh, '.0f'):>9}"
                  f"{fmt(mean(rs, 'cap_exceed_kwh'), '.1f'):>11}"
                  f"{fmt(mean(rs, 'cap_exceed_rate'), '.4f'):>12}"
                  f"{fmt(mean(rs, 'missed_rate'), '.4f'):>12}"
                  f"{fmt(mean(rs, 'unserved_kwh'), '.1f'):>13}"
                  f"{fmt(mean(rs, 'peak_import_kw'), '.1f'):>9}")


if __name__ == "__main__":
    args = sys.argv[1:] or sorted(glob.glob("results/e5_rules_cap*"))
    main([a.rstrip("/") for a in args if not a.endswith(".log")])