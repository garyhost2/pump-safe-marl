from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from itertools import combinations
from typing import Dict, List, Tuple

PLAYERS = ("lagrangian", "projection", "exact_calibration")

ARM_OF: Dict[frozenset, str] = {
    frozenset(): "mech-none+uniform",
    frozenset({"lagrangian"}): "mech-lagrangian+uniform",
    frozenset({"projection"}): "mech-projection+uniform",
    frozenset({"exact_calibration"}): "mech-none+exact",
    frozenset({"lagrangian", "projection"}): "mech-both+uniform",
    frozenset({"lagrangian", "exact_calibration"}): "mech-lagrangian+exact",
    frozenset({"projection", "exact_calibration"}): "mech-projection+exact",
    frozenset({"lagrangian", "projection", "exact_calibration"}): "mech-both+exact",
}

METRICS = ("safety_violation_rate", "soc_violation_rate", "avoidable_violation_rate",
           "cost", "peak_import_kw", "discomfort_rate", "ramping_rate")

LOWER_IS_BETTER = set(METRICS)


def load(results_dir: str) -> Dict[str, Dict[int, Dict[str, float]]]:
    out: Dict[str, Dict[int, Dict[str, float]]] = {}
    for path in glob.glob(os.path.join(results_dir, "*", "*.json")):
        name = os.path.basename(path)[: -len(".json")]
        if "__seed" not in name:
            continue
        arm, seed = name.split("__seed")
        with open(path) as fh:
            rec = json.load(fh)
        ev = rec.get("eval") or {}
        out.setdefault(arm, {})[int(seed)] = {
            m: (float(ev[m]) if ev.get(m) is not None else float("nan")) for m in METRICS}
    return out


def shapley(values: Dict[frozenset, float]) -> Dict[str, float]:
    n = len(PLAYERS)
    phi = {p: 0.0 for p in PLAYERS}
    others = {p: [q for q in PLAYERS if q != p] for p in PLAYERS}
    for p in PLAYERS:
        for k in range(n):
            for S in combinations(others[p], k):
                w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
                phi[p] += w * (values[frozenset(S) | {p}] - values[frozenset(S)])
    return phi


def _mean_ci(xs: List[float]) -> Tuple[float, float, float]:
    xs = [x for x in xs if not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262}.get(n - 1, 1.96)
    half = tcrit * sd / math.sqrt(n)
    return m, m - half, m + half


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/e3_mech")
    ap.add_argument("--out", default="results/shapley_mechanism.csv")
    args = ap.parse_args()

    data = load(args.results)
    missing = [a for a in ARM_OF.values() if a not in data]
    if missing:
        raise SystemExit(f"the factorial is incomplete, so no exact Shapley value "
                         f"exists. Missing arms: {missing}")
    seeds = sorted(set.intersection(*(set(data[a]) for a in ARM_OF.values())))
    if not seeds:
        raise SystemExit("no seed is present for every coalition")

    rows = []
    for metric in METRICS:
        per_seed: Dict[str, List[float]] = {p: [] for p in PLAYERS}
        totals: List[float] = []
        for s in seeds:
            v = {coal: data[arm][s][metric] for coal, arm in ARM_OF.items()}
            if any(math.isnan(x) for x in v.values()):
                continue
            phi = shapley(v)
            total = v[frozenset(PLAYERS)] - v[frozenset()]
            assert abs(sum(phi.values()) - total) < 1e-9 * max(1.0, abs(total)), \
                f"Shapley efficiency violated for {metric} seed {s}"
            for p in PLAYERS:
                per_seed[p].append(phi[p])
            totals.append(total)
        tot_m, tot_lo, tot_hi = _mean_ci(totals)
        biggest = max((abs(_mean_ci(per_seed[p])[0]) for p in PLAYERS), default=0.0)
        share_is_meaningful = (not math.isnan(tot_m)) and abs(tot_m) > 0.5 * biggest
        for p in PLAYERS:
            m, lo, hi = _mean_ci(per_seed[p])
            share = (m / tot_m) if share_is_meaningful else float("nan")
            rows.append({"metric": metric, "mechanism": p, "seeds": len(totals),
                         "shapley_mean": m, "ci_lo": lo, "ci_hi": hi,
                         "total_effect": tot_m, "share_of_total": share,
                         "excludes_zero": (not math.isnan(lo)) and (lo > 0 or hi < 0)})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    hdr = f"{'metric':<28}{'mechanism':<20}{'shapley':>13}{'95% CI':>26}{'share':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        ci = f"[{r['ci_lo']:+.3g}, {r['ci_hi']:+.3g}]"
        star = " *" if r["excludes_zero"] else ""
        sh = ("      --" if math.isnan(r["share_of_total"])
              else f"{r['share_of_total']:>8.1%}")
        print(f"{r['metric']:<28}{r['mechanism']:<20}{r['shapley_mean']:>13.4g}"
              f"{ci:>26}{sh}{star}")
    print("\n* interval excludes zero (paired over seeds)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
