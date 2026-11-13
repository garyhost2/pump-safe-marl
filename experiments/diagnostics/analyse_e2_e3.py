#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
KPIS = ["safety_violation_rate", "soc_violation_rate", "grid_violation_rate",
        "power_violation_rate", "cost", "discomfort_rate", "peak_import_kw",
        "cap_exceedance_kwh", "emission", "avg_daily_peak"]


def load(folder: Path) -> List[Dict]:
    out = []
    for p in sorted(folder.rglob("*.json")):
        if p.name in {"summary.json", "e2_caps.json"}:
            continue
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if r.get("status") != "ok":
            continue
        meta, ev = r.get("meta", {}), r.get("eval", {})
        arm = meta.get("arm")
        arm = arm.get("name") if isinstance(arm, dict) else arm
        sc = meta.get("scenario", {})
        row = {"arm": arm, "seed": int(meta.get("seed", 0)), "key": sc.get("key"),
               "subset": sc.get("subset_seed"), "cap": sc.get("grid_cap_kw"),
               "steps": r.get("eval_steps")}
        for k in KPIS:
            v = ev.get(k)
            row[k] = float(v) if isinstance(v, (int, float)) else np.nan
        out.append(row)
    return out


def t_crit(df: int) -> float:
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}
    if df <= 0:
        return float("nan")
    if df in table:
        return table[df]
    keys = sorted(table)
    return table[min(keys, key=lambda k: abs(k - df))]


def scenario_means(rows: List[Dict], kpi: str) -> Dict[object, float]:
    by: Dict[object, List[float]] = {}
    for r in rows:
        if np.isfinite(r.get(kpi, np.nan)):
            by.setdefault(r["subset"], []).append(r[kpi])
    return {s: float(np.mean(v)) for s, v in by.items()}


def paired_contrast(rows: List[Dict], a: str, b: str, kpi: str) -> Dict[str, float]:
    ma = scenario_means([r for r in rows if r["arm"] == a], kpi)
    mb = scenario_means([r for r in rows if r["arm"] == b], kpi)
    shared = sorted(set(ma) & set(mb), key=lambda x: (x is None, x))
    d = np.array([ma[s] - mb[s] for s in shared], dtype=float)
    n = d.size
    out = {"n_scenarios": n, "mean_diff": float(d.mean()) if n else float("nan")}
    if n >= 2:
        sd = float(d.std(ddof=1))
        se = sd / np.sqrt(n)
        tc = t_crit(n - 1)
        out.update(sd=sd, se=se, lo=out["mean_diff"] - tc * se, hi=out["mean_diff"] + tc * se,
                   t=(out["mean_diff"] / se) if se > 0 else float("nan"))
    return out


def holm(pairs: List[tuple]) -> List[tuple]:
    srt = sorted(pairs, key=lambda x: x[1])
    m, out, prev = len(srt), [], 0.0
    for i, (lab, p) in enumerate(srt):
        adj = max(prev, min(1.0, (m - i) * p))
        prev = adj
        out.append((lab, p, adj, adj <= 0.05))
    return out


def report_e2(rows: List[Dict], fh) -> None:
    subsets = sorted({r["subset"] for r in rows}, key=lambda x: (x is None, x))
    arms = sorted({r["arm"] for r in rows})
    print(f"\n## E2 — {len(rows)} records, {len(subsets)} scenarios, arms: {', '.join(arms)}", file=fh)
    print(f"\nReplication unit: building subset (n={len(subsets)}). Seeds averaged within.", file=fh)
    for kpi in ["safety_violation_rate", "soc_violation_rate", "grid_violation_rate", "cost"]:
        print(f"\n### {kpi}\n\n| arm | " + " | ".join(f"subset {s}" for s in subsets) +
              " | mean | 95% CI |", file=fh)
        print("|" + "---|" * (len(subsets) + 3), file=fh)
        for a in arms:
            m = scenario_means([r for r in rows if r["arm"] == a], kpi)
            vals = np.array([m.get(s, np.nan) for s in subsets], dtype=float)
            ok = vals[np.isfinite(vals)]
            cells = " | ".join("—" if not np.isfinite(v) else f"{v:.6g}" for v in vals)
            if ok.size >= 2:
                se = ok.std(ddof=1) / np.sqrt(ok.size)
                tc = t_crit(ok.size - 1)
                ci = f"[{ok.mean()-tc*se:.6g}, {ok.mean()+tc*se:.6g}]"
            else:
                ci = "n/a"
            print(f"| {a} | {cells} | {ok.mean():.6g} | {ci} |" if ok.size else
                  f"| {a} | {cells} | — | n/a |", file=fh)
    print("\n### Contrasts (paired across scenarios)\n", file=fh)
    print("| contrast | KPI | mean diff | 95% CI | n |", file=fh)
    print("|---|---|---|---|---|", file=fh)
    for a, b in [("rl+calibrated", "rbc+calibrated"), ("rl+calibrated", "rl"),
                 ("rl+calibrated", "rl+basic")]:
        if a not in arms or b not in arms:
            continue
        for kpi in ["cost", "safety_violation_rate", "soc_violation_rate"]:
            c = paired_contrast(rows, a, b, kpi)
            ci = (f"[{c['lo']:.6g}, {c['hi']:.6g}]" if "lo" in c else "n/a")
            print(f"| {a} − {b} | {kpi} | {c['mean_diff']:.6g} | {ci} | {c['n_scenarios']} |", file=fh)


def report_e3(rows: List[Dict], fh) -> None:
    arms = sorted({r["arm"] for r in rows})
    print(f"\n## E3 — {len(rows)} records, {len(arms)} arms\n", file=fh)
    print("Mechanism × battery-model calibration. Seeds are the only variation here "
          "(single scenario), so these are descriptive: no CI is reported.\n", file=fh)
    print("| mechanism | calibration | SOC rate | cap rate | total | cost | seeds |", file=fh)
    print("|---|---|---|---|---|---|---|", file=fh)
    for a in arms:
        sub = [r for r in rows if r["arm"] == a]
        if not sub:
            continue
        mech, _, calib = a.partition("+")
        g = lambda k: float(np.nanmean([r[k] for r in sub]))
        print(f"| {mech.replace('mech-','')} | {calib} | {g('soc_violation_rate'):.6g} | "
              f"{g('grid_violation_rate'):.6g} | {g('safety_violation_rate'):.6g} | "
              f"{g('cost'):.6g} | {len(sub)} |", file=fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", default="results/e2_subsets")
    ap.add_argument("--e3", default="results/e3_mech")
    ap.add_argument("--out", default="results/e2_e3_summary.md")
    args = ap.parse_args()

    e2 = load(REPO / args.e2)
    e3 = load(REPO / args.e3)
    out = REPO / args.out
    with out.open("w", encoding="utf-8") as fh:
        print("# E2 / E3 summary", file=fh)
        if e2:
            report_e2(e2, fh)
        else:
            print("\n## E2 — no completed records yet", file=fh)
        if e3:
            report_e3(e3, fh)
        else:
            print("\n## E3 — no completed records yet", file=fh)
    print(f"E2 records: {len(e2)} | E3 records: {len(e3)} -> {out}")
    print(out.read_text(encoding="utf-8")[:3000])


if __name__ == "__main__":
    main()
