#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

COLS = ["cost", "objective", "elec_kwh", "cost_dhw", "viol_end_of_hour", "viol_intra_hour", "weeks_missed", "weeks_total",
        "intervention_rate", "filter_share_dhw_heat", "filter_share_disinfection", "starts", "sh_degree_hours",
        "elec_dhw_kwh", "elec_sh_kwh", "time_below_floor_h", "dis_supply_hours", "violations_unexplained"]
NAMED_RULES = ("reactive", "price_aware", "thermia")


def collect(runs: Path):
    rows = []
    for ds in sorted(p for p in runs.iterdir() if p.is_dir()):
        for f in sorted((ds / "rules").glob("*__*.json")) if (ds / "rules").is_dir() else []:
            d = json.loads(f.read_text())
            r = {"dataset": ds.name, "controller": d["controller"], "kind": "rule", "learner": "", "arm": "",
                 "seed": "", "eval": "filtered" if d["filtered"] else "raw"}
            r.update({k: d["metrics"].get(k, np.nan) for k in COLS})
            rows.append(r)
        for run in sorted(ds.glob("*_seed*")):
            if not (run / "done.json").exists():
                continue
            meta = json.loads((run / "meta.json").read_text())
            curve = list(csv.DictReader(open(run / "curve.csv")))
            for mode in ("filtered", "raw"):
                d = json.loads((run / f"eval_{mode}.json").read_text())
                r = {"dataset": ds.name, "controller": f"{meta['learner']}_{meta['arm']}", "kind": "rl",
                     "learner": meta["learner"], "arm": meta["arm"], "seed": meta["seed"], "eval": mode,
                     "episodes": meta["episodes"], "train_last_cost": float(curve[-1]["cost"]),
                     "train_last_intervention_rate": float(curve[-1].get("intervention_rate") or np.nan)}
                r.update({k: d["metrics"].get(k, np.nan) for k in COLS})
                rows.append(r)
    return rows


def lp(runs: Path, ds: str):
    f = runs / ds / "rules" / "lp_bound.json"
    return json.loads(f.read_text())["cost_dhw_lb_total"] if f.exists() else None


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def tables(rows, runs: Path):
    out, summ = [], {}
    for ds in sorted({r["dataset"] for r in rows}):
        R = [r for r in rows if r["dataset"] == ds and r["eval"] == "filtered"]
        rules = {r["controller"]: r for r in R if r["kind"] == "rule"}
        named = {k: v for k, v in rules.items() if k in NAMED_RULES}
        best = min(named, key=lambda k: named[k]["cost"]) if named else None
        bc = named[best]["cost"] if best else np.nan
        lpb = lp(runs, ds)
        S = {"best_rule": best, "best_rule_cost": bc, "lp_dhw_cost_lb": lpb, "rules": {}, "rl": {}}
        out.append(f"\n### {ds}: held-out evaluation behind the filter\n")
        out.append(f"Best named rule (lowest cost): **{best}** ({fmt(bc)}). LP perfect-foresight lower bound "
                   f"on the DHW part of the cost: {fmt(lpb, 2)}.\n")
        hdr = ("| controller | n | cost | Δcost vs best rule (paired, mean ± sd) | 95 % CI | seeds cheaper | objective (cost + penalties) | elec kWh | DHW cost | "
               "viol. end-of-hour | viol. intra-hour | missed weeks / total | intervention rate | "
               "filter share DHW heat | filter share disinf. hours | starts |")
        out.append(hdr)
        out.append("|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
        for k in list(NAMED_RULES) + ["null"]:
            if k not in rules:
                continue
            r = rules[k]
            S["rules"][k] = {c: r[c] for c in COLS}
            d = r["cost"] - bc
            out.append(f"| {k} (rule) | 1 | {fmt(r['cost'])} | {fmt(d)} | – | – | {fmt(r['objective'])} | {fmt(r['elec_kwh'], 0)} | {fmt(r['cost_dhw'])} | "
                       f"{fmt(r['viol_end_of_hour'], 0)} | {fmt(r['viol_intra_hour'], 0)} | {r['weeks_missed']}/{r['weeks_total']} | "
                       f"{fmt(r['intervention_rate'], 3)} | {fmt(r['filter_share_dhw_heat'], 3)} | "
                       f"{fmt(r['filter_share_disinfection'], 3)} | {fmt(r['starts'], 0)} |")
        for k in sorted({r["controller"] for r in R if r["kind"] == "rl"}):
            G = sorted([r for r in R if r["controller"] == k], key=lambda r: r["seed"])
            n = len(G)
            ms = lambda c: (float(np.mean([g[c] for g in G])), float(np.std([g[c] for g in G], ddof=1)) if n > 1 else np.nan)
            dif = np.array([g["cost"] - bc for g in G])
            dm, dsd = float(dif.mean()), float(dif.std(ddof=1)) if n > 1 else np.nan
            from scipy.stats import t as tdist
            half = float(tdist.ppf(0.975, n - 1) * dsd / np.sqrt(n)) if n > 1 else np.nan
            cheaper = int((dif < 0).sum())
            row = {c: ms(c) for c in COLS}
            S["rl"][k] = {"n": n, "seeds": [g["seed"] for g in G], "dcost_mean": dm, "dcost_sd": dsd,
                          "dcost_ci95": [dm - half, dm + half], "seeds_cheaper": cheaper,
                          "dcost_per_seed": dif.tolist(), **{c: row[c] for c in COLS}}
            pm = lambda c, nd=1: f"{fmt(row[c][0], nd)} ± {fmt(row[c][1], nd)}"
            out.append(f"| {k} | {n} | {pm('cost')} | {fmt(dm)} ± {fmt(dsd)} | [{fmt(dm - half)}, {fmt(dm + half)}] | "
                       f"{cheaper}/{n} | {pm('objective')} | {pm('elec_kwh', 0)} | {pm('cost_dhw')} | {pm('viol_end_of_hour', 0)} | "
                       f"{pm('viol_intra_hour', 0)} | {pm('weeks_missed', 1)}/{G[0]['weeks_total']} | "
                       f"{pm('intervention_rate', 3)} | {pm('filter_share_dhw_heat', 3)} | "
                       f"{pm('filter_share_disinfection', 3)} | {pm('starts', 0)} |")
        Rr = [r for r in rows if r["dataset"] == ds and r["eval"] == "raw"]
        if Rr:
            out.append(f"\nDiagnostic, same policies on the RAW plant (no filter) on the held-out split:\n")
            out.append("| controller | n | cost | viol. intra-hour | missed weeks / total | SH degree-hours |")
            out.append("|---|---|---|---|---|---|")
            for k in sorted({r["controller"] for r in Rr}, key=lambda k: (k not in rules, k)):
                G = [r for r in Rr if r["controller"] == k]
                n = len(G)
                m = lambda c: float(np.mean([g[c] for g in G]))
                s = lambda c: float(np.std([g[c] for g in G], ddof=1)) if n > 1 else float("nan")
                out.append(f"| {k} | {n} | {fmt(m('cost'))} ± {fmt(s('cost'))} | {fmt(m('viol_intra_hour'), 0)} ± {fmt(s('viol_intra_hour'), 0)} | "
                           f"{fmt(m('weeks_missed'), 1)}/{G[0]['weeks_total']} | {fmt(m('sh_degree_hours'), 0)} |")
                S.setdefault("raw", {})[k] = {"n": n, "cost": m("cost"), "viol_intra_hour": m("viol_intra_hour"),
                                              "weeks_missed": m("weeks_missed")}
        summ[ds] = S
    return "\n".join(out), summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=".scratch/final")
    ap.add_argument("--out", default="experiments/dhw_v2_rl/results")
    a = ap.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(runs)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "per_seed.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md, summ = tables(rows, runs)
    (out / "tables.md").write_text(md + "\n")
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
