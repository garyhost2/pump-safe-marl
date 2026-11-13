#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

NAMED = ("reactive", "price_aware", "thermia")
COLS = ["cost", "objective", "elec_kwh", "cost_dhw", "viol_end_of_hour", "viol_intra_hour", "weeks_missed",
        "weeks_total", "intervention_rate", "filter_share_dhw_heat", "filter_share_disinfection", "starts",
        "sh_degree_hours", "violations_unexplained"]


def objective(m, w, cur, w_start=0.1, w_dis=50.0, w_sh=5.0):
    f = 1.0 / 10.28 if cur.upper() == "USD" else 1.0
    return m["cost"] + f * (w * m["viol_intra_hour"] + w_start * m["starts"] + w_dis * m["weeks_missed"] + w_sh * m["sh_degree_hours"])


def collect(runs: Path, weights=(20.0, 50.0)):
    rows = []
    for d in sorted((runs / "baselines").glob("*")):
        for f in sorted(d.glob("*__*.json")):
            j = json.loads(f.read_text())
            name = f.stem.split("__")[0]
            for w in weights:
                r = {"set": d.name, "w_comfort": w, "controller": name, "kind": "mpc" if name.startswith("mpc") else "rule",
                     "learner": "", "arm": "", "seed": "", "eval": "filtered" if j["filtered"] else "raw"}
                m = dict(j["metrics"])
                m["objective"] = objective(m, w, j["currency"])
                r.update({k: m.get(k, np.nan) for k in COLS})
                rows.append(r)
    for ds in ("sweden", "texas"):
        for run in sorted((runs / ds).glob("*_seed*")) if (runs / ds).is_dir() else []:
            if not (run / "done.json").exists():
                continue
            meta = json.loads((run / "meta.json").read_text())
            w = float(meta["reward_spec"]["w_comfort_h"])
            for f in sorted(run.glob("eval_*_*.json")):
                setname, mode = f.stem[len("eval_"):].rsplit("_", 1)
                setname = setname.replace("texas_eval", "texas")
                j = json.loads(f.read_text())
                m = dict(j["metrics"])
                m["objective"] = objective(m, w, meta["currency"], meta["reward_spec"]["w_start"],
                                           meta["reward_spec"]["w_dis"], meta["reward_spec"]["w_sh"])
                r = {"set": setname, "w_comfort": w, "controller": f"{meta['learner']}_{meta['arm']}", "kind": "rl",
                     "learner": meta["learner"], "arm": meta["arm"], "seed": meta["seed"], "eval": mode,
                     "steps": meta["steps_budget"], "train_years": "+".join(map(str, meta.get("train_years", []))) if ds == "sweden" else "texas_h1"}
                r.update({k: m.get(k, np.nan) for k in COLS})
                rows.append(r)
    return rows


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def paired(G, ref, col):
    from scipy.stats import t as tdist
    x = np.array([g[col] - ref[col] for g in G])
    n = len(x)
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    h = float(tdist.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return float(x.mean()), sd, float(x.mean() - h), float(x.mean() + h), int((x < 0).sum()), n


def tables(rows):
    out, summ, prs = [], {}, []
    for s in sorted({r["set"] for r in rows}):
        for w in sorted({r["w_comfort"] for r in rows}):
            R = [r for r in rows if r["set"] == s and r["w_comfort"] == w and r["eval"] == "filtered"]
            if not any(r["kind"] == "rl" for r in R):
                continue
            base = {r["controller"]: r for r in R if r["kind"] != "rl"}
            named = {k: base[k] for k in NAMED if k in base}
            best = min(named, key=lambda k: named[k]["objective"])
            mpc = next((base[k] for k in base if k.startswith("mpc_rho")), None)
            out.append(f"\n#### {s}, comfort weight {w:g} SEK per building-hour below 45 °C\n")
            out.append(f"Best named rule on the objective: **{best}** (objective {fmt(named[best]['objective'])}, cost "
                       f"{fmt(named[best]['cost'])}). MPC = 24-h LP, causal draw forecast.\n")
            hdr = ("| controller | n | objective | Δobj vs best rule [95 % CI] (better/n) | Δobj vs MPC [95 % CI] (better/n) | cost | "
                   "Δcost vs best rule [95 % CI] | DHW cost | viol. end-of-hour | viol. intra-hour | missed weeks | "
                   "intervention rate | filter share DHW heat | filter share disinf. | starts | SH K h |")
            out.append(hdr)
            out.append("|" + "|".join(["---"] * (hdr.count("|") - 1)) + "|")
            for k in [*NAMED, "null"] + sorted(x for x in base if x.startswith("mpc")):
                if k not in base:
                    continue
                r = base[k]
                out.append(f"| {k} | 1 | {fmt(r['objective'])} | {fmt(r['objective'] - named[best]['objective'])} | "
                           f"{fmt(r['objective'] - mpc['objective']) if mpc else '–'} | {fmt(r['cost'])} | "
                           f"{fmt(r['cost'] - named[best]['cost'])} | {fmt(r['cost_dhw'])} | {fmt(r['viol_end_of_hour'], 0)} | "
                           f"{fmt(r['viol_intra_hour'], 0)} | {r['weeks_missed']}/{r['weeks_total']} | {fmt(r['intervention_rate'], 3)} | "
                           f"{fmt(r['filter_share_dhw_heat'], 3)} | {fmt(r['filter_share_disinfection'], 3)} | {fmt(r['starts'], 0)} | "
                           f"{fmt(r['sh_degree_hours'], 0)} |")
            for k in sorted({r["controller"] for r in R if r["kind"] == "rl"}):
                G = sorted([r for r in R if r["controller"] == k], key=lambda r: r["seed"])
                n = len(G)
                ms = lambda c: (float(np.mean([g[c] for g in G])), float(np.std([g[c] for g in G], ddof=1)) if n > 1 else np.nan)
                pb = paired(G, named[best], "objective")
                pm = paired(G, mpc, "objective") if mpc else None
                pc = paired(G, named[best], "cost")
                for ref, p, col in (("best_rule", pb, "objective"), ("mpc", pm, "objective"), ("best_rule", pc, "cost")):
                    if p:
                        prs.append({"set": s, "w_comfort": w, "controller": k, "ref": ref if ref == "mpc" else best,
                                    "metric": col, "n": p[5], "diff_mean": p[0], "diff_sd": p[1], "ci95_lo": p[2],
                                    "ci95_hi": p[3], "seeds_below": p[4]})
                pmf = lambda c, nd=1: f"{fmt(ms(c)[0], nd)} ± {fmt(ms(c)[1], nd)}"
                pmtxt = f"{fmt(pm[0])} [{fmt(pm[2])}, {fmt(pm[3])}] ({pm[4]}/{n})" if pm else "–"
                out.append(f"| {k} | {n} | {pmf('objective')} | {fmt(pb[0])} [{fmt(pb[2])}, {fmt(pb[3])}] ({pb[4]}/{n}) | {pmtxt} | "
                           f"{pmf('cost')} | {fmt(pc[0])} [{fmt(pc[2])}, {fmt(pc[3])}] | {pmf('cost_dhw')} | {pmf('viol_end_of_hour', 0)} | "
                           f"{pmf('viol_intra_hour', 0)} | {pmf('weeks_missed')}/{G[0]['weeks_total']} | {pmf('intervention_rate', 3)} | "
                           f"{pmf('filter_share_dhw_heat', 3)} | {pmf('filter_share_disinfection', 3)} | {pmf('starts', 0)} | "
                           f"{pmf('sh_degree_hours', 0)} |")
                summ.setdefault(s, {}).setdefault(str(w), {})[k] = {"n": n, "objective": ms("objective"), "cost": ms("cost"),
                                                                    "viol_intra_hour": ms("viol_intra_hour"), "starts": ms("starts"),
                                                                    "dobj_best_rule": pb[:5], "dobj_mpc": pm[:5] if pm else None,
                                                                    "dcost_best_rule": pc[:5]}
            summ.setdefault(s, {}).setdefault(str(w), {})["_ref"] = {"best_rule": best, **{k: {c: base[k][c] for c in COLS} for k in base}}
    return "\n".join(out), summ, prs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=".scratch/fair")
    ap.add_argument("--out", default="experiments/dhw_v2_rl/results_fair")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(Path(a.runs))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "per_seed_fair.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    md, summ, prs = tables(rows)
    with open(out / "paired_fair.csv", "w", newline="") as f:
        if prs:
            w = csv.DictWriter(f, fieldnames=list(prs[0]))
            w.writeheader()
            w.writerows(prs)
    (out / "tables_fair.md").write_text(md + "\n")
    (out / "summary_fair.json").write_text(json.dumps(summ, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
