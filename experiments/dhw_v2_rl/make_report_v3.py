#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

BANDS = ((20, 23), (20, 22), (21, 22))
RULES = ("price_aware", "preheat_tuned", "reactive", "thermia")
SETS = ("karlstad_2025", "lulea_2025", "texas", "karlstad_2024")


def obj(m, cur, band):
    fx = 1.0 / 10.28 if cur.upper() == "USD" else 1.0
    lo, hi = band
    return m["cost"] + fx * (20.0 * m["viol_intra_hour"] + 0.1 * m["starts"] + 50.0 * m["weeks_missed"]
                             + 5.0 * m[f"band_{lo:g}_{hi:g}_below_kh"])


def row(name, kind, s, m, cur, seed=""):
    r = {"set": s, "controller": name, "kind": kind, "seed": seed, "currency": cur}
    for b in BANDS:
        r[f"objective_{b[0]}_{b[1]}"] = obj(m, cur, b)
    for k in ("cost", "elec_kwh", "elec_sh_kwh", "cost_dhw", "viol_end_of_hour", "viol_intra_hour", "weeks_missed",
              "weeks_total", "starts", "intervention_rate", "filter_share_dhw_heat", "filter_share_disinfection",
              "violations_unexplained") + tuple(f"band_{lo}_{hi}_{w}_kh" for lo, hi in BANDS for w in ("below", "above")):
        r[k] = m.get(k, np.nan)
    if "mpc_lp_fail_hours" in m:
        r["mpc_lp_fail_hours"] = m["mpc_lp_fail_hours"]
    return r


def collect(runs: Path):
    rows = []
    for d in sorted((runs / "baselines").glob("*")):
        for f in sorted(d.glob("*.json")):
            j = json.loads(f.read_text())
            kind = "mpc" if j["name"].startswith(("mpc", "pf")) else "rule"
            if j["name"].startswith("preheat_k"):
                kind = "tuning"
            rows.append(row(j["name"], kind, d.name, j["metrics"], j["currency"]))
    for r in sorted((runs / "runs").glob("v3*_seed*")):
        if not (r / "done.json").exists():
            continue
        meta = json.loads((r / "meta.json").read_text())
        for f in sorted(r.glob("eval_*.json")):
            j = json.loads(f.read_text())
            s = f.stem[len("eval_"):]
            rows.append(row(f"v3{meta['variant']}", "rl", s, j["metrics"], j["currency"], meta["seed"]))
    return rows


def fmt(x, nd=0):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}".replace(",", " ")


def paired(G, ref, col):
    from scipy.stats import t as tdist
    x = np.array([g[col] - ref[col] for g in G])
    n = len(x)
    sd = float(x.std(ddof=1)) if n > 1 else np.nan
    h = float(tdist.ppf(0.975, n - 1) * sd / np.sqrt(n)) if n > 1 else np.nan
    return float(x.mean()), sd, float(x.mean() - h), float(x.mean() + h), int((x < 0).sum()), n


def tables(rows):
    out, summ, prs = [], {}, []
    out.append("## Headroom (filtered, held-out sets)\n")
    out.append("Objective per comfort band (columns); headroom = Price-aware minus the controller (positive = "
               "the controller is better).\n")
    for s in SETS:
        R = {r["controller"]: r for r in rows if r["set"] == s and r["kind"] in ("rule", "mpc")}
        if "price_aware" not in R:
            continue
        cur = R["price_aware"]["currency"]
        out.append(f"\n### {s} ({cur})\n")
        out.append("| controller | cost | obj [20,23] | obj [20,22] | obj [21,22] | headroom vs Price-aware [20,23] / [20,22] / [21,22] | "
                   "DHW viol. h | K h <20 | K h <21 | starts | DHW cost |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|")
        order = ["price_aware", "preheat_tuned", "reactive", "thermia", "null", "mpc_setpoint", "mpc_band_20_23", "mpc_band_20_22",
                 "mpc_band_21_22", "pf_band_20_23", "pf_band_20_22", "pf_band_21_22"]
        pa = R["price_aware"]
        S = summ.setdefault(s, {})
        for k in order:
            if k not in R:
                continue
            r = R[k]
            hr = [pa[f"objective_{a}_{b}"] - r[f"objective_{a}_{b}"] for a, b in BANDS]
            S[k] = {"cost": r["cost"], **{f"objective_{a}_{b}": r[f"objective_{a}_{b}"] for a, b in BANDS},
                    "headroom_vs_price_aware": hr, "starts": r["starts"]}
            out.append(f"| {k} | {fmt(r['cost'], 1 if cur == 'USD' else 0)} | " + " | ".join(fmt(r[f'objective_{a}_{b}'], 1 if cur == 'USD' else 0) for a, b in BANDS)
                       + f" | {' / '.join(fmt(h, 1 if cur == 'USD' else 0) for h in hr)} | {fmt(r['viol_intra_hour'])} | "
                       f"{fmt(r['band_20_23_below_kh'])} | {fmt(r['band_21_22_below_kh'])} | {fmt(r['starts'])} | {fmt(r['cost_dhw'], 1 if cur == 'USD' else 0)} |")
    out.append("\n## RL v3 (filtered; objective at the [20, 23] band)\n")
    col = "objective_20_23"
    for s in SETS:
        R = [r for r in rows if r["set"] == s]
        rl = [r for r in R if r["kind"] == "rl"]
        if not rl:
            continue
        base = {r["controller"]: r for r in R if r["kind"] in ("rule", "mpc")}
        named = {k: base[k] for k in RULES if k in base}
        best = min(named, key=lambda k: named[k][col])
        refs = {"best rule (" + best + ")": named[best], "MPC-band (causal)": base.get("mpc_band_20_23"),
                "PF-oracle-band": base.get("pf_band_20_23")}
        out.append(f"\n### {s}\n")
        out.append(f"References: " + "; ".join(f"{k} {fmt(v[col])}" for k, v in refs.items() if v) + "\n")
        out.append("| variant | n | objective | Δ vs best rule [95 % CI] (better/n) | Δ vs MPC-band [95 % CI] (better/n) | Δ vs PF | cost | "
                   "DHW viol. h | K h <20 | starts | intervention rate | filter share DHW heat |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for v in sorted({r["controller"] for r in rl}):
            G = sorted([r for r in rl if r["controller"] == v], key=lambda r: r["seed"])
            n = len(G)
            ms = lambda c: (float(np.mean([g[c] for g in G])), float(np.std([g[c] for g in G], ddof=1)) if n > 1 else np.nan)
            cells = []
            for name, ref in refs.items():
                if ref is None:
                    cells.append("–"); continue
                p = paired(G, ref, col)
                prs.append({"set": s, "variant": v, "ref": name, "metric": col, "n": n, "diff_mean": p[0], "diff_sd": p[1],
                            "ci95_lo": p[2], "ci95_hi": p[3], "seeds_below": p[4]})
                cells.append(f"{fmt(p[0])} [{fmt(p[2])}, {fmt(p[3])}] ({p[4]}/{n})" if n > 1 else f"{fmt(p[0])} ({p[4]}/{n})")
            pm = lambda c, nd=0: f"{fmt(ms(c)[0], nd)} ± {fmt(ms(c)[1], nd)}"
            out.append(f"| {v} | {n} | {pm(col)} | {cells[0]} | {cells[1]} | {cells[2]} | {pm('cost')} | {pm('viol_intra_hour')} | "
                       f"{pm('band_20_23_below_kh')} | {pm('starts')} | {pm('intervention_rate', 3)} | {pm('filter_share_dhw_heat', 3)} |")
            summ.setdefault(s, {})[v] = {"n": n, "objective": ms(col), "cost": ms("cost"), "starts": ms("starts"),
                                         "viol": ms("viol_intra_hour"), "kh_below20": ms("band_20_23_below_kh")}
    return "\n".join(out), summ, prs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=".scratch/v3")
    ap.add_argument("--out", default="experiments/dhw_v2_rl/results_v3")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(Path(a.runs))
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "per_seed_v3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    md, summ, prs = tables(rows)
    with open(out / "paired_v3.csv", "w", newline="") as f:
        if prs:
            w = csv.DictWriter(f, fieldnames=list(prs[0])); w.writeheader(); w.writerows(prs)
    (out / "tables_v3.md").write_text(md + "\n")
    (out / "summary_v3.json").write_text(json.dumps(summ, indent=1, default=float))
    print(md)


if __name__ == "__main__":
    main()
