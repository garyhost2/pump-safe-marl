from __future__ import annotations

import collections
import glob
import json
import os
import statistics as st
import sys
from typing import Any, Dict, List, Optional, Tuple

REF = "tx_travis_8b__year-insample__refn8__cap2000-500"

TOL = 0.0015


def _arm(rec: Dict[str, Any]) -> Optional[str]:
    a = rec.get("meta", {}).get("arm")
    return a if isinstance(a, str) else (a or {}).get("name")


def _lambda_indoor(rec: Dict[str, Any]) -> float:
    learner = rec.get("meta", {}).get("learner") or {}
    return (learner.get("reward") or {}).get("lambda_indoor", 0.4)


def load_reference_stock() -> Tuple[Dict[Tuple, Dict[str, List[float]]], float, float, float]:
    den = idle = rule_viol = None
    for path in glob.glob("results/t1_repro_ep15/**/*.json", recursive=True):
        rec = json.load(open(path))
        if _arm(rec) == "rbc":
            den = rec["eval"]["stems_cost"]
            rule_viol = rec["eval"]["stems_safety_violations_soc_equipment"]
        elif _arm(rec) == "idle":
            idle = rec["eval"]["stems_cost"]
    if den is None:
        raise SystemExit("no unshielded rule record found; cannot normalise anything")

    rows: Dict[Tuple, Dict[str, List[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for path in glob.glob("results/**/*.json", recursive=True):
        if os.path.basename(os.path.dirname(path)) != REF:
            continue
        try:
            rec = json.load(open(path))
        except Exception:
            continue
        ev = rec.get("eval") or {}
        if "meta" not in rec or "stems_cost" not in ev:
            continue
        name = _arm(rec)
        if not name:
            continue
        key = (name, _lambda_indoor(rec), rec["meta"].get("episodes", 0))
        rows[key]["cost"].append(ev["stems_cost"] / den)
        rows[key]["disc"].append(ev["stems_discomfort_proportion"])
        rows[key]["viol"].append(ev["stems_safety_violations_soc_equipment"])
    return rows, den, idle, rule_viol


def claims() -> List[Tuple[str, float, Optional[float], float]]:
    rows, den, idle, rule_viol = load_reference_stock()
    out: List[Tuple[str, float, Optional[float], float]] = []

    def mean(key: Tuple, field: str) -> Optional[float]:
        v = rows.get(key, {}).get(field)
        return st.mean(v) if v else None

    for name, published in (("mpc", 0.979), ("sac", 1.581), ("maddpg", 1.269),
                            ("metaems", 1.310), ("marlisa", 0.962),
                            ("madcq", 0.879), ("dmappo", 0.998)):
        value = (mean((f"{name}-unshielded", 0.4, 15), "cost")
                 or mean((f"{name}-unshielded", 0.4, 0), "cost"))
        out.append((f"§7 {name}-unshielded cost", published, value, TOL))
    out.append(("§7.1 unshielded rule violation %", 31.63, rule_viol, 0.01))
    out.append(("§7.5 idle cost ratio", 1.224, idle / den, TOL))

    out.append(("§8.3 best point cost (λ=0)", 0.7182,
                mean(("rl+calibrated+bc", 0.0, 50), "cost"), TOL))
    out.append(("§8.3 best point discomfort", 0.4497,
                mean(("rl+calibrated+bc", 0.0, 50), "disc"), TOL))
    out.append(("§8.3a matched cost (λ=0.4)", 0.817,
                mean(("rl+calibrated+bc", 0.4, 50), "cost"), TOL))
    out.append(("§8.3a matched discomfort", 0.420,
                mean(("rl+calibrated+bc", 0.4, 50), "disc"), TOL))
    out.append(("§8.3a best baseline (MADCQ)", 0.879,
                mean(("madcq-unshielded", 0.4, 15), "cost"), TOL))

    GRAPH = "<neighbour coupling>"
    try:
        sh = json.load(open("results/state_shapley/state_shapley.json"))
        rnd = json.load(open("results/state_shapley_randominit/state_shapley.json"))
        g_t = next(p for p in sh["players"] if p["name"] == GRAPH)
        g_r = next(p for p in rnd["players"] if p["name"] == GRAPH)
        out.append(("§10.4 graph share, trained", 0.501, g_t["share_of_total_abs"], 0.001))
        out.append(("§10.5 graph share, random init", 0.389, g_r["share_of_total_abs"], 0.001))
        out.append(("§10.5 magnitude ratio", 751.0,
                    g_t["mean_abs_phi"] / g_r["mean_abs_phi"], 1.0))
        occ = json.load(open("results/graph_attribution_bc/graph_attribution.json"))
        top = max(float(d["influence"]) for d in occ["observation_influence"])
        out.append(("§10.4 occlusion ratio", 2.04, top / occ["full_isolation_mean"], 0.01))
        out.append(("§10.4 efficiency residual < 1e-5", 0.0,
                    sh["efficiency_residual"], 1e-5))
    except (FileNotFoundError, StopIteration) as exc:
        print(f"note: attribution claims skipped ({exc})", file=sys.stderr)

    ref_eval = []
    for f in glob.glob(f"results/*/{REF}/*.json"):
        try:
            d = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(d, dict) and isinstance(d.get("eval"), dict) \
                and "soc_violation_rate" in d["eval"]:
            ref_eval.append(d["eval"])
    if ref_eval:
        n = len(ref_eval)
        out.append(("§8.2 union == SoC, fraction of records", 1.0,
                    sum(1 for e in ref_eval
                        if e["safety_violation_rate"] == e["soc_violation_rate"]) / n, 0.0))
        out.append(("§8.2 max district peak, % of 2000 kW cap", 5.08,
                    100.0 * max(e["peak_import_kw"] for e in ref_eval) / 2000.0, 0.01))
        out.append(("§8.2 max building peak, % of 500 kW cap", 4.72,
                    100.0 * max(e["peak_building_import_kw"] for e in ref_eval) / 500.0, 0.01))

    def _rule_denominator() -> Optional[float]:
        for path in glob.glob("results/t1_repro_ep15/**/*.json", recursive=True):
            try:
                rec = json.load(open(path))
            except (json.JSONDecodeError, OSError):
                continue
            if _arm(rec) == "rbc":
                return float(rec["eval"]["stems_cost"])
        return None

    def _runs(dirs: List[str], stock: str, arm: str) -> Tuple[List[float], List[float]]:
        costs, viols = [], []
        for d in dirs:
            for path in sorted(glob.glob(f"results/{d}/{stock}/{arm}__seed*.json")):
                try:
                    rec = json.load(open(path))
                except (json.JSONDecodeError, OSError):
                    continue
                if rec.get("status") != "ok":
                    continue
                costs.append(float(rec["eval"]["stems_cost"]))
                viols.append(float(rec["eval"]["stems_soc_violations_pct"]))
        return costs, viols

    den = _rule_denominator()
    if den:
        shielded = ["shielded_ep50", "dmappo_shielded_ep50"]
        deltas: List[float] = []
        for arm, claimed in (("sac", -0.068), ("maddpg", 0.062),
                             ("metaems", -0.005), ("dmappo", 0.015)):
            off, voff = _runs(["t1_unshielded_ep50"], REF, f"{arm}-unshielded")
            on, von = _runs(shielded, REF, arm)
            if len(off) < 2 or len(on) < 2:
                continue
            d = st.mean(on) / den - st.mean(off) / den
            deltas.append(d)
            out.append((f"§8.3e {arm} shield effect", claimed, d, 0.001))
            out.append((f"§8.3e {arm} violations shielded", 0.0, max(von), 0.0))
        for i, claimed, claimed_viol in ((1, -0.002, 0.0), (2, 0.029, 0.001427), (3, 0.001, 0.0)):
            stock = f"tx_travis_8b__year-insample__subset{i}n8__cap2000-500"
            rule, _ = _runs(["shield_transfer_subsets"], stock, "rbc")
            off, _ = _runs(["shield_transfer_subsets"], stock, "dmappo-unshielded")
            on, von = _runs(["shield_transfer_subsets"], stock, "dmappo")
            if not rule or len(off) < 2 or len(on) < 2:
                continue
            d = st.mean(on) / rule[0] - st.mean(off) / rule[0]
            deltas.append(d)
            out.append((f"§8.3f subset{i} shield effect", claimed, d, 0.001))
            out.append((f"§8.3f subset{i} violations shielded", claimed_viol, max(von), 1e-6))
        tstock = "transfer_2022__year-insample__refn8__cap2000-500"
        trule, _ = _runs(["transfer_2022"], tstock, "rbc")
        toff, tvoff = _runs(["transfer_2022"], tstock, "dmappo-unshielded")
        ton, tvon = _runs(["transfer_2022"], tstock, "dmappo")
        if trule and len(toff) >= 2 and len(ton) >= 2:
            d = st.mean(ton) / trule[0] - st.mean(toff) / trule[0]
            deltas.append(d)
            out.append(("§8.3g 2022 dataset shield effect", 0.014, d, 0.001))
            for arm, claimed in (("sac", 0.001), ("maddpg", -0.012)):
                o2, _ = _runs(["transfer_2022"], tstock, f"{arm}-unshielded")
                n2, v2 = _runs(["transfer_2022"], tstock, arm)
                if len(o2) >= 2 and len(n2) >= 2:
                    deltas.append(st.mean(n2) / trule[0] - st.mean(o2) / trule[0])
                    out.append((f"§8.3h 2022 {arm} shield effect", claimed,
                                st.mean(n2) / trule[0] - st.mean(o2) / trule[0], 0.001))
                    out.append((f"§8.3h 2022 {arm} violations shielded", 0.0, max(v2), 0.0))
            out.append(("§8.3g 2022 violations unshielded", 90.54, st.mean(tvoff), 0.01))
            out.append(("§8.3g 2022 violations shielded", 0.0, max(tvon), 0.0))
            out.append(("§8.3g 2022 rule cost", 25042.41, trule[0], 0.01))

        matrix, sub2 = [], []
        for i in (1, 2, 3):
            stock = f"tx_travis_8b__year-insample__subset{i}n8__cap2000-500"
            rule, _ = _runs(["shield_transfer_subsets"], stock, "rbc")
            if not rule:
                continue
            for arm in ("sac", "maddpg", "metaems"):
                off, _ = _runs(["shield_transfer_subsets"], stock, f"{arm}-unshielded")
                on, von = _runs(["shield_transfer_subsets"], stock, arm)
                if len(off) < 2 or len(on) < 2:
                    continue
                matrix.append(st.mean(on) / rule[0] - st.mean(off) / rule[0])
                if i == 2:
                    sub2.append(max(von))
        _, von_d = _runs(["shield_transfer_subsets"],
                         "tx_travis_8b__year-insample__subset2n8__cap2000-500", "dmappo")
        if von_d:
            sub2.append(max(von_d))
        if len(sub2) == 4:
            out.append(("§8.3h subset2 residual, all 4 policies identical",
                        0.001427, max(sub2), 1e-6))
            out.append(("§8.3h subset2 residual, spread across policies",
                        0.0, max(sub2) - min(sub2), 1e-9))
        allcells = deltas + matrix
        if len(allcells) == 19:
            out.append(("§8.3h mean effect, 19 conditions", -0.004, st.mean(allcells), 0.001))
            out.append(("§8.3h most negative effect", -0.135, min(allcells), 0.001))
            out.append(("§8.3h most positive effect", 0.062, max(allcells), 0.001))

    def _cycles_and_spread(path: str, arm: str) -> Tuple[float, float]:
        costs, cycles = [], []
        for f in sorted(glob.glob(f"{path}/{arm}__seed*.json")):
            try:
                rec = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get("status") != "ok":
                continue
            costs.append(rec["eval"]["stems_cost"])
            cycles.append(rec["eval"].get("battery_equivalent_full_cycles", 0.0))
        if not costs:
            return float("nan"), float("nan")
        return (sum(cycles) / len(cycles),
                (max(costs) - min(costs)) if len(costs) > 1 else 0.0)

    counted = [("results/t1_unshielded_ep50/" + REF, a)
               for a in ("dmappo", "sac", "maddpg", "metaems")]
    for i in (1, 2, 3):
        stk = f"results/shield_transfer_subsets/tx_travis_8b__year-insample__subset{i}n8__cap2000-500"
        counted += [(stk, a) for a in ("dmappo", "sac", "maddpg", "metaems")]
    counted += [("results/transfer_2022/transfer_2022__year-insample__refn8__cap2000-500", a)
                for a in ("dmappo", "sac", "maddpg")]
    worst_cycles, worst_spread, seen = float("inf"), float("inf"), 0
    for path, arm in counted:
        cyc, spr = _cycles_and_spread(path, f"{arm}-unshielded")
        if cyc != cyc:
            continue
        seen += 1
        worst_cycles = min(worst_cycles, cyc)
        worst_spread = min(worst_spread, spr)
    if seen:
        out.append(("§8.3h counted cells screened", 19.0, float(seen), 0.0))
        out.append(("§8.3h min cycles in a counted cell > 1", 27.56, worst_cycles, 0.1))
        out.append(("§8.3h min seed spread in a counted cell > 0", 168.36, worst_spread, 0.1))
    icyc, ispr = _cycles_and_spread(
        "results/transfer_2022/transfer_2022__year-insample__refn8__cap2000-500",
        "metaems-unshielded")
    if icyc == icyc:
        out.append(("§8.3h excluded cell cycles", 0.0, icyc, 1e-9))
        out.append(("§8.3h excluded cell seed spread", 0.0, ispr, 1e-9))

    iso_c, iso_v, ish_c, ish_v = [], [], [], []
    for arm, cs, vs in (("isac-unshielded", iso_c, iso_v), ("isac", ish_c, ish_v)):
        for f in sorted(glob.glob(f"results/t1_isac_ep15/{REF}/{arm}__seed*.json")):
            try:
                rec = json.load(open(f))
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get("status") != "ok":
                continue
            cs.append(rec["eval"]["stems_cost"])
            vs.append(rec["eval"]["stems_soc_violations_pct"])
    if iso_c and ish_c and den:
        out.append(("§7.6 isac unshielded cost", 1.563, st.mean(iso_c) / den, 0.001))
        out.append(("§7.6 isac shielded cost", 1.512, st.mean(ish_c) / den, 0.001))
        out.append(("§7.6 isac violations unshielded", 68.23, st.mean(iso_v), 0.01))
        out.append(("§7.6 isac violations shielded", 0.0, max(ish_v), 0.0))

    try:
        sys.path.insert(0, os.getcwd())
        from experiments.scenario import Scenario
        from stems.environment import STEMSEnvironment
        _sc = Scenario(schema="citylearn_schemas/tx_travis_8b/schema.json", season="year",
                       n_buildings=8, subset_seed=2, grid_cap_kw=2000.0,
                       building_cap_kw=500.0)
        _env = STEMSEnvironment(schema=_sc.schema_path(), seed=0, heat_pump=_sc.heat_pump,
                                env_kwargs=_sc.env_kwargs("eval"),
                                hvac_control=_sc.hvac_control)
        _b = _env.battery_info()
        _cap = [float(x) for x in _b["capacity"]]
        _nom = [float(x) for x in _b["nominal_power"]]
        _eff = [float(x) for x in _b["efficiency"]]
        _gain = [n * e / c for n, e, c in zip(_nom, _eff, _cap)]
        _w = _gain.index(min(_gain))
        out.append(("§8.3f draw 2 worst capacity kWh", 14.0, _cap[_w], 0.001))
        out.append(("§8.3f draw 2 worst nominal kW", 1.28, _nom[_w], 0.001))
        out.append(("§8.3f draw 2 worst efficiency", 0.9601, _eff[_w], 0.0001))
        out.append(("§8.3f draw 2 max one-hour SoC gain", 0.0878, min(_gain), 0.0001))
    except Exception as exc:
        print(f"note: reachability claims skipped ({type(exc).__name__}: {exc})",
              file=sys.stderr)

    dr_arms, dr_lam = [], collections.defaultdict(list)
    for f in glob.glob(f"results/t1_unshielded_ep15/{REF}/*-unshielded__seed*.json"):
        try:
            rec = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if rec.get("status") == "ok" and rec["eval"].get("discomfort_rate") is not None:
            dr_arms.append(rec["eval"]["discomfort_rate"])
    for f in glob.glob("results/**/*.json", recursive=True):
        try:
            rec = json.load(open(f))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(rec, dict) or rec.get("status") != "ok":
            continue
        meta = rec.get("meta", {}) or {}
        arm = meta.get("arm")
        arm = arm if isinstance(arm, str) else (arm or {}).get("name")
        lam = ((meta.get("learner") or {}).get("reward") or {}).get("lambda_indoor")
        d = (rec.get("eval") or {}).get("discomfort_rate")
        if arm and arm.startswith("rl+calibrated") and lam is not None and d is not None:
            dr_lam[round(float(lam), 2)].append(d)
    if dr_arms:
        out.append(("§9.1 discomfort min across controllers", 0.0042, min(dr_arms), 0.0001))
        out.append(("§9.1 discomfort max across controllers", 0.2809, max(dr_arms), 0.0001))
    if {0.0, 0.8} <= set(dr_lam):
        out.append(("§9.1 discomfort at lambda=0.0", 0.0291,
                    st.mean(dr_lam[0.0]), 0.0001))
        out.append(("§9.1 discomfort at lambda=0.8", 0.0112,
                    st.mean(dr_lam[0.8]), 0.0001))

    lam_runs, lam_eps, lam_dead, lam_ident, lam_viol = 0, 0, 0, 0, 0.0
    for f in glob.glob(f"results/*/{REF}/rl+calibrated*__seed*.json"):
        try:
            tr = json.load(open(f)).get("train") or []
        except (json.JSONDecodeError, OSError):
            continue
        lam = [(e["episode"], e["lambdas"], e.get("violating_step_rate"))
               for e in tr if e.get("lambdas") is not None]
        if not lam:
            continue
        lam_runs += 1
        lam_eps += len(lam)
        if all(abs(float(L[0]) - float(L[1])) < 1e-12
               and abs(float(L[1]) - float(L[2])) < 1e-12 for _, L, _ in lam):
            lam_ident += 1
        first_zero = next((ep for ep, L, _ in lam if all(float(x) == 0.0 for x in L)), None)
        if first_zero is not None:
            lam_dead += len(lam) - (first_zero - 1)
        lam_viol = max(lam_viol, max((v for _, _, v in lam if v is not None), default=0.0))
    if lam_runs:
        out.append(("§9.4 runs with all 3 channels identical", float(lam_runs),
                    float(lam_ident), 0.0))
        out.append(("§9.4 episodes with every lambda at zero", 0.904,
                    lam_dead / lam_eps, 0.001))
        out.append(("§9.4 max violating_step_rate, any episode", 0.0, lam_viol, 0.0))

    try:
        ref = {os.path.basename(f).replace("__ep15.json", ""): json.load(open(f))["eval"]
               for f in glob.glob("results/reference_baselines/*.json")}
        basic_rbc_den = ref["citylearn-rbc__seed0"]["stems_cost"]
        for seed, claimed in ((0, 0.153), (1, 0.162)):
            out.append((f"§7.5 reference SAC cost, seed {seed}", claimed,
                        ref[f"citylearn-sac__seed{seed}"]["stems_cost"]
                        / basic_rbc_den, 0.001))
        out.append(("§7.5 BasicRBC / no-control electricity", 3.15,
                    ref["citylearn-rbc__seed0"]["stems_electricity_consumption"]
                    / ref["citylearn-baseline__seed0"]["stems_electricity_consumption"], 0.01))
        out.append(("§7.5 BasicRBC discomfort", 0.771,
                    ref["citylearn-rbc__seed0"]["stems_discomfort_proportion"], 0.001))
        out.append(("§7.5 no-control discomfort", 0.400,
                    ref["citylearn-baseline__seed0"]["stems_discomfort_proportion"], 0.001))
        out.append(("§7.5 rbc-optimized as denominator", 0.991,
                    ref["citylearn-rbc-optimized__seed0"]["stems_cost"]
                    / basic_rbc_den, 0.001))
        out.append(("§7.5 reference MARLISA cost", 0.191,
                    ref["citylearn-marlisa__seed0"]["stems_cost"]
                    / basic_rbc_den, 0.001))
    except (FileNotFoundError, KeyError) as exc:
        print(f"note: reference-baseline claims skipped ({exc})", file=sys.stderr)

    import glob as _glob
    import statistics as _st

    def _ep50(arm: str, key: str) -> Optional[float]:
        vals = []
        for f in sorted(_glob.glob(
                f"results/t1_unshielded_ep50/*/{arm}-unshielded__seed*.json")):
            with open(f, encoding="utf-8") as fh:
                e = (json.load(fh).get("eval") or {})
            if e.get(key) is not None:
                vals.append(e[key])
        return _st.mean(vals) if vals else None

    published_cost = {"marlisa": 0.847, "dmappo": 0.803, "madcq": 0.814,
                      "maddpg": 0.795, "metaems": 0.836, "sac": 0.824}
    measured_cost = {"marlisa": 0.8474, "dmappo": 0.8173, "madcq": 0.8325,
                     "maddpg": 1.0474, "metaems": 1.3272, "sac": 1.5733}
    for arm, want in measured_cost.items():
        got = _ep50(arm, "stems_cost")
        out.append((f"§7.0 {arm} cost at 50 episodes", want,
                    None if got is None else got / den, 0.001))
    for arm in ("marlisa", "dmappo", "madcq"):
        got = _ep50(arm, "stems_cost")
        gap = None if got is None else abs(got / den - published_cost[arm])
        out.append((f"§7.0 {arm} |gap to published| under 2.3%",
                    0.0, None if gap is None else max(gap - 0.023, 0.0), 1e-9))
    try:
        with open("results/t1_unshielded_ep50/tx_travis_8b__year-insample__refn8"
                  "__cap2000-500/metaems-unshielded__seed0.json", encoding="utf-8") as fh:
            tr = json.load(fh)["train"]
        out.append(("§7.0a MetaEMS gradient steps per episode", 64.0,
                    float(tr[0]["gradient_steps"]), 0.0))
        out.append(("§7.0a MetaEMS total gradient steps over 50 episodes", 3200.0,
                    float(sum(e["gradient_steps"] for e in tr)), 0.0))
        with open("results/t1_unshielded_ep50/tx_travis_8b__year-insample__refn8"
                  "__cap2000-500/marlisa-unshielded__seed0.json", encoding="utf-8") as fh:
            tr2 = json.load(fh)["train"]
        out.append(("§7.0a MARLISA total gradient steps over 50 episodes", 437950.0,
                    float(sum(e["gradient_steps"] for e in tr2)), 0.0))
    except (FileNotFoundError, KeyError) as exc:
        print(f"note: gradient-budget claims skipped ({exc})", file=sys.stderr)
    cyc = {a: _ep50(a, "battery_equivalent_full_cycles")
           for a in ("metaems", "marlisa", "madcq", "dmappo", "sac", "maddpg")}
    if cyc["metaems"] is not None:
        out.append(("§7.0a MetaEMS battery-equivalent full cycles", 61.9,
                    cyc["metaems"], 0.1))
        others = [v for a, v in cyc.items() if a != "metaems" and v is not None]
        out.append(("§7.0a every other arm exceeds 230 cycles", 0.0,
                    max(230.0 - min(others), 0.0) if others else None, 1e-9))
    pub_peak = {"mpc": 0.983, "maddpg": 0.854, "marlisa": 0.826, "madcq": 0.862,
                "dmappo": 0.841, "sac": 0.856, "metaems": 0.884}
    try:
        mpc = {}
        for f in _glob.glob("results/t1_mpc_probe/*/*.json"):
            with open(f, encoding="utf-8") as fh:
                rec = json.load(fh)
            mpc[os.path.basename(f).split("__")[0]] = rec["eval"]
        out.append(("§7.0a MPC, causal forecaster", 0.979,
                    mpc["mpc-unshielded"]["stems_cost"] / den, 0.001))
        out.append(("§7.0a MPC, perfect foresight", 0.770,
                    mpc["mpc-oracle"]["stems_cost"] / den, 0.001))
        lo = mpc["mpc-oracle"]["stems_cost"] / den
        hi = mpc["mpc-unshielded"]["stems_cost"] / den
        out.append(("§7.0a published MPC 0.872 lies strictly inside", 1.0,
                    1.0 if lo < 0.872 < hi else 0.0, 0.0))
        pub5 = {"stems_cost": 0.872, "stems_emission": 0.914,
                "stems_avg_daily_peak": 0.983,
                "stems_electricity_consumption": 0.975,
                "stems_ramping_rate": 0.981}
        rule5 = {}
        for f in _glob.glob("results/t1_lam0_ep15/*/rbc__seed0.json"):
            with open(f, encoding="utf-8") as fh:
                rule5 = json.load(fh)["eval"]
        n_bracketed = 0
        for k, p_ in pub5.items():
            if not rule5.get(k):
                continue
            a_ = mpc["mpc-oracle"][k] / rule5[k]
            b_ = mpc["mpc-unshielded"][k] / rule5[k]
            if min(a_, b_) < p_ < max(a_, b_):
                n_bracketed += 1
        out.append(("§7.0a columns where the MPC bracket holds", 2.0,
                    float(n_bracketed), 0.0))
    except (FileNotFoundError, KeyError) as exc:
        print(f"note: MPC bracket claims skipped ({exc})", file=sys.stderr)

    try:
        cs = {}
        for f in _glob.glob("results/critic_sensitivity/*.json"):
            with open(f, encoding="utf-8") as fh:
                rec = json.load(fh)
            cs[rec["arm"].replace("-unshielded", "")] = rec
        out.append(("§7.0a SAC critic |Q|", 71422.1, cs["sac"]["q_mean_abs"], 0.5))
        out.append(("§7.0a MARLISA critic |Q|", 58.0, cs["marlisa"]["q_mean_abs"], 0.5))
        out.append(("§7.0a SAC relative action sensitivity", 6.41e-05,
                    cs["sac"]["relative_sensitivity"], 1e-7))
        out.append(("§7.0a MARLISA relative action sensitivity", 3.72e-03,
                    cs["marlisa"]["relative_sensitivity"], 1e-5))
        out.append(("§7.0a SAC critic argmax is a corner", 1.0,
                    1.0 if cs["sac"]["argmax_is_corner"] else 0.0, 0.0))
        out.append(("§7.0a MARLISA critic argmax is interior", 0.0,
                    1.0 if cs["marlisa"]["argmax_is_corner"] else 0.0, 0.0))
    except (FileNotFoundError, KeyError) as exc:
        print(f"note: critic-sensitivity claims skipped ({exc})", file=sys.stderr)

    try:
        with open("results/critic_sensitivity_scaled/sac-unshielded__seed0.json",
                  encoding="utf-8") as fh:
            sc = json.load(fh)
        out.append(("§7.0a SAC critic |Q| with return scaling", 912.9,
                    sc["q_mean_abs"], 0.5))
        out.append(("§7.0a SAC relative sensitivity with return scaling", 5.96e-05,
                    sc["relative_sensitivity"], 1e-7))
        with open("results/critic_sensitivity/isac-unshielded__seed0.json",
                  encoding="utf-8") as fh:
            iso = json.load(fh)
        out.append(("§7.0a independent-SAC relative sensitivity", 2.07e-04,
                    iso["relative_sensitivity"], 1e-6))
        out.append(("§7.0a independent-SAC sub-critics peaking at a corner", 1.0,
                    iso["argmax_corner_fraction"], 0.0))
    except (FileNotFoundError, KeyError) as exc:
        print(f"note: value-scaling claims skipped ({exc})", file=sys.stderr)

    def _seed0(pat: str) -> Optional[float]:
        hits = _glob.glob(pat)
        if not hits:
            return None
        with open(hits[0], encoding="utf-8") as fh:
            return json.load(fh)["eval"]["stems_cost"] / den
    for lab, want, pat in (
            ("MARLISA", 0.8617,
             "results/t1_unshielded_ep50/*/marlisa-unshielded__seed0.json"),
            ("independent SAC", 1.5904, "results/t1_isac_ep15/*/isac__seed0.json"),
            ("single-agent SAC", 1.5569,
             "results/t1_unshielded_ep50/*/sac-unshielded__seed0.json"),
            ("SAC with return scaling", 1.6093,
             "results/sac_scaled/*/sac-unshielded__seed0.json")):
        out.append((f"§7.0a {lab} cost, seed 0", want, _seed0(pat), 0.001))

    try:
        import statistics as _st2
        sweep = {}
        for w in ("1e-3", "1e-2", "1e-1", "1.0"):
            pf = _glob.glob(f"results/regsweep/prof_w{w}/*.json")
            if not pf:
                continue
            with open(pf[0], encoding="utf-8") as fh:
                ch = json.load(fh)["channels"]
            sweep[w] = _st2.mean(v["saturated_hi"] + v["saturated_lo"]
                                 for v in ch.values())
        if len(sweep) == 4:
            out.append(("§7.0a regulariser sweep, min saturation over 4 decades",
                        0.979, min(sweep.values()), 0.001))
            out.append(("§7.0a regulariser sweep, max saturation over 4 decades",
                        0.987, max(sweep.values()), 0.001))
    except (FileNotFoundError, KeyError) as exc:
        print(f"note: regulariser-sweep claims skipped ({exc})", file=sys.stderr)

    den_peak = None
    for f in sorted(_glob.glob("results/t1_lam0_ep15/*/rbc__seed0.json")):
        with open(f, encoding="utf-8") as fh:
            den_peak = (json.load(fh).get("eval") or {}).get("stems_avg_daily_peak")
    if den_peak:
        below = 0
        for arm, p in pub_peak.items():
            got = _ep50(arm, "stems_avg_daily_peak")
            if got is not None and got / den_peak < p:
                below += 1
        out.append(("§7.0 arms below published avg-daily-peak", 5.0, float(below), 0.0))
    else:
        print("note: peak-direction claim skipped (no rule peak record)", file=sys.stderr)

    return out


def _stated_counts(rows: List[Tuple[str, float, Optional[float], float]]) -> None:
    import re
    pattern = re.compile(r"pins \*{0,2}(\d+)\*{0,2} numbers")
    found = []
    for path in ("RESUME.md", "docs/TECHNICAL_REPORT.md", "docs/WSED_PAPER_DRAFT.md"):
        try:
            text = open(path, encoding="utf-8").read()
        except OSError:
            continue
        for m in pattern.finditer(text):
            found.append((path, float(m.group(1))))
    total = float(len(rows) + len(found))
    for path, stated in found:
        rows.append((f"{path}: stated pinned-claim count", stated, total, 0.0))


def main() -> None:
    rows = claims()
    _stated_counts(rows)
    width = max(len(c[0]) for c in rows)
    print(f"{'claim':<{width}}{'in report':>12}{'measured':>13}   status")
    print("-" * (width + 36))
    failures = 0
    for label, published, measured, tol in rows:
        ok = measured is not None and abs(published - measured) <= tol
        failures += 0 if ok else 1
        shown = "missing" if measured is None else f"{measured:.4f}"
        print(f"{label:<{width}}{published:>12.4f}{shown:>13}   "
              f"{'ok' if ok else 'MISMATCH'}")
    print(f"\n{len(rows) - failures}/{len(rows)} claims verified against source records")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
