from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from stems.stems_table import stems_table_metrics

PUBLISHED_COST = {
    "MPC": 0.872,
    "Single-Agent SAC": 0.824,
    "MADDPG": 0.795,
    "MetaEMS": 0.836,
    "MARLISA": 0.847,
    "MADCQ": 0.814,
    "D-MAPPO": 0.803,
    "STEMS": 0.792,
}
PUBLISHED_CONSUMPTION = {
    "MPC": 0.975,
    "Single-Agent SAC": 0.813,
    "MADDPG": 0.856,
    "MetaEMS": 0.835,
    "MARLISA": 0.838,
    "MADCQ": 0.823,
    "D-MAPPO": 0.816,
    "STEMS": 0.805,
}

ARM_TO_PUBLISHED = {
    "mpc": "MPC",
    "sac": "Single-Agent SAC",
    "maddpg": "MADDPG",
    "metaems": "MetaEMS",
    "marlisa": "MARLISA",
    "madcq": "MADCQ",
    "dmappo": "D-MAPPO",
    "rl+calibrated": "STEMS",
}
BASELINE_ARM = "rbc"
EXTRA_ARMS = ("rbc", "idle", "rbc+calibrated")

ROW_ORDER = ["MPC", "Single-Agent SAC", "MADDPG", "MetaEMS", "MARLISA", "MADCQ",
             "D-MAPPO", "STEMS"]


class PriceOnlyBatteryAgent:

    def __init__(self, num_buildings: int, price_idx: int, low: float, high: float,
                 rate: float = 1.0) -> None:
        self.B = int(num_buildings)
        self.price_idx = int(price_idx)
        self.low = float(low)
        self.high = float(high)
        self.rate = float(rate)

    def reset(self) -> None:
        pass

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        actions = np.zeros((self.B, 3), dtype=np.float32)
        for i, obs in enumerate(obs_list):
            p = float(obs[self.price_idx])
            if p <= self.low:
                actions[i, 1] = self.rate
            elif p >= self.high:
                actions[i, 1] = -self.rate
        return actions

    def update(self, batch):
        return {}

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        pass


def month_index(schema_path: str, n_steps: int) -> np.ndarray:
    import pandas as pd

    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    root = Path(schema["root_directory"])
    name = next(v["energy_simulation"] for v in schema["buildings"].values()
                if v.get("energy_simulation"))
    months = pd.read_csv(root / name, usecols=["month"])["month"].to_numpy()
    start = int(schema.get("simulation_start_time_step") or 0)
    out = months[start:start + n_steps]
    if out.shape[0] != n_steps:
        raise RuntimeError(f"month series has {out.shape[0]} rows for {n_steps} steps")
    return out.astype(np.int16)


def month_bounds(months: np.ndarray) -> List[Tuple[int, int, int]]:
    out = []
    for m in range(1, 13):
        idx = np.flatnonzero(months == m)
        if idx.size == 0:
            raise RuntimeError(f"month {m} is absent from the evaluation window")
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        if hi - lo != idx.size:
            raise RuntimeError(f"month {m} is not contiguous in the evaluation window")
        out.append((m, lo, hi))
    return out


def _replay(item: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    os.chdir(REPO)

    from experiments.controllers import ARMS, build_controller
    from experiments.runner import _window_len, evaluate, make_config
    from experiments.scenario import Scenario
    from stems.environment import STEMSEnvironment
    from stems.observations import obs_index
    from stems.utils import set_seed

    arm_name, seed = item["arm"], int(item["seed"])
    record_path = Path(item["record"])
    out_dir = Path(item["out"])
    stored = json.loads(record_path.read_text(encoding="utf-8"))
    meta = stored["meta"]
    scenario = Scenario(**{k: meta["scenario"][k] for k in (
        "schema", "season", "subset_seed", "n_buildings", "days", "grid_cap_kw",
        "building_cap_kw", "hvac_control")})
    set_seed(seed)
    schema = scenario.schema_path()

    def env_for(phase: str):
        return STEMSEnvironment(schema=schema, seed=seed, heat_pump=True,
                                env_kwargs=scenario.env_kwargs(phase),
                                hvac_control=scenario.hvac_control)

    eval_env = env_for("eval")
    config = make_config(scenario, meta.get("learner") or {})

    if item.get("synthetic") == "price-only-battery":
        price = np.asarray(
            eval_env._env.buildings[0].pricing.electricity_pricing, dtype=np.float64)
        lo, hi = float(np.quantile(price, 0.25)), float(np.quantile(price, 0.75))
        controller = PriceOnlyBatteryAgent(eval_env.num_buildings,
                                          obs_index("electricity_pricing"), lo, hi)
    else:
        arm = ARMS[arm_name]
        controller = build_controller(arm, eval_env, config)
        if arm.learns:
            controller.load(str(record_path.with_name(f"{record_path.stem}_model")))
        if getattr(controller, "fleet_shield", None) is not None:
            evaluate(controller, env_for("train"), config,
                     _window_len(scenario.env_kwargs("train")))
            for inner in (getattr(controller, "base", None),
                          getattr(controller, "base_policy", None)):
                if inner is not None and hasattr(inner, "reset"):
                    inner.reset()

    res = evaluate(controller, eval_env, config,
                   _window_len(scenario.env_kwargs("eval")), return_accumulator=True)
    acc = res["accumulator"]
    kpis = res["kpis"]

    net = np.stack(acc._net_list, axis=0)
    price = np.stack(acc._price_list, axis=0)
    carbon = np.stack(acc._carbon_list, axis=0)
    months = month_index(schema, net.shape[0])

    monthly = []
    for m, lo, hi in month_bounds(months):
        sub = acc.slice_steps(lo, hi).compute_all()
        monthly.append({"month": m, "start": lo, "stop": hi,
                        "stems_cost": sub["stems_cost"],
                        "stems_electricity_consumption":
                            sub["stems_electricity_consumption"]})

    synthetic = bool(item.get("synthetic"))
    tag = f"{arm_name}__seed{seed}"
    np.savez_compressed(out_dir / f"{tag}.npz", net=net.astype(np.float32),
                        price=price.astype(np.float32),
                        carbon=carbon.astype(np.float32), months=months)
    payload = {
        "arm": arm_name, "seed": seed, "steps": int(net.shape[0]),
        "stored_cost": None if synthetic else stored["eval"].get("stems_cost"),
        "replayed_cost": kpis["stems_cost"],
        "stored_consumption": (None if synthetic
                               else stored["eval"].get("stems_electricity_consumption")),
        "replayed_consumption": kpis["stems_electricity_consumption"],
        "barrier_intervention_rate": kpis.get("barrier_intervention_rate"),
        "monthly": monthly,
        "code_fingerprint_stored": meta["code"]["fingerprint"],
    }
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return {"tag": tag, "status": "ok",
            "drift": (None if payload["stored_cost"] is None
                      else payload["replayed_cost"] - payload["stored_cost"])}


def _replay_guarded(item: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _replay(item)
    except Exception:
        return {"tag": f"{item['arm']}__seed{item['seed']}", "status": "error",
                "traceback": traceback.format_exc()}


def stage_dump(results: Path, out: Path, workers: int, arms: Optional[List[str]],
               seeds: Optional[List[int]]) -> None:
    from multiprocessing import get_context

    out.mkdir(parents=True, exist_ok=True)
    scenario_dirs = sorted(p for p in results.iterdir() if p.is_dir())
    if len(scenario_dirs) != 1:
        raise SystemExit(f"{results} holds {len(scenario_dirs)} scenario directories; "
                         "point --results at a single-scenario run")
    root = scenario_dirs[0]

    wanted = arms or (list(ARM_TO_PUBLISHED) + list(EXTRA_ARMS))
    items: List[Dict[str, Any]] = []
    for arm_name in wanted:
        records = sorted(root.glob(f"{arm_name}__seed*.json"))
        for rec in records:
            seed = int(rec.stem.split("seed")[-1])
            if seeds is not None and seed not in seeds:
                continue
            items.append({"arm": arm_name, "seed": seed, "record": str(rec),
                          "out": str(out)})
    base_rec = sorted(root.glob(f"{BASELINE_ARM}__seed*.json"))[0]
    if arms is None or "price-only-battery" in (arms or []):
        items.append({"arm": "price-only-battery", "seed": 0, "record": str(base_rec),
                      "out": str(out), "synthetic": "price-only-battery"})

    print(f"replaying {len(items)} runs with {workers} workers", flush=True)
    if workers <= 1:
        done = [_replay_guarded(it) for it in items]
    else:
        with get_context("spawn").Pool(workers) as pool:
            done = list(pool.imap_unordered(_replay_guarded, items))
    for d in sorted(done, key=lambda r: r["tag"]):
        if d["status"] != "ok":
            print(f"  {d['tag']:28s} ERROR\n{d['traceback']}", flush=True)
        else:
            drift = d["drift"]
            print(f"  {d['tag']:28s} ok   cost drift vs stored: "
                  f"{'n/a' if drift is None else format(drift, '+.6f')}", flush=True)


def shield_precheck(results: Path, unshielded: Path) -> Dict[str, Any]:
    import statistics as st

    def collect(root: Path, name: str) -> Optional[Dict[str, float]]:
        costs, cons = [], []
        for path in sorted(root.glob(f"{name}__seed*.json")):
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("status") != "ok":
                continue
            costs.append(rec["eval"]["stems_cost"])
            cons.append(rec["eval"]["stems_electricity_consumption"])
        if not costs:
            return None
        return {"n_seeds": len(costs), "cost_usd": st.mean(costs),
                "consumption_kwh": st.mean(cons)}

    shielded_root = next(p for p in sorted(results.iterdir()) if p.is_dir())
    unshielded_root = next(p for p in sorted(unshielded.iterdir()) if p.is_dir())
    base = collect(shielded_root, BASELINE_ARM)
    rows = {}
    for arm, label in ARM_TO_PUBLISHED.items():
        s = collect(shielded_root, arm)
        u = collect(unshielded_root, f"{arm}-unshielded")
        if s is None or u is None:
            continue
        rows[label] = {
            "shielded_cost_usd": s["cost_usd"], "unshielded_cost_usd": u["cost_usd"],
            "shielded_ratio": s["cost_usd"] / base["cost_usd"],
            "unshielded_ratio": u["cost_usd"] / base["cost_usd"],
            "shield_effect_on_ratio": (s["cost_usd"] - u["cost_usd"]) / base["cost_usd"],
            "published": PUBLISHED_COST[label],
            "n_seeds": [s["n_seeds"], u["n_seeds"]],
        }
    deltas = [abs(v["shield_effect_on_ratio"]) for v in rows.values()]
    return {"baseline_cost_usd": base["cost_usd"], "rows": rows,
            "max_abs_shield_effect_on_ratio": max(deltas) if deltas else None,
            "rows_still_above_published_unshielded":
                sum(1 for v in rows.values() if v["unshielded_ratio"] > v["published"]),
            "n_rows_compared": len(rows)}


EP50_DIRS = ("results/t1_unshielded_ep50", "results/t1_repro_ep50")
EP50_ARM_FOR = {
    "MPC": ("mpc-unshielded",),
    "Single-Agent SAC": ("sac-unshielded",),
    "MADDPG": ("maddpg-unshielded",),
    "MetaEMS": ("metaems-unshielded",),
    "MARLISA": ("marlisa-unshielded",),
    "MADCQ": ("madcq-unshielded",),
    "D-MAPPO": ("dmappo-unshielded",),
    "STEMS": ("rl+calibrated",),
}


def budget_premise_check(ep50_dirs, baseline_usd: float) -> Dict[str, Any]:
    import statistics as st

    out = {"baseline_usd": baseline_usd, "rows": {}}
    for label, names in EP50_ARM_FOR.items():
        best = None
        for d in ep50_dirs:
            root_dirs = [p for p in sorted(Path(d).glob("*")) if p.is_dir()]
            for root in root_dirs:
                for name in names:
                    costs, cons, eps = [], [], set()
                    for path in sorted(root.glob(f"{name}__seed*.json")):
                        rec = json.loads(path.read_text(encoding="utf-8"))
                        if rec.get("status") != "ok":
                            continue
                        costs.append(rec["eval"]["stems_cost"])
                        cons.append(rec["eval"]["stems_electricity_consumption"])
                        eps.add(rec["meta"]["episodes"])
                    if costs and (best is None or len(costs) > best["n_seeds"]):
                        best = {"arm": name, "n_seeds": len(costs),
                                "episodes": sorted(eps),
                                "cost_usd": st.mean(costs),
                                "consumption_kwh": st.mean(cons),
                                "cost_sd": (st.stdev(costs) if len(costs) > 1 else 0.0)}
        if best is None:
            continue
        best["ratio"] = best["cost_usd"] / baseline_usd
        best["gap"] = best["ratio"] - PUBLISHED_COST[label]
        best["published"] = PUBLISHED_COST[label]
        best["implied_baseline_usd"] = best["cost_usd"] / PUBLISHED_COST[label]
        out["rows"][label] = best
    gaps = {l: v["gap"] for l, v in out["rows"].items()}
    out["rows_within_0_025"] = [l for l, g in gaps.items() if abs(g) <= 0.025]
    out["rows_within_0_05"] = [l for l, g in gaps.items() if abs(g) <= 0.05]
    out["rows_still_above_0_05"] = [l for l, g in gaps.items() if g > 0.05]
    return out


def _load(out: Path) -> Dict[str, Dict[str, Any]]:
    per_arm: Dict[str, Dict[str, Any]] = {}
    for path in sorted(out.glob("*.json")):
        if path.name.startswith("_"):
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        per_arm.setdefault(rec["arm"], {"seeds": []})["seeds"].append(rec)
    for arm, blob in per_arm.items():
        blob["seeds"].sort(key=lambda r: r["seed"])
        nets, prices, carbons = [], [], []
        for rec in blob["seeds"]:
            z = np.load(out / f"{arm}__seed{rec['seed']}.npz")
            nets.append(z["net"].astype(np.float64))
            prices.append(z["price"].astype(np.float64))
            carbons.append(z["carbon"].astype(np.float64))
            blob["months"] = z["months"]
        blob["net"] = nets
        blob["price"] = prices
        blob["carbon"] = carbons
    return per_arm


def _cost(net: np.ndarray, price: np.ndarray, *, credit_exports: bool = False,
          district_netting: bool = False) -> float:
    if district_netting:
        signed = net.sum(axis=1)
        billed = signed if credit_exports else np.maximum(signed, 0.0)
        return float((billed * price[:, 0]).sum())
    billed = net if credit_exports else np.maximum(net, 0.0)
    return float((billed * price).sum())


def _mean(xs: List[float]) -> float:
    return float(np.mean(xs))


def hypothesis_1(per_arm, rows) -> Dict[str, Any]:
    base_monthly = np.array([[m["stems_cost"] for m in rec["monthly"]]
                             for rec in per_arm[BASELINE_ARM]["seeds"]]).mean(axis=0)
    base_annual = _mean([r["replayed_cost"] for r in per_arm[BASELINE_ARM]["seeds"]])
    base_cons_monthly = np.array(
        [[m["stems_electricity_consumption"] for m in rec["monthly"]]
         for rec in per_arm[BASELINE_ARM]["seeds"]]).mean(axis=0)
    out = {"baseline_monthly_cost": base_monthly.tolist(),
           "baseline_annual_cost": base_annual,
           "baseline_monthly_consumption_kwh": base_cons_monthly.tolist(),
           "rows": {}}
    for arm, label in rows:
        monthly = np.array([[m["stems_cost"] for m in rec["monthly"]]
                            for rec in per_arm[arm]["seeds"]]).mean(axis=0)
        annual = _mean([r["replayed_cost"] for r in per_arm[arm]["seeds"]])
        ratios = monthly / base_monthly
        out["rows"][label] = {
            "arm": arm,
            "annual_ratio": annual / base_annual,
            "monthly_mean_ratio": float(ratios.mean()),
            "monthly_ratios": ratios.tolist(),
            "monthly_cost_usd": monthly.tolist(),
            "annual_cost_usd": annual,
            "monthly_ratio_min": float(ratios.min()),
            "monthly_ratio_max": float(ratios.max()),
            "published": PUBLISHED_COST[label],
        }
    return out


def hypothesis_2(per_arm, rows, cost_fn=None) -> Dict[str, Any]:
    implied = {}
    for arm, label in rows:
        abs_cost = (_mean([r["replayed_cost"] for r in per_arm[arm]["seeds"]])
                    if cost_fn is None else cost_fn(arm))
        implied[label] = {"arm": arm, "absolute_cost_usd": abs_cost,
                          "published_ratio": PUBLISHED_COST[label],
                          "implied_baseline_usd": abs_cost / PUBLISHED_COST[label]}
    vals = np.array([v["implied_baseline_usd"] for v in implied.values()])
    candidates = {}
    for arm in per_arm:
        if arm in dict(rows):
            continue
        candidates[arm] = (_mean([r["replayed_cost"] for r in per_arm[arm]["seeds"]])
                           if cost_fn is None else cost_fn(arm))
    return {"implied": implied,
            "spread": {"min": float(vals.min()), "max": float(vals.max()),
                       "mean": float(vals.mean()), "median": float(np.median(vals)),
                       "cv": float(vals.std(ddof=0) / vals.mean()),
                       "max_over_min": float(vals.max() / vals.min())},
            "best_single_baseline": best_single_baseline(
                {l: v["absolute_cost_usd"] for l, v in implied.items()}),
            "baseline_scan": baseline_scan(
                {l: v["absolute_cost_usd"] for l, v in implied.items()}, candidates),
            "candidate_baselines_usd": candidates}


def best_single_baseline(costs: Dict[str, float]) -> Dict[str, Any]:
    labels = [l for l in ROW_ORDER if l in costs]
    c = np.array([costs[l] for l in labels])
    p = np.array([PUBLISHED_COST[l] for l in labels])
    b = float((c ** 2).sum() / (c * p).sum())
    resid = c / b - p
    return {"baseline_usd": b,
            "residuals": {l: float(r) for l, r in zip(labels, resid)},
            "ratios_at_best_baseline": {l: float(x) for l, x in zip(labels, c / b)},
            "mean_abs_residual": float(np.abs(resid).mean()),
            "max_abs_residual": float(np.abs(resid).max()),
            "rows_within_0_05": int((np.abs(resid) <= 0.05).sum()),
            "rows_within_0_10": int((np.abs(resid) <= 0.10).sum()),
            "n_rows": len(labels)}


def baseline_scan(costs: Dict[str, float], candidates: Dict[str, float],
                  lo: float = 12000.0, hi: float = 40000.0,
                  step: float = 5.0) -> Dict[str, Any]:
    labels = [l for l in ROW_ORDER if l in costs]
    c = np.array([costs[l] for l in labels])
    p = np.array([PUBLISHED_COST[l] for l in labels])
    grid = np.arange(lo, hi + step, step)
    resid = c[None, :] / grid[:, None] - p[None, :]
    out: Dict[str, Any] = {"tolerances": {}}
    for tol in (0.05, 0.10):
        inside = (np.abs(resid) <= tol)
        counts = inside.sum(axis=1)
        best = int(counts.max())
        where = grid[counts == best]
        k = int(np.flatnonzero(counts == best)[len(np.flatnonzero(counts == best)) // 2])
        out["tolerances"][f"{tol:.2f}"] = {
            "max_rows_explained": best, "of": len(labels),
            "baseline_window_usd": [float(where.min()), float(where.max())],
            "rows_explained_at_window_centre": [l for l, ok in zip(labels, inside[k]) if ok],
            "rows_missed_at_window_centre": [l for l, ok in zip(labels, inside[k]) if not ok],
        }
    out["runnable_candidates"] = {}
    for name, b in sorted(candidates.items()):
        r = c / b - p
        out["runnable_candidates"][name] = {
            "baseline_usd": b,
            "ratios": {l: float(x) for l, x in zip(labels, c / b)},
            "residuals": {l: float(x) for l, x in zip(labels, r)},
            "mean_abs_residual": float(np.abs(r).mean()),
            "rows_within_0_05": int((np.abs(r) <= 0.05).sum()),
            "rows_within_0_10": int((np.abs(r) <= 0.10).sum()),
        }
    return out


def hypothesis_3(per_arm, rows, variants) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name, kw in variants.items():
        def cost_of(arm: str, kw=kw) -> float:
            blob = per_arm[arm]
            vals = []
            for net, price in zip(blob["net"], blob["price"]):
                p = kw["price"](price)
                vals.append(_cost(net, p, credit_exports=kw.get("credit_exports", False),
                                  district_netting=kw.get("district_netting", False)))
            return _mean(vals)

        base = cost_of(BASELINE_ARM)
        rowdata = {}
        for arm, label in rows:
            c = cost_of(arm)
            rowdata[label] = {"arm": arm, "absolute_cost_usd": c, "ratio": c / base,
                              "published": PUBLISHED_COST[label],
                              "residual": c / base - PUBLISHED_COST[label],
                              "implied_baseline_usd": c / PUBLISHED_COST[label]}
        vals = np.array([r["implied_baseline_usd"] for r in rowdata.values()])
        res = np.array([r["residual"] for r in rowdata.values()])
        out[name] = {
            "baseline_cost_usd": base,
            "rows": rowdata,
            "mean_abs_residual": float(np.abs(res).mean()),
            "rows_within_0_05": int((np.abs(res) <= 0.05).sum()),
            "implied_baseline_cv": float(vals.std(ddof=0) / vals.mean()),
            "implied_baseline_range": [float(vals.min()), float(vals.max())],
            "best_single_baseline": best_single_baseline(
                {l: v["absolute_cost_usd"] for l, v in rowdata.items()}),
            "baseline_scan": baseline_scan(
                {l: v["absolute_cost_usd"] for l, v in rowdata.items()},
                {arm: cost_of(arm) for arm in per_arm if arm not in dict(rows)},
                lo=2000.0, hi=40000.0),
            "candidate_baselines_usd": {
                arm: cost_of(arm) for arm in per_arm if arm not in dict(rows)},
        }
    return out


def tou_reachability(per_arm, rows) -> Dict[str, Any]:
    base = per_arm[BASELINE_ARM]
    B_h = np.zeros(24)
    for net in base["net"]:
        imported = np.maximum(net, 0.0).sum(axis=1)
        for h in range(24):
            B_h[h] += imported[h::24].sum()
    B_h /= len(base["net"])
    out = {}
    for arm, label in rows:
        A_h = np.zeros(24)
        for net in per_arm[arm]["net"]:
            imported = np.maximum(net, 0.0).sum(axis=1)
            for h in range(24):
                A_h[h] += imported[h::24].sum()
        A_h /= len(per_arm[arm]["net"])
        hourly = A_h / B_h
        lo, hi = float(hourly.min()), float(hourly.max())
        pub = PUBLISHED_COST[label]
        out[label] = {"arm": arm, "hourly_ratio_min": lo, "hourly_ratio_max": hi,
                      "published": pub, "reachable": bool(lo <= pub <= hi),
                      "argmin_hour": int(hourly.argmin()),
                      "hourly_ratios": hourly.tolist()}
    return {"baseline_hourly_import_kwh": B_h.tolist(), "rows": out,
            "n_reachable": sum(1 for v in out.values() if v["reachable"]),
            "n_rows": len(out)}


def minimum_tariff_spread(per_arm, rows, *, baseline_usd: Optional[float] = None,
                          subset: Optional[List[str]] = None,
                          hi: float = 1e6) -> Dict[str, Any]:
    from scipy.optimize import linprog

    labels = [l for _, l in rows] if subset is None else list(subset)
    arm_of = {l: a for a, l in rows}

    def hour_profile(arm: str) -> np.ndarray:
        acc = np.zeros(24)
        for net in per_arm[arm]["net"]:
            imported = np.maximum(net, 0.0).sum(axis=1)
            for h in range(24):
                acc[h] += imported[h::24].sum()
        return acc / len(per_arm[arm]["net"])

    B_h = hour_profile(BASELINE_ARM)
    if baseline_usd is not None:
        B_h = B_h * (baseline_usd / float(B_h.sum()))
    A = np.vstack([hour_profile(arm_of[l]) for l in labels])
    r = np.array([PUBLISHED_COST[l] for l in labels])
    M = A - r[:, None] * B_h[None, :]

    def feasible(s: float) -> bool:
        res = linprog(c=np.zeros(24), A_eq=M, b_eq=np.zeros(len(labels)),
                      bounds=[(1.0, s)] * 24, method="highs")
        return bool(res.status == 0)

    nonneg = linprog(c=np.zeros(24), A_eq=np.vstack([M, np.ones((1, 24))]),
                     b_eq=np.concatenate([np.zeros(len(labels)), [24.0]]),
                     bounds=[(0.0, None)] * 24, method="highs")
    nonneg_ok = bool(nonneg.status == 0)
    zero_hours = (None if not nonneg_ok
                  else int((np.asarray(nonneg.x) < 1e-9).sum()))

    if not feasible(hi):
        return {"rows": labels, "feasible_at_any_spread": False,
                "min_peak_over_offpeak": None,
                "feasible_with_nonnegative_prices": nonneg_ok,
                "hours_priced_at_zero_in_witness": zero_hours}
    lo = 1.0
    if feasible(lo):
        return {"rows": labels, "feasible_at_any_spread": True,
                "min_peak_over_offpeak": 1.0,
                "note": "a flat tariff already reproduces these rows exactly"}
    for _ in range(60):
        mid = (lo * hi) ** 0.5
        if feasible(mid):
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.0001:
            break
    res = linprog(c=np.zeros(24), A_eq=M, b_eq=np.zeros(len(labels)),
                  bounds=[(1.0, hi)] * 24, method="highs")
    return {"rows": labels, "feasible_at_any_spread": True,
            "min_peak_over_offpeak": float(hi),
            "feasible_with_nonnegative_prices": nonneg_ok,
            "hours_priced_at_zero_in_witness": zero_hours,
            "witness_tariff_normalised": (None if res.status != 0
                                          else [float(x) for x in res.x]),
            "baseline_usd": (None if baseline_usd is None else float(baseline_usd))}


def stage_analyse(out: Path, results: Path, unshielded: Optional[Path]) -> Dict[str, Any]:
    per_arm = _load(out)
    missing = [a for a in list(ARM_TO_PUBLISHED) + [BASELINE_ARM] if a not in per_arm]
    if missing:
        raise SystemExit(f"no replayed series for {missing}; run --stage dump first")
    rows = [(arm, label) for arm, label in ARM_TO_PUBLISHED.items()]
    rows.sort(key=lambda t: ROW_ORDER.index(t[1]))

    price0 = per_arm[BASELINE_ARM]["price"][0]
    mean_price = float(price0.mean())
    levels = sorted(set(np.round(price0[:, 0], 6).tolist()))

    def flatten(k: float):
        def f(price: np.ndarray) -> np.ndarray:
            mu = float(price.mean())
            return mu + k * (price - mu)
        return f

    variants = {
        "tou_actual": {"price": lambda p: p},
        "flat_mean": {"price": flatten(0.0)},
        "tou_flattened_50pct": {"price": flatten(0.5)},
        "tou_flattened_25pct": {"price": flatten(0.25)},
        "tou_sharpened_150pct": {"price": flatten(1.5)},
        "tou_actual_export_credit": {"price": lambda p: p, "credit_exports": True},
        "tou_actual_district_netting": {"price": lambda p: p, "district_netting": True},
        "flat_mean_export_credit": {"price": flatten(0.0), "credit_exports": True},
    }

    report = {
        "tariff": {"mean_usd_per_kwh": mean_price, "levels": levels,
                   "peak_over_offpeak": float(max(levels) / min(levels))},
        "replay_drift": {
            arm: {"max_abs_cost_drift": float(np.max(np.abs(
                [r["replayed_cost"] - r["stored_cost"] for r in blob["seeds"]
                 if r["stored_cost"] is not None] or [0.0])))}
            for arm, blob in per_arm.items()},
        "h0_shield_precheck": (None if unshielded is None or not unshielded.is_dir()
                               else shield_precheck(results, unshielded)),
        "h0_budget_premise_check": budget_premise_check(
            [d for d in EP50_DIRS if Path(d).is_dir()],
            _mean([x["replayed_cost"] for x in per_arm[BASELINE_ARM]["seeds"]])),
        "h1_aggregation": hypothesis_1(per_arm, rows),
        "h2_normalising_rule": hypothesis_2(per_arm, rows),
        "h3_tariff_and_exports": hypothesis_3(per_arm, rows, variants),
        "h3_tou_reachability": tou_reachability(per_arm, rows),
        "h3_minimum_tariff_spread": {
            "all_eight_on_our_rule": minimum_tariff_spread(per_arm, rows),
            "all_eight_on_idle_strength_rule": minimum_tariff_spread(
                per_arm, rows,
                baseline_usd=_mean([x["replayed_cost"]
                                    for x in per_arm["idle"]["seeds"]])
                if "idle" in per_arm else None),
            "five_competitive_rows": minimum_tariff_spread(
                per_arm, rows,
                subset=["MPC", "MARLISA", "MADCQ", "D-MAPPO", "STEMS"]),
        },
    }

    cons_base = _mean([r["replayed_consumption"] for r in per_arm[BASELINE_ARM]["seeds"]])
    report["flat_tariff_prediction"] = {
        label: {
            "ours_cost_ratio_flat": report["h3_tariff_and_exports"]["flat_mean"]["rows"][label]["ratio"],
            "ours_consumption_ratio": _mean(
                [r["replayed_consumption"] for r in per_arm[arm]["seeds"]]) / cons_base,
            "published_cost_ratio": PUBLISHED_COST[label],
            "published_consumption_ratio": PUBLISHED_CONSUMPTION[label],
            "published_cost_minus_consumption":
                PUBLISHED_COST[label] - PUBLISHED_CONSUMPTION[label],
        }
        for arm, label in rows
    }

    h1 = report["h1_aggregation"]["rows"]
    monthly_implied = {label: h1[label]["monthly_mean_ratio"] / PUBLISHED_COST[label]
                       for label in ROW_ORDER}
    report["h1_plus_h2"] = {
        "monthly_mean_ratio_over_published": monthly_implied,
        "note": ("a scale factor here is the *multiple* of our baseline that the "
                 "monthly-mean reading would need; 1.0 would mean the published ratio "
                 "is reproduced exactly under monthly averaging"),
    }

    (out / "_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    _print_report(report)
    return report


def _print_report(r: Dict[str, Any]) -> None:
    t = r["tariff"]
    print(f"\ntariff as served: mean {t['mean_usd_per_kwh']:.5f} USD/kWh, levels "
          f"{t['levels']}, peak/offpeak {t['peak_over_offpeak']:.3f}")
    drift = max(v["max_abs_cost_drift"] for v in r["replay_drift"].values())
    print(f"replay drift vs stored records: max |delta cost| = {drift:.6f} USD\n")

    hb = r.get("h0_budget_premise_check")
    if hb and hb["rows"]:
        print("H0  premise check: does the gap survive a 50-episode budget? "
              f"(same baseline, {hb['baseline_usd']:.0f} USD)")
        print(f"{'row':18s} {'ep50 USD':>9s} {'ep50 ratio':>10s} {'published':>9s} "
              f"{'gap':>7s} {'seeds':>6s} {'implied base USD':>17s}")
        for label in ROW_ORDER:
            d = hb["rows"].get(label)
            if d is None:
                continue
            print(f"{label:18s} {d['cost_usd']:9.0f} {d['ratio']:10.3f} "
                  f"{d['published']:9.3f} {d['gap']:+7.3f} {d['n_seeds']:6d} "
                  f"{d['implied_baseline_usd']:17.0f}")
        print(f"  rows within 0.025 of published at 50 episodes, on our own rule: "
              f"{', '.join(hb['rows_within_0_025']) or 'none'}")
        print(f"  rows still more than 0.05 above: "
              f"{', '.join(hb['rows_still_above_0_05']) or 'none'}\n")

    h0 = r.get("h0_shield_precheck")
    if h0:
        print("H0  pre-check: is our own safety projection the cost inflator?")
        print(f"{'row':18s} {'shielded':>9s} {'unshielded':>11s} {'shield eff':>11s} "
              f"{'published':>9s}")
        for label in ROW_ORDER:
            d = h0["rows"].get(label)
            if d is None:
                continue
            print(f"{label:18s} {d['shielded_ratio']:9.3f} {d['unshielded_ratio']:11.3f} "
                  f"{d['shield_effect_on_ratio']:+11.3f} {d['published']:9.3f}")
        print(f"  largest shield effect on a cost ratio: "
              f"{h0['max_abs_shield_effect_on_ratio']:.3f}; rows still above published "
              f"unshielded: {h0['rows_still_above_published_unshielded']}"
              f"/{h0['n_rows_compared']}\n")

    print("H1  aggregation: annual-total ratio vs mean of twelve monthly ratios")
    print(f"{'row':18s} {'published':>9s} {'annual':>8s} {'monthly':>8s} {'shift':>8s} "
          f"{'gap_annual':>10s} {'gap_monthly':>11s} {'month ratio range':>19s}")
    h1 = r["h1_aggregation"]["rows"]
    for label in ROW_ORDER:
        d = h1[label]
        print(f"{label:18s} {d['published']:9.3f} {d['annual_ratio']:8.3f} "
              f"{d['monthly_mean_ratio']:8.3f} "
              f"{d['monthly_mean_ratio'] - d['annual_ratio']:+8.3f} "
              f"{d['annual_ratio'] - d['published']:+10.3f} "
              f"{d['monthly_mean_ratio'] - d['published']:+11.3f} "
              f"{d['monthly_ratio_min']:8.3f}-{d['monthly_ratio_max']:<10.3f}")
    bm = r["h1_aggregation"]["baseline_monthly_cost"]
    print(f"  baseline monthly cost USD, Jan-Dec: "
          f"{', '.join(format(x, '.0f') for x in bm)}")

    print("\nH2  normalising rule: baseline cost each published ratio implies")
    h2 = r["h2_normalising_rule"]
    print(f"{'row':18s} {'our abs USD':>12s} {'published':>9s} {'implied base USD':>17s}")
    for label in ROW_ORDER:
        d = h2["implied"][label]
        print(f"{label:18s} {d['absolute_cost_usd']:12.1f} {d['published_ratio']:9.3f} "
              f"{d['implied_baseline_usd']:17.1f}")
    s = h2["spread"]
    print(f"  implied baseline: {s['min']:.0f} to {s['max']:.0f} USD "
          f"(median {s['median']:.0f}, CV {s['cv']:.3f}, max/min {s['max_over_min']:.3f})")
    print("  candidate rules we can run: " + ", ".join(
        f"{k}={v:.0f}" for k, v in sorted(h2["candidate_baselines_usd"].items())))
    bb = h2["best_single_baseline"]
    print(f"  best single baseline {bb['baseline_usd']:.0f} USD: mean|resid| "
          f"{bb['mean_abs_residual']:.3f}, max {bb['max_abs_residual']:.3f}, within 0.05: "
          f"{bb['rows_within_0_05']}/{bb['n_rows']}, within 0.10: "
          f"{bb['rows_within_0_10']}/{bb['n_rows']}")
    for label in ROW_ORDER:
        print(f"      {label:18s} ratio {bb['ratios_at_best_baseline'][label]:6.3f} "
              f"vs published {PUBLISHED_COST[label]:.3f}  "
              f"resid {bb['residuals'][label]:+.3f}")
    sc = h2["baseline_scan"]
    for tol, d in sc["tolerances"].items():
        lo, hi = d["baseline_window_usd"]
        print(f"  sweeping the baseline, tolerance {tol}: at best "
              f"{d['max_rows_explained']}/{d['of']} rows, for baselines "
              f"{lo:.0f}-{hi:.0f} USD; missed: "
              f"{', '.join(d['rows_missed_at_window_centre']) or 'none'}")
    print("  at the baselines a rule actually produces:")
    for name, d in sc["runnable_candidates"].items():
        print(f"    {name:20s} {d['baseline_usd']:8.0f} USD  mean|resid| "
              f"{d['mean_abs_residual']:.3f}  within 0.05: {d['rows_within_0_05']}/8  "
              f"within 0.10: {d['rows_within_0_10']}/8")

    print("\nH3  tariff and export treatment")
    print(f"{'variant':30s} {'base USD':>9s} {'mean|resid|':>11s} {'within .05':>10s} "
          f"{'implied CV':>10s} {'implied range USD':>22s}")
    for name, d in r["h3_tariff_and_exports"].items():
        lo, hi = d["implied_baseline_range"]
        print(f"{name:30s} {d['baseline_cost_usd']:9.0f} {d['mean_abs_residual']:11.3f} "
              f"{d['rows_within_0_05']:10d} {d['implied_baseline_cv']:10.3f} "
              f"{lo:10.0f}-{hi:<11.0f}")

    reach = r["h3_tou_reachability"]
    print("\nH3  sharp bound: range of the cost ratio over every 24h-periodic tariff")
    print(f"{'row':18s} {'min':>7s} {'max':>7s} {'published':>9s} {'reachable':>10s}")
    for label in ROW_ORDER:
        d = reach["rows"][label]
        print(f"{label:18s} {d['hourly_ratio_min']:7.3f} {d['hourly_ratio_max']:7.3f} "
              f"{d['published']:9.3f} {'yes' if d['reachable'] else 'NO':>10s}")
    print(f"  published ratio reachable by some time-of-use tariff on our baseline: "
          f"{reach['n_reachable']}/{reach['n_rows']}")

    print("\nH3  flattest 24h tariff reproducing the published ratios exactly "
          f"(tariff in the data: {t['peak_over_offpeak']:.3f})")
    for name, d in r["h3_minimum_tariff_spread"].items():
        tail = (f"non-negative prices feasible: "
                f"{d.get('feasible_with_nonnegative_prices')}"
                + ("" if d.get("hours_priced_at_zero_in_witness") is None
                   else f", {d['hours_priced_at_zero_in_witness']}/24 hours priced at 0"))
        if not d.get("feasible_at_any_spread"):
            print(f"  {name:32s} infeasible at any peak-to-offpeak ratio; {tail}")
        else:
            print(f"  {name:32s} needs peak/offpeak >= "
                  f"{d['min_peak_over_offpeak']:.2f}  ({len(d['rows'])} rows); {tail}")

    print("\nH2+H3 jointly: best single baseline allowed under each tariff variant")
    print(f"{'variant':30s} {'rows<=.10':>9s} {'baseline window USD':>21s} "
          f"{'a runnable rule in window?':>27s}")
    for name, d in r["h3_tariff_and_exports"].items():
        t = d["baseline_scan"]["tolerances"]["0.10"]
        lo, hi = t["baseline_window_usd"]
        reach = [k for k, v in d["candidate_baselines_usd"].items() if lo <= v <= hi]
        print(f"{name:30s} {t['max_rows_explained']:4d}/{t['of']:<4d} "
              f"{lo:9.0f}-{hi:<11.0f} {', '.join(reach) or 'none':>27s}")

    print("\nflat-tariff prediction: cost ratio should equal consumption ratio")
    print(f"{'row':18s} {'ours flat':>9s} {'ours cons':>9s} {'pub cost':>9s} "
          f"{'pub cons':>9s} {'pub c-c':>8s}")
    for label in ROW_ORDER:
        d = r["flat_tariff_prediction"][label]
        print(f"{label:18s} {d['ours_cost_ratio_flat']:9.3f} "
              f"{d['ours_consumption_ratio']:9.3f} {d['published_cost_ratio']:9.3f} "
              f"{d['published_consumption_ratio']:9.3f} "
              f"{d['published_cost_minus_consumption']:+8.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("dump", "analyse"), required=True)
    ap.add_argument("--results", default="results/t1_repro_ep15")
    ap.add_argument("--out", default="experiments/diagnostics/cost_column")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--unshielded", default="results/t1_unshielded_ep15",
                    help="matching barrier='none' records, for the H0 pre-check")
    args = ap.parse_args()
    out = Path(args.out)
    if args.stage == "dump":
        stage_dump(Path(args.results), out, args.workers, args.arms, args.seeds)
    else:
        stage_analyse(out, Path(args.results), Path(args.unshielded))


if __name__ == "__main__":
    main()
