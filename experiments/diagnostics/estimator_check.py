from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

_CALENDAR_FROM_AUGUST = [31, 30, 31, 30, 31, 31, 28, 31, 30, 31, 30, 31]
_MONTH_NAMES = ["Aug", "Sep", "Oct", "Nov", "Dec", "Jan",
                "Feb", "Mar", "Apr", "May", "Jun", "Jul"]

_RATIO_COLUMNS = [
    ("cost", "stems_cost"),
    ("emission", "stems_emission"),
    ("avg_daily_peak", "stems_avg_daily_peak"),
    ("consumption", "stems_electricity_consumption"),
    ("ramping", "stems_ramping_rate"),
]


def month_windows(n_steps: int, hours_per_step: float = 1.0) -> List[Tuple[str, int, int]]:
    steps_per_day = int(round(24 / hours_per_step))
    full_year = sum(_CALENDAR_FROM_AUGUST) * steps_per_day
    if n_steps == full_year:
        out, cursor = [], 0
        for name, days in zip(_MONTH_NAMES, _CALENDAR_FROM_AUGUST):
            stop = cursor + days * steps_per_day
            out.append((name, cursor, stop))
            cursor = stop
        return out
    block = n_steps // 12
    if block == 0:
        raise ValueError(f"month_windows: {n_steps} steps is too short to split into 12")
    return [(f"block{i + 1}", i * block, (i + 1) * block if i < 11 else n_steps)
            for i in range(12)]


def monthly_kpis(accumulator, windows: List[Tuple[str, int, int]]) -> List[Dict[str, float]]:
    return [accumulator.slice_steps(a, b).compute_all() for _, a, b in windows]


def both_estimators(arm_months: List[Dict[str, float]],
                    rule_months: List[Dict[str, float]],
                    arm_annual: Dict[str, float],
                    rule_annual: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for label, key in _RATIO_COLUMNS:
        denom_annual = rule_annual[key]
        annual = arm_annual[key] / denom_annual if denom_annual else float("nan")
        per_month = []
        for arm_m, rule_m in zip(arm_months, rule_months):
            d = rule_m[key]
            per_month.append(arm_m[key] / d if d else float("nan"))
        finite = [r for r in per_month if r == r]
        monthly = sum(finite) / len(finite) if finite else float("nan")
        out[label] = {
            "annual_total_ratio": annual,
            "monthly_mean_ratio": monthly,
            "difference": monthly - annual,
            "per_month": per_month,
            "rule_monthly_level": [m[key] for m in rule_months],
        }
    return out


def one_run(arm_name: str, scenario, config, seed: int) -> Dict[str, Any]:
    from stems.environment import STEMSEnvironment
    from stems.utils import set_seed
    from experiments.controllers import ARMS, build_controller
    from experiments.runner import evaluate

    set_seed(seed)
    env = STEMSEnvironment(schema=scenario.schema_path(), seed=seed,
                           heat_pump=scenario.heat_pump,
                           env_kwargs=scenario.env_kwargs("eval"),
                           hvac_control=scenario.hvac_control,
                           allow_missing_obs=scenario.allow_missing_obs)
    controller = build_controller(ARMS[arm_name], env, config)
    return evaluate(controller, env, config, max_steps=10 ** 9,
                    return_accumulator=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="tx_travis_8b")
    ap.add_argument("--window", default="year")
    ap.add_argument("--arms", nargs="+",
                    default=["idle", "rbc+calibrated", "mpc"],
                    help="deterministic arms to compare against the unshielded rule")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--grid-cap", type=float, default=2000.0)
    ap.add_argument("--building-cap", type=float, default=500.0)
    ap.add_argument("--out", default="results/estimator_check")
    args = ap.parse_args()

    from experiments.runner import make_config
    from experiments.scenario import Scenario

    scenario = Scenario(season=args.window, grid_cap_kw=args.grid_cap,
                        building_cap_kw=args.building_cap)
    config = make_config(scenario)

    rule = one_run("rbc", scenario, config, args.seed)
    windows = month_windows(rule["steps"])
    rule_months = monthly_kpis(rule["accumulator"], windows)

    report: Dict[str, Any] = {
        "scenario": args.scenario, "window": args.window, "seed": args.seed,
        "steps": rule["steps"],
        "windows": [{"name": n, "start": a, "stop": b} for n, a, b in windows],
        "normaliser": "rbc (unshielded rule)",
        "arms": {},
    }
    for arm_name in args.arms:
        run = one_run(arm_name, scenario, config, args.seed)
        if run["steps"] != rule["steps"]:
            raise RuntimeError(
                f"{arm_name} ran {run['steps']} steps but the rule ran {rule['steps']}; "
                "the two estimators would not be comparable")
        report["arms"][arm_name] = both_estimators(
            monthly_kpis(run["accumulator"], windows), rule_months,
            run["kpis"], rule["kpis"])

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "estimator_check.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"{'arm':<16}{'column':<16}{'annual':>9}{'monthly':>9}{'diff':>9}")
    for arm_name, cols in report["arms"].items():
        for label, d in cols.items():
            print(f"{arm_name:<16}{label:<16}{d['annual_total_ratio']:>9.4f}"
                  f"{d['monthly_mean_ratio']:>9.4f}{d['difference']:>+9.4f}")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
