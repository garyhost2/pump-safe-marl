#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LEG = REPO / "experiments" / "diagnostics" / "legionella"
DHW = REPO / "experiments" / "diagnostics" / "dhw_band"
SIZINGS = (0.25, 0.5, 1.0, 2.0)
PLANT_ORDER = ("sizing0.25", "sizing0.5", "sizing1", "sizing2", "thermia_1_12")


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def legionella_tables() -> None:
    for steps in (336, 8759):
        d = _load(LEG / f"recourse_{steps}.json")
        if d is None:
            continue
        print(f"\n**ShadowTank, {d['steps']} h x {d['buildings']} buildings** -- missed windows "
              f"by sizing (0.25x / 0.5x / 1x / 2x); `R` = refused by `LegionellaSpec` "
              f"(reserve >= source output per step)\n")
        print("| reserve q | jit missed | recourse missed | jit monotone | recourse monotone |")
        print("|---|---|---|---|---|")
        v = d["monotone_in_source_size"]
        for q in (0.5, 0.75, 0.9, 0.95, 1.0):
            j, r = v[f"jit_q{q}"], v[f"recourse_q{q}"]
            fmt = lambda xs: " / ".join("R" if x is None else str(x) for x in xs)
            print(f"| {q} | {fmt(j['missed_by_sizing'])} | {fmt(r['missed_by_sizing'])} | "
                  f"{j['monotone_missed']} | {r['monotone_missed']} |")
        print(f"\nElectricity (kWh) and unmet hot water (kWh), {d['steps']} h:\n")
        print("| sizing | no cycle | jit q0.5 | recourse q0.5 | jit q0.9 | recourse q0.9 | "
              "unmet (no cycle) |")
        print("|---|---|---|---|---|---|---|")
        for m in SIZINGS:
            row = {}
            for x in d["runs"]:
                if x["sizing"] != m:
                    continue
                key = f"{x['trigger']}_{x['reserve_q']}"
                row[key] = "R" if "refused" in x else f"{x['electricity_kwh']:.1f}"
                if x["trigger"] == "none":
                    row["unmet"] = f"{x['unmet_kwh']:.2f}"
            print(f"| {m}x | {row.get('none_None')} | {row.get('jit_0.5')} | "
                  f"{row.get('recourse_0.5')} | {row.get('jit_0.9')} | "
                  f"{row.get('recourse_0.9')} | {row.get('unmet')} |")


def _runs(d, **kw):
    return [r for r in d["runs"] if all(r.get(k) == v for k, v in kw.items())]


def dhw_tables(name: str = "diagnostic_8759.json") -> None:
    d = _load(DHW / name)
    if d is None:
        return
    print(f"\n**DHWPlant, {d['steps']} h x {d['buildings']} buildings, `{name}`** "
          f"(barrier draw: {d['barrier_draw']}, guard: {d.get('guard', 'strict')}). "
          f"`viol` = building-steps with a post-step node outside its phase band "
          f"(floor/ceiling); `empty` = reported empty intersections; every Thermia "
          f"parameter is an unsourced fixture.\n")
    print("| plant | nodes | policy | shield | viol (floor/ceil) | empty | refusals "
          "(ceil/floor) | unmet kWh | events | missed | elec kWh | cost | starts |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for p in PLANT_ORDER:
        for r in _runs(d, plant=p):
            sh = "off" if r["shield"] == "off" else r["trigger"]
            emp = "-" if r["empty_intersections"] is None else r["empty_intersections"]
            print(f"| {p} | {r['nodes']} | {r['policy']} | {sh} | "
                  f"{r['floor_violation_building_steps']}/{r['ceiling_violation_building_steps']} | "
                  f"{emp} | {r['refusals_above_ceiling']}/{r['refusals_below_floor']} | "
                  f"{r['unmet_kwh']:.2f} | {r['completion_events']} | {r['missed_windows']} | "
                  f"{r['electricity_kwh']:.1f} | {r['cost']:.2f} | {r['compressor_starts']} |")
    print(f"\nClaim 2 floor condition fails from the worst in-band state on these "
          f"building-steps (of {d['steps'] * d['buildings']}): "
          f"{d['claim2_floor_condition_failures_from_worst_in_band_state']}")
    mx = max(r["model_vs_plant_max_abs_kwh"] for r in d["runs"])
    res = max(r["max_band_residual_on_nonempty_steps"] for r in d["runs"])
    print(f"\nAcross all {len(d['runs'])} runs: max |model - plant| = {mx} kWh; max band "
          f"residual on a non-empty step (model prediction) = {res}.")


def battery_table() -> None:
    d = _load(DHW / "battery_crossing.json")
    if d is None:
        return
    print(f"\n**Battery `safe_interval` crossings**: {d['crossed_total']} of "
          f"{d['evaluations_total']} evaluations crossed; max change from the flag "
          f"{max(r['max_abs_change_from_flag'] for r in d['rows'])}.")


if __name__ == "__main__":
    which = sys.argv[1:] or ["legionella", "dhw", "battery"]
    if "legionella" in which:
        legionella_tables()
    if "dhw" in which:
        dhw_tables()
        for extra in ("diagnostic_8759_forecastdraw.json", "diagnostic_8759_ceilingguard.json"):
            if (DHW / extra).exists():
                dhw_tables(extra)
    if "battery" in which:
        battery_table()
