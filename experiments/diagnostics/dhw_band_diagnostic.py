#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stems.dhw_barrier import (PHASE_DISINFECT, PHASE_NORMAL, CyclePhaseBarrier,
                               DHWStoreModel, HourOfDayDrawForecast, PhaseBands)
from stems.dhw_plant import DHWPlant, DHWPlantFleet, TankSpec
from stems.heatpump_thermia import ThermiaSpec, ThermiaUnit

SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
T_COLD, T_RATED, T_FLOOR, T_CEIL, T_LEG = 10.0, 60.0, 45.0, 55.0, 60.0
INITIAL_SOC = 0.80
PERIOD = 168.0
CHEAP_PRICE = 0.30
RESERVE_Q = 0.9
PLANTS = ("sizing0.25", "sizing0.5", "sizing1", "sizing2", "thermia_1_12")
NODES = ("1", "2k0", "2k0.1")
POLICIES = ("thermostat", "price_aware")
SHIELDS = (("off", None), ("on", "recourse"), ("on", "jit"))

_DATA: Dict[str, Any] = {}


def load_series(steps: int, npz: Optional[str]) -> Dict[str, Any]:
    if npz:
        z = np.load(npz)
        names = [str(x) for x in z["names"]]
        a = z["data"][:steps]
        col = lambda n: a[:, :, names.index(n)]
        return {"draw": col("dhw_demand"), "t_out": col("outdoor_dry_bulb_temperature")[:, 0],
                "price": col("electricity_pricing")[:, 0],
                "capacity": np.asarray(z["capacity"], dtype=np.float64),
                "loss": np.asarray(z["loss_coefficient"], dtype=np.float64),
                "source": f"cache {npz} of {SCHEMA}"}
    from stems.environment import STEMSEnvironment
    from stems.observations import obs_indices
    i_d, i_t, i_p = obs_indices("dhw_demand", "outdoor_dry_bulb_temperature",
                                "electricity_pricing")
    kw = {"env_kwargs": {"episode_time_steps": [(0, steps)]}} if steps < 8759 else {}
    env = STEMSEnvironment(schema=SCHEMA, seed=0, heat_pump=True, **kw)
    info = env.dhw_info()
    obs, _ = env.reset()
    rows = [obs]
    zero = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    while len(rows) < steps:
        obs, _r, term, trunc, _i = env.step(zero)
        if term or trunc:
            break
        rows.append(obs)
    pick = lambda i: np.array([[float(o[i]) for o in r] for r in rows])
    return {"draw": pick(i_d), "t_out": pick(i_t)[:, 0], "price": pick(i_p)[:, 0],
            "capacity": np.asarray(info["capacity"], dtype=np.float64),
            "loss": np.asarray(info["loss_coefficient"], dtype=np.float64),
            "source": f"STEMSEnvironment({SCHEMA}, seed=0), zero actions"}


def unit_spec(plant: str, p_hp: float) -> ThermiaSpec:
    if plant == "thermia_1_12":
        return ThermiaSpec(power_min_kw=1.0, power_max_kw=12.0, eta_ref=0.45,
                           t_sink_ref_c=35.0, eta_slope_per_k=-0.004, capacity_ref_kw=12.0,
                           t_source_ref_c=0.0, capacity_slope_per_k_sink=-0.12,
                           capacity_slope_per_k_source=0.10, t_sink_max_c=65.0,
                           element_kw=6.0, element_efficiency=1.0, cycling_penalty_kwh=0.0)
    return ThermiaSpec(power_min_kw=max(0.2, p_hp * 0.2), power_max_kw=max(0.5, p_hp),
                       eta_ref=0.45, t_sink_ref_c=35.0, eta_slope_per_k=-0.004,
                       capacity_ref_kw=max(0.5, p_hp), t_source_ref_c=0.0,
                       capacity_slope_per_k_sink=-0.01, capacity_slope_per_k_source=0.02,
                       t_sink_max_c=65.0, element_kw=0.0, element_efficiency=1.0,
                       cycling_penalty_kwh=0.0)


def build(plant: str, nodes: str, data: Dict[str, Any]):
    draw = data["draw"]
    p_hp = draw.mean(axis=0) * 24.0 / 4.0
    if plant.startswith("sizing"):
        p_hp = p_hp * float(plant[len("sizing"):])
    n = 1 if nodes == "1" else 2
    kappa = float(nodes.split("k")[1]) if n == 2 else 0.0
    tanks = [TankSpec(capacity_kwh=float(data["capacity"][i]), t_cold_c=T_COLD,
                      t_rated_c=T_RATED, loss_coefficient=float(data["loss"][i]), nodes=n,
                      top_fraction=0.5, node_conduction=kappa, charge_node="top",
                      provenance={"capacity_kwh": "schema dhw_storage.capacity",
                                  "loss_coefficient": "schema dhw_storage"})
             for i in range(draw.shape[1])]
    units = [unit_spec(plant, float(p_hp[i])) for i in range(draw.shape[1])]
    plants = [DHWPlant(tanks[i], ThermiaUnit(units[i]), initial_soc=INITIAL_SOC)
              for i in range(draw.shape[1])]
    model = DHWStoreModel(tanks, units, allow_element=False)
    bands = PhaseBands.from_temperatures(tanks[0], floor_c=T_FLOOR, ceiling_c=T_CEIL,
                                         disinfect_c=T_LEG, bottom="ceiling")
    return plants, model, bands, units, p_hp


def floor_condition_failures(model: DHWStoreModel, bands: PhaseBands,
                             data: Dict[str, Any]) -> int:
    lo = float(bands.normal[0, 0])
    E = model.caps * lo
    fails = 0
    for t in range(data["draw"].shape[0]):
        s = model.next_node_soc(E, np.ones(model.B), data["draw"][t], data["t_out"][t])
        fails += int(np.sum(s[:, 0] < lo))
    return fails


def run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    data = _DATA
    t0 = time.time()
    draw, t_out, price = data["draw"], data["t_out"], data["price"]
    T, B = draw.shape
    plants, model, bands, units, p_hp = build(cfg["plant"], cfg["nodes"], data)
    fleet = DHWPlantFleet(plants)
    fleet.reset(np.full(B, INITIAL_SOC))
    shielded = cfg["shield"] == "on"
    reserve = np.quantile(draw, RESERVE_Q, axis=0)
    fc = HourOfDayDrawForecast(B)
    barrier = CyclePhaseBarrier(
        model, bands, soc_legionella=1.0, period_steps=PERIOD,
        trigger=cfg["trigger"] or "jit", forecaster=fc,
        safety_stock_kwh=reserve if cfg["trigger"] == "recourse" else 0.0,
        jit_reserve_kwh=reserve if cfg["trigger"] == "jit" else 0.0,
        t_legionella_c=T_LEG, on_empty="floor_first",
        guard_includes_floor=cfg.get("guard", "strict") == "strict")
    rows = model._rows
    sink_caps = model._sink_caps
    n = model.n
    floor_viol = np.zeros(B, dtype=np.int64)
    ceil_viol = np.zeros(B, dtype=np.int64)
    top_floor_viol = np.zeros(B, dtype=np.int64)
    top_ceil_viol = np.zeros(B, dtype=np.int64)
    max_below_c = 0.0
    max_above_c = 0.0
    modified = 0
    events = np.zeros(B, dtype=np.int64)
    prev_done = np.zeros(B, dtype=bool)
    unmet_draws = 0
    elec = np.zeros(B)
    cost = np.zeros(B)
    max_fwd = 0.0
    span = T_RATED - T_COLD
    for t in range(T):
        E = model.energy_of(plants)
        node_soc = E / model.caps
        d, ts = draw[t], float(t_out[t])
        cheap = price[t] < CHEAP_PRICE
        if cfg["policy"] == "thermostat":
            q_req = None
            target = np.full(B, 0.80)
        else:
            due = barrier.k >= 120.0
            q_req = np.where(due, PHASE_DISINFECT, PHASE_NORMAL)
            in_d = (barrier.q == PHASE_DISINFECT) if shielded else (q_req == PHASE_DISINFECT)
            target = np.where(in_d & due, 1.0 if cheap else 0.71, 0.88 if cheap else 0.71)
        E_d, _dd, _sv, _um = model._after_draw(E, d)
        need = np.maximum(target * sink_caps - E_d[rows, model.sink], 0.0)
        u_pol = np.clip(need / model.request_scale_kwh, 0.0, 1.0)
        d_bar = d if cfg["barrier_draw"] == "known" else fc.forecast(1)[0]
        if shielded:
            dl = barrier.deadline(E, d_bar, ts)
            barrier.request_phase(barrier.desired_phase(q_req, dl["must"]), node_soc)
            res = barrier.project(u_pol, E, d_bar, ts, a_min=dl["a_min"])
            u = res.action
            modified += int(np.sum(np.abs(u - u_pol) > 1e-12))
        else:
            u = u_pol
            if q_req is not None:
                barrier.q = q_req.astype(np.int64)
        q_now = barrier.q.copy()
        pred = model.step(E, u, d, ts)
        steps = fleet.step(d, model.request_kwh(u), ts, allow_element=model.allow_element)
        E2 = model.energy_of(plants)
        max_fwd = max(max_fwd, float(np.max(np.abs(E2 - pred.energy_kwh))))
        e_step = np.array([s.electricity_kwh for s in steps])
        elec += e_step
        cost += e_step * price[t]
        unmet_draws += int(sum(s.unmet_kwh > 1e-9 for s in steps))
        s2 = E2 / model.caps
        lo, hi = bands.lo_hi(q_now)
        below, above = s2 < lo, s2 > hi
        floor_viol += below.any(axis=1)
        ceil_viol += above.any(axis=1)
        top_floor_viol += below[:, 0]
        top_ceil_viol += above[:, 0]
        if below.any():
            max_below_c = max(max_below_c, float(np.max(np.where(below, lo - s2, 0.0))) * span)
        if above.any():
            max_above_c = max(max_above_c, float(np.max(np.where(above, s2 - hi, 0.0))) * span)
        ob = barrier.observe(pred.post_charge_sink_soc, pred.pre_loss_node_soc)
        done = ob["completed"]
        events += (done & ~prev_done).astype(np.int64)
        prev_done = done
        fc.update(d)
    rep = fleet.report()
    br = barrier.report()
    return {
        **cfg,
        "steps": T, "buildings": B,
        "source_kw_rule": [round(float(x), 4) for x in p_hp],
        "unit_power_max_kw": [u.power_max_kw for u in units],
        "band_violation_building_steps": int(floor_viol.sum() + ceil_viol.sum()),
        "floor_violation_building_steps": int(floor_viol.sum()),
        "ceiling_violation_building_steps": int(ceil_viol.sum()),
        "top_floor_violations": int(top_floor_viol.sum()),
        "top_ceiling_violations": int(top_ceil_viol.sum()),
        "max_below_floor_c": max_below_c, "max_above_ceiling_c": max_above_c,
        "refusals": int(sum(br["refusals"])),
        "refusals_above_ceiling": int(sum(br["refusals_above_ceiling"])),
        "refusals_below_floor": int(sum(br["refusals_below_floor"])),
        "empty_intersections": int(sum(br["empty_steps"])) if shielded else None,
        "empty_by_reason": br["empty_by_reason"] if shielded else None,
        "unmet_kwh": rep["unmet_kwh"], "unmet_draw_steps": unmet_draws,
        "cycles_completed_raw": int(sum(br["cycles_completed"])),
        "completion_events": int(events.sum()),
        "missed_windows": int(sum(br["missed_windows"])),
        "windows_bottom_node_reached_60c": int(sum(br["windows_bottom_node_reached_level"])),
        "mean_window_bottom_peak_c": (None if br["mean_window_bottom_peak_soc"] is None
                                      else T_COLD + br["mean_window_bottom_peak_soc"] * span),
        "electricity_kwh": float(elec.sum()), "cost": float(cost.sum()),
        "compressor_starts": int(sum(rep["starts"])),
        "cycling_steps": int(sum(rep["cycling_steps"])),
        "spill_kwh": rep["spill_kwh"],
        "steps_in_D": int(sum(br["steps_in_D"])),
        "relaxed_ceiling_after_completion_steps":
            int(sum(br["relaxed_ceiling_after_completion_steps"])),
        "forced_steps": int(sum(br["forced_steps"])),
        "deadline_clipped_steps": int(sum(br["deadline_clipped_steps"])),
        "actions_modified": modified if shielded else None,
        "max_band_residual_on_nonempty_steps": br["max_band_residual_on_nonempty_steps"],
        "model_vs_plant_max_abs_kwh": max_fwd,
        "seconds": round(time.time() - t0, 1),
    }


def _init(data: Dict[str, Any]) -> None:
    _DATA.update(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8759)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--draws-npz", default=None,
                    help="cache of the environment's series (same schema); default: read "
                         "them from the environment")
    ap.add_argument("--barrier-draw", default="known", choices=("known", "forecast"))
    ap.add_argument("--guard", default="strict", choices=("strict", "ceiling_only"),
                    help="strict: both bands, floor included (the brief's rule); "
                         "ceiling_only: M4.3's second form, floor not checked")
    ap.add_argument("--plants", nargs="*", default=list(PLANTS))
    ap.add_argument("--nodes", nargs="*", default=list(NODES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = load_series(args.steps, args.draws_npz)
    _init(data)
    cfgs = [{"plant": p, "nodes": nd, "policy": pol, "shield": sh, "trigger": trig,
             "barrier_draw": args.barrier_draw, "guard": args.guard}
            for p, nd, pol, (sh, trig) in itertools.product(args.plants, args.nodes,
                                                            POLICIES, SHIELDS)]
    with mp.get_context("fork").Pool(args.workers, initializer=_init,
                                     initargs=(data,)) as pool:
        results = pool.map(run_one, cfgs, chunksize=1)

    floor_cond = {}
    for p, nd in itertools.product(args.plants, args.nodes):
        _pl, model, bands, _u, _p = build(p, nd, data)
        floor_cond[f"{p}/{nd}"] = floor_condition_failures(model, bands, data)

    fixtures = {p: [{k: getattr(unit_spec(p, float(x)), k) for k in ThermiaSpec.OPEN_QUESTIONS}
                    for x in data["draw"].mean(axis=0) * 6.0 *
                    (float(p[len("sizing"):]) if p.startswith("sizing") else 1.0)]
                for p in args.plants}
    out = Path(args.out) if args.out else (REPO / "experiments" / "diagnostics" / "dhw_band"
                                           / f"diagnostic_{args.steps}"
                                             f"{'' if args.barrier_draw == 'known' else '_forecastdraw'}"
                                             f"{'' if args.guard == 'strict' else '_ceilingguard'}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": SCHEMA, "series_source": data["source"], "steps": int(data["draw"].shape[0]),
        "buildings": int(data["draw"].shape[1]),
        "temperatures_chosen_not_sourced": {"t_cold_c": T_COLD, "t_rated_c": T_RATED,
                                            "band_c": [T_FLOOR, T_CEIL],
                                            "disinfection_c": T_LEG},
        "conduction_values_chosen_not_sourced": [0.0, 0.1],
        "bottom_node_band": "ceiling only (chosen): no node above the phase ceiling",
        "source_temperature": "outdoor dry-bulb series as a stand-in (no ground loop modelled)",
        "reserve_quantile_over_whole_run": RESERVE_Q,
        "barrier_draw": args.barrier_draw, "guard": args.guard,
        "completion_rule": ("charged node filled to rated capacity within the step "
                            "(post-charge, before conduction and loss) -- a convention"),
        "unsourced_fixtures": fixtures,
        "claim2_floor_condition_failures_from_worst_in_band_state": floor_cond,
        "runs": results,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [f"{data['draw'].shape[0]} steps x {data['draw'].shape[1]} buildings; every "
             f"Thermia parameter is an UNSOURCED FIXTURE; barrier draw = {args.barrier_draw}",
             f"{'plant':>13}{'nodes':>6}{'policy':>12}{'shield':>10}{'viol':>7}{'refus':>6}"
             f"{'empty':>6}{'unmet':>8}{'events':>7}{'missed':>7}{'elec':>8}{'cost':>8}"
             f"{'starts':>7}{'fwd err':>9}"]
    for r in results:
        sh = "off" if r["shield"] == "off" else r["trigger"]
        lines.append(
            f"{r['plant']:>13}{r['nodes']:>6}{r['policy']:>12}{sh:>10}"
            f"{r['band_violation_building_steps']:>7}{r['refusals']:>6}"
            f"{'-' if r['empty_intersections'] is None else r['empty_intersections']:>6}"
            f"{r['unmet_kwh']:>8.3f}{r['completion_events']:>7}{r['missed_windows']:>7}"
            f"{r['electricity_kwh']:>8.1f}{r['cost']:>8.2f}{r['compressor_starts']:>7}"
            f"{r['model_vs_plant_max_abs_kwh']:>9.1e}")
    lines.append(f"Claim 2 floor-condition failures (building-steps, worst in-band state): "
                 f"{floor_cond}")
    out.with_suffix(".txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"written to {out}")


if __name__ == "__main__":
    main()
