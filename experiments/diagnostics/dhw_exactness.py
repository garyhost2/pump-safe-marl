#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.diagnostics.dhw_band_diagnostic import build, load_series
from stems.dhw_barrier import DHWStoreModel
from stems.dhw_plant import DHWPlant, DHWPlantFleet, TankSpec
from stems.heatpump_thermia import ThermiaSpec, ThermiaUnit


def _unit(pmax: float, elem: float) -> ThermiaSpec:
    return ThermiaSpec(power_min_kw=0.3 * pmax, power_max_kw=pmax, eta_ref=0.45,
                       t_sink_ref_c=35.0, eta_slope_per_k=-0.004, capacity_ref_kw=pmax,
                       t_source_ref_c=0.0, capacity_slope_per_k_sink=-0.05,
                       capacity_slope_per_k_source=0.03, t_sink_max_c=58.0,
                       element_kw=elem, element_efficiency=1.0, cycling_penalty_kwh=0.0)


def section_random(seed: int = 0, steps: int = 400, B: int = 6) -> dict:
    rng = np.random.default_rng(seed)
    worst_fwd, worst_inv, binding, wrong_side, total, configs = 0.0, 0.0, 0, 0, 0, 0
    for nodes, node, kappa, elem, allow in itertools.product((1, 2), ("top", "bottom"),
                                                             (0.0, 0.2), (0.0, 1.5),
                                                             (True, False)):
        if nodes == 1 and (node == "bottom" or kappa > 0):
            continue
        configs += 1
        caps = rng.uniform(4, 12, B)
        loss = rng.uniform(0.001, 0.009, B)
        pmax = rng.uniform(0.3, 4, B)
        tanks = [TankSpec(capacity_kwh=float(caps[i]), t_cold_c=10.0, t_rated_c=60.0,
                          loss_coefficient=float(loss[i]), nodes=nodes,
                          top_fraction=0.5 if i % 2 else 0.4, node_conduction=kappa,
                          charge_node=node, allow_inversion=True) for i in range(B)]
        units = [_unit(float(pmax[i]), elem) for i in range(B)]
        plants = [DHWPlant(tanks[i], ThermiaUnit(units[i]),
                           initial_soc=float(rng.uniform(0.3, 1.0))) for i in range(B)]
        m = DHWStoreModel(tanks, units, allow_element=allow)
        lo = np.full((B, nodes), 0.70)
        hi = np.full((B, nodes), 0.90)
        if nodes == 2:
            lo[:, 1] = -np.inf
        for _ in range(steps):
            E = m.energy_of(plants)
            draw = np.where(rng.random(B) < 0.4, rng.exponential(1.5, B), 0.0)
            t_src = float(rng.uniform(-5, 35))
            u = rng.uniform(0, 1, B)
            a_lo, a_hi, status = m.node_bounds(E, lo, hi, draw, t_src)
            pred = m.step(E, u, draw, t_src)
            for cand, side in ((a_lo, "lo"), (a_hi, "hi")):
                for j in range(nodes):
                    req = m.request_kwh(cand[:, j])
                    for i in range(B):
                        bound = lo[i, j] if side == "lo" else hi[i, j]
                        if status[i, j] != 0 or not np.isfinite(bound):
                            continue
                        if not ((cand[i, j] > 0) if side == "lo" else (cand[i, j] < 1)):
                            continue
                        q = DHWPlant(tanks[i], ThermiaUnit(units[i]))
                        q.energy_kwh = E[i].copy()
                        soc = q.step(draw[i], req[i], t_src, allow_element=allow).node_soc[j]
                        binding += 1
                        worst_inv = max(worst_inv, abs(soc - bound))
                        wrong_side += int((soc < bound) if side == "lo" else (soc > bound))
            for i in range(B):
                plants[i].step(draw[i], m.request_kwh(u)[i], t_src, allow_element=allow)
            worst_fwd = max(worst_fwd, float(np.abs(m.energy_of(plants) - pred.energy_kwh).max()))
            total += B
    return {"configurations": configs, "building_steps": total,
            "max_forward_abs_error_kwh": worst_fwd, "binding_endpoints": binding,
            "max_endpoint_abs_error_soc": worst_inv, "endpoints_on_wrong_side": wrong_side}


def section_limit(data: dict) -> list:
    out = []
    for plant in ("sizing0.25", "sizing1", "thermia_1_12"):
        plants, model, _bands, _u, _p = build(plant, "1", data)
        fleet = DHWPlantFleet(plants)
        fleet.reset(np.full(model.B, 0.8))
        differ = higher = n = 0
        worst = 0.0
        for t in range(data["draw"].shape[0]):
            E = model.energy_of(plants)
            d, ts = data["draw"][t], float(data["t_out"][t])
            pre = np.array([p.max_charge_thermal_kwh(ts, allow_element=False) for p in plants])
            post = model.step(E, np.zeros(model.B), d, ts).capacity_kw * model.dt
            diff = post - pre
            differ += int(np.sum(np.abs(diff) > 1e-12))
            higher += int(np.sum(diff > 1e-12))
            worst = max(worst, float(np.abs(diff).max()))
            n += model.B
            E_d, *_ = model._after_draw(E, d)
            need = np.maximum(0.8 * model._sink_caps - E_d[model._rows, model.sink], 0.0)
            fleet.step(d, model.request_kwh(np.clip(need / model.request_scale_kwh, 0, 1)),
                       ts, allow_element=False)
        out.append({"plant": plant, "nodes": 1, "policy": "thermostat 50 degC",
                    "building_steps": n, "pre_draw_limit_differs": differ,
                    "applied_limit_higher": higher, "max_abs_difference_kwh": worst})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws-npz", default=None)
    ap.add_argument("--out", default=str(REPO / "experiments" / "diagnostics" / "dhw_band"
                                         / "exactness.json"))
    args = ap.parse_args()
    rnd = section_random()
    print(json.dumps(rnd))
    data = load_series(8759, args.draws_npz)
    lim = section_limit(data)
    for r in lim:
        print(json.dumps(r))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"random": rnd, "limit": lim,
                                          "series_source": data["source"]}, indent=2) + "\n")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
