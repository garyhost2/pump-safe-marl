from __future__ import annotations

import argparse
import json

import numpy as np

from stems.dhw_plant import DHWPlant, DHWPlantFleet, TankSpec
from stems.heatpump_thermia import ThermiaSpec, ThermiaUnit
from stems.observations import obs_indices

IDX_DHW_DEMAND, IDX_T_OUT = obs_indices("dhw_demand", "outdoor_dry_bulb_temperature")


def draws(env, steps: int) -> np.ndarray:
    obs, _ = env.reset()
    rows = [[float(o[IDX_DHW_DEMAND]) for o in obs]]
    t_out = [float(obs[0][IDX_T_OUT])]
    zero = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    for _ in range(steps - 1):
        obs, _r, term, trunc, _i = env.step(zero)
        rows.append([float(o[IDX_DHW_DEMAND]) for o in obs])
        t_out.append(float(obs[0][IDX_T_OUT]))
        if term or trunc:
            break
    return np.asarray(rows), np.asarray(t_out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=336)
    ap.add_argument("--nodes", type=int, default=1, choices=(1, 2))
    ap.add_argument("--source-multiplier", type=float, default=1.0)
    ap.add_argument("--out", default="experiments/diagnostics/dhw_plant_before_after.json")
    args = ap.parse_args()

    from stems.environment import STEMSEnvironment

    env = STEMSEnvironment(seed=0, heat_pump=True,
                           env_kwargs={"episode_time_steps": [(0, args.steps)]})
    info = env.dhw_info()
    cap = np.asarray(info["capacity"], dtype=np.float64)
    p_nom = np.asarray(info["nominal_power"], dtype=np.float64)
    eff = np.asarray(info["efficiency"], dtype=np.float64)
    loss = np.asarray(info["loss_coefficient"], dtype=np.float64)

    d, t_out = draws(env, args.steps)
    peak = d.max(axis=0)
    mean_daily = d.mean(axis=0) * 24.0
    p_hp = mean_daily / 4.0 * args.source_multiplier

    ratio = p_nom / np.maximum(peak, 1e-9)
    citylearn = {
        "nominal_power_kw": p_nom.tolist(),
        "peak_draw_kwh_per_h": peak.tolist(),
        "nominal_over_peak": ratio.tolist(),
        "unmet_kwh": 0.0,
        "why_zero": ("ElectricHeater.autosize sets nominal_power to max(demand/eff) * "
                     "safety_factor and Building.update_energy_from_dhw_device serves "
                     "every draw in the hour it occurs, so with nominal/peak >= 1 the "
                     "unmet quantity cannot be non-zero under ANY controller"),
        "electricity_kwh": float((d / eff).sum()),
    }

    fleet = DHWPlantFleet([
        DHWPlant(TankSpec(capacity_kwh=float(cap[i]), t_cold_c=10.0, t_rated_c=60.0,
                          loss_coefficient=float(loss[i]), nodes=args.nodes,
                          node_conduction=0.0, charge_node="top",
                          provenance={"capacity_kwh": "schema dhw_storage.capacity",
                                      "loss_coefficient": "schema dhw_storage"}),
                 ThermiaUnit(ThermiaSpec(
                     power_min_kw=max(0.2, float(p_hp[i]) * 0.2),
                     power_max_kw=max(0.5, float(p_hp[i])),
                     eta_ref=0.45, t_sink_ref_c=35.0, eta_slope_per_k=-0.004,
                     capacity_ref_kw=max(0.5, float(p_hp[i])), t_source_ref_c=0.0,
                     capacity_slope_per_k_sink=-0.01, capacity_slope_per_k_source=0.02,
                     t_sink_max_c=65.0, element_kw=0.0, element_efficiency=1.0,
                     cycling_penalty_kwh=0.0)),
                 initial_soc=0.5)
        for i in range(env.num_buildings)])

    soc_target = 0.9
    for t in range(d.shape[0]):
        req = []
        for i, p in enumerate(fleet.plants):
            gap = max(soc_target - p.soc, 0.0) * p.tank.capacity_kwh
            req.append(gap + d[t, i])
        fleet.step(d[t], req, float(t_out[t]), allow_element=False)

    rep = fleet.report()
    print(f"schema draws: {d.shape[0]} steps x {d.shape[1]} buildings, "
          f"total {d.sum():.1f} kWh\n")
    print("CityLearn plant (autosized electric heater)")
    print(f"  nominal power / peak hourly draw : "
          f"{ratio.min():.2f} to {ratio.max():.2f}  (>= 1 in every house)")
    print(f"  unmet hot water                  : 0.000 kWh, BY CONSTRUCTION")
    print(f"  electricity                      : {citylearn['electricity_kwh']:.1f} kWh\n")
    print(f"Power-limited plant (P_hp = mean daily / 4 h x {args.source_multiplier}, "
          f"{args.nodes} node(s), no element)")
    print(f"  source power per house           : "
          f"{p_hp.min():.2f} to {p_hp.max():.2f} kW")
    print(f"  unmet hot water                  : {rep['unmet_kwh']:.3f} kWh")
    print(f"  electricity                      : {rep['electricity_kwh']:.1f} kWh")
    print(f"  spill                            : {rep['spill_kwh']:.3f} kWh")
    print(f"  compressor starts per house      : {rep['starts']}")
    print(f"  steps spent cycling per house    : {rep['cycling_steps']}")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"steps": int(d.shape[0]), "buildings": int(d.shape[1]),
                   "nodes": args.nodes, "source_multiplier": args.source_multiplier,
                   "source_kw": p_hp.tolist(), "citylearn": citylearn,
                   "power_limited": rep,
                   "unsourced": ["eta_ref", "eta_slope_per_k", "t_sink_ref_c",
                                 "capacity_slope_per_k_sink",
                                 "capacity_slope_per_k_source", "t_sink_max_c",
                                 "power_min_kw", "t_cold_c", "t_rated_c"]}, fh, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
