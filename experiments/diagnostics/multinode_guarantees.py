from __future__ import annotations

import json
import math
import sys
from typing import Dict, List, Optional

import numpy as np

from experiments.scenario import Scenario
from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig
from stems.environment import STEMSEnvironment
from stems.observations import OBS_NAMES, obs_indices

STOCKS = (("reference", None), ("subset1", 1), ("subset2", 2), ("subset3", 3))
SECTIONS = ("recover", "down", "guard", "tank")


def _env(seed: Optional[int]) -> STEMSEnvironment:
    sc = Scenario(schema="citylearn_schemas/tx_travis_8b/schema.json", season="year",
                  n_buildings=8, subset_seed=seed, grid_cap_kw=2000.0,
                  building_cap_kw=500.0)
    return STEMSEnvironment(schema=sc.schema_path(), seed=0, heat_pump=sc.heat_pump,
                            env_kwargs=sc.env_kwargs("eval"),
                            hvac_control=sc.hvac_control)


def _bounds(soc_rate: np.ndarray, cfg: CBFConfig, saf: SafetyConfig):
    margin = (saf.soc_margin if saf.robust_margins else 0.0) + saf.soc_tolerance
    buf = 0.5 * soc_rate * float(saf.invariance_horizon) if saf.anticipatory else 0.0
    mid = 0.5 * (cfg.SOC_min + cfg.SOC_max)
    return (np.minimum(cfg.SOC_min + margin + buf, mid - 1e-3),
            np.maximum(cfg.SOC_max - margin - buf, mid + 1e-3))


def _roll(bm, i: int, x0: float, target: float, charging: bool, limit: int = 60):
    B = bm.B
    a = np.ones(B) if charging else -np.ones(B)
    x, traj = float(x0), []
    for _ in range(limit):
        x = float(bm.next_soc(np.full(B, x), a)[i])
        traj.append(x)
        if (x >= target) if charging else (x <= target):
            break
    hit = next((k + 1 for k, v in enumerate(traj)
                if (v >= target if charging else v <= target)), None)
    return traj, hit


def section_recover(cfg: CBFConfig, saf: SafetyConfig) -> List[Dict]:
    rows = []
    for name, seed in STOCKS:
        env = _env(seed)
        bm, info = env.battery_model(), env.battery_info()
        names = [b.name for b in env._env.buildings]
        lo, _ = _bounds(info["soc_rate"].astype(float), cfg, saf)
        for i in range(bm.B):
            gain = lambda x: float(bm.next_soc(np.full(bm.B, x), np.ones(bm.B))[i] - x)
            grid = np.linspace(0.0, float(lo[i]), 201)
            g = np.array([gain(float(x)) for x in grid])
            r_min_reported = gain(cfg.SOC_min)
            r_min_enforced = gain(float(lo[i]))
            _, n_rep = _roll(bm, i, 0.0, cfg.SOC_min, True)
            _, n_enf = _roll(bm, i, 0.0, float(lo[i]), True)
            rows.append({
                "stock": name, "index": i, "building": names[i],
                "capacity_kwh": float(bm.capacity[i]),
                "nominal_power_kw": float(bm.nominal_power[i]),
                "efficiency_nameplate": float(info["efficiency"][i]),
                "loss_coefficient": float(bm.loss[i]),
                "soc_rate": float(info["soc_rate"][i]),
                "datasheet_P_eta_over_C": float(bm.nominal_power[i]
                                                * info["efficiency"][i]
                                                / bm.capacity[i]),
                "plant_gain_at_zero": gain(0.0),
                "plant_gain_at_SOC_min": r_min_reported,
                "gain_monotone": bool(np.all(np.diff(g) <= 1e-12)),
                "enforced_lo": float(lo[i]),
                "steps_to_SOC_min_simulated": n_rep,
                "steps_to_SOC_min_bound": math.ceil(cfg.SOC_min / r_min_reported),
                "steps_to_enforced_lo_simulated": n_enf,
                "steps_to_enforced_lo_bound": math.ceil(float(lo[i]) / r_min_enforced),
                "violating_steps_simulated": (n_rep or 1) - 1,
                "violating_steps_bound": math.ceil(cfg.SOC_min / r_min_reported) - 1,
            })
    return rows


def section_down(cfg: CBFConfig, saf: SafetyConfig) -> List[Dict]:
    rows = []
    for name, seed in STOCKS:
        env = _env(seed)
        bm, info = env.battery_model(), env.battery_info()
        _, hi = _bounds(info["soc_rate"].astype(float), cfg, saf)
        for i in range(bm.B):
            drop = lambda x: float(x - bm.next_soc(np.full(bm.B, x),
                                                   -np.ones(bm.B))[i])
            grid = np.linspace(float(hi[i]), 1.0, 201)
            d = np.array([drop(float(x)) for x in grid])
            rho_min = float(d.min())
            _, n_rep = _roll(bm, i, 1.0, cfg.SOC_max, False)
            _, n_enf = _roll(bm, i, 1.0, float(hi[i]), False)
            rows.append({
                "stock": name, "index": i,
                "capacity_kwh": float(bm.capacity[i]),
                "nominal_power_kw": float(bm.nominal_power[i]),
                "enforced_hi": float(hi[i]),
                "drop_at_one": drop(1.0), "drop_at_hi": drop(float(hi[i])),
                "drop_min": rho_min,
                "drop_min_is_at_x0": bool(abs(rho_min - drop(1.0)) < 1e-12),
                "steps_to_SOC_max_simulated": n_rep,
                "steps_to_SOC_max_bound": math.ceil((1.0 - cfg.SOC_max) / rho_min),
                "steps_to_enforced_hi_simulated": n_enf,
                "steps_to_enforced_hi_bound": math.ceil((1.0 - float(hi[i])) / rho_min),
            })
    return rows


def section_guard(cfg: CBFConfig, saf: SafetyConfig) -> List[Dict]:
    i_soc, i_net = obs_indices("electrical_storage_soc", "net_electricity_consumption")
    rows = []
    for name, seed in STOCKS:
        env = _env(seed)
        bm, info = env.battery_model(), env.battery_info()
        states = []
        for _ in range(bm.B):
            s = np.zeros(len(OBS_NAMES), dtype=np.float32)
            s[i_soc] = 0.0
            s[i_net] = 500.0
            states.append(s)
        acts = np.zeros((bm.B, 3), dtype=np.float32)
        acts[:, 1] = -1.0
        for cap_kw, label in ((2000.0, "published"), (1.0, "cap below base load")):
            sh = CBFShield(config=CBFConfig(SOC_min=cfg.SOC_min, SOC_max=cfg.SOC_max,
                                            P_grid_max=cap_kw, P_building_max=cap_kw),
                           num_buildings=bm.B, battery_model=bm,
                           nominal_power=info["nominal_power"].astype(float),
                           action_scale=1.0, elec_idx=1, safety_cfg=saf,
                           enforce_soc=True)
            lo_b, hi_b = sh.enforced_soc_bounds()
            a_lo, _ = bm.safe_interval(np.zeros(bm.B), lo_b, hi_b, a_max=1.0)
            out = sh.project(acts.copy(), states)
            rows.append({"stock": name, "cap_kw": cap_kw, "label": label,
                         "recovery_floor_a_lo": [round(float(v), 6) for v in a_lo],
                         "projected_elec_action": [round(float(v), 6)
                                                   for v in out[:, 1]],
                         "never_cut_below_floor": bool(
                             np.all(out[:, 1] >= np.asarray(a_lo) - 1e-6)),
                         "equals_floor": bool(np.allclose(out[:, 1], a_lo, atol=1e-6)),
                         "devices_at_full_charge": int(np.sum(out[:, 1] >= 1.0 - 1e-6))})
    return rows


def section_tank() -> List[Dict]:
    rows = []
    for name, seed in STOCKS:
        env = _env(seed)
        for b in env._env.buildings:
            C = float(b.dhw_storage.capacity)
            draw = np.asarray(b.energy_simulation.dhw_demand, dtype=float)
            beta = draw / max(C / 2.0, 1e-9)
            served = draw > 1e-9
            rows.append({
                "stock": name, "building": b.name,
                "dhw_capacity_kwh": C,
                "dhw_device_kw": float(b.dhw_device.nominal_power),
                "dhw_loss_coefficient": float(b.dhw_storage.loss_coefficient),
                "draw_hours": int(served.sum()),
                "draw_max_kwh": float(draw.max()),
                "draw_max_over_capacity": float(draw.max() / max(C, 1e-9)),
                "beta_max": float(beta.max()),
                "beta_p99": float(np.percentile(beta, 99)),
                "hours_beta_ge_1": int((beta >= 1.0).sum()),
                "share_of_draw_hours_beta_ge_1": (float((beta[served] >= 1.0).mean())
                                                  if served.any() else 0.0),
            })
    return rows


def main(argv: List[str]) -> int:
    out_path = None
    if "--json" in argv:
        k = argv.index("--json")
        out_path = argv[k + 1]
        argv = argv[:k] + argv[k + 2:]
    want = [a for a in argv if a in SECTIONS] or list(SECTIONS)
    cfg, saf = CBFConfig(), SafetyConfig()
    result: Dict[str, List[Dict]] = {}

    if "recover" in want:
        rows = result["recover"] = section_recover(cfg, saf)
        bad = [r for r in rows
               if r["steps_to_SOC_min_simulated"] > r["steps_to_SOC_min_bound"]
               or r["steps_to_enforced_lo_simulated"] > r["steps_to_enforced_lo_bound"]
               or r["violating_steps_simulated"] != r["violating_steps_bound"]
               or not r["gain_monotone"]]
        print(f"recover: {len(rows)} devices, {len(bad)} outside the bound or "
              f"non-monotone")
        for r in rows:
            print(f"  {r['stock']:>9} i={r['index']} C={r['capacity_kwh']:7.3f} "
                  f"P={r['nominal_power_kw']:6.3f} eta={r['efficiency_nameplate']:.4f} "
                  f"Peta/C={r['datasheet_P_eta_over_C']:.4f} "
                  f"gain(0)={r['plant_gain_at_zero']:.5f} "
                  f"gain(0.1)={r['plant_gain_at_SOC_min']:.5f} "
                  f"N_sim={r['steps_to_SOC_min_simulated']} "
                  f"N_bound={r['steps_to_SOC_min_bound']} "
                  f"viol={r['violating_steps_simulated']}")
        tot = sum(r["violating_steps_simulated"] for r in rows if r["stock"] == "subset2")
        print(f"  subset2 predicted violating building-steps per episode: {tot} "
              f"= {100.0 * tot / 70072:.6f}% of 8 x 8759")

    if "down" in want:
        rows = result["down"] = section_down(cfg, saf)
        bad = [r for r in rows
               if r["steps_to_SOC_max_simulated"] > r["steps_to_SOC_max_bound"]
               or r["steps_to_enforced_hi_simulated"] > r["steps_to_enforced_hi_bound"]]
        print(f"down: {len(rows)} devices, {len(bad)} outside the bound; "
              f"worst rate at x0 on "
              f"{sum(r['drop_min_is_at_x0'] for r in rows)}/{len(rows)}")
        ref = [r for r in rows if r["stock"] == "reference"]
        print("  reference, still above SOC_max after one step: "
              f"{sum(1 for r in ref if r['steps_to_SOC_max_simulated'] > 1)}/{len(ref)}"
              f"; after two: "
              f"{sum(1 for r in ref if r['steps_to_SOC_max_simulated'] > 2)}/{len(ref)}")
        print("  steps to SOC_max, simulated vs bound: "
              f"{sorted({r['steps_to_SOC_max_simulated'] for r in rows})} vs "
              f"{sorted({r['steps_to_SOC_max_bound'] for r in rows})}")

    if "guard" in want:
        rows = result["guard"] = section_guard(cfg, saf)
        print("guard: recovery action against the power guards, base load 500 kW/house")
        for r in rows:
            print(f"  {r['stock']:>9} cap={r['cap_kw']:>7.1f} kW ({r['label']}): "
                  f"never cut below the recovery floor -> {r['never_cut_below_floor']}, "
                  f"equals the floor -> {r['equals_floor']}, "
                  f"{r['devices_at_full_charge']}/8 at full charge")

    if "tank" in want:
        rows = result["tank"] = section_tank()
        print("tank: beta = hourly draw / node energy capacity (C/2)")
        for r in rows:
            print(f"  {r['stock']:>9} C={r['dhw_capacity_kwh']:6.3f} "
                  f"P_hp={r['dhw_device_kw']:6.3f} "
                  f"draw_max={r['draw_max_kwh']:6.3f} "
                  f"draw_max/C={r['draw_max_over_capacity']:.3f} "
                  f"beta_max={r['beta_max']:.3f} beta_p99={r['beta_p99']:.3f} "
                  f"hours_beta>=1={r['hours_beta_ge_1']:4d} "
                  f"({100 * r['share_of_draw_hours_beta_ge_1']:.2f}% of draw hours)")
        print(f"  devices with beta_max >= 1: "
              f"{sum(r['beta_max'] >= 1.0 for r in rows)}/{len(rows)}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
