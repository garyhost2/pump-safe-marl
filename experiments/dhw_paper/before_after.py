#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_matrix import PERIOD, T_COLD, T_RATED, load
from stems.dhw_plant import DHWPlant, TankSpec
from stems.dhw_stratified import ForecastBundle, hold_met
from stems.heatpump_hplib import HplibUnit, brine_temperature_c, calibra_eco_8

VARIANTS = {
    "A committed": dict(charging="top_node", draw_model="top_first", loss_first=False,
                        t_max_c=60.0, t_use_c=10.0),
    "B +loss first, 65 degC store": dict(charging="top_node", draw_model="top_first",
                                         loss_first=True, t_max_c=65.0, t_use_c=10.0),
    "C +plug-flow draw, 40 degC usable": dict(charging="top_node", draw_model="plug_flow",
                                              loss_first=True, t_max_c=65.0, t_use_c=40.0),
    "D stratified charging": dict(charging="stratified", t_max_c=65.0, t_use_c=40.0),
    "E D with all kWh usable": dict(charging="stratified", t_max_c=65.0, t_use_c=10.0),
}


def run(name, kw, data):
    t0 = time.time()
    draw, price = data["draw"], data["price_dynamic"]
    T, B = draw.shape
    span = T_RATED - T_COLD
    fb = ForecastBundle(price, data["t_out"], brine_temperature_c, weather=False, seed=None)
    plants = []
    for i in range(B):
        tank = TankSpec(capacity_kwh=float(data["capacity"][i]), t_cold_c=T_COLD,
                        t_rated_c=T_RATED, loss_coefficient=float(data["loss"][i]), nodes=2,
                        top_fraction=0.5, node_conduction=0.0, **kw)
        plants.append(DHWPlant(tank, HplibUnit(calibra_eco_8()), initial_soc=0.9))
    strat = kw["charging"] == "stratified"
    t_dis = 60.0 if kw["t_max_c"] == 60.0 else 65.0
    due = np.zeros(B, dtype=bool)
    old_def = np.zeros((T // PERIOD, B), dtype=bool)
    new_def = np.zeros((T // PERIOD, B), dtype=bool)
    peak = np.zeros((T // PERIOD, B))
    elec = cost = unmet = heat = loss = served = 0.0
    below = above = 0
    for t in range(T):
        w = t // PERIOD
        if t % PERIOD == 2:
            due[:] = True
        src = fb.source_now(t)
        for i, p in enumerate(plants):
            d = float(draw[t, i])
            target = t_dis if due[i] else 55.0
            if strat:
                req = p.charge_limit_kwh(src, draw_kwh=d, t_supply_c=target)
                s = p.step(d, req, src, t_supply_c=target)
                charged_full = False
            else:
                E = p._post_draw_energy(d)
                req = max(p.tank.soc_of_temperature(target) * p._caps[0] - E[0], 0.0)
                s = p.step(d, req, src)
                charged_full = (E[0] + s.charge_thermal_kwh) / p._caps[0] + 1e-9 >= 1.0
            elec += s.electricity_kwh
            heat += s.charge_thermal_kwh
            loss += s.loss_kwh
            served += s.served_kwh
            cost += s.electricity_kwh * price[t]
            unmet += s.unmet_kwh
            below += s.top_temperature_c < 45.0 - 1e-9
            above += s.top_temperature_c > 55.0 + 1e-9
            pd = np.array(s.post_draw_node_soc)
            if w < old_def.shape[0]:
                if strat:
                    charged_full = s.node_soc[0] + 1e-9 >= 1.0
                old_def[w, i] |= bool(charged_full)
                new_def[w, i] |= bool(hold_met(pd, 1.0))
                peak[w, i] = max(peak[w, i], T_COLD + span * max(s.node_soc[-1], pd[-1]))
            if due[i] and s.top_temperature_c >= 60.0 - 1e-9:
                due[i] = False
    return {"variant": name, **kw, "unmet_kwh": unmet, "electricity_kwh": elec,
            "cost_usd": cost, "starts": int(sum(p.unit.total_starts for p in plants)),
            "heat_kwh": heat, "standing_loss_kwh": loss, "served_kwh": served,
            "seasonal_cop": heat / elec if elec > 0 else float("nan"),
            "stored_change_kwh": sum(float(p.energy_kwh.sum()) - 0.9 * float(p._caps.sum())
                                     for p in plants),
            "comfort_below_steps": int(below), "comfort_above_steps": int(above),
            "weeks_old_definition": int(old_def.sum()),
            "weeks_hold_definition": int(new_def.sum()), "weeks": int(old_def.size),
            "mean_week_bottom_peak_c": float(peak.mean()),
            "max_week_bottom_peak_c": float(peak.max()),
            "seconds": round(time.time() - t0, 1)}


def main():
    data = load()
    out = []
    for name, kw in VARIANTS.items():
        r = run(name, kw, data)
        out.append(r)
        print(f"{name:<36} unmet={r['unmet_kwh']:8.2f} elec={r['electricity_kwh']:8.1f} "
              f"old-def weeks={r['weeks_old_definition']:3d} hold weeks="
              f"{r['weeks_hold_definition']:3d}/{r['weeks']} bottom peak="
              f"{r['mean_week_bottom_peak_c']:5.1f} below={r['comfort_below_steps']} "
              f"t={r['seconds']}s", flush=True)
    path = REPO / "results" / "dhw_paper" / "before_after.json"
    path.write_text(json.dumps({"series": data["source"], "heat_pump": "calibra_eco_8 (hplib)",
                                "runs": out}, indent=2) + "\n")
    print("written", path)


if __name__ == "__main__":
    main()
