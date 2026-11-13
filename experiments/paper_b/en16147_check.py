#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CERT = {"cop_dhw": 3.06, "standby_w": 34.0, "t_ref_c": 50.9}

_S, _M = 0.105, 3.0
XL = [("07:00", _S, _M), ("07:15", 1.82, 6.0), ("07:26", _S, _M), ("07:45", 4.42, 10.0), ("08:01", _S, _M),
      ("08:15", _S, _M), ("08:30", _S, _M), ("08:45", _S, _M), ("09:00", _S, _M), ("09:30", _S, _M),
      ("10:00", _S, _M), ("10:30", _S, _M), ("11:00", _S, _M), ("11:30", _S, _M), ("11:45", _S, _M),
      ("12:45", 0.735, 4.0), ("14:30", _S, _M), ("15:00", _S, _M), ("15:30", _S, _M), ("16:00", _S, _M),
      ("16:30", _S, _M), ("17:00", _S, _M), ("18:00", _S, _M), ("18:15", _S, _M), ("18:30", _S, _M),
      ("19:00", _S, _M), ("20:30", 0.735, 4.0), ("20:46", 4.42, 10.0), ("21:15", _S, _M), ("21:30", 4.42, 10.0)]
XL_KWH = 19.07


def xl_minutes() -> np.ndarray:
    from stems.dhw_v2.tank import CV
    d = np.zeros(24 * 60)
    for hhmm, e, flow in XL:
        h, m = (int(x) for x in hhmm.split(":"))
        rate = flow * CV * (40.0 - 10.0)
        k = h * 60 + m
        left = e
        while left > 1e-12:
            x = min(left, rate)
            d[k] += x
            left -= x
            k += 1
    return d.reshape(24, 60)


def run(start_c=47.0, stop_c=53.0, sensor_height=0.6, t_supply_c=55.0, *, tapping=True, days=10, skip=2,
        nodes=10, speed_frac=None):
    from stems.dhw_v2 import tank as tk
    from stems.dhw_v2.calibration import DHW_SPEED_FRAC
    from stems.dhw_v2.heatpump import RXT12Map
    from stems.dhw_v2.plant import simulate_hour, tank_arrays
    hp = RXT12Map()
    f = DHW_SPEED_FRAC if speed_frac is None else speed_frac
    spec = tk.rxt12_tank(nodes, sensor_height=sensor_height)
    tank = tank_arrays([spec])
    r = np.array([f * float(hp.q_max(0.0, t_supply_c))])
    p = np.array([float(hp.p_el(r[0], 0.0, t_supply_c))])
    draws = xl_minutes() if tapping else np.zeros((24, 60))
    T = np.full((1, nodes), stop_c)
    on = np.zeros(1, bool)
    comp = np.zeros(1, bool)
    acc = dict(tap=0.0, unmet=0.0, elec_c=0.0, elec_sb=0.0, heat=0.0, loss=0.0, starts=0.0, run_h=0.0, top_e=0.0,
               top_min=np.inf, hours_top_below_40=0.0, tmean=0.0, n=0)
    e_of = lambda X: float(((X - 10.0) * tank["node_vol"][:, None] * tk.CV).sum())
    e0 = None
    for day in range(days):
        if day == skip:
            e0 = e_of(T)
        for h in range(24):
            th = {"on": on, "leg_armed": np.zeros(1, bool), "leg_active": np.zeros(1, bool), "start_c": start_c,
                  "stop_c": stop_c, "t_s_leg": t_supply_c, "r_leg": r, "p_leg": p, "leg_target_c": 99.0}
            res = simulate_hour(T, tank, draws[h][:, None], 1.0 / 60.0, u_d=np.zeros(1), r_d=r, p_d=p,
                                t_s=np.array([t_supply_c]), t_cold=np.array([10.0]), thermo=th, comp_on0=comp,
                                series=True)
            T, on, comp = res["T_end"], res["on_end"], res["comp_on_end"]
            if day >= skip:
                acc["tap"] += float(res["served"][0]); acc["unmet"] += float(res["unmet"][0])
                acc["elec_c"] += float(res["elec_d"][0])
                acc["elec_sb"] += hp.p_standby_kw * max(1.0 - float(res["time_comp"][0]), 0.0)
                acc["heat"] += float(res["heat_d"][0]); acc["loss"] += float(res["loss"][0])
                acc["starts"] += float(res["starts"][0]); acc["run_h"] += float(res["time_comp"][0])
                acc["top_e"] += float((res["series"]["top"][:, 0] * draws[h]).sum())
                acc["top_min"] = min(acc["top_min"], float(res["top_min"][0]))
                acc["tmean"] += float(res["series"]["mean"][:, 0].mean()); acc["n"] += 1
    nd = days - skip
    elec = acc["elec_c"] + acc["elec_sb"]
    out = {"start_c": start_c, "stop_c": stop_c, "sensor_height": sensor_height, "t_supply_c": t_supply_c,
           "q_dhw_kw": float(r[0]), "cop_charge_B0": float(r[0] / p[0]), "days_measured": nd,
           "elec_kwh_per_day": elec / nd, "elec_standby_kwh_per_day": acc["elec_sb"] / nd,
           "heat_kwh_per_day": acc["heat"] / nd, "store_loss_kwh_per_day": acc["loss"] / nd,
           "starts_per_day": acc["starts"] / nd, "compressor_h_per_day": acc["run_h"] / nd,
           "store_mean_c": acc["tmean"] / acc["n"], "store_energy_change_kwh": e_of(T) - e0,
           "mean_power_w": 1000.0 * elec / (24.0 * nd)}
    if tapping:
        out.update(tapped_kwh_per_day=acc["tap"] / nd, unmet_kwh_per_day=acc["unmet"] / nd,
                   cop_dhw=acc["tap"] / elec, top_min_c=acc["top_min"],
                   outlet_mean_c=acc["top_e"] / max(float(draws.sum()) * nd, 1e-12))
    return out


GRID = [(47.0, 53.0, 0.6), (45.0, 53.0, 0.6), (49.0, 53.0, 0.6), (47.0, 51.0, 0.6), (47.0, 55.0, 0.6),
        (47.0, 53.0, 0.3), (47.0, 53.0, 0.5), (47.0, 53.0, 0.8), (44.0, 50.0, 0.5), (50.0, 55.0, 0.8)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    rows = []
    for st, sp, sh in GRID:
        tc = run(st, sp, sh, tapping=True)
        sb = run(st, sp, sh, tapping=False)
        rows.append({"start_c": st, "stop_c": sp, "sensor_height": sh, "cop_dhw": tc["cop_dhw"],
                     "gap_cop": tc["cop_dhw"] - CERT["cop_dhw"], "gap_cop_pct": 100.0 * (tc["cop_dhw"] / CERT["cop_dhw"] - 1.0),
                     "standby_w": sb["mean_power_w"], "gap_standby_w": sb["mean_power_w"] - CERT["standby_w"],
                     "unmet_kwh_per_day": tc["unmet_kwh_per_day"], "top_min_c": tc["top_min_c"],
                     "outlet_mean_c": tc["outlet_mean_c"], "tapping": tc, "standby": sb})
    cop = np.array([r["cop_dhw"] for r in rows])
    sbw = np.array([r["standby_w"] for r in rows])
    res = {"NOT_QUOTABLE": "pre-fix (HW2, HW14); XL profile typed from memory",
           "certified": CERT, "xl_profile_kwh": float(xl_minutes().sum()), "rows": rows,
           "assumed_default_row": rows[0]["cop_dhw"], "cop_range": [float(cop.min()), float(cop.max())],
           "standby_range_w": [float(sbw.min()), float(sbw.max())]}
    print(f"certified: COP_DHW {CERT['cop_dhw']}, standby {CERT['standby_w']} W, reference temperature {CERT['t_ref_c']} degC")
    print("start stop sensor |  COP_DHW   gap    | standby W  gap | unmet kWh/d  top min  outlet mean  starts/d")
    for r in rows:
        print(f"{r['start_c']:5.0f} {r['stop_c']:4.0f} {r['sensor_height']:5.1f}  | {r['cop_dhw']:7.2f} {r['gap_cop_pct']:+6.1f} % |"
              f" {r['standby_w']:7.1f} {r['gap_standby_w']:+6.1f} | {r['unmet_kwh_per_day']:8.2f} {r['top_min_c']:9.1f}"
              f" {r['outlet_mean_c']:10.1f} {r['tapping']['starts_per_day']:9.1f}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
