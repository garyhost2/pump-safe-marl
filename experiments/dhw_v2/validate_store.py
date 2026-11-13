#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from stems.dhw_v2 import tank as tk
from stems.dhw_v2.heatpump import RXT12Map

OUT = Path(__file__).resolve().parent / "smoke_results"


def v40(N, T0, flow=10.0, dt_min=1.0):
    h = 184.0 / N
    T = np.full(N, T0)
    vol = 0.0
    heat = 0.0
    for _ in range(10000):
        if T[0] < 40.0:
            break
        D = flow * dt_min * tk.CV * (T[0] - 10.0)
        T2, served, unmet, v = tk.draw(T, h, D, 10.0, 40.0)
        vol += v
        heat += served
        T = T2
    t_mean = 10.0 + heat / (tk.CV * vol) if vol > 0 else float("nan")
    return {"V_tapped_l": vol, "mean_outlet_c": t_mean, "V40_l": heat / (tk.CV * 30.0)}


def calibrate_t0(N):
    lo, hi = 41.0, 70.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if v40(N, mid)["mean_outlet_c"] < 50.9:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def heat_up(N, stop_c=52.0, t_s=55.0, t_src=0.0, dt_min=1.0):
    hp = RXT12Map()
    spec = tk.rxt12_tank(N)
    h = spec.node_volume_l
    q = float(hp.q_max(t_src, t_s))
    p = float(hp.p_el(q, t_src, t_s))
    T = np.full(N, 10.0)
    sidx = tk.sensor_index(spec)
    t_stop, t_full = None, None
    for k in range(1, 2000):
        T, _ = tk.loss(T, spec.ua_w_k, 20.0, dt_min / 60, h)
        a = tk.accept_max(T, h, t_s)
        T, Q, _ = tk.charge(T, h, min(q * dt_min / 60, a), t_s)
        if t_stop is None and T[sidx] >= stop_c:
            t_stop = k * dt_min / 60
        if t_full is None and T.min() >= t_s - 0.05:
            t_full = k * dt_min / 60
            break
    return {"q_kw": q, "p_el_kw": p, "cop": q / p, "t_stop_h": t_stop, "t_full_h": t_full,
            "implied_mean_kw_if_1h55_to_stop": None}


def main():
    out = {"certified": {"V40_l": 227, "reference_hot_water_c": 50.9, "heat_up_h": 1 + 55 / 60,
                         "standby_w": 34.0, "source": "KEYMARK 012-C700403 EN 16147 XL"}}
    for N in (2, 10, 50):
        t0 = calibrate_t0(N)
        r = v40(N, t0)
        r["T0_c"] = t0
        out[f"V40_N{N}"] = r
        out[f"heat_up_N{N}"] = heat_up(N)
    e = 184.0 * tk.CV * (out["V40_N10"]["T0_c"] - 10.0)
    out["implied_mean_charge_kw_for_certified_heat_up"] = e / out["certified"]["heat_up_h"]
    out["ua"] = {m: tk.calibrate_ua(m) for m in ("pes_minus_psb_cop", "pes_cop", "pes_as_heat")}
    from stems.dhw_v2.calibration import DHW_SPEED_FRAC, calibrate_dhw_speed
    out["dhw_speed_calibration"] = {f"N{N}": calibrate_dhw_speed(N) for N in (10, 20)}
    out["dhw_speed_calibration"]["DHW_SPEED_FRAC_default"] = DHW_SPEED_FRAC
    out["dhw_speed_calibration"]["note"] = ("N = 2 cannot be calibrated: its V40 start state needs "
                                           f"{out['V40_N2']['T0_c']:.1f} degC > 55 degC supply")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "validate_store.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
