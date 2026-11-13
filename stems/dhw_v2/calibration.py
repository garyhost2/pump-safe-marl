from __future__ import annotations

from typing import Dict

import numpy as np

from stems.dhw_v2 import tank as tk

__all__ = ["V40_CERTIFIED_L", "HEAT_UP_CERTIFIED_H", "v40", "v40_start_temperature",
           "heat_up_hours", "calibrate_dhw_speed", "DHW_SPEED_FRAC"]

V40_CERTIFIED_L = 227.0
T_REF_CERTIFIED_C = 50.9
HEAT_UP_CERTIFIED_H = 1.0 + 55.0 / 60.0

DHW_SPEED_FRAC = 0.4579


def v40(N: int, T0: float, flow_l_min: float = 10.0, dt_min: float = 1.0, volume_l: float = 184.0) -> Dict[str, float]:
    h = volume_l / N
    T = np.full(N, float(T0))
    vol = heat = 0.0
    for _ in range(100000):
        if T[0] < 40.0:
            break
        D = flow_l_min * dt_min * tk.CV * (T[0] - 10.0)
        T, served, _, v = tk.draw(T, h, D, 10.0, 40.0)
        vol += float(v)
        heat += float(served)
    return {"V_tapped_l": vol, "mean_outlet_c": 10.0 + heat / (tk.CV * vol) if vol > 0 else float("nan"),
            "V40_l": heat / (tk.CV * 30.0)}


def v40_start_temperature(N: int) -> float:
    lo, hi = 41.0, 70.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if v40(N, mid)["mean_outlet_c"] < T_REF_CERTIFIED_C:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def heat_up_hours(frac: float, N: int = 10, *, t_s: float = 55.0, t_src: float = 0.0,
                  target_kwh: float = None, dt_min: float = 1.0, hp=None) -> float:
    if hp is None:
        from stems.dhw_v2.heatpump import RXT12Map
        hp = RXT12Map()
    spec = tk.rxt12_tank(N)
    h = spec.node_volume_l
    q = frac * float(hp.q_max(t_src, t_s))
    T = np.full(N, 10.0)
    for k in range(1, 100000):
        T, _ = tk.loss(T, spec.ua_w_k, spec.t_room_c, dt_min / 60.0, h)
        a = float(tk.accept_max(T, h, t_s))
        T, _, _ = tk.charge(T, h, min(q * dt_min / 60.0, a), t_s)
        e = float(((T - 10.0) * h * tk.CV).sum())
        if e >= target_kwh:
            return k * dt_min / 60.0
        if a <= 1e-12:
            return float("inf")
    return float("inf")


def calibrate_dhw_speed(N: int = 10, heat_up_h: float = HEAT_UP_CERTIFIED_H) -> Dict[str, float]:
    T0 = v40_start_temperature(N)
    target = 184.0 * tk.CV * (T0 - 10.0)
    lo, hi = 0.05, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if heat_up_hours(mid, N, target_kwh=target, dt_min=0.5) > heat_up_h:
            lo = mid
        else:
            hi = mid
    f = 0.5 * (lo + hi)
    from stems.dhw_v2.heatpump import RXT12Map
    hp = RXT12Map()
    q = f * float(hp.q_max(0.0, 55.0))
    return {"dhw_speed_frac": f, "T0_c": T0, "target_kwh": target, "q_kw_B0W55": q,
            "cop_B0W55": float(hp.cop(q, 0.0, 55.0)), "full_speed_heat_up_h": heat_up_hours(1.0, N, target_kwh=target)}
