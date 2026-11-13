from __future__ import annotations

import numpy as np

__all__ = ["plugflow_draw", "accept_max", "plugflow_charge", "conduct", "inlet_case"]

INLET_NONE, INLET_TOP, INLET_BOTTOM = 0, 1, 2


def _soc(E, caps):
    return E / caps


def plugflow_draw(E, caps, draw_kwh, soc_use):
    E = np.asarray(E, dtype=np.float64)
    caps = np.broadcast_to(np.asarray(caps, dtype=np.float64), E.shape)
    D = np.maximum(np.asarray(draw_kwh, dtype=np.float64), 0.0)
    use = np.asarray(soc_use, dtype=np.float64)
    n = E.shape[-1]
    C1 = caps[..., 0]
    s1 = E[..., 0] / C1
    ok1 = s1 >= use
    if n == 1:
        cap1 = np.where(ok1, C1 * s1, 0.0)
        served = np.minimum(D, cap1)
        v = np.where(ok1 & (s1 > 0.0), served / np.where(s1 > 0.0, s1, 1.0), 0.0)
        v = np.minimum(v, C1)
        E1 = np.maximum(C1 - v, 0.0) * s1
        return E1[..., None], served, np.maximum(D - served, 0.0)
    C2 = caps[..., 1]
    s2 = E[..., 1] / C2
    ok2 = ok1 & (s2 >= use)
    first = C1 * s1
    second = C2 * s2
    in_first = ok1 & (D <= first)
    in_second = ok2 & ~in_first & (D <= first + second)
    safe = lambda x: np.where(x > 0.0, x, 1.0)
    v = np.where(in_first, D / safe(s1),
                 np.where(in_second, C1 + (D - first) / safe(s2),
                          np.where(ok2, C1 + C2, np.where(ok1, C1, 0.0))))
    served = np.where(in_first | in_second, D,
                      np.where(ok2, first + second, np.where(ok1, first, 0.0)))
    C = C1 + C2
    p = np.maximum(C1 - v, 0.0)
    q = np.maximum(0.0, np.minimum(C - v, C1) - np.maximum(C1 - v, 0.0))
    r = np.maximum(0.0, C2 - v)
    E1 = p * s1 + q * s2
    E2 = r * s2
    return np.stack([E1, E2], axis=-1), served, np.maximum(D - served, 0.0)


def inlet_case(E, caps, soc_s):
    E = np.asarray(E, dtype=np.float64)
    caps = np.broadcast_to(np.asarray(caps, dtype=np.float64), E.shape)
    s_s = np.asarray(soc_s, dtype=np.float64)
    s1 = E[..., 0] / caps[..., 0]
    s_bot = E[..., -1] / caps[..., -1]
    if E.shape[-1] == 1:
        return np.where(s_s > s1, INLET_TOP, INLET_NONE)
    return np.where(s_s >= s1, np.where(s_s > s_bot, INLET_TOP, INLET_NONE),
                    np.where(s_s > s_bot, INLET_BOTTOM, INLET_NONE))


def accept_max(E, caps, soc_s):
    E = np.asarray(E, dtype=np.float64)
    caps = np.broadcast_to(np.asarray(caps, dtype=np.float64), E.shape)
    s_s = np.asarray(soc_s, dtype=np.float64)
    case = inlet_case(E, caps, s_s)
    if E.shape[-1] == 1:
        return np.where(case == INLET_TOP, caps[..., 0] * s_s - E[..., 0], 0.0)
    top = (caps[..., 0] * s_s - E[..., 0]) + (caps[..., 1] * s_s - E[..., 1])
    bot = caps[..., 1] * s_s - E[..., 1]
    out = np.where(case == INLET_TOP, top, np.where(case == INLET_BOTTOM, bot, 0.0))
    return np.maximum(out, 0.0)


def plugflow_charge(E, caps, heat_kwh, soc_s):
    E = np.asarray(E, dtype=np.float64)
    caps = np.broadcast_to(np.asarray(caps, dtype=np.float64), E.shape)
    s_s = np.asarray(soc_s, dtype=np.float64)
    Q = np.maximum(np.asarray(heat_kwh, dtype=np.float64), 0.0)
    case = inlet_case(E, caps, s_s)
    safe = lambda x: np.where(x > 0.0, x, 1.0)
    C1 = caps[..., 0]
    s1 = E[..., 0] / C1
    if E.shape[-1] == 1:
        w = np.minimum(np.where(case == INLET_TOP, Q / safe(s_s - s1), 0.0), C1)
        E1 = np.minimum(w, C1) * s_s + np.maximum(C1 - w, 0.0) * s1
        return np.where(case == INLET_TOP, E1, E[..., 0])[..., None]
    C2 = caps[..., 1]
    s2 = E[..., 1] / C2
    C = C1 + C2
    q_first = C2 * (s_s - s2)
    w_top = np.where(Q <= q_first, Q / safe(s_s - s2),
                     C2 + (Q - q_first) / safe(s_s - s1))
    w_top = np.minimum(w_top, C)
    a = np.minimum(np.maximum(w_top - C1, 0.0), C2)
    b = np.maximum(0.0, np.minimum(w_top + C1, C) - np.maximum(w_top, C1))
    c = np.maximum(0.0, C2 - w_top)
    E1_top = np.minimum(w_top, C1) * s_s + np.maximum(C1 - w_top, 0.0) * s1
    E2_top = a * s_s + b * s1 + c * s2
    w_bot = np.minimum(Q / safe(s_s - s2), C2)
    E2_bot = np.minimum(w_bot, C2) * s_s + np.maximum(C2 - w_bot, 0.0) * s2
    E1 = np.where(case == INLET_TOP, E1_top, E[..., 0])
    E2 = np.where(case == INLET_TOP, E2_top,
                  np.where(case == INLET_BOTTOM, E2_bot, E[..., 1]))
    return np.stack([E1, E2], axis=-1)


def conduct(E, caps, kappa):
    E = np.asarray(E, dtype=np.float64)
    if E.shape[-1] != 2:
        return E
    caps = np.broadcast_to(np.asarray(caps, dtype=np.float64), E.shape)
    k = np.asarray(kappa, dtype=np.float64)
    c0, c1 = caps[..., 0], caps[..., 1]
    c_eff = np.where(c0 == c1, c0, 2.0 * c0 * c1 / (c0 + c1))
    moved = k * (E[..., 0] / c0 - E[..., 1] / c1) / 2.0 * c_eff
    moved = np.where(k > 0.0, moved, 0.0)
    return np.stack([E[..., 0] - moved, E[..., 1] + moved], axis=-1)


def loss_and_draw(E, draw_kwh, tank):
    E = np.asarray(E, dtype=np.float64)
    keep = 1.0 - np.asarray(tank["loss"], dtype=np.float64)
    E_l = E * keep[..., None]
    loss_kwh = (E - E_l).sum(axis=-1)
    E_d, served, unmet = plugflow_draw(E_l, tank["caps"], draw_kwh, tank["soc_use"])
    return E_d, served, unmet, loss_kwh


def charge_step(E_d, request_kwh, t_source_c, t_supply_c, tank, hp):
    from stems.heatpump_hplib import deliver
    caps = tank["caps"]
    soc_s = (np.asarray(t_supply_c, dtype=np.float64) - tank["t_cold"]) / tank["span"]
    acc = accept_max(E_d, caps, soc_s)
    d = deliver(hp, request_kwh, acc, t_source_c, t_supply_c)
    E_c = plugflow_charge(E_d, caps, d.heat_kwh, soc_s)
    E_k = conduct(E_c, caps, tank["kappa"])
    upper = np.asarray(caps, dtype=np.float64) * np.asarray(tank["soc_max"])[..., None]
    E_n = np.minimum(np.maximum(E_k, 0.0), upper)
    return {"E_next": E_n, "heat": d.heat_kwh, "accept": acc, "electricity": d.electricity_kwh,
            "on": d.on, "cycling": d.cycling, "capacity_kw": d.capacity_kw, "cop": d.cop,
            "compressor": d.compressor_kwh, "element": d.element_kwh}


def request_scale_kwh(t_source_c, t_supply_c, hp, allow_element=True):
    from stems.heatpump_hplib import deliver
    big = np.full(np.broadcast(np.asarray(t_source_c), np.asarray(t_supply_c)).shape, np.inf)
    d = deliver(hp, big, big, t_source_c, t_supply_c)
    out = d.capacity_kw * hp["dt"]
    if allow_element:
        out = out + hp["element_kw"] * hp["dt"]
    return out


def tank_arrays(tanks, column=False):
    f = lambda xs: np.array([float(x) for x in xs], dtype=np.float64)
    caps = np.stack([t.node_capacities_kwh for t in tanks]).astype(np.float64)
    out = {"caps": caps, "loss": f(t.loss_coefficient for t in tanks),
           "soc_use": f(t.soc_use for t in tanks), "kappa": f(t.node_conduction for t in tanks),
           "soc_max": f(t.soc_max for t in tanks), "t_cold": f(t.t_cold_c for t in tanks),
           "span": f(t.t_rated_c - t.t_cold_c for t in tanks)}
    if column:
        out = {k: (v[:, None, :] if v.ndim == 2 else v[:, None]) for k, v in out.items()}
    return out
