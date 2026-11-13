from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np

CV = 4.18 / 3600.0
EPS = 1e-12

__all__ = ["CV", "TankSpec2", "rxt12_tank", "citylearn_tank", "calibrate_ua", "loss",
           "draw", "accept_max", "charge", "buoyancy", "v40_l", "usable_volume_l",
           "sensor_index", "aggregate_nodes"]


@dataclass(frozen=True)
class TankSpec2:
    volume_l: float = 184.0
    nodes: int = 2
    ua_w_k: float = 1.71
    t_room_c: float = 20.0
    t_use_c: float = 40.0
    t_max_c: float = 70.0
    sensor_height: float = 0.6
    loss_model: str = "ua"
    citylearn_loss_per_h: float = 0.0
    name: str = "rxt12_builtin_184l"
    provenance: Dict[str, str] = field(default_factory=dict, compare=False)

    @property
    def node_volume_l(self) -> float:
        return self.volume_l / self.nodes

    @property
    def node_capacity_kwh_k(self) -> float:
        return self.node_volume_l * CV

    def with_nodes(self, n: int) -> "TankSpec2":
        return replace(self, nodes=int(n))


def calibrate_ua(method: str = "pes_minus_psb_cop", *, hp=None, t_store_c: float = 50.9,
                 t_amb_c: float = 20.0) -> Dict[str, float]:
    pes, psb = 34.0, 17.0
    if hp is None:
        from stems.dhw_v2.heatpump import RXT12Map
        hp = RXT12Map()
    qmin = float(hp.q_min(0.0))
    cop = float(hp.cop(qmin, 0.0, 55.0))
    if method == "pes_minus_psb_cop":
        heat = (pes - psb) * cop
    elif method == "pes_cop":
        heat = pes * cop
    elif method == "pes_as_heat":
        heat = pes
    else:
        raise ValueError(method)
    dT = t_store_c - t_amb_c
    return {"ua_w_k": heat / dT, "loss_w": heat, "cop_reheat": cop, "pes_w": pes, "psb_w": psb,
            "t_store_c": t_store_c, "t_amb_c": t_amb_c, "method": method}


def rxt12_tank(nodes: int = 2, calibration: str = "pes_minus_psb_cop", **over) -> TankSpec2:
    cal = calibrate_ua(calibration)
    prov = {"volume_l": "Calibra RXT datasheet 'Water volume 184 l'; KEYMARK EN 16147",
            "ua_w_k": f"calibrate_ua('{calibration}'): {cal['loss_w']:.1f} W at "
                      f"{cal['t_store_c']}-{cal['t_amb_c']} degC",
            "t_use_c": "v1 definition (usable water >= 40 degC)",
            "t_max_c": "WTOL 70 degC; compressor alone reaches 70 (Thermia, 2026-10-19)",
            "sensor_height": "ASSUMPTION: Thermia tank sensor in the upper-middle (0.6 H)"}
    kw = dict(volume_l=184.0, nodes=nodes, ua_w_k=cal["ua_w_k"], provenance=prov)
    kw.update(over)
    return TankSpec2(**kw)


def citylearn_tank(capacity_kwh: float, loss_per_h: float, nodes: int = 2, *,
                   t_cold_c: float = 10.0, t_rated_c: float = 60.0, **over) -> TankSpec2:
    vol = capacity_kwh / (CV * (t_rated_c - t_cold_c))
    prov = {"volume_l": f"CityLearn autosized capacity {capacity_kwh:.3f} kWh over "
                        f"{t_cold_c}-{t_rated_c} degC", "loss": "CityLearn loss_coefficient"}
    kw = dict(volume_l=vol, nodes=nodes, ua_w_k=0.0, loss_model="citylearn",
              citylearn_loss_per_h=float(loss_per_h), name="citylearn_autosized", provenance=prov)
    kw.update(over)
    return TankSpec2(**kw)


def sensor_index(spec: TankSpec2) -> int:
    pos_from_top = 1.0 - float(spec.sensor_height)
    return int(min(spec.nodes - 1, max(0, np.floor(pos_from_top * spec.nodes - 1e-9))))


def _shift(A, x, fill, limit=None):
    A = np.asarray(A, dtype=np.float64)
    N = A.shape[-1]
    bshape = A.shape[:-1]
    x = np.broadcast_to(np.clip(np.asarray(x, dtype=np.float64), 0.0, N), bshape)
    L = np.broadcast_to(np.full(bshape, N) if limit is None else np.asarray(limit), bshape)[..., None]
    fillb = np.broadcast_to(np.asarray(fill, dtype=np.float64), bshape)[..., None]
    m = np.floor(x).astype(np.int64)
    r = (x - m)[..., None]
    idx = np.arange(N)

    def gather(i):
        vals = np.take_along_axis(A, np.minimum(i, N - 1), axis=-1)
        return np.where(i < L, vals, fillb)

    i0 = idx + m[..., None]
    out = (1.0 - r) * gather(i0) + r * gather(i0 + 1)
    return np.where(idx < L, out, A)


def loss(T, spec_ua_w_k, t_room_c, dt_h, node_volume_l, *, model="ua", citylearn_loss_per_h=0.0,
         t_cold_c=10.0):
    T = np.asarray(T, dtype=np.float64)
    N = T.shape[-1]
    Cn = np.asarray(node_volume_l, dtype=np.float64) * CV
    if model == "ua":
        ua_node = np.asarray(spec_ua_w_k, dtype=np.float64) / 1000.0 / N
        k = np.exp(-ua_node * dt_h / Cn)
        tr = np.asarray(t_room_c, dtype=np.float64)
        T2 = tr[..., None] + (T - tr[..., None]) * np.asarray(k)[..., None]
    elif model == "citylearn":
        keep = (1.0 - np.asarray(citylearn_loss_per_h, dtype=np.float64)) ** dt_h
        tc = np.asarray(t_cold_c, dtype=np.float64)[..., None]
        T2 = tc + (T - tc) * np.asarray(keep)[..., None]
    else:
        raise ValueError(model)
    lost = ((T - T2) * np.asarray(Cn)[..., None]).sum(axis=-1)
    return T2, lost


def draw(T, node_volume_l, D_kwh, t_cold_c, t_use_c):
    T = np.asarray(T, dtype=np.float64)
    N = T.shape[-1]
    h = np.asarray(node_volume_l, dtype=np.float64)[..., None]
    tc = np.asarray(t_cold_c, dtype=np.float64)
    D = np.maximum(np.asarray(D_kwh, dtype=np.float64), 0.0)
    ok = np.cumprod(T >= np.asarray(t_use_c, dtype=np.float64)[..., None], axis=-1).astype(bool)
    e = np.where(ok, h * CV * np.maximum(T - tc[..., None], 0.0), 0.0)
    ec = np.cumsum(e, axis=-1)
    avail = ec[..., -1]
    served = np.minimum(D, avail)
    reach = ec >= served[..., None] - 1e-15
    k = np.argmax(reach, axis=-1)
    prev = np.where(k > 0, np.take_along_axis(ec, np.maximum(k - 1, 0)[..., None], -1)[..., 0], 0.0)
    ek = np.take_along_axis(e, k[..., None], -1)[..., 0]
    frac = np.where(ek > EPS, (served - prev) / np.where(ek > EPS, ek, 1.0), 0.0)
    x = np.where(served > EPS, k + np.clip(frac, 0.0, 1.0), 0.0)
    T2 = _shift(T, x, tc)
    return T2, served, D - served, x * h[..., 0]


def accept_max(T, node_volume_l, t_s):
    T = np.asarray(T, dtype=np.float64)
    t_s = np.asarray(t_s, dtype=np.float64)
    h = np.asarray(node_volume_l, dtype=np.float64)[..., None]
    return (h * CV * np.maximum(t_s[..., None] - T, 0.0)).sum(axis=-1)


def charge(T, node_volume_l, Q_kwh, t_s):
    T = np.asarray(T, dtype=np.float64)
    N = T.shape[-1]
    t_s = np.asarray(t_s, dtype=np.float64)
    h = np.asarray(node_volume_l, dtype=np.float64)
    R = T[..., ::-1]
    colder = T < t_s[..., None]
    n_sub = colder.sum(axis=-1)
    g = np.where(R < t_s[..., None], h[..., None] * CV * (t_s[..., None] - R), 0.0)
    g = np.where(np.arange(N) < n_sub[..., None], g, 0.0)
    gc = np.cumsum(g, axis=-1)
    acc = gc[..., -1]
    Q = np.clip(np.asarray(Q_kwh, dtype=np.float64), 0.0, acc)
    reach = gc >= Q[..., None] - 1e-15
    k = np.argmax(reach, axis=-1)
    prev = np.where(k > 0, np.take_along_axis(gc, np.maximum(k - 1, 0)[..., None], -1)[..., 0], 0.0)
    gk = np.take_along_axis(g, k[..., None], -1)[..., 0]
    frac = np.where(gk > EPS, (Q - prev) / np.where(gk > EPS, gk, 1.0), 0.0)
    x = np.where(Q > EPS, k + np.clip(frac, 0.0, 1.0), 0.0)
    R2 = _shift(R, x, t_s, limit=n_sub)
    return R2[..., ::-1].copy(), Q, x * h


def buoyancy(T):
    T = np.array(T, dtype=np.float64, copy=True)
    N = T.shape[-1]
    if N == 1 or not np.any(T[..., :-1] < T[..., 1:] - 1e-12):
        return T
    flat = T.reshape(-1, N)
    for row in range(flat.shape[0]):
        v = flat[row]
        if not np.any(v[:-1] < v[1:] - 1e-12):
            continue
        blocks = []
        for t in v:
            blocks.append([t, 1])
            while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
                s, c = blocks.pop()
                blocks[-1][0] += s
                blocks[-1][1] += c
        flat[row] = np.concatenate([[s / c] * c for s, c in blocks])
    return flat.reshape(T.shape)


def v40_l(T, node_volume_l, t_cold_c, t_use_c=40.0):
    T = np.asarray(T, dtype=np.float64)
    tc = np.asarray(t_cold_c, dtype=np.float64)
    h = np.asarray(node_volume_l, dtype=np.float64)[..., None]
    ok = np.cumprod(T >= t_use_c, axis=-1).astype(bool)
    e = np.where(ok, h * CV * (T - tc[..., None]), 0.0).sum(axis=-1)
    return e / (CV * (40.0 - tc))


def usable_volume_l(T, node_volume_l, t_thr_c=40.0):
    T = np.asarray(T, dtype=np.float64)
    ok = np.cumprod(T >= t_thr_c, axis=-1).astype(bool)
    return ok.sum(axis=-1) * np.asarray(node_volume_l, dtype=np.float64)


def aggregate_nodes(T, n_out: int):
    T = np.asarray(T, dtype=np.float64)
    N = T.shape[-1]
    if N == n_out:
        return T.copy()
    edges_in = np.linspace(0.0, 1.0, N + 1)
    edges_out = np.linspace(0.0, 1.0, n_out + 1)
    W = np.zeros((n_out, N))
    for j in range(n_out):
        lo, hi = edges_out[j], edges_out[j + 1]
        W[j] = np.clip(np.minimum(edges_in[1:], hi) - np.maximum(edges_in[:-1], lo), 0.0, None)
    W /= W.sum(axis=1, keepdims=True)
    return T @ W.T
