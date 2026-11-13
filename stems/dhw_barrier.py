from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from stems.dhw_plant import DHWPlant, TankSpec
from stems.heatpump_thermia import ThermiaSpec

__all__ = [
    "PHASE_NORMAL", "PHASE_DISINFECT", "PHASE_NAMES",
    "NODE_OK", "FLOOR_UNREACHABLE", "CEILING_UNAVOIDABLE",
    "StoreStep", "ProjectionResult", "DHWStoreModel", "PhaseBands",
    "HourOfDayDrawForecast", "CyclePhaseBarrier",
]

PHASE_NORMAL = 0
PHASE_DISINFECT = 1
PHASE_NAMES = ("N", "D")

NODE_OK = 0
FLOOR_UNREACHABLE = 1
CEILING_UNAVOIDABLE = 2

_ON_EMPTY = ("floor_first", "ceiling_first")


@dataclass
class StoreStep:

    energy_kwh: np.ndarray
    node_soc: np.ndarray
    pre_loss_node_soc: np.ndarray
    post_charge_sink_soc: np.ndarray
    served_kwh: np.ndarray
    unmet_kwh: np.ndarray
    delivered_kwh: np.ndarray
    accepted_kwh: np.ndarray
    spill_kwh: np.ndarray
    capacity_kw: np.ndarray
    charge_limit_kwh: np.ndarray


@dataclass
class ProjectionResult:

    action: np.ndarray
    requested: np.ndarray
    a_lo: np.ndarray
    a_hi: np.ndarray
    node_lo: np.ndarray
    node_hi: np.ndarray
    status: np.ndarray
    empty: np.ndarray
    conflict: np.ndarray
    on_empty: str
    predicted_node_soc: np.ndarray
    floor_breach: np.ndarray
    ceiling_breach: np.ndarray

    @property
    def modified(self) -> np.ndarray:
        return self.action != self.requested

    @property
    def residual_when_nonempty(self) -> float:
        ok = ~self.empty
        if not ok.any():
            return 0.0
        return float(max(self.floor_breach[ok].max(), self.ceiling_breach[ok].max()))


def _col(x: np.ndarray) -> np.ndarray:
    return np.asarray(x)[:, None]


class DHWStoreModel:

    def __init__(self, tanks: Sequence[TankSpec], units: Sequence[ThermiaSpec], *,
                 allow_element: bool = True) -> None:
        tanks, units = list(tanks), list(units)
        if not tanks or len(tanks) != len(units):
            raise ValueError("need one ThermiaSpec per TankSpec, and at least one store")
        nodes = {int(t.nodes) for t in tanks}
        if len(nodes) != 1:
            raise ValueError(f"every store in one model must have the same number of "
                             f"nodes, got {sorted(nodes)}")
        f = lambda xs: np.array([float(x) for x in xs], dtype=np.float64)
        self.tanks, self.units = tanks, units
        self.B, self.n = len(tanks), nodes.pop()
        self.caps = np.stack([t.node_capacities_kwh for t in tanks]).astype(np.float64)
        self.loss = f(t.loss_coefficient for t in tanks)
        self.kappa = f(t.node_conduction for t in tanks)
        self.sink = np.array([0 if t.charge_node == "top" else t.nodes - 1 for t in tanks],
                             dtype=np.int64)
        self.t_cold = f(t.t_cold_c for t in tanks)
        self.t_rated = f(t.t_rated_c for t in tanks)
        self.cap_ref = f(u.capacity_ref_kw for u in units)
        self.t_sink_ref = f(u.t_sink_ref_c for u in units)
        self.slope_sink = f(u.capacity_slope_per_k_sink for u in units)
        self.t_source_ref = f(u.t_source_ref_c for u in units)
        self.slope_source = f(u.capacity_slope_per_k_source for u in units)
        self.power_max = f(u.power_max_kw for u in units)
        self.t_sink_max = f(u.t_sink_max_c for u in units)
        self.element_kw = f(u.element_kw for u in units)
        self.dt = f(u.dt_hours for u in units)
        self.allow_element = bool(allow_element)
        self.request_scale_kwh = (self.power_max
                                  + (self.element_kw if self.allow_element else 0.0)
                                  ) * self.dt
        self._rows = np.arange(self.B)
        self._sink_caps = self.caps[self._rows, self.sink]
        self._inputs: Optional[Dict[str, np.ndarray]] = None
        self.last_projection: Optional[ProjectionResult] = None

    @classmethod
    def from_plants(cls, plants: Sequence[DHWPlant], *,
                    allow_element: bool = True) -> "DHWStoreModel":
        return cls([p.tank for p in plants], [p.unit.spec for p in plants],
                   allow_element=allow_element)


    def request_kwh(self, action: np.ndarray) -> np.ndarray:
        return np.broadcast_to(np.asarray(action, dtype=np.float64),
                               (self.B,)) * self.request_scale_kwh

    def energy_of(self, plants: Sequence[DHWPlant]) -> np.ndarray:
        return np.stack([np.asarray(p.energy_kwh, dtype=np.float64) for p in plants])

    def _t_source(self, t_source_c) -> np.ndarray:
        return _col(np.broadcast_to(np.asarray(t_source_c, dtype=np.float64),
                                    (self.B,)).astype(np.float64))

    def _capacity_kw(self, t_source: np.ndarray, t_sink: np.ndarray) -> np.ndarray:
        cap = (_col(self.cap_ref) + _col(self.slope_sink) * (t_sink - _col(self.t_sink_ref))
               + _col(self.slope_source) * (t_source - _col(self.t_source_ref)))
        cap = np.clip(cap, 0.0, _col(self.power_max))
        return np.where(t_sink >= _col(self.t_sink_max), 0.0, cap)

    def _after_draw(self, energy_kwh, draw_kwh):
        E = np.array(energy_kwh, dtype=np.float64).reshape(self.B, self.n).copy()
        d = np.maximum(np.broadcast_to(np.asarray(draw_kwh, dtype=np.float64),
                                       (self.B,)).astype(np.float64), 0.0)
        take0 = np.minimum(E[:, 0], d)
        E[:, 0] = E[:, 0] - take0
        remaining = d - take0
        served = take0.copy()
        if self.n == 2:
            go = ~(remaining <= 1e-12)
            take1 = np.where(go, np.minimum(E[:, 1], remaining), 0.0)
            E[:, 1] = E[:, 1] - take1
            served = served + take1
        unmet = np.maximum(d - served, 0.0)
        return E, d, served, unmet

    def _charge(self, E_d: np.ndarray, u: np.ndarray, t_source: np.ndarray) -> Dict:
        r, s, k = self._rows, self.sink, u.shape[1]
        Es = _col(E_d[r, s])
        cs = _col(self._sink_caps)
        sink_c = _col(self.t_cold) + (Es / cs) * _col(self.t_rated - self.t_cold)
        cap_kw = self._capacity_kw(t_source, sink_c)
        request = np.maximum(u * _col(self.request_scale_kwh), 0.0)
        comp_max = cap_kw * _col(self.dt)
        comp = np.minimum(request, comp_max)
        comp = np.where((comp <= 0.0) | (comp_max <= 0.0), 0.0, comp)
        remaining = request - comp
        if self.allow_element:
            ok = (remaining > 1e-12) & _col(self.element_kw > 0.0)
            elem = np.where(ok, np.minimum(remaining, _col(self.element_kw * self.dt)), 0.0)
        else:
            elem = np.zeros_like(remaining)
        delivered = comp + elem
        headroom = cs - Es
        accepted = np.minimum(delivered, np.maximum(headroom, 0.0))
        spill = delivered - accepted
        Ek = np.repeat(E_d[:, None, :], k, axis=1)
        sink_new = Es + accepted
        Ek[r, :, s] = sink_new
        post_charge = sink_new / cs
        if self.n == 2:
            c0, c1 = _col(self.caps[:, 0]), _col(self.caps[:, 1])
            on = _col(self.kappa > 0.0)
            soc_top = Ek[:, :, 0] / c0
            soc_bot = Ek[:, :, 1] / c1
            c_eff = np.where(c0 == c1, c0, 2.0 * c0 * c1 / (c0 + c1))
            moved = _col(self.kappa) * (soc_top - soc_bot) / 2.0 * c_eff
            e0 = Ek[:, :, 0] - moved
            e1 = Ek[:, :, 1] + moved
            Ek[:, :, 0] = np.where(on, e0, Ek[:, :, 0])
            Ek[:, :, 1] = np.where(on, e1, Ek[:, :, 1])
        Ek = np.clip(Ek, 0.0, self.caps[:, None, :])
        pre_loss = Ek
        E_next = Ek * (1.0 - self.loss)[:, None, None]
        limit = np.minimum(comp_max + (_col(self.element_kw * self.dt)
                                       if self.allow_element else 0.0),
                           np.maximum(headroom, 0.0))
        return {"E_next": E_next, "pre_loss": pre_loss, "post_charge": post_charge,
                "delivered": delivered, "accepted": accepted, "spill": spill,
                "cap_kw": cap_kw, "limit": limit}

    def step(self, energy_kwh, action, draw_kwh, t_source_c) -> StoreStep:
        E_d, d, served, unmet = self._after_draw(energy_kwh, draw_kwh)
        u = _col(np.broadcast_to(np.asarray(action, dtype=np.float64), (self.B,)))
        o = self._charge(E_d, u, self._t_source(t_source_c))
        E = o["E_next"][:, 0, :]
        return StoreStep(
            energy_kwh=E, node_soc=E / self.caps,
            pre_loss_node_soc=o["pre_loss"][:, 0, :] / self.caps,
            post_charge_sink_soc=o["post_charge"][:, 0],
            served_kwh=served, unmet_kwh=unmet,
            delivered_kwh=o["delivered"][:, 0], accepted_kwh=o["accepted"][:, 0],
            spill_kwh=o["spill"][:, 0], capacity_kw=o["cap_kw"][:, 0],
            charge_limit_kwh=o["limit"][:, 0])

    def next_node_soc(self, energy_kwh, action, draw_kwh, t_source_c) -> np.ndarray:
        return self.step(energy_kwh, action, draw_kwh, t_source_c).node_soc

    def charge_limit_kwh(self, energy_kwh, draw_kwh, t_source_c) -> np.ndarray:
        return self.step(energy_kwh, np.zeros(self.B), draw_kwh, t_source_c).charge_limit_kwh


    def _bisect(self, g, level: np.ndarray, is_floor: np.ndarray, a_max: np.ndarray,
                iters: int):
        B, k = level.shape
        amax = np.repeat(_col(a_max), k, axis=1)
        zeros = np.zeros((B, k))
        f0, f1 = g(zeros), g(amax)
        floor_col = np.broadcast_to(is_floor, (B, k))
        ok0 = np.where(floor_col, f0 >= level, f1 <= level)
        bad = np.where(floor_col, f1 < level, f0 > level)
        left, right = zeros.copy(), amax.copy()
        active = ~ok0 & ~bad
        for _ in range(iters):
            if not active.any():
                break
            mid = 0.5 * (left + right)
            fm = g(mid)
            pred = np.where(floor_col, fm >= level, fm <= level)
            new_left = np.where(floor_col, np.where(pred, left, mid), np.where(pred, mid, left))
            new_right = np.where(floor_col, np.where(pred, mid, right), np.where(pred, right, mid))
            stalled = (mid == left) | (mid == right)
            left = np.where(active, new_left, left)
            right = np.where(active, new_right, right)
            active = active & ~stalled
        u = np.where(floor_col,
                     np.where(ok0, 0.0, np.where(bad, amax, right)),
                     np.where(ok0, amax, np.where(bad, 0.0, left)))
        return u, ok0, bad

    def node_bounds(self, energy_kwh, lo, hi, draw_kwh, t_source_c, a_max=1.0,
                    iters: int = 64):
        B, n = self.B, self.n
        lo = np.broadcast_to(np.asarray(lo, dtype=np.float64), (B, n))
        hi = np.broadcast_to(np.asarray(hi, dtype=np.float64), (B, n))
        if np.any(lo > hi):
            raise ValueError("a node band has lo > hi")
        amax = np.broadcast_to(np.asarray(a_max, dtype=np.float64), (B,)).astype(np.float64)
        E_d, _d, _s, _u = self._after_draw(energy_kwh, draw_kwh)
        ts = self._t_source(t_source_c)
        node_of = np.tile(np.arange(n), 2)
        sel = np.broadcast_to(node_of[None, :, None], (B, 2 * n, 1))

        def g(u):
            soc = self._charge(E_d, u, ts)["E_next"] / self.caps[:, None, :]
            return np.take_along_axis(soc, sel, axis=2)[:, :, 0]

        level = np.concatenate([lo, hi], axis=1)
        is_floor = np.arange(2 * n) < n
        u, _ok0, bad = self._bisect(g, level, is_floor, amax, iters)
        a_lo, a_hi = u[:, :n], u[:, n:]
        status = np.full((B, n), NODE_OK, dtype=np.int64)
        status[bad[:, :n]] = FLOOR_UNREACHABLE
        status[bad[:, n:]] = CEILING_UNAVOIDABLE
        return a_lo, a_hi, status

    def action_for_charged_level(self, energy_kwh, target, draw_kwh, t_source_c,
                                 a_max=1.0, iters: int = 64):
        amax = np.broadcast_to(np.asarray(a_max, dtype=np.float64), (self.B,)).astype(np.float64)
        E_d, _d, _s, _u = self._after_draw(energy_kwh, draw_kwh)
        ts = self._t_source(t_source_c)
        g = lambda u: self._charge(E_d, u, ts)["post_charge"]
        level = _col(np.broadcast_to(np.asarray(target, dtype=np.float64), (self.B,)))
        u, _ok0, bad = self._bisect(g, level, np.array([True]), amax, iters)
        return u[:, 0], ~bad[:, 0]

    def project(self, energy_kwh, action, lo, hi, draw_kwh, t_source_c, a_max=1.0,
                on_empty: str = "floor_first", iters: int = 64,
                need_interval: bool = True) -> ProjectionResult:
        if on_empty not in _ON_EMPTY:
            raise ValueError(f"on_empty must be one of {_ON_EMPTY}, got {on_empty!r}")
        B, n = self.B, self.n
        amax = np.broadcast_to(np.asarray(a_max, dtype=np.float64), (B,)).astype(np.float64)
        lo_b = np.broadcast_to(np.asarray(lo, dtype=np.float64), (B, n))
        hi_b = np.broadcast_to(np.asarray(hi, dtype=np.float64), (B, n))
        req = np.clip(np.broadcast_to(np.asarray(action, dtype=np.float64), (B,)), 0.0, amax)
        if not need_interval:
            pred = self.next_node_soc(energy_kwh, req, draw_kwh, t_source_c)
            if np.all((pred >= lo_b) & (pred <= hi_b)):
                nan = np.full((B, n), np.nan)
                z = np.zeros((B, n))
                return ProjectionResult(
                    action=req.copy(), requested=req, a_lo=np.full(B, np.nan),
                    a_hi=np.full(B, np.nan), node_lo=nan, node_hi=nan.copy(),
                    status=np.zeros((B, n), dtype=np.int64), empty=np.zeros(B, dtype=bool),
                    conflict=np.zeros(B, dtype=bool), on_empty=on_empty,
                    predicted_node_soc=pred, floor_breach=z, ceiling_breach=z.copy())
        node_lo, node_hi, status = self.node_bounds(energy_kwh, lo, hi, draw_kwh,
                                                    t_source_c, amax, iters)
        a_lo, a_hi = node_lo.max(axis=1), node_hi.min(axis=1)
        alone_bad = (status != NODE_OK).any(axis=1)
        conflict = ~alone_bad & (a_lo > a_hi)
        empty = alone_bad | conflict
        inside = np.clip(req, a_lo, np.maximum(a_hi, a_lo))
        fallback = a_lo if on_empty == "floor_first" else a_hi
        out = np.where(empty, fallback, inside).astype(np.float64)
        pred = self.next_node_soc(energy_kwh, out, draw_kwh, t_source_c)
        return ProjectionResult(
            action=out, requested=req, a_lo=a_lo, a_hi=a_hi, node_lo=node_lo,
            node_hi=node_hi, status=status, empty=empty, conflict=conflict,
            on_empty=on_empty, predicted_node_soc=pred,
            floor_breach=np.maximum(lo_b - pred, 0.0),
            ceiling_breach=np.maximum(pred - hi_b, 0.0))


    def set_step_inputs(self, draw_kwh, t_source_c, energy_kwh=None) -> None:
        self._inputs = {"draw": np.asarray(draw_kwh, dtype=np.float64),
                        "t_source": np.asarray(t_source_c, dtype=np.float64),
                        "energy": (None if energy_kwh is None
                                   else np.asarray(energy_kwh, dtype=np.float64))}

    def safe_interval(self, soc, lo, hi, a_max: float = 1.0, *, energy_kwh=None,
                      draw_kwh=None, t_source_c=None) -> Tuple[np.ndarray, np.ndarray]:
        inp = self._inputs or {}
        draw = draw_kwh if draw_kwh is not None else inp.get("draw")
        t_src = t_source_c if t_source_c is not None else inp.get("t_source")
        if draw is None or t_src is None:
            raise ValueError("DHWStoreModel.safe_interval needs the step's draw and source "
                             "temperature: pass them or call set_step_inputs first")
        energy = energy_kwh if energy_kwh is not None else inp.get("energy")
        if energy is None:
            s = np.asarray(soc, dtype=np.float64).reshape(self.B, -1)
            if s.shape[1] == 1 and self.n > 1:
                s = np.repeat(s, self.n, axis=1)
            energy = s * self.caps
        lo_a, hi_a = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
        if lo_a.ndim < 2:
            lo_bn = np.full((self.B, self.n), -np.inf)
            hi_bn = np.full((self.B, self.n), np.inf)
            lo_bn[:, 0] = np.broadcast_to(lo_a, (self.B,))
            hi_bn[:, 0] = np.broadcast_to(hi_a, (self.B,))
        else:
            lo_bn, hi_bn = lo_a, hi_a
        res = self.project(energy, np.zeros(self.B), lo_bn, hi_bn, draw, t_src, a_max)
        self.last_projection = res
        a_lo = np.where(res.empty, res.action, res.a_lo)
        a_hi = np.where(res.empty, res.action, res.a_hi)
        lo32, hi32 = a_lo.astype(np.float32), a_hi.astype(np.float32)
        up = lo32.astype(np.float64) < a_lo
        lo32[up] = np.nextafter(lo32[up], np.float32(np.inf))
        dn = hi32.astype(np.float64) > a_hi
        hi32[dn] = np.nextafter(hi32[dn], np.float32(-np.inf))
        collapse = lo32 > hi32
        hi32[collapse] = lo32[collapse]
        return lo32, hi32


@dataclass
class PhaseBands:

    normal: np.ndarray
    disinfect: np.ndarray

    def __post_init__(self) -> None:
        self.normal = np.asarray(self.normal, dtype=np.float64).reshape(-1, 2)
        self.disinfect = np.asarray(self.disinfect, dtype=np.float64).reshape(-1, 2)
        if self.normal.shape != self.disinfect.shape:
            raise ValueError("both phases must band the same nodes")
        if np.any(self.normal[:, 0] > self.normal[:, 1]) or np.any(
                self.disinfect[:, 0] > self.disinfect[:, 1]):
            raise ValueError("a band has lo > hi")

    @property
    def n(self) -> int:
        return int(self.normal.shape[0])

    def lo_hi(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d = (np.asarray(q) == PHASE_DISINFECT)[:, None]
        lo = np.where(d, self.disinfect[None, :, 0], self.normal[None, :, 0])
        hi = np.where(d, self.disinfect[None, :, 1], self.normal[None, :, 1])
        return lo, hi

    @classmethod
    def from_temperatures(cls, tank: TankSpec, *, floor_c: float, ceiling_c: float,
                          disinfect_c: float, bottom: str = "ceiling") -> "PhaseBands":
        lo = tank.soc_of_temperature(floor_c)
        hi_n, hi_d = tank.soc_of_temperature(ceiling_c), tank.soc_of_temperature(disinfect_c)
        top_n, top_d = [lo, hi_n], [lo, hi_d]
        if tank.nodes == 1:
            return cls(normal=[top_n], disinfect=[top_d])
        if bottom == "ceiling":
            return cls(normal=[top_n, [-np.inf, hi_n]], disinfect=[top_d, [-np.inf, hi_d]])
        if bottom == "none":
            return cls(normal=[top_n, [-np.inf, np.inf]], disinfect=[top_d, [-np.inf, np.inf]])
        raise ValueError("bottom must be 'ceiling' or 'none'")


class HourOfDayDrawForecast:

    def __init__(self, B: int, period: int = 24, start_index: int = 0) -> None:
        self.B, self.period = int(B), int(period)
        self.sums = np.zeros((self.period, self.B))
        self.counts = np.zeros(self.period)
        self.total = np.zeros(self.B)
        self.n = 0
        self.t = int(start_index)

    def forecast(self, horizon: int) -> np.ndarray:
        horizon = max(int(horizon), 1)
        hrs = (self.t + np.arange(horizon)) % self.period
        cnt = self.counts[hrs][:, None]
        overall = self.total / self.n if self.n else np.zeros(self.B)
        mean = self.sums[hrs] / np.maximum(cnt, 1.0)
        return np.where(cnt > 0, mean, overall[None, :])

    def update(self, draw_kwh) -> None:
        d = np.broadcast_to(np.asarray(draw_kwh, dtype=np.float64), (self.B,))
        h = self.t % self.period
        self.sums[h] += d
        self.counts[h] += 1.0
        self.total += d
        self.n += 1
        self.t += 1


class CyclePhaseBarrier:

    def __init__(self, model: DHWStoreModel, bands: PhaseBands, *, soc_legionella: float,
                 period_steps: float = 168.0, trigger: str = "recourse",
                 forecaster: Optional[HourOfDayDrawForecast] = None,
                 safety_stock_kwh=0.0, jit_reserve_kwh=0.0,
                 t_legionella_c: Optional[float] = None, on_empty: str = "floor_first",
                 guard_includes_floor: bool = True, a_max: float = 1.0,
                 completion_tol: float = 1e-9) -> None:
        if bands.n != model.n:
            raise ValueError(f"bands cover {bands.n} nodes, the model has {model.n}")
        if trigger not in ("jit", "recourse"):
            raise ValueError("trigger must be 'jit' or 'recourse'")
        if trigger == "recourse" and forecaster is None:
            raise ValueError("the recourse trigger re-plans against a draw forecast; pass "
                             "forecaster=")
        if on_empty not in _ON_EMPTY:
            raise ValueError(f"on_empty must be one of {_ON_EMPTY}")
        B = model.B
        self.model, self.bands = model, bands
        self.soc_legionella = float(soc_legionella)
        self.period_steps = float(period_steps)
        self.trigger = trigger
        self.forecaster = forecaster
        self.safety_stock_kwh = np.broadcast_to(
            np.asarray(safety_stock_kwh, dtype=np.float64), (B,)).astype(np.float64)
        self.jit_reserve_kwh = np.broadcast_to(
            np.asarray(jit_reserve_kwh, dtype=np.float64), (B,)).astype(np.float64)
        if t_legionella_c is None:
            t_legionella_c = model.t_cold + self.soc_legionella * (model.t_rated - model.t_cold)
        self.t_legionella_c = np.broadcast_to(np.asarray(t_legionella_c, dtype=np.float64),
                                              (B,)).astype(np.float64)
        self.on_empty = on_empty
        self.guard_includes_floor = bool(guard_includes_floor)
        self.a_max = float(a_max)
        self.tol = float(completion_tol)
        self.reset()

    def reset(self) -> None:
        B = self.model.B
        self.q = np.full(B, PHASE_NORMAL, dtype=np.int64)
        self.k = np.zeros(B)
        self.t = 0
        self.pending_exit = np.zeros(B, dtype=bool)
        self.cycles_completed = np.zeros(B, dtype=np.int64)
        self.missed_windows = np.zeros(B, dtype=np.int64)
        self.transitions = np.zeros(B, dtype=np.int64)
        self.refusals = np.zeros(B, dtype=np.int64)
        self.refusals_above_ceiling = np.zeros(B, dtype=np.int64)
        self.refusals_below_floor = np.zeros(B, dtype=np.int64)
        self.refusal_log: List[Tuple[int, int, str, str, str]] = []
        self.empty_steps = np.zeros(B, dtype=np.int64)
        self.empty_by_reason = {"floor_unreachable": 0, "ceiling_unavoidable": 0,
                                "conflict": 0}
        self.empty_log: List[Tuple[int, int, str]] = []
        self.forced_steps = np.zeros(B, dtype=np.int64)
        self.deadline_clipped_steps = np.zeros(B, dtype=np.int64)
        self.phase_steps = np.zeros((B, 2), dtype=np.int64)
        self.relaxed_after_completion_steps = np.zeros(B, dtype=np.int64)
        self.max_nonempty_residual = 0.0
        self._window_bottom_peak = np.zeros(B)
        self.windows_bottom_reached = np.zeros(B, dtype=np.int64)
        self.window_bottom_peaks: List[np.ndarray] = []


    def feasible(self, node_soc: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lo, hi = self.bands.lo_hi(q)
        s = np.asarray(node_soc, dtype=np.float64).reshape(self.model.B, self.model.n)
        return (s <= hi).all(axis=1), (s >= lo).all(axis=1)

    def request_phase(self, desired: np.ndarray, node_soc: np.ndarray) -> np.ndarray:
        desired = np.broadcast_to(np.asarray(desired, dtype=np.int64), self.q.shape)
        change = desired != self.q
        if not change.any():
            return np.zeros_like(change)
        ceil_from, floor_from = self.feasible(node_soc, self.q)
        ceil_to, floor_to = self.feasible(node_soc, desired)
        ceil_ok = ceil_from & ceil_to
        floor_ok = (floor_from & floor_to) if self.guard_includes_floor else np.ones_like(ceil_ok)
        admissible = ceil_ok & floor_ok
        accept = change & admissible
        refuse = change & ~admissible
        for b in np.where(refuse)[0]:
            reason = "above_ceiling" if not ceil_ok[b] else "below_floor"
            if reason == "above_ceiling":
                self.refusals_above_ceiling[b] += 1
            else:
                self.refusals_below_floor[b] += 1
            self.refusal_log.append((self.t, int(b), PHASE_NAMES[self.q[b]],
                                     PHASE_NAMES[desired[b]], reason))
        self.refusals += refuse.astype(np.int64)
        self.transitions += accept.astype(np.int64)
        self.q = np.where(accept, desired, self.q)
        return accept


    def steps_to_deadline(self) -> np.ndarray:
        return np.maximum(self.period_steps - self.k, 0.0)

    def charged_level(self, energy_kwh: np.ndarray) -> np.ndarray:
        m = self.model
        return np.asarray(energy_kwh, dtype=np.float64)[m._rows, m.sink] / m._sink_caps

    def _jit_steps(self, energy_kwh, t_source_c) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = self.model
        gap = np.maximum(self.soc_legionella - self.charged_level(energy_kwh), 0.0)
        cap = m._capacity_kw(m._t_source(t_source_c), _col(self.t_legionella_c))[:, 0]
        gross = cap * m.dt + (m.element_kw * m.dt if m.allow_element else 0.0)
        rate = np.maximum((gross - self.jit_reserve_kwh) / m._sink_caps
                          - m.loss * self.soc_legionella, 1e-6)
        return np.ceil(gap / rate), gap, rate

    def _recourse_steps(self, energy_kwh, draw_now, t_source_c, forecast: np.ndarray,
                        horizon: np.ndarray) -> np.ndarray:
        m = self.model
        H = int(np.max(horizon)) if np.size(horizon) else 0
        steps = np.full(m.B, np.inf)
        E = np.asarray(energy_kwh, dtype=np.float64).copy()
        full = np.full(m.B, self.a_max)
        for j in range(max(H, 1)):
            d = (np.asarray(draw_now, dtype=np.float64) + self.safety_stock_kwh if j == 0
                 else forecast[min(j, forecast.shape[0] - 1)])
            st = m.step(E, full, d, t_source_c)
            done = (st.post_charge_sink_soc + self.tol >= self.soc_legionella) & ~np.isfinite(steps)
            steps = np.where(done, j + 1, steps)
            E = st.energy_kwh
            if np.all(np.isfinite(steps) | (j + 1 >= horizon)):
                break
        return np.where(np.isfinite(steps), steps, horizon + 1.0)

    def deadline(self, energy_kwh, draw_now, t_source_c,
                 forecast: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        m = self.model
        L = self.steps_to_deadline()
        if self.trigger == "jit":
            steps, gap, rate = self._jit_steps(energy_kwh, t_source_c)
            must = (L - steps <= 0.0) & (gap > 0.0)
            a_min = np.where(must, np.minimum(gap, rate) / rate * self.a_max, 0.0)
        else:
            if forecast is None:
                forecast = self.forecaster.forecast(int(max(np.max(L), 1)))
            steps = self._recourse_steps(energy_kwh, draw_now, t_source_c, forecast, L)
            gap = np.maximum(self.soc_legionella - self.charged_level(energy_kwh), 0.0)
            must = (L <= steps) & (gap > 0.0)
            a_min = np.zeros(m.B)
            if must.any():
                a_need, _reach = m.action_for_charged_level(
                    energy_kwh, self.soc_legionella, draw_now, t_source_c, self.a_max)
                a_min = np.where(must, a_need, 0.0)
        return {"must": must, "a_min": a_min, "steps_needed": steps,
                "steps_to_deadline": L}


    def project(self, u_policy, energy_kwh, draw_kwh, t_source_c,
                a_min: Optional[np.ndarray] = None) -> ProjectionResult:
        m = self.model
        want = np.clip(np.broadcast_to(np.asarray(u_policy, dtype=np.float64), (m.B,)),
                       0.0, self.a_max)
        if a_min is not None:
            want = np.maximum(want, np.asarray(a_min, dtype=np.float64))
            self.forced_steps += (np.asarray(a_min) > 0.0).astype(np.int64)
        lo, hi = self.bands.lo_hi(self.q)
        res = m.project(energy_kwh, want, lo, hi, draw_kwh, t_source_c, self.a_max,
                        self.on_empty, need_interval=False)
        if a_min is not None:
            self.deadline_clipped_steps += ((res.action < np.asarray(a_min)) &
                                            ~res.empty).astype(np.int64)
        self.max_nonempty_residual = max(self.max_nonempty_residual,
                                         res.residual_when_nonempty)
        for b in np.where(res.empty)[0]:
            self.empty_steps[b] += 1
            if res.conflict[b]:
                why = "conflict"
            elif (res.status[b] == FLOOR_UNREACHABLE).any():
                why = "floor_unreachable"
            else:
                why = "ceiling_unavoidable"
            self.empty_by_reason[why] += 1
            self.empty_log.append((self.t, int(b), why))
        return res

    def observe(self, post_charge_sink_soc: np.ndarray,
                pre_loss_node_soc: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        self.phase_steps[self.model._rows, self.q] += 1
        self.relaxed_after_completion_steps += (self.pending_exit
                                                & (self.q == PHASE_DISINFECT)).astype(np.int64)
        if pre_loss_node_soc is not None:
            self._window_bottom_peak = np.maximum(
                self._window_bottom_peak, np.asarray(pre_loss_node_soc)[:, -1])
        done = np.asarray(post_charge_sink_soc, dtype=np.float64) + self.tol >= self.soc_legionella
        self.k = self.k + 1.0
        expired = (~done) & (self.k >= self.period_steps)
        self.cycles_completed += done.astype(np.int64)
        self.missed_windows += expired.astype(np.int64)
        ended = done | expired
        if ended.any():
            peaks = self._window_bottom_peak.copy()
            self.windows_bottom_reached += (ended & (peaks + self.tol >= self.soc_legionella)
                                            ).astype(np.int64)
            self.window_bottom_peaks.append(np.where(ended, peaks, np.nan))
            self._window_bottom_peak = np.where(ended, 0.0, self._window_bottom_peak)
        self.k = np.where(ended, 0.0, self.k)
        self.pending_exit = (self.pending_exit | done) & (self.q == PHASE_DISINFECT)
        self.t += 1
        return {"completed": done, "expired": expired}

    def desired_phase(self, policy_request: Optional[np.ndarray],
                      must: np.ndarray) -> np.ndarray:
        want = self.q.copy() if policy_request is None else np.asarray(
            policy_request, dtype=np.int64).copy()
        want = np.where(self.pending_exit & ~must, PHASE_NORMAL, want)
        return np.where(must, PHASE_DISINFECT, want)

    def report(self) -> Dict[str, Any]:
        peaks = (np.array(self.window_bottom_peaks) if self.window_bottom_peaks
                 else np.zeros((0, self.model.B)))
        return {
            "trigger": self.trigger,
            "cycles_completed": self.cycles_completed.tolist(),
            "missed_windows": self.missed_windows.tolist(),
            "transitions": self.transitions.tolist(),
            "refusals": self.refusals.tolist(),
            "refusals_above_ceiling": self.refusals_above_ceiling.tolist(),
            "refusals_below_floor": self.refusals_below_floor.tolist(),
            "empty_steps": self.empty_steps.tolist(),
            "empty_by_reason": dict(self.empty_by_reason),
            "forced_steps": self.forced_steps.tolist(),
            "deadline_clipped_steps": self.deadline_clipped_steps.tolist(),
            "steps_in_N": self.phase_steps[:, 0].tolist(),
            "steps_in_D": self.phase_steps[:, 1].tolist(),
            "relaxed_ceiling_after_completion_steps":
                self.relaxed_after_completion_steps.tolist(),
            "max_band_residual_on_nonempty_steps": self.max_nonempty_residual,
            "windows_bottom_node_reached_level": self.windows_bottom_reached.tolist(),
            "mean_window_bottom_peak_soc": (float(np.nanmean(peaks)) if peaks.size
                                            and np.isfinite(peaks).any() else None),
            "guard_includes_floor": self.guard_includes_floor,
            "on_empty": self.on_empty,
            "safety_stock_kwh": self.safety_stock_kwh.tolist(),
            "jit_reserve_kwh": self.jit_reserve_kwh.tolist(),
        }
