from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from stems.heatpump_thermia import ThermiaUnit


@dataclass
class TankSpec:

    capacity_kwh: float
    t_cold_c: float
    t_rated_c: float
    loss_coefficient: float
    nodes: int = 1
    top_fraction: float = 0.5
    node_conduction: float = 0.0
    charge_node: str = "top"
    allow_inversion: bool = False
    charging: str = "top_node"
    draw_model: Optional[str] = None
    loss_first: Optional[bool] = None
    t_max_c: Optional[float] = None
    t_use_c: Optional[float] = None
    dt_hours: float = 1.0
    provenance: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.charging not in ("top_node", "stratified"):
            raise ValueError("charging must be 'top_node' or 'stratified'")
        if self.draw_model is None:
            self.draw_model = "plug_flow" if self.charging == "stratified" else "top_first"
        if self.draw_model not in ("top_first", "plug_flow"):
            raise ValueError("draw_model must be 'top_first' or 'plug_flow'")
        if self.loss_first is None:
            self.loss_first = self.charging == "stratified"
        if self.t_max_c is None:
            self.t_max_c = self.t_rated_c
        if self.t_use_c is None:
            self.t_use_c = self.t_cold_c
        if self.t_max_c < self.t_rated_c:
            raise ValueError("t_max_c must be >= t_rated_c")
        if not self.t_cold_c <= self.t_use_c <= self.t_max_c:
            raise ValueError("t_use_c must lie between t_cold_c and t_max_c")
        if self.charging == "stratified" and self.charge_node != "top":
            raise ValueError("stratified charging picks its inlet by buoyancy; leave "
                             "charge_node at its default")
        if self.nodes not in (1, 2):
            raise ValueError(f"nodes must be 1 or 2, got {self.nodes}")
        if self.capacity_kwh <= 0.0:
            raise ValueError("capacity_kwh must be positive")
        if self.t_rated_c <= self.t_cold_c:
            raise ValueError(f"t_rated_c ({self.t_rated_c}) must exceed t_cold_c "
                             f"({self.t_cold_c}); otherwise soc is not a temperature")
        if not 0.0 <= self.loss_coefficient < 1.0:
            raise ValueError("loss_coefficient is a per-step fraction in [0, 1)")
        if not 0.0 < self.top_fraction < 1.0:
            raise ValueError("top_fraction must lie strictly in (0, 1)")
        if not 0.0 <= self.node_conduction <= 1.0:
            raise ValueError("node_conduction is a per-step fraction in [0, 1]")
        if self.charge_node not in ("top", "bottom"):
            raise ValueError("charge_node must be 'top' or 'bottom'")
        if self.nodes == 2 and self.charge_node == "bottom" and not self.allow_inversion:
            raise ValueError(
                "charging the bottom node of a two-node tank can leave warmer water "
                "below cooler water, which this model does not overturn because buoyancy "
                "is not modelled. Pass allow_inversion=True to accept that, or charge the "
                "top node, which is what a high-coil stratified unit does")


    def temperature_of_soc(self, soc: float) -> float:
        return self.t_cold_c + float(soc) * (self.t_rated_c - self.t_cold_c)

    def soc_of_temperature(self, t_c: float) -> float:
        return float(np.clip((t_c - self.t_cold_c) / (self.t_rated_c - self.t_cold_c),
                             0.0, self.soc_max))

    @property
    def soc_max(self) -> float:
        return float((self.t_max_c - self.t_cold_c) / (self.t_rated_c - self.t_cold_c))

    @property
    def soc_use(self) -> float:
        return float((self.t_use_c - self.t_cold_c) / (self.t_rated_c - self.t_cold_c))

    @property
    def node_capacities_kwh(self) -> np.ndarray:
        if self.nodes == 1:
            return np.array([self.capacity_kwh], dtype=np.float64)
        return np.array([self.capacity_kwh * self.top_fraction,
                         self.capacity_kwh * (1.0 - self.top_fraction)],
                        dtype=np.float64)


@dataclass
class PlantStep:

    draw_kwh: float
    served_kwh: float
    unmet_kwh: float
    charge_thermal_kwh: float
    electricity_kwh: float
    spill_kwh: float
    loss_kwh: float
    soc: float
    node_soc: List[float]
    node_temperature_c: List[float]
    top_temperature_c: float
    bottom_temperature_c: float
    cycling: bool
    starts: int
    capacity_kw: float
    cop_used: float
    post_draw_node_soc: Optional[List[float]] = None


class DHWPlant:

    def __init__(self, tank: TankSpec, unit, initial_soc: float = 0.5) -> None:
        self.tank = tank
        self.unit = unit
        caps = tank.node_capacities_kwh
        self.energy_kwh = caps * float(initial_soc)
        self._caps = caps
        self._upper = caps * tank.soc_max
        self._stratified = tank.charging == "stratified"
        if self._stratified:
            if not hasattr(unit, "account"):
                raise TypeError("stratified charging needs an HplibUnit "
                                "(stems.heatpump_hplib): its operating point is the supply "
                                "temperature, which ThermiaUnit does not take")
            from stems.dhw_physics import tank_arrays
            from stems.heatpump_hplib import spec_arrays_of
            self._tank_arr = tank_arrays([tank])
            self._hp_arr = spec_arrays_of([unit.spec])
            self._hp_arr_noel = dict(self._hp_arr, element_kw=np.zeros(1))

    def reset(self, initial_soc: float = 0.5) -> None:
        self.energy_kwh = self._caps * float(initial_soc)
        self.unit.reset()

    @property
    def soc(self) -> float:
        return float(self.energy_kwh.sum() / self._caps.sum())

    @property
    def node_soc(self) -> np.ndarray:
        return self.energy_kwh / self._caps

    def node_temperatures_c(self) -> np.ndarray:
        return np.array([self.tank.temperature_of_soc(s) for s in self.node_soc])

    def top_temperature_c(self) -> float:
        return float(self.node_temperatures_c()[0])

    def bottom_temperature_c(self) -> float:
        return float(self.node_temperatures_c()[-1])


    def _draw_legacy(self, E: np.ndarray, draw: float):
        E = np.array(E, dtype=np.float64)
        if self.tank.draw_model == "plug_flow":
            from stems.dhw_physics import plugflow_draw
            E2, served, unmet = plugflow_draw(E[None, :], self._caps[None, :],
                                              np.array([draw]), np.array([self.tank.soc_use]))
            return E2[0], float(served[0]), float(unmet[0])
        served = 0.0
        remaining = draw
        for i in range(len(E)):
            take = min(E[i], remaining)
            E[i] -= take
            served += float(take)
            remaining -= float(take)
            if remaining <= 1e-12:
                break
        return E, served, max(draw - served, 0.0)

    def _post_draw_energy(self, draw_kwh: float) -> np.ndarray:
        draw = max(float(draw_kwh), 0.0)
        if self._stratified:
            from stems.dhw_physics import loss_and_draw
            E_d, _s, _u, _l = loss_and_draw(self.energy_kwh[None, :], np.array([draw]),
                                            self._tank_arr)
            return E_d[0]
        E = self.energy_kwh.copy()
        if self.tank.loss_first:
            E = E * (1.0 - self.tank.loss_coefficient)
        return self._draw_legacy(E, draw)[0]

    def max_charge_thermal_kwh(self, t_source_c: float, *, draw_kwh: float,
                               allow_element: bool = True,
                               t_supply_c: Optional[float] = None) -> float:
        if self._stratified:
            if t_supply_c is None:
                raise ValueError("stratified charging needs t_supply_c")
            from stems.dhw_physics import request_scale_kwh
            hp = self._hp_arr if allow_element else self._hp_arr_noel
            return float(request_scale_kwh(np.array([t_source_c]), np.array([t_supply_c]),
                                           hp, allow_element)[0])
        E = self._post_draw_energy(draw_kwh)
        node = 0 if self.tank.charge_node == "top" else len(E) - 1
        sink = self.tank.temperature_of_soc(E[node] / self._caps[node])
        return self.unit.max_thermal_kwh(t_source_c, sink, allow_element=allow_element)

    def charge_limit_kwh(self, t_source_c: float, *, draw_kwh: float,
                         allow_element: bool = True,
                         t_supply_c: Optional[float] = None) -> float:
        src = self.max_charge_thermal_kwh(t_source_c, draw_kwh=draw_kwh,
                                          allow_element=allow_element, t_supply_c=t_supply_c)
        E = self._post_draw_energy(draw_kwh)
        if self._stratified:
            from stems.dhw_physics import accept_max
            soc_s = (float(t_supply_c) - self.tank.t_cold_c) / (self.tank.t_rated_c
                                                                 - self.tank.t_cold_c)
            acc = float(accept_max(E[None, :], self._caps[None, :], np.array([soc_s]))[0])
            return float(min(src, acc))
        node = 0 if self.tank.charge_node == "top" else len(E) - 1
        return float(min(src, max(float(self._upper[node] - E[node]), 0.0)))


    def step(self, draw_kwh: float, charge_request_kwh: float, t_source_c: float, *,
             allow_element: bool = True, t_supply_c: Optional[float] = None) -> PlantStep:
        if self._stratified:
            return self._step_stratified(draw_kwh, charge_request_kwh, t_source_c,
                                         allow_element=allow_element, t_supply_c=t_supply_c)
        tank = self.tank
        draw = max(float(draw_kwh), 0.0)
        loss = 0.0
        if tank.loss_first:
            loss = float((self.energy_kwh * tank.loss_coefficient).sum())
            self.energy_kwh = self.energy_kwh * (1.0 - tank.loss_coefficient)

        self.energy_kwh, served, unmet = self._draw_legacy(self.energy_kwh, draw)
        post_draw = [float(x) for x in self.node_soc]

        sink_node = 0 if tank.charge_node == "top" else len(self.energy_kwh) - 1
        sink_c = float(self.node_temperatures_c()[sink_node])
        delivery = self.unit.deliver(max(float(charge_request_kwh), 0.0),
                                     t_source_c, sink_c, allow_element=allow_element)
        headroom = float(self._upper[sink_node] - self.energy_kwh[sink_node])
        accepted = min(delivery.thermal_kwh, max(headroom, 0.0))
        spill = delivery.thermal_kwh - accepted
        self.energy_kwh[sink_node] += accepted

        if len(self.energy_kwh) == 2 and tank.node_conduction > 0.0:
            c0, c1 = float(self._caps[0]), float(self._caps[1])
            c_eff = c0 if c0 == c1 else 2.0 * c0 * c1 / (c0 + c1)
            soc_top, soc_bot = self.node_soc
            moved = tank.node_conduction * (soc_top - soc_bot) / 2.0 * c_eff
            self.energy_kwh[0] -= moved
            self.energy_kwh[1] += moved
        self.energy_kwh = np.clip(self.energy_kwh, 0.0, self._upper)
        if not tank.loss_first:
            loss = float((self.energy_kwh * tank.loss_coefficient).sum())
            self.energy_kwh *= (1.0 - tank.loss_coefficient)

        temps = self.node_temperatures_c()
        return PlantStep(
            draw_kwh=draw, served_kwh=served, unmet_kwh=unmet,
            charge_thermal_kwh=accepted, electricity_kwh=delivery.electricity_kwh,
            spill_kwh=float(spill), loss_kwh=loss, soc=self.soc,
            node_soc=[float(x) for x in self.node_soc],
            node_temperature_c=[float(x) for x in temps],
            top_temperature_c=float(temps[0]), bottom_temperature_c=float(temps[-1]),
            cycling=delivery.cycling, starts=delivery.starts,
            capacity_kw=delivery.capacity_kw, cop_used=delivery.cop_used,
            post_draw_node_soc=post_draw)

    def _step_stratified(self, draw_kwh, charge_request_kwh, t_source_c, *, allow_element,
                         t_supply_c) -> PlantStep:
        if t_supply_c is None:
            raise ValueError("stratified charging needs the supply temperature t_supply_c")
        from stems.dhw_physics import charge_step, loss_and_draw
        draw = max(float(draw_kwh), 0.0)
        E_d, served, unmet, loss = loss_and_draw(self.energy_kwh[None, :], np.array([draw]),
                                                 self._tank_arr)
        hp = self._hp_arr if allow_element else self._hp_arr_noel
        out = charge_step(E_d, np.array([max(float(charge_request_kwh), 0.0)]),
                          np.array([float(t_source_c)]), np.array([float(t_supply_c)]),
                          self._tank_arr, hp)
        on, cyc = bool(out["on"][0]), bool(out["cycling"][0])
        starts = self.unit.account(on, cyc)
        elec = float(out["electricity"][0]) + starts * self.unit.spec.cycling_penalty_kwh
        self.energy_kwh = out["E_next"][0].copy()
        temps = self.node_temperatures_c()
        return PlantStep(
            draw_kwh=draw, served_kwh=float(served[0]), unmet_kwh=float(unmet[0]),
            charge_thermal_kwh=float(out["heat"][0]), electricity_kwh=elec, spill_kwh=0.0,
            loss_kwh=float(loss[0]), soc=self.soc,
            node_soc=[float(x) for x in self.node_soc],
            node_temperature_c=[float(x) for x in temps],
            top_temperature_c=float(temps[0]), bottom_temperature_c=float(temps[-1]),
            cycling=cyc, starts=starts, capacity_kw=float(out["capacity_kw"][0]),
            cop_used=float(out["cop"][0]),
            post_draw_node_soc=[float(x) for x in (E_d[0] / self._caps)])

class DHWPlantFleet:

    def __init__(self, plants: List[DHWPlant]) -> None:
        self.plants = plants
        self.total_unmet_kwh = 0.0
        self.total_electricity_kwh = 0.0
        self.total_spill_kwh = 0.0
        self.steps: List[List[PlantStep]] = []

    @property
    def B(self) -> int:
        return len(self.plants)

    def reset(self, initial_soc: Optional[np.ndarray] = None) -> None:
        for i, p in enumerate(self.plants):
            p.reset(0.5 if initial_soc is None else float(initial_soc[i]))
        self.total_unmet_kwh = self.total_electricity_kwh = self.total_spill_kwh = 0.0
        self.steps = []

    def step(self, draws_kwh, charge_requests_kwh, t_source_c, *,
             allow_element: bool = True, t_supply_c=None) -> List[PlantStep]:
        out = []
        B = len(self.plants)
        t_src = np.broadcast_to(np.asarray(t_source_c, dtype=np.float64), (B,))
        t_sup = (None if t_supply_c is None
                 else np.broadcast_to(np.asarray(t_supply_c, dtype=np.float64), (B,)))
        for i, p in enumerate(self.plants):
            kw = {} if t_sup is None else {"t_supply_c": float(t_sup[i])}
            s = p.step(float(draws_kwh[i]), float(charge_requests_kwh[i]), float(t_src[i]),
                       allow_element=allow_element, **kw)
            self.total_unmet_kwh += s.unmet_kwh
            self.total_electricity_kwh += s.electricity_kwh
            self.total_spill_kwh += s.spill_kwh
            out.append(s)
        self.steps.append(out)
        return out

    def report(self) -> Dict[str, object]:
        return {"buildings": self.B,
                "steps": len(self.steps),
                "unmet_kwh": self.total_unmet_kwh,
                "electricity_kwh": self.total_electricity_kwh,
                "spill_kwh": self.total_spill_kwh,
                "starts": [p.unit.total_starts for p in self.plants],
                "cycling_steps": [p.unit.total_cycling_steps for p in self.plants],
                "nodes": [p.tank.nodes for p in self.plants]}
