from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from stems.environment import thermostat_step
from stems.observations import obs_indices

__all__ = ["StorageMPC", "DEFAULT_HORIZON_H"]

DEFAULT_HORIZON_H = 12

_IDX_SOC_ELEC, _IDX_SOC_DHW, _IDX_DHW_DEMAND = obs_indices(
    "electrical_storage_soc", "dhw_storage_soc", "dhw_demand")
_IDX_T_IN, _IDX_T_COOL, _IDX_T_HEAT = obs_indices(
    "indoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_cooling_set_point",
    "indoor_dry_bulb_temperature_heating_set_point")


class StorageMPC:

    def __init__(
        self,
        num_buildings: int,
        action_dim: int,
        battery,
        tank,
        forecaster,
        *,
        elec_idx: int = 1,
        dhw_idx: int = 0,
        hvac_idx: int = -1,
        hvac_control: str = "setpoint",
        horizon: int = DEFAULT_HORIZON_H,
        soc_min: float = 0.1,
        soc_max: float = 0.9,
        dhw_soc_cap: float = 0.95,
        dhw_action_bound: Optional[np.ndarray] = None,
        p_grid_max: float = 300.0,
        p_building_max: float = 80.0,
        hours_per_step: float = 1.0,
        linearisations: int = 2,
        penalty_scale: float = 1000.0,
        terminal_soc: str = "hold",
        solver: Optional[str] = None,
    ) -> None:
        if terminal_soc not in ("hold", "none"):
            raise ValueError(f"terminal_soc must be 'hold' or 'none', got {terminal_soc!r}")
        if hvac_control not in ("setpoint", "power"):
            raise ValueError(f"hvac_control must be 'setpoint' or 'power', got {hvac_control!r}")
        if battery is None:
            raise ValueError("StorageMPC needs the exact BatteryModel; pass "
                             "STEMSEnvironment.battery_model()")
        self.B = int(num_buildings)
        self.action_dim = int(action_dim)
        self.battery = battery
        self.tank = tank
        self.forecaster = forecaster
        self.elec_idx = int(elec_idx)
        self.dhw_idx = int(dhw_idx) if tank is not None else -1
        self.hvac_idx = int(hvac_idx)
        self.hvac_control = hvac_control
        self.H = max(1, int(horizon))
        self.soc_min, self.soc_max = float(soc_min), float(soc_max)
        self.dhw_soc_cap = float(dhw_soc_cap)
        self.p_grid_max, self.p_building_max = float(p_grid_max), float(p_building_max)
        self.dt = float(hours_per_step)
        self.linearisations = max(1, int(linearisations))
        self.penalty_scale = float(penalty_scale)
        self.terminal_soc = terminal_soc
        self._solver = solver
        self.a_dhw = (np.ones(self.B) if dhw_action_bound is None
                      else np.asarray(dhw_action_bound, dtype=np.float64).reshape(-1))
        self._batt_keep = 1.0 - np.asarray(battery.loss, dtype=np.float64).reshape(-1)
        self._tank_keep = (np.ones(self.B) if tank is None
                           else 1.0 - np.asarray(tank.loss, dtype=np.float64).reshape(-1))
        self._problem = None
        self._coeff0: Dict[str, np.ndarray] = {}
        self.diagnostics: Dict[str, Any] = {}
        self.reset()

    def reset(self) -> None:
        self._u_hvac = np.zeros(self.B, dtype=np.float32)
        self._last_plan: Optional[Dict[str, np.ndarray]] = None
        self._last_obs: Optional[List[np.ndarray]] = None
        if hasattr(self.forecaster, "reset"):
            self.forecaster.reset()

    def notify_executed(self, executed: np.ndarray, hvac_idx: Optional[int] = None) -> None:
        if self._last_obs is not None:
            self.forecaster.note_applied(self._last_obs, np.asarray(executed, dtype=np.float64))
        if self.hvac_control == "power" and self.hvac_idx >= 0:
            self._u_hvac = np.asarray(executed, dtype=np.float32)[:, self.hvac_idx].copy()

    def observe(self, next_obs_list, ev_draw_kwh=None) -> None:
        self.forecaster.observe(next_obs_list)

    def select_action(self, obs_list, history=None, explore: bool = False) -> np.ndarray:
        obs_list = [np.asarray(o, dtype=np.float64) for o in obs_list]
        self._last_obs = obs_list
        forecast = self.forecaster.predict(obs_list, self.H)
        plan = self._solve(obs_list, forecast)
        self._last_plan = plan

        coeff = self._coeff0
        soc_b0 = np.clip(np.array([float(o[_IDX_SOC_ELEC]) for o in obs_list]), 0.0, 1.0)
        idle_b = self.battery.next_soc(soc_b0, np.zeros(self.B))
        target_b = idle_b + coeff["gc_b"][0] * plan["c"][0] - coeff["gd_b"][0] * plan["d"][0]
        actions = np.zeros((self.B, self.action_dim), dtype=np.float32)
        actions[:, self.elec_idx] = np.clip(
            self.battery.action_for_soc(soc_b0, target_b), -1.0, 1.0)
        if self.dhw_idx >= 0 and self.tank is not None:
            soc_t0 = np.clip(np.array([float(o[_IDX_SOC_DHW]) for o in obs_list]), 0.0, 1.0)
            demand0 = np.maximum(np.asarray(forecast.dhw_demand_kwh[0], dtype=np.float64), 0.0)
            idle_t = self.tank.next_soc(soc_t0, np.zeros(self.B), demand0)
            target_t = (idle_t + coeff["gc_t"][0] * plan["h"][0]
                        - coeff["gd_t"][0] * plan["g"][0])
            actions[:, self.dhw_idx] = np.clip(
                self.tank.action_for_soc(soc_t0, target_t, demand0, self.a_dhw), -1.0, 1.0)
        if self.hvac_idx >= 0:
            actions[:, self.hvac_idx] = self._hvac_action(obs_list)
        self.forecaster.note_applied(obs_list, actions.astype(np.float64))
        return actions

    def _hvac_action(self, obs_list) -> np.ndarray:
        if self.hvac_control == "setpoint":
            return np.zeros(self.B, dtype=np.float32)
        col = lambda i: np.array([float(o[i]) for o in obs_list], dtype=np.float32)
        if any(len(o) <= _IDX_T_HEAT for o in obs_list):
            return np.zeros(self.B, dtype=np.float32)
        self._u_hvac = thermostat_step(self._u_hvac, col(_IDX_T_IN), col(_IDX_T_HEAT),
                                       col(_IDX_T_COOL), np.zeros(self.B, dtype=np.float32))
        return self._u_hvac

    def _battery_corners(self, soc: np.ndarray):
        ones = np.ones_like(soc)
        idle = self.battery.next_soc(soc, 0.0 * ones)
        up = self.battery.next_soc(soc, ones)
        down = self.battery.next_soc(soc, -ones)
        e_up = self.battery.accepted_kwh(soc, ones)
        e_down = self.battery.accepted_kwh(soc, -ones)
        return (idle, np.maximum(up - idle, 0.0), np.maximum(idle - down, 0.0),
                np.maximum(e_up, 0.0), np.maximum(-e_down, 0.0))

    def _tank_corners(self, soc: np.ndarray, demand: np.ndarray):
        ones = np.ones_like(soc)
        idle = self.tank.next_soc(soc, 0.0 * ones, demand)
        up = self.tank.next_soc(soc, self.a_dhw, demand)
        e_up = self.tank.drawn_kwh(soc, self.a_dhw, demand)
        ref = np.maximum(soc, self.dhw_soc_cap)
        idle_ref = self.tank.next_soc(ref, 0.0 * ones, demand)
        down_ref = self.tank.next_soc(ref, -ones, demand)
        e_down = self.tank.drawn_kwh(ref, -ones, demand)
        return (idle, np.maximum(up - idle, 0.0), np.maximum(idle_ref - down_ref, 0.0),
                np.maximum(e_up, 0.0), np.maximum(-e_down, 0.0))

    def _coefficients(self, soc_b_traj: np.ndarray, soc_t_traj: np.ndarray,
                      demand: np.ndarray, plan: Optional[Dict[str, np.ndarray]] = None
                      ) -> Dict[str, np.ndarray]:
        H, B = self.H, self.B
        out = {k: np.zeros((H, B)) for k in
               ("off_b", "gc_b", "gd_b", "pc_b", "pd_b",
                "off_t", "gc_t", "gd_t", "pc_t", "pd_t")}
        tol = 1e-4
        for k in range(H):
            idle, gc, gd, pc, pd = self._battery_corners(soc_b_traj[k])
            out["off_b"][k] = idle - self._batt_keep * soc_b_traj[k]
            out["gc_b"][k], out["gd_b"][k] = gc, gd
            out["pc_b"][k], out["pd_b"][k] = pc, pd
            if plan is not None:
                c, d = plan["c"][k], plan["d"][k]
                target = idle + gc * c - gd * d
                e = self.battery.accepted_kwh(
                    soc_b_traj[k], self.battery.action_for_soc(soc_b_traj[k], target))
                out["pc_b"][k] = np.where(c > tol, np.maximum(e, 0.0) / np.maximum(c, tol), pc)
                out["pd_b"][k] = np.where(d > tol, np.maximum(-e, 0.0) / np.maximum(d, tol), pd)
            if self.tank is not None:
                idle, gc, gd, pc, pd = self._tank_corners(soc_t_traj[k], demand[k])
                out["off_t"][k] = idle - self._tank_keep * soc_t_traj[k]
                out["gc_t"][k], out["gd_t"][k] = gc, gd
                out["pc_t"][k], out["pd_t"][k] = pc, pd
                if plan is not None:
                    h, g = plan["h"][k], plan["g"][k]
                    target = idle + gc * h - gd * g
                    a = self.tank.action_for_soc(soc_t_traj[k], target, demand[k], self.a_dhw)
                    e = self.tank.drawn_kwh(soc_t_traj[k], a, demand[k])
                    out["pc_t"][k] = np.where(h > tol, np.maximum(e, 0.0) / np.maximum(h, tol), pc)
                    out["pd_t"][k] = np.where(g > tol, np.maximum(-e, 0.0) / np.maximum(g, tol), pd)
        return out

    def _build(self):
        import cvxpy as cp

        H, B = self.H, self.B
        v = {n: cp.Variable((H, B), nonneg=True) for n in ("c", "d", "h", "g", "imp")}
        v["soc_b"] = cp.Variable((H + 1, B))
        v["soc_t"] = cp.Variable((H + 1, B))
        v["s_soc"] = cp.Variable((H + 1, B), nonneg=True)
        v["s_tank"] = cp.Variable((H + 1, B), nonneg=True)
        v["s_bld"] = cp.Variable((H, B), nonneg=True)
        v["s_grid"] = cp.Variable(H, nonneg=True)
        v["s_term"] = cp.Variable(B, nonneg=True)

        p = {n: cp.Parameter((H, B)) for n in
             ("base", "off_b", "off_t")}
        for n in ("gc_b", "gd_b", "pc_b", "pd_b", "gc_t", "gd_t", "pc_t", "pd_t"):
            p[n] = cp.Parameter((H, B), nonneg=True)
        p["price"] = cp.Parameter(H, nonneg=True)
        p["soc_b0"] = cp.Parameter(B)
        p["soc_t0"] = cp.Parameter(B)
        p["rho"] = cp.Parameter(nonneg=True)
        p["tank_on"] = cp.Parameter(nonneg=True)

        keep_b = np.ascontiguousarray(np.broadcast_to(self._batt_keep, (H, B)))
        keep_t = np.ascontiguousarray(np.broadcast_to(self._tank_keep, (H, B)))
        e_batt = cp.multiply(p["pc_b"], v["c"]) - cp.multiply(p["pd_b"], v["d"])
        e_tank = cp.multiply(p["pc_t"], v["h"]) - cp.multiply(p["pd_t"], v["g"])

        cons = [
            v["soc_b"][0] == p["soc_b0"],
            v["soc_t"][0] == p["soc_t0"],
            v["soc_b"][1:] == cp.multiply(keep_b, v["soc_b"][:-1]) + p["off_b"]
            + cp.multiply(p["gc_b"], v["c"]) - cp.multiply(p["gd_b"], v["d"]),
            v["soc_t"][1:] == cp.multiply(keep_t, v["soc_t"][:-1]) + p["off_t"]
            + cp.multiply(p["gc_t"], v["h"]) - cp.multiply(p["gd_t"], v["g"]),
            v["c"] + v["d"] <= 1.0,
            v["h"] + v["g"] <= 1.0,
            v["soc_b"] >= self.soc_min - v["s_soc"],
            v["soc_b"] <= self.soc_max + v["s_soc"],
            v["soc_t"] >= -v["s_tank"],
            v["soc_t"] <= self.dhw_soc_cap + v["s_tank"],
            v["imp"] >= p["base"] + e_batt + e_tank,
            v["imp"] <= self.p_building_max + v["s_bld"],
            cp.sum(v["imp"], axis=1) <= self.p_grid_max + v["s_grid"],
        ]
        if self.terminal_soc == "hold":
            cons.append(v["soc_b"][H] >= p["soc_b0"] - v["s_term"])
        else:
            cons.append(v["s_term"] == 0.0)
        cons += [v["h"] <= p["tank_on"], v["g"] <= p["tank_on"]]

        energy_cost = (p["price"] @ cp.sum(v["imp"], axis=1)) * self.dt
        slack = (cp.sum(v["s_grid"]) + cp.sum(v["s_bld"]) + cp.sum(v["s_soc"])
                 + cp.sum(v["s_tank"]) + cp.sum(v["s_term"]))
        problem = cp.Problem(cp.Minimize(energy_cost + p["rho"] * slack), cons)
        self._problem = (problem, v, p)
        return self._problem

    def _solve(self, obs_list, forecast) -> Dict[str, np.ndarray]:
        import cvxpy as cp

        H, B = self.H, self.B
        problem, v, p = self._problem or self._build()

        soc_b0 = np.clip(np.array([float(o[_IDX_SOC_ELEC]) for o in obs_list]), 0.0, 1.0)
        soc_t0 = (np.clip(np.array([float(o[_IDX_SOC_DHW]) for o in obs_list]), 0.0, 1.0)
                  if self.tank is not None else np.zeros(B))
        demand = np.asarray(forecast.dhw_demand_kwh, dtype=np.float64)
        base = np.asarray(forecast.base_kw, dtype=np.float64)
        price = np.maximum(np.asarray(forecast.price, dtype=np.float64), 0.0)

        p["base"].value = base
        p["price"].value = price
        p["soc_b0"].value = soc_b0
        p["soc_t0"].value = soc_t0
        p["rho"].value = float(self.penalty_scale * max(price.max(), 1e-3))
        p["tank_on"].value = 0.0 if self.tank is None else 1.0

        soc_b_traj = np.broadcast_to(soc_b0, (H, B)).copy()
        soc_t_traj = np.broadcast_to(soc_t0, (H, B)).copy()
        plan = {k: np.zeros((H, B)) for k in ("c", "d", "h", "g")}
        last_plan = None
        status = "not solved"
        for _ in range(self.linearisations):
            coeffs = self._coefficients(soc_b_traj, soc_t_traj, demand, last_plan)
            for name, value in coeffs.items():
                p[name].value = value
            try:
                problem.solve(solver=self._solver or cp.SCIPY,
                              scipy_options={"method": "highs"})
            except Exception:
                problem.solve()
            status = problem.status
            if v["c"].value is None:
                break
            for key in ("c", "d", "h", "g"):
                plan[key] = np.clip(np.asarray(v[key].value, dtype=np.float64), 0.0, 1.0)
            soc_b_traj = np.asarray(v["soc_b"].value, dtype=np.float64)[:H]
            soc_t_traj = np.asarray(v["soc_t"].value, dtype=np.float64)[:H]
            last_plan = plan
        self._coeff0 = coeffs

        self.diagnostics = {
            "status": status,
            "grid_slack_kwh": (float(np.sum(v["s_grid"].value))
                               if v["s_grid"].value is not None else float("nan")),
            "soc_slack": (float(np.sum(v["s_soc"].value))
                          if v["s_soc"].value is not None else float("nan")),
            "objective": (float(problem.value) if problem.value is not None
                          else float("nan")),
            "linearisations": self.linearisations,
        }
        return plan

    def planned_soc(self) -> Optional[np.ndarray]:
        if self._problem is None:
            return None
        value = self._problem[1]["soc_b"].value
        return None if value is None else np.asarray(value, dtype=np.float64)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        return {}

    def save(self, path: str) -> None:
        return None

    def load(self, path: str) -> None:
        return None
