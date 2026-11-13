from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.observations import obs_index, obs_indices

__all__ = [
    "RCThermalModel",
    "ThermalComfortBarrier",
    "build_comfort_barriers",
    "SECONDS_PER_HOUR",
    "SPAN_K",
    "T_ZERO_C",
    "T_HIGH_C",
]

IDX_T_IN, IDX_T_OUT, IDX_T_COOL, IDX_OCCUPANT = obs_indices(
    "indoor_dry_bulb_temperature", "outdoor_dry_bulb_temperature",
    "indoor_dry_bulb_temperature_cooling_set_point", "occupant_count")
IDX_T_HEAT = obs_index("indoor_dry_bulb_temperature_heating_set_point")

SECONDS_PER_HOUR = 3600.0

SPAN_K = 40.0

T_ZERO_C = 0.0
T_HIGH_C = T_ZERO_C + SPAN_K

_MIN_CAPACITANCE_J_PER_K = 1.0e5
_MIN_UA_W_PER_K = 1.0


@dataclass(frozen=True)
class RCThermalFit:

    r2: np.ndarray
    rmse_k: np.ndarray
    samples: np.ndarray
    tau_h: np.ndarray

    def summary(self) -> Dict[str, float]:
        return {"r2_min": float(np.min(self.r2)), "r2_median": float(np.median(self.r2)),
                "rmse_k_max": float(np.max(self.rmse_k)),
                "tau_h_min": float(np.min(self.tau_h)),
                "tau_h_max": float(np.max(self.tau_h)),
                "samples_min": float(np.min(self.samples))}


class RCThermalModel:

    def __init__(self, capacitance_j_per_k, ua_w_per_k, gain_w,
                 dt_s: float = SECONDS_PER_HOUR, provenance: str = "unspecified") -> None:
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        self.C = f(capacitance_j_per_k)
        self.UA = f(ua_w_per_k)
        self.Phi_g = f(gain_w)
        if not (self.C.shape == self.UA.shape == self.Phi_g.shape):
            raise ValueError("C, UA and Phi_g must have one entry per building")
        if np.any(self.C < _MIN_CAPACITANCE_J_PER_K):
            raise ValueError(
                f"thermal capacitance below {_MIN_CAPACITANCE_J_PER_K:g} J/K is not a "
                f"building envelope: got {self.C.min():g} J/K")
        if np.any(self.UA < _MIN_UA_W_PER_K):
            raise ValueError(
                f"envelope UA below {_MIN_UA_W_PER_K:g} W/K is not a building: "
                f"got {self.UA.min():g} W/K")
        self.dt_s = float(dt_s)
        self.provenance = str(provenance)
        self.B = int(self.C.size)

    @property
    def tau_h(self) -> np.ndarray:
        return self.C / self.UA / SECONDS_PER_HOUR

    def drift_k(self, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
        t_in = np.asarray(t_in, dtype=np.float64).reshape(-1)
        t_out = np.asarray(t_out, dtype=np.float64).reshape(-1)
        return (self.dt_s / self.C) * (self.UA * (t_out - t_in) + self.Phi_g)

    def temperature_gain_k(self, thermal_power_w: np.ndarray) -> np.ndarray:
        phi = np.asarray(thermal_power_w, dtype=np.float64).reshape(-1)
        return (self.dt_s / self.C) * phi

    def next_temperature_c(self, t_in, t_out, thermal_power_w) -> np.ndarray:
        t_in = np.asarray(t_in, dtype=np.float64).reshape(-1)
        return t_in + self.drift_k(t_in, t_out) + self.temperature_gain_k(thermal_power_w)

    def describe(self) -> Dict[str, object]:
        return {"capacitance_j_per_k": self.C.tolist(), "ua_w_per_k": self.UA.tolist(),
                "gain_w": self.Phi_g.tolist(), "dt_s": self.dt_s,
                "tau_h": self.tau_h.tolist(), "provenance": self.provenance}

    @classmethod
    def identify(cls, t_in: np.ndarray, t_out: np.ndarray, thermal_power_w: np.ndarray,
                 dt_s: float = SECONDS_PER_HOUR, provenance: str = "least squares",
                 ) -> Tuple["RCThermalModel", RCThermalFit]:
        t_in = np.asarray(t_in, dtype=np.float64)
        t_out = np.asarray(t_out, dtype=np.float64)
        phi = np.asarray(thermal_power_w, dtype=np.float64)
        if t_in.ndim != 2 or t_in.shape != t_out.shape or t_in.shape != phi.shape:
            raise ValueError(f"t_in, t_out and thermal_power_w must all be (T, B); got "
                             f"{t_in.shape}, {t_out.shape}, {phi.shape}")
        T, B = t_in.shape
        if T < 24:
            raise ValueError(f"need at least 24 steps to identify an envelope, got {T}")

        C = np.empty(B)
        UA = np.empty(B)
        Phi_g = np.empty(B)
        r2 = np.empty(B)
        rmse = np.empty(B)
        n = np.empty(B)
        for b in range(B):
            dT = np.diff(t_in[:, b])
            X = np.column_stack([(t_out[:-1, b] - t_in[:-1, b]), phi[:-1, b],
                                 np.ones(T - 1)])
            ok = np.isfinite(dT) & np.isfinite(X).all(axis=1)
            if ok.sum() < 24:
                raise ValueError(f"building {b}: only {int(ok.sum())} usable samples")
            coef, *_ = np.linalg.lstsq(X[ok], dT[ok], rcond=None)
            a, bb, c = (float(v) for v in coef)
            if bb <= 0.0 or a <= 0.0:
                raise ValueError(
                    f"building {b}: the 1R1C fit is not physical (a={a:.3e} K/K per "
                    f"step, b={bb:.3e} K/W per step). A positive a and b are required "
                    "for C and UA to be positive; this dataset cannot support a hard "
                    "thermal guarantee.")
            C[b] = dt_s / bb
            UA[b] = a / bb
            Phi_g[b] = c / bb
            resid = dT[ok] - X[ok] @ coef
            ss_res = float(resid @ resid)
            ss_tot = float(((dT[ok] - dT[ok].mean()) ** 2).sum())
            r2[b] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            rmse[b] = float(np.sqrt(ss_res / ok.sum()))
            n[b] = float(ok.sum())
        model = cls(C, UA, Phi_g, dt_s=dt_s, provenance=provenance)
        return model, RCThermalFit(r2=r2, rmse_k=rmse, samples=n, tau_h=model.tau_h)


class ThermalComfortBarrier(DeadlineStorageBarrier):

    def __init__(self, rc: RCThermalModel, direction: int, action_index: int,
                 thermal_power_w: np.ndarray, cop: np.ndarray,
                 action_bound: np.ndarray, tolerance_k: float = 2.0,
                 occupied_only: bool = True, name: str = "comfort") -> None:
        if direction not in (1, -1):
            raise ValueError(f"direction must be +1 (heating) or -1 (cooling), got {direction!r}")
        f = lambda x: np.asarray(x, dtype=np.float32).reshape(-1)
        self.rc = rc
        self.direction = int(direction)
        self.theta = float(tolerance_k)
        self.occupied_only = bool(occupied_only)
        self.phi_max_w = np.maximum(f(thermal_power_w), 1.0)
        self._cop = np.maximum(f(cop), 1e-3)
        rate = rc.temperature_gain_k(self.phi_max_w) / SPAN_K
        capacity = rc.C * SPAN_K / 3.6e6
        super().__init__(
            rate=rate,
            action_bound=f(action_bound),
            action_index=int(action_index),
            capacity=capacity,
            efficiency=self._cop,
            requirement_fn=self._requirement,
            soc_fn=self._state,
            margin=0.0,
            soc_cap=1.0,
            name=name,
        )


    def _col(self, obs_list: List[np.ndarray], idx: int) -> np.ndarray:
        return np.array([float(o[idx]) for o in obs_list], dtype=np.float32)

    def predicted_temperature_c(self, obs_list: List[np.ndarray]) -> np.ndarray:
        t_in = self._col(obs_list, IDX_T_IN)
        t_out = self._col(obs_list, IDX_T_OUT)
        return (t_in + self.rc.drift_k(t_in, t_out)).astype(np.float32)

    def _state(self, obs_list: List[np.ndarray]) -> np.ndarray:
        t_pred = self.predicted_temperature_c(obs_list)
        x = (t_pred - T_ZERO_C) if self.direction > 0 else (T_HIGH_C - t_pred)
        return np.clip(x / SPAN_K, 0.0, 1.0).astype(np.float32)

    def limit_c(self, obs_list: List[np.ndarray]) -> np.ndarray:
        if self.direction > 0:
            return (self._col(obs_list, IDX_T_HEAT) - self.theta).astype(np.float32)
        return (self._col(obs_list, IDX_T_COOL) + self.theta).astype(np.float32)

    def _requirement(self, obs_list: List[np.ndarray]) -> DeadlineRequirement:
        limit = self.limit_c(obs_list)
        x = (limit - T_ZERO_C) if self.direction > 0 else (T_HIGH_C - limit)
        required = np.clip(x / SPAN_K, 0.0, 1.0)
        occupied = (self._col(obs_list, IDX_OCCUPANT) > 0.0 if self.occupied_only
                    else np.ones(len(obs_list), dtype=bool))
        return DeadlineRequirement(
            soc=required,
            steps_to_deadline=np.zeros(len(obs_list), dtype=np.float32),
            active=occupied)


    def project(self, actions: np.ndarray, obs_list: List[np.ndarray]) -> np.ndarray:
        actions = np.asarray(actions, dtype=np.float32).copy()
        u = self.urgency(obs_list)
        must = (u["slack"] <= 0.0) & u["active"] & (u["gap"] > 0.0)
        a_req = np.minimum(self.action_for_soc_gain(u["gap"], u["rate"]),
                           self.action_bound)
        col = actions[:, self.action_index]
        if self.direction > 0:
            actions[:, self.action_index] = np.where(must, np.maximum(col, a_req), col)
        else:
            actions[:, self.action_index] = np.where(must, np.minimum(col, -a_req), col)
        return actions

    def electrical_power_w(self, obs_list: List[np.ndarray]) -> np.ndarray:
        return (self.phi_max_w / self._cop).astype(np.float32)


def build_comfort_barriers(env, comfort_cfg, rc: Optional[RCThermalModel] = None,
                           ) -> List[ThermalComfortBarrier]:
    if not getattr(comfort_cfg, "enabled", False):
        return []
    if env.hvac_action_index < 0:
        raise RuntimeError(
            "the comfort barrier needs a heat-pump action; this environment exposes "
            f"{env.action_names}")
    if env.hvac_control != "power":
        raise RuntimeError(
            "the comfort barrier needs hvac_control='power'. Under 'setpoint' the "
            "heat-pump action column is a +/-1.5 degC set-point offset and an internal "
            "integral thermostat issues the power command, so projecting that column "
            "cannot bound the indoor temperature. Run the scenario with "
            "hvac_control='power', or leave comfort as the soft reward term.")
    if env.heating_setpoint_idx is None:
        raise RuntimeError(
            "the comfort barrier needs a heating set point in the observation vector; "
            "run the environment with heat_pump=True")
    if rc is None:
        raise RuntimeError(
            "the comfort barrier needs an RCThermalModel. There is no default envelope: "
            "a hard thermal constraint built on an invented capacitance is a guarantee "
            "about nothing. Identify one with RCThermalModel.identify on logged "
            "(T_in, T_out, Phi) series -- stems.comfort.identify_from_citylearn does "
            "this for a CityLearn schema -- or supply measured parameters.")
    if rc.B != env.num_buildings:
        raise ValueError(f"RC model covers {rc.B} buildings, environment has "
                         f"{env.num_buildings}")

    hp = env.heat_pump_info()
    bounds = env._action_bounds()[:, env.hvac_action_index].astype(np.float32)
    theta = float(getattr(comfort_cfg, "tolerance_k", 2.0))
    occupied_only = bool(getattr(comfort_cfg, "occupied_only", True))
    from stems.thermal import CoPModel

    cop = CoPModel(hp["efficiency_heat"], hp["target_heat"], hp["efficiency_cool"],
                   hp["target_cool"], hp["nominal_power_heat"], hp["nominal_power_cool"])
    t_design_heat = float(getattr(comfort_cfg, "design_t_out_heating_c", -10.0))
    t_design_cool = float(getattr(comfort_cfg, "design_t_out_cooling_c", 35.0))
    cop_h = cop.cop(np.full(env.num_buildings, t_design_heat), heating=True)
    cop_c = cop.cop(np.full(env.num_buildings, t_design_cool), heating=False)

    barriers = [
        ThermalComfortBarrier(rc, +1, env.hvac_action_index,
                              thermal_power_w=1000.0 * hp["nominal_power_heat"] * cop_h,
                              cop=cop_h, action_bound=bounds, tolerance_k=theta,
                              occupied_only=occupied_only, name="comfort_heating"),
    ]
    if getattr(comfort_cfg, "enforce_cooling", True):
        barriers.append(
            ThermalComfortBarrier(rc, -1, env.hvac_action_index,
                                  thermal_power_w=1000.0 * hp["nominal_power_cool"] * cop_c,
                                  cop=cop_c, action_bound=bounds, tolerance_k=theta,
                                  occupied_only=occupied_only, name="comfort_cooling"))
    return barriers


def identify_from_rollout(env, steps: int = 672, seed: int = 0,
                          provenance: Optional[str] = None,
                          ) -> Tuple[RCThermalModel, RCThermalFit]:
    if env.hvac_control != "power":
        raise RuntimeError("identify_from_rollout needs hvac_control='power'")
    if env.hvac_action_index < 0:
        raise RuntimeError("identify_from_rollout needs a heat-pump action")
    from stems.thermal import CoPModel

    hp = env.heat_pump_info()
    cop_model = CoPModel(hp["efficiency_heat"], hp["target_heat"],
                         hp["efficiency_cool"], hp["target_cool"],
                         hp["nominal_power_heat"], hp["nominal_power_cool"])
    rng = np.random.default_rng(seed)
    j = env.hvac_action_index
    B, A = env.num_buildings, env.action_dim

    obs, _ = env.reset()
    t_in, t_out, phi = [], [], []
    for _ in range(int(steps)):
        a = np.zeros((B, A), dtype=np.float32)
        a[:, j] = rng.choice([-1.0, 0.0, 1.0], size=B).astype(np.float32)
        t_in.append(np.array([float(o[IDX_T_IN]) for o in obs], dtype=np.float64))
        t_out.append(np.array([float(o[IDX_T_OUT]) for o in obs], dtype=np.float64))
        heating = a[:, j] > 0.0
        cop = np.where(heating, cop_model.cop(t_out[-1], heating=True),
                       cop_model.cop(t_out[-1], heating=False))
        p_nom = np.where(heating, hp["nominal_power_heat"], hp["nominal_power_cool"])
        phi.append(1000.0 * a[:, j] * p_nom * cop)
        obs, _, term, trunc, _ = env.step(a)
        if term or trunc:
            break
    t_in.append(np.array([float(o[IDX_T_IN]) for o in obs], dtype=np.float64))
    t_out.append(t_out[-1])
    phi.append(np.zeros(B))
    return RCThermalModel.identify(
        np.stack(t_in), np.stack(t_out), np.stack(phi),
        dt_s=float(getattr(env._env, "seconds_per_time_step", SECONDS_PER_HOUR)),
        provenance=provenance or f"least squares on a {len(phi) - 1}-step open-loop "
                                 f"excitation rollout, seed {seed}")


def identify_from_citylearn(env, provenance: Optional[str] = None,
                            ) -> Tuple[RCThermalModel, RCThermalFit]:
    if env.using_mock:
        raise RuntimeError("identify_from_citylearn needs a real CityLearn environment")
    t_in, t_out, phi = [], [], []
    for b in env._env.buildings:
        sim = b.energy_simulation
        t_in.append(np.asarray(sim.indoor_dry_bulb_temperature, dtype=np.float64))
        t_out.append(np.asarray(b.weather.outdoor_dry_bulb_temperature, dtype=np.float64))
        heating = np.asarray(getattr(sim, "heating_demand", 0.0), dtype=np.float64)
        cooling = np.asarray(getattr(sim, "cooling_demand", 0.0), dtype=np.float64)
        heating = np.zeros_like(t_in[-1]) if heating.ndim == 0 else heating
        cooling = np.zeros_like(t_in[-1]) if cooling.ndim == 0 else cooling
        phi.append(1000.0 * (heating - cooling))
    n = min(len(a) for a in t_in)
    stack = lambda xs: np.stack([x[:n] for x in xs], axis=1)
    return RCThermalModel.identify(
        stack(t_in), stack(t_out), stack(phi),
        dt_s=float(getattr(env._env, "seconds_per_time_step", SECONDS_PER_HOUR)),
        provenance=provenance or "least squares on the CityLearn dataset series")
