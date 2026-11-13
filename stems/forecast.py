from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from stems.observations import obs_indices

__all__ = ["Forecast", "CausalForecaster", "OracleForecaster", "PRICE_PRED_LEAD_H",
           "record_idle_episode"]

_IDX_PRICE, _IDX_PRICE_1, _IDX_PRICE_2 = obs_indices(
    "electricity_pricing", "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2")
_IDX_DHW_DEMAND = obs_indices("dhw_demand")[0]
_IDX_NET = obs_indices("net_electricity_consumption")[0]
_IDX_SOC_ELEC = obs_indices("electrical_storage_soc")[0]
_IDX_SOC_DHW = obs_indices("dhw_storage_soc")[0]

PRICE_PRED_LEAD_H = (6, 12)

DAY = 24


@dataclass(frozen=True)
class Forecast:

    price: np.ndarray
    base_kw: np.ndarray
    dhw_demand_kwh: np.ndarray

    def __post_init__(self) -> None:
        H = self.price.shape[0]
        if self.base_kw.shape[0] != H or self.dhw_demand_kwh.shape[0] != H:
            raise ValueError(
                f"forecast horizons disagree: price {self.price.shape}, "
                f"base {self.base_kw.shape}, dhw {self.dhw_demand_kwh.shape}")

    @property
    def horizon(self) -> int:
        return int(self.price.shape[0])


class _BaseDecomposition:

    def __init__(self, battery, tank, elec_idx: int, dhw_idx: int) -> None:
        self.battery = battery
        self.tank = tank
        self.elec_idx = int(elec_idx)
        self.dhw_idx = int(dhw_idx)

    def __call__(self, obs_list: Sequence[np.ndarray], applied: Optional[np.ndarray],
                 next_obs_list: Sequence[np.ndarray]) -> np.ndarray:
        net = np.array([float(o[_IDX_NET]) for o in next_obs_list], dtype=np.float64)
        if applied is None:
            return net
        applied = np.asarray(applied, dtype=np.float64)
        storage = np.zeros_like(net)
        if self.battery is not None and self.elec_idx >= 0:
            soc = np.array([float(o[_IDX_SOC_ELEC]) for o in obs_list], dtype=np.float64)
            storage += self.battery.accepted_kwh(soc, applied[:, self.elec_idx])
        if self.tank is not None and self.dhw_idx >= 0:
            soc = np.array([float(o[_IDX_SOC_DHW]) for o in obs_list], dtype=np.float64)
            demand = np.array([float(o[_IDX_DHW_DEMAND]) for o in obs_list], dtype=np.float64)
            storage += self.tank.drawn_kwh(soc, applied[:, self.dhw_idx], demand)
        return net - storage


class CausalForecaster:

    def __init__(self, num_buildings: int, battery=None, tank=None,
                 elec_idx: int = 1, dhw_idx: int = 0, memory: int = DAY) -> None:
        self.B = int(num_buildings)
        self.memory = max(int(memory), DAY)
        self._decompose = _BaseDecomposition(battery, tank, elec_idx, dhw_idx)
        self.reset()

    def reset(self) -> None:
        self._price: List[float] = []
        self._base: List[np.ndarray] = []
        self._demand: List[np.ndarray] = []
        self._pending: Optional[tuple] = None
        self._total = 0

    def note_applied(self, obs_list: Sequence[np.ndarray], applied: np.ndarray) -> None:
        self._pending = ([np.asarray(o, dtype=np.float64) for o in obs_list],
                         np.asarray(applied, dtype=np.float64))

    def observe(self, next_obs_list: Sequence[np.ndarray]) -> None:
        if self._pending is None:
            return
        obs_list, applied = self._pending
        base = self._decompose(obs_list, applied, next_obs_list)
        self._price.append(float(obs_list[0][_IDX_PRICE]))
        self._base.append(base)
        self._demand.append(np.array([float(o[_IDX_DHW_DEMAND]) for o in obs_list],
                                     dtype=np.float64))
        self._total += 1
        cap = self.memory + 2
        if len(self._base) > cap:
            del self._price[:-cap], self._base[:-cap], self._demand[:-cap]
        self._pending = None

    @property
    def steps_observed(self) -> int:
        return self._total

    def _seasonal(self, series: List[np.ndarray], k: int, fallback: np.ndarray
                  ) -> np.ndarray:
        n = len(series)
        idx = n - DAY + k
        if idx < 0 or idx >= n or n < DAY + 1:
            return fallback
        correction = series[-1] - series[n - 1 - DAY]
        return series[idx] + correction

    def _price_at(self, obs: np.ndarray, k: int) -> float:
        anchors = {0: float(obs[_IDX_PRICE]),
                   PRICE_PRED_LEAD_H[0]: float(obs[_IDX_PRICE_1]),
                   PRICE_PRED_LEAD_H[1]: float(obs[_IDX_PRICE_2])}
        if k in anchors:
            return anchors[k]
        n = len(self._price)
        idx = n - DAY + k
        if 0 <= idx < n and n >= DAY:
            return float(self._price[idx])
        held = max(a for a in anchors if a <= k) if any(a <= k for a in anchors) else 0
        return anchors[held]

    def predict(self, obs_list: Sequence[np.ndarray], horizon: int) -> Forecast:
        H = max(1, int(horizon))
        obs_list = [np.asarray(o, dtype=np.float64) for o in obs_list]
        price = np.array([self._price_at(obs_list[0], k) for k in range(H)],
                         dtype=np.float64)
        now_demand = np.array([float(o[_IDX_DHW_DEMAND]) for o in obs_list],
                              dtype=np.float64)
        last_base = (self._base[-1] if self._base
                     else np.array([float(o[_IDX_NET]) for o in obs_list], dtype=np.float64))
        base = np.stack([self._seasonal(self._base, k, last_base) for k in range(H)])
        demand = np.stack([now_demand if k == 0
                           else self._seasonal(self._demand, k, now_demand)
                           for k in range(H)])
        return Forecast(price=price, base_kw=base,
                        dhw_demand_kwh=np.maximum(demand, 0.0))


class OracleForecaster:

    def __init__(self, price: np.ndarray, base_kw: np.ndarray,
                 dhw_demand_kwh: np.ndarray) -> None:
        self.price = np.asarray(price, dtype=np.float64).reshape(-1)
        self.base = np.asarray(base_kw, dtype=np.float64)
        self.demand = np.asarray(dhw_demand_kwh, dtype=np.float64)
        if self.base.shape[0] != self.price.shape[0] or \
                self.demand.shape[0] != self.price.shape[0]:
            raise ValueError(
                f"recording lengths disagree: price {self.price.shape}, "
                f"base {self.base.shape}, demand {self.demand.shape}")
        self.T, self.B = self.base.shape
        self.reset()

    def reset(self) -> None:
        self.t = 0

    def note_applied(self, obs_list: Sequence[np.ndarray], applied: np.ndarray) -> None:
        return None

    def observe(self, next_obs_list: Sequence[np.ndarray]) -> None:
        self.t = min(self.t + 1, self.T)

    @property
    def steps_observed(self) -> int:
        return self.t

    def predict(self, obs_list: Sequence[np.ndarray], horizon: int) -> Forecast:
        H = max(1, int(horizon))
        idx = np.clip(np.arange(self.t, self.t + H), 0, self.T - 1)
        return Forecast(price=self.price[idx].copy(),
                        base_kw=self.base[idx].copy(),
                        dhw_demand_kwh=np.maximum(self.demand[idx].copy(), 0.0))


def record_idle_episode(env, max_steps: int) -> OracleForecaster:
    obs, _ = env.reset()
    price: List[float] = []
    base: List[np.ndarray] = []
    demand: List[np.ndarray] = []
    for _ in range(int(max_steps)):
        price.append(float(obs[0][_IDX_PRICE]))
        demand.append(np.array([float(o[_IDX_DHW_DEMAND]) for o in obs], dtype=np.float64))
        actions = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
        nxt, _, term, trunc, _ = env.step(actions)
        base.append(np.array([float(o[_IDX_NET]) for o in nxt], dtype=np.float64))
        obs = nxt
        if term or trunc:
            break
    return OracleForecaster(np.array(price), np.stack(base), np.stack(demand))
