from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

__all__ = [
    "PlantModel",
    "Environment",
    "PlantProvider",
    "EVProvider",
    "missing_capabilities",
]


@runtime_checkable
class PlantModel(Protocol):

    def next_soc(self, soc: np.ndarray, action: np.ndarray, *args: Any,
                 **kwargs: Any) -> np.ndarray:
        pass

    def action_for_soc(self, soc: np.ndarray, target: np.ndarray, *args: Any,
                       **kwargs: Any) -> np.ndarray:
        pass


@runtime_checkable
class Environment(Protocol):

    @property
    def num_buildings(self) -> int: ...

    @property
    def obs_dim(self) -> int: ...

    @property
    def action_dim(self) -> int: ...

    @property
    def obs_names(self) -> List[str]: ...

    @property
    def action_names(self) -> List[str]: ...

    @property
    def executed_actions(self) -> np.ndarray: ...

    def index_of(self, name: str) -> int:
        pass

    def reset(self) -> Tuple[List[np.ndarray], Dict[str, Any]]: ...

    def step(self, actions: np.ndarray
             ) -> Tuple[List[np.ndarray], List[float], bool, bool, Dict[str, Any]]: ...


@runtime_checkable
class PlantProvider(Protocol):

    @property
    def electrical_storage_action_index(self) -> int: ...

    @property
    def dhw_action_index(self) -> int: ...

    @property
    def hvac_action_index(self) -> int: ...

    def battery_model(self) -> PlantModel: ...

    def battery_info(self) -> Dict[str, np.ndarray]:
        pass

    def dhw_tank_model(self) -> Optional[PlantModel]: ...

    def dhw_info(self) -> Dict[str, np.ndarray]:
        pass


@runtime_checkable
class EVProvider(Protocol):

    def ev_action_indices(self) -> List[int]: ...

    def ev_obs_layout(self) -> List[Dict[str, int]]: ...

    def ev_fleet_model(self, slot: int = 0) -> Any: ...

    @property
    def ev_draw_kwh(self) -> np.ndarray: ...

    @property
    def ev_departures(self) -> List[Dict[str, float]]: ...


_PROTOCOLS = {"Environment": Environment, "PlantProvider": PlantProvider,
              "EVProvider": EVProvider}


def missing_capabilities(adapter: object) -> Dict[str, Sequence[str]]:
    out: Dict[str, Sequence[str]] = {}
    cls = type(adapter)
    for name, proto in _PROTOCOLS.items():
        missing = [m for m in protocol_attrs(proto)
                   if not (hasattr(cls, m) or hasattr(adapter, m))]
        if missing:
            out[name] = sorted(missing)
    return out


def protocol_attrs(proto):
    attrs = getattr(proto, "__protocol_attrs__", None)
    if attrs is None:
        import typing
        attrs = typing._get_protocol_attrs(proto)
    return set(attrs)

