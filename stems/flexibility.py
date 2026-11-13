from __future__ import annotations

from dataclasses import dataclass, field
from typing import (Any, Callable, ClassVar, Dict, List, Optional, Sequence,
                    Tuple)

import numpy as np

from stems.deadline import (DeadlineRequirement, DeadlineStorageBarrier,
                            coupled_feasibility, prioritise)
from stems.protocols import EVProvider, PlantProvider

__all__ = [
    "FlexibleLoad",
    "FlexibilityPortfolio",
    "LOAD_KINDS",
]

LOAD_KINDS: Tuple[str, ...] = ("battery", "dhw", "legionella", "ev", "envelope")

SocFn = Callable[[Sequence[np.ndarray]], np.ndarray]


@dataclass
class FlexibleLoad:

    name: str
    kind: str
    action_index: int
    plant: Optional[Any] = None
    soc_fn: Optional[SocFn] = None
    band: Optional[Tuple[Any, Any]] = None
    barrier: Optional[DeadlineStorageBarrier] = None
    action_bound: float = 1.0
    capacity_kwh: Optional[np.ndarray] = None
    power_kw: Optional[np.ndarray] = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.kind not in LOAD_KINDS:
            raise ValueError(f"unknown load kind {self.kind!r}; choose from {LOAD_KINDS}")
        if self.barrier is None and self.band is None:
            raise ValueError(
                f"flexible load {self.name!r} has neither a band nor a deadline, so it "
                "constrains nothing. A load with no constraint is not a load; leave it "
                "out of the portfolio.")
        if self.capacity_kwh is not None:
            self.capacity_kwh = np.asarray(self.capacity_kwh, dtype=np.float64).reshape(-1)
        if self.power_kw is not None:
            self.power_kw = np.asarray(self.power_kw, dtype=np.float64).reshape(-1)
        if self.barrier is not None and self.barrier.action_index != self.action_index:
            raise ValueError(
                f"load {self.name!r} owns action column {self.action_index} but its "
                f"barrier projects onto column {self.barrier.action_index}")


    @property
    def has_band(self) -> bool:
        return self.band is not None and hasattr(self.plant, "safe_interval")

    @property
    def has_deadline(self) -> bool:
        return self.barrier is not None

    @property
    def exact_projection(self) -> bool:
        band_ok = (not self.has_band) or self.plant is not None
        deadline_ok = (not self.has_deadline) or self.barrier.exact_projection
        return bool(band_ok and deadline_ok)

    def project_band(self, actions: np.ndarray,
                     obs_list: Sequence[np.ndarray]) -> np.ndarray:
        if not self.has_band or self.soc_fn is None:
            return actions
        soc = np.asarray(self.soc_fn(obs_list), dtype=np.float32)
        lo, hi = self.band
        a_lo, a_hi = self.plant.safe_interval(soc, lo, hi, a_max=self.action_bound)
        actions = np.asarray(actions, dtype=np.float32).copy()
        actions[:, self.action_index] = np.clip(actions[:, self.action_index], a_lo, a_hi)
        return actions

    def project_deadline(self, actions: np.ndarray,
                         obs_list: Sequence[np.ndarray]) -> np.ndarray:
        if self.barrier is None:
            return actions
        return self.barrier.project(actions, list(obs_list))

    def project(self, actions: np.ndarray,
                obs_list: Sequence[np.ndarray]) -> np.ndarray:
        return self.project_deadline(self.project_band(actions, obs_list), obs_list)


    def urgency(self, obs_list: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
        if self.barrier is None:
            return {}
        return self.barrier.urgency(list(obs_list))

    def energy_still_required_kwh(self, obs_list: Sequence[np.ndarray]) -> np.ndarray:
        if self.barrier is None:
            return np.zeros(len(obs_list), dtype=np.float32)
        return self.barrier.energy_still_required_kwh(list(obs_list))

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "action_index": self.action_index,
                "plant": type(self.plant).__name__ if self.plant is not None else None,
                "has_band": self.has_band, "has_deadline": self.has_deadline,
                "exact_projection": self.exact_projection,
                "capacity_kwh": (None if self.capacity_kwh is None
                                 else [round(float(c), 4) for c in self.capacity_kwh]),
                "power_kw": (None if self.power_kw is None
                             else [round(float(p), 4) for p in self.power_kw]),
                "notes": self.notes}


@dataclass
class FlexibilityPortfolio:

    loads: List[FlexibleLoad] = field(default_factory=list)
    cap_kw: float = float("inf")
    dt_hours: float = 1.0
    coordination: str = "independent"
    net_index: int = 0

    COORDINATIONS: ClassVar[Tuple[str, ...]] = ("independent", "proportional", "edf")

    def __post_init__(self) -> None:
        if self.coordination not in self.COORDINATIONS:
            raise ValueError(f"coordination must be one of {self.COORDINATIONS}, "
                             f"got {self.coordination!r}")
        names = [l.name for l in self.loads]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate load names in the portfolio: {names}")


    @classmethod
    def from_barriers(cls, barriers: Sequence[DeadlineStorageBarrier],
                      cap_kw: float = float("inf"), dt_hours: float = 1.0,
                      coordination: str = "independent",
                      net_index: int = 0) -> "FlexibilityPortfolio":
        loads = [cls._load_from_barrier(b) for b in barriers]
        return cls(loads=loads, cap_kw=cap_kw, dt_hours=dt_hours,
                   coordination=coordination, net_index=net_index)

    @staticmethod
    def _load_from_barrier(barrier: DeadlineStorageBarrier) -> FlexibleLoad:
        kind = {"ev": "ev", "dhw": "dhw", "legionella": "legionella"}.get(
            barrier.name, "envelope" if barrier.name.startswith("comfort") else None)
        if kind is None:
            kind = "envelope"
        power = getattr(barrier, "_p_charge", None)
        return FlexibleLoad(name=barrier.name, kind=kind,
                            action_index=barrier.action_index, barrier=barrier,
                            action_bound=float(np.max(barrier.action_bound)),
                            capacity_kwh=barrier.capacity,
                            power_kw=None if power is None else np.asarray(power),
                            notes="wrapped from an existing barrier")

    @classmethod
    def from_environment(cls, env: PlantProvider, cbf_cfg: Any,
                         soc_band: Optional[Tuple[Any, Any]] = None,
                         dhw_barrier: Optional[DeadlineStorageBarrier] = None,
                         ev_barrier: Optional[DeadlineStorageBarrier] = None,
                         legionella_barrier: Optional[DeadlineStorageBarrier] = None,
                         extra: Sequence[DeadlineStorageBarrier] = (),
                         coordination: str = "independent"
                         ) -> "FlexibilityPortfolio":
        from stems.observations import obs_index

        loads: List[FlexibleLoad] = []
        e = env.electrical_storage_action_index
        if e >= 0:
            info = env.battery_info()
            lo, hi = soc_band if soc_band is not None else (cbf_cfg.SOC_min, cbf_cfg.SOC_max)
            idx = obs_index("electrical_storage_soc")
            loads.append(FlexibleLoad(
                name="battery", kind="battery", action_index=e,
                plant=env.battery_model(),
                soc_fn=lambda obs, i=idx: np.array([float(o[i]) for o in obs],
                                                   dtype=np.float32),
                band=(lo, hi), barrier=None,
                capacity_kwh=info["capacity"], power_kw=info["nominal_power"],
                notes="band only: a house battery owes nothing by a deadline"))
        for barrier in (dhw_barrier, legionella_barrier, ev_barrier, *extra):
            if barrier is not None:
                loads.append(cls._load_from_barrier(barrier))
        return cls(loads=loads, cap_kw=float(cbf_cfg.P_grid_max),
                   coordination=coordination,
                   net_index=obs_index("net_electricity_consumption"))


    @property
    def barriers(self) -> List[DeadlineStorageBarrier]:
        return [l.barrier for l in self.loads if l.barrier is not None]

    def __len__(self) -> int:
        return len(self.loads)

    def __iter__(self):
        return iter(self.loads)

    def of_kind(self, kind: str) -> List[FlexibleLoad]:
        return [l for l in self.loads if l.kind == kind]

    def describe(self) -> List[Dict[str, Any]]:
        return [l.describe() for l in self.loads]


    def project_deadlines(self, actions: np.ndarray, obs_list: Sequence[np.ndarray],
                          cap_kw: Optional[float] = None) -> np.ndarray:
        for barrier in self.barriers:
            actions = barrier.project(actions, list(obs_list))
        if self.coordination != "independent":
            actions = self.allocate_under_cap(actions, obs_list, cap_kw)
        return actions

    def project(self, actions: np.ndarray, obs_list: Sequence[np.ndarray],
                cap_kw: Optional[float] = None) -> np.ndarray:
        for load in self.loads:
            actions = load.project_band(actions, obs_list)
        return self.project_deadlines(actions, obs_list, cap_kw)

    def allocate_under_cap(self, actions: np.ndarray, obs_list: Sequence[np.ndarray],
                           cap_kw: Optional[float] = None) -> np.ndarray:
        cap = self.cap_kw if cap_kw is None else float(cap_kw)
        net = np.array([float(o[self.net_index]) for o in obs_list], dtype=np.float32)
        headroom = max(cap - float(np.maximum(net, 0.0).sum()), 0.0)

        entries, requested_total = [], 0.0
        for load in self.loads:
            if load.power_kw is None or load.barrier is None:
                continue
            u = load.barrier.urgency(list(obs_list))
            kw = np.maximum(actions[:, load.action_index], 0.0) * load.power_kw
            for i in range(len(kw)):
                if kw[i] > 1e-9:
                    entries.append((float(u["steps_to_deadline"][i]), load, i, float(kw[i])))
                    requested_total += float(kw[i])

        if requested_total <= headroom + 1e-9 or requested_total <= 1e-9:
            return actions
        if self.coordination == "proportional":
            scale = headroom / requested_total
            for _, load, i, kw in entries:
                actions[i, load.action_index] = actions[i, load.action_index] * scale
            return actions
        entries.sort(key=lambda e: (e[0], -e[3]))
        budget = headroom
        for _, load, i, kw in entries:
            grant = min(kw, budget)
            budget -= grant
            j = load.action_index
            actions[i, j] = actions[i, j] * (grant / kw) if kw > 1e-9 else 0.0
        return actions


    def feasibility(self, obs_list: Sequence[np.ndarray],
                    cap_kw: Optional[float] = None) -> Optional[Dict[str, Any]]:
        barriers = self.barriers
        if not barriers:
            return None
        cap = self.cap_kw if cap_kw is None else float(cap_kw)
        report = coupled_feasibility(barriers, list(obs_list), power_cap_kw=cap,
                                     dt_hours=self.dt_hours)
        if not report["feasible"]:
            report["priority"] = prioritise(barriers, list(obs_list), power_cap_kw=cap,
                                            dt_hours=self.dt_hours)
        return report
