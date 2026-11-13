from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

__all__ = ["DegradationModel", "DegradationAccountant", "CITYLEARN_KAPPA_RANGE"]

CITYLEARN_KAPPA_RANGE = (1e-5, 1e-4)

MODES = ("none", "throughput", "throughput+dod")


@dataclass(frozen=True)
class DegradationModel:

    kappa: np.ndarray
    capacity_kwh: np.ndarray
    mode: str = "throughput"
    dod_exponent: float = 0.0
    calendar_loss_per_second: Optional[float] = None
    price_per_kwh: float = 0.0
    dt_s: float = 3600.0
    reversal_threshold: float = 0.05
    source: str = "unspecified"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        f = lambda x: np.asarray(x, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "kappa", f(self.kappa))
        object.__setattr__(self, "capacity_kwh", f(self.capacity_kwh))
        if self.kappa.shape != self.capacity_kwh.shape:
            raise ValueError("kappa and capacity_kwh need one entry per building")
        if np.any(self.kappa < 0.0):
            raise ValueError("kappa must be non-negative")
        if self.dod_exponent < 0.0:
            raise ValueError("dod_exponent p must be non-negative: a deeper cycle "
                             "cannot cost less per kWh than a shallow one")
        if self.mode == "throughput+dod" and self.dod_exponent == 0.0:
            pass

    @property
    def B(self) -> int:
        return int(self.kappa.size)

    @classmethod
    def from_environment(cls, env, cfg) -> "DegradationModel":
        info = env.battery_degradation_info()
        return cls(kappa=info["capacity_loss_coefficient"],
                   capacity_kwh=info["capacity"],
                   mode=cfg.mode,
                   dod_exponent=float(cfg.dod_exponent),
                   reversal_threshold=float(cfg.reversal_threshold),
                   calendar_loss_per_second=(cfg.calendar_loss_per_second
                                             if cfg.calendar else None),
                   price_per_kwh=float(cfg.price_eur_per_kwh),
                   dt_s=float(info["seconds_per_time_step"]),
                   source=str(info["source"]))

    def describe(self) -> Dict[str, object]:
        return {"mode": self.mode, "kappa": self.kappa.tolist(),
                "capacity_kwh": self.capacity_kwh.tolist(),
                "dod_exponent": self.dod_exponent,
                "reversal_threshold": self.reversal_threshold,
                "calendar_loss_per_second": self.calendar_loss_per_second,
                "price_per_kwh": self.price_per_kwh, "dt_s": self.dt_s,
                "source": self.source}


class DegradationAccountant:

    def __init__(self, model: DegradationModel, limit_kwh_per_episode: Optional[float] = None,
                 episode_steps: Optional[int] = None) -> None:
        self.model = model
        B = model.B
        self.limit_kwh_per_episode = (None if limit_kwh_per_episode is None
                                      else float(limit_kwh_per_episode))
        self.episode_steps = None if episode_steps is None else max(int(episode_steps), 1)
        self._B = B
        self.reset()

    def reset(self) -> None:
        m = self.model
        self.capacity_kwh = m.capacity_kwh.astype(np.float64).copy()
        self.cumulative_loss_kwh = np.zeros(self._B, dtype=np.float64)
        self.throughput_kwh = np.zeros(self._B, dtype=np.float64)
        self._ref_soc: Optional[np.ndarray] = None
        self._ext_soc: Optional[np.ndarray] = None
        self._direction = np.zeros(self._B, dtype=np.int8)
        self._pending_fade = np.zeros(self._B, dtype=np.float64)
        self.half_cycles = np.zeros(self._B, dtype=np.int64)
        self.mean_half_cycle_depth = np.zeros(self._B, dtype=np.float64)
        self._depth_sum = np.zeros(self._B, dtype=np.float64)
        self.steps = 0

    @property
    def budget_per_step_kwh(self) -> Optional[float]:
        if self.limit_kwh_per_episode is None or self.episode_steps is None:
            return None
        return self.limit_kwh_per_episode / self.episode_steps

    def step(self, soc_prev: np.ndarray, soc_next: np.ndarray) -> np.ndarray:
        m = self.model
        if m.mode == "none":
            self.steps += 1
            return np.zeros(self._B, dtype=np.float64)
        soc_prev = np.asarray(soc_prev, dtype=np.float64).reshape(-1)
        soc_next = np.asarray(soc_next, dtype=np.float64).reshape(-1)
        d_soc = soc_next - soc_prev
        energy_kwh = np.abs(d_soc) * self.capacity_kwh
        self.throughput_kwh += energy_kwh

        fade = (m.kappa * m.capacity_kwh * energy_kwh
                / (2.0 * np.maximum(self.capacity_kwh, 1e-9)))

        if m.mode == "throughput+dod":
            fade = self._accrue_half_cycle(soc_prev, soc_next, fade)

        if m.calendar_loss_per_second is not None:
            fade = fade + m.calendar_loss_per_second * m.capacity_kwh * m.dt_s

        self.capacity_kwh = np.maximum(self.capacity_kwh - fade, 0.0)
        self.cumulative_loss_kwh += fade
        self.steps += 1
        return fade

    def _accrue_half_cycle(self, soc_prev: np.ndarray, soc_next: np.ndarray,
                           fade: np.ndarray) -> np.ndarray:
        m = self.model
        if self._ref_soc is None:
            self._ref_soc = soc_prev.copy()
            self._ext_soc = soc_prev.copy()
        thr = float(m.reversal_threshold)
        ext = self._ext_soc
        d = self._direction

        step_dir = np.sign(soc_next - soc_prev).astype(np.int8)
        starting = (d == 0) & (np.abs(soc_next - self._ref_soc) > thr)
        d = np.where(starting, step_dir, d).astype(np.int8)
        ext = np.where(starting, soc_next, ext)

        extending = (d != 0) & ((soc_next - ext) * d > 0.0)
        ext = np.where(extending, soc_next, ext)

        retracement = (ext - soc_next) * d
        closing = (d != 0) & ~extending & (retracement > thr)

        committed = np.zeros(self._B, dtype=np.float64)
        if np.any(closing):
            depth = np.abs(ext - self._ref_soc)
            scale = np.power(np.clip(depth, 1e-9, 1.0), m.dod_exponent)
            committed = np.where(closing, self._pending_fade * scale, 0.0)
            self._pending_fade = np.where(closing, 0.0, self._pending_fade)
            self._ref_soc = np.where(closing, ext, self._ref_soc)
            self.half_cycles += closing.astype(np.int64)
            self._depth_sum += np.where(closing, depth, 0.0)
            self.mean_half_cycle_depth = np.where(
                self.half_cycles > 0,
                self._depth_sum / np.maximum(self.half_cycles, 1), 0.0)
            ext = np.where(closing, soc_next, ext)
            d = np.where(closing, -d, d).astype(np.int8)

        self._ext_soc = ext
        self._direction = d
        self._pending_fade += fade
        return committed

    def flush(self) -> np.ndarray:
        if self.model.mode != "throughput+dod" or self._ref_soc is None:
            return np.zeros(self._B, dtype=np.float64)
        depth = np.abs(self._ext_soc - self._ref_soc)
        scale = np.power(np.clip(depth, 1e-9, 1.0), self.model.dod_exponent)
        pending = self._pending_fade * scale
        self._pending_fade = np.zeros(self._B, dtype=np.float64)
        self.capacity_kwh = np.maximum(self.capacity_kwh - pending, 0.0)
        self.cumulative_loss_kwh += pending
        return pending

    def constraint_cost(self, fade_kwh: np.ndarray) -> np.ndarray:
        budget = self.budget_per_step_kwh
        if budget is None:
            return np.zeros(self._B, dtype=np.float32)
        return (np.asarray(fade_kwh, dtype=np.float64) > budget).astype(np.float32)

    def reward_penalty(self, fade_kwh: np.ndarray) -> np.ndarray:
        return (self.model.price_per_kwh
                * np.asarray(fade_kwh, dtype=np.float64)).astype(np.float64)

    def kpis(self, prefix: str = "battery") -> Dict[str, float]:
        m = self.model
        rated = float(m.capacity_kwh.sum())
        lost = float(self.cumulative_loss_kwh.sum())
        out = {
            f"{prefix}_capacity_loss_kwh": lost,
            f"{prefix}_capacity_loss_fraction": (lost / rated if rated > 1e-9
                                                 else float("nan")),
            f"{prefix}_throughput_kwh": float(self.throughput_kwh.sum()),
            f"{prefix}_degradation_cost": float(m.price_per_kwh * lost),
            f"{prefix}_capacity_loss_kwh_worst_building": float(
                self.cumulative_loss_kwh.max()) if self._B else float("nan"),
        }
        if m.mode == "throughput+dod":
            out[f"{prefix}_half_cycles"] = float(self.half_cycles.mean())
            out[f"{prefix}_mean_half_cycle_depth"] = float(self.mean_half_cycle_depth.mean())
        return out
