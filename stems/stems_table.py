from __future__ import annotations

from typing import Dict, Optional

import numpy as np

COMFORT_LOW_C = 20.0
COMFORT_HIGH_C = 26.0
_SETPOINT_TOLERANCE_C = 2.0


def stems_table_metrics(
    net: np.ndarray,
    price: np.ndarray,
    carbon: np.ndarray,
    t_in: np.ndarray,
    soc: np.ndarray,
    *,
    soc_min: float,
    soc_max: float,
    p_building_max: float,
    hours_per_step: float = 1.0,
    indoor_temperature_available: bool = True,
    t_set: Optional[np.ndarray] = None,
    t_heat: Optional[np.ndarray] = None,
    occupant: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    net = np.asarray(net, dtype=np.float64)
    T, B = net.shape
    signed = net.sum(axis=1)
    imported = np.maximum(net, 0.0)

    cost = float((imported * np.asarray(price, dtype=np.float64)).sum())
    emission = float((imported * np.asarray(carbon, dtype=np.float64)).sum())

    steps_per_day = max(1, int(round(24 / hours_per_step)))
    daily_peaks = [float(signed[s:s + steps_per_day].max())
                   for s in range(0, T, steps_per_day)]
    avg_daily_peak = float(np.mean(daily_peaks)) if daily_peaks else 0.0

    electricity_consumption = float(imported.sum())

    ramping_rate = float(np.abs(np.diff(signed)).mean()) if T > 1 else 0.0

    t_in = np.asarray(t_in, dtype=np.float64)
    if not indoor_temperature_available or not np.isfinite(t_in).all():
        comfort_breach = np.zeros_like(t_in, dtype=bool)
        discomfort_proportion = None
        fixed_band_proportion = None
    elif t_set is not None and occupant is not None:
        t_set_a = np.asarray(t_set, dtype=np.float64)
        occ = np.asarray(occupant, dtype=np.float64) > 0.0
        if t_heat is None:
            breach = np.abs(t_in - t_set_a) > _SETPOINT_TOLERANCE_C
        else:
            t_heat_a = np.asarray(t_heat, dtype=np.float64)
            breach = (((t_in - t_set_a) > _SETPOINT_TOLERANCE_C)
                      | ((t_heat_a - t_in) > _SETPOINT_TOLERANCE_C))
        comfort_breach = breach & occ
        n_occ = float(occ.sum())
        discomfort_proportion = float(comfort_breach.sum() / n_occ) if n_occ > 0 else 0.0
        fixed_band_proportion = float(
            ((t_in < COMFORT_LOW_C) | (t_in > COMFORT_HIGH_C)).mean())
    else:
        comfort_breach = (t_in < COMFORT_LOW_C) | (t_in > COMFORT_HIGH_C)
        discomfort_proportion = None
        fixed_band_proportion = float(comfort_breach.mean())

    soc = np.asarray(soc, dtype=np.float64)
    soc_breach = (soc < soc_min) | (soc > soc_max)
    equipment_breach = np.abs(net) > p_building_max
    soc_equipment = soc_breach | equipment_breach
    with_comfort = soc_equipment | comfort_breach

    return {
        "stems_cost": cost,
        "stems_emission": emission,
        "stems_avg_daily_peak": avg_daily_peak,
        "stems_electricity_consumption": electricity_consumption,
        "stems_ramping_rate": ramping_rate,
        "stems_discomfort_proportion": discomfort_proportion,
        "stems_discomfort_fixed_band": fixed_band_proportion,
        "stems_safety_violations_soc_equipment": float(soc_equipment.mean()) * 100.0,
        "stems_safety_violations_with_comfort": (None if discomfort_proportion is None
                                                 else float(with_comfort.mean()) * 100.0),
        "stems_soc_violations_pct": float(soc_breach.mean()) * 100.0,
        "stems_equipment_violations_pct": float(equipment_breach.mean()) * 100.0,
    }


NORMALISED_KEYS = ("stems_cost", "stems_emission", "stems_avg_daily_peak",
                   "stems_electricity_consumption", "stems_ramping_rate")


def normalise_to_baseline(row: Dict[str, float],
                          baseline: Dict[str, float]) -> Dict[str, float]:
    out = dict(row)
    for k in NORMALISED_KEYS:
        b = baseline.get(k)
        if b is None or abs(b) < 1e-12:
            out[k + "_norm"] = float("nan")
        else:
            out[k + "_norm"] = float(row[k]) / float(b)
    return out
