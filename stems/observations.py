from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

OBS_NAMES: List[str] = [
    "day_type",
    "hour",
    "outdoor_dry_bulb_temperature",
    "outdoor_dry_bulb_temperature_predicted_1",
    "outdoor_dry_bulb_temperature_predicted_2",
    "outdoor_dry_bulb_temperature_predicted_3",
    "diffuse_solar_irradiance",
    "diffuse_solar_irradiance_predicted_1",
    "diffuse_solar_irradiance_predicted_2",
    "diffuse_solar_irradiance_predicted_3",
    "direct_solar_irradiance",
    "direct_solar_irradiance_predicted_1",
    "direct_solar_irradiance_predicted_2",
    "direct_solar_irradiance_predicted_3",
    "carbon_intensity",
    "indoor_dry_bulb_temperature",
    "non_shiftable_load",
    "solar_generation",
    "dhw_storage_soc",
    "electrical_storage_soc",
    "net_electricity_consumption",
    "electricity_pricing",
    "electricity_pricing_predicted_1",
    "electricity_pricing_predicted_2",
    "cooling_demand",
    "dhw_demand",
    "occupant_count",
    "indoor_dry_bulb_temperature_cooling_set_point",
]

HEATPUMP_OBS_NAMES: List[str] = [
    "indoor_dry_bulb_temperature_heating_set_point",
    "heating_electricity_consumption",
]

CANONICAL_OBS_NAMES: List[str] = OBS_NAMES + HEATPUMP_OBS_NAMES

T_OUT_PRED_LEAD_H: Tuple[float, float, float] = (6.0, 12.0, 24.0)

OBS_DIM = len(OBS_NAMES)
ACTION_DIM = 3

EV_SLOT_FIELDS: List[str] = [
    "connected_state",
    "departure_time",
    "required_soc_departure",
    "soc",
    "battery_capacity",
]

EV_ACTION_PREFIX = "electric_vehicle_storage"

_CANONICAL_INDEX: Dict[str, int] = {n: i for i, n in enumerate(CANONICAL_OBS_NAMES)}

if len(_CANONICAL_INDEX) != len(CANONICAL_OBS_NAMES):
    raise RuntimeError("CANONICAL_OBS_NAMES contains a duplicate observation name")


def obs_index(name: str) -> int:
    try:
        return _CANONICAL_INDEX[name]
    except KeyError:
        raise KeyError(
            f"{name!r} is not a canonical STEMS observation. Electric-vehicle slot "
            "fields have no fixed index -- ask the environment "
            "(STEMSEnvironment.index_of / .ev_obs_layout). Known names: "
            f"{CANONICAL_OBS_NAMES}"
        ) from None


def obs_indices(*names: str) -> Tuple[int, ...]:
    return tuple(obs_index(n) for n in names)


def ev_slot_obs_names(slot: int) -> List[str]:
    return [f"ev{slot}_{f}" for f in EV_SLOT_FIELDS]


def ev_native_obs_names(charger_id: str) -> Dict[str, str]:
    base = f"connected_electric_vehicle_at_charger_{charger_id}"
    return {
        "connected_state": f"electric_vehicle_charger_{charger_id}_connected_state",
        "departure_time": f"{base}_departure_time",
        "required_soc_departure": f"{base}_required_soc_departure",
        "soc": f"{base}_soc",
        "battery_capacity": f"{base}_battery_capacity",
    }


def ev_slot_action_name(slot: int) -> str:
    return f"{EV_ACTION_PREFIX}_{slot}"


def schema_observation_names(heat_pump: bool = True) -> List[str]:
    return list(CANONICAL_OBS_NAMES) if heat_pump else list(OBS_NAMES)


def selected_obs_names(heat_pump: bool = False, ev_slots: int = 0) -> List[str]:
    names = list(OBS_NAMES)
    if heat_pump:
        names += list(HEATPUMP_OBS_NAMES)
    for slot in range(int(ev_slots)):
        names += ev_slot_obs_names(slot)
    return names


def index_in(names: Sequence[str], name: str) -> int:
    try:
        return list(names).index(name)
    except ValueError:
        raise KeyError(
            f"{name!r} is not in this environment's observation vector. Present: "
            f"{list(names)}"
        ) from None
