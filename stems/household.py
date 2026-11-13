from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol, Sequence, Tuple

import numpy as np

LB_TO_KG = 0.45359237

EPA_TD_LOSS_FRACTION = 0.051


class Tariff(Protocol):

    name: str
    source: str

    def bill_usd(
        self, import_kwh: np.ndarray, month: np.ndarray, hour: np.ndarray
    ) -> Dict[str, np.ndarray]:
        ...


@dataclass(frozen=True)
class HourlyEnergyTariff:

    price_usd_per_kwh: np.ndarray
    name: str
    source: str

    def bill_usd(
        self, import_kwh: np.ndarray, month: np.ndarray, hour: np.ndarray
    ) -> Dict[str, np.ndarray]:
        p = np.asarray(self.price_usd_per_kwh, dtype=np.float64)
        E = np.asarray(import_kwh, dtype=np.float64)
        if p.shape != (E.shape[0],):
            raise ValueError(
                f"price vector has shape {p.shape}, expected ({E.shape[0]},) to match "
                "the number of simulation steps")
        energy = (E * p[:, None]).sum(axis=0)
        zero = np.zeros_like(energy)
        return {"energy_usd": energy, "fixed_usd": zero.copy(),
                "adders_usd": zero.copy(), "total_usd": energy.copy()}


@dataclass(frozen=True)
class TieredMonthlyTariff:

    customer_charge_usd_per_month: float
    tier_upper_kwh: Tuple[float, ...]
    tier_rate_usd_per_kwh: Tuple[float, ...]
    rider_usd_per_kwh: object
    name: str
    source: str

    @property
    def time_varying(self) -> bool:
        return np.ndim(self.rider_usd_per_kwh) > 0

    def __post_init__(self) -> None:
        if len(self.tier_rate_usd_per_kwh) != len(self.tier_upper_kwh) + 1:
            raise ValueError(
                f"{len(self.tier_upper_kwh)} tier boundaries need "
                f"{len(self.tier_upper_kwh) + 1} rates, got "
                f"{len(self.tier_rate_usd_per_kwh)}")
        if list(self.tier_upper_kwh) != sorted(set(self.tier_upper_kwh)):
            raise ValueError("tier_upper_kwh must be strictly increasing")

    def _month_energy_charge(self, monthly_kwh: np.ndarray) -> np.ndarray:
        edges = np.concatenate([[0.0], np.asarray(self.tier_upper_kwh, dtype=np.float64),
                                [np.inf]])
        rates = np.asarray(self.tier_rate_usd_per_kwh, dtype=np.float64)
        charge = np.zeros_like(monthly_kwh)
        for k, rate in enumerate(rates):
            lo, hi = edges[k], edges[k + 1]
            in_tier = np.clip(monthly_kwh, lo, hi) - lo
            charge += rate * in_tier
        return charge

    def bill_usd(
        self, import_kwh: np.ndarray, month: np.ndarray, hour: np.ndarray
    ) -> Dict[str, np.ndarray]:
        E = np.asarray(import_kwh, dtype=np.float64)
        m = np.asarray(month, dtype=np.int64)
        if m.shape != (E.shape[0],):
            raise ValueError(
                f"month column has shape {m.shape}, expected ({E.shape[0]},)")
        months = np.unique(m)
        energy = np.zeros(E.shape[1], dtype=np.float64)
        for mm in months:
            energy += self._month_energy_charge(E[m == mm].sum(axis=0))
        fixed = np.full(E.shape[1],
                        self.customer_charge_usd_per_month * float(months.size))
        if self.time_varying:
            r = np.asarray(self.rider_usd_per_kwh, dtype=np.float64)
            if r.shape != (E.shape[0],):
                raise ValueError(
                    f"rider vector has shape {r.shape}, expected ({E.shape[0]},)")
            adders = (E * r[:, None]).sum(axis=0)
        else:
            adders = float(self.rider_usd_per_kwh) * E.sum(axis=0)
        return {"energy_usd": energy, "fixed_usd": fixed, "adders_usd": adders,
                "total_usd": energy + fixed + adders}


class EmissionFactor(Protocol):

    name: str
    source: str

    def co2_kg(self, import_kwh: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class HourlyEmissionFactor:

    kg_per_kwh: np.ndarray
    name: str
    source: str

    def co2_kg(self, import_kwh: np.ndarray) -> np.ndarray:
        c = np.asarray(self.kg_per_kwh, dtype=np.float64)
        E = np.asarray(import_kwh, dtype=np.float64)
        if c.shape != (E.shape[0],):
            raise ValueError(
                f"carbon series has shape {c.shape}, expected ({E.shape[0]},)")
        return (E * c[:, None]).sum(axis=0)


@dataclass(frozen=True)
class ConstantEmissionFactor:

    kg_per_kwh: float
    name: str
    source: str

    def co2_kg(self, import_kwh: np.ndarray) -> np.ndarray:
        return float(self.kg_per_kwh) * np.asarray(
            import_kwh, dtype=np.float64).sum(axis=0)


def egrid_constant_factor(
    lb_co2_per_mwh: float,
    name: str,
    source: str,
    delivered: bool = True,
) -> ConstantEmissionFactor:
    kg_per_kwh = lb_co2_per_mwh * LB_TO_KG / 1000.0
    if delivered:
        kg_per_kwh /= (1.0 - EPA_TD_LOSS_FRACTION)
    basis = "delivered" if delivered else "generated"
    return ConstantEmissionFactor(
        kg_per_kwh=kg_per_kwh, name=name,
        source=f"{source} [{lb_co2_per_mwh} lb CO2/MWh, {basis} basis]")


@dataclass(frozen=True)
class ComfortBand:

    tolerance_c: float
    source: str
    occupied_only: bool = True


@dataclass(frozen=True)
class HouseholdCase:

    tariff: Tariff
    emissions: EmissionFactor
    comfort: ComfortBand
    district_cap_kw: Optional[float]
    ev_penetration: Optional[float]
    hours_per_step: float = 1.0

    @property
    def provenance(self) -> Dict[str, str]:
        return {
            "tariff": f"{self.tariff.name} -- {self.tariff.source}",
            "emissions": f"{self.emissions.name} -- {self.emissions.source}",
            "comfort": f"tolerance {self.comfort.tolerance_c} degC, "
                       f"occupied_only={self.comfort.occupied_only} -- "
                       f"{self.comfort.source}",
            "district_cap_kw": repr(self.district_cap_kw),
            "ev_penetration": repr(self.ev_penetration),
        }


@dataclass
class HouseholdBenefit:

    bill_usd: np.ndarray
    bill_components: Dict[str, np.ndarray]
    co2_kg: np.ndarray
    import_kwh: np.ndarray
    export_kwh: np.ndarray
    gross_load_kwh: Optional[np.ndarray]
    pv_kwh: Optional[np.ndarray]
    comfort_hours_in_band: np.ndarray
    comfort_hours_assessed: np.ndarray
    own_peak_kw: np.ndarray
    coincident_peak_kw: np.ndarray
    district_peak_kw: float
    district_peak_step: int
    cap_exceedance_kwh: Optional[float]
    steps: int
    hours_per_step: float
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def comfort_fraction_in_band(self) -> np.ndarray:
        assessed = np.maximum(self.comfort_hours_assessed, 1e-12)
        return self.comfort_hours_in_band / assessed

    def annualised(self) -> Dict[str, np.ndarray]:
        hours = self.steps * self.hours_per_step
        k = 8760.0 / hours
        return {"bill_usd": self.bill_usd * k, "co2_kg": self.co2_kg * k,
                "import_kwh": self.import_kwh * k, "scale_factor": np.float64(k)}


def household_benefit(
    net_kw: np.ndarray,
    month: np.ndarray,
    hour: np.ndarray,
    indoor_c: np.ndarray,
    cooling_set_c: np.ndarray,
    heating_set_c: np.ndarray,
    occupant_count: np.ndarray,
    case: HouseholdCase,
    pv_kw: Optional[np.ndarray] = None,
) -> HouseholdBenefit:
    net = np.asarray(net_kw, dtype=np.float64)
    if net.ndim != 2:
        raise ValueError(f"net_kw must be (T, B), got shape {net.shape}")
    T, B = net.shape
    h = float(case.hours_per_step)

    imp_kw = np.maximum(net, 0.0)
    exp_kw = np.maximum(-net, 0.0)
    import_kwh = imp_kw * h

    components = case.tariff.bill_usd(import_kwh, np.asarray(month), np.asarray(hour))
    co2 = case.emissions.co2_kg(import_kwh)

    theta = np.asarray(indoor_c, dtype=np.float64)
    lo = np.asarray(heating_set_c, dtype=np.float64) - case.comfort.tolerance_c
    hi = np.asarray(cooling_set_c, dtype=np.float64) + case.comfort.tolerance_c
    for nm, arr in (("indoor_c", theta), ("cooling_set_c", hi), ("heating_set_c", lo)):
        if arr.shape != (T, B):
            raise ValueError(f"{nm} must be ({T}, {B}), got {arr.shape}")
    in_band = (theta >= lo) & (theta <= hi)
    if case.comfort.occupied_only:
        assessed = np.asarray(occupant_count, dtype=np.float64) > 0.0
    else:
        assessed = np.ones((T, B), dtype=bool)
    comfort_in = (in_band & assessed).sum(axis=0) * h
    comfort_assessed = assessed.sum(axis=0) * h

    district = imp_kw.sum(axis=1)
    peak_step = int(np.argmax(district))
    own_peak = imp_kw.max(axis=0)
    coincident = imp_kw[peak_step, :].copy()

    if case.district_cap_kw is None:
        exceed = None
    else:
        exceed = float(
            np.maximum(district - float(case.district_cap_kw), 0.0).sum() * h)

    if pv_kw is None:
        gross_kwh = pv_total = None
    else:
        pv = np.abs(np.asarray(pv_kw, dtype=np.float64))
        if pv.shape != (T, B):
            raise ValueError(f"pv_kw must be ({T}, {B}), got {pv.shape}")
        gross_kwh = ((net + pv) * h).sum(axis=0)
        pv_total = (pv * h).sum(axis=0)

    return HouseholdBenefit(
        bill_usd=components["total_usd"], bill_components=components, co2_kg=co2,
        import_kwh=import_kwh.sum(axis=0), export_kwh=(exp_kw * h).sum(axis=0),
        gross_load_kwh=gross_kwh, pv_kwh=pv_total,
        comfort_hours_in_band=comfort_in, comfort_hours_assessed=comfort_assessed,
        own_peak_kw=own_peak, coincident_peak_kw=coincident,
        district_peak_kw=float(district[peak_step]), district_peak_step=peak_step,
        cap_exceedance_kwh=exceed, steps=T, hours_per_step=h,
        provenance=case.provenance)


def benefit_delta(
    treatment: HouseholdBenefit, reference: HouseholdBenefit
) -> Dict[str, np.ndarray]:
    if treatment.bill_usd.shape != reference.bill_usd.shape:
        raise ValueError(
            f"treatment covers {treatment.bill_usd.shape[0]} households, reference "
            f"{reference.bill_usd.shape[0]}")
    if treatment.steps != reference.steps:
        raise ValueError(
            f"treatment ran {treatment.steps} steps, reference {reference.steps}; "
            "a benefit delta between different windows is not meaningful")
    out = {
        "bill_saved_usd": reference.bill_usd - treatment.bill_usd,
        "bill_saved_frac": np.where(
            reference.bill_usd > 0.0,
            (reference.bill_usd - treatment.bill_usd)
            / np.maximum(reference.bill_usd, 1e-12), np.nan),
        "co2_avoided_kg": reference.co2_kg - treatment.co2_kg,
        "import_avoided_kwh": reference.import_kwh - treatment.import_kwh,
        "export_forgone_kwh": reference.export_kwh - treatment.export_kwh,
        "comfort_hours_gained": (treatment.comfort_hours_in_band
                                 - reference.comfort_hours_in_band),
        "own_peak_cut_kw": reference.own_peak_kw - treatment.own_peak_kw,
        "coincident_peak_cut_kw": (reference.coincident_peak_kw
                                   - treatment.coincident_peak_kw),
    }
    if treatment.gross_load_kwh is not None and reference.gross_load_kwh is not None:
        out["gross_load_saved_kwh"] = (reference.gross_load_kwh
                                       - treatment.gross_load_kwh)
    return out


CITYLEARN_2022_TARIFF_NAME = "citylearn_challenge_2022_phase_all"
CITYLEARN_2022_TARIFF_SOURCE = (
    "pricing.csv shipped with the CityLearn citylearn_challenge_2022_phase_all dataset "
    "(Nweye, Sankaranarayanan & Nagy 2023, doi:10.18738/T8/0YLJ6Q). Time-of-use, five "
    "distinct prices spanning 0.21-0.54 USD/kWh, peak 16:00-20:00 local. The "
    "neighbourhood is in Fontana, California (Nweye et al., NeurIPS 2022 Competition "
    "Track, PMLR 220:85-103), NOT Travis County, Texas."
)

CITYLEARN_2022_CARBON_SOURCE = (
    "carbon_intensity.csv shipped with citylearn_challenge_2022_phase_all "
    "(doi:10.18738/T8/0YLJ6Q); hourly, mean 0.1565 kg CO2/kWh over 8760 h. California "
    "grid, NOT ERCOT."
)


_AE_TARIFF_SOURCE = (
    "City of Austin Electric Tariff effective 2025-11-01 (FY2026), "
    "https://austinenergy.com/-/media/project/websites/shared/pdfs/rates/tariff.pdf, "
    "retrieved 2026-10-06")

AE_RES_CUSTOMER_CHARGE_USD_PER_MONTH = 16.50
AE_RES_TIER_UPPER_KWH = (300.0, 900.0, 2000.0)
AE_RES_TIER_RATE_USD_PER_KWH = (0.04640, 0.05138, 0.07525, 0.10884)

AE_RES_PSA_USD_PER_KWH = 0.04118

AE_RES_CBC_USD_PER_KWH = 0.00564 + 0.00254 + 0.00457

AE_REGULATORY_SECONDARY_USD_PER_KWH = 0.01338

AE_RES_FLAT_RIDERS_USD_PER_KWH = (
    AE_RES_PSA_USD_PER_KWH + AE_RES_CBC_USD_PER_KWH
    + AE_REGULATORY_SECONDARY_USD_PER_KWH)

AE_TOU_OFF_PEAK_USD_PER_KWH = 0.02677
AE_TOU_MID_PEAK_USD_PER_KWH = 0.04118
AE_TOU_ON_PEAK_USD_PER_KWH = 0.08442

_AE_TOU_ON_PEAK_HOURS = frozenset({16, 17, 18})
_AE_TOU_MID_PEAK_HOURS = frozenset(range(8, 16)) | frozenset({19, 20, 21, 22})
_AE_WEEKEND_DAY_TYPES = frozenset({6, 7})


def austin_tou_psa_vector(hour: np.ndarray, day_type: np.ndarray) -> np.ndarray:
    h = np.asarray(hour, dtype=np.int64)
    dt_ = np.asarray(day_type, dtype=np.int64)
    if h.shape != dt_.shape:
        raise ValueError(f"hour {h.shape} and day_type {dt_.shape} must match")
    bad = sorted(set(np.unique(h).tolist()) - set(range(1, 25)))
    if bad:
        raise ValueError(f"hour must be 1..24 (CityLearn convention), saw {bad}")
    bad = sorted(set(np.unique(dt_).tolist()) - set(range(1, 8)))
    if bad:
        raise ValueError(f"day_type must be 1..7 (1=Monday), saw {bad}")
    psa = np.full(h.shape, AE_TOU_OFF_PEAK_USD_PER_KWH, dtype=np.float64)
    weekday = ~np.isin(dt_, list(_AE_WEEKEND_DAY_TYPES))
    psa[weekday & np.isin(h, list(_AE_TOU_MID_PEAK_HOURS))] = \
        AE_TOU_MID_PEAK_USD_PER_KWH
    psa[weekday & np.isin(h, list(_AE_TOU_ON_PEAK_HOURS))] = \
        AE_TOU_ON_PEAK_USD_PER_KWH
    return psa


def austin_energy_residential() -> TieredMonthlyTariff:
    return TieredMonthlyTariff(
        customer_charge_usd_per_month=AE_RES_CUSTOMER_CHARGE_USD_PER_MONTH,
        tier_upper_kwh=AE_RES_TIER_UPPER_KWH,
        tier_rate_usd_per_kwh=AE_RES_TIER_RATE_USD_PER_KWH,
        rider_usd_per_kwh=AE_RES_FLAT_RIDERS_USD_PER_KWH,
        name="austin_energy_residential_inside_city",
        source=(f"{_AE_TARIFF_SOURCE}; Residential Service Inside City Limits (p. 4), "
                f"Regulatory Charge Secondary energy basis (p. 27). Customer charge "
                f"16.50 USD/mo; tiers 4.640/5.138/7.525/10.884 c/kWh at 300/900/2000 "
                f"kWh; flat riders PSA 4.118 + CBC 1.275 + Regulatory 1.338 c/kWh. "
                f"No time-of-use component."),
    )


def austin_energy_residential_tou_pilot(
    hour: np.ndarray, day_type: np.ndarray
) -> TieredMonthlyTariff:
    psa = austin_tou_psa_vector(hour, day_type)
    rider = psa + AE_RES_CBC_USD_PER_KWH + AE_REGULATORY_SECONDARY_USD_PER_KWH
    return TieredMonthlyTariff(
        customer_charge_usd_per_month=AE_RES_CUSTOMER_CHARGE_USD_PER_MONTH,
        tier_upper_kwh=AE_RES_TIER_UPPER_KWH,
        tier_rate_usd_per_kwh=AE_RES_TIER_RATE_USD_PER_KWH,
        rider_usd_per_kwh=rider,
        name="austin_energy_residential_tou_pilot",
        source=(f"{_AE_TARIFF_SOURCE}; Residential Service Inside City Limits (p. 4) "
                f"with the Pilot Programs Residential Time-of-Use power supply charges "
                f"(p. 37) in lieu of the standard PSA: off-peak 2.677, mid-peak 4.118, "
                f"on-peak 8.442 c/kWh; weekday on-peak 15:00-18:00, mid-peak "
                f"07:00-15:00 and 18:00-22:00, off-peak 22:00-07:00, weekends off-peak "
                f"all day. Pilot limited to 100 individual meters (p. 36)."),
    )


AE_VOS_NON_DEMAND_USD_PER_KWH = 0.0761 + 0.023


def value_of_solar_credit_usd(pv_output_kwh: np.ndarray) -> np.ndarray:
    pv = np.asarray(pv_output_kwh, dtype=np.float64)
    if pv.ndim != 2:
        raise ValueError(f"pv_output_kwh must be (T, B), got {pv.shape}")
    return AE_VOS_NON_DEMAND_USD_PER_KWH * pv.sum(axis=0)


def value_of_solar_bill_usd(
    tariff: Tariff,
    gross_load_kwh: np.ndarray,
    pv_output_kwh: np.ndarray,
    month: np.ndarray,
    hour: np.ndarray,
) -> Dict[str, np.ndarray]:
    gross = np.asarray(gross_load_kwh, dtype=np.float64)
    pv = np.abs(np.asarray(pv_output_kwh, dtype=np.float64))
    if gross.shape != pv.shape:
        raise ValueError(
            f"gross_load_kwh {gross.shape} and pv_output_kwh {pv.shape} must match")
    charges = tariff.bill_usd(gross, month, hour)
    credit = value_of_solar_credit_usd(pv)
    out = {k: v.copy() for k, v in charges.items()}
    out["vos_credit_usd"] = credit
    out["net_usd"] = charges["total_usd"] - credit
    return out


def egrid_erct_delivered() -> ConstantEmissionFactor:
    return egrid_constant_factor(
        lb_co2_per_mwh=771.1, name="egrid2022_erct_delivered",
        source=("US EPA, Greenhouse Gas Equivalencies Calculator - Calculations and "
                "References, eGRID Regions table, ERCT 'ERCOT All' total output "
                "(baseload) rate, eGRID2022 (year 2022 data)"),
        delivered=True)


def citylearn_2022_tariff(price_usd_per_kwh: Sequence[float]) -> HourlyEnergyTariff:
    return HourlyEnergyTariff(
        price_usd_per_kwh=np.asarray(price_usd_per_kwh, dtype=np.float64),
        name=CITYLEARN_2022_TARIFF_NAME, source=CITYLEARN_2022_TARIFF_SOURCE)


def citylearn_2022_emissions(kg_per_kwh: Sequence[float]) -> HourlyEmissionFactor:
    return HourlyEmissionFactor(
        kg_per_kwh=np.asarray(kg_per_kwh, dtype=np.float64),
        name="citylearn_challenge_2022_phase_all",
        source=CITYLEARN_2022_CARBON_SOURCE)
