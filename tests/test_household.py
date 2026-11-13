from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from stems.household import (
    AE_REGULATORY_SECONDARY_USD_PER_KWH,
    AE_RES_CBC_USD_PER_KWH,
    AE_RES_FLAT_RIDERS_USD_PER_KWH,
    AE_RES_PSA_USD_PER_KWH,
    AE_TOU_MID_PEAK_USD_PER_KWH,
    AE_TOU_OFF_PEAK_USD_PER_KWH,
    AE_TOU_ON_PEAK_USD_PER_KWH,
    AE_VOS_NON_DEMAND_USD_PER_KWH,
    ComfortBand,
    ConstantEmissionFactor,
    HourlyEmissionFactor,
    HourlyEnergyTariff,
    HouseholdCase,
    TieredMonthlyTariff,
    austin_energy_residential,
    austin_energy_residential_tou_pilot,
    austin_tou_psa_vector,
    benefit_delta,
    egrid_constant_factor,
    egrid_erct_delivered,
    household_benefit,
    value_of_solar_credit_usd,
)

REPO = Path(__file__).resolve().parents[1]


def _case(tariff, emissions, cap=None, pen=None, tol=1.0):
    return HouseholdCase(
        tariff=tariff, emissions=emissions,
        comfort=ComfortBand(tolerance_c=tol, source="test"),
        district_cap_kw=cap, ev_penetration=pen)


def test_tiered_tariff_charges_each_block_at_its_own_rate():
    t = austin_energy_residential()
    E = np.full((1, 1), 1000.0)
    out = t.bill_usd(E, np.ones(1, dtype=int), np.ones(1, dtype=int))
    expected_energy = 300 * 0.04640 + 600 * 0.05138 + 100 * 0.07525
    assert out["energy_usd"][0] == pytest.approx(expected_energy, abs=1e-9)
    assert out["fixed_usd"][0] == pytest.approx(16.50, abs=1e-9)
    assert out["adders_usd"][0] == pytest.approx(
        1000.0 * AE_RES_FLAT_RIDERS_USD_PER_KWH, abs=1e-9)
    assert out["total_usd"][0] == pytest.approx(
        expected_energy + 16.50 + 1000.0 * AE_RES_FLAT_RIDERS_USD_PER_KWH, abs=1e-9)


def test_tiers_are_monthly_not_annual():
    t = austin_energy_residential()
    month = np.repeat(np.arange(1, 13), 1)
    E = np.full((12, 1), 300.0)
    out = t.bill_usd(E, month, np.ones(12, dtype=int))
    assert out["energy_usd"][0] == pytest.approx(12 * 300 * 0.04640, abs=1e-9)
    assert out["fixed_usd"][0] == pytest.approx(12 * 16.50, abs=1e-9)


def test_tier_boundary_is_exact_at_300_kwh():
    t = austin_energy_residential()
    at = t.bill_usd(np.full((1, 1), 300.0), np.ones(1, dtype=int),
                    np.ones(1, dtype=int))["energy_usd"][0]
    just_over = t.bill_usd(np.full((1, 1), 301.0), np.ones(1, dtype=int),
                           np.ones(1, dtype=int))["energy_usd"][0]
    assert at == pytest.approx(300 * 0.04640, abs=1e-9)
    assert just_over - at == pytest.approx(0.05138, abs=1e-9)


def test_tier_rate_count_mismatch_is_rejected():
    with pytest.raises(ValueError, match="need 4 rates"):
        TieredMonthlyTariff(
            customer_charge_usd_per_month=1.0, tier_upper_kwh=(1.0, 2.0, 3.0),
            tier_rate_usd_per_kwh=(0.1, 0.2), rider_usd_per_kwh=0.0,
            name="bad", source="test")


def test_non_monotone_tier_bounds_are_rejected():
    with pytest.raises(ValueError, match="strictly increasing"):
        TieredMonthlyTariff(
            customer_charge_usd_per_month=1.0, tier_upper_kwh=(300.0, 200.0),
            tier_rate_usd_per_kwh=(0.1, 0.2, 0.3), rider_usd_per_kwh=0.0,
            name="bad", source="test")


def test_standard_austin_tariff_is_indifferent_to_when_energy_is_used():
    t = austin_energy_residential()
    month = np.ones(24, dtype=int)
    hour = np.arange(1, 25)
    peaky = np.zeros((24, 1)); peaky[16:21, 0] = 20.0
    flat = np.full((24, 1), 100.0 / 24.0)
    a = t.bill_usd(peaky, month, hour)["total_usd"][0]
    b = t.bill_usd(flat, month, hour)["total_usd"][0]
    assert a == pytest.approx(b, abs=1e-9)
    assert not t.time_varying


def test_tou_pilot_tariff_does_reward_shifting():
    hour = np.arange(1, 25)
    day_type = np.ones(24, dtype=int)
    t = austin_energy_residential_tou_pilot(hour, day_type)
    assert t.time_varying
    month = np.ones(24, dtype=int)
    on_peak = np.zeros((24, 1)); on_peak[15:18, 0] = 100.0 / 3.0
    off_peak = np.zeros((24, 1)); off_peak[0:3, 0] = 100.0 / 3.0
    a = t.bill_usd(on_peak, month, hour)["total_usd"][0]
    b = t.bill_usd(off_peak, month, hour)["total_usd"][0]
    saved = a - b
    assert saved == pytest.approx(
        100.0 * (AE_TOU_ON_PEAK_USD_PER_KWH - AE_TOU_OFF_PEAK_USD_PER_KWH), abs=1e-9)
    assert saved > 0.0


def test_tou_periods_partition_the_week_with_the_published_durations():
    hour = np.tile(np.arange(1, 25), 7)
    day_type = np.repeat(np.arange(1, 8), 24)
    psa = austin_tou_psa_vector(hour, day_type)
    weekday = day_type <= 5
    assert int((psa == AE_TOU_ON_PEAK_USD_PER_KWH).sum()) == 3 * 5
    assert int((psa == AE_TOU_MID_PEAK_USD_PER_KWH).sum()) == 12 * 5
    assert int((psa == AE_TOU_OFF_PEAK_USD_PER_KWH).sum()) == 9 * 5 + 48
    assert int((~weekday).sum()) == 48
    assert np.all(psa[~weekday] == AE_TOU_OFF_PEAK_USD_PER_KWH)


def test_mid_peak_equals_the_standard_psa():
    assert AE_TOU_MID_PEAK_USD_PER_KWH == pytest.approx(AE_RES_PSA_USD_PER_KWH)


def test_tou_vector_rejects_out_of_range_calendar_columns():
    with pytest.raises(ValueError, match="hour must be 1..24"):
        austin_tou_psa_vector(np.array([0]), np.array([1]))
    with pytest.raises(ValueError, match="day_type must be 1..7"):
        austin_tou_psa_vector(np.array([1]), np.array([0]))


def test_tou_hour_mapping_matches_the_clock_intervals():
    psa = austin_tou_psa_vector(np.arange(1, 25), np.ones(24, dtype=int))
    on = set((np.arange(1, 25)[psa == AE_TOU_ON_PEAK_USD_PER_KWH]).tolist())
    assert on == {16, 17, 18}
    off = set((np.arange(1, 25)[psa == AE_TOU_OFF_PEAK_USD_PER_KWH]).tolist())
    assert off == {23, 24, 1, 2, 3, 4, 5, 6, 7}


def test_egrid_conversion_matches_a_hand_computation():
    ef = egrid_constant_factor(771.1, name="x", source="y", delivered=False)
    assert ef.kg_per_kwh == pytest.approx(771.1 * 0.45359237 / 1000.0, rel=1e-12)
    assert ef.kg_per_kwh == pytest.approx(0.3497651, abs=1e-7)
    delivered = egrid_erct_delivered()
    assert delivered.kg_per_kwh == pytest.approx(ef.kg_per_kwh / (1 - 0.051), rel=1e-12)
    assert delivered.kg_per_kwh == pytest.approx(0.3685617, abs=1e-7)


def test_erct_factor_is_more_than_double_the_shipped_california_series():
    ratio = egrid_erct_delivered().kg_per_kwh / 0.1565
    assert 2.3 < ratio < 2.5


def test_constant_factor_makes_co2_proportional_to_energy():
    ef = egrid_erct_delivered()
    peaky = np.zeros((24, 1)); peaky[16:21, 0] = 20.0
    flat = np.full((24, 1), 100.0 / 24.0)
    assert ef.co2_kg(peaky)[0] == pytest.approx(ef.co2_kg(flat)[0], abs=1e-9)


def test_hourly_factor_does_distinguish_when_energy_is_used():
    c = np.linspace(0.05, 0.30, 24)
    ef = HourlyEmissionFactor(kg_per_kwh=c, name="x", source="y")
    early = np.zeros((24, 1)); early[0, 0] = 10.0
    late = np.zeros((24, 1)); late[23, 0] = 10.0
    assert ef.co2_kg(late)[0] > ef.co2_kg(early)[0]


def test_emission_series_length_mismatch_is_rejected():
    ef = HourlyEmissionFactor(kg_per_kwh=np.ones(5), name="x", source="y")
    with pytest.raises(ValueError, match="expected \\(3,\\)"):
        ef.co2_kg(np.ones((3, 2)))


def test_value_of_solar_credits_gross_output_not_exports():
    pv = np.full((10, 2), 3.0)
    credit = value_of_solar_credit_usd(pv)
    assert credit[0] == pytest.approx(30.0 * AE_VOS_NON_DEMAND_USD_PER_KWH, abs=1e-9)
    assert credit[0] == pytest.approx(30.0 * 0.0991, abs=1e-9)


def test_import_only_billing_ignores_exports():
    t = HourlyEnergyTariff(price_usd_per_kwh=np.full(3, 0.2), name="x", source="y")
    case = _case(t, ConstantEmissionFactor(0.4, "x", "y"))
    net = np.array([[-5.0], [0.0], [5.0]])
    b = household_benefit(
        net, month=np.ones(3, dtype=int), hour=np.arange(1, 4),
        indoor_c=np.full((3, 1), 22.0), cooling_set_c=np.full((3, 1), 24.0),
        heating_set_c=np.full((3, 1), 20.0), occupant_count=np.ones((3, 1)), case=case)
    assert b.import_kwh[0] == pytest.approx(5.0)
    assert b.export_kwh[0] == pytest.approx(5.0)
    assert b.bill_usd[0] == pytest.approx(1.0)
    assert b.co2_kg[0] == pytest.approx(2.0)


def test_coincident_peak_is_measured_at_the_district_peak_not_the_household_peak():
    t = HourlyEnergyTariff(price_usd_per_kwh=np.zeros(2), name="x", source="y")
    case = _case(t, ConstantEmissionFactor(0.0, "x", "y"))
    net = np.array([[10.0, 1.0],
                    [2.0, 8.0]])
    b = household_benefit(
        net, month=np.ones(2, dtype=int), hour=np.arange(1, 3),
        indoor_c=np.full((2, 2), 22.0), cooling_set_c=np.full((2, 2), 24.0),
        heating_set_c=np.full((2, 2), 20.0), occupant_count=np.ones((2, 2)), case=case)
    assert b.district_peak_step == 0
    assert b.district_peak_kw == pytest.approx(11.0)
    assert b.own_peak_kw.tolist() == [10.0, 8.0]
    assert b.coincident_peak_kw.tolist() == [10.0, 1.0]
    assert b.own_peak_kw[1] > b.coincident_peak_kw[1]


def test_comfort_counts_only_occupied_hours_when_asked():
    t = HourlyEnergyTariff(price_usd_per_kwh=np.zeros(4), name="x", source="y")
    indoor = np.array([[22.0], [30.0], [22.0], [30.0]])
    occ = np.array([[1.0], [1.0], [0.0], [0.0]])
    kw = dict(month=np.ones(4, dtype=int), hour=np.arange(1, 5),
              indoor_c=indoor, cooling_set_c=np.full((4, 1), 24.0),
              heating_set_c=np.full((4, 1), 20.0), occupant_count=occ)
    occupied = household_benefit(
        np.zeros((4, 1)), case=_case(t, ConstantEmissionFactor(0.0, "x", "y"), tol=0.0),
        **kw)
    assert occupied.comfort_hours_assessed[0] == pytest.approx(2.0)
    assert occupied.comfort_hours_in_band[0] == pytest.approx(1.0)
    assert occupied.comfort_fraction_in_band[0] == pytest.approx(0.5)

    case_all = HouseholdCase(
        tariff=t, emissions=ConstantEmissionFactor(0.0, "x", "y"),
        comfort=ComfortBand(tolerance_c=0.0, source="test", occupied_only=False),
        district_cap_kw=None, ev_penetration=None)
    everyone = household_benefit(np.zeros((4, 1)), case=case_all, **kw)
    assert everyone.comfort_hours_assessed[0] == pytest.approx(4.0)
    assert everyone.comfort_hours_in_band[0] == pytest.approx(2.0)


def test_comfort_tolerance_widens_the_band_symmetrically():
    t = HourlyEnergyTariff(price_usd_per_kwh=np.zeros(2), name="x", source="y")
    kw = dict(month=np.ones(2, dtype=int), hour=np.arange(1, 3),
              indoor_c=np.array([[19.5], [24.5]]),
              cooling_set_c=np.full((2, 1), 24.0), heating_set_c=np.full((2, 1), 20.0),
              occupant_count=np.ones((2, 1)))
    tight = household_benefit(
        np.zeros((2, 1)), case=_case(t, ConstantEmissionFactor(0, "x", "y"), tol=0.0),
        **kw)
    loose = household_benefit(
        np.zeros((2, 1)), case=_case(t, ConstantEmissionFactor(0, "x", "y"), tol=1.0),
        **kw)
    assert tight.comfort_hours_in_band[0] == pytest.approx(0.0)
    assert loose.comfort_hours_in_band[0] == pytest.approx(2.0)


def test_cap_exceedance_is_energy_above_the_cap():
    t = HourlyEnergyTariff(price_usd_per_kwh=np.zeros(3), name="x", source="y")
    case = _case(t, ConstantEmissionFactor(0.0, "x", "y"), cap=10.0)
    net = np.array([[6.0, 6.0], [4.0, 4.0], [5.0, 8.0]])
    b = household_benefit(
        net, month=np.ones(3, dtype=int), hour=np.arange(1, 4),
        indoor_c=np.full((3, 2), 22.0), cooling_set_c=np.full((3, 2), 24.0),
        heating_set_c=np.full((3, 2), 20.0), occupant_count=np.ones((3, 2)), case=case)
    assert b.cap_exceedance_kwh == pytest.approx(2.0 + 3.0)


def test_benefit_delta_signs_favour_the_household():
    t = HourlyEnergyTariff(price_usd_per_kwh=np.full(2, 0.5), name="x", source="y")
    case = _case(t, ConstantEmissionFactor(0.4, "x", "y"))
    kw = dict(month=np.ones(2, dtype=int), hour=np.arange(1, 3),
              cooling_set_c=np.full((2, 1), 24.0), heating_set_c=np.full((2, 1), 20.0),
              occupant_count=np.ones((2, 1)))
    ref = household_benefit(np.array([[10.0], [10.0]]),
                            indoor_c=np.full((2, 1), 30.0), case=case, **kw)
    trt = household_benefit(np.array([[4.0], [4.0]]),
                            indoor_c=np.full((2, 1), 22.0), case=case, **kw)
    d = benefit_delta(trt, ref)
    assert d["bill_saved_usd"][0] == pytest.approx(6.0)
    assert d["bill_saved_frac"][0] == pytest.approx(0.6)
    assert d["co2_avoided_kg"][0] == pytest.approx(4.8)
    assert d["import_avoided_kwh"][0] == pytest.approx(12.0)
    assert d["comfort_hours_gained"][0] == pytest.approx(2.0)
    assert d["own_peak_cut_kw"][0] == pytest.approx(6.0)


def test_benefit_delta_refuses_windows_of_different_length():
    t2 = HourlyEnergyTariff(price_usd_per_kwh=np.zeros(2), name="x", source="y")
    t3 = HourlyEnergyTariff(price_usd_per_kwh=np.zeros(3), name="x", source="y")
    ef = ConstantEmissionFactor(0.0, "x", "y")
    def mk(T, tar):
        return household_benefit(
            np.zeros((T, 1)), month=np.ones(T, dtype=int), hour=np.arange(1, T + 1),
            indoor_c=np.full((T, 1), 22.0), cooling_set_c=np.full((T, 1), 24.0),
            heating_set_c=np.full((T, 1), 20.0), occupant_count=np.ones((T, 1)),
            case=_case(tar, ef))
    with pytest.raises(ValueError, match="different windows"):
        benefit_delta(mk(2, t2), mk(3, t3))


def test_annualisation_scale_factor_is_exact_for_a_full_year():
    t = HourlyEnergyTariff(price_usd_per_kwh=np.ones(8760), name="x", source="y")
    b = household_benefit(
        np.ones((8760, 1)), month=np.ones(8760, dtype=int),
        hour=np.tile(np.arange(1, 25), 365), indoor_c=np.full((8760, 1), 22.0),
        cooling_set_c=np.full((8760, 1), 24.0), heating_set_c=np.full((8760, 1), 20.0),
        occupant_count=np.ones((8760, 1)),
        case=_case(t, ConstantEmissionFactor(0.0, "x", "y")))
    assert float(b.annualised()["scale_factor"]) == pytest.approx(1.0)
    assert b.annualised()["bill_usd"][0] == pytest.approx(b.bill_usd[0])


def test_case_provenance_names_every_source():
    case = _case(austin_energy_residential(), egrid_erct_delivered(), cap=30.0, pen=0.5)
    p = case.provenance
    assert "austinenergy.com" in p["tariff"]
    assert "eGRID" in p["emissions"]
    assert "30.0" in p["district_cap_kw"]
    assert "0.5" in p["ev_penetration"]


def test_hourly_tariff_total_equals_the_metrics_cost_definition():
    rng = np.random.default_rng(0)
    net = rng.normal(2.0, 4.0, size=(96, 5))
    price = rng.uniform(0.1, 0.6, size=96)
    metrics_cost = float((np.maximum(net, 0.0) * price[:, None]).sum())
    t = HourlyEnergyTariff(price_usd_per_kwh=price, name="x", source="y")
    b = household_benefit(
        net, month=np.ones(96, dtype=int), hour=np.tile(np.arange(1, 25), 4),
        indoor_c=np.full((96, 5), 22.0), cooling_set_c=np.full((96, 5), 24.0),
        heating_set_c=np.full((96, 5), 20.0), occupant_count=np.ones((96, 5)),
        case=_case(t, ConstantEmissionFactor(0.0, "x", "y")))
    assert float(b.bill_usd.sum()) == pytest.approx(metrics_cost, rel=1e-12)


def test_price_vector_length_mismatch_is_rejected():
    t = HourlyEnergyTariff(price_usd_per_kwh=np.ones(5), name="x", source="y")
    with pytest.raises(ValueError, match="expected \\(3,\\)"):
        t.bill_usd(np.ones((3, 2)), np.ones(3, dtype=int), np.ones(3, dtype=int))


def _shipped_series():
    schema = json.loads(
        (REPO / "citylearn_schemas" / "tx_travis_8b" / "schema.json").read_text())
    included = [v for v in schema["buildings"].values() if v.get("include", True)]
    if not included or not included[0].get("pricing"):
        pytest.skip("tx_travis_8b schema has no pricing file wired")
    import pandas as pd
    pricing = Path(included[0]["pricing"])
    carbon = Path(included[0]["carbon_intensity"])
    if not pricing.exists() or not carbon.exists():
        pytest.skip("shipped pricing/carbon not present in the local CityLearn cache")
    return (pd.read_csv(pricing)["electricity_pricing"].to_numpy(),
            pd.read_csv(carbon).iloc[:, 0].to_numpy())


def test_shipped_tariff_is_several_times_the_real_local_rate():
    price, _ = _shipped_series()
    austin_tier1 = 0.04640 + AE_RES_FLAT_RIDERS_USD_PER_KWH
    assert austin_tier1 == pytest.approx(0.11371, abs=1e-5)
    assert price.min() == pytest.approx(0.21, abs=1e-6)
    assert price.max() == pytest.approx(0.54, abs=1e-6)
    assert price.mean() / austin_tier1 > 2.0


def test_shipped_tariff_peak_spread_exceeds_the_local_tou_pilot_spread():
    price, _ = _shipped_series()
    shipped_ratio = price.max() / price.min()
    pilot_on = 0.04640 + AE_TOU_ON_PEAK_USD_PER_KWH + AE_RES_CBC_USD_PER_KWH \
        + AE_REGULATORY_SECONDARY_USD_PER_KWH
    pilot_off = 0.04640 + AE_TOU_OFF_PEAK_USD_PER_KWH + AE_RES_CBC_USD_PER_KWH \
        + AE_REGULATORY_SECONDARY_USD_PER_KWH
    pilot_ratio = pilot_on / pilot_off
    assert shipped_ratio == pytest.approx(0.54 / 0.21, rel=1e-9)
    assert pilot_ratio == pytest.approx(0.15695 / 0.09930, rel=1e-3)
    assert shipped_ratio > pilot_ratio


def _bench(net, pv, price=0.2, T=None):
    T = net.shape[0] if T is None else T
    t = HourlyEnergyTariff(price_usd_per_kwh=np.full(T, price), name="x", source="y")
    return household_benefit(
        net, month=np.ones(T, dtype=int), hour=np.tile(np.arange(1, 25),
                                                       int(np.ceil(T / 24)))[:T],
        indoor_c=np.full(net.shape, 22.0), cooling_set_c=np.full(net.shape, 24.0),
        heating_set_c=np.full(net.shape, 20.0), occupant_count=np.ones(net.shape),
        case=_case(t, ConstantEmissionFactor(0.4, "x", "y")), pv_kw=pv)


def test_household_energy_balance_closes_exactly():
    rng = np.random.default_rng(1)
    pv = np.abs(rng.normal(2.0, 1.0, size=(240, 4)))
    net = rng.normal(1.0, 3.0, size=(240, 4))
    b = _bench(net, pv)
    lhs = b.import_kwh - b.export_kwh + b.pv_kwh
    assert np.allclose(lhs, b.gross_load_kwh, atol=1e-9)


def test_import_can_fall_while_gross_consumption_rises():
    pv = np.full((2, 1), 10.0)
    ref = _bench(np.array([[-8.0], [2.0]]), pv)
    trt = _bench(np.array([[-2.0], [1.0]]), pv)
    d = benefit_delta(trt, ref)
    assert d["import_avoided_kwh"][0] == pytest.approx(1.0)
    assert d["export_forgone_kwh"][0] == pytest.approx(6.0)
    assert d["gross_load_saved_kwh"][0] == pytest.approx(-5.0)
    assert d["import_avoided_kwh"][0] > 0.0 > d["gross_load_saved_kwh"][0]
    assert (d["import_avoided_kwh"][0] - d["export_forgone_kwh"][0]
            - d["gross_load_saved_kwh"][0]) == pytest.approx(0.0, abs=1e-9)


def test_delta_has_no_field_called_energy_saved():
    ref = _bench(np.ones((4, 1)), np.ones((4, 1)))
    trt = _bench(np.full((4, 1), 0.5), np.ones((4, 1)))
    assert "energy_saved_kwh" not in benefit_delta(trt, ref)


def test_value_of_solar_bill_is_blind_to_self_consumption():
    from stems.household import value_of_solar_bill_usd

    t = austin_energy_residential()
    month, hour = np.ones(2, dtype=int), np.array([1, 2])
    pv = np.full((2, 1), 5.0)
    gross = np.full((2, 1), 6.0)
    a = value_of_solar_bill_usd(t, gross, pv, month, hour)
    b = value_of_solar_bill_usd(t, gross, pv, month, hour)
    assert a["net_usd"][0] == pytest.approx(b["net_usd"][0])
    assert a["vos_credit_usd"][0] == pytest.approx(
        10.0 * AE_VOS_NON_DEMAND_USD_PER_KWH)


def test_value_of_solar_bill_penalises_round_trip_losses():
    from stems.household import value_of_solar_bill_usd

    t = austin_energy_residential()
    month, hour = np.ones(24, dtype=int), np.arange(1, 25)
    pv = np.full((24, 1), 2.0)
    no_battery = np.full((24, 1), 1.0)
    with_battery = np.full((24, 1), 1.0 + 10.0 / 24)
    a = value_of_solar_bill_usd(t, no_battery, pv, month, hour)["net_usd"][0]
    b = value_of_solar_bill_usd(t, with_battery, pv, month, hour)["net_usd"][0]
    assert b > a
    assert b - a == pytest.approx(10.0 * (0.04640 + AE_RES_FLAT_RIDERS_USD_PER_KWH),
                                  abs=1e-9)


def test_pv_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="pv_kw must be"):
        _bench(np.ones((4, 2)), np.ones((4, 3)))
