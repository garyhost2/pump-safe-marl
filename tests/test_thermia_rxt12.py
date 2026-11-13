import math

import numpy as np
import pytest

from stems.thermia_rxt12 import CalibraRXT12, fit, fit_report, load_data


@pytest.fixture(scope="module")
def hp():
    return CalibraRXT12()


@pytest.fixture(scope="module")
def report():
    return fit_report()


def test_data_file_is_the_certified_block():
    d = load_data()
    assert d["certificate"]["registration"] == "012-C700403 / Issue 1"
    assert len(d["operating_points"]) == 21
    pts = {(p["source"], p["t_supply_c"], p["q_kw"]): p["cop"] for p in d["operating_points"]}
    assert pts[("brine", 35, 6.20)] == 4.87
    assert pts[("brine", 55, 10.23)] == 2.93
    assert pts[("water", 35, 13.85)] == 4.73


def test_fit_quality_matches_the_documented_numbers(report):
    assert report["n_points"] == 21
    assert report["in_sample_rms"] < 0.0260 and report["in_sample_max"] < 0.0480
    assert report["loo_rms"] < 0.0420 and report["loo_max"] < 0.0910
    assert report["brine_loo_rms"] < 0.0330 and report["brine_loo_max"] < 0.0600


def test_every_brine_point_reproduced_within_five_percent(hp):
    for p in load_data()["operating_points"]:
        if p["source"] != "brine":
            continue
        c = float(hp.cop(p["q_kw"], p["t_source_c"], p["t_supply_c"]))
        assert abs(c - p["cop"]) / p["cop"] < 0.05, p


def test_fit_is_deterministic():
    a, b = fit(), fit()
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-9, abs=1e-12)


def test_cop_falls_with_supply_and_rises_with_source(hp):
    q = 6.0
    sup = np.arange(25.0, 70.0, 1.0)
    c = hp.cop(q, 0.0, sup)
    assert np.all(np.diff(c) < 0), "COP must fall as the supply temperature rises"
    src = np.arange(-5.0, 10.0, 1.0)
    c2 = hp.cop(q, src, 45.0)
    assert np.all(np.diff(c2) > 0), "COP must rise as the source temperature rises"


def test_cop_physically_bounded_over_the_operating_range(hp):
    K = 273.15
    for ts in (-5.0, 0.0, 5.0, 10.0):
        for tw in (25.0, 35.0, 45.0, 55.0, 60.0, 65.0, 70.0):
            for q in np.linspace(float(hp.q_min(ts)), hp.q_max(ts, tw), 6):
                c = float(hp.cop(q, ts, tw))
                carnot = (tw + K) / (tw - ts)
                assert 1.0 < c < carnot, (ts, tw, q, c, carnot)


def test_full_speed_penalty_is_present(hp):
    qmax = hp.q_max(0.0, 35.0)
    assert float(hp.cop(qmax, 0.0, 35.0)) < float(hp.cop(0.6 * qmax, 0.0, 35.0))


def test_electric_input_never_exceeds_datasheet_limit(hp):
    for ts in (-5.0, 0.0, 10.0):
        for tw in (35.0, 55.0, 65.0, 70.0):
            qm = hp.q_max(ts, tw)
            assert qm / float(hp.cop(qm, ts, tw)) <= hp.p_el_max + 1e-6


def test_no_compressor_output_above_70c(hp):
    assert hp.q_max(0.0, 70.5) == 0.0
    op = hp.operate(5.0, 0.0, 72.0)
    assert op.q_compressor_kw == 0.0 and op.regime == "above_limit" and op.p_aux_kw == pytest.approx(5.0)


def test_extrapolation_flag_marks_the_uncertified_disinfection_range(hp):
    assert not hp.operate(6.0, 0.0, 45.0).extrapolated
    assert hp.operate(6.0, 0.0, 60.0).extrapolated
    assert hp.operate(6.0, -4.0, 35.0).extrapolated


def test_energy_balance_and_regimes(hp):
    off = hp.operate(0.0, 0.0, 45.0)
    assert off.regime == "off" and off.q_total_kw == 0.0 and off.p_total_kw == pytest.approx(0.017)
    mod = hp.operate(6.0, 0.0, 45.0)
    assert mod.regime == "modulating" and mod.q_compressor_kw == pytest.approx(mod.cop * mod.p_compressor_kw)
    cyc = hp.operate(1.0, 0.0, 45.0)
    assert cyc.regime == "cycling" and cyc.q_compressor_kw == pytest.approx(1.0) and 0 < cyc.duty < 1
    assert cyc.cop < float(hp.cop(float(hp.q_min(0.0)), 0.0, 45.0))
    big = hp.operate(15.0, 0.0, 45.0)
    assert big.regime == "full+aux" and big.q_total_kw == pytest.approx(15.0)
    assert big.aux_step_kw in (3.0, 6.0, 9.0) and big.p_aux_kw == pytest.approx(big.q_aux_kw)


def test_capacity_range_consistent_with_datasheet(hp):
    assert 2.5 < float(hp.q_min(0.0)) < 3.6
    assert 10.5 < hp.q_max(0.0, 35.0) < 12.5


def test_en16147_heat_up_is_within_capacity(hp):
    dhw = load_data()["dhw_en16147_xl"]
    e_kwh = dhw["tank_l"] * 4.18 * (dhw["reference_hot_water_temp_c"] - 10.0) / 3600.0
    p_mean = e_kwh / dhw["heat_up_time_h"]
    assert float(hp.q_min(0.0)) < p_mean < hp.q_max(0.0, 55.0)
    assert not math.isnan(p_mean)
