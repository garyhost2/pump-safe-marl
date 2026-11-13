from __future__ import annotations

import numpy as np
import pytest

from stems.stems_table import (COMFORT_HIGH_C, COMFORT_LOW_C, NORMALISED_KEYS,
                               normalise_to_baseline, stems_table_metrics)

KW = dict(soc_min=0.2, soc_max=0.8, p_building_max=10.0, hours_per_step=1.0)


def _rollout(T=48, B=2):
    net = np.zeros((T, B))
    net[:, 0] = 3.0
    net[:, 1] = 2.0
    net[5, 1] = -4.0
    price = np.full((T, B), 0.1)
    carbon = np.full((T, B), 0.5)
    t_in = np.full((T, B), 23.0)
    t_in[7, 0] = 27.5
    t_in[9, 1] = 18.0
    soc = np.full((T, B), 0.5)
    soc[11, 0] = 0.9
    return net, price, carbon, t_in, soc


def test_cost_and_emission_bill_imports_only_and_never_credit_exports():
    net, price, carbon, t_in, soc = _rollout()
    m = stems_table_metrics(net, price, carbon, t_in, soc, **KW)
    imported = np.maximum(net, 0.0)
    assert m["stems_cost"] == pytest.approx(float((imported * price).sum()))
    assert m["stems_emission"] == pytest.approx(float((imported * carbon).sum()))
    assert m["stems_cost"] > float((net * price).sum())


def test_avg_daily_peak_is_the_plain_mean_of_signed_daily_maxima():
    net, price, carbon, t_in, soc = _rollout()
    m = stems_table_metrics(net, price, carbon, t_in, soc, **KW)
    signed = net.sum(axis=1)
    expected = np.mean([signed[0:24].max(), signed[24:48].max()])
    assert m["stems_avg_daily_peak"] == pytest.approx(float(expected))


def test_avg_daily_peak_is_not_floored_at_zero():
    net = np.full((24, 1), -2.0)
    z = np.zeros((24, 1))
    m = stems_table_metrics(net, z + 0.1, z + 0.5, z + 23.0, z + 0.5, **KW)
    assert m["stems_avg_daily_peak"] == pytest.approx(-2.0)


def test_ramping_is_the_mean_absolute_step_change_of_the_district_series():
    net, price, carbon, t_in, soc = _rollout()
    m = stems_table_metrics(net, price, carbon, t_in, soc, **KW)
    assert m["stems_ramping_rate"] == pytest.approx(
        float(np.abs(np.diff(net.sum(axis=1))).mean()))


def test_without_set_points_the_published_discomfort_column_is_not_reported():
    net, price, carbon, t_in, soc = _rollout()
    m = stems_table_metrics(net, price, carbon, t_in, soc, **KW)
    assert m["stems_discomfort_proportion"] is None
    assert m["stems_discomfort_fixed_band"] == pytest.approx(2 / 96)
    assert (COMFORT_LOW_C, COMFORT_HIGH_C) == (20.0, 26.0)


def test_discomfort_is_the_occupied_set_point_band_of_the_paper():
    import numpy as np
    net, price, carbon, t_in, soc = _rollout()
    t_set = np.full_like(t_in, 24.0)
    occ = np.ones_like(t_in)
    t_in = np.full_like(t_in, 24.0)
    t_in[0, 0] = 30.0
    occ[1, :] = 0.0
    t_in[1, :] = 40.0
    m = stems_table_metrics(net, price, carbon, t_in, soc,
                            t_set=t_set, occupant=occ, **KW)
    n_occ = float((occ > 0).sum())
    assert m["stems_discomfort_proportion"] == pytest.approx(1 / n_occ)
    assert n_occ < t_in.size


def test_safety_is_reported_under_both_readings_and_they_differ():
    net, price, carbon, t_in, soc = _rollout()
    m = stems_table_metrics(net, price, carbon, t_in, soc, **KW)
    assert m["stems_safety_violations_soc_equipment"] == pytest.approx(100 * 1 / 96)
    assert m["stems_safety_violations_with_comfort"] is None


def test_the_with_comfort_reading_is_bounded_below_by_the_discomfort_proportion():
    import numpy as np
    net, price, carbon, t_in, soc = _rollout()
    t_set = np.full_like(t_in, 24.0)
    occ = np.ones_like(t_in)
    m = stems_table_metrics(net, price, carbon, t_in, soc,
                            t_set=t_set, occupant=occ, **KW)
    assert (m["stems_safety_violations_with_comfort"]
            >= 100.0 * m["stems_discomfort_proportion"] - 1e-9)


def test_normalisation_divides_only_the_five_metrics_stems_normalises():
    net, price, carbon, t_in, soc = _rollout()
    base = stems_table_metrics(net, price, carbon, t_in, soc, **KW)
    row = normalise_to_baseline(base, base)
    for k in NORMALISED_KEYS:
        assert row[k + "_norm"] == pytest.approx(1.0)
    assert "stems_discomfort_proportion_norm" not in row
    assert "stems_safety_violations_soc_equipment_norm" not in row


def test_the_published_table_reports_a_safety_rate_below_its_own_discomfort_rate():
    from experiments.diagnostics.stems_table_compare import PUBLISHED

    row = PUBLISHED["STEMS"]
    assert row["violations"] < row["discomfort"], (
        "if this ever stops holding the transcription has drifted from the paper")
    assert row["discomfort"] / row["violations"] > 2.0


def test_the_published_rule_based_row_is_the_most_comfortable_in_the_table():
    from experiments.diagnostics.stems_table_compare import PUBLISHED

    rule = PUBLISHED["Rule-Based"]["discomfort"]
    others = {k: v["discomfort"] for k, v in PUBLISHED.items() if k != "Rule-Based"}
    assert all(v > rule for v in others.values()), others
    assert min(others.values()) == PUBLISHED["STEMS"]["discomfort"]


def test_a_non_finite_indoor_temperature_is_not_scored_as_perfect_comfort():
    import numpy as np
    from stems.stems_table import stems_table_metrics

    T, B = 48, 3
    net = np.full((T, B), 1.0)
    base = dict(net=net, price=np.full((T, B), 0.2), carbon=np.full((T, B), 0.3),
                soc=np.full((T, B), 0.5), soc_min=0.1, soc_max=0.9,
                p_building_max=1e6)

    occ = np.ones((T, B))
    t_set = np.full((T, B), 24.0)
    good = stems_table_metrics(t_in=np.full((T, B), 23.0),
                               t_set=t_set, occupant=occ, **base)
    assert good["stems_discomfort_proportion"] == pytest.approx(0.0)
    assert good["stems_safety_violations_with_comfort"] is not None

    t_bad = np.full((T, B), 23.0)
    t_bad[5, 1] = np.nan
    bad = stems_table_metrics(t_in=t_bad, t_set=t_set, occupant=occ, **base)
    assert bad["stems_discomfort_proportion"] is None, (
        "a NaN indoor temperature must not be reported as a comfort score")
    assert bad["stems_safety_violations_with_comfort"] is None
    for k in ("stems_cost", "stems_avg_daily_peak", "stems_soc_violations_pct"):
        assert bad[k] == pytest.approx(good[k])
