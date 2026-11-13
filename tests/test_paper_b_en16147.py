import numpy as np
import pytest

from experiments.paper_b.en16147_check import CERT, XL, XL_KWH, run, xl_minutes


def test_xl_profile_sums_to_the_reference_energy():
    assert len(XL) == 30
    assert sum(e for _, e, _ in XL) == pytest.approx(XL_KWH, abs=1e-9)
    d = xl_minutes()
    assert d.shape == (24, 60) and d.sum() == pytest.approx(XL_KWH, abs=1e-9)
    assert d[:7].sum() == 0.0 and d[22:].sum() == 0.0
    assert d.max() <= 10.0 * 4.18 / 3600.0 * 30.0 + 1e-12


def test_tapping_cycle_runs_and_balances():
    r = run(days=5, skip=2)
    assert r["tapped_kwh_per_day"] + r["unmet_kwh_per_day"] == pytest.approx(XL_KWH, abs=1e-6)
    bal = r["heat_kwh_per_day"] - r["tapped_kwh_per_day"] - r["store_loss_kwh_per_day"] - r["store_energy_change_kwh"] / r["days_measured"]
    assert abs(bal) < 1e-6
    assert 2.0 < r["cop_dhw"] < r["cop_charge_B0"]
    assert abs(r["cop_dhw"] / CERT["cop_dhw"] - 1.0) < 0.15


def test_standby_test_runs():
    r = run(tapping=False, days=5, skip=2)
    assert 17.0 < r["mean_power_w"] < 60.0
    assert "cop_dhw" not in r
