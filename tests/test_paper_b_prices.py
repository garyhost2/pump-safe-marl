import json
from pathlib import Path

import numpy as np
import pytest

from stems.dhw_v2.data import PRICE_AREAS, PRICE_DIRS, apply_price_area, load_area_spot

from stems.paths import SWEDEN_PACKAGES
PKG = SWEDEN_PACKAGES / "karlstad_2025"
needs_pkg = pytest.mark.skipif(not PKG.is_dir(), reason="Swedish site-year packages not on this machine")
needs_all = pytest.mark.skipif(not all(any((d / f"price_{a}.csv").is_file() for d in PRICE_DIRS) for a in PRICE_AREAS),
                               reason="price_SE1..SE4.csv not on this machine")


@needs_all
def test_all_four_areas_load_and_cover_2022_2025():
    for a in PRICE_AREAS:
        t, sp, bad = load_area_spot(a)
        assert len(t) == 35064 and str(t[0]) == "2022-01-01T00:00:00" and str(t[-1]) == "2025-12-31T23:00:00"
        assert np.all(np.diff(t).astype(int) == 3600)
        assert np.isfinite(sp).all()
        assert bad.sum() == {"SE1": 0, "SE2": 5, "SE3": 0, "SE4": 1}[a]


def test_qc_json_has_se2_se4_like_se1_se3():
    f = next(d / "price_qc.json" for d in PRICE_DIRS if (d / "price_qc.json").is_file())
    qc = json.loads(f.read_text())
    for z in ("SE2", "SE4"):
        for k in ("raw_intervals", "missing_hours", "provider_dst_end_label_artifacts",
                  "hours_with_wrong_interval_count", "per_year"):
            assert k in qc[z]
        assert [qc[z]["per_year"][y]["hours"] for y in ("2022", "2023", "2024", "2025")] == [8760, 8760, 8784, 8760]
        assert qc[z]["provider_dst_end_label_artifacts"] == 4
    assert qc["SE2"]["missing_hours"] == 5 and qc["SE4"]["missing_hours"] == 1
    assert len(qc["SE2"]["provider_short_days_non_dst"]) == 4


def test_unknown_area_is_refused():
    with pytest.raises(ValueError):
        load_area_spot("SE5")


@needs_pkg
@needs_all
def test_price_area_swap_changes_only_the_spot_term():
    from stems.dhw_v2.data import load_package
    d = load_package(PKG, houses=["H01"])
    assert d.meta["location"]["bidding_zone"] == "SE3"
    same = apply_price_area(d, "SE3")
    assert np.array_equal(same.price, d.price) and same.meta["price_area"]["area"] == "SE3"
    d4 = apply_price_area(d, "SE4")
    t, s3, _ = load_area_spot("SE3")
    _, s4, _ = load_area_spot("SE4")
    i = np.searchsorted(t, d.time_utc)
    assert np.allclose(d4.price - d.price, 1.25 * (s4[i] - s3[i]), atol=1e-9)
    assert d4.meta["price_area"]["filled_spot_hours"] == 1
    assert d4.price.mean() > d.price.mean()
    assert d.price is not d4.price and d4.name.endswith("@SE4")
    import csv
    f = next(p / "price_SE3.csv" for p in PRICE_DIRS if (p / "price_SE3.csv").is_file())
    with open(f) as fh:
        rd = csv.DictReader(fh)
        retail = np.array([float(r["retail_total_SEK_per_kWh"]) for r in rd])
    assert np.allclose(retail[i], d.price, atol=2e-5)


@needs_pkg
@needs_all
def test_load_site_and_eval_sets_take_a_price_area():
    from experiments.dhw_v2_rl.train_v2 import eval_sets
    from stems.dhw_v2_rl.common import load_site
    d, bh = load_site("karlstad", [2025], price_area="SE2")
    assert d.meta["price_area"]["area"] == "SE2" and bh is not None
    out = eval_sets(["karlstad:2025@SE4"])
    assert list(out) == ["karlstad_2025@SE4"]
