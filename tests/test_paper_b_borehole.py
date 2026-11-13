import numpy as np
import pytest

from stems.dhw_v2.borehole import Borehole, BoreholeSpec, reference_extraction_kw, spinup_sequence
from stems.dhw_v2.building import DEFAULT_SE_SFH
from stems.dhw_v2.data import DistrictData, HouseData
from stems.dhw_v2.heatpump import RXT12Map
from stems.dhw_v2.plant import DistrictPlant, PlantConfig


def year_district(T=8760, houses=("A", "B")):
    t = np.arange(T)
    t_out = 6.0 - 12.0 * np.cos(2 * np.pi * t / 8760.0)
    hs = [HouseData(i, np.full(T, 0.2 + 0.1 * j), 60, np.full(T, 8.0), dict(DEFAULT_SE_SFH))
          for j, i in enumerate(houses)]
    return DistrictData("yr", "test", np.datetime64("2025-01-01T00:00:00") + t.astype("timedelta64[h]"),
                        t % 24, (t // 24) % 7, t_out, np.zeros(T), np.ones(T), "SEK", 7.0, hs)


def test_spinup_sequence_ends_at_the_run_start():
    p = np.arange(10.0)
    assert spinup_sequence(p, 0, 4).tolist() == [6, 7, 8, 9]
    assert spinup_sequence(p, 3, 5).tolist() == [8, 9, 0, 1, 2]
    assert spinup_sequence(p, 3, 23)[-3:].tolist() == [0, 1, 2] and len(spinup_sequence(p, 3, 23)) == 23


def test_zero_load_spinup_leaves_undisturbed_ground():
    b = Borehole(BoreholeSpec(t_ground_c=8.0, spinup_years=1))
    assert b.spin_up(np.zeros(8760)) == pytest.approx(8.0, abs=1e-9)


def test_constant_load_spinup_matches_the_g_function():
    import pygfunction as gt
    s = BoreholeSpec(t_ground_c=8.0, spinup_years=3)
    b = Borehole(s)
    q_kw, n = 2.0, 3 * 8760
    tw = b.spin_up(np.full(n, q_kw))
    g = gt.gfunction.gFunction([gt.boreholes.Borehole(s.H_m, s.D_m, s.r_b_m, 0.0, 0.0)], s.alpha,
                               time=np.array([n * 3600.0]), boundary_condition="UHTR", method="similarities").gFunc[0]
    expect = s.t_ground_c - (q_kw * 1000.0 / s.H_m) * g / (2.0 * np.pi * s.k_s_w_mk)
    assert tw == pytest.approx(expect, abs=0.05)
    assert tw < 8.0 - 1.0
    for _ in range(200):
        b.advance(q_kw)
    assert np.isfinite(b.t_wall)


def test_reference_extraction_is_seasonal_and_positive():
    d = year_district()
    q = reference_extraction_kw(d, d.houses[0], RXT12Map(), store_loss_kw=0.05)
    assert q.shape == (8760,) and np.all(q >= 0)
    assert q[:720].mean() > 3.0 * q[4000:4720].mean()
    assert 0.5 < q.mean() < 6.0


def test_plant_starts_colder_with_spinup_and_is_reproducible():
    d = year_district()
    run = d.subset(hours=48)
    assert run.parent is d and run.parent_offset == 0
    p0 = DistrictPlant(run, cfg=PlantConfig())
    p2 = DistrictPlant(run, cfg=PlantConfig(), borehole=BoreholeSpec(t_ground_c=d.t_ground_c, spinup_years=2))
    assert p0.spinup_info is None
    w0, w2 = p0.context()["t_wall"], p2.context()["t_wall"]
    assert np.allclose(w0, d.t_ground_c)
    assert np.all(w2 < w0 - 0.5)
    assert np.all(p2.context()["t_src_d"] < p0.context()["t_src_d"] - 0.5)
    assert p2.spinup_info["years"] == 2 and not p2.spinup_info["short_profile"]
    again = DistrictPlant(run, cfg=PlantConfig(), borehole=BoreholeSpec(t_ground_c=d.t_ground_c, spinup_years=2))
    assert np.array_equal(again.context()["t_wall"], w2)
    p2.step(u_d=np.ones(2), u_s=np.ones(2))
    assert np.array_equal(again.context()["t_wall"], w2)
    p5 = DistrictPlant(run, cfg=PlantConfig(), borehole=BoreholeSpec(t_ground_c=d.t_ground_c, spinup_years=5))
    w5 = p5.context()["t_wall"]
    assert np.all(w5 < w2) and np.all((w2 - w5) < 0.5 * (w0 - w2))


def test_spinup_ends_at_the_window_start_not_at_new_year():
    from stems.dhw_v2_rl.common import slice_hours
    d = year_district()
    jan = slice_hours(d, 0, 24)
    jul = slice_hours(d, 4380, 4404)
    assert jul.parent is d and jul.parent_offset == 4380
    bs = BoreholeSpec(t_ground_c=d.t_ground_c, spinup_years=2)
    wj = DistrictPlant(jan, cfg=PlantConfig(), borehole=bs).context()["t_wall"]
    ws = DistrictPlant(jul, cfg=PlantConfig(), borehole=bs).context()["t_wall"]
    assert not np.allclose(wj, ws)
    assert np.all(ws < d.t_ground_c)


def test_short_parent_is_flagged():
    d = year_district(T=240)
    p = DistrictPlant(d, cfg=PlantConfig(), borehole=BoreholeSpec(t_ground_c=7.0, spinup_years=1))
    assert p.spinup_info["short_profile"]
