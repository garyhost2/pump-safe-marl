import numpy as np
import pytest

from stems.dhw_v2.building import DEFAULT_SE_SFH
from stems.dhw_v2.data import DistrictData, HouseData
from stems.dhw_v2.plant import DistrictPlant, PlantConfig


def cold_district(T=72, t_out=-25.0, scale=1.5):
    b = dict(DEFAULT_SE_SFH)
    for k in ("H_ve_W_K", "H_tr_w_W_K", "H_tr_op_W_K"):
        b[k] = b[k] * scale
    h = HouseData("H", np.zeros(T), 60, np.full(T, 8.0), b)
    t = np.arange(T)
    return DistrictData("cold", "test", (np.datetime64("2025-01-06T00:00:00") + t.astype("timedelta64[h]")),
                        t % 24, (t // 24) % 7, np.full(T, t_out), np.zeros(T), np.ones(T), "SEK", 5.0, [h])


def run(aux, T=72, **kw):
    d = cold_district(T, **kw)
    pl = DistrictPlant(d, cfg=PlantConfig(brine="correlation", aux_heater=aux))
    out = []
    for _ in range(T):
        ctx = pl.context()
        u_s = np.clip(ctx["sh_need_set_kw"] / ctx["q_max_s"], 0.0, 1.0)
        out.append(pl.step(u_d=np.zeros(1), u_s=u_s))
    return {k: np.array([float(r[k][0]) for r in out]) for k in
            ("t_in", "elec_aux_kwh", "heat_aux_kwh", "aux_step_kw", "elec_kwh", "elec_sh_kwh",
             "elec_standby_kwh", "elec_dhw_kwh", "cost", "heat_sh_kwh")}, pl


def test_without_the_element_the_cold_house_falls_below_20():
    r, _ = run("off")
    assert r["t_in"].min() < 19.0
    assert r["elec_aux_kwh"].sum() == 0.0


def test_element_holds_the_indoor_minimum():
    r, pl = run("auto_min")
    t_min = pl.buildings[0].t_min
    assert r["t_in"].min() >= t_min - 1e-6
    assert r["elec_aux_kwh"].sum() > 0.0
    assert np.array_equal(r["elec_aux_kwh"], r["heat_aux_kwh"])
    assert np.allclose(r["elec_kwh"], r["elec_sh_kwh"] + r["elec_dhw_kwh"] + r["elec_standby_kwh"] + r["elec_aux_kwh"])
    assert np.allclose(r["cost"], r["elec_kwh"])


def test_element_runs_in_3_6_9_kw_steps_and_is_capped_at_9_kw():
    r, _ = run("auto_min")
    assert set(np.unique(r["aux_step_kw"])) <= {0.0, 3.0, 6.0, 9.0}
    on = r["elec_aux_kwh"] > 0
    assert np.all(r["aux_step_kw"][on] >= r["elec_aux_kwh"][on] - 1e-9)
    assert np.all(r["aux_step_kw"][on] - r["elec_aux_kwh"][on] < 3.0 + 1e-9)
    assert r["elec_aux_kwh"].max() <= 9.0 + 1e-12
    big, _ = run("auto_min", scale=6.0)
    assert big["elec_aux_kwh"].max() == pytest.approx(9.0)
    assert big["t_in"].min() < 20.0


def test_auto_set_reaches_the_set_point_and_uses_more_than_auto_min():
    a, pl = run("auto_set")
    b, _ = run("auto_min")
    assert a["t_in"][-1] == pytest.approx(pl.buildings[0].t_set, abs=1e-6)
    assert a["elec_aux_kwh"].sum() > b["elec_aux_kwh"].sum()


def test_element_idle_when_the_compressor_suffices():
    r, _ = run("auto_min", t_out=0.0, scale=1.0)
    assert r["elec_aux_kwh"].sum() == 0.0 and r["t_in"].min() > 20.5


def test_bad_mode_is_refused():
    with pytest.raises(ValueError):
        PlantConfig(aux_heater="on")


def test_element_stays_under_the_electric_caps():
    d = cold_district(48)
    pl = DistrictPlant(d, cfg=PlantConfig(brine="correlation"))
    for _ in range(48):
        ctx = pl.context()
        r = pl.step(u_d=np.zeros(1), u_s=np.clip(ctx["sh_need_set_kw"] / ctx["q_max_s"], 0, 1), cap_house_kw=6.0)
        assert r["elec_kwh"][0] <= 6.0 + 1e-9 or r["elec_aux_kwh"][0] == 0.0
    pl = DistrictPlant(d, cfg=PlantConfig(brine="correlation"))
    for _ in range(48):
        ctx = pl.context()
        r = pl.step(u_d=np.zeros(1), u_s=np.clip(ctx["sh_need_set_kw"] / ctx["q_max_s"], 0, 1), cap_district_kw=5.5)
        assert r["elec_kwh"].sum() <= 5.5 + 1e-9 or r["elec_aux_kwh"].sum() == 0.0
