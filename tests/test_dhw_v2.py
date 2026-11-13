import json

import numpy as np
import pytest

from stems import dhw_physics as P
from stems.dhw_v2 import tank as tk
from stems.dhw_v2.building import DEFAULT_SE_SFH, RC1R1C, RC5R1C, DemandBuilding, heating_curve_c
from stems.dhw_v2.heatpump import HplibMap, RXT12Map
from stems.thermia_rxt12 import CalibraRXT12


@pytest.fixture(scope="module")
def hp():
    return RXT12Map()


def test_map_matches_scalar_unit(hp):
    u = CalibraRXT12()
    for ts in (-3.0, 0.0, 5.0, 10.0):
        for tw in (35.0, 45.0, 55.0, 65.0):
            assert np.isclose(hp.q_max(ts, tw), u.q_max(ts, tw), rtol=1e-6)
            for q in (1.0, 2.5, 4.0, 7.0, float(u.q_max(ts, tw))):
                op = u.operate(q, ts, tw)
                assert np.isclose(hp.p_el(q, ts, tw), op.p_compressor_kw, rtol=1e-9)


def test_map_limits_and_flags(hp):
    assert hp.q_max(0.0, 70.5) == 0.0
    assert hp.q_max(0.0, 70.0) > 0.0
    f = hp.flags(np.array([0.0, 0.0, -2.0]), np.array([55.0, 60.0, 50.0]))
    assert f["extrap_supply"].tolist() == [False, True, False]
    assert f["extrap_source"].tolist() == [False, False, True]
    q = hp.q_max(0.0, 65.0)
    assert hp.p_el(q, 0.0, 65.0) <= hp.p_el_max + 1e-9


def test_chord_is_an_upper_bound(hp):
    for ts in (-5.0, 0.0, 8.0, 20.0):
        for tw in (25.0, 40.0, 55.0, 65.0, 70.0):
            p0, kap = hp.chord(ts, tw)
            q = np.linspace(1e-4, float(hp.q_max(ts, tw)), 2001)
            assert np.all(hp.p_el(q, ts, tw) <= p0 + kap * q + 1e-12)


def test_hplib_adapter_runs():
    h = HplibMap()
    assert h.q_max(0.0, 55.0) > 0 and h.p_el(2.0, 0.0, 55.0) > 0
    assert h.flags(0.0, 50.0)["extrap_supply"]


def test_two_nodes_reproduce_v1_closed_form():
    rng = np.random.default_rng(1)
    tc, span, V = 10.0, 50.0, 184.0
    h = V / 2
    C = h * tk.CV * span
    caps = np.array([C, C])
    for _ in range(500):
        T = np.sort(rng.uniform(15, 68, 2))[::-1]
        D = rng.uniform(0, 10)
        E = (T - tc) / span * C
        T2, sv, _, _ = tk.draw(T, h, D, tc, 40.0)
        Ed, s2, _ = P.plugflow_draw(E, caps, D, (40 - tc) / span)
        assert np.allclose(T2, tc + Ed / C * span, atol=1e-9) and np.isclose(sv, s2)
        ts = rng.uniform(45, 70)
        assert np.isclose(tk.accept_max(T, h, ts), P.accept_max(E, caps, (ts - tc) / span))
        Q = rng.uniform() * tk.accept_max(T, h, ts)
        Tc, _, _ = tk.charge(T, h, Q, ts)
        assert np.allclose(Tc, tc + P.plugflow_charge(E, caps, Q, (ts - tc) / span) / C * span, atol=1e-9)


@pytest.mark.parametrize("N", [2, 5, 10, 40])
def test_store_energy_conservation_and_order(N):
    rng = np.random.default_rng(N)
    V = 184.0
    h = V / N
    for _ in range(200):
        T = np.sort(rng.uniform(12, 68, N))[::-1]
        D = rng.uniform(0, 8)
        T2, served, unmet, vol = tk.draw(T, h, D, 10.0, 40.0)
        assert np.isclose(served + unmet, D)
        assert np.isclose(((T - 10) * h * tk.CV).sum() - ((T2 - 10) * h * tk.CV).sum(), served, atol=1e-9)
        assert np.all(np.diff(T2) <= 1e-9)
        ts = rng.uniform(45, 70)
        a = tk.accept_max(T2, h, ts)
        Q = rng.uniform() * a
        T3, q, _ = tk.charge(T2, h, Q, ts)
        assert np.isclose(((T3 - T2) * h * tk.CV).sum(), q, atol=1e-9)
        assert np.all(np.diff(T3) <= 1e-9)
        Tf, qf, _ = tk.charge(T2, h, a, ts)
        assert np.allclose(np.where(T2 < ts, Tf, T2), np.where(T2 < ts, ts, T2), atol=1e-9)


def test_loss_ua_and_calibration():
    cal = tk.calibrate_ua()
    assert np.isclose(cal["loss_w"], (34.0 - 17.0) * cal["cop_reheat"])
    assert 1.0 < cal["ua_w_k"] < 3.5
    assert tk.calibrate_ua("pes_cop")["ua_w_k"] > cal["ua_w_k"] > tk.calibrate_ua("pes_as_heat")["ua_w_k"]
    t = tk.rxt12_tank(2)
    assert t.volume_l == 184.0 and np.isclose(t.ua_w_k, cal["ua_w_k"])
    T = np.array([[50.9, 50.9]])
    T2, lost = tk.loss(T, t.ua_w_k, 20.0, 1.0, t.node_volume_l)
    assert np.isclose(lost[0], t.ua_w_k * 30.9 / 1000.0, rtol=1e-2)


def test_buoyancy_and_aggregation():
    T = np.array([60.0, 50.0, 40.0])
    assert np.array_equal(tk.buoyancy(T), T)
    inv = np.array([40.0, 60.0, 30.0])
    b = tk.buoyancy(inv)
    assert np.all(np.diff(b) <= 1e-12) and np.isclose(b.sum(), inv.sum())
    T10 = np.linspace(60, 20, 10)
    assert np.isclose(tk.aggregate_nodes(T10, 2).mean(), T10.mean())
    assert tk.sensor_index(tk.rxt12_tank(2)) == 0 and tk.sensor_index(tk.rxt12_tank(10)) == 3


def test_v40_of_a_full_store():
    T = np.full(10, 55.0)
    assert np.isclose(tk.v40_l(T, 18.4, 10.0), 184.0 * 45.0 / 30.0)


def test_borehole_cools_under_extraction_and_recovers():
    from stems.dhw_v2.borehole import Borehole, BoreholeSpec
    b = Borehole(BoreholeSpec(t_ground_c=8.0))
    assert b.wall_temperature() == 8.0
    for _ in range(240):
        b.advance(6.0)
    cold = b.wall_temperature()
    assert 2.0 < cold < 7.5
    assert b.inlet(6.0) < cold
    for _ in range(240):
        b.advance(0.0)
    assert cold < b.wall_temperature() < 8.0


def test_brine_correlation_is_v1():
    from stems.dhw_v2.borehole import BrineCorrelation
    from stems.heatpump_hplib import brine_temperature_c
    t = np.linspace(-5, 25, 100)
    bc = BrineCorrelation(t)
    for _ in range(50):
        bc.advance(0.0)
    assert np.isclose(bc.wall_temperature(), brine_temperature_c(t[27:51].mean()))
    assert np.isclose(bc.series[0], brine_temperature_c(t[0]))


@pytest.mark.parametrize("cls,p", [(RC5R1C, DEFAULT_SE_SFH),
                                   (RC1R1C, {"H_W_K": 250.0, "C_J_K": 2.0e7, "internal_gains_W": 300})])
def test_building_need_is_exact(cls, p):
    for t_e in (-15.0, 0.0, 10.0):
        b = cls(p, t_init_c=19.0)
        need = b.need_w(21.0, t_e, 100.0)
        assert need > 0.0
        assert np.isclose(b.step(need, t_e, 100.0), 21.0, atol=1e-6)


def test_default_archetype_design_load():
    b = RC5R1C(DEFAULT_SE_SFH)
    for _ in range(24 * 10):
        b.step(b.need_w(21.0, -18.0, 0.0), -18.0, 0.0)
    assert 8.5e3 < b.need_w(21.0, -18.0, 0.0) < 10.5e3


def test_heating_curve():
    assert np.isclose(heating_curve_c(0.0), 40.0)
    assert np.isnan(heating_curve_c(18.0))
    assert heating_curve_c(-30.0) == 55.0


def test_demand_building_debt():
    b = DemandBuilding({"heating_demand_kwh": np.array([2.0, 1.0])})
    b.step(1500.0, 0.0, 0.0)
    assert np.isclose(b.debt, 0.5)
    assert np.isclose(b.need_w(21.0, 0, 0), 1500.0)


def _write_package(root, dt_min=10, hours=48):
    from datetime import datetime, timedelta
    root.mkdir()
    (root / "draws").mkdir()
    (root / "buildings").mkdir()
    t0 = datetime.fromisoformat("2019-01-07T00:00:00+01:00")
    rows_w = ["time,T_out,GHI"] + [f"{(t0 + timedelta(hours=i)).isoformat()},{-5 + i % 5},{max(0, 100 * (i % 24 - 8))}"
                                   for i in range(hours)]
    rows_p = ["time,price"] + [f"{(t0 + timedelta(hours=i)).isoformat()},{1.0 + 0.1 * (i % 24)}" for i in range(hours)]
    (root / "weather.csv").write_text("\n".join(rows_w) + "\n")
    (root / "price.csv").write_text("\n".join(rows_p) + "\n")
    n = hours * 60 // dt_min
    rows_d = ["time,volume_l"] + [f"{(t0 + timedelta(minutes=dt_min * i)).isoformat()},{5.0 if i % 7 == 0 else 0.0}"
                                  for i in range(n)]
    (root / "draws" / "H01.csv").write_text("\n".join(rows_d) + "\n")
    (root / "buildings" / "H01.json").write_text(json.dumps(DEFAULT_SE_SFH | {"t_draw_c": 40.0}))
    man = {"format": "stems-dhw-v2", "version": 1, "name": "test", "currency": "SEK", "t_cold_c": 8.0,
           "houses": [{"id": "H01", "draws": "draws/H01.csv", "building": "buildings/H01.json"}],
           "sources": {"weather": "synthetic", "price": "synthetic", "draws": "synthetic", "buildings": "placeholder"}}
    (root / "manifest.json").write_text(json.dumps(man))


def test_package_loader_roundtrip(tmp_path):
    from stems.dhw_v2.data import load_package
    _write_package(tmp_path / "pkg")
    d = load_package(tmp_path / "pkg")
    assert d.T == 48 and d.B == 1 and d.houses[0].draw_dt_min == 10 and d.currency == "SEK"
    e = d.houses[0].draw_kwh
    assert np.isclose(e[0], 5.0 * tk.CV * 32.0)
    sub = d.draws_substep(5)
    assert sub.shape == (48, 12, 1) and np.isclose(sub.sum(), e.sum())
    assert d.day_of_week[0] == 0 and d.hour_of_day[0] == 0


def test_package_rejects_naive_time(tmp_path):
    from stems.dhw_v2.data import load_package
    _write_package(tmp_path / "pkg")
    w = (tmp_path / "pkg" / "weather.csv").read_text().replace("+01:00", "")
    (tmp_path / "pkg" / "weather.csv").write_text(w)
    with pytest.raises(ValueError):
        load_package(tmp_path / "pkg")


def test_texas_loader_same_structure():
    from stems.dhw_v2.data import load_texas
    d = load_texas(steps=200)
    assert d.T == 200 and d.B == 8 and d.houses[0].draw_dt_min == 60
    sub = d.draws_substep(5)
    assert sub.shape == (200, 12, 8)
    assert np.allclose(sub.sum(axis=1), d.hourly_draw_kwh())
    assert d.houses[0].building["model"] == "demand"


@pytest.fixture(scope="module")
def texas2():
    from stems.dhw_v2.data import load_texas
    return load_texas(steps=24 * 8).subset([0, 1])


def test_plant_time_sharing_and_intra_hour_min(texas2):
    from stems.dhw_v2.plant import DistrictPlant
    pl = DistrictPlant(texas2)
    h = pl.tank["node_vol"]
    for t in range(48):
        rec = pl.step(u_d=np.array([0.3, 0.0]) if t % 3 == 0 else np.zeros(2), u_s=np.array([0.2, 0.5]))
        assert np.all(rec["dhw_time_h"] + rec["sh_time_h"] <= 1.0 + 1e-9)
        assert np.all(rec["top_min_c"] <= rec["top_end_c"] + 1e-9)


def test_plant_energy_balance_exact(texas2):
    from stems.dhw_v2.plant import DistrictPlant
    pl = DistrictPlant(texas2)
    h = pl.tank["node_vol"]
    for t in range(30):
        E0 = ((pl.T - 10.0) * h[:, None] * tk.CV).sum(axis=1)
        rec = pl.step(u_d=np.array([0.5, 1.0]) * (t % 2), u_s=np.zeros(2))
        E1 = ((pl.T - 10.0) * h[:, None] * tk.CV).sum(axis=1)
        lost_rel = rec["loss_kwh"]
        assert np.allclose(E1 - E0, rec["heat_dhw_kwh"] - rec["served_kwh"] - lost_rel, atol=1e-9)


def test_thermostat_native(texas2):
    from stems.dhw_v2.plant import DistrictPlant
    pl = DistrictPlant(texas2)
    pl.T[:] = 44.0
    th = {"start_c": 45.0, "stop_c": 52.0, "leg_armed": np.array([True, False])}
    rec = pl.step(thermostat=th)
    assert rec["leg_started"][0] and not rec["leg_started"][1]
    assert rec["dhw_time_h"][1] > 0 and pl.T[1, 0] >= 52.0 - 1e-6


def test_dt_one_minute_runs(texas2):
    from stems.dhw_v2.plant import DistrictPlant, PlantConfig
    pl = DistrictPlant(texas2, cfg=PlantConfig(dt_min=1))
    rec = pl.step(u_d=np.ones(2))
    assert pl.S == 60 and np.all(rec["heat_dhw_kwh"] >= 0)


def test_polygon_projection_matches_cvxpy():
    cp = pytest.importorskip("cvxpy")
    from stems.dhw_v2.filter import _house_polygon, project_polygon
    rng = np.random.default_rng(0)
    for _ in range(60):
        lo = rng.uniform(0, 0.5, 2)
        A, b = _house_polygon(lo[0], lo[1], rng.uniform(1, 4), rng.uniform(0.5, 3), rng.uniform(2.5, 6))
        p = rng.uniform(-0.5, 1.5, 2)
        x = cp.Variable(2)
        pr = cp.Problem(cp.Minimize(cp.sum_squares(x - p)), [A @ x <= b])
        pr.solve()
        if pr.status != "optimal":
            continue
        assert np.allclose(project_polygon(p, A, b), x.value, atol=1e-5)


def test_joint_qp_district_cap_matches_cvxpy():
    cp = pytest.importorskip("cvxpy")
    from stems.dhw_v2.filter import solve_joint_qp
    rng = np.random.default_rng(3)
    for _ in range(20):
        B = 3
        prop = rng.uniform(0, 1, (B, 2))
        lo = np.zeros((B, 2))
        coef = rng.uniform(0.5, 4, (B, 2))
        const = rng.uniform(0, 0.1, B)
        cap = np.full(B, np.inf)
        capD = 0.4 * float(np.sum(coef * prop) + const.sum())
        xs = solve_joint_qp(prop, lo, coef, const, cap, capD)
        assert np.sum(coef * xs) + const.sum() <= capD + 1e-9
        X = cp.Variable((B, 2))
        cons = [X >= 0, X <= 1, cp.sum(X, axis=1) <= 1, cp.sum(cp.multiply(coef, X)) + const.sum() <= capD]
        cp.Problem(cp.Minimize(cp.sum_squares(X - prop)), cons).solve()
        assert np.allclose(xs, X.value, atol=1e-4)


def test_margin_tracker_one_sided_bound():
    from stems.dhw_v2.filter import MarginTracker
    rng = np.random.default_rng(0)
    for alpha in (0.05, 0.01, 0.5):
        m = MarginTracker(1, alpha=alpha, eta=0.05)
        hits = 0
        for _ in range(20000):
            e = rng.exponential(1.0, 1) - 0.5 * (rng.uniform() < 0.7)
            hits += int(e[0] > m.m[0])
            m.update(e)
        assert hits / 20000 <= m.upper_bound()[0] + 1e-12


def test_certificate_indicator_monotone_in_charge_and_draw(texas2):
    from stems.dhw_v2.plant import DistrictPlant, simulate_hour
    pl = DistrictPlant(texas2)
    ctx = pl.context()
    rng = np.random.default_rng(5)
    for _ in range(100):
        T = np.sort(rng.uniform(15, 62, 2))[::-1][None, :]
        D = rng.uniform(0, 5)
        u = np.linspace(0, 1, 41)
        dr = np.full((12, 1), D / 12)
        r = simulate_hour(T, {k: (v[:1] if isinstance(v, np.ndarray) else v) for k, v in pl.tank.items()},
                          dr, 1 / 12, u_d=u, r_d=ctx["r_d"][0], p_d=ctx["p_d"][0], t_s=55.0, t_cold=10.0)
        ok = r["top_min"] >= 45.0
        assert np.all(np.diff(ok.astype(int)) >= 0)
        hi = r["top_min"] >= 40.0 + 1e-6
        assert np.all(np.diff(r["top_min"])[hi[:-1] & hi[1:]] >= -1e-9)
        Ds = np.linspace(0, 6, 31)
        r2 = simulate_hour(T, {k: (v[:1] if isinstance(v, np.ndarray) else v) for k, v in pl.tank.items()},
                           np.full((12, 31), 1.0) * Ds / 12, 1 / 12, u_d=0.3, r_d=ctx["r_d"][0],
                           p_d=ctx["p_d"][0], t_s=55.0, t_cold=10.0)
        ok2 = r2["top_min"] >= 45.0
        assert np.all(np.diff(ok2.astype(int)) <= 0)


def test_certificate_holds_on_the_plant_when_draw_is_certified(texas2):
    from stems.dhw_v2.env import DHWDistrictEnv, EnvConfig
    from stems.dhw_v2.filter import FilterConfig
    env = DHWDistrictEnv(texas2, EnvConfig(filter=FilterConfig(forecast="oracle", lookahead=False), hours=96))
    env.reset()
    rng = np.random.default_rng(0)
    n_ok = 0
    while True:
        a = np.c_[rng.uniform(0, 0.3, 2), rng.uniform(0, 0.3, 2), np.zeros(2)]
        T_pre = env.plant.T.copy()
        obs, r, term, _, info = env.step(a)
        f = info["filter"]
        ok = ~f["infeasible_c"] & (f["s_c"] <= 1e-9) & (T_pre[:, 0] >= 45.0)
        assert np.all(info["record"]["top_min_c"][ok] >= 45.0 - 1e-6)
        n_ok += ok.sum()
        if term:
            break
    assert n_ok > 100


def test_filter_respects_house_and_district_caps(texas2):
    from stems.dhw_v2.env import DHWDistrictEnv, EnvConfig
    from stems.dhw_v2.filter import FilterConfig
    env = DHWDistrictEnv(texas2, EnvConfig(filter=FilterConfig(cap_house_kw=3.0, cap_district_kw=4.0), hours=48))
    env.reset()
    while True:
        obs, r, term, _, info = env.step(np.ones((2, 3)) * [1.0, 1.0, 0.0])
        eb = info["filter"]["elec_bound_kw"]
        assert np.all(eb <= 3.0 + 1e-9) and eb.sum() <= 4.0 + 1e-9
        rec = info["record"]
        assert np.all(rec["elec_kwh"] - rec["elec_aux_kwh"] <= eb + 1e-9)
        assert np.all(rec["elec_kwh"] <= 3.0 + 1e-9) and rec["elec_kwh"].sum() <= 4.0 + 1e-9
        if term:
            break


def test_env_interface(texas2):
    from stems.dhw_v2.env import OBS_NAMES, DHWDistrictEnv, EnvConfig
    from stems.dhw_v2.filter import FilterConfig
    env = DHWDistrictEnv(texas2, EnvConfig(filter=FilterConfig(), hours=24))
    obs, info = env.reset()
    assert obs.shape == (2, len(OBS_NAMES)) and obs.dtype == np.float32
    obs, rew, term, trunc, info = env.step({a: np.array([0.5, 0.2, 0.0]) for a in env.agents})
    assert rew.shape == (2,) and np.all(np.isfinite(rew))
    for k in ("proposed", "executed", "filter", "delta", "m", "violation", "declared_infeasible", "record"):
        assert k in info
    assert "extrap_supply" in info["record"] and "extrap_source" in info["record"]
    with pytest.raises(ValueError):
        env.step_command({"thermostat": {"start_c": 45, "stop_c": 52}})


def test_identification_recovers_ua(texas2):
    from stems.dhw_v2.controllers import ThermiaLogic
    from stems.dhw_v2.env import DHWDistrictEnv, EnvConfig
    from stems.dhw_v2.identify import LogRecorder, identify
    env = DHWDistrictEnv(texas2, EnvConfig(hours=24 * 8))
    env.reset()
    ctl = ThermiaLogic(2, stop_c=54.0)
    log = LogRecorder(2, sigma_t=0.0, sigma_p=0.0)
    vol = np.array([t.volume_l for t in env._tanks])
    while True:
        T0, ctx = env.plant.T.copy(), env.plant.context()
        obs, r, term, _, info = env.step_command(ctl.act(env))
        ctl.update(info["record"])
        log.record(T0, env.plant.T, ctx, info["record"], np.where(info["record"]["dis_supply_used"], 65.0, 55.0), vol)
        if term:
            break
    idf = identify(log)
    assert abs(idf["ua_w_k"] / env._tanks[0].ua_w_k - 1.0) < 0.25
    assert idf["n_dhw_hours"] > 5


def test_dhw_speed_calibration_regenerates():
    from stems.dhw_v2.calibration import (DHW_SPEED_FRAC, HEAT_UP_CERTIFIED_H, calibrate_dhw_speed,
                                          heat_up_hours)
    c = calibrate_dhw_speed(10)
    assert abs(c["dhw_speed_frac"] - DHW_SPEED_FRAC) < 2e-3
    assert abs(heat_up_hours(DHW_SPEED_FRAC, 10, target_kwh=c["target_kwh"], dt_min=0.5) - HEAT_UP_CERTIFIED_H) < 0.03


def test_v21_defaults():
    from stems.dhw_v2.calibration import DHW_SPEED_FRAC
    from stems.dhw_v2.controllers import ThermiaLogic
    from stems.dhw_v2.env import EnvConfig
    from stems.dhw_v2.filter import FilterConfig
    from stems.dhw_v2.plant import PlantConfig
    assert EnvConfig().tank_nodes == 10
    assert PlantConfig().dhw_speed_frac == DHW_SPEED_FRAC and PlantConfig().dis_speed_frac is None
    th = ThermiaLogic(1)
    assert th.start_c > PlantConfig().t_floor_c and th.stop_c > th.start_c
    f = FilterConfig()
    assert f.reserve_h == 24 and f.trigger == "robust"


def test_hour_of_day_conformal_bound():
    from stems.dhw_v2.filter import HourOfDayQuantile
    q = HourOfDayQuantile(1, beta=0.1)
    assert np.isinf(q.quantile(3)[0])
    for d in range(5):
        q.update(3, np.array([float(d)]))
    assert np.isinf(q.quantile(3)[0])
    for d in range(5, 30):
        q.update(3, np.array([float(d % 10)]))
    assert q.quantile(3)[0] == 9.0
    qi = HourOfDayQuantile(1, beta=0.1, method="interp")
    assert qi.quantile(0)[0] == 0.0


def _filter_for(texas2, H=8, cap=None):
    from stems.dhw_v2.filter import FilterConfig, FilterModel, JointFilter, week_calendar
    from stems.dhw_v2.plant import DistrictPlant
    pl = DistrictPlant(texas2, tanks=tk.rxt12_tank(10))
    ctx = pl.context()
    wid, left = week_calendar(texas2.hour_of_day, texas2.day_of_week)
    f = JointFilter(FilterModel(pl.tank, pl.hp), FilterConfig(reserve_h=H), 2, pl.S, pl.dt_h, wid, left)
    rng = np.random.default_rng(0)
    for h in range(48):
        f.forecaster.update(h % 24, rng.exponential(0.6, 2))
    m = np.array([0.8, 1.2])
    u_max = np.array([1.0, 0.35]) if cap else np.ones(2)
    R = f._prepare_reserve(ctx, 7, m, ctx["t_cold"], u_max)
    return f, R, pl


def _brute(f, R, X):
    ok = np.ones(X.shape[:-1], bool)
    for j in range(f.cfg.reserve_h):
        r = f._sim(X, R["Dq"][j], R["u"], R["t_s"], R["r"], R["p"], R["t_cold"])
        ok &= r["top_min"] >= f.cfg.floor_c - 1e-9
        X = r["T_end"]
    return ok


@pytest.mark.parametrize("cap", [False, True])
def test_reserve_tail_equals_brute_force_rollout(texas2, cap):
    f, R, _ = _filter_for(texas2, H=8, cap=cap)
    rng = np.random.default_rng(1)
    X = np.sort(rng.uniform(15, 56, (2, 40, 10)), axis=-1)[..., ::-1]
    X[:, :5] = 55.0
    assert np.array_equal(f._tail(X, 0, R), _brute(f, R, X))


def test_reserve_monotone_in_state(texas2):
    f, R, _ = _filter_for(texas2, H=8, cap=True)
    rng = np.random.default_rng(2)
    X = np.sort(rng.uniform(20, 52, (2, 60, 10)), axis=-1)[..., ::-1]
    Y = np.minimum(X + rng.uniform(0, 4, X.shape), 55.0)
    Y = np.sort(Y, axis=-1)[..., ::-1]
    Y = np.maximum(Y, X)
    okX, okY = f._tail(X, 0, R), f._tail(Y, 0, R)
    assert np.all(okY | ~okX)


def test_robust_trigger_completes_every_week_for_a_null_proposer(texas2):
    from stems.dhw_v2.controllers import NullController
    from stems.dhw_v2.env import DHWDistrictEnv, EnvConfig
    from stems.dhw_v2.filter import FilterConfig
    env = DHWDistrictEnv(texas2, EnvConfig(filter=FilterConfig(reserve_h=0), hours=24 * 8))
    env.reset()
    ctl = NullController(2)
    while True:
        obs, r, term, _, info = env.step_command(ctl.act(env))
        if term:
            break
    S = env.summary()
    assert S["weeks_full"] == 1 and S["weeks_completed"] == [1, 1]


def test_hold_rules_and_mesh():
    from stems.dhw_v2.plant import bottom_weights, simulate_hour, tank_arrays
    assert np.allclose(bottom_weights(10), [0] * 8 + [0.5, 0.5])
    out = {}
    for N in (10, 20):
        ta = tank_arrays([tk.rxt12_tank(N)])
        draws = np.full((12, 1), 1.0 / 12)
        r = simulate_hour(np.full((1, N), 64.0), ta, draws, 1 / 12, u_d=1.0, r_d=4.7, p_d=1.5,
                          t_s=65.0, t_cold=10.0, hold_rule="bottom20")
        out[N] = (bool(r["hold"][0]), bool(r["hold_all"][0]), float(r["bot_min"][0]))
    assert out[10][0] and out[20][0]
    assert not out[20][1]
