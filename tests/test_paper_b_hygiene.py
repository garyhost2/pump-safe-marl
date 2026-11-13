import numpy as np
import pytest

from stems.dhw_v2 import tank as tk
from stems.dhw_v2.building import DEFAULT_SE_SFH
from stems.dhw_v2.data import DistrictData, HouseData
from stems.dhw_v2.hygiene import (REGIMES, SCORED, HygieneScorer, HygieneSpec, ThermiaRuleScorer, get_regime,
                                  period_calendar, regime_overrides)
from stems.dhw_v2.plant import PlantConfig, simulate_hour, tank_arrays

S = 12


def calendar(T, start_dow=0, start_hour=0):
    t = np.arange(T) + start_hour
    return t % 24, (start_dow + t // 24) % 7


def rec(bot=20.0, nmin=20.0, mean=20.0, top=50.0, sensor=45.0, used=False, served=0.0, delivered=np.nan, B=1):
    f = lambda x: np.broadcast_to(np.asarray(x, float).reshape(-1, 1) if np.ndim(x) else float(x), (S, B)).copy()
    return {"series": {"bot": f(bot), "nmin": f(nmin), "mean": f(mean), "top": f(top), "sensor": f(sensor)},
            "dis_supply_used": np.full(B, used), "served_kwh": np.full(B, served),
            "delivered_min_c": np.full(B, delivered)}


def test_the_four_regimes_of_the_plan():
    assert list(REGIMES) == ["none", "weekly_bottom20_60_1h", "daily_whole_60_1h", "w551_70_3min"]
    n, w, d, x = REGIMES.values()
    assert n.t_c is None and n.period_h is None and n.t_tap_min_c == 50.0
    assert (w.t_c, w.hold_min, w.volume, w.period_h) == (60.0, 60.0, "bottom20", 168)
    assert (d.t_c, d.hold_min, d.volume, d.period_h) == (60.0, 60.0, "whole", 24)
    assert (x.t_c, x.hold_min, x.volume, x.t_supply_c) == (70.0, 3.0, "whole", 70.0)
    assert w.hold_steps(5) == 12 and x.hold_steps(5) == 1 and x.hold_steps(1) == 3
    assert [s.name for s in SCORED][-1] == "w551_70_3min_tol1K" and SCORED[-1].threshold_c == 69.0


def test_a_regime_is_a_free_specification():
    s = HygieneSpec("mine", "55 degC, 30 min, store mean, daily", 55.0, 30.0, "whole_mean", 24, 60.0)
    o = regime_overrides(s)
    assert o["plant"] == {"t_dis_c": 55.0, "t_supply_dis_c": 60.0, "hold_rule": "whole_mean", "hold_min": 30.0}
    assert o["env"]["period_h"] == 24
    with pytest.raises(ValueError):
        HygieneSpec("bad", "", 60.0, 60.0, "top", 168)
    with pytest.raises(ValueError):
        HygieneSpec("bad", "", 60.0, 60.0, "whole", 72)


def test_regime_overrides():
    assert regime_overrides("none") == {"plant": {}, "filter": {"disinfection": False},
                                        "env": {"hygiene": False, "regime": "none"}}
    o = regime_overrides("daily_whole_60_1h")
    assert o["plant"]["hold_rule"] == "all_nodes" and o["env"] == {"period_h": 24, "hygiene": True, "regime": "daily_whole_60_1h"}
    o = regime_overrides("w551_70_3min", tol_k=1.0)
    assert o["plant"] == {"t_dis_c": 69.0, "t_supply_dis_c": 70.0, "hold_rule": "all_nodes", "hold_min": 3.0}
    assert o["env"]["regime"] == "w551_70_3min~tol1K"
    PlantConfig(**o["plant"])


def test_period_calendar_week_equals_the_filters_and_day_is_local_midnight():
    from stems.dhw_v2.filter import week_calendar
    hod, dow = calendar(24 * 16, start_dow=3, start_hour=5)
    a, b = period_calendar(hod, dow, 168), week_calendar(hod, dow)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    pid, left = period_calendar(hod, dow, 24)
    assert pid[0] == 0 and pid[18] == 0 and pid[19] == 1 and left[0] == 19 and left[19] == 24


def test_hold_run_may_cross_a_clock_hour_but_must_be_uninterrupted():
    hod, dow = calendar(168)
    sc = HygieneScorer(1, 5, hod, dow)
    up6_end = np.r_[np.full(6, 20.0), np.full(6, 61.0)]
    up6_start = np.r_[np.full(6, 61.0), np.full(6, 20.0)]
    for t in range(168):
        sc.update(t, rec(bot=up6_end if t == 10 else up6_start if t == 11 else 20.0))
    w = sc.summary()["weekly_bottom20_60_1h"]
    assert w["holds_completed"] == [1] and w["periods_full"] == 1 and w["periods_missed"] == [0]
    assert w["hours_at_or_above"] == [1.0] and w["max_gap_h"] == [11.5]
    sc = HygieneScorer(1, 5, hod, dow)
    eleven = np.r_[np.full(11, 61.0), 59.9]
    for t in range(168):
        sc.update(t, rec(bot=eleven if t in (10, 11) else 20.0))
    w = sc.summary()["weekly_bottom20_60_1h"]
    assert w["holds_completed"] == [0] and w["periods_missed"] == [1]


def test_daily_whole_tank_counts_days_and_only_whole_periods():
    hod, dow = calendar(24 * 3 + 7)
    sc = HygieneScorer(2, 5, hod, dow)
    for t in range(24 * 3 + 7):
        hot = t in (5, 30)
        sc.update(t, rec(nmin=60.0 if hot else 30.0, bot=65.0, B=2))
    d = sc.summary()["daily_whole_60_1h"]
    assert d["periods_full"] == 3 and d["periods_completed"] == [2, 2] and d["periods_missed"] == [1, 1]
    assert d["max_gap_h"] == [25.0, 25.0]
    assert sc.summary()["weekly_bottom20_60_1h"]["holds_completed"] == [1, 1]


def test_w551_needs_70_for_3_minutes_and_the_tolerance_is_reported():
    hod, dow = calendar(168)
    sc = HygieneScorer(1, 5, hod, dow)
    one = np.r_[np.full(5, 30.0), 69.5, np.full(6, 30.0)]
    for t in range(168):
        sc.update(t, rec(nmin=one if t == 3 else 30.0))
    s = sc.summary()
    assert s["w551_70_3min"]["holds_completed"] == [0] and s["w551_70_3min"]["tol_k"] == 0.0
    assert s["w551_70_3min_tol1K"]["holds_completed"] == [1] and s["w551_70_3min_tol1K"]["tol_k"] == 1.0
    sc = HygieneScorer(1, 5, hod, dow)
    for t in range(168):
        sc.update(t, rec(nmin=np.r_[np.full(5, 30.0), 70.0, np.full(6, 30.0)] if t == 3 else 30.0))
    assert sc.summary()["w551_70_3min"]["periods_missed"] == [0]
    sc1 = HygieneScorer(1, 1, hod, dow)
    assert sc1.st["w551_70_3min"]["need"] == 3


def test_regime_none_scores_only_the_tap():
    hod, dow = calendar(48)
    sc = HygieneScorer(1, 5, hod, dow)
    for t in range(48):
        sc.update(t, rec(served=0.5 if t % 2 else 0.0, delivered=48.0 if t == 7 else 53.0))
    s = sc.summary()
    assert "periods_full" not in s["none"] and s["none"]["t_c"] is None
    assert s["tap_50c"]["draw_hours"] == [24] and s["tap_50c"]["draw_hours_delivered_below"] == [1]
    assert s["tap_50c"]["min_delivered_c"] == [48.0]


def test_thermia_rule_done_at_60_no_hold_within_7_h_reported_per_sensor():
    sc = ThermiaRuleScorer(1, 5, weights=(0.5, 0.5), sensor_height=0.6)
    ramp_top = np.r_[np.full(7, 55.0), np.full(5, 60.0)]
    sc.update(0, rec(used=False))
    sc.update(1, rec(top=ramp_top, sensor=40.0, used=True))
    for t in range(2, 12):
        sc.update(t, rec(top=62.0, sensor=40.0 if t < 4 else 58.5, used=True))
    s = sc.summary()
    assert s["top"]["programs"] == [1] and s["top"]["done_within_limit"] == [1] and s["top"]["alarms"] == [0]
    assert s["top"]["mean_minutes_to_target"] == [35.0]
    assert s["middle"]["done_within_limit"] == [0] and s["middle"]["alarms"] == [1]
    assert s["weighted"]["done_within_limit"] == [1] and s["weighted"]["max_minutes_to_target"] == [180.0]
    assert s["limit_h"] == 7.0 and s["target_c"] == 60.0 and s["middle_sensor_height"] == 0.6


def test_thermia_rule_one_program_per_start_and_alarm_at_7_h():
    sc = ThermiaRuleScorer(2, 5)
    for t in range(9):
        sc.update(t, rec(top=np.array([58.0]), sensor=50.0, used=True, B=2))
    s = sc.summary()
    assert s["top"]["programs"] == [1, 1] and s["top"]["alarms"] == [1, 1] and s["top"]["open_at_end"] == [0, 0]
    sc.update(9, rec(used=False, B=2))
    sc.update(10, rec(top=61.0, used=True, B=2))
    s = sc.summary()
    assert s["top"]["programs"] == [2, 2] and s["top"]["done_within_limit"] == [1, 1]
    assert s["top"]["at_target_at_start"] == [1, 1]


def test_a_regime_miss_is_not_a_thermia_failure():
    hod, dow = calendar(168)
    hy, th = HygieneScorer(1, 5, hod, dow), ThermiaRuleScorer(1, 5)
    for t in range(168):
        r = rec(top=61.0 if t == 50 else 50.0, bot=40.0, used=t == 50)
        hy.update(t, r)
        th.update(t, r)
    assert th.summary()["top"]["done_within_limit"] == [1] and th.summary()["top"]["alarms"] == [0]
    assert hy.summary()["weekly_bottom20_60_1h"]["periods_missed"] == [1]
    assert "Thermia" in hy.summary()["_note"] and "alarms" not in hy.summary()["weekly_bottom20_60_1h"]


def hour(T0, u_d, t_s, hold_rule="all_nodes", t_dis=60.0, hold_steps=None, n=10, dt_min=5, draw=0.0):
    tank = tank_arrays([tk.rxt12_tank(n)])
    Ssub = 60 // dt_min
    return simulate_hour(np.full((1, n), float(T0)) if np.ndim(T0) == 0 else T0, tank, np.full((Ssub, 1), draw / Ssub),
                         dt_min / 60.0, u_d=np.array([u_d]), r_d=np.array([5.0]), p_d=np.array([1.7]),
                         t_s=np.array([t_s]), t_cold=np.array([8.0]), t_dis_c=t_dis, hold_rule=hold_rule,
                         hold_steps=hold_steps, series=True)


def test_plant_hold_flag_default_is_the_whole_clock_hour():
    r = hour(61.0, 0.0, 65.0)
    assert r["hold"][0] and r["hold_all"][0] and r["series"]["nmin"].shape == (12, 1)
    T = np.full((1, 10), 61.0)
    T[0, -1] = 59.0
    r = hour(T, 1.0, 65.0)
    assert not r["hold"][0]
    assert hour(T, 1.0, 65.0, hold_steps=1)["hold"][0]
    assert not hour(T, 0.0, 65.0, hold_steps=1)["hold"][0]


def test_strict_w551_is_unreachable_at_the_70_degc_supply_limit_but_reachable_with_1_k():
    o = regime_overrides("w551_70_3min")["plant"]
    r = None
    T = np.full((1, 10), 50.0)
    for _ in range(6):
        r = hour(T, 1.0, o["t_supply_dis_c"], t_dis=o["t_dis_c"], hold_steps=1)
        T = r["T_end"]
    assert np.allclose(T, 70.0, atol=1e-6)
    assert not r["hold"][0] and r["series"]["nmin"].max() < 70.0
    assert hour(T, 1.0, 70.0, t_dis=69.0, hold_steps=1)["hold"][0]


def small_district(T=96, B=2):
    t = np.arange(T)
    rng = np.random.default_rng(0)
    hs = [HouseData(f"H{j}", rng.uniform(0.0, 0.5, T) * (rng.random(T) < 0.4), 60, np.full(T, 8.0),
                    dict(DEFAULT_SE_SFH)) for j in range(B)]
    return DistrictData("small", "test", np.datetime64("2025-03-03T00:00:00") + t.astype("timedelta64[h]"),
                        t % 24, (t // 24) % 7, np.full(T, 2.0), np.zeros(T), 1.0 + 0.5 * np.sin(t / 3.0), "SEK", 7.0, hs)


def run_env(regime, ctl="reactive", T=96, tol=None, filtered=True):
    from stems.dhw_v2.controllers import ReactiveController, ThermiaWeekly
    from stems.dhw_v2.env import DHWDistrictEnv, EnvConfig
    from stems.dhw_v2.filter import FilterConfig
    o = regime_overrides(regime, tol)
    env = DHWDistrictEnv(small_district(T), EnvConfig(
        plant=PlantConfig(brine="correlation", **o["plant"]),
        filter=FilterConfig(reserve_h=6, **o["filter"]) if filtered else None, **o["env"]))
    env.reset()
    c = ReactiveController(env.B) if ctl == "reactive" else ThermiaWeekly(env.B, dow=0, hour=3)
    while True:
        cmd = c.act(env)
        _, _, term, _, info = env.step_command(cmd)
        c.update(info["record"])
        if term:
            break
    return env, env.summary(), c


def test_env_regime_none_never_uses_the_disinfection_supply():
    env, s, _ = run_env("none")
    assert sum(s["dis_supply_hours"]) == 0 and s["regime"]["hygiene"] is False
    assert s["thermia_rule"]["top"]["programs"] == [0, 0]
    assert set(s["hygiene"]) >= {"none", "weekly_bottom20_60_1h", "daily_whole_60_1h", "w551_70_3min", "tap_50c"}


def test_env_daily_regime_owes_one_hold_per_day():
    env, s, _ = run_env("daily_whole_60_1h")
    assert s["regime"]["period_h"] == 24 and s["hold_rule"] == "all_nodes" and s["weeks_full"] == 4
    assert env.filter.cfg.hold_rule == "all_nodes" and env.filter.hours_left[0] == 24
    assert s["weeks_completed"] == [4, 4]
    assert s["hygiene"]["daily_whole_60_1h"]["periods_completed"] == [4, 4]
    assert s["hygiene"]["weekly_bottom20_60_1h"]["holds_completed"][0] >= 4


def test_env_weekly_regime_is_the_previous_default():
    env, s, _ = run_env("weekly_bottom20_60_1h")
    assert (s["regime"]["period_h"], s["regime"]["t_dis_c"], s["regime"]["hold_min"], s["hold_rule"]) == (168, 60.0, 60.0, "bottom20")
    assert env.filter.cfg.t_supply_dis_c == 65.0 and env.filter.model.hold_steps == 12


def test_env_w551_with_tolerance_runs_at_70_supply():
    env, s, _ = run_env("w551_70_3min", tol=1.0)
    assert env.filter.cfg.t_supply_dis_c == 70.0 and env.filter.cfg.t_dis_c == 69.0 and env.filter.model.hold_steps == 1
    assert s["hygiene"]["w551_70_3min_tol1K"]["holds_completed"][0] >= 1
    assert s["hygiene"]["w551_70_3min"]["holds_completed"] == [0, 0]
    assert sum(s["hours_extrap_supply"]) > 0


def test_thermia_weekly_program_no_hold_and_alarm():
    env, s, c = run_env("none", ctl="thermia_weekly", filtered=False)
    assert c.programs.tolist() == [1, 1] and c.done.tolist() == [1, 1] and c.alarms.tolist() == [0, 0]
    th = s["thermia_rule"]["top"]
    assert th["programs"] == [1, 1] and th["done_within_limit"] == [1, 1] and th["alarms"] == [0, 0]
    assert all(m <= 7 * 60 for m in th["max_minutes_to_target"])
    from stems.dhw_v2.controllers import ThermiaWeekly
    with pytest.raises(ValueError):
        ThermiaWeekly(1, end_sensor="bottom")
    from stems.dhw_v2.env import DHWDistrictEnv, EnvConfig
    env = DHWDistrictEnv(small_district(48), EnvConfig(plant=PlantConfig(brine="correlation"), hygiene=False))
    env.reset()
    c = ThermiaWeekly(env.B, dow=0, hour=3, target_c=99.0, end_sensor="weighted")
    n_dis = np.zeros(env.B)
    for _ in range(48):
        cmd = c.act(env)
        n_dis += cmd["dis"]
        env.step_command(cmd)
    assert c.alarms.tolist() == [1, 1] and c.done.tolist() == [0, 0] and n_dis.tolist() == [7, 7]
