from __future__ import annotations

import numpy as np
import pytest

from stems.dhw_barrier import PHASE_DISINFECT, PHASE_NORMAL, PhaseBands
from stems.dhw_physics import accept_max, plugflow_charge, plugflow_draw
from stems.dhw_plant import DHWPlant, TankSpec
from stems.dhw_stratified import (DisinfectionBarrier, StratifiedStoreModel, hold_met,
                                  ReactiveController)
from stems.heatpump_hplib import (HPLIB_GENERIC, HplibUnit, brine_temperature_c,
                                  calibra_eco_8)
from stems.heatpump_thermia import ThermiaSpec, ThermiaUnit


def _tank(**over) -> TankSpec:
    kw = dict(capacity_kwh=10.0, t_cold_c=10.0, t_rated_c=60.0, loss_coefficient=0.005,
              nodes=2, charging="stratified", t_max_c=65.0, t_use_c=40.0)
    kw.update(over)
    return TankSpec(**kw)


def _fleet(B=5, seed=0, kappa=0.0):
    rng = np.random.default_rng(seed)
    tanks = [_tank(capacity_kwh=float(rng.uniform(5, 12)),
                   loss_coefficient=float(rng.uniform(0.001, 0.009)),
                   top_fraction=float(rng.choice([0.3, 0.4, 0.5, 0.6])), node_conduction=kappa)
             for _ in range(B)]
    specs = [calibra_eco_8(p_th_max_kw=float(rng.uniform(1.0, 8.0))) for _ in range(B)]
    plants = [DHWPlant(tanks[i], HplibUnit(specs[i]), initial_soc=float(rng.uniform(0.3, 1.05)))
              for i in range(B)]
    return tanks, specs, plants, rng


@pytest.mark.parametrize("kappa", [0.0, 0.2])
def test_energy_is_conserved_over_a_step(kappa):
    tanks, specs, plants, rng = _fleet(kappa=kappa)
    for _ in range(200):
        for p in plants:
            E0 = p.energy_kwh.sum()
            draw = float(rng.exponential(1.5)) if rng.random() < 0.5 else 0.0
            s = p.step(draw, float(rng.uniform(0, 10)), float(rng.uniform(0, 15)),
                       t_supply_c=float(rng.choice([45.0, 55.0, 65.0])))
            E1 = p.energy_kwh.sum()
            balance = E0 - s.loss_kwh - s.served_kwh + s.charge_thermal_kwh
            assert E1 == pytest.approx(balance, abs=1e-12)
            assert s.spill_kwh == 0.0


def test_the_charge_front_moves_down_monotonically():
    caps = np.array([5.0, 5.0])
    E0 = np.array([0.4 * 5.0, 0.1 * 5.0])
    soc_s = 0.9
    acc = float(accept_max(E0, caps, soc_s))
    prev = E0 / caps
    for Q in np.linspace(0.0, acc, 401):
        s = plugflow_charge(E0, caps, Q, soc_s) / caps
        assert s[0] >= s[1] - 1e-12, "the profile inverted"
        assert np.all(s >= prev - 1e-12), "more heat lowered a node"
        prev = s
    first = plugflow_charge(E0, caps, 5.0 * (soc_s - 0.1), soc_s) / caps
    assert first[0] == pytest.approx(soc_s, abs=1e-12)
    assert first[1] == pytest.approx(0.4, abs=1e-12)


@pytest.mark.parametrize("top_fraction", [0.3, 0.5, 0.7])
@pytest.mark.parametrize("t_supply", [55.0, 60.0, 65.0])
def test_a_full_charge_brings_both_nodes_to_the_supply_temperature(top_fraction, t_supply):
    p = DHWPlant(_tank(top_fraction=top_fraction), HplibUnit(calibra_eco_8(p_th_max_kw=50.0)),
                 initial_soc=0.3)
    p.step(0.0, 1e6, 10.0, t_supply_c=t_supply)
    assert p.node_temperatures_c() == pytest.approx([t_supply, t_supply], abs=1e-9)


def test_a_draw_takes_hot_water_from_the_top_and_cold_water_enters_the_bottom():
    caps = np.array([5.0, 5.0])
    E = caps * 0.9
    E2, served, unmet = plugflow_draw(E, caps, 4.0, 0.6)
    s = E2 / caps
    assert served == pytest.approx(4.0) and unmet == 0.0
    assert s[0] == pytest.approx(0.9, abs=1e-12), "the top still delivers 55 degC"
    assert s[1] < 0.9


def test_returned_water_cooler_than_the_top_enters_the_bottom():
    caps = np.array([5.0, 5.0])
    E = np.array([1.06 * 5.0, 0.4 * 5.0])
    E2 = plugflow_charge(E, caps, 1.0, 0.9) / caps
    assert E2[0] == pytest.approx(1.06, abs=1e-12)
    assert 0.4 < E2[1] <= 0.9 and E2[0] >= E2[1]


def test_water_colder_than_the_usable_temperature_is_not_delivered():
    caps = np.array([5.0, 5.0])
    E = caps * 0.5
    E2, served, unmet = plugflow_draw(E, caps, 2.0, 0.6)
    assert served == 0.0 and unmet == pytest.approx(2.0)
    assert np.array_equal(E2, E)


def _legacy(loss_first):
    tank = TankSpec(capacity_kwh=8.0, t_cold_c=10.0, t_rated_c=60.0, loss_coefficient=0.005,
                    nodes=1, loss_first=loss_first)
    spec = ThermiaSpec(power_min_kw=0.5, power_max_kw=20.0, eta_ref=0.45, t_sink_ref_c=35.0,
                       eta_slope_per_k=-0.004, capacity_ref_kw=20.0, t_source_ref_c=0.0,
                       capacity_slope_per_k_sink=0.0, capacity_slope_per_k_source=0.0,
                       t_sink_max_c=99.0, element_kw=0.0, element_efficiency=1.0,
                       cycling_penalty_kwh=0.0)
    return DHWPlant(tank, ThermiaUnit(spec), initial_soc=0.5)


def test_with_the_loss_first_the_post_step_state_reaches_the_set_temperature():
    old, new = _legacy(False), _legacy(True)
    old.step(0.0, 100.0, 5.0)
    new.step(0.0, 100.0, 5.0)
    assert old.soc == pytest.approx(1.0 - 0.005), "committed order: capped at 1 - loss"
    assert new.soc == 1.0


def test_the_stratified_charge_limit_is_the_heat_the_step_accepts():
    p = DHWPlant(_tank(), HplibUnit(calibra_eco_8()), initial_soc=0.95)
    lim = p.charge_limit_kwh(8.0, draw_kwh=3.0, t_supply_c=65.0)
    assert lim < p.max_charge_thermal_kwh(8.0, draw_kwh=3.0, t_supply_c=65.0) + 1e-12
    s = p.step(3.0, 1e3, 8.0, t_supply_c=65.0)
    assert s.charge_thermal_kwh == lim


@pytest.mark.parametrize("kappa", [0.0, 0.2])
def test_the_model_reproduces_the_stratified_plant_bit_for_bit(kappa):
    tanks, specs, plants, rng = _fleet(kappa=kappa)
    m = StratifiedStoreModel(tanks, specs)
    for _ in range(80):
        E = m.energy_of(plants)
        draw = np.where(rng.random(m.B) < 0.4, rng.exponential(1.5, m.B), 0.0)
        t_src = float(rng.uniform(0.0, 15.0))
        sup = rng.choice([55.0, 65.0], m.B)
        u = rng.uniform(0.0, 1.0, m.B)
        pred = m.step(E, u, draw, t_src, t_supply_c=sup)
        req = m.request_kwh(u, t_src, sup)
        steps = [p.step(draw[i], req[i], t_src, t_supply_c=sup[i]) for i, p in enumerate(plants)]
        assert np.array_equal(m.energy_of(plants), pred.energy_kwh)
        assert np.array_equal(np.array([s.unmet_kwh for s in steps]), pred.unmet_kwh)
        assert np.array_equal(np.array([s.post_draw_node_soc for s in steps]),
                              pred.post_draw_node_soc)
        assert np.array_equal(np.array([s.electricity_kwh for s in steps]),
                              pred.electricity_kwh)


def test_every_node_successor_is_non_decreasing_in_the_action():
    tanks, specs, plants, rng = _fleet(seed=3)
    m = StratifiedStoreModel(tanks, specs)
    grid = np.linspace(0.0, 1.0, 201)
    for _ in range(30):
        E = m.energy_of(plants) * rng.uniform(0.2, 1.0)
        d = rng.exponential(1.0, m.B)
        m.t_supply_c = rng.choice([55.0, 65.0], m.B)
        succ = np.stack([m.step(E, np.full(m.B, g), d, 8.0).node_soc for g in grid])
        assert np.all(np.diff(succ, axis=0) >= -1e-12)


def test_interval_endpoints_are_exact_inverses_on_the_protected_side():
    tanks, specs, plants, rng = _fleet(seed=4)
    m = StratifiedStoreModel(tanks, specs)
    bands = PhaseBands.from_temperatures(tanks[0], floor_c=45.0, ceiling_c=55.0,
                                         disinfect_c=65.0)
    binding = 0
    for _ in range(40):
        E = m.energy_of(plants) * rng.uniform(0.3, 1.0)
        d = np.where(rng.random(m.B) < 0.5, rng.exponential(1.5, m.B), 0.0)
        m.t_supply_c = np.full(m.B, 55.0)
        lo, hi = bands.lo_hi(np.zeros(m.B, dtype=int))
        a_lo, a_hi, status = m.node_bounds(E, lo, hi, d, 8.0)
        for b in range(m.B):
            if status[b, 0] == 0 and 0.0 < a_lo[b, 0] < 1.0:
                s = m.step(E, np.where(np.arange(m.B) == b, a_lo[b, 0], 0.0), d, 8.0).node_soc
                assert s[b, 0] >= lo[b, 0]
                assert s[b, 0] - lo[b, 0] < 1e-12
                binding += 1
    assert binding > 5


def test_the_hold_needs_every_node_through_a_full_step():
    p = DHWPlant(_tank(), HplibUnit(calibra_eco_8(p_th_max_kw=50.0)), initial_soc=0.5)
    s = p.step(0.0, 1e6, 8.0, t_supply_c=65.0)
    assert not hold_met(np.array(s.post_draw_node_soc), 1.0)
    s = p.step(0.5, 1e6, 8.0, t_supply_c=65.0)
    assert not hold_met(np.array(s.post_draw_node_soc), 1.0)
    s = p.step(0.0, 1e6, 8.0, t_supply_c=65.0)
    assert hold_met(np.array(s.post_draw_node_soc), 1.0)


def test_the_hplib_rows_are_reproduced():
    u = calibra_eco_8()
    assert float(u.capacity_kw(0.0, 35.0)) == pytest.approx(8.0)
    p1, p2, p3, p4 = HPLIB_GENERIC[2]["cop"]
    assert float(u.cop(0.0, 52.0)) == pytest.approx(p2 * 52 + p3 + p4 * -7.0)
    assert float(u.cop(0.0, 52.0)) == pytest.approx(3.0, abs=0.01)
    assert float(u.capacity_kw(5.0, 65.1)) == 0.0
    assert float(brine_temperature_c(0.0)) == pytest.approx(5.0647)


def _barrier(trigger, use_filter=True, B=2):
    tanks = [_tank(capacity_kwh=8.0) for _ in range(B)]
    specs = [calibra_eco_8() for _ in range(B)]
    plants = [DHWPlant(t, HplibUnit(s), initial_soc=0.9) for t, s in zip(tanks, specs)]
    m = StratifiedStoreModel(tanks, specs)
    bands = PhaseBands.from_temperatures(tanks[0], floor_c=45.0, ceiling_c=55.0,
                                         disinfect_c=65.0)
    b = DisinfectionBarrier(m, bands, trigger=trigger, use_filter=use_filter,
                            safety_stock_kwh=1.0, jit_reserve_kwh=1.0)
    return plants, m, b


@pytest.mark.parametrize("trigger", ["jit", "recourse", "timer"])
@pytest.mark.parametrize("use_filter", [False, True])
def test_every_trigger_completes_every_window_without_draws(trigger, use_filter):
    plants, m, b = _barrier(trigger, use_filter)
    ctl = ReactiveController(m.B)
    for t in range(3 * 168):
        E = m.energy_of(plants)
        T = E / m.caps * 50.0 + 10.0
        u, q = ctl.act(T[:, 0], T[:, 1], owed_d=b.in_d_owed)
        u, sup, _ = b.decide(u, q, E, np.zeros(m.B), 8.0,
                             draw_forecast=np.zeros((168, m.B)))
        req = m.request_kwh(u, 8.0, sup)
        steps = [p.step(0.0, req[i], 8.0, t_supply_c=sup[i]) for i, p in enumerate(plants)]
        b.observe(np.array([s.post_draw_node_soc for s in steps]))
    assert b.missed_windows.sum() == 0
    assert np.all(b.completions == 3)


def test_recourse_forces_earlier_when_draws_are_forecast():
    plants, m, b = _barrier("recourse")
    E = m.energy_of(plants)
    b.k[:] = 160.0
    quiet = b._recourse_steps(E, np.zeros(m.B), np.full(8, 8.0), np.zeros((8, m.B)))
    busy = b._recourse_steps(E, np.zeros(m.B), np.full(8, 8.0), np.full((8, m.B), 0.8))
    assert np.all(busy >= quiet)
    assert np.any(busy > quiet)
