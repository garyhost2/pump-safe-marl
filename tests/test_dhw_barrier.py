from __future__ import annotations

import itertools

import numpy as np
import pytest

from stems.battery import BatteryModel
from stems.cbf import CBFShield
from stems.dhw_barrier import (CEILING_UNAVOIDABLE, FLOOR_UNREACHABLE, NODE_OK,
                               PHASE_DISINFECT, PHASE_NORMAL, CyclePhaseBarrier,
                               DHWStoreModel, HourOfDayDrawForecast, PhaseBands)
from stems.dhw_plant import DHWPlant, TankSpec
from stems.flexibility import FlexibleLoad
from stems.heatpump_thermia import ThermiaSpec, ThermiaUnit
from stems.legionella import LegionellaCycleBarrier, LegionellaSpec, LegionellaStack


def _spec(pmax: float = 2.0, **over) -> ThermiaSpec:
    kw = dict(power_min_kw=0.3 * pmax, power_max_kw=pmax, eta_ref=0.45, t_sink_ref_c=35.0,
              eta_slope_per_k=-0.004, capacity_ref_kw=pmax, t_source_ref_c=0.0,
              capacity_slope_per_k_sink=-0.05, capacity_slope_per_k_source=0.03,
              t_sink_max_c=58.0, element_kw=0.0, element_efficiency=1.0,
              cycling_penalty_kwh=0.0)
    kw.update(over)
    return ThermiaSpec(**kw)


def _tank(**over) -> TankSpec:
    kw = dict(capacity_kwh=8.0, t_cold_c=10.0, t_rated_c=60.0, loss_coefficient=0.004,
              nodes=1)
    kw.update(over)
    return TankSpec(**kw)


def _fleet(nodes, charge_node, kappa, element, B=4, seed=0):
    rng = np.random.default_rng(seed)
    tanks = [_tank(capacity_kwh=float(rng.uniform(4, 12)),
                   loss_coefficient=float(rng.uniform(0.001, 0.009)), nodes=nodes,
                   top_fraction=0.5 if i % 2 else 0.4, node_conduction=kappa,
                   charge_node=charge_node, allow_inversion=True) for i in range(B)]
    specs = [_spec(float(rng.uniform(0.3, 4.0)), element_kw=element) for _ in range(B)]
    plants = [DHWPlant(tanks[i], ThermiaUnit(specs[i]),
                       initial_soc=float(rng.uniform(0.3, 1.0))) for i in range(B)]
    return tanks, specs, plants, rng


CONFIGS = [(1, "top", 0.0, 0.0), (1, "top", 0.0, 1.5), (2, "top", 0.0, 0.0),
           (2, "top", 0.2, 1.5), (2, "bottom", 0.2, 0.0), (2, "bottom", 0.0, 1.5)]


@pytest.mark.parametrize("nodes,charge_node,kappa,element", CONFIGS)
@pytest.mark.parametrize("allow", [True, False])
def test_the_model_reproduces_the_plant_bit_for_bit(nodes, charge_node, kappa, element,
                                                    allow):
    tanks, specs, plants, rng = _fleet(nodes, charge_node, kappa, element)
    m = DHWStoreModel(tanks, specs, allow_element=allow)
    for _ in range(60):
        E = m.energy_of(plants)
        draw = np.where(rng.random(m.B) < 0.4, rng.exponential(1.5, m.B), 0.0)
        t_src = float(rng.uniform(-5.0, 35.0))
        u = rng.uniform(0.0, 1.0, m.B)
        pred = m.step(E, u, draw, t_src)
        req = m.request_kwh(u)
        steps = [p.step(draw[i], req[i], t_src, allow_element=allow)
                 for i, p in enumerate(plants)]
        assert np.array_equal(m.energy_of(plants), pred.energy_kwh)
        assert np.array_equal(np.array([s.unmet_kwh for s in steps]), pred.unmet_kwh)
        assert np.array_equal(np.array([s.spill_kwh for s in steps]), pred.spill_kwh)


@pytest.mark.parametrize("nodes,charge_node,kappa,element", CONFIGS)
def test_interval_endpoints_are_exact_inverses_on_the_protected_side(nodes, charge_node,
                                                                     kappa, element):
    tanks, specs, plants, rng = _fleet(nodes, charge_node, kappa, element, seed=1)
    m = DHWStoreModel(tanks, specs)
    lo = np.full((m.B, nodes), 0.70)
    hi = np.full((m.B, nodes), 0.90)
    worst, binding = 0.0, 0
    for _ in range(40):
        E = np.stack([p.tank.node_capacities_kwh for p in plants]) * rng.uniform(
            0.4, 1.0, (m.B, nodes))
        draw = np.where(rng.random(m.B) < 0.5, rng.exponential(1.0, m.B), 0.0)
        t_src = float(rng.uniform(-5.0, 35.0))
        a_lo, a_hi, status = m.node_bounds(E, lo, hi, draw, t_src)
        for a, bound, side in ((a_lo, lo, "lo"), (a_hi, hi, "hi")):
            for j in range(nodes):
                for i, p in enumerate(plants):
                    if status[i, j] != NODE_OK:
                        continue
                    interior = a[i, j] > 0.0 if side == "lo" else a[i, j] < 1.0
                    if not interior:
                        continue
                    q = DHWPlant(p.tank, ThermiaUnit(p.unit.spec))
                    q.energy_kwh = E[i].copy()
                    soc = q.step(draw[i], m.request_kwh(a[:, j])[i], t_src).node_soc[j]
                    if side == "lo":
                        assert soc >= bound[i, j], "the floor endpoint must satisfy the floor"
                    else:
                        assert soc <= bound[i, j], "the ceiling endpoint must satisfy it"
                    worst = max(worst, abs(soc - bound[i, j]))
                    binding += 1
    assert binding > 20
    assert worst < 1e-14, f"one-step inverse error {worst:.3e} is not machine precision"


def test_the_charge_limit_is_read_at_the_post_draw_sink():
    tank = _tank(loss_coefficient=0.0)
    spec = _spec(3.0, t_sink_max_c=58.0)
    p = DHWPlant(tank, ThermiaUnit(spec), initial_soc=0.98)
    m = DHWStoreModel([tank], [spec])
    assert p.max_charge_thermal_kwh(5.0, draw_kwh=0.0) == 0.0
    assert p.max_charge_thermal_kwh(5.0, draw_kwh=2.0) > 0.0
    E = m.energy_of([p])
    limit = m.charge_limit_kwh(E, 2.0, 5.0)[0]
    assert limit > 0.0
    assert p.charge_limit_kwh(5.0, draw_kwh=2.0) == limit
    s = p.step(2.0, 10.0, 5.0)
    assert s.charge_thermal_kwh == limit


@pytest.mark.parametrize("charge_node,kappa", [("top", 0.0), ("top", 0.3), ("bottom", 0.3)])
def test_every_node_successor_is_non_decreasing_in_the_action(charge_node, kappa):
    tanks, specs, plants, rng = _fleet(2, charge_node, kappa, 1.5, seed=2)
    m = DHWStoreModel(tanks, specs)
    us = np.linspace(0.0, 1.0, 201)
    for _ in range(10):
        E = m.energy_of(plants) * rng.uniform(0.3, 1.0, (m.B, 2))
        draw = rng.exponential(1.0, m.B)
        socs = np.stack([m.next_node_soc(E, np.full(m.B, u), draw, 10.0) for u in us])
        assert np.all(np.diff(socs, axis=0) >= -1e-15)


def test_an_unreachable_floor_is_reported_with_its_fallback_not_a_midpoint():
    tank, spec = _tank(), _spec(0.5)
    m = DHWStoreModel([tank], [spec])
    E = np.array([[0.72 * 8.0]])
    res = m.project(E, [0.2], [[0.70]], [[0.90]], [3.0], 10.0)
    assert res.empty[0] and not res.conflict[0]
    assert res.status[0, 0] == FLOOR_UNREACHABLE
    assert res.action[0] == 1.0, "floor_first commands the best effort toward the floor"
    assert res.floor_breach[0, 0] > 0.0, "the breach the fallback cannot avoid is recorded"


def test_a_ceiling_the_store_already_exceeds_is_reported():
    tank, spec = _tank(loss_coefficient=0.0), _spec(2.0)
    m = DHWStoreModel([tank], [spec])
    res = m.project(np.array([[0.99 * 8.0]]), [0.5], [[0.70]], [[0.90]], [0.0], 10.0)
    assert res.empty[0] and res.status[0, 0] == CEILING_UNAVOIDABLE
    assert res.action[0] == 0.0


def test_two_nodes_whose_intervals_do_not_overlap_are_a_reported_conflict():
    tank = _tank(capacity_kwh=8.0, loss_coefficient=0.0, nodes=2, node_conduction=0.5)
    m = DHWStoreModel([tank], [_spec(4.0)])
    E = np.array([[0.6 * 4.0, 0.2 * 4.0]])
    lo, hi = np.array([[-np.inf, 0.38]]), np.array([[0.70, np.inf]])
    a_lo, a_hi, status = m.node_bounds(E, lo, hi, [0.0], 10.0)
    assert (status == NODE_OK).all()
    floor_first = m.project(E, [0.0], lo, hi, [0.0], 10.0, on_empty="floor_first")
    ceil_first = m.project(E, [0.0], lo, hi, [0.0], 10.0, on_empty="ceiling_first")
    for r in (floor_first, ceil_first):
        assert r.empty[0] and r.conflict[0]
    assert floor_first.action[0] == pytest.approx(a_lo[0, 1])
    assert floor_first.ceiling_breach[0, 0] > 0.0 and floor_first.floor_breach[0, 1] == 0.0
    assert ceil_first.action[0] == pytest.approx(a_hi[0, 0])
    assert ceil_first.floor_breach[0, 1] > 0.0 and ceil_first.ceiling_breach[0, 0] == 0.0


def test_safe_interval_gives_a_tank_a_band_inside_the_framework():
    tank, spec = _tank(), _spec(3.0)
    m = DHWStoreModel([tank], [spec])
    p = DHWPlant(tank, ThermiaUnit(spec), initial_soc=0.85)
    E = m.energy_of([p])
    load = FlexibleLoad(name="dhw", kind="dhw", action_index=0, plant=m,
                        soc_fn=lambda obs: (E / m.caps)[:, 0], band=(0.70, 0.90))
    assert load.has_band
    m.set_step_inputs(draw_kwh=[0.4], t_source_c=10.0, energy_kwh=E)
    out = load.project_band(np.array([[1.0]], dtype=np.float32), [np.zeros(4)])
    assert out.dtype == np.float32 and out[0, 0] < 1.0
    soc = p.step(0.4, m.request_kwh(out[:, 0].astype(np.float64))[0], 10.0).node_soc[0]
    assert 0.70 <= soc <= 0.90, "the float32 action, rounded inward, keeps the band"
    assert m.last_projection is not None and not m.last_projection.empty[0]


def _barrier(soc0=0.8, nodes=1, pmax=6.0, trigger="jit", **kw):
    tank = _tank(nodes=nodes, loss_coefficient=0.004)
    spec = _spec(pmax, t_sink_max_c=65.0, capacity_slope_per_k_sink=0.0)
    p = DHWPlant(tank, ThermiaUnit(spec), initial_soc=soc0)
    m = DHWStoreModel([tank], [spec], allow_element=False)
    bands = PhaseBands.from_temperatures(tank, floor_c=45.0, ceiling_c=55.0,
                                         disinfect_c=60.0)
    fc = HourOfDayDrawForecast(1)
    b = CyclePhaseBarrier(m, bands, soc_legionella=1.0, period_steps=168.0,
                          trigger=trigger, forecaster=fc, **kw)
    return p, m, b


def test_the_bands_are_the_temperatures_on_the_top_node():
    tank = _tank(nodes=2)
    bands = PhaseBands.from_temperatures(tank, floor_c=45.0, ceiling_c=55.0,
                                         disinfect_c=60.0)
    np.testing.assert_allclose(bands.normal[0], [0.70, 0.90])
    np.testing.assert_allclose(bands.disinfect[0], [0.70, 1.00])
    assert bands.normal[1, 0] == -np.inf


def test_entry_into_disinfection_is_admissible_from_the_comfort_band():
    p, m, b = _barrier(soc0=0.85)
    acc = b.request_phase([PHASE_DISINFECT], m.energy_of([p]) / m.caps)
    assert acc[0] and b.q[0] == PHASE_DISINFECT and b.refusals[0] == 0


def test_an_exit_above_55_degc_is_refused_recorded_and_keeps_the_relaxed_ceiling():
    p, m, b = _barrier(soc0=0.95)
    b.q[:] = PHASE_DISINFECT
    acc = b.request_phase([PHASE_NORMAL], m.energy_of([p]) / m.caps)
    assert not acc[0] and b.q[0] == PHASE_DISINFECT
    assert b.refusals[0] == 1 and b.refusals_above_ceiling[0] == 1
    assert b.refusal_log[-1][2:] == ("D", "N", "above_ceiling")
    lo, hi = b.bands.lo_hi(b.q)
    assert hi[0, 0] == pytest.approx(1.0), "the relaxed ceiling stays in force"


def test_an_exit_at_or_below_55_degc_is_admitted():
    p, m, b = _barrier(soc0=0.90)
    b.q[:] = PHASE_DISINFECT
    assert b.request_phase([PHASE_NORMAL], m.energy_of([p]) / m.caps)[0]
    assert b.q[0] == PHASE_NORMAL


def test_a_state_below_the_floor_cannot_change_phase_under_the_strict_guard():
    p, m, b = _barrier(soc0=0.60)
    assert not b.request_phase([PHASE_DISINFECT], m.energy_of([p]) / m.caps)[0]
    assert b.refusals_below_floor[0] == 1
    p2, m2, b2 = _barrier(soc0=0.60, guard_includes_floor=False)
    assert b2.request_phase([PHASE_DISINFECT], m2.energy_of([p2]) / m2.caps)[0]


def test_arbitrary_phase_requests_never_leave_the_set_without_a_reported_empty_step():
    rng = np.random.default_rng(3)
    p, m, b = _barrier(soc0=0.80, pmax=6.0)
    checked = 0
    for t in range(400):
        E = m.energy_of([p])
        draw = np.array([rng.exponential(0.6) if rng.random() < 0.3 else 0.0])
        b.request_phase(rng.integers(0, 2, 1), E / m.caps)
        res = b.project(rng.uniform(0.0, 1.0, 1), E, draw, 10.0)
        p.step(draw[0], m.request_kwh(res.action)[0], 10.0, allow_element=False)
        soc = p.node_soc[0]
        lo, hi = b.bands.lo_hi(b.q)
        if not res.empty[0]:
            assert lo[0, 0] <= soc <= hi[0, 0]
            checked += 1
        b.observe(np.array([soc]))
    assert checked > 300
    assert b.transitions[0] >= 3 and b.refusals[0] > 0


def test_a_completed_cycle_requests_the_exit_until_the_guard_admits_it():
    p, m, b = _barrier(soc0=0.85, pmax=6.0)
    b.q[:] = PHASE_DISINFECT
    E = m.energy_of([p])
    res = b.project([1.0], E, [0.0], 10.0)
    st = m.step(E, res.action, [0.0], 10.0)
    p.step(0.0, m.request_kwh(res.action)[0], 10.0, allow_element=False)
    out = b.observe(st.post_charge_sink_soc)
    assert out["completed"][0] and b.cycles_completed[0] == 1
    assert b.pending_exit[0]
    want = b.desired_phase(None, np.array([False]))
    assert want[0] == PHASE_NORMAL
    assert not b.request_phase(want, m.energy_of([p]) / m.caps)[0]


def test_the_recourse_plan_is_longer_when_the_forecast_expects_draws():
    p, m, b = _barrier(soc0=0.75, pmax=1.0, trigger="recourse")
    E = m.energy_of([p])
    quiet = b.deadline(E, [0.0], 10.0, forecast=np.zeros((168, 1)))["steps_needed"][0]
    busy = b.deadline(E, [0.0], 10.0, forecast=np.full((168, 1), 0.5))["steps_needed"][0]
    assert busy > quiet


def test_the_legionella_barrier_still_defaults_to_the_just_in_time_trigger():
    spec = LegionellaSpec(capacity_kwh=[4.0], heat_source_kw=[1.5], t_cold_c=15.0,
                          t_rated_c=60.0, t_normal_c=50.0, t_legionella_c=60.0,
                          cop_legionella=2.0, element_efficiency=1.0,
                          t_heat_pump_max_c=55.0)
    stack = LegionellaStack.build(spec, action_index=0)
    assert stack.barrier.trigger == "jit" and stack.barrier.forecaster is None
    with pytest.raises(ValueError, match="forecast"):
        LegionellaStack.build(spec, action_index=0, trigger="recourse")


def _legionella_misses(trigger: str) -> int:
    spec = LegionellaSpec(capacity_kwh=[4.0], heat_source_kw=[1.5], t_cold_c=15.0,
                          t_rated_c=60.0, t_normal_c=50.0, t_legionella_c=60.0,
                          cop_legionella=2.0, element_efficiency=1.0,
                          t_heat_pump_max_c=55.0, period_hours=24.0)
    fc = HourOfDayDrawForecast(1) if trigger == "recourse" else None
    stack = LegionellaStack.build(spec, action_index=0, trigger=trigger, forecaster=fc)
    for t in range(24 * 8):
        draw = np.array([1.0 if t % 24 >= 21 else 0.0])
        soc = stack.tank.soc
        need = np.maximum(spec.soc_normal - soc, 0.0) * spec.capacity_kwh + draw
        a = np.clip(need / spec.heat_source_kw, 0.0, 1.0).astype(np.float32)[:, None]
        a = stack.load.project(a, [np.zeros(40, dtype=np.float32)])
        stack.observe(a[:, 0], draw)
    return int(stack.barrier.missed_windows.sum())


def test_recourse_absorbs_a_burst_the_just_in_time_trigger_does_not():
    jit, recourse = _legionella_misses("jit"), _legionella_misses("recourse")
    assert jit > 0, "the burst in the forced hours defeats a just-in-time schedule"
    assert recourse < jit


def test_the_hour_of_day_forecast_is_causal():
    fc = HourOfDayDrawForecast(1, period=24)
    assert fc.forecast(3)[0, 0] == 0.0
    for t in range(24):
        fc.update([float(t)])
    np.testing.assert_allclose(fc.forecast(2)[:, 0], [0.0, 1.0])


def test_the_battery_interval_default_on_crossed_bounds_is_still_the_midpoint():
    bm = BatteryModel.linear(np.array([0.2]))
    old = bm.safe_interval(np.array([0.5]), np.array([0.6]), np.array([0.4]))
    assert bm.last_crossed[0], "lo above hi crosses, and the model now says so"
    assert old[0][0] == old[1][0] == pytest.approx(0.0, abs=1e-6)
    lo_f, hi_f = bm.safe_interval(np.array([0.5]), np.array([0.6]), np.array([0.4]),
                                  on_crossed="floor")
    assert lo_f[0] == hi_f[0] == pytest.approx(0.5, abs=1e-5)
    with pytest.raises(ValueError):
        bm.safe_interval(np.array([0.5]), 0.1, 0.9, on_crossed="nearest")


def test_the_shield_counts_crossings_and_defaults_to_the_old_call():
    assert CBFShield(num_buildings=1, soc_rate=np.array([0.2])).on_crossed == "midpoint"
    with pytest.raises(ValueError):
        CBFShield(num_buildings=1, soc_rate=np.array([0.2]), on_crossed="bad")
