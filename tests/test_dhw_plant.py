from __future__ import annotations

import numpy as np
import pytest

from stems.dhw_plant import DHWPlant, DHWPlantFleet, TankSpec
from stems.heatpump_thermia import ThermiaSpec, ThermiaUnit


def _unit(power_max_kw: float = 2.0, **over) -> ThermiaUnit:
    kw = dict(power_min_kw=0.5, power_max_kw=power_max_kw,
              eta_ref=0.45, t_sink_ref_c=35.0, eta_slope_per_k=-0.004,
              capacity_ref_kw=power_max_kw, t_source_ref_c=0.0,
              capacity_slope_per_k_sink=-0.01, capacity_slope_per_k_source=0.02,
              t_sink_max_c=65.0, element_kw=0.0, element_efficiency=1.0,
              cycling_penalty_kwh=0.0)
    kw.update(over)
    return ThermiaUnit(ThermiaSpec(**kw))


def _tank(**over) -> TankSpec:
    kw = dict(capacity_kwh=8.0, t_cold_c=10.0, t_rated_c=60.0,
              loss_coefficient=0.005, nodes=1)
    kw.update(over)
    return TankSpec(**kw)


def test_a_burst_larger_than_the_store_leaves_hot_water_unmet():
    p = DHWPlant(_tank(), _unit(), initial_soc=0.1)
    s = p.step(draw_kwh=5.0, charge_request_kwh=0.0, t_source_c=5.0)
    assert s.unmet_kwh > 0.0
    assert s.served_kwh == pytest.approx(0.8)
    assert s.unmet_kwh == pytest.approx(4.2)


def test_the_source_cannot_refill_within_one_step():
    p = DHWPlant(_tank(), _unit(power_max_kw=2.0), initial_soc=0.0)
    s = p.step(draw_kwh=0.0, charge_request_kwh=8.0, t_source_c=5.0)
    assert s.charge_thermal_kwh <= 2.0 + 1e-9
    assert p.soc < 0.3


def test_a_full_tank_and_no_draw_matches_citylearn_when_the_limit_does_not_bind():
    p = DHWPlant(_tank(loss_coefficient=0.0), _unit(power_max_kw=10.0), initial_soc=0.5)
    s = p.step(draw_kwh=1.0, charge_request_kwh=1.0, t_source_c=5.0)
    assert s.unmet_kwh == 0.0
    assert s.charge_thermal_kwh == pytest.approx(1.0)


def test_unmet_accumulates_over_a_fleet():
    fleet = DHWPlantFleet([DHWPlant(_tank(), _unit(), initial_soc=0.05) for _ in range(3)])
    fleet.step([4.0, 4.0, 4.0], [0.0, 0.0, 0.0], 5.0)
    r = fleet.report()
    assert r["unmet_kwh"] > 0.0
    assert r["buildings"] == 3 and r["steps"] == 1


def test_two_nodes_let_the_top_be_hot_while_the_bottom_is_cold():
    t = _tank(nodes=2, top_fraction=0.5, loss_coefficient=0.0)
    p = DHWPlant(t, _unit(power_max_kw=10.0), initial_soc=0.0)
    p.energy_kwh[0] = p._caps[0]
    assert p.top_temperature_c() == pytest.approx(60.0)
    assert p.bottom_temperature_c() == pytest.approx(10.0)
    assert p.soc == pytest.approx(0.5)
    assert t.temperature_of_soc(p.soc) == pytest.approx(35.0)


def test_a_draw_empties_the_top_node_first():
    t = _tank(nodes=2, loss_coefficient=0.0)
    p = DHWPlant(t, _unit(power_max_kw=10.0), initial_soc=1.0)
    s = p.step(draw_kwh=2.0, charge_request_kwh=0.0, t_source_c=5.0)
    assert s.node_soc[0] < s.node_soc[1], "the tap draws from the top"


def test_charging_the_top_node_raises_the_delivered_temperature_first():
    t = _tank(nodes=2, loss_coefficient=0.0, charge_node="top")
    p = DHWPlant(t, _unit(power_max_kw=10.0), initial_soc=0.0)
    before = p.top_temperature_c()
    s = p.step(draw_kwh=0.0, charge_request_kwh=2.0, t_source_c=5.0)
    assert p.top_temperature_c() > before
    assert s.node_soc[1] == pytest.approx(0.0), "the bottom is untouched"


@pytest.mark.parametrize("top_fraction", [0.3, 0.4, 0.5, 0.6, 0.7])
@pytest.mark.parametrize("kappa", [0.05, 0.2, 1.0])
def test_conduction_conserves_energy_for_any_node_split(top_fraction, kappa):
    t = _tank(nodes=2, top_fraction=top_fraction, loss_coefficient=0.0,
              node_conduction=kappa)
    p = DHWPlant(t, _unit(power_max_kw=10.0), initial_soc=0.0)
    p.energy_kwh[0] = p._caps[0] * 0.9
    p.energy_kwh[1] = p._caps[1] * 0.1
    before = float(p.energy_kwh.sum())
    p.step(draw_kwh=0.0, charge_request_kwh=0.0, t_source_c=5.0)
    assert float(p.energy_kwh.sum()) == pytest.approx(before, abs=1e-12)


def test_full_conduction_equalises_the_nodes_for_any_split():
    for f in (0.3, 0.5, 0.7):
        t = _tank(nodes=2, top_fraction=f, loss_coefficient=0.0, node_conduction=1.0)
        p = DHWPlant(t, _unit(power_max_kw=10.0), initial_soc=0.0)
        p.energy_kwh[0] = p._caps[0] * 0.9
        p.energy_kwh[1] = p._caps[1] * 0.1
        p.step(draw_kwh=0.0, charge_request_kwh=0.0, t_source_c=5.0)
        top, bottom = p.node_soc
        assert top == pytest.approx(bottom, abs=1e-12), f


def test_conduction_moves_heat_from_the_hot_node_to_the_cold_one():
    t = _tank(nodes=2, loss_coefficient=0.0, node_conduction=0.2)
    p = DHWPlant(t, _unit(power_max_kw=10.0), initial_soc=0.0)
    p.energy_kwh[0] = p._caps[0]
    top_before, bottom_before = p.node_soc
    p.step(draw_kwh=0.0, charge_request_kwh=0.0, t_source_c=5.0)
    top_after, bottom_after = p.node_soc
    assert top_after < top_before and bottom_after > bottom_before


def test_one_node_is_still_available_as_the_reference():
    p = DHWPlant(_tank(nodes=1), _unit(), initial_soc=0.5)
    assert p.top_temperature_c() == p.bottom_temperature_c(), "mixed tank: one number"


def test_the_charge_limit_falls_as_the_store_gets_hotter():
    p = DHWPlant(_tank(), _unit(power_max_kw=10.0, capacity_slope_per_k_sink=-0.2),
                 initial_soc=0.2)
    cold = p.max_charge_thermal_kwh(5.0, draw_kwh=0.0)
    p.energy_kwh[:] = p._caps * 0.95
    hot = p.max_charge_thermal_kwh(5.0, draw_kwh=0.0)
    assert hot < cold, "a hotter store presents a hotter sink and gets less power"


def test_a_store_above_the_sink_ceiling_cannot_be_charged_by_the_compressor():
    p = DHWPlant(_tank(), _unit(power_max_kw=10.0, t_sink_max_c=50.0), initial_soc=0.95)
    assert p.top_temperature_c() > 50.0
    assert p.max_charge_thermal_kwh(5.0, draw_kwh=0.0, allow_element=False) == 0.0


def test_charging_the_bottom_of_a_two_node_tank_must_be_opted_into():
    with pytest.raises(ValueError, match="buoyancy"):
        TankSpec(capacity_kwh=8.0, t_cold_c=10.0, t_rated_c=60.0,
                 loss_coefficient=0.005, nodes=2, charge_node="bottom")
    ok = TankSpec(capacity_kwh=8.0, t_cold_c=10.0, t_rated_c=60.0,
                  loss_coefficient=0.005, nodes=2, charge_node="bottom",
                  allow_inversion=True)
    assert ok.charge_node == "bottom"


def test_spill_is_reported_when_the_store_cannot_hold_the_heat():
    p = DHWPlant(_tank(loss_coefficient=0.0), _unit(power_max_kw=10.0), initial_soc=0.95)
    s = p.step(draw_kwh=0.0, charge_request_kwh=5.0, t_source_c=5.0)
    assert s.spill_kwh > 0.0
    assert p.soc == pytest.approx(1.0)


def test_the_soc_temperature_mapping_matches_the_legionella_convention():
    t = _tank()
    assert t.soc_of_temperature(45.0) == pytest.approx(0.70)
    assert t.soc_of_temperature(55.0) == pytest.approx(0.90)
    assert t.soc_of_temperature(60.0) == pytest.approx(1.00)
    assert t.temperature_of_soc(0.95) == pytest.approx(57.5)


@pytest.mark.parametrize("bad,match", [
    ({"nodes": 3}, "nodes must be 1 or 2"),
    ({"capacity_kwh": 0.0}, "capacity_kwh must be positive"),
    ({"t_rated_c": 10.0}, "must exceed t_cold_c"),
    ({"loss_coefficient": 1.0}, "per-step fraction"),
    ({"top_fraction": 0.0}, "strictly in"),
])
def test_impossible_tanks_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        _tank(**bad)
