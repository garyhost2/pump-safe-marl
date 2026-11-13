from __future__ import annotations

import numpy as np
import pytest

from stems.heatpump_thermia import ThermiaSpec, ThermiaUnit


def _spec(**over) -> ThermiaSpec:
    kw = dict(
        power_min_kw=1.0, power_max_kw=12.0,
        eta_ref=0.45, t_sink_ref_c=35.0, eta_slope_per_k=-0.004,
        capacity_ref_kw=12.0, t_source_ref_c=0.0,
        capacity_slope_per_k_sink=-0.12, capacity_slope_per_k_source=0.10,
        t_sink_max_c=65.0,
        element_kw=6.0, element_efficiency=1.0,
        cycling_penalty_kwh=0.0,
    )
    kw.update(over)
    return ThermiaSpec(**kw)


def test_capacity_falls_as_the_sink_gets_hotter():
    s = _spec()
    caps = [float(s.capacity_kw(0.0, t)) for t in (35.0, 45.0, 55.0, 60.0)]
    assert caps == sorted(caps, reverse=True), caps
    assert caps[0] > caps[-1], "no derating at all means the ceiling is the only physics"


def test_capacity_is_zero_at_and_above_the_ceiling():
    s = _spec(t_sink_max_c=55.0)
    assert float(s.capacity_kw(0.0, 54.9)) > 0.0
    assert float(s.capacity_kw(0.0, 55.0)) == 0.0
    assert float(s.capacity_kw(0.0, 70.0)) == 0.0


def test_capacity_rises_with_a_warmer_ground_loop():
    s = _spec()
    assert float(s.capacity_kw(5.0, 45.0)) > float(s.capacity_kw(-5.0, 45.0))


def test_cop_falls_as_the_sink_gets_hotter():
    s = _spec()
    cops = [float(s.cop(0.0, t)) for t in (35.0, 45.0, 55.0)]
    assert cops == sorted(cops, reverse=True), cops


def test_cop_and_capacity_are_independent_functions():
    base = _spec()
    steeper_capacity = _spec(capacity_slope_per_k_sink=-0.40)
    steeper_efficiency = _spec(eta_slope_per_k=-0.010)

    assert float(steeper_capacity.capacity_kw(0.0, 55.0)) < float(base.capacity_kw(0.0, 55.0))
    assert float(steeper_capacity.cop(0.0, 55.0)) == pytest.approx(float(base.cop(0.0, 55.0)))

    assert float(steeper_efficiency.cop(0.0, 55.0)) < float(base.cop(0.0, 55.0))
    assert float(steeper_efficiency.capacity_kw(0.0, 55.0)) == pytest.approx(
        float(base.capacity_kw(0.0, 55.0)))


def test_a_unit_can_be_efficient_at_a_sink_it_cannot_supply():
    s = _spec(t_sink_max_c=58.0)
    cop_below, cop_above = float(s.cop(0.0, 57.0)), float(s.cop(0.0, 59.0))
    assert cop_above == pytest.approx(cop_below, rel=0.1)
    assert cop_above > 1.5, "and is still a usable efficiency, not a degenerate one"
    assert float(s.capacity_kw(0.0, 57.0)) > 0.0
    assert float(s.capacity_kw(0.0, 59.0)) == 0.0, (
        "efficiency says nothing about availability - that is the whole separation")


def test_a_request_inside_the_envelope_does_not_cycle():
    u = ThermiaUnit(_spec())
    d = u.deliver(6.0, 0.0, 45.0)
    assert not d.cycling
    assert d.thermal_kwh == pytest.approx(6.0)
    assert d.element_thermal_kwh == 0.0


def test_a_request_below_minimum_modulation_cycles_and_counts_a_start():
    u = ThermiaUnit(_spec(power_min_kw=2.0))
    d = u.deliver(0.5, 0.0, 45.0)
    assert d.cycling and d.starts == 1
    assert d.duty == pytest.approx(0.25)
    assert d.thermal_kwh == pytest.approx(0.5)
    assert u.total_cycling_steps == 1


def test_cycling_is_counted_every_step_it_happens():
    u = ThermiaUnit(_spec(power_min_kw=2.0))
    for _ in range(5):
        u.deliver(0.5, 0.0, 45.0)
    assert u.total_cycling_steps == 5
    assert u.total_starts == 5, "each cycling step contains an off->on transition"


def test_a_start_is_counted_when_the_unit_comes_on_from_rest():
    u = ThermiaUnit(_spec())
    u.deliver(0.0, 0.0, 45.0)
    assert u.total_starts == 0
    u.deliver(6.0, 0.0, 45.0)
    assert u.total_starts == 1
    u.deliver(6.0, 0.0, 45.0)
    assert u.total_starts == 1, "staying on is not a new start"


def test_the_cycling_penalty_is_charged_per_start_when_priced():
    free = ThermiaUnit(_spec(power_min_kw=2.0, cycling_penalty_kwh=0.0))
    paid = ThermiaUnit(_spec(power_min_kw=2.0, cycling_penalty_kwh=0.05))
    d_free = free.deliver(0.5, 0.0, 45.0)
    d_paid = paid.deliver(0.5, 0.0, 45.0)
    assert d_paid.electricity_kwh == pytest.approx(d_free.electricity_kwh + 0.05)


def test_the_element_covers_what_the_compressor_cannot():
    u = ThermiaUnit(_spec(t_sink_max_c=55.0, element_kw=6.0, element_efficiency=1.0))
    d = u.deliver(3.0, 0.0, 60.0)
    assert d.compressor_thermal_kwh == 0.0
    assert d.element_thermal_kwh == pytest.approx(3.0)
    assert d.electricity_kwh == pytest.approx(3.0), "element at efficiency 1.0"


def test_without_the_element_a_request_above_the_ceiling_is_unmet():
    u = ThermiaUnit(_spec(t_sink_max_c=55.0))
    d = u.deliver(3.0, 0.0, 60.0, allow_element=False)
    assert d.thermal_kwh == 0.0
    assert d.unmet_thermal_kwh == pytest.approx(3.0)


def test_a_ceiling_below_the_disinfection_target_makes_the_lift_resistive():
    low = ThermiaUnit(_spec(t_sink_max_c=55.0))
    high = ThermiaUnit(_spec(t_sink_max_c=65.0))
    at_60_low = low.deliver(2.0, 0.0, 60.0)
    at_60_high = high.deliver(2.0, 0.0, 60.0)
    assert at_60_low.element_thermal_kwh == pytest.approx(2.0)
    assert at_60_high.element_thermal_kwh == 0.0
    assert at_60_high.electricity_kwh < at_60_low.electricity_kwh, (
        "a unit that reaches the set point on the compressor must be cheaper there")


def test_every_physical_parameter_is_required():
    import dataclasses
    required = {f.name for f in dataclasses.fields(ThermiaSpec)
                if f.default is dataclasses.MISSING
                and f.default_factory is dataclasses.MISSING}
    assert set(ThermiaSpec.OPEN_QUESTIONS) <= required, (
        "a physical parameter with a default is a number this repository invented")


def test_unsourced_lists_what_the_run_is_guessing():
    s = _spec()
    assert set(s.unsourced) == set(ThermiaSpec.OPEN_QUESTIONS)
    s = _spec(provenance={"power_min_kw": "datasheet rev 3, table 2"})
    assert "power_min_kw" not in s.unsourced
    assert "eta_ref" in s.unsourced


def test_describe_carries_the_unsourced_list_for_the_run_record():
    u = ThermiaUnit(_spec())
    d = u.describe()
    assert d["kind"] == "ThermiaUnit"
    assert set(d["unsourced"]) == set(ThermiaSpec.OPEN_QUESTIONS)


@pytest.mark.parametrize("bad,match", [
    ({"power_min_kw": 12.0, "power_max_kw": 12.0}, "fixed-speed"),
    ({"eta_ref": 1.5}, "Carnot efficiency"),
    ({"eta_slope_per_k": 0.01}, "more efficient"),
    ({"capacity_slope_per_k_sink": 0.5}, "rises with sink temperature"),
    ({"capacity_slope_per_k_source": -0.5}, ">= 0"),
    ({"capacity_ref_kw": 0.0}, "must be positive"),
    ({"cycling_penalty_kwh": -1.0}, ">= 0"),
])
def test_physically_impossible_parameters_are_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        _spec(**bad)


def test_no_derating_must_be_declared_rather_than_implied():
    s = _spec(capacity_slope_per_k_sink=0.0,
              provenance={"capacity_slope_per_k_sink": "chosen: no derating, stated"})
    assert float(s.capacity_kw(0.0, 35.0)) == float(s.capacity_kw(0.0, 54.0))
    assert "capacity_slope_per_k_sink" not in s.unsourced


def test_the_cop_clamp_is_reported_rather_than_absorbed():
    u = ThermiaUnit(_spec())
    d = u.deliver(1.0, 44.9, 45.0)
    assert d.cop_clamped
    assert d.cop_used <= 8.0
