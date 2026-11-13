from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.controllers import (ARMS, COMPARISON_POLICIES,
                                     LEARNING_COMPARISON_POLICIES, MECHANISMS,
                                     PLANT_MODELS, PRE_REGISTERED_ARMS,
                                     PRE_REGISTERED_POLICIES, build_controller)

COMPARISON_CONTROLLERS = ("sac", "dmappo", "mpc", "mpc-oracle", "maddpg", "marlisa",
                          "madcq", "metaems", "mappo-cc", "isac", "paper-stems")
MECHANISM_2X2 = tuple(f"mech-{m}+{p}"
                      for m in ("none", "lagrangian", "projection", "both")
                      for p in ("uniform", "linear", "exact"))
COMFORT_ARMS = ("rl+calibrated+comfort", "rbc+calibrated+comfort")
DEGRADATION_ARMS = ("rl+calibrated+degr-throughput", "rl+calibrated+degr-dod")

IMPLEMENTED_BY_BASELINES_TRACK = COMPARISON_CONTROLLERS
IMPLEMENTED_BY_CONSTRAINTS_TRACK = MECHANISM_2X2 + COMFORT_ARMS + DEGRADATION_ARMS
IMPLEMENTED_BY_METHOD_TRACKS = IMPLEMENTED_BY_BASELINES_TRACK + IMPLEMENTED_BY_CONSTRAINTS_TRACK
ALL_PRE_REGISTERED: tuple = ()


def test_nothing_remains_reserved():
    assert ALL_PRE_REGISTERED == ()
    assert tuple(PRE_REGISTERED_ARMS) == ()
    assert all(a.implemented for a in ARMS.values()), \
        [n for n, a in ARMS.items() if not a.implemented]


def test_the_pre_registered_set_is_exactly_what_we_declared():
    assert set(PRE_REGISTERED_ARMS) == set(ALL_PRE_REGISTERED)
    assert len(PRE_REGISTERED_ARMS) == 0
    assert len(IMPLEMENTED_BY_METHOD_TRACKS) == 11 + 16 == 27


@pytest.mark.parametrize("name", IMPLEMENTED_BY_METHOD_TRACKS)
def test_every_method_track_arm_resolves_and_is_implemented(name):
    assert name in ARMS and ARMS[name].name == name
    assert ARMS[name].implemented is True
    assert name not in PRE_REGISTERED_ARMS


def test_the_comparison_controllers_cover_table_one_plus_the_two_arms_it_lacked():
    assert len(COMPARISON_CONTROLLERS) == 11
    assert set(COMPARISON_POLICIES) == set(COMPARISON_CONTROLLERS)
    assert not set(PRE_REGISTERED_POLICIES) & set(COMPARISON_CONTROLLERS)
    assert COMPARISON_POLICIES["mappo-cc"] != COMPARISON_POLICIES["dmappo"]
    for policy, note in COMPARISON_POLICIES.items():
        assert len(note) > 30, f"{policy} needs a note saying what it is"
    for name in COMPARISON_CONTROLLERS:
        assert ARMS[name].implemented is True, f"{name} still reports no builder"
        assert ARMS[name].is_comparison is True
        assert name not in PRE_REGISTERED_ARMS


def test_the_two_model_predictive_arms_are_a_pair():
    assert ARMS["mpc"].policy == "mpc" and ARMS["mpc-oracle"].policy == "mpc-oracle"
    assert ARMS["mpc"].barrier == ARMS["mpc-oracle"].barrier == "calibrated"
    assert ARMS["mpc"].learns is False and ARMS["mpc-oracle"].learns is False
    assert {"mpc", "mpc-oracle"}.isdisjoint(LEARNING_COMPARISON_POLICIES)


def test_the_mechanism_cross_is_complete():
    assert len(MECHANISM_2X2) == 12
    seen = {(ARMS[n].mechanism, ARMS[n].plant_model) for n in MECHANISM_2X2}
    assert seen == {(m, p) for m in ("none", "lagrangian", "projection", "both")
                    for p in ("uniform", "linear", "exact")}
    for name in MECHANISM_2X2:
        arm = ARMS[name]
        assert arm.policy == "rl"
        assert arm.mechanism in MECHANISMS and arm.mechanism != "auto"


def test_plant_model_is_read_from_barrier_not_stored_twice():
    assert ARMS["rl+calibrated"].plant_model == "exact"
    assert ARMS["rl+linear"].plant_model == "linear"
    assert ARMS["rl+basic"].plant_model == "uniform"
    assert ARMS["rl"].plant_model is None
    for plant, barrier in PLANT_MODELS.items():
        assert ARMS[f"mech-both+{plant}"].barrier == barrier
        assert ARMS[f"mech-both+{plant}"].plant_model == plant


def test_the_comfort_and_degradation_flags_are_set():
    for name in COMFORT_ARMS:
        assert ARMS[name].comfort_barrier is True
    assert ARMS["rl+calibrated+degr-throughput"].degradation == "throughput"
    assert ARMS["rl+calibrated+degr-dod"].degradation == "throughput+dod"
    assert all(ARMS[n].learns for n in ("rl+calibrated+comfort",
                                        "rl+calibrated+degr-throughput"))
    assert ARMS["rbc+calibrated+comfort"].learns is False
    assert ARMS["rl+calibrated"].comfort_barrier is False
    assert ARMS["rl+calibrated"].degradation == "none"


def test_every_pre_existing_arm_still_builds_as_before():
    existing = ("idle", "idle+calibrated", "rbc", "rbc+calibrated", "rl", "rl+basic",
                "rl+linear", "rl+calibrated", "rl-res+calibrated", "rl+calibrated+pen",
                "rbc-offpeak+calibrated", "rbc-never+calibrated", "rl+calibrated+own",
                "rl+calibrated+floor", "hp-shift", "rl-hp", "rl+calibrated+meanpool")
    for name in existing:
        arm = ARMS[name]
        assert arm.implemented is True, f"{name} became unimplemented"
        assert arm.mechanism == "auto"
        assert name not in PRE_REGISTERED_ARMS


def test_arm_names_are_unique_and_match_their_keys():
    assert len({a.name for a in ARMS.values()}) == len(ARMS)
    for key, arm in ARMS.items():
        assert key == arm.name


def test_the_learning_comparison_arms_now_ask_for_a_training_budget():
    for name in COMPARISON_CONTROLLERS:
        expected = name not in ("mpc", "mpc-oracle")
        assert ARMS[name].learns is expected, (
            f"{name}.learns should be {expected}: a learner with no training budget "
            "enters the table untrained, and a budget given to the MPC is wasted")
