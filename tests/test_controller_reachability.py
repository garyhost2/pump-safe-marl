from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import stems
from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment

from experiments.controllers import ARMS, COMPARISON_POLICIES, build_controller

NOT_CONTROLLERS = {
    "STEMSConfig", "STEMSEnvironment", "BuildingGraph", "STEncoder", "CBFShield",
    "STEMSReward", "MetricsCalculator", "ReplayBuffer", "EpisodeBuffer", "HistoryBuffer",
    "CausalForecaster", "OracleForecaster", "CONTROLLERS_REACHABLE_FROM_ARMS",
}


def _exported_controllers():
    return sorted(set(stems.__all__) - NOT_CONTROLLERS - stems.FRAMEWORK_EXPORTS
                  - {"FRAMEWORK_EXPORTS"})


def test_every_exported_controller_declares_an_arm():
    declared = stems.CONTROLLERS_REACHABLE_FROM_ARMS
    exported = _exported_controllers()
    assert set(declared) == set(exported), (
        "stems.__all__ and CONTROLLERS_REACHABLE_FROM_ARMS disagree; a controller was "
        "exported or removed without updating the mapping.\n"
        f"  exported, not declared: {sorted(set(exported) - set(declared))}\n"
        f"  declared, not exported: {sorted(set(declared) - set(exported))}")


@pytest.mark.parametrize("name", sorted(stems.CONTROLLERS_REACHABLE_FROM_ARMS))
def test_the_declared_arms_exist(name):
    for arm in stems.CONTROLLERS_REACHABLE_FROM_ARMS[name]:
        assert arm in ARMS, f"{name} claims arm {arm!r}, which is not in ARMS"


def test_no_controller_is_exported_without_being_built_by_some_arm():
    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    built = set()
    for arm_name in sorted(COMPARISON_POLICIES):
        controller = build_controller(ARMS[arm_name], env, STEMSConfig())
        built.add(type(controller.base).__name__)
    for arm_name in ("rbc", "rl"):
        controller = build_controller(ARMS[arm_name], env, STEMSConfig())
        built.add(type(getattr(controller, "base", controller)).__name__)
    unreachable = [n for n in _exported_controllers() if n not in built]
    assert not unreachable, (
        f"exported from stems but built by no arm: {unreachable}. This is audit A3 "
        "recurring: a controller nothing runs is a controller nothing checks.")


def test_the_retired_mpc_agent_name_is_not_exported_as_a_controller():
    assert "MPCAgent" not in stems.__all__
    assert not hasattr(stems, "MPCAgent")


def test_every_comparison_arm_builds_a_distinct_controller_class():
    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    classes = {}
    for arm_name in sorted(COMPARISON_POLICIES):
        classes[arm_name] = type(build_controller(ARMS[arm_name], env, STEMSConfig()).base)
    assert classes["mpc"] is classes["mpc-oracle"] is stems.StorageMPC
    learners = {k: v for k, v in classes.items() if k not in ("mpc", "mpc-oracle")}
    assert len(set(learners.values())) == len(learners), (
        f"two comparison arms share an implementation: {learners}")
