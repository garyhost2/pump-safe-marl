from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.scenario import Scenario
from stems.environment import STEMSEnvironment

CRASHING = Scenario(season="winter", subset_seed=1, days=3)
REFERENCE = Scenario(season="winter", subset_seed=None, days=3)


def _env(scenario: Scenario, patch: bool) -> STEMSEnvironment:
    return STEMSEnvironment(schema=scenario.schema_path(), seed=0, heat_pump=True,
                            env_kwargs=scenario.env_kwargs("train"),
                            patch_t0_double_count=patch)


def _idle(env: STEMSEnvironment) -> np.ndarray:
    return np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)


def _assert_load_counted_once(env: STEMSEnvironment) -> None:
    for b in env._env.buildings:
        output = b.energy_from_heating_device[0]
        temperature = b.weather.outdoor_dry_bulb_temperature[0]
        expected = b.heating_device.get_input_power(output, temperature, heating=True)
        assert b.heating_device.electricity_consumption[0] == pytest.approx(expected, rel=1e-6), b.name


def test_citylearn_clears_the_reset_load_itself_so_the_patch_is_a_no_op():
    env = _env(CRASHING, patch=True)
    assert "t0_thermal_double_count" in env.citylearn_native_fixes
    assert "t0_thermal_double_count" not in env.citylearn_patches


def test_unpatched_first_step_runs_and_counts_the_load_once():
    env = _env(CRASHING, patch=False)
    env.reset()
    env.step(_idle(env))
    _assert_load_counted_once(env)


def test_patched_first_step_runs_and_counts_the_load_once():
    env = _env(CRASHING, patch=True)
    env.reset()
    env.step(_idle(env))
    for b in env._env.buildings:
        output = b.energy_from_heating_device[0]
        temperature = b.weather.outdoor_dry_bulb_temperature[0]
        expected = b.heating_device.get_input_power(output, temperature, heating=True)
        assert b.heating_device.electricity_consumption[0] == pytest.approx(expected, rel=1e-6), b.name


def test_patch_holds_across_episode_resets():
    env = _env(CRASHING, patch=True)
    for _ in range(2):
        env.reset()
        done = False
        while not done:
            done = env.step(_idle(env))[2]


def test_full_dhw_charge_is_not_clipped_on_the_first_step():
    drawn = {}
    for patch in (False, True):
        env = _env(REFERENCE, patch=patch)
        env.reset()
        action = _idle(env)
        action[:, env.dhw_action_index] = 1.0
        env.step(action)
        drawn[patch] = np.array([b.dhw_device.electricity_consumption[0] for b in env._env.buildings])
        nominal = np.array([b.dhw_device.nominal_power for b in env._env.buildings])
    np.testing.assert_allclose(drawn[True], drawn[False], rtol=0, atol=1e-9)
    np.testing.assert_allclose(drawn[False], nominal, rtol=1e-6)
