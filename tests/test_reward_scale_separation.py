from __future__ import annotations

import numpy as np
import pytest

from stems.config import RewardConfig
from stems.observations import obs_index
from stems.reward import STEMSReward

_IDX_NET = obs_index("net_electricity_consumption")
_IDX_PRICE = obs_index("electricity_pricing")
_OBS_DIM = 30
B = 2


def _obs(net: float, price: float = 0.3) -> np.ndarray:
    o = np.zeros(_OBS_DIM, np.float32)
    o[_IDX_NET] = net
    o[_IDX_PRICE] = price
    return o


def test_unset_scales_fall_back_to_the_caps():
    cfg = RewardConfig()
    assert cfg.scale_grid_kw is None and cfg.scale_building_kw is None
    assert cfg.resolve_scales(33.5, 11.5) == (33.5, 11.5)
    assert cfg.resolve_scales(2000.0, 500.0) == (2000.0, 500.0)


def test_pinned_scales_ignore_the_caps():
    cfg = RewardConfig(scale_grid_kw=300.0, scale_building_kw=80.0)
    binding = cfg.resolve_scales(33.5, 11.5)
    loose = cfg.resolve_scales(2000.0, 500.0)
    assert binding == loose == (300.0, 80.0)


def test_one_scale_pinned_and_one_unset_resolves_independently():
    cfg = RewardConfig(scale_grid_kw=300.0)
    assert cfg.resolve_scales(33.5, 11.5) == (300.0, 11.5)
    cfg = RewardConfig(scale_building_kw=80.0)
    assert cfg.resolve_scales(33.5, 11.5) == (33.5, 80.0)


def test_moving_the_cap_no_longer_moves_the_objective():
    cfg = RewardConfig(scale_grid_kw=300.0, scale_building_kw=80.0)
    obs = [_obs(12.0) for _ in range(B)]
    nxt = [_obs(12.0) for _ in range(B)]

    rewards = []
    for caps in ((33.5, 11.5), (2000.0, 500.0)):
        grid, building = cfg.resolve_scales(*caps)
        fn = STEMSReward(cfg, B, grid, building)
        rewards.append(sum(fn.compute(obs, np.zeros((B, 3), np.float32), nxt,
                                      prev_net_consumption=[0.0] * B)))
    assert rewards[0] == pytest.approx(rewards[1], abs=0.0), (
        "pinned scales must make the objective independent of the caps")


def test_unpinned_scales_do_move_the_objective():
    cfg = RewardConfig()
    obs = [_obs(12.0) for _ in range(B)]
    nxt = [_obs(12.0) for _ in range(B)]

    rewards = []
    for caps in ((33.5, 11.5), (2000.0, 500.0)):
        grid, building = cfg.resolve_scales(*caps)
        fn = STEMSReward(cfg, B, grid, building)
        rewards.append(sum(fn.compute(obs, np.zeros((B, 3), np.float32), nxt,
                                      prev_net_consumption=[0.0] * B)))
    assert rewards[0] != pytest.approx(rewards[1]), (
        "with scales unset the objective follows the caps; if this ever passes, the "
        "fallback has stopped reading the caps and property 1 above is broken")


def test_non_positive_scales_are_refused():
    for bad in ({"scale_grid_kw": 0.0}, {"scale_building_kw": -1.0}):
        with pytest.raises(ValueError, match="must be positive"):
            RewardConfig(**bad).resolve_scales(300.0, 80.0)


def test_scales_are_recorded_in_the_reward_config_dataclass():
    import dataclasses
    names = {f.name for f in dataclasses.fields(RewardConfig)}
    assert {"scale_grid_kw", "scale_building_kw"} <= names
