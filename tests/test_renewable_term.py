from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.config import RewardConfig
from stems.reward import STEMSReward
import stems.reward as R

B = 2
XI = 0.6


def _obs(solar: float, load: float, net: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    o = rng.normal(size=40).astype(np.float32)
    o[R._IDX_SOLAR], o[R._IDX_LOAD], o[R._IDX_NET] = solar, load, net
    o[R._IDX_PRICE], o[R._IDX_T_IN] = 0.2, 22.0
    return o


def _reward(xi: float, actions: np.ndarray, next_net: float) -> list:
    cfg = RewardConfig(xi=xi)
    fn = STEMSReward(cfg, B, scale_grid_kw=100.0, scale_building_kw=20.0)
    obs = [_obs(3.0, 5.0, 2.0, i) for i in range(B)]
    nxt = [_obs(3.0, 5.0, next_net, i) for i in range(B)]
    return fn.compute(obs, actions, nxt, prev_net_consumption=[1.0] * B)


def test_the_renewable_term_adds_the_same_amount_whatever_the_action():
    a1 = np.zeros((B, 3), dtype=np.float32)
    a2 = np.full((B, 3), 0.8, dtype=np.float32)
    d1 = np.array(_reward(XI, a1, 2.0)) - np.array(_reward(0.0, a1, 2.0))
    d2 = np.array(_reward(XI, a2, 7.5)) - np.array(_reward(0.0, a2, 7.5))
    expected = XI * min(3.0 / 5.0, 1.0)
    assert np.allclose(d1, expected, atol=1e-6), d1
    assert np.allclose(d2, expected, atol=1e-6), d2
    assert np.allclose(d1, d2, atol=1e-6), (d1, d2)
    assert expected > 0.0


def test_the_renewable_term_reads_only_exogenous_channels():
    from stems.environment import OBS_NAMES

    assert OBS_NAMES[R._IDX_SOLAR] == "solar_generation"
    assert OBS_NAMES[R._IDX_LOAD] == "non_shiftable_load"


def test_the_repository_default_leaves_the_term_off():
    assert RewardConfig().xi == 0.0
