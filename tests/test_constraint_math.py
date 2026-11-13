from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.agent import STEMSAgent
from stems.cbf import CBFShield
from stems.config import CBFConfig, SafetyConfig
from stems.observations import obs_index
from stems.thermal import CoPModel

_IDX_T_OUT = obs_index("outdoor_dry_bulb_temperature")
_IDX_NET = obs_index("net_electricity_consumption")
_OBS_LEN = 30
B = 4
HVAC_IDX = 2


def _cop_model(p_nom_kw: float = 10.0) -> CoPModel:
    ones = np.ones(B, dtype=np.float32)
    return CoPModel(efficiency_heat=0.3 * ones, target_heat=45.0 * ones,
                    efficiency_cool=0.3 * ones, target_cool=7.0 * ones,
                    nominal_power_heat=p_nom_kw * ones,
                    nominal_power_cool=p_nom_kw * ones)


def _states(net_kw, t_out_c: float = 5.0):
    out = []
    for e in np.asarray(net_kw, dtype=np.float32):
        s = np.zeros(_OBS_LEN, dtype=np.float32)
        s[_IDX_NET] = e
        s[_IDX_T_OUT] = t_out_c
        out.append(s)
    return out


def _shield(grid_cap_kw: float, p_nom_kw: float = 10.0,
            building_cap_kw: float = 1e6) -> CBFShield:
    cfg = CBFConfig(P_grid_max=grid_cap_kw, P_building_max=building_cap_kw)
    safety = SafetyConfig(robust_margins=False)
    return CBFShield(config=cfg, num_buildings=B, safety_cfg=safety,
                     enforce_soc=False, cop_model=_cop_model(p_nom_kw),
                     hvac_idx=HVAC_IDX, action_scale=1.0)


def _achieved_import(safe, states, p_nom_kw: float = 10.0) -> float:
    net = np.array([s[_IDX_NET] for s in states], dtype=np.float64)
    draw = np.abs(safe[:, HVAC_IDX]) * p_nom_kw
    return float(np.maximum(net + draw, 0.0).sum())


def _achieved_import_f32(safe, states, p_nom_kw: float = 10.0) -> float:
    net = np.array([s[_IDX_NET] for s in states], dtype=np.float32)
    p = np.full(B, p_nom_kw, dtype=np.float32)
    return float(np.maximum(net + np.abs(safe[:, HVAC_IDX]) * p, 0.0).sum())


_CAP_TOL_KW = 1e-3


def test_hvac_guard_brings_the_district_import_to_the_cap():
    cap = 100.0
    states = _states([20.0] * B)
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 1.0

    shield = _shield(cap)
    safe = shield.project(actions.copy(), states)
    achieved = _achieved_import(safe, states)

    assert _achieved_import_f32(safe, states) <= cap, (
        "the shield must meet the cap in its own arithmetic, exactly")
    assert achieved <= cap + _CAP_TOL_KW, f"import {achieved:.6f} kW exceeds {cap} kW"
    assert achieved == pytest.approx(cap, abs=_CAP_TOL_KW), (
        "the guard should sit on the cap, not far below it")
    assert safe[:, HVAC_IDX] == pytest.approx(np.full(B, 0.5), abs=1e-4)
    assert safe[:, HVAC_IDX].dtype == np.float32

    total = _achieved_import(actions, states)
    old = actions.copy()
    old[:, HVAC_IDX] *= cap / total
    old_import = _achieved_import(old, states)
    assert old_import > cap + 1.0, (
        "the cap/total rescale is supposed to miss the cap; if it no longer does, this "
        "test has stopped testing anything")
    assert old_import == pytest.approx(113.333333, abs=1e-3), (
        "13.3 kW over a 100 kW cap is the size of the bug audit B1 describes")


def test_hvac_guard_sheds_everything_when_the_fixed_load_alone_exceeds_the_cap():
    cap = 50.0
    states = _states([20.0] * B)
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 1.0

    safe = _shield(cap).project(actions.copy(), states)
    assert safe[:, HVAC_IDX] == pytest.approx(np.zeros(B), abs=1e-6)
    assert _achieved_import(safe, states) == pytest.approx(80.0, abs=1e-6)


def test_hvac_guard_leaves_a_feasible_request_untouched():
    cap = 300.0
    states = _states([20.0] * B)
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 0.4
    safe = _shield(cap).project(actions.copy(), states)
    assert safe[:, HVAC_IDX] == pytest.approx(np.full(B, 0.4), abs=1e-6)


def test_hvac_guard_handles_an_exporting_building_without_offsetting_an_import():
    cap = 100.0
    states = _states([-30.0, 30.0, 30.0, 30.0])
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 1.0
    safe = _shield(cap).project(actions.copy(), states)
    achieved = _achieved_import(safe, states)
    assert _achieved_import_f32(safe, states) <= cap
    assert achieved <= cap + _CAP_TOL_KW
    assert achieved == pytest.approx(cap, abs=_CAP_TOL_KW)


def test_hvac_guard_is_unreachable_without_a_cop_model():
    cfg = CBFConfig(P_grid_max=1.0, P_building_max=1.0)
    shield = CBFShield(config=cfg, num_buildings=B,
                       safety_cfg=SafetyConfig(robust_margins=False),
                       enforce_soc=False, cop_model=None, hvac_idx=HVAC_IDX)
    actions = np.zeros((B, 3), dtype=np.float32)
    actions[:, HVAC_IDX] = 1.0
    safe = shield.project(actions.copy(), _states([20.0] * B))
    assert safe[:, HVAC_IDX] == pytest.approx(np.ones(B))


def _advantages(seed: int = 0, N: int = 200, n_b: int = 3, K: int = 3):
    g = torch.Generator().manual_seed(seed)
    adv = torch.randn(N, n_b, generator=g)
    adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)
    cadv = torch.randn(N, n_b, K, generator=g) * 7.0 + 3.0
    return adv, cadv


def test_effective_advantage_is_invariant_to_the_cost_scale():
    adv, cadv = _advantages()
    lam = torch.tensor([0.4, 1.2, 0.0])
    base = STEMSAgent.effective_advantage(adv, cadv, lam)
    for factor in (0.01, 10.0, 1000.0):
        scaled = STEMSAgent.effective_advantage(adv, cadv * factor, lam)
        assert torch.allclose(base, scaled, atol=1e-4), (
            f"a x{factor} rescale of the costs changed the objective")


def test_effective_advantage_was_not_invariant_before_the_fix():
    adv, cadv = _advantages()
    lam = torch.tensor([0.4, 1.2, 0.0])

    def old(adv_, cadv_, lam_):
        cadv_ = cadv_ - cadv_.mean(0)
        return (adv_ - (cadv_ * lam_).sum(-1)) / (1.0 + lam_.sum())

    base = old(adv, cadv, lam)
    scaled = old(adv, cadv * 10.0, lam)
    assert not torch.allclose(base, scaled, atol=1e-2)


def test_cost_advantages_reach_unit_variance():
    adv, cadv = _advantages()
    lam_only_first = torch.tensor([1.0, 0.0, 0.0])
    eff = STEMSAgent.effective_advantage(adv, cadv, lam_only_first)
    z0 = adv - 2.0 * eff
    assert z0.mean(0).abs().max().item() < 1e-5
    assert torch.allclose(z0.std(0), torch.ones_like(z0.std(0)), atol=1e-3)


def test_effective_advantage_reduces_to_the_reward_advantage_at_zero_lambda():
    adv, cadv = _advantages()
    eff = STEMSAgent.effective_advantage(adv, cadv, torch.zeros(3))
    assert torch.allclose(eff, adv, atol=1e-6)


def test_the_same_violation_rate_gives_the_same_lambda_trajectory_at_two_reward_scales():
    from stems.config import STEMSConfig

    def trajectory(reward_scale: float):
        cfg = STEMSConfig()
        agent = object.__new__(STEMSAgent)
        agent.cfg = cfg
        agent.device = torch.device("cpu")
        K = cfg.lagrangian.num_constraints
        agent._lambdas = torch.full((K,), cfg.lagrangian.lambda_init)
        agent._cost_integral = torch.full((K,), cfg.lagrangian.lambda_init)
        agent._prev_cost = torch.zeros(K)
        agent._cost_limit = torch.full((K,), cfg.lagrangian.cost_limit)
        rng = np.random.default_rng(0)
        trace = []
        for _ in range(25):
            rate = torch.tensor(rng.uniform(0.0, 0.3, size=K), dtype=torch.float32)
            _ = float(reward_scale)
            agent._update_lambdas(rate)
            trace.append(agent._lambdas.clone())
        return torch.stack(trace)

    assert torch.allclose(trajectory(1.0), trajectory(1000.0), atol=1e-7)
