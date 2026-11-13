from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.agent import STEMSAgent
from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.graph import BuildingGraph
from stems.utils import set_seed

TEACHER_ACTION = 0.5
DEMO_STEPS = 64


def _agent(cfg=None):
    set_seed(0)
    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    cfg = cfg or STEMSConfig()
    info = env.get_building_info()
    graph = BuildingGraph(env.num_buildings, info["positions"], info["features"], cfg.graph)
    agent = STEMSAgent(env.obs_dim, env.action_dim, env.num_buildings, graph, config=cfg,
                       battery_info=env.battery_info(), use_cbf=False,
                       electrical_storage_action_index=env.electrical_storage_action_index,
                       hvac_action_index=env.hvac_action_index)
    return agent, env, cfg


def _demos(agent, env, cfg, action=TEACHER_ACTION):
    rng = np.random.default_rng(0)
    B, W = env.num_buildings, cfg.transformer.window_size
    return {
        "obs": rng.normal(size=(DEMO_STEPS, B, env.obs_dim)).astype(np.float32),
        "history": rng.normal(size=(DEMO_STEPS, B, W, env.obs_dim)).astype(np.float32),
        "actions": np.full((DEMO_STEPS, B, env.action_dim), action, dtype=np.float32),
    }


def _log_std_on(agent, demos):
    with torch.no_grad():
        o = torch.tensor(demos["obs"], dtype=torch.float32)
        h = torch.tensor(demos["history"], dtype=torch.float32)
        o = agent.obs_normalizer(o)
        h = agent.obs_normalizer(h.reshape(-1, agent.obs_dim)).view(h.shape)
        r = agent.encoder.batch_forward(o, agent.adj, h)
        return float(torch.cat([agent.actors[b % len(agent.actors)](r[:, b])[1]
                                for b in range(agent.B)]).mean())


def _mean_action(agent, demos):
    return agent.select_action(list(demos["obs"][0]), demos["history"][0], explore=False)


def test_the_warm_start_is_off_unless_an_arm_asks_for_it():
    cfg = STEMSConfig()
    assert cfg.training.bc_episodes == 0
    assert cfg.training.bc_epochs == 0


def test_zero_epochs_is_a_no_op_that_reports_itself_as_one():
    agent, env, cfg = _agent()
    stats = agent.behaviour_clone(_demos(agent, env, cfg), epochs=0, lr=1e-3)
    assert stats["bc_ran"] is False
    assert stats["bc_steps"] == 0


def test_cloning_moves_the_policy_toward_the_teacher():
    agent, env, cfg = _agent()
    demos = _demos(agent, env, cfg)
    before = float(np.abs(_mean_action(agent, demos) - TEACHER_ACTION).mean())
    agent.behaviour_clone(demos, epochs=40, lr=3e-3)
    after = float(np.abs(_mean_action(agent, demos) - TEACHER_ACTION).mean())
    assert after < before, f"cloning did not approach the teacher: {before} -> {after}"
    assert after < 0.1, f"cloned policy is still {after} away from a constant teacher"


def test_cloning_reduces_the_negative_log_likelihood_of_the_teachers_actions():
    agent, env, cfg = _agent()
    stats = agent.behaviour_clone(_demos(agent, env, cfg), epochs=20, lr=3e-3)
    assert stats["bc_ran"] is True
    assert stats["bc_nll_last"] < stats["bc_nll_first"]


def test_the_collapsed_spread_is_recorded_and_then_reset():
    agent, env, cfg = _agent()
    stats = agent.behaviour_clone(_demos(agent, env, cfg), epochs=40, lr=3e-3)
    assert stats["bc_fitted_log_std"] < -1.5
    assert stats["bc_reset_log_std"] == pytest.approx(cfg.training.bc_reset_log_std)
    with torch.no_grad():
        x = torch.zeros(1, agent.actors[0].trunk[0].in_features)
        log_std = float(agent.actors[0](x)[1].mean())
    assert log_std == pytest.approx(cfg.training.bc_reset_log_std, abs=1e-5)


def test_the_fitted_spread_can_be_kept_when_asked_for():
    cfg = STEMSConfig()
    cfg.training.bc_reset_log_std = None
    agent, env, cfg = _agent(cfg)
    demos = _demos(agent, env, cfg)
    stats = agent.behaviour_clone(demos, epochs=40, lr=3e-3)
    assert stats["bc_reset_log_std"] is None
    kept = _log_std_on(agent, demos)
    assert kept == pytest.approx(stats["bc_fitted_log_std"], abs=1e-4), \
        "the fitted spread should survive untouched when no reset is asked for"
    assert kept < -1.5, "fitting a constant teacher should have collapsed the spread"


def test_an_empty_demonstration_set_is_refused_rather_than_silently_skipped():
    agent, env, cfg = _agent()
    empty = {"obs": np.zeros((0, env.num_buildings, env.obs_dim), dtype=np.float32),
             "history": np.zeros((0, env.num_buildings, cfg.transformer.window_size,
                                  env.obs_dim), dtype=np.float32),
             "actions": np.zeros((0, env.num_buildings, env.action_dim), dtype=np.float32)}
    with pytest.raises(ValueError, match="no teacher steps"):
        agent.behaviour_clone(empty, epochs=5, lr=1e-3)


def test_the_warm_start_leaves_the_critics_at_initialisation():
    agent, env, cfg = _agent()
    before = [p.detach().clone() for p in agent.critics.parameters()]
    agent.behaviour_clone(_demos(agent, env, cfg), epochs=20, lr=3e-3)
    after = list(agent.critics.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after))
