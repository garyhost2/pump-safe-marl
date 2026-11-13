from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.paper_stems import PaperSTEMS
from stems.utils import EpisodeBuffer, HistoryBuffer, set_seed

from experiments.controllers import ARMS, build_controller

B, OBS, ACT, WINDOW = 3, 11, 2, 7


def _agent(**kw):
    set_seed(0)
    adj = np.eye(B, dtype=np.float32) + 0.1 * (1.0 - np.eye(B, dtype=np.float32))
    return PaperSTEMS(OBS, ACT, num_buildings=B, adjacency=adj,
                      window_size=WINDOW, spatial_dim=16, temporal_dim=8,
                      repr_dim=12, hidden_dim=16, **kw)


def _batch(n=40, rng=None):
    rng = rng or np.random.default_rng(0)
    return {
        "obs": list(rng.normal(size=(n, B, OBS)).astype(np.float32)),
        "actions": list(rng.uniform(-0.9, 0.9, size=(n, B, ACT)).astype(np.float32)),
        "rewards": list(rng.normal(size=(n, B)).astype(np.float32)),
        "next_obs": list(rng.normal(size=(n, B, OBS)).astype(np.float32)),
        "dones": [0.0] * (n - 1) + [1.0],
    }


def test_the_window_gather_puts_the_building_axis_second():
    agent = _agent()
    obs = torch.arange(5 * B * OBS, dtype=torch.float32).reshape(5, B, OBS)
    rows = torch.tensor([[0, 0, 0, 0, 1, 2, 3], [1, 2, 3, 4, 4, 4, 4]])
    win = agent._windows(obs, rows)
    assert win.shape == (2, B, WINDOW, OBS), win.shape
    for n in range(2):
        for b in range(B):
            for w in range(WINDOW):
                assert torch.equal(win[n, b, w], obs[rows[n, w], b])


def test_the_stored_action_is_the_safe_one_not_the_proposal():

    class Clamp:
        def project(self, actions, states):
            return np.full_like(np.asarray(actions, dtype=np.float32), -0.5)

    agent = _agent(shield=Clamp())
    obs = [np.random.default_rng(1).normal(size=OBS).astype(np.float32)
           for _ in range(B)]
    hist = np.stack([np.tile(o, (WINDOW, 1)) for o in obs])
    executed = agent.select_action(obs, hist, explore=True)

    assert np.allclose(executed, -0.5), executed
    assert not np.allclose(agent._last_nominal_actions, -0.5)
    assert np.allclose(np.tanh(agent._last_pre_tanh), -0.5, atol=1e-5)


def test_the_shared_encoder_receives_gradient():
    agent = _agent()
    before = [p.detach().clone() for p in agent.encoder.parameters()]
    agent.update(_batch())
    after = list(agent.encoder.parameters())
    moved = sum(1 for a, b in zip(after, before) if not torch.allclose(a, b))
    assert moved > 0, "no encoder parameter changed, so psi is not being trained"


@pytest.mark.parametrize("clip_eps", [None, 0.2])
def test_update_runs_and_reports_finite_losses_under_both_readings(clip_eps):
    agent = _agent(clip_eps=clip_eps, epochs=2, minibatch_size=16)
    stats = agent.update(_batch())
    assert set(stats) >= {"policy", "value", "entropy"}
    assert all(np.isfinite(v) for v in stats.values()), stats


def test_the_arm_builds_acts_in_range_and_round_trips_through_disk(tmp_path):
    set_seed(0)
    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    cfg = STEMSConfig()
    controller = build_controller(ARMS["paper-stems"], env, cfg)
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    a = controller.select_action(obs, hist.get(), explore=False)
    assert a.shape == (env.num_buildings, env.action_dim)
    assert np.all(np.abs(a) <= 1.0 + 1e-6)

    controller.save(str(tmp_path / "m"))
    reloaded = build_controller(ARMS["paper-stems"], env, cfg)
    reloaded.load(str(tmp_path / "m"))
    b = reloaded.select_action(obs, hist.get(), explore=False)
    assert np.allclose(a, b, atol=1e-6), "deterministic action changed across save/load"


def test_it_moves_toward_a_known_optimum():
    set_seed(0)
    env = STEMSEnvironment(force_mock=True, heat_pump=True)
    cfg = STEMSConfig()
    controller = build_controller(ARMS["paper-stems-unshielded"], env, cfg)
    h, e = env.hvac_action_index, env.electrical_storage_action_index
    target_h, target_e = 0.6, -0.3
    steps, episodes = 80, 14

    def episode(explore, buf=None):
        hb = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
        obs, _ = env.reset()
        hb.prime(obs)
        for k in range(steps):
            window = hb.get()
            actions = controller.select_action(obs, window, explore=explore)
            nxt = env.step(actions)[0]
            rewards = [-4.0 * (a[h] - target_h) ** 2 - 4.0 * (a[e] - target_e) ** 2
                       for a in actions]
            hb.update(nxt)
            if buf is not None:
                buf.add(obs=obs, actions=actions, rewards=rewards, next_obs=nxt,
                        done=(k == steps - 1), history=window, next_history=hb.get(),
                        raw_actions=controller._last_raw_actions,
                        safe_actions=controller._last_safe_actions,
                        pre_tanh=controller._last_pre_tanh,
                        behaviour_log_probs=controller._last_log_probs,
                        constraint_costs=np.zeros((env.num_buildings, 3), np.float32))
            obs = nxt
        a = controller._last_raw_actions
        return float(a[:, h].mean()), float(a[:, e].mean())

    start = episode(explore=False)
    for _ in range(episodes):
        buf = EpisodeBuffer()
        episode(explore=True, buf=buf)
        controller.update(buf.get_batch())
    end = episode(explore=False)

    d0 = abs(start[0] - target_h) + abs(start[1] - target_e)
    d1 = abs(end[0] - target_h) + abs(end[1] - target_e)
    assert d1 < d0, (f"did not move toward the optimum: start {start} (d={d0:.3f}) "
                     f"-> end {end} (d={d1:.3f}), target ({target_h}, {target_e})")
