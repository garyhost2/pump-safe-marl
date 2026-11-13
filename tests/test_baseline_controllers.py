from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stems.config import STEMSConfig
from stems.environment import STEMSEnvironment
from stems.observations import obs_index
from stems.utils import EpisodeBuffer, HistoryBuffer, set_seed

from experiments.controllers import (ARMS, COMPARISON_POLICIES, ComparisonController,
                                     build_controller)

COMPARISON_POLICIES = set(COMPARISON_POLICIES)

STEPS, EPISODES = 80, 14
TARGET_HVAC, TARGET_BATT = 0.6, -0.3

TEST_OVERRIDES = dict(hidden_dim=64, batch_size=64, start_steps=64,
                      updates_per_step=1.0)


def _env():
    set_seed(0)
    return STEMSEnvironment(force_mock=True, heat_pump=True)


def _reward(actions, h, e):
    return [-4.0 * (a[h] - TARGET_HVAC) ** 2 - 4.0 * (a[e] - TARGET_BATT) ** 2
            for a in actions]


def _episode(controller, env, cfg, explore, buf=None):
    h, e = env.hvac_action_index, env.electrical_storage_action_index
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    for k in range(STEPS):
        window = hist.get()
        actions = controller.select_action(obs, window, explore=explore)
        nxt = env.step(actions)[0]
        if hasattr(controller, "observe"):
            controller.observe(nxt)
        rewards = _reward(controller._last_raw_actions, h, e)
        hist.update(nxt)
        if buf is not None:
            buf.add(obs=obs, actions=actions, rewards=rewards, next_obs=nxt,
                    done=(k == STEPS - 1), history=window, next_history=hist.get(),
                    raw_actions=controller._last_raw_actions,
                    safe_actions=controller._last_safe_actions,
                    pre_tanh=controller._last_pre_tanh,
                    behaviour_log_probs=controller._last_log_probs,
                    constraint_costs=np.zeros((env.num_buildings, 3), np.float32))
        obs = nxt
    raw = controller._last_raw_actions
    return float(raw[:, h].mean()), float(raw[:, e].mean())


def _train(controller, env, cfg, episodes=EPISODES):
    for _ in range(episodes):
        buf = EpisodeBuffer()
        _episode(controller, env, cfg, explore=True, buf=buf)
        controller.update(buf.get_batch())
    return _episode(controller, env, cfg, explore=False)


def _build(name, env, **overrides):
    cfg = STEMSConfig()
    controller = build_controller(ARMS[name], env, cfg)
    if overrides:
        base = type(controller.base)
        kwargs = dict(obs_dim=env.obs_dim, action_dim=env.action_dim,
                      num_buildings=env.num_buildings, **overrides)
        controller.base = base(**kwargs)
    return controller, cfg


@pytest.mark.parametrize("name", sorted(COMPARISON_POLICIES))
def test_the_arm_builds_and_acts_in_range(name):
    env = _env()
    controller = build_controller(ARMS[name], env, STEMSConfig())
    cfg = STEMSConfig()
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    actions = controller.select_action(obs, hist.get(), explore=False)
    assert actions.shape == (env.num_buildings, env.action_dim)
    assert np.isfinite(actions).all()
    assert (np.abs(actions) <= 1.0 + 1e-6).all(), f"{name} left the action box"


@pytest.mark.parametrize("name", sorted(COMPARISON_POLICIES))
def test_the_arm_presents_the_attributes_the_runner_reads(name):
    env = _env()
    controller = build_controller(ARMS[name], env, STEMSConfig())
    for attr in ("elec_idx", "control_indices", "_last_raw_actions", "_last_safe_actions",
                 "_last_pre_tanh", "_last_log_probs", "_last_nominal_actions",
                 "fleet_shield", "update", "save", "load", "observe"):
        assert hasattr(controller, attr), f"{name} is missing {attr!r}"
    assert isinstance(controller, ComparisonController)


def test_single_agent_sac_moves_to_the_known_optimum():
    env = _env()
    controller, cfg = _build("sac", env, **TEST_OVERRIDES)
    hvac, batt = _train(controller, env, cfg)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_dmappo_moves_to_the_known_optimum():
    env = _env()
    controller, cfg = _build("dmappo", env)
    hvac, batt = _train(controller, env, cfg, episodes=2 * EPISODES)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_maddpg_moves_to_the_known_optimum():
    env = _env()
    controller, cfg = _build("maddpg", env, **TEST_OVERRIDES)
    hvac, batt = _train(controller, env, cfg)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_marlisa_moves_to_the_known_optimum():
    env = _env()
    controller, cfg = _build("marlisa", env, **TEST_OVERRIDES)
    hvac, batt = _train(controller, env, cfg)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_mappo_with_a_centralised_critic_moves_to_the_known_optimum():
    env = _env()
    controller = build_controller(ARMS["mappo-cc"], env, STEMSConfig())
    hvac, batt = _train(controller, env, STEMSConfig())
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"
    assert batt < -0.1, f"battery action {batt:+.3f} did not move toward {TARGET_BATT}"


def test_metaems_moves_to_the_known_optimum():
    env = _env()
    controller, cfg = _build("metaems", env, hidden_dim=64, batch_size=64,
                             start_steps=64, inner_steps=8, lr_outer=0.7)
    optimum = np.array([TARGET_HVAC, TARGET_BATT])
    before = np.linalg.norm(np.array(_episode(controller, env, cfg, explore=False)) - optimum)
    after = np.linalg.norm(np.array(_train(controller, env, cfg, episodes=2 * EPISODES))
                           - optimum)
    assert after < 0.6 * before, (
        f"distance to the optimum went from {before:.3f} to {after:.3f}; Reptile did "
        "not meta-learn the shared task")


def test_madcq_moves_to_the_known_optimum_on_its_unconstrained_axis():
    env = _env()
    controller, cfg = _build("madcq", env, hidden_dim=64, batch_size=64, start_steps=64,
                             updates_per_step=0.25, battery_model=env.battery_model(),
                             elec_idx=env.electrical_storage_action_index,
                             eps_decay_steps=400)
    hvac, _ = _train(controller, env, cfg)
    assert hvac > 0.2, f"HVAC action {hvac:+.3f} did not move toward {TARGET_HVAC}"


def test_madcq_never_proposes_an_action_outside_the_safe_set():
    env = _env()
    controller = build_controller(ARMS["madcq"], env, STEMSConfig())
    agent, battery = controller.base, env.battery_model()
    cfg = STEMSConfig()
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    soc_idx = obs_index("electrical_storage_soc")
    e = env.electrical_storage_action_index
    worst = 0.0
    for _ in range(40):
        soc = np.array([float(o[soc_idx]) for o in obs], dtype=np.float64)
        lo, hi = battery.safe_interval(soc, np.full(env.num_buildings, cfg.cbf.SOC_min),
                                       np.full(env.num_buildings, cfg.cbf.SOC_max))
        raw = agent.select_action(obs, hist.get(), explore=True)
        worst = max(worst, float(np.maximum(lo - raw[:, e], raw[:, e] - hi).max()))
        obs = env.step(controller.select_action(obs, hist.get(), explore=True))[0]
        hist.update(obs)
    assert worst <= 0.02, (
        f"MADCQ proposed a battery action {worst:.4f} outside the exact plant's safe "
        "interval; the constrained maximisation is not being applied")


def test_dmappo_uses_the_behaviour_log_probability_from_the_trajectory():
    env = _env()
    controller, cfg = _build("dmappo", env)
    controller.base.epochs = 1
    controller.base.minibatch_size = 10_000
    buf = EpisodeBuffer()
    _episode(controller, env, cfg, explore=True, buf=buf)
    stats = controller.update(buf.get_batch())
    assert stats["recomputed_behaviour_logp"] is False
    assert abs(stats["approx_kl"]) < 1e-4, stats
    assert stats["clip_frac"] == 0.0


@pytest.mark.parametrize("name", sorted(COMPARISON_POLICIES - {"mpc", "mpc-oracle"}))
def test_the_stored_behaviour_samples_are_not_all_the_same_object(name):
    env = _env()
    controller = build_controller(ARMS[name], env, STEMSConfig())
    cfg = STEMSConfig()
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    seen_pre, seen_logp = [], []
    for _ in range(6):
        actions = controller.select_action(obs, hist.get(), explore=True)
        seen_pre.append(controller._last_pre_tanh)
        seen_logp.append(controller._last_log_probs)
        obs = env.step(actions)[0]
        hist.update(obs)
    assert len({id(a) for a in seen_pre}) == len(seen_pre), "pre-tanh buffer is reused"
    assert len({id(a) for a in seen_logp}) == len(seen_logp), "log-prob buffer is reused"
    stacked = np.stack(seen_pre)
    assert stacked.std(axis=0).max() > 1e-6, (
        f"{name}'s stored pre-squash samples are identical across steps")


def test_dmappo_has_no_hard_coded_battery_model():
    import inspect

    import stems.baselines as baselines

    source = inspect.getsource(baselines.DMAPPOAgent)
    assert "_cbf_penalty" not in source
    assert "0.1" not in source.split('"""')[2] if source.count('"""') > 2 else True


def test_marlisa_builds_the_same_augmentation_when_acting_and_when_updating():
    env = _env()
    controller = build_controller(ARMS["marlisa"], env, STEMSConfig())
    agent = controller.base
    obs, _ = env.reset()
    actions = agent.select_action(obs, None, explore=True)
    for i in range(env.num_buildings):
        seen = agent.augment(obs, actions, i)
        assert seen.shape[0] == agent.aug_dim[i]
        if i > 0:
            block = seen[env.obs_dim:env.obs_dim + i * env.action_dim]
            assert np.allclose(block, actions[:i].reshape(-1)), (
                f"building {i}'s sequential block is not buildings 0..{i - 1}'s actions")
        rebuilt = agent.augment(obs, actions, i)
        assert np.array_equal(seen, rebuilt)


def test_madcq_does_not_enumerate_the_joint_action_space():
    env = _env()
    controller = build_controller(ARMS["madcq"], env, STEMSConfig())
    head = controller.base.q[0].heads
    assert head.out_features == env.action_dim * controller.base.n_bins
    assert head.out_features < 11 ** 3


def test_metaems_is_reptile_over_more_than_one_task():
    env = _env()
    controller = build_controller(ARMS["metaems"], env, STEMSConfig())
    agent = controller.base
    assert len(agent.buffers) == env.num_buildings, "one replay per task"
    assert agent.inner_steps >= 1
    assert hasattr(agent, "adapt"), "a meta-learner must expose test-time adaptation"


def test_single_agent_sac_is_actually_single_agent():
    env = _env()
    controller = build_controller(ARMS["sac"], env, STEMSConfig())
    agent = controller.base
    first = agent.policy.trunk[0]
    assert first.in_features == env.num_buildings * env.obs_dim
    assert agent.policy.mean_head.out_features == env.num_buildings * env.action_dim


def test_the_mpc_agent_name_refuses_rather_than_resurrecting_the_old_one():
    from stems.baselines import MPCAgent

    with pytest.raises(NotImplementedError) as excinfo:
        MPCAgent()
    assert "stems.mpc.StorageMPC" in str(excinfo.value)


def test_the_presquash_penalty_is_exactly_zero_when_disabled():
    import torch

    from stems.baselines import _presquash_penalty

    mean = torch.randn(16, 4) * 5.0
    log_std = torch.randn(16, 4)
    assert float(_presquash_penalty(mean, log_std, 0.0)) == 0.0
    assert float(_presquash_penalty(mean, log_std, -1.0)) == 0.0
    on = float(_presquash_penalty(mean, log_std, 1e-3))
    assert on > 0.0
    expected = 1e-3 * float(mean.pow(2).mean() + log_std.pow(2).mean())
    assert abs(on - expected) < 1e-7


@pytest.mark.parametrize("agent_name", ["sac", "marlisa"])
def test_policy_regularisation_pulls_the_presquash_mean_in(agent_name):
    import torch

    from stems.baselines import MARLISAAgent, SingleAgentSAC

    O = 6 if agent_name == "sac" else 28

    def build(reg):
        torch.manual_seed(0)
        np.random.seed(0)
        if agent_name == "sac":
            return SingleAgentSAC(obs_dim=O, action_dim=2, num_buildings=2,
                                  hidden_dim=32, batch_size=32, start_steps=16,
                                  policy_reg=reg)
        return MARLISAAgent(obs_dim=O, action_dim=2, num_buildings=2,
                            hidden_dim=32, batch_size=32, start_steps=16,
                            policy_reg=reg)

    rng = np.random.default_rng(0)
    n, B, A = 192, 2, 2
    actions = rng.uniform(-1.0, 1.0, size=(n, B, A)).astype(np.float32)
    batch = {
        "obs": rng.normal(size=(n, B, O)).astype(np.float32),
        "actions": actions,
        "rewards": np.abs(actions).sum(axis=2).astype(np.float32) * 10.0,
        "next_obs": rng.normal(size=(n, B, O)).astype(np.float32),
        "dones": np.zeros(n, dtype=np.float32),
    }

    def presquash_abs_mean(agent):
        policy = agent.policy if agent_name == "sac" else agent.policies[0]
        width = policy.trunk[0].in_features
        probe = torch.as_tensor(
            rng.normal(size=(64, width)), dtype=torch.float32)
        with torch.no_grad():
            m, _ = policy(probe)
        return float(m.abs().mean())

    plain, regularised = build(0.0), build(1.0)
    for _ in range(4):
        plain.update({k: v.copy() for k, v in batch.items()})
        regularised.update({k: v.copy() for k, v in batch.items()})

    assert presquash_abs_mean(regularised) < presquash_abs_mean(plain), (
        f"{agent_name}: regularised pre-squash mean "
        f"{presquash_abs_mean(regularised):.4f} not below plain "
        f"{presquash_abs_mean(plain):.4f}")
