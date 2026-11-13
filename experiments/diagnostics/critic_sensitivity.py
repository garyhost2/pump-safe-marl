from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def _q_of(agent, obs_joint: torch.Tensor, act_joint: torch.Tensor,
          per_building: bool) -> torch.Tensor:
    if not per_building:
        return torch.min(agent.q1(torch.cat([obs_joint, act_joint], -1)),
                         agent.q2(torch.cat([obs_joint, act_joint], -1))).mean()
    vals = []
    B = agent.B
    o = obs_joint.view(obs_joint.shape[0], B, -1)
    a = act_joint.view(act_joint.shape[0], B, -1)
    for i in range(B):
        if hasattr(agent, "augment"):
            oi = torch.stack([
                torch.as_tensor(np.asarray(
                    agent.augment(list(o[j].numpy()), a[j].numpy(), i),
                    dtype=np.float32))
                for j in range(o.shape[0])])
        else:
            oi = o[:, i]
        if hasattr(agent, "obs_norms"):
            oi = agent.obs_norms[i](oi)
        vals.append(torch.min(agent.q1[i](torch.cat([oi, a[:, i]], -1)),
                              agent.q2[i](torch.cat([oi, a[:, i]], -1))).mean())
    return torch.stack(vals).mean()


def _independent(agent, members, ctrl, env, cfg, grid_n, n_states, seed):
    from stems.utils import HistoryBuffer

    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    per_b = [[] for _ in range(env.num_buildings)]
    for k in range(n_states * 6):
        a = ctrl.select_action(obs, hist.get(), explore=False)
        nxt, _, term, trunc, _ = env.step(a)
        if k % 6 == 0:
            for i in range(env.num_buildings):
                per_b[i].append(np.asarray(obs[i], np.float32))
        hist.update(nxt)
        obs = nxt
        if term or trunc:
            break
    elec = env.electrical_storage_action_index
    grid = torch.linspace(-1.0, 1.0, grid_n)
    spans, mags, argmaxes, absa = [], [], [], []
    with torch.no_grad():
        for i, sub in enumerate(members):
            S = torch.as_tensor(np.stack(per_b[i][:n_states]))
            o = sub._norm(S)
            a0, _ = sub.policy.sample(o, deterministic=True)
            qs = torch.zeros(grid_n)
            for gi, g in enumerate(grid):
                a = a0.clone()
                a[:, elec] = g
                qs[gi] = torch.min(sub.q1(torch.cat([o, a], -1)),
                                   sub.q2(torch.cat([o, a], -1))).mean()
            q = qs.numpy()
            spans.append(float(q.max() - q.min()))
            mags.append(float(np.abs(q).mean()))
            argmaxes.append(float(grid[int(q.argmax())]))
            absa.append(float(a0[:, elec].abs().mean()))
    mag, span = float(np.mean(mags)), float(np.mean(spans))
    corner_frac = float(np.mean([abs(x) > 0.99 for x in argmaxes]))
    return {"arm": "isac-unshielded", "seed": seed, "n_states": int(n_states),
            "per_building": True, "q_mean_abs": mag, "q_span": span,
            "q_min": None, "q_max": None,
            "relative_sensitivity": span / max(mag, 1e-12),
            "argmax_action": float(np.mean(argmaxes)),
            "argmax_per_building": [float(x) for x in argmaxes],
            "argmax_corner_fraction": corner_frac,
            "argmax_is_corner": bool(corner_frac > 0.5),
            "policy_abs_action": float(np.mean(absa))}


def sensitivity(root: Path, arm: str, seed: int, n_states: int = 100,
                grid_n: int = 41) -> Dict[str, Any]:
    from experiments.controllers import ARMS, build_controller
    from experiments.runner import make_config
    from experiments.scenario import Scenario
    from stems.environment import STEMSEnvironment
    from stems.utils import HistoryBuffer, set_seed

    name = arm if arm.endswith("-unshielded") else f"{arm}-unshielded"
    hits = sorted(root.glob(f"*/{name}__seed{seed}.json"))
    if not hits:
        raise SystemExit(f"no record for {name} seed {seed} under {root}")
    meta = json.loads(hits[-1].read_text(encoding="utf-8"))["meta"]
    model_dir = hits[-1].with_name(f"{hits[-1].stem}_model")

    sc = Scenario(**{k: meta["scenario"][k] for k in (
        "schema", "season", "subset_seed", "n_buildings", "days",
        "grid_cap_kw", "building_cap_kw", "hvac_control")})
    set_seed(seed)
    env = STEMSEnvironment(schema=sc.schema_path(), seed=seed, heat_pump=True,
                           env_kwargs=sc.env_kwargs("eval"),
                           hvac_control=sc.hvac_control)
    cfg = make_config(sc, meta.get("learner") or {})
    ctrl = build_controller(ARMS[name], env, cfg)
    ctrl.load(str(model_dir))
    agent = ctrl.base

    hist = HistoryBuffer(env.num_buildings, env.obs_dim, cfg.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    states, every = [], max(1, (6 * n_states) // n_states)
    for k in range(n_states * 6):
        a = ctrl.select_action(obs, hist.get(), explore=False)
        nxt, _, term, trunc, _ = env.step(a)
        if k % 6 == 0:
            states.append(np.concatenate([np.asarray(o, np.float32) for o in obs]))
        hist.update(nxt)
        obs = nxt
        if term or trunc:
            break
    S = torch.as_tensor(np.stack(states[:n_states]))

    members = getattr(agent, "agents", None)
    if members is not None:
        return _independent(agent, members, ctrl, env, cfg, grid_n, n_states, seed)

    per_building = hasattr(agent, "q1") and isinstance(agent.q1, torch.nn.ModuleList)
    elec = env.electrical_storage_action_index
    out: Dict[str, Any] = {"arm": name, "seed": seed, "n_states": int(S.shape[0]),
                           "per_building": bool(per_building)}
    with torch.no_grad():
        if per_building:
            B = agent.B
            a0 = torch.as_tensor(np.stack([
                ctrl.select_action([s for s in S[j].view(B, -1).numpy()],
                                   None, explore=False).reshape(-1)
                for j in range(S.shape[0])]))
            o_in = S
        else:
            o_in = agent._norm(S)
            a0, _ = agent.policy.sample(o_in, deterministic=True)
        cols = [b * env.action_dim + elec for b in range(env.num_buildings)]
        grid = torch.linspace(-1.0, 1.0, grid_n)
        qs = torch.zeros(grid_n)
        for gi, g in enumerate(grid):
            a = a0.clone()
            a[:, cols] = g
            qs[gi] = _q_of(agent, o_in, a, per_building)
        q = qs.numpy()
        out.update({
            "q_min": float(q.min()), "q_max": float(q.max()),
            "q_mean_abs": float(np.abs(q).mean()),
            "q_span": float(q.max() - q.min()),
            "relative_sensitivity": float((q.max() - q.min()) / max(abs(q).mean(), 1e-12)),
            "argmax_action": float(grid[int(q.argmax())]),
            "argmax_is_corner": bool(abs(float(grid[int(q.argmax())])) > 0.99),
            "policy_abs_action": float(a0[:, cols].abs().mean()),
        })
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results/t1_unshielded_ep50")
    p.add_argument("--arms", nargs="+", default=["sac", "marlisa", "madcq"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/critic_sensitivity")
    args = p.parse_args(argv)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in args.arms:
        try:
            rec = sensitivity(Path(args.root), arm, args.seed)
        except Exception as exc:
            print(f"[skip] {arm}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        rows.append(rec)
        (outdir / f"{rec['arm']}__seed{args.seed}.json").write_text(
            json.dumps(rec, indent=1), encoding="utf-8")

    print("%-10s %13s %11s %13s %9s %9s" % (
        "arm", "|Q|", "Q span", "rel.sens.", "argmax", "|a| policy"))
    for r in rows:
        print("%-10s %13.1f %11.3f %13.2e %9.2f %9.3f" % (
            r["arm"].replace("-unshielded", ""), r["q_mean_abs"], r["q_span"],
            r["relative_sensitivity"], r["argmax_action"], r["policy_abs_action"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
