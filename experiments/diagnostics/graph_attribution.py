from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def collect_states(agent, env, config, n_states: int):
    from stems.utils import HistoryBuffer

    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    states, windows = [], []
    n, done = 0, False
    while not done and len(states) < n_states:
        w = hist.get()
        states.append(np.stack(obs, axis=0))
        windows.append(np.array(w, dtype=np.float32))
        a = agent.select_action(obs, w, explore=False)
        nxt, _, term, trunc, _ = env.step(a)
        if hasattr(agent, "observe"):
            agent.observe(nxt, env.ev_draw_kwh)
        hist.update(nxt)
        obs = nxt
        n += 1
        done = bool(term or trunc)
    return np.asarray(states), np.asarray(windows)


def committed_actions(agent, states, windows, adj=None):
    import torch

    A = agent.adj if adj is None else adj
    outs = []
    with torch.no_grad():
        for k in range(0, len(states), 128):
            x = torch.tensor(states[k:k + 128], dtype=torch.float32, device=agent.device)
            h = torch.tensor(windows[k:k + 128], dtype=torch.float32, device=agent.device)
            o = agent.obs_normalizer(x)
            hh = agent.obs_normalizer(h.reshape(-1, agent.obs_dim)).view(h.shape)
            r = agent.encoder.batch_forward(o, A, hh)
            acts = []
            for b in range(agent.B):
                dist = agent.actors[b % len(agent.actors)].distribution(r[:, b])
                acts.append(torch.tanh(dist.mean))
            outs.append(torch.stack(acts, dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="rl+calibrated")
    ap.add_argument("--season", default="year")
    ap.add_argument("--grid-cap", type=float, default=2000.0)
    ap.add_argument("--building-cap", type=float, default=500.0)
    ap.add_argument("--states", type=int, default=400)
    ap.add_argument("--out", default="results/graph_attribution")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from stems.environment import STEMSEnvironment
    from stems.utils import set_seed
    from experiments.controllers import ARMS, build_controller
    from experiments.runner import make_config
    from experiments.scenario import Scenario

    record_path = args.model[: -len("_model")] + ".json" if args.model.endswith("_model") \
        else None
    learner, scen_kw = None, {}
    if record_path and os.path.exists(record_path):
        meta = json.load(open(record_path)).get("meta", {})
        learner = meta.get("learner") or {}
        stored = meta.get("scenario") or {}
        scen_kw = {k: v for k, v in stored.items()
                   if k in Scenario.__dataclass_fields__}
    scenario = Scenario(**scen_kw) if scen_kw else Scenario(
        season=args.season, grid_cap_kw=args.grid_cap, building_cap_kw=args.building_cap)
    config = make_config(scenario, learner)
    set_seed(0)
    env = STEMSEnvironment(schema=scenario.schema_path(), seed=0,
                           heat_pump=scenario.heat_pump,
                           env_kwargs=scenario.env_kwargs("eval"),
                           hvac_control=scenario.hvac_control,
                           allow_missing_obs=scenario.allow_missing_obs)
    agent = build_controller(ARMS[args.arm], env, config)
    if not hasattr(agent, "load"):
        raise SystemExit(f"arm {args.arm!r} is not a learned agent; nothing to attribute")
    agent.load(args.model)

    states, windows = collect_states(agent, env, config, args.states)
    base = committed_actions(agent, states, windows)
    names = list(env._selected_obs_names)
    B = agent.B

    null = float(np.abs(committed_actions(agent, states, windows, agent.adj) - base).mean())

    adj0 = agent.adj.clone()
    edge = np.zeros((B, B), dtype=np.float64)
    for i in range(B):
        for j in range(B):
            A = adj0.clone()
            A[i, j] = 0.0
            moved = np.abs(committed_actions(agent, states, windows, A) - base)
            edge[i, j] = float(moved[:, i, :].mean())

    obs_inf = []
    means = states.reshape(-1, states.shape[-1]).mean(axis=0)
    for c in range(states.shape[-1]):
        s2 = states.copy()
        s2[:, :, c] = means[c]
        w2 = windows.copy()
        w2[:, :, :, c] = means[c]
        moved = float(np.abs(committed_actions(agent, s2, w2) - base).mean())
        obs_inf.append({"observation": names[c], "influence": moved})
    obs_inf.sort(key=lambda d: -d["influence"])

    isolate = np.zeros(B)
    for i in range(B):
        A = adj0.clone()
        A[i, :] = 0.0
        moved = np.abs(committed_actions(agent, states, windows, A) - base)
        isolate[i] = float(moved[:, i, :].mean())

    off = edge.copy()
    np.fill_diagonal(off, np.nan)
    report: Dict[str, Any] = {
        "model": args.model, "arm": args.arm, "states": int(len(states)),
        "null_change_cutting_nothing": null,
        "adjacency_diagonal_is_zero": bool(np.allclose(adj0.cpu().numpy().diagonal(), 0.0)),
        "self_loop_cut_change": float(np.diag(edge).mean()),
        "full_isolation_mean": float(isolate.mean()),
        "full_isolation_max": float(isolate.max()),
        "neighbour_influence_mean": float(np.nanmean(off)),
        "neighbour_influence_max": float(np.nanmax(off)),
        "strongest_edge": [int(x) for x in np.unravel_index(np.nanargmax(off), off.shape)],
        "edge_influence": edge.tolist(),
        "adjacency": adj0.cpu().numpy().tolist(),
        "observation_influence": obs_inf[:15],
    }
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "graph_attribution.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"model {args.model}")
    print(f"  states evaluated                 {len(states)}")
    print(f"  null (cut nothing)               {null:.3e}   <- must be 0")
    print(f"  adjacency has no self-loops      {report['adjacency_diagonal_is_zero']}"
          f"  (cutting A[i,i] moves {report['self_loop_cut_change']:.1e})")
    print(f"  single neighbour edge, mean      {report['neighbour_influence_mean']:.4e}")
    print(f"  strongest single edge            {report['neighbour_influence_max']:.4e}"
          f"  (building {report['strongest_edge'][1]} -> {report['strongest_edge'][0]})")
    print(f"  ALL neighbours cut, mean         {report['full_isolation_mean']:.4e}")
    top = obs_inf[0]
    print(f"  strongest single observation     {top['influence']:.4e}  ({top['observation']})")
    print(f"  -> one observation outweighs total isolation by "
          f"{top['influence'] / max(report['full_isolation_mean'], 1e-12):.1f}x")
    print("\n  observations the action leans on:")
    for d in obs_inf[:8]:
        print(f"    {d['influence']:.4e}  {d['observation']}")
    print(f"\nwrote {args.out}/graph_attribution.json")


if __name__ == "__main__":
    main()
