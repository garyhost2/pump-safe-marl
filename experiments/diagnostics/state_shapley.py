from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.diagnostics.graph_attribution import collect_states

GRAPH_PLAYER = "<neighbour coupling>"


def masked_actions(agent, states, windows, present_channels, graph_on, background,
                   action_index):
    import torch

    A = agent.adj
    if not graph_on:
        A = A * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)

    absent = ~present_channels
    x = states.copy()
    w = windows.copy()
    if absent.any():
        x[:, :, absent] = states[background][:, :, absent]
        w[:, :, :, absent] = windows[background][:, :, :, absent]

    outs = []
    with torch.no_grad():
        for k in range(0, len(x), 256):
            xb = torch.tensor(x[k:k + 256], dtype=torch.float32, device=agent.device)
            hb = torch.tensor(w[k:k + 256], dtype=torch.float32, device=agent.device)
            o = agent.obs_normalizer(xb)
            hh = agent.obs_normalizer(hb.reshape(-1, agent.obs_dim)).view(hb.shape)
            r = agent.encoder.batch_forward(o, A, hh)
            acts = []
            for b in range(agent.B):
                dist = agent.actors[b % len(agent.actors)].distribution(r[:, b])
                acts.append(torch.tanh(dist.mean)[:, action_index])
            outs.append(torch.stack(acts, dim=1).cpu().numpy())
    return np.concatenate(outs, axis=0).mean(axis=0)


def shapley(agent, states, windows, action_index, permutations: int,
            rng: np.random.Generator):
    K = states.shape[2]
    n_players = K + 1
    T, B = states.shape[0], states.shape[1]
    draws = np.empty((permutations, n_players, B), dtype=np.float64)
    grand = empty = None

    for p in range(permutations):
        background = rng.permutation(T)
        order = rng.permutation(n_players)
        present = np.zeros(K, dtype=bool)
        graph_on = False
        prev = masked_actions(agent, states, windows, present, graph_on,
                              background, action_index)
        if empty is None:
            empty = prev.copy()
        for player in order:
            if player == K:
                graph_on = True
            else:
                present[player] = True
            cur = masked_actions(agent, states, windows, present, graph_on,
                                 background, action_index)
            draws[p, player] = cur - prev
            prev = cur
        if grand is None:
            grand = prev.copy()

    names = [f"channel_{i}" for i in range(K)] + [GRAPH_PLAYER]
    return draws, names, grand, empty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="rl+calibrated")
    ap.add_argument("--season", default="year")
    ap.add_argument("--grid-cap", type=float, default=2000.0)
    ap.add_argument("--building-cap", type=float, default=500.0)
    ap.add_argument("--states", type=int, default=256)
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--action", default="electrical_storage",
                    help="which action the attribution explains")
    ap.add_argument("--random-init", action="store_true",
                    help="SANITY CONTROL: skip loading the checkpoint and attribute a "
                         "randomly initialised network instead. An attribution that "
                         "reports the same thing for a trained and an untrained model "
                         "is measuring the architecture, not what was learned "
                         "(the model-randomisation test of Adebayo et al., 2018). "
                         "States are still collected from the TRAINED policy so the "
                         "two runs see the same inputs.")
    ap.add_argument("--out", default="results/state_shapley")
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)
    from stems.environment import STEMSEnvironment
    from stems.utils import set_seed
    from experiments.controllers import ARMS, build_controller
    from experiments.runner import make_config
    from experiments.scenario import Scenario

    record_path = (args.model[: -len("_model")] + ".json"
                   if args.model.endswith("_model") else None)
    learner, scen_kw = None, {}
    if record_path and os.path.exists(record_path):
        meta = json.load(open(record_path)).get("meta", {})
        learner = meta.get("learner") or {}
        stored = meta.get("scenario") or {}
        scen_kw = {k: v for k, v in stored.items() if k in Scenario.__dataclass_fields__}
    scenario = Scenario(**scen_kw) if scen_kw else Scenario(
        season=args.season, grid_cap_kw=args.grid_cap, building_cap_kw=args.building_cap)
    config = make_config(scenario, learner)
    set_seed(args.seed)
    env = STEMSEnvironment(schema=scenario.schema_path(), seed=args.seed,
                           heat_pump=scenario.heat_pump,
                           env_kwargs=scenario.env_kwargs("eval"),
                           hvac_control=scenario.hvac_control,
                           allow_missing_obs=scenario.allow_missing_obs)
    agent = build_controller(ARMS[args.arm], env, config)
    if not hasattr(agent, "load"):
        raise SystemExit(f"arm {args.arm!r} is not a learned agent; nothing to attribute")
    agent.load(args.model)

    action_index = {"electrical_storage": env.electrical_storage_action_index,
                    "hvac": env.hvac_action_index}.get(args.action, 0)
    if action_index < 0:
        raise SystemExit(f"action {args.action!r} is not controlled on this schema")

    states, windows = collect_states(agent, env, config, args.states)
    obs_names = list(env._selected_obs_names)
    if args.random_init:
        fresh = build_controller(ARMS[args.arm], env, config)
        agent.encoder, agent.actors = fresh.encoder, fresh.actors
    rng = np.random.default_rng(args.seed)
    draws, names, grand, empty = shapley(agent, states, windows, action_index,
                                         args.permutations, rng)
    names = obs_names + [GRAPH_PLAYER]

    phi = draws.mean(axis=0)
    se = draws.std(axis=0, ddof=1) / np.sqrt(draws.shape[0])

    EFFICIENCY_TOL = 1e-5
    residual = float(np.abs(phi.sum(axis=0) - (grand - empty)).max())
    assert residual < EFFICIENCY_TOL, (
        f"efficiency violated by {residual:.3e} (tolerance {EFFICIENCY_TOL:.0e}); the "
        "marginals must telescope to v(grand) - v(empty), so this is an estimator bug")

    mag = np.abs(phi).mean(axis=1)
    rank = np.argsort(-mag)

    report: Dict[str, Any] = {
        "model": args.model, "arm": args.arm, "action": args.action,
        "random_init": bool(args.random_init),
        "states": int(states.shape[0]), "buildings": int(states.shape[1]),
        "permutations": args.permutations, "seed": args.seed,
        "efficiency_residual": float(residual),
        "grand_coalition_action": grand.tolist(),
        "empty_coalition_action": empty.tolist(),
        "players": [
            {"name": names[j],
             "mean_abs_phi": float(mag[j]),
             "phi_per_building": phi[j].tolist(),
             "se_per_building": se[j].tolist(),
             "share_of_total_abs": float(mag[j] / mag.sum()) if mag.sum() else float("nan")}
            for j in range(len(names))],
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "state_shapley.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"efficiency residual {residual:.2e}  (exact by construction)")
    print(f"action span across buildings: empty {empty.mean():+.4f} -> "
          f"grand {grand.mean():+.4f}")
    print(f"\n{'player':<42}{'|phi|':>10}{'share':>9}{'95% CI (bldg mean)':>24}")
    print("-" * 85)
    for j in rank[:14]:
        lo = phi[j].mean() - 1.96 * se[j].mean()
        hi = phi[j].mean() + 1.96 * se[j].mean()
        mark = "  <-- graph" if names[j] == GRAPH_PLAYER else ""
        print(f"{names[j]:<42}{mag[j]:>10.5f}{report['players'][j]['share_of_total_abs']:>9.1%}"
              f"   [{lo:+.5f}, {hi:+.5f}]{mark}")
    gi = names.index(GRAPH_PLAYER)
    if gi not in rank[:14]:
        print(f"{names[gi]:<42}{mag[gi]:>10.5f}"
              f"{report['players'][gi]['share_of_total_abs']:>9.1%}   <-- graph")
    print(f"\ngraph rank: {list(rank).index(gi) + 1} of {len(names)}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
