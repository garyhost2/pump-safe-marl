from __future__ import annotations

import argparse
import dataclasses
import json
from typing import Dict, List, Tuple

import numpy as np

from stems.observations import obs_index

_IDX_NET = obs_index("net_electricity_consumption")

TERM_WEIGHTS = ("mu", "alpha_grid", "alpha_build", "beta_ramp", "lambda_indoor", "xi")

CAP_SETTINGS = {"binding (33.5/11.5)": (33.5, 11.5),
                "non-binding (2000/500)": (2000.0, 500.0)}


def rollout(steps: int, caps: Tuple[float, float], seed: int = 0):
    from experiments.controllers import ARMS, build_controller
    from stems.config import CBFConfig, STEMSConfig
    from stems.environment import STEMSEnvironment
    from stems.utils import HistoryBuffer

    config = STEMSConfig()
    config.cbf = CBFConfig(P_grid_max=caps[0], P_building_max=caps[1])
    env = STEMSEnvironment(seed=seed, heat_pump=True,
                           env_kwargs={"episode_time_steps": [(0, steps)]})
    agent = build_controller(ARMS["rbc+calibrated"], env, config)

    obs, _ = env.reset()
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    hist.prime(obs)
    prev_net = [float(o[_IDX_NET]) for o in obs]
    transitions = []
    for _ in range(steps - 1):
        actions = agent.select_action(obs, hist.get(), explore=False)
        nxt, _r, term, trunc, _i = env.step(actions)
        agent.observe(nxt, env.ev_draw_kwh)
        transitions.append((obs, np.asarray(actions, dtype=np.float32), nxt, list(prev_net)))
        prev_net = [float(o[_IDX_NET]) for o in nxt]
        obs = nxt
        hist.update(obs)
        if term or trunc:
            break
    return env, transitions


def term_contributions(transitions, reward_cfg, B: int, scales: Tuple[float, float],
                       heating_idx) -> Dict[str, float]:
    from stems.reward import STEMSReward

    def total(cfg) -> float:
        fn = STEMSReward(cfg, B, scales[0], scales[1], heating_setpoint_idx=heating_idx)
        out = 0.0
        for obs, actions, nxt, prev in transitions:
            out += float(sum(fn.compute(obs, actions, nxt, prev_net_consumption=prev)))
        return out

    full = total(reward_cfg)
    contributions = {}
    for name in TERM_WEIGHTS:
        stripped = dataclasses.replace(reward_cfg, **{name: 0.0})
        contributions[name] = full - total(stripped)
    contributions["_total"] = full
    return contributions


def shares(contributions: Dict[str, float]) -> Dict[str, float]:
    mass = sum(abs(contributions[k]) for k in TERM_WEIGHTS)
    if mass <= 0.0:
        return {k: float("nan") for k in TERM_WEIGHTS}
    return {k: 100.0 * abs(contributions[k]) / mass for k in TERM_WEIGHTS}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pinned-scales", type=float, nargs=2, default=(300.0, 80.0),
                    metavar=("GRID_KW", "BUILDING_KW"))
    ap.add_argument("--out", default="experiments/diagnostics/reward_scale_check.json")
    args = ap.parse_args()

    from stems.config import RewardConfig

    rows: List[Dict] = []
    for label, caps in CAP_SETTINGS.items():
        env, transitions = rollout(args.steps, caps, args.seed)
        B, heating_idx = env.num_buildings, env.heating_setpoint_idx
        for mode, cfg in (("scales unset (follows the caps)", RewardConfig()),
                          (f"scales pinned {tuple(args.pinned_scales)}",
                           RewardConfig(scale_grid_kw=args.pinned_scales[0],
                                        scale_building_kw=args.pinned_scales[1]))):
            resolved = cfg.resolve_scales(*caps)
            contrib = term_contributions(transitions, cfg, B, resolved, heating_idx)
            rows.append({"caps": label, "mode": mode, "resolved_scales": resolved,
                         "steps": len(transitions),
                         "contributions": contrib, "shares_pct": shares(contrib)})

    hdr = f"{'caps':24s} {'mode':34s} " + "".join(f"{t:>14s}" for t in TERM_WEIGHTS)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        line = f"{r['caps']:24s} {r['mode']:34s} "
        line += "".join(f"{r['shares_pct'][t]:13.2f}%" for t in TERM_WEIGHTS)
        print(line)

    stability = {}
    for r in rows:
        s = r["shares_pct"]
        stability[(r["caps"], r["mode"])] = s["alpha_grid"] + s["alpha_build"] + s["beta_ramp"]
    print("\nthree stability terms, combined share of the objective:")
    for (caps, mode), v in stability.items():
        print(f"  {caps:24s} {mode:34s} {v:6.2f}%")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"steps": args.steps, "seed": args.seed,
                   "pinned_scales": list(args.pinned_scales), "rows": rows}, fh, indent=2)
    print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
