#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "citylearn_schemas/tx_travis_8b/schema.json"
COLS = ("hour", "day_type", "outdoor_dry_bulb_temperature",
        "outdoor_dry_bulb_temperature_predicted_1", "outdoor_dry_bulb_temperature_predicted_2",
        "outdoor_dry_bulb_temperature_predicted_3", "electricity_pricing",
        "electricity_pricing_predicted_1", "electricity_pricing_predicted_2", "dhw_demand")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8759)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    from stems.environment import STEMSEnvironment
    from stems.observations import obs_indices
    idx = obs_indices(*COLS)
    kw = {"env_kwargs": {"episode_time_steps": [(0, args.steps)]}} if args.steps < 8759 else {}
    env = STEMSEnvironment(schema=SCHEMA, seed=0, heat_pump=True, **kw)
    info = env.dhw_info()
    obs, _ = env.reset()
    rows = [obs]
    zero = np.zeros((env.num_buildings, env.action_dim), dtype=np.float32)
    while len(rows) < args.steps:
        obs, _r, term, trunc, _i = env.step(zero)
        if term or trunc:
            break
        rows.append(obs)
    a = np.array([[[float(o[i]) for i in idx] for o in r] for r in rows])
    out = Path(args.out) if args.out else REPO / "results" / "dhw_paper" / f"series_{len(rows)}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    names = [getattr(b, "name", str(i)) for i, b in enumerate(env.env.buildings)] \
        if hasattr(env, "env") else []
    np.savez(out, data=a, cols=np.array(COLS), capacity=np.asarray(info["capacity"], float),
             loss_coefficient=np.asarray(info["loss_coefficient"], float),
             buildings=np.array(names), schema=SCHEMA,
             source=f"STEMSEnvironment({SCHEMA}, seed=0, heat_pump=True), zero actions")
    print(out, a.shape, "capacity", np.round(info["capacity"], 3).tolist(),
          "loss", np.round(info["loss_coefficient"], 5).tolist())


if __name__ == "__main__":
    main()
