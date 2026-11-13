from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def profile(root: Path, arm_name: str, seed: int) -> Dict[str, Any]:
    from experiments.controllers import ARMS, build_controller
    from experiments.runner import make_config
    from experiments.scenario import Scenario
    from stems.environment import STEMSEnvironment
    from stems.utils import HistoryBuffer, set_seed

    hits = sorted(root.glob(f"*/{arm_name}__seed{seed}.json"))
    if not hits:
        raise SystemExit(f"no record for {arm_name} seed {seed} under {root}")
    path = hits[-1]
    stored = json.loads(path.read_text(encoding="utf-8"))
    meta = stored["meta"]
    model_dir = path.with_name(f"{path.stem}_model")

    scenario = Scenario(**{k: meta["scenario"][k] for k in (
        "schema", "season", "subset_seed", "n_buildings", "days",
        "grid_cap_kw", "building_cap_kw", "hvac_control")})
    set_seed(seed)
    env = STEMSEnvironment(schema=scenario.schema_path(), seed=seed, heat_pump=True,
                           env_kwargs=scenario.env_kwargs("eval"),
                           hvac_control=scenario.hvac_control)
    config = make_config(scenario, meta.get("learner") or {})
    controller = build_controller(ARMS[arm_name], env, config)
    if ARMS[arm_name].learns:
        controller.load(str(model_dir))

    from experiments.runner import _window_len

    max_steps = _window_len(scenario.env_kwargs("eval"))
    hist = HistoryBuffer(env.num_buildings, env.obs_dim, config.transformer.window_size)
    obs, _ = env.reset()
    hist.prime(obs)
    recorded: List[np.ndarray] = []
    executed: List[np.ndarray] = []
    n, done = 0, False
    while not done:
        action = controller.select_action(obs, hist.get(), explore=False)
        recorded.append(np.asarray(action, dtype=np.float32).copy())
        nxt, _, term, trunc, _ = env.step(action)
        executed.append(np.asarray(env.executed_actions, dtype=np.float32).copy())
        hist.update(nxt)
        obs = nxt
        n += 1
        done = bool(term or trunc) or n >= max_steps

    a = np.stack(recorded)
    names = ["dhw_storage", "electrical_storage", "hvac"]
    out: Dict[str, Any] = {
        "arm": arm_name, "seed": seed, "steps": int(a.shape[0]),
        "buildings": int(a.shape[1]), "stored_cost": stored["eval"].get("stems_cost"),
        "stored_electricity": stored["eval"].get("stems_electricity_consumption"),
        "channels": {},
    }
    for j, nm in enumerate(names):
        c = a[:, :, j]
        out["channels"][nm] = {
            "mean": float(c.mean()), "sd": float(c.std()),
            "abs_mean": float(np.abs(c).mean()),
            "p05": float(np.percentile(c, 5)), "p50": float(np.percentile(c, 50)),
            "p95": float(np.percentile(c, 95)),
            "saturated_hi": float((c > 0.99).mean()), "saturated_lo": float((c < -0.99).mean()),
            "near_zero": float((np.abs(c) < 0.01).mean()),
            "step_change_abs_mean": float(np.abs(np.diff(c, axis=0)).mean()),
        }
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="results/t1_unshielded_ep50")
    p.add_argument("--arms", nargs="+", default=["sac", "marlisa", "madcq", "dmappo"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/action_profile")
    args = p.parse_args(argv)

    root = Path(args.root)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in args.arms:
        name = arm if arm.endswith("-unshielded") else f"{arm}-unshielded"
        rec = profile(root, name, args.seed)
        rows.append(rec)
        (outdir / f"{name}__seed{args.seed}.json").write_text(
            json.dumps(rec, indent=1), encoding="utf-8")
        print(f"[done] {name}")

    hdr = ("arm", "chan", "mean", "sd", "|a|", "sat+", "sat-", "~0", "d|a|")
    print("\n%-9s %-20s %7s %6s %6s %6s %6s %6s %6s" % hdr)
    for r in rows:
        for nm, c in r["channels"].items():
            print("%-9s %-20s %7.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f" % (
                r["arm"].replace("-unshielded", ""), nm, c["mean"], c["sd"], c["abs_mean"],
                c["saturated_hi"], c["saturated_lo"], c["near_zero"],
                c["step_change_abs_mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
