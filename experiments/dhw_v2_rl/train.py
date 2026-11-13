#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def get_split(dataset, package=None):
    from stems.dhw_v2_rl.common import package_split, texas_split
    return texas_split() if dataset == "texas" else package_split(package)


def track_from(js):
    from stems.dhw_v2_rl.common import TrackConfig
    return TrackConfig(**(json.loads(js) if js else {}))


def evaluate_policy(pol, data, track, filtered: bool):
    from stems.dhw_v2_rl.common import make_env, metrics, run_episode
    env = make_env(data, track, filtered=filtered, k_correction=0.0)
    S = run_episode(env, lambda e, X: pol.env_action(X))
    S["tank_nodes"] = env.cfg.tank_nodes
    return {"metrics": metrics(S), "summary": {k: S[k] for k in S if k not in ("filter",)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="texas")
    ap.add_argument("--package", default=None)
    ap.add_argument("--learner", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--arm", default="a", choices=["a", "b", "c"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--out", default="results_dhw_v2_rl")
    ap.add_argument("--track", default="", help="JSON TrackConfig overrides")
    ap.add_argument("--hp", default="", help="JSON learner hyper-parameter overrides")
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-hours", type=int, default=None, help="smoke: truncate the training split")
    ap.add_argument("--eval-hours", type=int, default=None, help="smoke: truncate the evaluation split")
    a = ap.parse_args(argv)

    import torch
    from stems.dhw_v2_rl.common import ARMS, ObsBuilder, dumps, make_env, slice_hours
    from stems.dhw_v2_rl.learners import PPO, SAC

    tr, ev, split = get_split(a.dataset, a.package)
    if a.train_hours:
        tr = slice_hours(tr, 0, a.train_hours)
    if a.eval_hours:
        ev = slice_hours(ev, 0, a.eval_hours)
    track = track_from(a.track)
    arm = ARMS[a.arm]
    run = f"{a.learner}_{a.arm}_seed{a.seed}{a.tag}"
    out = Path(a.out) / a.dataset / run
    out.mkdir(parents=True, exist_ok=True)
    env = make_env(tr, track, filtered=arm["filtered"], k_correction=arm["k"])
    d = ObsBuilder(env).dim
    rs = 1.0 / float(np.mean(tr.price))
    hp = json.loads(a.hp) if a.hp else {}
    rs *= float(hp.pop("reward_mult", 1.0))
    if a.learner == "ppo":
        steps = tr.T
        roll = hp.pop("rollout", max(1, steps // 8))
        pol = PPO(d, env.B, seed=a.seed, reward_scale=rs, rollout=roll,
                  total_updates=a.episodes * int(np.ceil(steps / roll)), **hp)
    else:
        pol = SAC(d, env.B, seed=a.seed, reward_scale=rs, obs_norm_steps=tr.T, **hp)
    meta = {"run": run, "dataset": a.dataset, "split": split, "learner": a.learner, "arm": a.arm,
            "arm_label": arm["label"], "seed": a.seed, "episodes": a.episodes, "track": track.__dict__,
            "hp": pol.hp, "obs_dim": d, "houses": env.agents, "weights": {k: getattr(env.cfg, k) for k in
                                                                          ("w_comfort", "w_sh", "w_dis", "k_correction")},
            "train_hours": tr.T, "eval_hours": ev.T, "reward_scale_1_over_mean_price": rs}
    (out / "meta.json").write_text(dumps(meta))
    rows = []
    t0 = time.time()
    for ep in range(a.episodes):
        st = pol.train_episode(env)
        st["episode"] = ep
        rows.append(st)
        keys = sorted({k for r in rows for k in r})
        with open(out / "curve.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        (out / "heartbeat.json").write_text(json.dumps({"episode": ep + 1, "of": a.episodes,
                                                        "elapsed_s": time.time() - t0, "last": st}))
        print(f"[{run}] ep {ep + 1}/{a.episodes} cost {st['cost']:.1f} viol {st['viol_intra_hour']:.0f} "
              f"miss {st['weeks_missed']} ir {st.get('intervention_rate', float('nan')):.3f} {st['wall_s']:.0f}s", flush=True)
    torch.save(pol.state_dict(), out / "policy.pt")
    t1 = time.time()
    ef = evaluate_policy(pol, ev, track, True)
    (out / "eval_filtered.json").write_text(dumps(ef))
    er = evaluate_policy(pol, ev, track, False)
    (out / "eval_raw.json").write_text(dumps(er))
    (out / "done.json").write_text(json.dumps({"train_s": t1 - t0, "eval_s": time.time() - t1,
                                               "filtered": ef["metrics"], "raw": er["metrics"]}, indent=1))
    print(f"[{run}] done: eval filtered cost {ef['metrics']['cost']:.1f}, raw cost {er['metrics']['cost']:.1f}", flush=True)


if __name__ == "__main__":
    main()
