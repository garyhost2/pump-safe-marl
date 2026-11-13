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


def eval_sets(items):
    from stems.dhw_v2_rl.common import load_site, texas_split
    out = {}
    for it in items:
        if it == "texas":
            out["texas_eval"] = (texas_split()[1], None)
            continue
        it, _, area = it.partition("@")
        site, ys = it.split(":")
        d, bh = load_site(site, [int(y) for y in ys.split("+")], price_area=area or None)
        out[f"{site}_{ys}" + (f"@{area}" if area else "")] = (d, bh)
    return out


def evaluate(pol, data, bh, track, spec, price_ref, filtered):
    from stems.dhw_v2_rl.common import make_env_v2, metrics, run_episode
    env = make_env_v2(data, track, spec, filtered=filtered, k_correction=0.0, borehole=bh, price_ref=price_ref)
    S = run_episode(env, lambda e, X: pol.env_action(X))
    S["tank_nodes"] = env.cfg.tank_nodes
    m = metrics(S)
    m["objective_v2"] = spec.objective(m, data.currency)
    return {"metrics": m, "summary": {k: S[k] for k in S if k not in ("filter",)}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sweden", choices=["sweden", "texas"])
    ap.add_argument("--site", default="karlstad")
    ap.add_argument("--train-years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--eval", nargs="+", default=["karlstad:2025", "lulea:2025"])
    ap.add_argument("--learner", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--arm", default="a", choices=["a", "b", "c"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--chunk-hours", type=int, default=2184)
    ap.add_argument("--reward", default="", help="JSON RewardSpec overrides (SEK-equivalent)")
    ap.add_argument("--hp", default="", help="JSON learner overrides")
    ap.add_argument("--track", default="")
    ap.add_argument("--out", default=".scratch/fair")
    ap.add_argument("--tag", default="")
    ap.add_argument("--eval-hours", type=int, default=None, help="smoke only: truncate evaluation sets")
    a = ap.parse_args(argv)

    import torch
    from stems.dhw_v2_rl.common import (ARMS, ObsBuilder, RewardSpec, TrackConfig, dumps, load_site, make_env_v2,
                                        slice_hours, texas_split, window_starts)
    from stems.dhw_v2_rl.learners import PPO, SAC

    if a.dataset == "texas":
        tr, _, _ = texas_split()
        bh = None
    else:
        tr, bh = load_site(a.site, a.train_years)
    track = TrackConfig(**(json.loads(a.track) if a.track else {}))
    spec = RewardSpec(**(json.loads(a.reward) if a.reward else {}))
    arm = ARMS[a.arm]
    price_ref = float(np.mean(tr.price))
    run = f"{a.learner}_{a.arm}_seed{a.seed}{a.tag}"
    out = Path(a.out) / a.dataset / run
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print(f"[{run}] already done")
        return
    hbf = out / "heartbeat.json"
    if hbf.exists() and time.time() - hbf.stat().st_mtime < 1800:
        print(f"[{run}] running in another pool (fresh heartbeat), skipped")
        return
    hbf.write_text(json.dumps({"steps": 0, "of": a.steps, "episode": 0, "elapsed_s": 0.0, "started": time.time()}))
    chunk = a.chunk_hours if a.chunk_hours and a.chunk_hours < tr.T else tr.T
    starts = window_starts(tr, chunk) if chunk < tr.T else np.array([0])
    rng = np.random.default_rng(10_000 + a.seed)
    env0 = make_env_v2(slice_hours(tr, 0, min(chunk, 48)), track, spec, filtered=arm["filtered"],
                       k_correction=arm["k"], borehole=bh, price_ref=price_ref)
    d = ObsBuilder(env0).dim
    B = env0.B
    rs = 1.0 / price_ref
    hp = json.loads(a.hp) if a.hp else {}
    rs *= float(hp.pop("reward_mult", 1.0))
    if a.learner == "ppo":
        roll = hp.pop("rollout", 546)
        pol = PPO(d, B, seed=a.seed, reward_scale=rs, rollout=roll, total_updates=int(np.ceil(a.steps / roll)), **hp)
    else:
        pol = SAC(d, B, seed=a.seed, reward_scale=rs, obs_norm_steps=min(chunk, 4368), **hp)
    extra = spec.extra(tr)
    meta = {"run": run, "dataset": a.dataset, "site": a.site, "train_years": a.train_years, "train_hours": tr.T,
            "chunk_hours": chunk, "n_window_starts": int(len(starts)), "eval": a.eval, "learner": a.learner,
            "arm": a.arm, "arm_label": arm["label"], "seed": a.seed, "steps_budget": a.steps,
            "reward_spec": spec.__dict__, "reward_fx": spec.fx(tr), "env_weights": spec.env_weights(tr),
            "k_correction": arm["k"], "track": track.__dict__, "hp": pol.hp, "obs_dim": d, "borehole": bh,
            "price_ref": price_ref, "currency": tr.currency}
    (out / "meta.json").write_text(dumps(meta))
    rows, steps, ep = [], 0, 0
    t0 = time.time()
    while steps < a.steps:
        s0 = int(rng.choice(starts))
        env = make_env_v2(slice_hours(tr, s0, s0 + chunk), track, spec, filtered=arm["filtered"],
                          k_correction=arm["k"], borehole=bh, price_ref=price_ref)
        st = pol.train_episode(env, extra=extra)
        steps += env.data.T
        st.update(episode=ep, start_hour=s0, steps=steps, elapsed_s=time.time() - t0)
        S = env.summary()
        tot = lambda k: float(np.sum(S[k]))
        st["objective_v2"] = st["cost"] + spec.fx(tr) * (spec.w_comfort_h * tot("hours_below_floor") + spec.w_start * tot("starts")
                                                        + spec.w_dis * st["weeks_missed"] + spec.w_sh * st["sh_degree_hours"])
        st["hours"] = env.data.T
        rows.append(st)
        ep += 1
        keys = sorted({k for r in rows for k in r})
        with open(out / "curve.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        (out / "heartbeat.json").write_text(json.dumps({"steps": steps, "of": a.steps, "episode": ep,
                                                        "elapsed_s": time.time() - t0, "last": st}))
        print(f"[{run}] ep {ep} steps {steps}/{a.steps} obj {st['objective_v2']:.0f} cost {st['cost']:.0f} "
              f"viol {st['viol_intra_hour']:.0f} {st['wall_s']:.0f}s", flush=True)
    torch.save(pol.state_dict(), out / "policy.pt")
    t1 = time.time()
    res = {}
    for name, (data, ebh) in eval_sets(a.eval).items():
        if a.eval_hours:
            data = slice_hours(data, 0, a.eval_hours)
        for filtered in (True, False):
            mode = "filtered" if filtered else "raw"
            r = evaluate(pol, data, ebh, track, spec, price_ref, filtered)
            (out / f"eval_{name}_{mode}.json").write_text(dumps(r))
            res[f"{name}__{mode}"] = r["metrics"]
    (out / "done.json").write_text(json.dumps({"train_s": t1 - t0, "eval_s": time.time() - t1, "eval": res}, indent=1))
    print(f"[{run}] done " + ", ".join(f"{k}: {v['objective_v2']:.0f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
