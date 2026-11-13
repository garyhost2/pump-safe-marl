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

W_COMFORT_H, W_START, W_DIS, W_SH, W_ABOVE = 20.0, 0.1, 50.0, 5.0, 0.0


def extra_v3(info):
    rec = info["record"]
    return (W_COMFORT_H * np.asarray(rec["below_floor"], float) + W_START * np.asarray(rec["starts"], float)
            + W_ABOVE * np.maximum(np.asarray(rec["t_in"], float) - 23.0, 0.0))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="A", choices=list("ABCDZ"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500_000)
    ap.add_argument("--chunk-hours", type=int, default=2184)
    ap.add_argument("--eval", nargs="+", default=["karlstad:2024", "karlstad:2025", "lulea:2025"])
    ap.add_argument("--bc-dir", default=".scratch/v3/bc")
    ap.add_argument("--out", default=".scratch/v3")
    ap.add_argument("--eval-hours", type=int, default=None)
    a = ap.parse_args(argv)
    import torch
    from experiments.dhw_v2_rl.train_v2 import eval_sets
    from stems.dhw_v2_rl.common import RewardSpec, TrackConfig, dumps, load_site, make_env_v2, metrics, slice_hours, window_starts
    from stems.dhw_v2_rl.rl_v3 import PPOv3, SetpointLayer, run_v3, train_price_ref, PREHEAT_BASE

    run = f"v3{a.variant}_seed{a.seed}"
    out = Path(a.out) / "runs" / run
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print(f"[{run}] done already"); return
    hb = out / "heartbeat.json"
    if hb.exists() and time.time() - hb.stat().st_mtime < 1800:
        print(f"[{run}] running elsewhere"); return
    hb.write_text("{}")
    tr, bh = load_site("karlstad", [2022, 2023, 2024])
    pref = train_price_ref()
    spec = RewardSpec(w_comfort_h=W_COMFORT_H, w_start=W_START, w_dis=W_DIS, w_sh=W_SH)
    track = TrackConfig()
    mode = "absolute" if a.variant == "D" else "residual"
    chunk = a.chunk_hours
    starts = window_starts(tr, chunk)
    rng = np.random.default_rng(20_000 + a.seed)
    B, d = tr.B, 101
    roll = 546
    pol = PPOv3(d, B, seed=a.seed, reward_scale=1.0 / pref, rollout=roll, total_updates=int(np.ceil(a.steps / roll)),
                zero_init=(a.variant != "D"), bc_coef=1.0 if a.variant == "A" else 0.0)
    meta = {"run": run, "variant": a.variant, "seed": a.seed, "steps_budget": a.steps, "chunk_hours": chunk, "mode": mode,
            "preheat_base": PREHEAT_BASE, "price_ref": pref, "hp": pol.hp, "reward": {"w_comfort_h": W_COMFORT_H,
            "w_start": W_START, "w_dis": W_DIS, "w_sh_below20": W_SH, "w_above23": W_ABOVE, "k_correction": 1.0},
            "borehole": bh, "train": "karlstad 2022-2024", "eval": a.eval}
    rows, t0 = [], time.time()
    if a.variant in "AC":
        X, Y = [], []
        for y in (2022, 2023, 2024):
            z = np.load(Path(a.bc_dir) / f"bc_karlstad_{y}_mpc_band_20_23.npz")
            ok = np.isfinite(z["plan"]).all(axis=1)
            Xy = z["X"][ok].reshape(-1, d)
            ds = np.clip((z["plan"][ok] - z["sp"][ok][:, None]) / 1.5, -1.0, 1.0).reshape(-1)
            X.append(Xy); Y.append(np.stack([np.zeros_like(ds), ds], 1))
        X, Y = np.concatenate(X), np.concatenate(Y)
        hist = pol.fit_bc(X, Y)
        meta["bc"] = {"samples": int(len(X)), "mse_first": hist[0], "mse_last": hist[-1],
                      "target_delta_s_mean": float(Y[:, 1].mean()), "target_delta_s_std": float(Y[:, 1].std()),
                      "delta_d_target": "0 (DHW kept at the baseline: the MPC charge decision has no unique setpoint inverse)"}
    (out / "meta.json").write_text(dumps(meta))
    steps, ep = 0, 0
    if a.variant in "ABD":
        while steps < a.steps:
            s0 = int(rng.choice(starts))
            env = make_env_v2(slice_hours(tr, s0, s0 + chunk), track, spec, filtered=True, k_correction=1.0,
                              borehole=bh, price_ref=pref)
            layer = SetpointLayer(B, mode)
            stats, wall = run_v3(env, pol, layer, train=True, extra=extra_v3, price_ref=pref)
            if ep == 0 and not pol.freeze_obs:
                pol.freeze_obs = True
            steps += env.data.T
            S = env.summary()
            tot = lambda k: float(np.sum(S[k]))
            row = {"episode": ep, "start_hour": s0, "steps": steps, "cost": tot("cost"), "viol_intra_hour": tot("hours_below_floor"),
                   "starts": tot("starts"), "sh_degree_hours": tot("sh_degree_hours_below_min"),
                   "intervention_rate": float(np.mean(S["intervention_rate"])), "bc_w": float(stats[-1][1]) if stats else 0.0,
                   "std": float(pol.pi.log_std.exp().mean()), "wall_s": wall, "elapsed_s": time.time() - t0}
            rows.append(row)
            ep += 1
            with open(out / "curve.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row)); w.writeheader(); w.writerows(rows)
            hb.write_text(json.dumps({"steps": steps, "of": a.steps, "elapsed_s": time.time() - t0, "last": row}))
            print(f"[{run}] ep {ep} steps {steps} cost {row['cost']:.0f} viol {row['viol_intra_hour']:.0f} std {row['std']:.3f} {wall:.0f}s", flush=True)
    torch.save(pol.state_dict(), out / "policy.pt")
    t1 = time.time()
    res = {}
    for name, (data, ebh) in eval_sets(a.eval).items():
        if a.eval_hours:
            data = slice_hours(data, 0, a.eval_hours)
        env = make_env_v2(data, track, spec, filtered=True, k_correction=0.0, borehole=ebh, price_ref=pref)
        S = run_v3(env, pol, SetpointLayer(B, mode), train=False, price_ref=pref)
        m = metrics(S)
        res[name] = m
        (out / f"eval_{name}.json").write_text(dumps({"metrics": m, "currency": data.currency}))
    (out / "done.json").write_text(json.dumps({"train_s": t1 - t0, "eval_s": time.time() - t1, "eval": res}, indent=1))
    print(f"[{run}] done " + ", ".join(f"{k}: cost {v['cost']:.0f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    main()
