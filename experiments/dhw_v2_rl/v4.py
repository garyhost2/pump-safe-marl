#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROOT = Path(os.environ.get("V4_ROOT", REPO / ".scratch" / "v4"))
TRAIN_YEARS = (2022, 2023)
VAL = "karlstad:2024"
TEST = ("karlstad:2025", "lulea:2025")
BETAS = {1: 0.5, 2: 0.25, 3: 0.1, 4: 0.0, 5: 0.0}
N_IT = 5


def obj(m, cur="SEK", band=(20, 23)):
    fx = 1.0 / 10.28 if cur.upper() == "USD" else 1.0
    lo, hi = band
    return m["cost"] + fx * (20.0 * m["viol_intra_hour"] + 0.1 * m["starts"] + 50.0 * m["weeks_missed"]
                             + 5.0 * m[f"band_{lo:g}_{hi:g}_below_kh"])


def env_for(setname, k):
    from experiments.dhw_v2_rl.train_v2 import eval_sets
    from stems.dhw_v2_rl.common import RewardSpec, TrackConfig, make_env_v2
    from stems.dhw_v2_rl.rl_v3 import train_price_ref
    (data, bh), = eval_sets([setname]).values()
    if os.environ.get("V4_HOURS"):
        from stems.dhw_v2_rl.common import slice_hours
        data = slice_hours(data, 0, int(os.environ["V4_HOURS"]))
    return make_env_v2(data, TrackConfig(), RewardSpec(), filtered=True, k_correction=k, borehole=bh,
                       price_ref=train_price_ref("karlstad", TRAIN_YEARS)), data


def save_npz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    os.replace(tmp, path)


def write_json(path, o):
    from stems.dhw_v2_rl.common import dumps
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(dumps(o))
    os.replace(tmp, path)


def band_tag(band):
    return f"{band[0]}_{band[1]}"


def t_expert(a):
    from stems.dhw_v2_rl.common import metrics
    from stems.dhw_v2_rl.mpc_v3 import MPCBandController
    from stems.dhw_v2_rl.rl_v4 import LayerV4, rollout, train_extra
    band = tuple(a.band)
    rec = a.record
    env, data = env_for(a.set, 1.0 if rec else 0.0)
    t0 = time.time()
    S, D = rollout(env, LayerV4(env.B), expert=MPCBandController(env.B, band=band), expert_exec=True, record=rec,
                   extra=train_extra if rec else None)
    m = metrics(S); m["wall_s"] = time.time() - t0
    tag = f"E{band_tag(band)}"
    if rec:
        save_npz(ROOT / "data" / f"{tag}_{a.set.replace(':', '_')}.npz", D)
        m["label_ok_frac"] = float(D["ok"].mean())
    write_json(ROOT / "evals" / tag / f"{a.set.replace(':', '_')}{'_train' if rec else ''}.json",
               {"metrics": m, "currency": data.currency, "k_correction": 1.0 if rec else 0.0})


def load_agg(seed, it, band=(20, 23), setpoint_only=False):
    X, Y = [], []
    srcs = [ROOT / "data" / f"E{band_tag(band)}_karlstad_{y}.npz" for y in TRAIN_YEARS]
    srcs += [ROOT / "data" / f"G{band_tag(band)}_seed{seed}_it{i}_{y}.npz" for i in range(1, it + 1) for y in TRAIN_YEARS]
    for f in srcs:
        z = np.load(f)
        ok = z["ok"].reshape(-1)
        X.append(z["X"].reshape(-1, 101)[ok].astype(np.float32))
        y = z["Y"].reshape(-1, 2)[ok]
        if setpoint_only:
            y = np.stack([np.zeros(len(y), np.float32), y[:, 1]], 1)
        Y.append(y)
    return np.concatenate(X), np.concatenate(Y), [f.name for f in srcs]


def t_fit(a):
    import torch
    from stems.dhw_v2_rl.rl_v4 import PolicyV4
    band = tuple(a.band)
    rd = ROOT / "runs" / f"G{band_tag(band)}_seed{a.seed}"
    rd.mkdir(parents=True, exist_ok=True)
    variants = [("", False)] + ([("s", True)] if a.it == 0 and band == (20, 23) else [])
    for suf, sp_only in variants:
        X, Y, srcs = load_agg(a.seed, a.it, band, sp_only)
        p = PolicyV4(seed=1000 * a.seed + a.it)
        t0 = time.time()
        info = p.fit_bc(X, Y)
        info.update({"sources": srcs, "fit_s": time.time() - t0, "setpoint_only": sp_only, "it": a.it, "band": band})
        name = f"pol_it{a.it}{suf}"
        torch.save(p.state_dict(), rd / f"{name}.pt")
        write_json(rd / f"{name}.json", info)


def t_roll(a):
    from stems.dhw_v2_rl.mpc_v3 import MPCBandController
    from stems.dhw_v2_rl.rl_v4 import LayerV4, PolicyV4, rollout, train_extra
    from stems.dhw_v2_rl.common import metrics
    band = tuple(a.band)
    rd = ROOT / "runs" / f"G{band_tag(band)}_seed{a.seed}"
    pol = PolicyV4.load(str(rd / f"pol_it{a.it - 1}.pt"))
    env, data = env_for(f"karlstad:{a.year}", 1.0)
    rng = np.random.default_rng(7919 * a.seed + 31 * a.it + a.year)
    t0 = time.time()
    S, D = rollout(env, LayerV4(env.B), pol, expert=MPCBandController(env.B, band=band), beta=BETAS[a.it], rng=rng,
                   record=True, extra=train_extra)
    save_npz(ROOT / "data" / f"G{band_tag(band)}_seed{a.seed}_it{a.it}_{a.year}.npz", D)
    m = metrics(S); m["wall_s"] = time.time() - t0; m["beta"] = BETAS[a.it]
    m["label_gap_mse"] = np.nanmean((D["A"] - D["Y"]) ** 2, axis=(0, 1)).tolist()
    write_json(rd / f"roll_it{a.it}_{a.year}.json", {"metrics": m, "currency": data.currency, "k_correction": 1.0})


def evaluate_policy(pol, setname):
    from stems.dhw_v2_rl.common import metrics
    from stems.dhw_v2_rl.rl_v4 import LayerV4, rollout
    env, data = env_for(setname, 0.0)
    t0 = time.time()
    S, _ = rollout(env, LayerV4(env.B), pol)
    m = metrics(S); m["wall_s"] = time.time() - t0
    return {"metrics": m, "currency": data.currency, "objective_20_23": obj(m, data.currency)}


def t_eval(a):
    from stems.dhw_v2_rl.rl_v4 import PolicyV4
    pol = PolicyV4.load(a.policy)
    for s in a.sets:
        f = Path(a.out) / f"eval_{s.replace(':', '_')}.json"
        if f.exists():
            continue
        write_json(f, evaluate_policy(pol, s))


def t_select(a):
    band = tuple(a.band)
    rd = ROOT / "runs" / f"G{band_tag(band)}_seed{a.seed}"
    vals = {}
    for i in range(N_IT + 1):
        j = json.loads((rd / f"val_it{i}" / "eval_karlstad_2024.json").read_text())
        vals[i] = j["objective_20_23"]
    best = min(vals, key=vals.get)
    shutil.copy(rd / f"pol_it{best}.pt", rd / "pol_G.pt")
    write_json(rd / "select.json", {"val_objective_20_23": vals, "best_it": best})


def _windows(chunk):
    from stems.dhw_v2_rl.common import load_site, window_starts
    tr, bh = load_site("karlstad", list(TRAIN_YEARS))
    return tr, bh, window_starts(tr, chunk)


def t_ppo(a):
    import csv
    import torch
    from stems.dhw_v2_rl.common import RewardSpec, TrackConfig, make_env_v2, slice_hours
    from stems.dhw_v2_rl.rl_v3 import train_price_ref
    from stems.dhw_v2_rl.rl_v4 import AnchoredPPO, LayerV4, PolicyV4, ppo_episode, train_extra
    rd = ROOT / "runs" / f"GP_seed{a.seed}"
    rd.mkdir(parents=True, exist_ok=True)
    g = ROOT / "runs" / f"G20_23_seed{a.seed}"
    init = PolicyV4.load(str(g / "pol_G.pt"))
    sel = json.loads((g / "select.json").read_text())
    pref = train_price_ref("karlstad", TRAIN_YEARS)
    chunk, roll = 2184, 546
    tr, bh, starts = _windows(chunk)
    ag = AnchoredPPO(init, tr.B, seed=a.seed, reward_scale=1.0 / pref, rollout=roll,
                     total_updates=int(np.ceil(a.steps / roll)))
    val0 = sel["val_objective_20_23"][str(sel["best_it"])] if str(sel["best_it"]) in sel["val_objective_20_23"] else \
        sel["val_objective_20_23"][sel["best_it"]]
    hist = [{"steps": 0, "val_objective_20_23": val0}]
    best = (val0, 0)
    torch.save(init.state_dict(), rd / "pol_best.pt")
    rng = np.random.default_rng(31_000 + a.seed)
    spec, track = RewardSpec(), TrackConfig()
    steps, ep, rows, t0 = 0, 0, [], time.time()
    next_val = a.val_every
    while steps < a.steps:
        s0 = int(rng.choice(starts))
        env = make_env_v2(slice_hours(tr, s0, s0 + chunk), track, spec, filtered=True, k_correction=1.0, borehole=bh,
                          price_ref=pref)
        st = ppo_episode(env, ag, LayerV4(tr.B), train_extra)
        steps += env.data.T
        S = env.summary()
        rows.append({"episode": ep, "steps": steps, "cost": float(np.sum(S["cost"])),
                     "viol": float(np.sum(S["hours_below_floor"])), "anchor_w": float(st[-1][1]),
                     "anchor_term": float(st[-1][0][2]), "elapsed_s": time.time() - t0})
        ep += 1
        with open(rd / "curve.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        if steps >= next_val or steps >= a.steps:
            next_val += a.val_every
            v = evaluate_policy(ag.pol, VAL)
            hist.append({"steps": steps, "val_objective_20_23": v["objective_20_23"], "cost": v["metrics"]["cost"]})
            torch.save(ag.pol.state_dict(), rd / f"pol_s{steps}.pt")
            if v["objective_20_23"] < best[0]:
                best = (v["objective_20_23"], steps)
                torch.save(ag.pol.state_dict(), rd / "pol_best.pt")
            write_json(rd / "val_hist.json", {"hist": hist, "best": best, "hp": ag.hp, "init_it": sel["best_it"]})
            print(f"[GP{a.seed}] steps {steps} val {v['objective_20_23']:.0f} best {best}", flush=True)
    torch.save(ag.pol.state_dict(), rd / "pol_last.pt")
    for f in rd.glob("pol_s*.pt"):
        f.unlink()
    write_json(rd / "done.json", {"hist": hist, "best": best, "hp": ag.hp, "train_s": time.time() - t0})


def t_iql(a):
    import csv
    import torch
    from stems.dhw_v2_rl.common import RewardSpec, TrackConfig, make_env_v2, slice_hours
    from stems.dhw_v2_rl.rl_v3 import train_price_ref
    from stems.dhw_v2_rl.rl_v4 import IQL, LayerV4, PolicyV4, Replay, rollout, train_extra
    rd = ROOT / "runs" / f"Q_seed{a.seed}"
    rd.mkdir(parents=True, exist_ok=True)
    g = ROOT / "runs" / f"G20_23_seed{a.seed}"
    init = PolicyV4.load(str(g / "pol_it0.pt"))
    pref = train_price_ref("karlstad", TRAIN_YEARS)
    rep = Replay()
    srcs = [ROOT / "data" / f"E20_23_karlstad_{y}.npz" for y in TRAIN_YEARS]
    srcs += [ROOT / "data" / f"G20_23_seed{a.seed}_it{i}_{y}.npz" for i in range(1, N_IT + 1) for y in TRAIN_YEARS]
    for f in srcs:
        z = np.load(f)
        rep.add_episode(z["X"], z["A"], z["R"])
    q = IQL(init, seed=a.seed, reward_scale=1.0 / pref)
    t0 = time.time()
    log = []
    beta0 = q.hp["beta"]
    q.hp["beta"] = 0.0
    opp = q.op
    q.op = torch.optim.SGD(q.pol.pi.parameters(), lr=0.0)
    for s in range(a.v_steps):
        out = q.step(rep)
        if s % 2000 == 0:
            log.append({"phase": "value", "step": s, "lv": out[0], "lq": out[1], "adv_sd": out[3]})
    sds = [q.step(rep)[3] for _ in range(50)]
    adv_sd = float(np.mean(sds))
    q.hp["beta"] = beta0 / max(adv_sd, 1e-6)
    q.op = opp
    for s in range(a.offline_steps):
        out = q.step(rep)
        if s % 2000 == 0:
            log.append({"phase": "offline", "step": s, "lv": out[0], "lq": out[1], "lp": out[2], "adv_sd": out[3],
                        "w_mean": out[4]})
    torch.save(q.pol.state_dict(), rd / "pol_Q0.pt")
    info = {"n_transitions": len(rep), "sources": [f.name for f in srcs], "adv_sd": adv_sd, "beta_eff": q.hp["beta"],
            "hp": q.hp, "offline_s": time.time() - t0}
    v = evaluate_policy(q.pol, VAL)
    hist = [{"steps": 0, "val_objective_20_23": v["objective_20_23"], "cost": v["metrics"]["cost"]}]
    best = (v["objective_20_23"], 0)
    torch.save(q.pol.state_dict(), rd / "pol_best.pt")
    write_json(rd / "val_hist.json", {"hist": hist, "best": best, "info": info})
    chunk = 2184
    tr, bh, starts = _windows(chunk)
    rng = np.random.default_rng(41_000 + a.seed)
    steps, next_val, rows = 0, a.val_every, []
    spec, track = RewardSpec(), TrackConfig()
    while steps < a.online_steps:
        s0 = int(rng.choice(starts))
        env = make_env_v2(slice_hours(tr, s0, s0 + chunk), track, spec, filtered=True, k_correction=1.0, borehole=bh,
                          price_ref=pref)
        S, D = rollout(env, LayerV4(tr.B), q.pol, noise=a.noise, rng=rng, record=True, extra=train_extra)
        rep.add_episode(D["X"], D["A"], D["R"])
        T = D["R"].shape[0]
        for _ in range(T):
            out = q.step(rep)
        steps += T
        rows.append({"steps": steps, "cost": float(np.sum(S["cost"])), "lq": out[1], "lp": out[2], "adv_sd": out[3],
                     "elapsed_s": time.time() - t0})
        with open(rd / "curve.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        if steps >= next_val or steps >= a.online_steps:
            next_val += a.val_every
            v = evaluate_policy(q.pol, VAL)
            hist.append({"steps": steps, "val_objective_20_23": v["objective_20_23"], "cost": v["metrics"]["cost"]})
            if v["objective_20_23"] < best[0]:
                best = (v["objective_20_23"], steps)
                torch.save(q.pol.state_dict(), rd / "pol_best.pt")
            write_json(rd / "val_hist.json", {"hist": hist, "best": best, "info": info})
            print(f"[Q{a.seed}] steps {steps} val {v['objective_20_23']:.0f} best {best}", flush=True)
    torch.save(q.pol.state_dict(), rd / "pol_last.pt")
    with open(rd / "iql_log.json", "w") as f:
        json.dump(log, f)
    write_json(rd / "done.json", {"hist": hist, "best": best, "info": info, "train_s": time.time() - t0})


PY = sys.executable


def task_list(seeds, extra_bands=True, ppo_steps=250_000, online_steps=100_000):
    T = []
    D, R, E = ROOT / "data", ROOT / "runs", ROOT / "evals"
    me = ["-B", str(Path(__file__).resolve())]

    def add(name, args, needs, outs):
        T.append({"name": name, "cmd": [PY] + me + args, "needs": [str(x) for x in needs], "outs": [str(x) for x in outs]})

    bands = [(20, 23)] + ([(21, 22)] if extra_bands else [])
    for b in bands:
        bt = band_tag(b)
        for y in TRAIN_YEARS:
            add(f"E{bt}_{y}", ["expert", "--set", f"karlstad:{y}", "--band", *map(str, b), "--record"], [],
                [D / f"E{bt}_karlstad_{y}.npz"])
        for s in ([VAL] + list(TEST)):
            add(f"E{bt}_{s}", ["expert", "--set", s, "--band", *map(str, b)], [], [E / f"E{bt}" / f"{s.replace(':', '_')}.json"])
    for seed in seeds:
        for b in bands:
            bt = band_tag(b)
            rd = R / f"G{bt}_seed{seed}"
            add(f"fit{bt}_s{seed}_it0", ["fit", "--seed", str(seed), "--it", "0", "--band", *map(str, b)],
                [D / f"E{bt}_karlstad_{y}.npz" for y in TRAIN_YEARS], [rd / "pol_it0.pt"])
            sets_c = [VAL] + list(TEST)
            add(f"evalC{bt}_s{seed}", ["eval", "--policy", str(rd / "pol_it0.pt"), "--out", str(rd / "val_it0"), "--sets", *sets_c],
                [rd / "pol_it0.pt"], [rd / "val_it0" / f"eval_{s.replace(':', '_')}.json" for s in sets_c])
            if b != (20, 23):
                continue
            add(f"evalCs_s{seed}", ["eval", "--policy", str(rd / "pol_it0s.pt"), "--out", str(R / f"Cs_seed{seed}"), "--sets", *sets_c],
                [rd / "pol_it0.pt"], [R / f"Cs_seed{seed}" / f"eval_{s.replace(':', '_')}.json" for s in sets_c])
            for i in range(1, N_IT + 1):
                for y in TRAIN_YEARS:
                    add(f"roll_s{seed}_it{i}_{y}", ["roll", "--seed", str(seed), "--it", str(i), "--year", str(y)],
                        [rd / f"pol_it{i - 1}.pt"], [D / f"G{bt}_seed{seed}_it{i}_{y}.npz"])
                add(f"fit_s{seed}_it{i}", ["fit", "--seed", str(seed), "--it", str(i)],
                    [D / f"G{bt}_seed{seed}_it{i}_{y}.npz" for y in TRAIN_YEARS], [rd / f"pol_it{i}.pt"])
                add(f"val_s{seed}_it{i}", ["eval", "--policy", str(rd / f"pol_it{i}.pt"), "--out", str(rd / f"val_it{i}"), "--sets", VAL],
                    [rd / f"pol_it{i}.pt"], [rd / f"val_it{i}" / "eval_karlstad_2024.json"])
            add(f"select_s{seed}", ["select", "--seed", str(seed)],
                [rd / f"val_it{i}" / "eval_karlstad_2024.json" for i in range(N_IT + 1)], [rd / "select.json"])
            add(f"evalG_s{seed}", ["eval", "--policy", str(rd / "pol_G.pt"), "--out", str(R / f"G_seed{seed}"), "--sets", VAL, *TEST],
                [rd / "select.json"], [R / f"G_seed{seed}" / f"eval_{s.replace(':', '_')}.json" for s in TEST])
            gp = R / f"GP_seed{seed}"
            add(f"ppo_s{seed}", ["ppo", "--seed", str(seed), "--steps", str(ppo_steps)], [rd / "select.json"], [gp / "done.json"])
            for k in ("best", "last"):
                add(f"evalGP{k}_s{seed}", ["eval", "--policy", str(gp / f"pol_{k}.pt"), "--out", str(gp / f"eval_{k}"), "--sets", VAL, *TEST],
                    [gp / "done.json"], [gp / f"eval_{k}" / f"eval_{s.replace(':', '_')}.json" for s in TEST])
            qd = R / f"Q_seed{seed}"
            add(f"iql_s{seed}", ["iql", "--seed", str(seed), "--online-steps", str(online_steps)],
                [D / f"G{bt}_seed{seed}_it{N_IT}_{y}.npz" for y in TRAIN_YEARS] + [rd / "pol_it0.pt"], [qd / "done.json"])
            for k in ("Q0", "best", "last"):
                add(f"evalQ{k}_s{seed}", ["eval", "--policy", str(qd / f"pol_{k}.pt"), "--out", str(qd / f"eval_{k}"), "--sets", VAL, *TEST],
                    [qd / "done.json"], [qd / f"eval_{k}" / f"eval_{s.replace(':', '_')}.json" for s in TEST])
    return T


def t_queue(a):
    tasks = task_list(a.seeds, extra_bands=not a.no_2122, ppo_steps=a.ppo_steps, online_steps=a.online_steps)
    if a.only:
        tasks = [t for t in tasks if any(t["name"].startswith(p) for p in a.only)]
    logd = ROOT / "logs"
    logd.mkdir(parents=True, exist_ok=True)
    done = lambda t: all(Path(o).exists() for o in t["outs"])
    ready = lambda t: all(Path(n).exists() for n in t["needs"])
    pend = [t for t in tasks if not done(t)]
    run = {}
    fails = []
    print(f"[queue] {len(pend)} tasks pending of {len(tasks)}", flush=True)
    while pend or run:
        for n, (p, t, t0) in list(run.items()):
            if p.poll() is not None:
                del run[n]
                ok = p.returncode == 0 and done(t)
                print(f"[queue] {'done' if ok else 'FAIL'} {n} rc={p.returncode} {time.time() - t0:.0f}s", flush=True)
                if not ok:
                    fails.append(n)
        for t in list(pend):
            if len(run) >= a.workers:
                break
            if ready(t):
                pend.remove(t)
                f = open(logd / f"{t['name']}.log", "w")
                env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                run[t["name"]] = (subprocess.Popen(t["cmd"], stdout=f, stderr=subprocess.STDOUT, cwd=str(REPO), env=env), t, time.time())
                print(f"[queue] start {t['name']} ({len(run)} running, {len(pend)} pending)", flush=True)
        if not run and pend and not any(ready(t) for t in pend):
            print("[queue] blocked: " + ", ".join(t["name"] for t in pend[:10]), flush=True)
            break
        time.sleep(5)
    print(f"[queue] finished; failures: {fails}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("expert"); p.add_argument("--set"); p.add_argument("--band", type=int, nargs=2, default=[20, 23]); p.add_argument("--record", action="store_true")
    p = sp.add_parser("fit"); p.add_argument("--seed", type=int); p.add_argument("--it", type=int); p.add_argument("--band", type=int, nargs=2, default=[20, 23])
    p = sp.add_parser("roll"); p.add_argument("--seed", type=int); p.add_argument("--it", type=int); p.add_argument("--year", type=int); p.add_argument("--band", type=int, nargs=2, default=[20, 23])
    p = sp.add_parser("eval"); p.add_argument("--policy"); p.add_argument("--out"); p.add_argument("--sets", nargs="+")
    p = sp.add_parser("select"); p.add_argument("--seed", type=int); p.add_argument("--band", type=int, nargs=2, default=[20, 23])
    p = sp.add_parser("ppo"); p.add_argument("--seed", type=int); p.add_argument("--steps", type=int, default=250_000); p.add_argument("--val-every", type=int, default=50_000)
    p = sp.add_parser("iql"); p.add_argument("--seed", type=int); p.add_argument("--v-steps", type=int, default=20_000)
    p.add_argument("--offline-steps", type=int, default=60_000); p.add_argument("--online-steps", type=int, default=100_000)
    p.add_argument("--val-every", type=int, default=25_000); p.add_argument("--noise", type=float, default=0.1)
    p = sp.add_parser("queue"); p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); p.add_argument("--workers", type=int, default=10)
    p.add_argument("--only", nargs="*", default=None); p.add_argument("--no-2122", action="store_true")
    p.add_argument("--ppo-steps", type=int, default=250_000); p.add_argument("--online-steps", type=int, default=100_000)
    a = ap.parse_args(argv)
    {"expert": t_expert, "fit": t_fit, "roll": t_roll, "eval": t_eval, "select": t_select, "ppo": t_ppo, "iql": t_iql,
     "queue": t_queue}[a.cmd](a)


if __name__ == "__main__":
    main()
