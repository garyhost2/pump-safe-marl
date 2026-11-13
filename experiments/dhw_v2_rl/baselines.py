#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RULES = ("reactive", "price_aware", "thermia", "null")


def _controller(name, B):
    from stems.dhw_v2.controllers import NullController, PriceAwareController, ReactiveController, ThermiaLogic
    if name == "reactive":
        return ReactiveController(B)
    if name == "price_aware":
        return PriceAwareController(B)
    if name == "thermia":
        return ThermiaLogic(B, mode="hourly")

    return NullController(B)


def run_rule(args):
    name, filtered, dataset, package, track_js, out, eval_hours, split_name = args
    from experiments.dhw_v2_rl.train import get_split, track_from
    from stems.dhw_v2_rl.common import dumps, make_env, metrics, run_episode, slice_hours
    tr, ev, split = get_split(dataset, package)
    if split_name == "train":
        ev = tr
    if eval_hours:
        ev = slice_hours(ev, 0, eval_hours)
    track = track_from(track_js)
    env = make_env(ev, track, filtered=filtered, k_correction=0.0)
    ctl = _controller(name, env.B)
    if hasattr(ctl, "reset"):
        ctl.reset()
    rates = {"r_d": [], "p_d": []}

    def act(e, X):
        ctx = e.plant.context()
        rates["r_d"].append(np.asarray(ctx["r_d"], float).copy())
        rates["p_d"].append(np.asarray(ctx["p_d"], float).copy())
        return ctl.act(e)

    S = run_episode(env, act, update=ctl.update)
    S["tank_nodes"] = env.cfg.tank_nodes
    tag = f"{name}__{'filtered' if filtered else 'raw'}"
    res = {"run": tag, "controller": name, "filtered": filtered, "split": split, "track": track.__dict__,
           "metrics": metrics(S), "summary": {k: S[k] for k in S if k != "filter"}}
    p = Path(out) / dataset / ("rules" if split_name == "eval" else "rules_train")
    p.mkdir(parents=True, exist_ok=True)
    (p / f"{tag}.json").write_text(dumps(res))
    if name == "reactive" and filtered:
        np.savez_compressed(p / "rates_reactive_filtered.npz", r_d=np.array(rates["r_d"]), p_d=np.array(rates["p_d"]))
    return tag, res["metrics"]["cost"], S["wall_episode_s"]


def lp_bound(dataset, package, track_js, out, eval_hours=None):
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix
    from experiments.dhw_v2_rl.train import get_split, track_from
    from stems.dhw_v2.tank import rxt12_tank
    from stems.dhw_v2_rl.common import slice_hours
    tr, ev, split = get_split(dataset, package)
    if eval_hours:
        ev = slice_hours(ev, 0, eval_hours)
    track = track_from(track_js)
    p = Path(out) / dataset / "rules"
    z = np.load(p / "rates_reactive_filtered.npz")
    r_d, p_d = z["r_d"], z["p_d"]
    T = r_d.shape[0]
    D = ev.hourly_draw_kwh()[:T]
    spec = rxt12_tank(track.tank_nodes or 2)
    vol = spec.volume_l
    loss_min = spec.ua_w_k / spec.nodes * max(45.0 - spec.t_room_c, 0.0) / 1000.0
    C = vol * 4.186 / 3600.0
    rows = np.arange(T)
    A = coo_matrix((np.concatenate([-np.ones(T), np.ones(T), -np.ones(T - 1)]),
                    (np.concatenate([rows, rows, rows[1:]]),
                     np.concatenate([rows, T + rows, T + rows[:-1]]))), shape=(T, 2 * T)).tocsr()

    def solve(i, E0, Emax):
        c = np.concatenate([ev.price[:T] * p_d[:, i] / r_d[:, i], np.zeros(T)])
        b = -(D[:, i] + loss_min)
        b[0] += E0
        bounds = list(zip(np.zeros(T), r_d[:, i])) + [(0.0, Emax)] * T
        return linprog(c, A_eq=A, b_eq=b, bounds=bounds, method="highs")

    res = []
    for i, h in enumerate(ev.houses):
        tc = float(np.mean(h.t_cold_c[:T]))
        E0 = C * (55.0 - tc)
        r = solve(i, E0, C * (65.0 - tc))
        r55 = solve(i, E0, C * (55.0 - tc))
        if r.status != 0 or r55.status != 0:
            res.append({"house": h.id, "status": f"{r.message} / {r55.message}"})
            continue
        q = r.x[:T]
        res.append({"house": h.id, "cost_dhw_lb": float(r.fun), "cost_dhw_ref55": float(r55.fun),
                    "heat_kwh": float(q.sum()), "elec_kwh": float(np.sum(q * p_d[:, i] / r_d[:, i])),
                    "draw_kwh": float(D[:, i].sum())})
    tot = float(sum(x.get("cost_dhw_lb", np.nan) for x in res))
    outd = {"split": split, "hours": int(T), "per_house": res, "cost_dhw_lb_total": tot,
            "loss_min_kwh_per_h": loss_min,
            "cost_dhw_ref55_total": float(sum(x.get("cost_dhw_ref55", np.nan) for x in res)),
            "ref55_note": "same LP with the store capped at 55 degC (the normal DHW supply): NOT a strict "
                          "bound (the plant can store above 55 degC in disinfection mode), a tighter reference",
            "definition": "(per house, perfect foresight of draws and prices; RELAXATION, hence a lower bound\non the DHW part of the cost up to the brine-temperature caveat below):\n    min sum_t price_t * q_t / COP_t   s.t.   E_{t+1} = E_t + q_t - d_t,  0 <= E_t <= E_max,\n                                             0 <= q_t <= r_t\nwith E the store's heat above the cold-water temperature (E_0 = 55 degC store, E_max = the whole\nstore at the 65 degC disinfection supply), r_t / COP_t the plant's full-speed DHW output and COP\nat the 55 degC supply recorded hour by hour on the Reactive+filter run (the brine temperature\ndepends weakly on the control history, so COP_t is the Reactive run's -- caveat), no standing\nloss beyond the smallest loss of a comfort-feasible store (top node at 45 degC, the other nodes at\nroom temperature), no comfort floor on E, no disinfection energy. Every omission lowers the bound;\nit bounds controllers that keep the comfort floor (hours below the floor make it inapplicable)."}
    (p / "lp_bound.json").write_text(json.dumps(outd, indent=1))
    return tot


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="texas")
    ap.add_argument("--package", default=None)
    ap.add_argument("--track", default="")
    ap.add_argument("--out", default="results_dhw_v2_rl")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval-hours", type=int, default=None)
    ap.add_argument("--split", default="eval", choices=["eval", "train"])
    ap.add_argument("--rules", nargs="+", default=list(RULES))
    a = ap.parse_args(argv)
    jobs = [(n, f, a.dataset, a.package, a.track, a.out, a.eval_hours, a.split) for n in a.rules for f in (True, False)]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for tag, cost, w in pool.imap_unordered(run_rule, jobs):
            print(f"{tag}: cost {cost:.1f} ({w:.0f} s)", flush=True)
    if a.split == "train" or "reactive" not in a.rules:
        return
    t = lp_bound(a.dataset, a.package, a.track, a.out, a.eval_hours)
    print(f"LP DHW-cost lower bound: {t:.2f}", flush=True)


if __name__ == "__main__":
    main()
