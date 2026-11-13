#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent / "smoke_results"

RUNS = {
    "reactive__off": ("reactive", None, None, {}),
    "reactive__oracle": ("reactive", None, "oracle", {}),
    "thermia_native__off": ("thermia", "native", None, {}),
    "thermia_hourly__oracle": ("thermia", "hourly", "oracle", {}),
    "thermia_hourly__off": ("thermia", "hourly", None, {}),
    "null__oracle": ("null", None, "oracle", {}),
    "null__oracle_onehour": ("null", None, "oracle", {"reserve_h": 0}),
    "reactive__identified": ("reactive", None, "identified", {}),
    "thermia_hourly__identified": ("thermia", "hourly", "identified", {}),
    "reactive__oracle_a01": ("reactive", None, "oracle", {"alpha": 0.01}),
    "reactive__oracle_a001": ("reactive", None, "oracle", {"alpha": 0.001, "eta": 0.5}),
    "null__oracle_a01": ("null", None, "oracle", {"alpha": 0.01}),
    "null__oracle_a001": ("null", None, "oracle", {"alpha": 0.001, "eta": 0.5}),
    "reactive__oracle_cap3": ("reactive", None, "oracle", {"cap_house": 3.0, "cap_district": 5.0}),
    "reactive__identified_cap3": ("reactive", None, "identified", {"cap_house": 3.0, "cap_district": 5.0}),
    "reactive__off_N2": ("reactive", None, None, {"nodes": 2}),
    "reactive__oracle_N2": ("reactive", None, "oracle", {"nodes": 2}),
    "thermia_native__off_N2": ("thermia", "native", None, {"nodes": 2}),
    "thermia_hourly__oracle_N2": ("thermia", "hourly", "oracle", {"nodes": 2}),
    "reactive__off_N20": ("reactive", None, None, {"nodes": 20}),
    "thermia_native__off_N20": ("thermia", "native", None, {"nodes": 20}),
    "null__oracle_N20": ("null", None, "oracle", {"nodes": 20}),
    "null__oracle_N2": ("null", None, "oracle", {"nodes": 2}),
    "thermia_native__off_fullspeed": ("thermia", "native", None, {"dhw_speed": 1.0}),
    "reactive__off_fullspeed": ("reactive", None, None, {"dhw_speed": 1.0}),
    "thermia_native__off_start45": ("thermia", "native", None, {"start": 45.0}),
    "thermia_native__off_start46": ("thermia", "native", None, {"start": 46.0}),
    "thermia_native__off_start48": ("thermia", "native", None, {"start": 48.0}),
    "thermia_native__off_stop52": ("thermia", "native", None, {"stop": 52.0}),
    "thermia_native__off_stop55": ("thermia", "native", None, {"stop": 55.0}),
    "thermia_native__off_sensor050": ("thermia", "native", None, {"sensor": 0.5}),
    "thermia_native__off_sensor075": ("thermia", "native", None, {"sensor": 0.75}),
    "null__oracle_trig_jit": ("null", None, "oracle", {"trigger": "jit"}),
    "null__oracle_trig_recourse": ("null", None, "oracle", {"trigger": "recourse"}),
    "null__oracle_trig_quantile": ("null", None, "oracle", {"trigger": "quantile"}),
    "null__oracle_trig_robust_b010": ("null", None, "oracle", {"beta": 0.10}),
    "null__oracle_trig_robust_R0": ("null", None, "oracle", {"retry": 0}),
    "reactive__oracle_trig_timer": ("reactive", None, "oracle", {"trigger": "none"}),
    "thermia_hourly__oracle_trig_thermia": ("thermia", "hourly", "oracle", {"trigger": "none"}),
    "null__stress_onehour_4w": ("null", None, "oracle", {"draw_scale": 2.0, "cap_house": 0.8, "reserve_h": 0, "hours": 672}),
    "null__stress_lookahead1_4w": ("null", None, "oracle", {"draw_scale": 2.0, "cap_house": 0.8, "reserve_h": 0, "lookahead": True, "hours": 672}),
    "null__stress_reserve24_4w": ("null", None, "oracle", {"draw_scale": 2.0, "cap_house": 0.8, "hours": 672}),
    "reactive__off_v1plant": ("reactive", None, None, {"heatpump": "hplib", "brine": "correlation", "nodes": 2, "dhw_speed": 1.0}),
    "reactive__off_dt1_2w": ("reactive", None, None, {"dt": 1, "hours": 336}),
    "reactive__off_dt5_2w": ("reactive", None, None, {"dt": 5, "hours": 336}),
}
CORE = ["reactive__off", "reactive__oracle", "thermia_native__off", "thermia_hourly__oracle"]


def _env(data, ov, filt, identified=None, hours=None):
    from stems.dhw_v2.calibration import DHW_SPEED_FRAC
    from stems.dhw_v2.env import DHWDistrictEnv, EnvConfig
    from stems.dhw_v2.filter import FilterConfig
    from stems.dhw_v2.plant import PlantConfig
    pc = PlantConfig(dt_min=int(ov.get("dt", 5)), heatpump=ov.get("heatpump", "rxt12"),
                     brine=ov.get("brine", "borehole"), dhw_speed_frac=float(ov.get("dhw_speed", DHW_SPEED_FRAC)))
    fc = None
    if filt is not None:
        fc = FilterConfig(alpha=float(ov.get("alpha", 0.05)), eta_kwh=float(ov.get("eta", 0.05)),
                          beta=float(ov.get("beta", 0.02)), retry_h=int(ov.get("retry", 2)),
                          reserve_h=int(ov.get("reserve_h", 24)),
                          lookahead=bool(ov.get("lookahead", False)), trigger=ov.get("trigger", "robust"),
                          cap_house_kw=ov.get("cap_house"), cap_district_kw=ov.get("cap_district"))
    cfg = EnvConfig(plant=pc, tank_nodes=int(ov.get("nodes", 10)), sensor_height=float(ov.get("sensor", 0.6)),
                    filter=fc, filter_model=filt or "oracle", identified=identified,
                    hours=hours if hours is not None else ov.get("hours"))
    return DHWDistrictEnv(data, cfg)


def _controller(name, mode, B, ov=None):
    from stems.dhw_v2.controllers import NullController, ReactiveController, ThermiaLogic
    ov = ov or {}
    if name == "reactive":
        return ReactiveController(B)
    if name == "null":
        return NullController(B)
    return ThermiaLogic(B, mode=mode, start_c=float(ov.get("start", 47.0)), stop_c=float(ov.get("stop", 53.0)))


def _data(ov, houses=(0, 1)):
    from stems.dhw_v2.data import load_texas
    data = load_texas().subset(list(ov.get("houses", houses)))
    sc = float(ov.get("draw_scale", 1.0))
    if sc != 1.0:
        for h in data.houses:
            h.draw_kwh = h.draw_kwh * sc
    return data


def commission_and_identify(data, ov, hours=672, seed=0):
    from stems.dhw_v2.identify import LogRecorder, identify
    env = _env(data, ov, None, hours=hours)
    env.reset()
    ctl = _controller("thermia", "native", env.B, ov)
    log = LogRecorder(2, seed=seed)
    vol = np.array([t.volume_l for t in env._tanks])
    while True:
        T0 = env.plant.T.copy()
        ctx = env.plant.context()
        obs, r, term, _, info = env.step_command(ctl.act(env))
        rec = info["record"]
        ctl.update(rec)
        t_s = np.where(rec["dis_supply_used"], env.cfg.plant.t_supply_dis_c, env.cfg.plant.t_supply_dhw_c)
        log.record(T0, env.plant.T, ctx, rec, t_s, vol)
        if term:
            break
    idf = identify(log)
    idf["nodes"] = 2
    idf["commissioning_hours"] = hours
    return idf


def run(name, houses=(0, 1)):
    ctl_name, mode, filt, ov = RUNS[name]
    data = _data(ov, houses)
    t0 = time.time()
    idf = commission_and_identify(data, ov) if filt == "identified" else None
    t_id = time.time() - t0
    env = _env(data, ov, filt, identified=idf)
    env.reset()
    ctl = _controller(ctl_name, mode, env.B, ov)
    t1 = time.time()
    while True:
        obs, r, term, _, info = env.step_command(ctl.act(env))
        ctl.update(info["record"])
        if term:
            break
    S = env.summary()
    S.update(run=name, controller=ctl_name, mode=mode, filter=filt, overrides=ov,
             wall_run_s=time.time() - t1, wall_identify_s=t_id, identified=idf,
             dt_min=env.cfg.plant.dt_min, tank_nodes=env.cfg.tank_nodes,
             data=data.source, plant=env.plant.describe())
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(S, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return name, S["wall_run_s"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=list(RUNS))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for name, w in pool.imap_unordered(run, args.runs):
            print(f"{name}: {w:.1f} s", flush=True)


if __name__ == "__main__":
    main()
