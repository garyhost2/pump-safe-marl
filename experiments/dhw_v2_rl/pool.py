#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="texas")
    ap.add_argument("--package", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--episodes", type=int, nargs="+", default=[40], help="one value, or one per learner")
    ap.add_argument("--learners", nargs="+", default=["ppo", "sac"])
    ap.add_argument("--arms", nargs="+", default=["a", "b", "c"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="results_dhw_v2_rl")
    ap.add_argument("--track", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--extra", default="", help="extra args for train.py (one string)")
    a = ap.parse_args(argv)
    eps = dict(zip(a.learners, a.episodes if len(a.episodes) == len(a.learners) else a.episodes * len(a.learners)))
    jobs = []
    for s in a.seeds:
        for L in a.learners:
            for arm in a.arms:
                jobs.append((L, arm, s))
    outd = Path(a.out) / a.dataset
    outd.mkdir(parents=True, exist_ok=True)
    logs = outd / "_logs"
    logs.mkdir(exist_ok=True)
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    running, done, failed = {}, [], []
    pending = [j for j in jobs if not (outd / f"{j[0]}_{j[1]}_seed{j[2]}{a.tag}" / "done.json").exists()]
    skipped = len(jobs) - len(pending)
    while pending or running:
        while pending and len(running) < a.workers:
            L, arm, s = pending.pop(0)
            name = f"{L}_{arm}_seed{s}{a.tag}"
            cmd = [py, "-B", str(REPO / "experiments/dhw_v2_rl/train.py"), "--dataset", a.dataset, "--learner", L,
                   "--arm", arm, "--seed", str(s), "--episodes", str(eps[L]), "--out", a.out, "--tag", a.tag]
            if a.package:
                cmd += ["--package", a.package]
            if a.track:
                cmd += ["--track", a.track]
            if a.extra:
                cmd += a.extra.split()
            f = open(logs / f"{name}.log", "w")
            running[name] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)), f, time.time())
        for name in list(running):
            p, f, t0 = running[name]
            rc = p.poll()
            if rc is not None:
                f.close()
                (done if rc == 0 else failed).append({"run": name, "rc": rc, "wall_s": time.time() - t0})
                del running[name]
        (outd / f"_pool_heartbeat{a.tag}.json").write_text(json.dumps({
            "time": time.strftime("%Y-%m-%d %H:%M:%S"), "total": len(jobs), "skipped_done": skipped,
            "running": sorted(running), "pending": len(pending), "done": done, "failed": failed,
            "finished": not pending and not running}, indent=1))
        time.sleep(10)
    print(json.dumps({"done": len(done), "failed": failed}))


if __name__ == "__main__":
    main()
