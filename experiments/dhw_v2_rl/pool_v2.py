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
    ap.add_argument("jobs")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    jobs = json.loads(Path(a.jobs).read_text())
    logs = Path(a.jobs).with_suffix("")
    logs = logs.parent / (logs.name + "_logs")
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    pending, running, done, failed = list(jobs), {}, [], []
    hb = Path(str(a.jobs) + ".heartbeat.json")
    while pending or running:
        while pending and len(running) < a.workers:
            j = pending.pop(0)
            f = open(logs / f"{j['name']}.log", "w")
            cmd = [sys.executable, "-B", str(REPO / j.get("script", "experiments/dhw_v2_rl/train_v2.py"))] + j["args"]
            running[j["name"]] = (subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)), f, time.time())
        for n in list(running):
            p, f, t0 = running[n]
            if p.poll() is not None:
                f.close()
                (done if p.returncode == 0 else failed).append({"name": n, "rc": p.returncode, "wall_s": time.time() - t0})
                del running[n]
        hb.write_text(json.dumps({"time": time.strftime("%H:%M:%S"), "running": sorted(running), "pending": len(pending),
                                  "done": len(done), "failed": failed, "finished": not pending and not running}, indent=1))
        time.sleep(15)
    print(json.dumps({"done": len(done), "failed": failed}))


if __name__ == "__main__":
    main()
