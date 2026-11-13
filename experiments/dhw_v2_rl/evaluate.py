#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def one(args):
    run, out, track_js, package = args
    from experiments.dhw_v2_rl.train import evaluate_policy, get_split, track_from
    from stems.dhw_v2_rl.common import dumps
    from stems.dhw_v2_rl.learners import load_policy
    run, out = Path(run), Path(out)
    meta = json.loads((run / "meta.json").read_text())
    dst = out / run.name
    if (dst / "done.json").exists():
        return run.name, "skip"
    dst.mkdir(parents=True, exist_ok=True)
    tr, ev, _ = get_split(meta["dataset"], package)
    base = dict(meta["track"])
    ov = json.loads(track_js)
    for k, v in ov.items():
        base[k] = {**base.get(k, {}), **v} if isinstance(v, dict) else v
    track = track_from(json.dumps(base))
    pol = load_policy(run / "policy.pt")
    ef = evaluate_policy(pol, ev, track, True)
    meta = dict(meta, eval_track=base, eval_override=ov)
    (dst / "meta.json").write_text(dumps(meta))
    for f in ("curve.csv", "eval_raw.json"):
        shutil.copy(run / f, dst / f)
    (dst / "eval_filtered.json").write_text(dumps(ef))
    (dst / "done.json").write_text(json.dumps({"filtered": ef["metrics"], "override": ov}, indent=1))
    return run.name, ef["metrics"]["cost"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--track", required=True)
    ap.add_argument("--package", default=None)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args(argv)
    runs = sorted(p for p in Path(a.runs).glob("*_seed*") if (p / "done.json").exists())
    jobs = [(str(r), a.out, a.track, a.package) for r in runs]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, c in pool.imap_unordered(one, jobs):
            print(name, c, flush=True)


if __name__ == "__main__":
    main()
