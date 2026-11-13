#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

LADDER = [("no filter", "rl"),
          ("uniform rate", "rl+basic"),
          ("linear", "rl+linear"),
          ("exact inverse", "rl+calibrated")]
FOCAL = "exact inverse"
KPI = "avoidable_violation_rate"

FOCAL_BLUE = "#1f5fa8"
GREY = "#8a8f98"
LIGHT = "#b9bec6"
MID = "#6f757e"
META = "#6f757e"


def load(e2_dir: str) -> dict:
    acc: dict = {}
    for p in sorted(glob.glob(os.path.join(e2_dir, "*", "*.json"))):
        r = json.load(open(p, encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        meta = r["meta"]
        arm = meta["arm"]
        arm = arm["name"] if isinstance(arm, dict) else arm
        v = r.get("eval", {}).get(KPI)
        if isinstance(v, (int, float)):
            acc.setdefault((meta["scenario"]["subset_seed"], arm), []).append(float(v))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", default="results/e2_subsets")
    ap.add_argument("--out", default="docs/figures/fig_calibration_ladder.png")
    a = ap.parse_args()

    means = load(a.e2)
    subsets = sorted({s for s, _ in means})
    series = {lab: np.array([means[(s, arm)] for s in subsets]) for lab, arm in LADDER}
    seeds = len({0, 1, 2, 3, 4})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogFormatterMathtext

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "font.family": "DejaVu Sans"})

    x = np.arange(len(subsets))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    style = {"no filter": dict(color=GREY, lw=1.3, ms=4, ls=(0, (5, 1.5))),
             "uniform rate": dict(color=LIGHT, lw=1.3, ms=4, ls="-"),
             "linear": dict(color=MID, lw=1.3, ms=4, ls="-"),
             FOCAL: dict(color=FOCAL_BLUE, lw=2.4, ms=6.5, ls="-")}
    for lab, _ in LADDER:
        ax.plot(x, series[lab], marker="o", zorder=3 if lab == FOCAL else 2,
                **style[lab])

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(LogFormatterMathtext())
    ax.set_xticks(x)
    ax.set_xticklabels([f"subset {s}" for s in subsets])
    ax.set_ylabel("avoidable violation rate  (lower = safer)")
    ax.set_xlim(-0.35, len(subsets) - 1 + 1.5)
    ax.set_title("Safety is set by the plant model inside the filter, not by the learner",
                 loc="left")

    ends = sorted(((series[lab][-1], lab) for lab, _ in LADDER), reverse=True)
    offsets, prev = {}, None
    for val, lab in ends:
        off = 1.0
        if prev is not None and prev / max(val, 1e-18) < 2.0:
            off = 0.62
        offsets[lab] = off
        prev = val * off
    for lab, _ in LADDER:
        ax.text(x[-1] + 0.12, series[lab][-1] * offsets[lab], lab,
                color=style[lab]["color"], va="center", ha="left", fontsize=7)

    n_low = int(sum(series[FOCAL][i] <= min(series[l][i] for l, _ in LADDER)
                    for i in range(len(subsets))))
    note = (f"{seeds} seeds per point; replication unit is the building subset. "
            f"Exact inverse is lowest in {n_low} of {len(subsets)} subsets.")

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=2.5, width=0.7)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.012, 0.015, note, fontsize=6, color=META, va="bottom", ha="left")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print(f"wrote {a.out}")
    for lab, _ in LADDER:
        print(f"  {lab:14s} " + " ".join(f"{v:.3e}" for v in series[lab]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())