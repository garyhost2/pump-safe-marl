#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

ORACLE = "mpc-oracle"
RULE = "rbc+calibrated"
FOCAL_BLUE = "#1f5fa8"
GREY = "#8a8f98"
RULE_GREY = "#4a5058"
META = "#6f757e"

NICE = {"rl+calibrated+pen": "projection + intervention penalty",
        "rl-res+calibrated": "residual over rule",
        "rl+calibrated": "projection only",
        "rl+calibrated+meanpool": "projection, mean pooling",
        "maddpg": "MADDPG",
        "sac": "SAC",
        RULE: "rule + barrier (not learned)",
        ORACLE: "MPC oracle"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/e8_arch")
    ap.add_argument("--out", default="docs/figures/fig_architecture_invariance.png")
    a = ap.parse_args()

    acc = collections.defaultdict(list)
    for p in glob.glob(os.path.join(a.dir, "*", "*.json")):
        r = json.load(open(p, encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        arm = r["meta"]["arm"]
        arm = arm["name"] if isinstance(arm, dict) else arm
        acc[arm].append(r["eval"])

    def mean(arm, key):
        return float(np.mean([x[key] for x in acc[arm]
                              if isinstance(x.get(key), (int, float))]))

    oracle_cost = mean(ORACLE, "cost")
    seen, arms = set(), []
    for arm in sorted(acc, key=lambda k: (mean(k, "cost"), len(k), k)):
        sig = (round(mean(arm, "cost"), 6), round(mean(arm, "discomfort_rate"), 8))
        if sig in seen and arm != ORACLE:
            continue
        seen.add(sig)
        arms.append(arm)

    excess = [100 * (mean(x, "cost") - oracle_cost) / oracle_cost for x in arms]
    soc = [mean(x, "soc_violation_rate") for x in arms]
    y = np.arange(len(arms))[::-1]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 6, "ytick.labelsize": 7,
                         "font.family": "DejaVu Sans"})
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(7.2, 3.2),
                                   gridspec_kw={"width_ratios": [2.4, 1.0],
                                                "wspace": 0.08})

    colours = [RULE_GREY if x == RULE else GREY if x == ORACLE else FOCAL_BLUE
               for x in arms]
    for yi, xi, c in zip(y, excess, colours):
        axl.plot([0, xi], [yi, yi], color=c, lw=1.0, alpha=0.5, zorder=1)
    axl.scatter(excess, y, s=34, color=colours, zorder=3)
    for yi, xi in zip(y, excess):
        axl.annotate(f"{xi:+.1f}%", (xi, yi), xytext=(6, 0), textcoords="offset points",
                     va="center", fontsize=6, color=META)
    axl.set_yticks(y)
    axl.set_yticklabels([NICE.get(x, x) for x in arms])
    axl.set_xlabel("operating cost above the MPC oracle  (lower = better)")
    axl.set_xlim(-4, max(excess) * 1.22)
    axl.axvline(0, color=GREY, lw=0.8, ls=(0, (4, 2)))
    axl.set_title("What the learner controls", loc="left")

    axr.scatter(soc, y, s=34, color=colours, zorder=3)
    axr.set_yticks(y)
    axr.set_yticklabels([])
    axr.set_xlim(-1, 1)
    axr.set_xticks([0])
    axr.set_xticklabels(["0"])
    axr.set_xlabel("state-of-charge\nviolation rate")
    axr.set_title("What the filter controls", loc="left")

    for ax in (axl, axr):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=2.5, width=0.7)
    axr.spines["left"].set_visible(False)
    axr.tick_params(axis="y", length=0)

    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print(f"wrote {a.out}")
    for arm, e, s in zip(arms, excess, soc):
        print(f"  {arm:<26}{e:>+8.1f}%   soc {s:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())