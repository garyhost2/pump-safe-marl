from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY = "#888888"
ORDER = ["lagrangian", "projection", "exact_calibration"]
LABEL = {"lagrangian": "Lagrangian\nmultipliers",
         "projection": "Barrier\nprojection",
         "exact_calibration": "Exact plant\ninverse"}
HUE = {"lagrangian": "#b07aa1", "projection": "#1f6f8b", "exact_calibration": "#e58606"}

PANELS = [
    ("safety_violation_rate",
     "Safety is split between two mechanisms;\nthe plant inverse earns its share only in company",
     "Shapley value, violation rate"),
    ("cost",
     "The mechanism that buys the most safety\nis also the one that costs money",
     "Shapley value, cost (currency units)"),
]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.0,
        "axes.titlesize": 8.0, "axes.labelsize": 8.0, "axes.titlelocation": "left",
        "xtick.labelsize": 6.0, "ytick.labelsize": 6.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "figure.dpi": 200.0,
    })


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/shapley_mechanism.csv")
    ap.add_argument("--out", default="docs/figures/fig_shapley_mechanism.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv)))
    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    for (metric, title, xlabel), ax in zip(PANELS, axes):
        rs = {r["mechanism"]: r for r in rows if r["metric"] == metric}
        y = np.arange(len(ORDER))[::-1]
        for yi, mech in zip(y, ORDER):
            r = rs[mech]
            m, lo, hi = float(r["shapley_mean"]), float(r["ci_lo"]), float(r["ci_hi"])
            ax.plot([lo, hi], [yi, yi], color=HUE[mech], lw=2.0,
                    solid_capstyle="round", zorder=2)
            ax.plot([m], [yi], "o", color=HUE[mech], ms=6, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)
        ax.axvline(0, color=GREY, lw=0.9, zorder=1)
        ax.set_yticks(y)
        ax.set_yticklabels([LABEL[m] for m in ORDER])
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left")
        ax.margins(y=0.25)
        ax.text(0.5, -0.33, "lower = better", transform=ax.transAxes, fontsize=6,
                ha="center", color=GREY)

    axes[0].annotate("zero effect alone;\n$-0.202$ added to the projection",
                     xy=(-0.034, 0), xytext=(-0.33, 0.55), fontsize=6.3, color=HUE["exact_calibration"],
                     arrowprops=dict(arrowstyle="-", color=HUE["exact_calibration"],
                                     lw=0.8, shrinkA=0, shrinkB=3))
    axes[1].annotate("costs $+549$", xy=(549, 2), xytext=(120, 1.5), fontsize=6.3,
                     color=HUE["lagrangian"],
                     arrowprops=dict(arrowstyle="-", color=HUE["lagrangian"],
                                     lw=0.8, shrinkA=0, shrinkB=3))

    fig.tight_layout(w_pad=3.0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
