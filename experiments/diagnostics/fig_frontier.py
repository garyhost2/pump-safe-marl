from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.diagnostics.stems_table_compare import PUBLISHED

GREY, FOCAL, WARM, MARK = "#888888", "#1f6f8b", "#b07aa1", "#e58606"


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
    ap.add_argument("--csv", default="results/frontier.csv")
    ap.add_argument("--episodes", type=int, default=50,
                    help="which budget to draw as the frontier")
    ap.add_argument("--subset", default="ref", help="building subset to plot")
    ap.add_argument("--out", default="docs/figures/fig_frontier.png")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(args.csv))
            if r["xi"] in ("0.0", "0") and r.get("subset", "ref") == args.subset]
    ours = sorted((r for r in rows
                   if int(r["episodes"]) == args.episodes and r["arm"] == "rl+calibrated"),
                  key=lambda r: float(r["lambda_indoor"]))
    warm = sorted((r for r in rows
                   if int(r["episodes"]) == args.episodes and r["arm"] == "rl+calibrated+bc"),
                  key=lambda r: float(r["lambda_indoor"]))
    if not ours:
        raise SystemExit(f"no rl+calibrated rows at {args.episodes} episodes in "
                         f"{args.csv}; the sweep has not reached that budget yet")

    style()
    fig, ax = plt.subplots(figsize=(5.2, 3.6), layout="constrained")

    for name, p in PUBLISHED.items():
        focal = name == "STEMS"
        ax.plot(p["discomfort"], p["cost"], "s" if focal else "o",
                ms=6 if focal else 4,
                color=MARK if focal else GREY, zorder=3 if focal else 2,
                markeredgecolor="white", markeredgewidth=0.6)
        ax.annotate(name, (p["discomfort"], p["cost"]), textcoords="offset points",
                    xytext=(6, 4) if not focal else (6, -2), fontsize=5.6,
                    color=MARK if focal else GREY,
                    bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none",
                              alpha=0.75))

    def draw(points, colour, label, marker="o", anchor=-1, offset=(6, 6), connect=True):
        if not points:
            return
        x = [float(r["discomfort_proportion"]) for r in points]
        y = [float(r["cost"]) for r in points]
        ax.plot(x, y, ("-" if connect else "") + marker, color=colour,
                lw=1.4 if connect else 0, ms=5, zorder=4,
                markeredgecolor="white", markeredgewidth=0.6, linestyle="-" if connect else "none")
        ends = [(0, (6, 7)), (len(points) - 1, (6, -10))] if len(points) > 1 else [(0, (6, 7))]
        for k, off in ends:
            r, xi_, yi_ = points[k], x[k], y[k]
            ax.annotate(f"$\\lambda$={float(r['lambda_indoor']):g}", (xi_, yi_),
                        textcoords="offset points", xytext=off, fontsize=5.6,
                        color=colour, ha="left",
                        bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none",
                                  alpha=0.8))
        ax.annotate(label, (x[anchor], y[anchor]), textcoords="offset points",
                    xytext=offset, fontsize=6.4, color=colour, weight="bold")

    draw(ours, FOCAL, "ours", anchor=-1, offset=(6, 6))
    draw(warm, WARM, "ours, warm-started", marker="^", anchor=0, offset=(10, -4),
         connect=False)

    ax.axhline(1.0, color=GREY, lw=0.8, ls=":")
    ax.text(ax.get_xlim()[1], 1.005, "rule-based cost ", fontsize=6, color=GREY,
            ha="right", va="bottom")
    ax.set_xlabel("discomfort proportion (hours outside 20–26 °C)")
    ax.set_ylabel("cost, normalised to the rule")
    ax.set_title(f"One point each, or the curve they sit on ({args.episodes} episodes)",
                 loc="left")
    ax.text(1.0, -0.22, "lower and left = better", transform=ax.transAxes,
            fontsize=6, ha="right", color=GREY)
    ax.text(0.0, -0.22, "published rows state no comfort weight: compare along x, not y alone",
            transform=ax.transAxes, fontsize=6, ha="left", color=GREY)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}  ({len(ours)} frontier points, "
          f"{len(warm)} warm-started, {len(PUBLISHED)} published)")


if __name__ == "__main__":
    main()
