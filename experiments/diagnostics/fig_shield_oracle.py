from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY = "#888888"
FOCAL = "#1f6f8b"
MARK = "#e58606"
SOC_SPLIT = 0.101815
DHW_SPLIT = 0.998481
MIN_CELL = 10


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
    ap.add_argument("--dir", default="results/shield_oracle")
    ap.add_argument("--out", default="docs/figures/fig_shield_oracle.png")
    args = ap.parse_args()

    z = np.load(os.path.join(args.dir, "shield_dataset.npz"), allow_pickle=True)
    X, y, names = z["X"], z["y"], list(z["feature_names"])
    report = json.load(open(os.path.join(args.dir, "shield_oracle.json")))
    ib, idh = names.index("electrical_storage_soc"), names.index("dhw_storage_soc")

    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), layout="constrained")

    nb = 40
    xb = np.linspace(0, X[:, ib].max(), nb + 1)
    yb = np.linspace(0, 1.0, nb + 1)
    n_all, _, _ = np.histogram2d(X[:, ib], X[:, idh], bins=[xb, yb])
    n_hit, _, _ = np.histogram2d(X[y == 1, ib], X[y == 1, idh], bins=[xb, yb])
    rate = np.where(n_all >= MIN_CELL, n_hit / np.maximum(n_all, 1), np.nan)

    ax = axes[0]
    im = ax.pcolormesh(xb, yb, rate.T, cmap="magma_r", vmin=0, vmax=1, shading="flat")
    ax.axvline(SOC_SPLIT, color=FOCAL, lw=1.4, ls="--")
    ax.axhline(DHW_SPLIT, color=FOCAL, lw=1.4, ls="--")
    ax.annotate("SOC $\\leq$ 0.1018 recovers\nthe configured floor, 0.10",
                xy=(SOC_SPLIT, 0.30), xytext=(0.22, 0.22), fontsize=6.2, color=FOCAL,
                arrowprops=dict(arrowstyle="-", color=FOCAL, lw=0.8, shrinkA=0, shrinkB=2))
    ax.set_xlabel("battery state of charge")
    ax.set_ylabel("hot-water store state of charge")
    ax.set_title("Where the shield corrects the command", loc="left")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label("intervention rate", fontsize=6)
    cb.ax.tick_params(labelsize=6)

    ax = axes[1]
    lad = report["fidelity_ladder"]
    d = [r["depth"] for r in lad]
    ba = [r["heldout_balanced_accuracy"] for r in lad]
    ax.plot(d, ba, "-o", color=FOCAL, ms=5, lw=1.4, zorder=3)
    k = d.index(2)
    ax.plot([2], [ba[k]], "o", ms=10, mfc="none", mec=MARK, mew=1.6, zorder=4)
    ax.annotate(f"two conditions, {ba[k]:.3f}", xy=(2, ba[k]), xytext=(2.35, 0.80),
                fontsize=6.4, color=MARK,
                arrowprops=dict(arrowstyle="-", color=MARK, lw=0.8, shrinkA=0, shrinkB=5))
    ax.axhline(0.5, color=GREY, lw=0.9, ls=":")
    ax.text(max(d), 0.515, "chance", fontsize=6, color=GREY, ha="right")
    ax.set_xticks(d)
    ax.set_xlabel("conditions allowed (tree depth)")
    ax.set_ylabel("balanced accuracy, held-out half-year")
    ax.set_title("Depth 3 buys nothing over depth 2", loc="left")
    ax.set_ylim(0.45, 1.04)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
