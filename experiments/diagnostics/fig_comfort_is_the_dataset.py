from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOCAL, GREYC, MARK = "#1f6f8b", "#c9c9c9", "#e58606"
DRAWS = 20000


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
    ap.add_argument("--report",
                    default="results/comfort_is_the_dataset/comfort_is_the_dataset.json")
    ap.add_argument("--schema", default="citylearn_schemas/tx_travis_8b/schema.json")
    ap.add_argument("--out", default="docs/figures/fig_comfort_is_the_dataset.png")
    args = ap.parse_args()

    R = json.load(open(args.report))
    rates = R["per_building_rate"]
    schema = json.load(open(args.schema))
    ours = {n for n, b in schema["buildings"].items() if b.get("include", True)}
    rule, stems = R["published_rule"], R["published_stems"]

    keys = list(rates)
    vals = np.array([rates[k] for k in keys])
    order = np.argsort(vals)
    sv, is_ours = vals[order], np.array([keys[i] in ours for i in order])

    style()
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), layout="constrained")

    ax = axes[0]
    x = np.arange(len(sv))
    ax.bar(x[~is_ours], sv[~is_ours], color=GREYC, width=1.0, linewidth=0)
    ax.bar(x[is_ours], sv[is_ours], color=FOCAL, width=1.0, linewidth=0)
    ax.axhline(rule, color=MARK, lw=1.2, ls="--")
    ax.text(1, rule + 0.022,
            f"STEMS Table I reports {rule:.3f} (rule)\nand {stems:.3f} (their method)",
            fontsize=6.2, color=MARK, va="bottom")
    ax.set_xlabel(f"the {len(sv)} Travis County buildings, sorted")
    ax.set_ylabel("hours outside 20–26 °C, dataset only")
    ax.set_title("No controller ran: this is the weather file", loc="left")
    ax.set_xlim(-1, len(sv))
    ax.text(len(sv) * 0.52, sv.max() * 0.92, f"our {int(is_ours.sum())} buildings",
            fontsize=6.4, color=FOCAL)

    ax = axes[1]
    rng = np.random.default_rng(0)
    k = int(is_ours.sum())
    draws = np.array([vals[rng.choice(len(vals), k, replace=False)].mean()
                      for _ in range(DRAWS)])
    ax.hist(draws, bins=60, color=GREYC, edgecolor="none")
    top = ax.get_ylim()[1]
    for v, c, lbl in [(rule, MARK, f"their rule\n{rule:.3f}"),
                      (R["subset_dataset_rate"], FOCAL,
                       f"our subset\n{R['subset_dataset_rate']:.3f}")]:
        ax.axvline(v, color=c, lw=1.4)
        ax.text(v, top * 0.80, "  " + lbl, color=c, fontsize=6.3, va="top")
    ax.set_xlabel(f"mean discomfort of a random {k}-building subset")
    ax.set_ylabel(f"subsets out of {DRAWS:,}")
    ax.set_title("Both figures are atypical draws from one stock", loc="left")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
