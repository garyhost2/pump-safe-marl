#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from before_after import VARIANTS
from run_matrix import PERIOD, T_COLD, T_RATED, load
from stems.dhw_plant import DHWPlant, TankSpec
from stems.dhw_stratified import ForecastBundle
from stems.heatpump_hplib import HplibUnit, brine_temperature_c, calibra_eco_8


def trace(kw, data, b, t0, t1):
    tank = TankSpec(capacity_kwh=float(data["capacity"][b]), t_cold_c=T_COLD,
                    t_rated_c=T_RATED, loss_coefficient=float(data["loss"][b]), nodes=2,
                    top_fraction=0.5, node_conduction=0.0, **kw)
    p = DHWPlant(tank, HplibUnit(calibra_eco_8()), initial_soc=0.9)
    fb = ForecastBundle(data["price_dynamic"], data["t_out"], brine_temperature_c,
                        weather=False, seed=None)
    strat = kw["charging"] == "stratified"
    t_dis = 60.0 if kw["t_max_c"] == 60.0 else 65.0
    due = False
    rows = []
    for t in range(t1):
        if t % PERIOD == 2:
            due = True
        d = float(data["draw"][t, b])
        src = fb.source_now(t)
        target = t_dis if due else 55.0
        if strat:
            s = p.step(d, p.charge_limit_kwh(src, draw_kwh=d, t_supply_c=target), src,
                       t_supply_c=target)
        else:
            E = p._post_draw_energy(d)
            s = p.step(d, max(p.tank.soc_of_temperature(target) * p._caps[0] - E[0], 0.0), src)
        if due and s.top_temperature_c >= 60.0 - 1e-9:
            due = False
        if t >= t0:
            rows.append((t, d, s.node_temperature_c[0], s.node_temperature_c[1],
                         s.charge_thermal_kwh))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", type=int, default=0)
    args = ap.parse_args()
    data = load()
    t0, t1 = PERIOD, 3 * PERIOD
    old = trace(VARIANTS["A committed"], data, args.house, t0, t1)
    new = trace(VARIANTS["D stratified charging"], data, args.house, t0, t1)
    out = REPO / "results" / "dhw_paper"
    np.savetxt(out / f"trace_house{args.house}.csv",
               np.column_stack([old, new[:, 2:]]), delimiter=",",
               header="t,draw_kwh,committed_top_c,committed_bottom_c,committed_heat_kwh,"
                      "stratified_top_c,stratified_bottom_c,stratified_heat_kwh",
               comments="")
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 4.2), sharex=True,
                             gridspec_kw={"hspace": 0.45, "height_ratios": [2, 2, 1]})
    days = (old[:, 0] - t0) / 24.0
    for ax, arr, title in ((axes[0], old, "committed plant: the bottom node is never heated (top held at 60 °C: the cycle never ends)"),
                           (axes[1], new, "stratified charging: the charge front reaches "
                                          "the bottom")):
        ax.axhspan(45, 55, color="#0072B2", alpha=0.08, lw=0)
        ax.axhline(60, color="#D55E00", lw=0.8, ls="--")
        ax.plot(days, arr[:, 2], color="#0072B2", lw=1.0, label="top node")
        ax.plot(days, arr[:, 3], color="#E69F00", lw=1.0, label="bottom node")
        ax.set_ylim(5, 70)
        ax.set_ylabel("°C", fontsize=7)
        ax.set_title(title, loc="left", fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6)
    axes[0].text(0.3, 62.0, "top node", fontsize=6, color="#0072B2", ha="left", va="bottom")
    axes[0].text(0.3, 37.5, "bottom node", fontsize=6, color="#E69F00", ha="left", va="bottom")
    axes[1].text(2.0, 52.5, "top and bottom node (55 °C supply)", fontsize=6, color="#555555",
                 ha="left", va="top")
    for ax in axes[:2]:
        ax.text(days[-1], 60.8, "60 °C", fontsize=6, color="#D55E00", ha="right")
        ax.text(days[-1], 46, "comfort band 45–55 °C", fontsize=6, color="#0072B2", ha="right")
    axes[2].bar(days, old[:, 1], width=1 / 24, color="#555555")
    axes[2].set_ylabel("draw\n(kWh/h)", fontsize=7)
    axes[2].set_xlabel(f"days from hour {t0} (house {args.house}; weekly cycle due at "
                       f"hour 2 of each week)", fontsize=7)
    for sp in ("top", "right"):
        axes[2].spines[sp].set_visible(False)
    axes[2].tick_params(labelsize=6)
    from figcheck import check_figure, record
    record(check_figure(fig, "fig_trace.png"), out / "figure_checks.json")
    fig.savefig(out / "fig_trace.png", dpi=300, bbox_inches="tight")
    print("written", out / "fig_trace.png")


if __name__ == "__main__":
    main()
