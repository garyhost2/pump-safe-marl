#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RES = REPO / "results" / "dhw_paper"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from figcheck import check_figure, record

FIGCHECK = RES / "figure_checks.json"
KEYS = ["plant", "controller", "trigger", "filter", "barrier_draw", "price", "weather"]
METRICS = ["windows_missed", "completion_rate", "comfort_below_steps",
           "comfort_below_degree_h", "comfort_above_steps", "comfort_above_degree_h",
           "comfort_above_steps_outside_D", "top_below_45_after_draw_steps", "unmet_kwh",
           "electricity_kwh", "cost_usd", "starts", "gaps_over_168h",
           "max_gap_between_hold_steps_h", "refusals", "empty_intersections",
           "modified_steps", "steps_in_D", "electricity_in_D_kwh", "seasonal_cop",
           "model_vs_plant_max_abs_kwh", "band_violations_unreported", "hold_met_steps",
           "mean_window_bottom_peak_c"]
TRIG_LABEL = {"none": "no disinfection", "timer": "weekly timer", "jit": "just-in-time",
              "recourse": "recourse (committed)", "recourse_q": "recourse, quantile plan"}
CTL_LABEL = {"reactive": "Reactive", "price_aware": "Price-aware"}
OKABE = {"none": "#999999", "timer": "#E69F00", "jit": "#D55E00", "recourse": "#56B4E9",
         "recourse_q": "#0072B2"}


def load_jsonl(name):
    p = RES / f"{name}.jsonl"
    if not p.exists():
        return pd.DataFrame()
    rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["filter"] = df["filter"].astype(bool)
    g = df.groupby(KEYS, dropna=False)
    out = g.size().rename("n_runs").to_frame()
    for m in METRICS:
        if m not in df:
            continue
        col = pd.to_numeric(df[m], errors="coerce")
        df[m] = col
        out[m] = g[m].mean()
        out[m + "_min"] = g[m].min()
        out[m + "_max"] = g[m].max()
    out = out.reset_index()
    base = out[(out.trigger == "none")].set_index(["plant", "controller", "price", "weather",
                                                   "filter"])
    inc, per = [], []
    for _, r in out.iterrows():
        k = (r.plant, r.controller, r.price, r.weather, bool(r["filter"]))
        if r.trigger == "none" or k not in base.index:
            inc.append(np.nan)
            per.append(np.nan)
            continue
        e0 = base.loc[k, "electricity_kwh"]
        e0 = float(e0.iloc[0]) if hasattr(e0, "iloc") else float(e0)
        done = 416.0 - r.windows_missed
        inc.append(r.electricity_kwh - e0)
        per.append((r.electricity_kwh - e0) / done if done > 0 else np.nan)
    out["electricity_increment_kwh"] = inc
    out["electricity_per_completed_disinfection_kwh"] = per
    return out


def fmt(r, m, nd=0):
    v, lo, hi = r[m], r.get(m + "_min"), r.get(m + "_max")
    if pd.isna(v):
        return "-"
    s = f"{v:.{nd}f}"
    if r.n_runs > 1 and lo is not None and not pd.isna(lo) and (hi - lo) > 10 ** (-nd) / 2:
        s += f" [{lo:.{nd}f}, {hi:.{nd}f}]"
    return s


def headline(agg, price, weather="off", barrier_draw="known", plant="calibra8", extra=False):
    a = agg[(agg.price == price) & (agg.weather == weather) & (agg.plant == plant)
            & ((agg.barrier_draw == barrier_draw) | (agg.trigger == "none"))]
    order = {"none": 0, "timer": 1, "jit": 2, "recourse": 3, "recourse_q": 4}
    a = a.assign(o=a.trigger.map(order)).sort_values(["controller", "o", "filter"])
    lines = ["| controller | disinfection trigger | filter | missed windows (of 416) | "
             "completion | top < 45 °C (building-h) | below 45 °C (K·h) | "
             "top > 55 °C (building-h) | above 55 °C (K·h) | unmet (kWh) | electricity "
             "(kWh) | cost (USD) | starts | kWh per completed disinfection |" +
             (" steps reported infeasible | unreported band violations |" if extra else ""),
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|" +
             ("---|---|" if extra else "")]
    for _, r in a.iterrows():
        lines.append(
            f"| {CTL_LABEL[r.controller]} | {TRIG_LABEL[r.trigger]} | "
            f"{'on' if r['filter'] else 'off'} | {fmt(r, 'windows_missed')} | "
            f"{r.completion_rate * 100:.1f} % | {fmt(r, 'comfort_below_steps')} | "
            f"{fmt(r, 'comfort_below_degree_h', 1)} | {fmt(r, 'comfort_above_steps')} | "
            f"{fmt(r, 'comfort_above_degree_h', 0)} | {fmt(r, 'unmet_kwh', 2)} | "
            f"{fmt(r, 'electricity_kwh', 1)} | {fmt(r, 'cost_usd', 2)} | {fmt(r, 'starts')} | "
            f"{fmt(r, 'electricity_per_completed_disinfection_kwh', 2)} |" +
            ((f" {fmt(r, 'empty_intersections') if r['filter'] else '-'} | "
              f"{fmt(r, 'band_violations_unreported') if r['filter'] else '-'} |") if extra else ""))
    return "\n".join(lines)


def deltas(agg, axis, a_val, b_val, metrics=("windows_missed", "comfort_below_steps",
                                              "unmet_kwh", "electricity_kwh", "cost_usd",
                                              "starts")):
    keys = [k for k in KEYS if k != axis]
    A = agg[agg[axis] == a_val].set_index(keys)
    B = agg[agg[axis] == b_val].set_index(keys)
    common = A.index.intersection(B.index)
    rows = []
    for k in common:
        ra, rb = A.loc[k], B.loc[k]
        if isinstance(ra, pd.DataFrame):
            ra, rb = ra.iloc[0], rb.iloc[0]
        rows.append(dict(zip(keys, k)) | {m: float(rb[m] - ra[m]) for m in metrics})
    return pd.DataFrame(rows)


def fig_completion(agg, path):
    sub = agg[(agg.plant == "calibra8") & (agg.weather == "off") &
              ((agg.barrier_draw == "known") | (agg.trigger == "none"))]
    trig_order = ["none", "timer", "jit", "recourse", "recourse_q"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    for ax, price in zip(axes, ("dynamic", "flat")):
        s = sub[sub.price == price]
        ylabels, y = [], 0
        for ctl in ("reactive", "price_aware"):
            for trig in trig_order:
                rows = s[(s.controller == ctl) & (s.trigger == trig)]
                if rows.empty:
                    continue
                for _, r in rows.iterrows():
                    mk = "o" if not r["filter"] else "s"
                    ax.plot(r.windows_missed, y + (0.22 if r["filter"] else -0.22), mk,
                            color=OKABE[trig], ms=4.5, mfc=OKABE[trig] if r["filter"] else "white",
                            mew=1.1)
                ylabels.append((y, f"{CTL_LABEL[ctl]}: {TRIG_LABEL[trig]}"))
                y += 1
            y += 0.5
        ax.set_yticks([v for v, _ in ylabels])
        ax.set_yticklabels([t for _, t in ylabels])
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(-0.3, 600)
        ax.set_xticks([0, 1, 10, 100, 416])
        ax.set_xticklabels(["0", "1", "10", "100", "416"])
        ax.set_title({"dynamic": "Time-of-use price:\nthe price-aware policy places every cycle",
                      "flat": "Flat price:\ncompletion depends on the trigger"}[price],
                     loc="left", fontsize=8)
        ax.set_xlabel("missed weekly windows (of 416, lower = better)", fontsize=7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6)
    axes[0].plot([], [], "o", color="k", mfc="white", label="filter off")
    axes[0].plot([], [], "s", color="k", label="filter on (step's draw known)")
    axes[0].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.62, 0.99))
    record(check_figure(fig, path.name), FIGCHECK)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_tradeoff(agg, path):
    sub = agg[(agg.plant == "calibra8") & (agg.weather == "off") & (agg.trigger != "none") &
              (agg.barrier_draw == "known")]
    on = sub[sub["filter"]].set_index(["controller", "trigger", "price"])
    off = sub[~sub["filter"]].set_index(["controller", "trigger", "price"])
    common = on.index.intersection(off.index)
    k = "electricity_per_completed_disinfection_kwh"
    assert (on.loc[common, "windows_missed"] == off.loc[common, "windows_missed"]).all()
    assert (on.loc[common, k] - off.loc[common, k]).abs().max() <= 0.0105
    short = {"timer": "weekly timer", "jit": "just-in-time", "recourse": "recourse",
             "recourse_q": "quantile recourse"}
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, price in zip(axes, ("dynamic", "flat")):
        s = off.reset_index()
        s = s[s.price == price]
        pa = s[s.controller == "price_aware"]
        pa_same = len(pa) > 1 and pa.windows_missed.nunique() == 1 and pa[k].nunique() == 1
        for _, r in s.iterrows():
            mk = "o" if r.controller == "reactive" else "^"
            coincident = r.controller == "price_aware" and pa_same
            col = "#000000" if coincident else OKABE[r.trigger]
            ax.plot(r.windows_missed, r[k], mk, color=col, ms=5.5, mfc="white", mew=1.4)
            if coincident:
                if r.trigger != "jit":
                    continue
                lab, off_xy, ha = "price-aware: all three triggers coincide", (5, -10), "left"
            else:
                lab = ("price-aware, " if r.controller == "price_aware" else "") + short[r.trigger]
                off_xy, ha = ((-6, 3), "right") if r.windows_missed > 30 else ((5, 3), "left")
            ax.annotate(lab, (r.windows_missed, r[k]), xytext=off_xy, textcoords="offset points",
                        fontsize=6, color=col, ha=ha)
        ax.set_xlabel("missed weekly windows (of 416)", fontsize=7)
        ax.set_title({"dynamic": "Time-of-use price", "flat": "Flat price"}[price], loc="left",
                     fontsize=8)
        ax.set_xlim(-2, 48)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("net extra electricity per\ncompleted disinfection (kWh)", fontsize=7)
    handles = [plt.Line2D([], [], ls="", marker="o", color="k", mfc="white", label="reactive"),
               plt.Line2D([], [], ls="", marker="^", color="k", mfc="white",
                          label="price-aware")]
    axes[0].legend(handles=handles, frameon=False, fontsize=6, loc="lower left")
    record(check_figure(fig, path.name), FIGCHECK)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_before_after(ba, path):
    runs = ba["runs"]
    names = [r["variant"].replace("degC", "°C") for r in runs]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True,
                             gridspec_kw={"wspace": 0.08})
    y = np.arange(len(names))
    vals = [("mean_week_bottom_peak_c", "mean weekly peak of the bottom node (°C)"),
            ("weeks_hold_definition", "weeks with every node ≥ 60 °C\nfor a full hour (of 416)"),
            ("unmet_kwh", "unmet hot water (kWh/yr)")]
    for ax, (k, lab) in zip(axes, vals):
        v = [r[k] for r in runs]
        ax.hlines(y, 0, v, color="#0072B2", lw=1.5)
        ax.plot(v, y, "o", color="#0072B2", ms=4)
        for yy, vv in zip(y, v):
            txt = "0" if vv == 0 else (f"{vv:.0f}" if vv >= 10 else f"{vv:.1f}")
            ax.text(vv, yy - 0.28, txt, fontsize=6, ha="center", va="bottom")
        ax.set_xlabel(lab, fontsize=7)
        ax.margins(x=0.15, y=0.12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=6)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(names)
    axes[0].invert_yaxis()
    record(check_figure(fig, path.name), FIGCHECK)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


DOC = REPO / "docs" / "DHW_PAPER_RESULTS.md"


def fill_doc(sections, path=DOC):
    import re
    if not path.exists():
        return 0
    text = path.read_text()
    n = 0
    for key, body in sections.items():
        pat = re.compile(rf"(<!-- table:{re.escape(key)} -->)\n.*?(<!-- /table:{re.escape(key)} -->)",
                         re.S)
        text, k = pat.subn(lambda m: m.group(1) + "\n" + body.strip("\n") + "\n" + m.group(2),
                           text)
        n += k
    path.write_text(text)
    return n


ROW_FIELDS = ["plant", "controller", "trigger", "filter", "barrier_draw", "price", "weather",
              "n_runs", "windows_missed", "completion_rate", "comfort_below_steps",
              "comfort_below_degree_h", "comfort_above_steps", "comfort_above_degree_h",
              "top_below_45_after_draw_steps", "unmet_kwh", "electricity_kwh", "cost_usd",
              "starts", "electricity_per_completed_disinfection_kwh", "empty_intersections",
              "band_violations_unreported", "gaps_over_168h", "max_gap_between_hold_steps_h",
              "seasonal_cop", "model_vs_plant_max_abs_kwh"]


def main():
    main_df = load_jsonl("main")
    pl_df = load_jsonl("power_limited")
    bf_df = load_jsonl("base_filter")
    frames = [d for d in (main_df, pl_df, bf_df) if not d.empty]
    df = pd.concat(frames, ignore_index=True)
    agg = aggregate(df)
    agg.to_csv(RES / "table_all_configs.csv", index=False)
    md = []
    sections = {}
    md.append("### Table 1 -- dynamic (time-of-use) price, weather forecast off, step's draw "
              "known to the filter\n")
    md.append(headline(agg, "dynamic"))
    sections["1"] = md[-1]
    md.append("\n### Table 2 -- flat price, weather forecast off\n")
    md.append(headline(agg, "flat"))
    sections["2"] = md[-1]
    t3 = []
    md.append("\n### Table 3 -- the filter given a causal forecast of the step's draw "
              "(no certificate), weather off\n")
    for price in ("dynamic", "flat"):
        md.append(f"\n{price} price:\n")
        a = agg[(agg.plant == "calibra8") & (agg.price == price) & (agg.weather == "off") &
                (agg["filter"]) & (agg.trigger != "none")]
        lines = ["| controller | trigger | filter's draw | missed windows | top < 45 °C "
                 "(building-h) | unmet (kWh) | electricity (kWh) | cost (USD) | "
                 "unreported band violations |", "|---|---|---|---|---|---|---|---|---|"]
        for _, r in a.sort_values(["controller", "trigger", "barrier_draw"]).iterrows():
            lines.append(f"| {CTL_LABEL[r.controller]} | {TRIG_LABEL[r.trigger]} | "
                         f"{r.barrier_draw} | {fmt(r, 'windows_missed')} | "
                         f"{fmt(r, 'comfort_below_steps')} | {fmt(r, 'unmet_kwh', 2)} | "
                         f"{fmt(r, 'electricity_kwh', 1)} | {fmt(r, 'cost_usd', 2)} | "
                         f"{'-' if pd.isna(r.band_violations_unreported) else int(r.band_violations_unreported)} |")
        md.append("\n".join(lines))
        t3.append(f"{price} price:\n\n" + md[-1])
    sections["3"] = "\n\n".join(t3)
    for axis, a_val, b_val, title in (("weather", "off", "on", "weather forecast on minus off"),
                                      ("price", "dynamic", "flat", "flat minus dynamic price"),
                                      ("filter", False, True, "filter on minus off")):
        d = deltas(agg[(agg.plant == "calibra8") & (agg.barrier_draw == "known")],
                   axis, a_val, b_val)
        if d.empty:
            continue
        d.to_csv(RES / f"table_delta_{axis}.csv", index=False)
        md.append(f"\n### Differences: {title} (same everything else; seeds averaged)\n")
        lines = ["| controller | trigger | " + " | ".join(k for k in ("filter", "price", "weather")
                                                      if k != axis) +
                 " | Δ missed windows | Δ top < 45 °C (building-h) | Δ unmet (kWh) | "
                 "Δ electricity (kWh) | Δ cost (USD) | Δ starts |",
                 "|---|---|" + "---|" * 2 + "---|---|---|---|---|---|"]
        order = {"none": 0, "timer": 1, "jit": 2, "recourse": 3, "recourse_q": 4}
        d = d.assign(o=d.trigger.map(order)).sort_values([c for c in ("controller", "o", "filter",
                                                                    "price", "weather")
                                                           if c in d.columns and c != axis])
        for _, r in d.iterrows():
            extra = " | ".join(str(r[k]) if k != "filter" else ("on" if r[k] else "off")
                               for k in ("filter", "price", "weather") if k != axis)
            lines.append(f"| {CTL_LABEL[r.controller]} | {TRIG_LABEL[r.trigger]} | {extra} | "
                         f"{r.windows_missed:+.1f} | {r.comfort_below_steps:+.0f} | "
                         f"{r.unmet_kwh:+.2f} | {r.electricity_kwh:+.1f} | {r.cost_usd:+.2f} | "
                         f"{r.starts:+.0f} |")
        md.append("\n".join(lines))
        sections[f"delta_{axis}"] = md[-1]
    if not pl_df.empty:
        md.append("\n### Table 4 -- power-limited units (the design note's sizing rule, "
                  "dynamic price, weather off)\n")
        t4 = []
        for plant in sorted(pl_df.plant.unique()):
            md.append(f"\n{plant}:\n")
            md.append(headline(agg, "dynamic", plant=plant, extra=True))
            t4.append(f"{plant}:\n\n" + md[-1])
        sections["4"] = "\n\n".join(t4)
    ba_path = RES / "before_after.json"
    if ba_path.exists():
        ba = json.loads(ba_path.read_text())
        lines = ["| variant | unmet (kWh) | electricity (kWh) | cost (USD) | starts | top < 45 °C "
                 "(building-h) | top > 55 °C (building-h) | weeks, committed convention | "
                 "weeks, hold definition | mean weekly bottom peak (°C) |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for r in ba["runs"]:
            lines.append(f"| {r['variant']} | {r['unmet_kwh']:.2f} | {r['electricity_kwh']:.1f} | "
                         f"{r['cost_usd']:.2f} | {r['starts']} | {r['comfort_below_steps']} | "
                         f"{r['comfort_above_steps']} | {r['weeks_old_definition']} | "
                         f"{r['weeks_hold_definition']} | {r['mean_week_bottom_peak_c']:.1f} |")
        md.insert(0, "### Table 0 -- before/after on the year (same draws, heat pump and "
                     "controller)\n\n" + "\n".join(lines) + "\n")
        sections["0"] = "\n".join(lines)
        R = {r["variant"].split()[0]: r for r in ba["runs"]}
        if "heat_kwh" in R["A"]:
            A, D = R["A"], R["D"]
            more_heat = (D["heat_kwh"] - A["heat_kwh"]) / A["seasonal_cop"]
            cop_part = D["heat_kwh"] * (1.0 / D["seasonal_cop"] - 1.0 / A["seasonal_cop"])
            bal = ["| variant | heat delivered (kWh) | hot water served (kWh) | standing loss "
                   "(kWh) | change in stored heat (kWh) | seasonal COP |",
                   "|---|---|---|---|---|---|"]
            for k in ("A", "D"):
                r = R[k]
                bal.append(f"| {r['variant'].replace('degC', '°C')} | {r['heat_kwh']:.1f} | "
                           f"{r['served_kwh']:.1f} | {r['standing_loss_kwh']:.1f} | "
                           f"{r['stored_change_kwh']:+.1f} | {r['seasonal_cop']:.2f} |")
            bal.append("")
            bal.append(f"Electricity difference D − A: {D['electricity_kwh'] - A['electricity_kwh']:+.1f} "
                       f"kWh = {more_heat:.1f} kWh for the extra heat at A's seasonal COP "
                       f"+ {cop_part:.1f} kWh because D's COP is evaluated at the supply "
                       f"temperature it produces (55/65 °C) and A's at the top node's "
                       f"pre-charge temperature.")
            sections["0_balance"] = "\n".join(bal)
        fig_before_after(ba, RES / "fig_before_after.png")
    (RES / "tables.md").write_text("\n".join(md) + "\n")
    keep = agg[((agg.barrier_draw == "known") | (agg.trigger == "none") |
                (agg.barrier_draw == "forecast"))]
    rows = []
    for _, r in keep.iterrows():
        d = {}
        for f in ROW_FIELDS:
            v = r.get(f)
            if isinstance(v, (np.floating, float)):
                v = None if pd.isna(v) else round(float(v), 4)
            elif isinstance(v, (np.integer,)):
                v = int(v)
            elif isinstance(v, (np.bool_,)):
                v = bool(v)
            d[f] = v
        d["table"] = ("power_limited" if r.plant != "calibra8" else
                      "filter_forecast_draw" if r.barrier_draw == "forecast" and r.trigger != "none"
                      else "main")
        rows.append(d)
    if ba_path.exists():
        for r in ba["runs"]:
            rows.append({"table": "before_after", "variant": r["variant"],
                         **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                            if k in ("unmet_kwh", "electricity_kwh", "cost_usd", "starts",
                                     "weeks_hold_definition", "weeks_old_definition",
                                     "mean_week_bottom_peak_c", "heat_kwh",
                                     "standing_loss_kwh", "seasonal_cop")}})
    (RES / "results_rows.json").write_text(json.dumps(rows, indent=1) + "\n")
    n_filled = fill_doc(sections)
    print("doc blocks filled:", n_filled, "of", len(sections))
    fig_completion(agg, RES / "fig_completion.png")
    fig_tradeoff(agg, RES / "fig_tradeoff.png")
    print("written", RES / "tables.md", len(agg), "configurations")


if __name__ == "__main__":
    main()
