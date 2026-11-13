#!/usr/bin/env python3
import json
from pathlib import Path

D = Path(__file__).resolve().parent / "smoke_results"
def _g(d, k, f=sum):
    v = d.get(k)
    return None if v is None else f(v)


COLS = [("wall s", lambda d: d["wall_run_s"], "{:.0f}"),
        ("elec kWh", lambda d: sum(d["elec_kwh"]), "{:.0f}"),
        ("viol h", lambda d: sum(d["violation_hours"]), "{:.0f}"),
        ("inherited", lambda d: sum(d["violations_inherited"]), "{:.0f}"),
        ("RF fail", lambda d: _g(d, "recursive_feasibility_failures"), "{:.0f}"),
        ("unexpl.", lambda d: _g(d, "violations_unexplained"), "{:.0f}"),
        ("announced", lambda d: _g(d, "violations_announced_prev_hour"), "{:.0f}"),
        ("deliv<45 h", lambda d: sum(d["hours_delivered_below_floor"]), "{:.0f}"),
        ("min<45 h", lambda d: sum(d["time_below_floor_h"]), "{:.1f}"),
        ("unmet kWh", lambda d: sum(d["unmet_kwh"]), "{:.1f}"),
        ("weeks b20", lambda d: f'{sum(d["weeks_completed_bottom20"])}/{len(d["houses"]) * d["weeks_full"]}', "{}"),
        ("weeks mean", lambda d: f'{sum(d["weeks_completed_whole_mean"])}', "{}"),
        ("weeks all-nodes", lambda d: f'{sum(d["weeks_completed_all_nodes"])}', "{}"),
        ("max gap h", lambda d: max(d["max_gap_between_holds_h"]), "{:.0f}"),
        ("dis-supply h", lambda d: sum(d["dis_supply_hours"]), "{:.0f}"),
        ("kWh@extrap sup", lambda d: sum(d["elec_at_extrap_supply_kwh"]), "{:.0f}"),
        ("SH K h<Tmin", lambda d: sum(d["sh_degree_hours_below_min"]), "{:.0f}"),
        ("interv. rate", lambda d: _g(d, "intervention_rate", lambda v: sum(v) / len(v)), "{:.3f}"),
        ("P(d>m)", lambda d: sum(d["exceed_freq_delta_gt_m"]) / len(d["exceed_freq_delta_gt_m"]), "{:.4f}"),
        ("bound P", lambda d: _g(d, "exceed_freq_upper_bound", max), "{:.4f}"),
        ("brine 24h drop K", lambda d: max(d["brine_max_24h_drop_k"]), "{:.1f}"),
        ("h over cap", lambda d: (f'{sum(d["hours_over_house_cap"]):.0f}/{d.get("hours_over_district_cap", 0)}' if "hours_over_house_cap" in d else None), "{}")]


def main():
    rows = []
    for f in sorted(D.glob("*__*.json")):
        d = json.loads(f.read_text())
        cells = []
        for _, fn, fmt in COLS:
            try:
                v = fn(d)
            except KeyError:
                v = None
            cells.append("-" if v is None else fmt.format(v))
        rows.append(f"| {d['run']} | {d['hours']} | {len(d['houses'])} | " + " | ".join(cells) + " |")
    head = "| run | hours | houses | " + " | ".join(c for c, _, _ in COLS) + " |"
    sep = "|" + "---|" * (len(COLS) + 3)
    txt = ("Smoke runs v2.1, tx_travis_8b houses 0-1 (sums over houses; PROVISIONAL: Texas data, 2 houses,"
           " one year, in-sample, every hour source-extrapolated). 'viol h' = hours whose intra-hour top-node"
           " minimum fell below 45 C; 'inherited' = of those, hours that STARTED below 45 C; 'RF fail' ="
           " recursive-feasibility failures (hour started >= 45 C but could not be certified although the"
           " previous hour was certified and within its quantile); 'unexpl.' = violations with no draw"
           " exceedance in hour t or in the t-1 plan for t, not announced at t-1, not declared infeasible, not"
           " inherited (T3(iv)); 'announced' = violations announced one hour earlier (T2); 'deliv<45 h' = hours"
           " in which water delivered at the tap was below 45 C; 'weeks b20/mean/all-nodes' = calendar weeks"
           " with a 1-h hold under the bottom-20 % mean (primary), whole-volume mean, all-nodes rules;"
           " 'bound P' = one-sided T3 bound alpha + (range+eta)/(eta T) on P(d>m).\n\n"
           + head + "\n" + sep + "\n" + "\n".join(rows) + "\n")
    (D / "SMOKE_TABLE.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
