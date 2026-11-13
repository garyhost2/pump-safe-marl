#!/usr/bin/env python3
import glob
import json
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
sm = lambda x: int(np.sum(x))
L = ["# Paper B smoke runs -- PRE-FIX, NOT QUOTABLE (HW2 and HW14 open)", "",
     "Short runs that prove the pipeline end to end. Not results. 8 Karlstad houses, 4 weeks from the first",
     "Monday of 2025 (672 h), borehole spun up 5 years, auxiliary heater `auto_min`. `regime ok` = whole periods",
     "in which the run's OWN regime was met, over all houses, by the sub-step scorer. `Thermia rule` = programs /",
     "done within 7 h / alarms at the top, middle and weighted sensor (the ending sensor is not known). The",
     "regime columns are our criteria; a miss there is not a Thermia failure.", "",
     "| run | price | regime aimed at | controller | filter | cost SEK | elec kWh | aux kWh | dis-supply h | regime ok | h below 45 C | tap < 50 C draw-h | Thermia rule top / middle / weighted |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for f in sorted(glob.glob(src + "/*.json")):
    d = json.load(open(f))
    sp, s, m = d["spec"], d["summary"], d["metrics"]
    h, th = s["hygiene"], s["thermia_rule"]
    reg = sp.get("regime") or "weekly_bottom20_60_1h"
    key = reg + ("_tol1K" if sp.get("regime_tol_k") else "")
    ok = "n/a" if reg == "none" else f"{sm(h[key]['periods_completed'])}/{h[key]['periods_full'] * len(s['houses'])}"
    tr = " / ".join(f"{sm(th[k]['programs'])}-{sm(th[k]['done_within_limit'])}-{sm(th[k]['alarms'])}" for k in ("top", "middle", "weighted"))
    L.append(f"| {sp['name']} | {d['price_area']['area']} | {reg}{' (1 K tol)' if sp.get('regime_tol_k') else ''} | {sp['ctl']}"
             f"{'/' + sp['kw']['end_sensor'] if sp['ctl'] == 'thermia_weekly' else ''} | {'on' if sp.get('filtered', True) else 'off'} |"
             f" {m['cost']:.0f} | {m['elec_kwh']:.0f} | {m['elec_aux_kwh']:.0f} | {m['dis_supply_hours']:.0f} | {ok} |"
             f" {m['viol_intra_hour']:.0f} | {sm(h['tap_50c']['draw_hours_delivered_below'])}/{sm(h['tap_50c']['draw_hours'])} | {tr} |")
L += ["", "Every regime is scored on every run; the full per-regime, per-house counts are in each JSON under `summary.hygiene`."]
open(out, "w").write("\n".join(L) + "\n")
print(out, len(L) - 12, "rows")
