import json

import numpy as np
import pandas as pd

from data_pipeline.common import SMHI_RAW, SWEDEN_RAW, log

Y0, Y1 = 2020, 2025
SITES = {
    "karlstad": dict(T=93220, GHI=93235, donors=[93520, 92410], lon=13.3374,
                     names={93220: "Karlstad Flygplats", 93235: "Karlstad Sol", 93520: "Sunne A", 92410: "Arvika A"}),
    "lulea": dict(T=162860, GHI=162015, donors=[161940, 162790], lon=22.1240,
                  names={162860: "Lulea-Kallax Flygplats", 162015: "Lulea Sol", 161940: "Boden", 162790: "Rodkallen A"}),
}


def outputs():
    return ["weather/weather_karlstad.csv", "weather/weather_lulea.csv", "weather/weather_qc.json"]


def parse(fn):
    lines = open(fn, encoding="utf-8-sig").read().splitlines()
    i = [k for k, line in enumerate(lines) if line.startswith("Datum;")][0]
    rows = []
    for line in lines[i + 1:]:
        f = line.split(";")
        if len(f) < 4 or not f[0][:4].isdigit():
            continue
        rows.append((f[0] + " " + f[1], f[2], f[3]))
    d = pd.DataFrame(rows, columns=["t", "v", "q"])
    d["t"] = pd.to_datetime(d.t, utc=True)
    d["v"] = pd.to_numeric(d.v, errors="coerce")
    d = d[d.t.dt.minute == 0].drop_duplicates("t").set_index("t")
    return d


def runs(mask):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def donor_fill(target, donors, i0, length, win=15 * 24):
    lo, hi = max(0, i0 - win), min(len(target), i0 + length + win)
    for k, d in enumerate(donors):
        seg = d[i0:i0 + length]
        if np.isnan(seg).any():
            continue
        tw, dw = target[lo:hi], d[lo:hi]
        ok = ~np.isnan(tw) & ~np.isnan(dw)
        if ok.sum() < 48:
            continue
        b, a = np.polyfit(dw[ok], tw[ok], 1)
        return a + b * seg, k
    return None, None


def fill_T(T, donors):
    T = T.copy()
    flag = np.zeros(len(T), dtype=np.int8)
    for i0, length in runs(np.isnan(T)):
        span = np.arange(i0, i0 + length)
        if length <= 2 and i0 > 0 and i0 + length < len(T):
            T[i0:i0 + length] = np.interp(span, [i0 - 1, i0 + length], [T[i0 - 1], T[i0 + length]])
            flag[i0:i0 + length] = 1
            continue
        v, k = donor_fill(T, donors, i0, length)
        if v is not None:
            T[i0:i0 + length] = v
            flag[i0:i0 + length] = 2
        else:
            T[i0:i0 + length] = np.interp(span, [i0 - 1, i0 + length], [T[i0 - 1], T[i0 + length]])
            flag[i0:i0 + length] = 3
    return T, flag


def validate(T, donors, rng, n=300, length=11):
    idx = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31 23:00", freq="h", tz="UTC")
    starts = np.where((idx.hour == 19))[0]
    starts = [s for s in starts if s + length + 2 < len(T) and s > 2 and not np.isnan(T[s - 1:s + length + 1]).any()]
    pick = rng.choice(starts, size=min(n, len(starts)), replace=False)
    e_d, e_l = [], []
    for s in pick:
        truth = T[s:s + length].copy()
        Tm = T.copy()
        Tm[s:s + length] = np.nan
        v, k = donor_fill(Tm, donors, s, length)
        lin = np.interp(np.arange(s, s + length), [s - 1, s + length], [T[s - 1], T[s + length]])
        if v is not None:
            e_d.append(v - truth)
            e_l.append(lin - truth)
    e_d, e_l = np.concatenate(e_d), np.concatenate(e_l)
    return dict(windows=int(len(pick)), window_h=length, rmse_donor_K=round(float(np.sqrt(np.mean(e_d ** 2))), 3),
                bias_donor_K=round(float(np.mean(e_d)), 3), rmse_linear_K=round(float(np.sqrt(np.mean(e_l ** 2))), 3),
                max_abs_donor_K=round(float(np.max(np.abs(e_d))), 2))


def run(raw=SMHI_RAW, out=SWEDEN_RAW / "weather"):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261026)
    idx = pd.date_range(f"{Y0}-01-01", f"{Y1}-12-31 23:00", freq="h", tz="UTC")
    qc = {}
    for site, c in SITES.items():
        t = parse(raw / f"p1_{c['T']}_corrected-archive.csv")
        g = parse(raw / f"p11_{c['GHI']}_corrected-archive.csv")
        dn = [parse(raw / f"p1_{s}_corrected-archive.csv").v.reindex(idx).values for s in c["donors"]]
        T = t.v.reindex(idx).values.astype(float)
        Tq = t.q.reindex(idx)
        n_missing = int(np.isnan(T).sum())
        r = runs(np.isnan(T))
        lens = np.array([x[1] for x in r]) if r else np.array([0])
        val = validate(T, dn, rng)
        Tf, flag = fill_T(T, dn)
        gs = g.copy()
        gs.index = gs.index - pd.Timedelta(hours=1)
        G = gs.v.reindex(idx).values.astype(float)
        Gq = gs.q.reindex(idx)
        gflag = np.where(Gq.values == "Y", 1, 0).astype(np.int8)
        gmiss = np.isnan(G)
        if gmiss.any():
            G = pd.Series(G).interpolate(limit_direction="both").values
            gflag[gmiss] = 2
        prof = pd.Series(G, index=idx).groupby(idx.hour).mean()
        centroid = float(((prof.index + 0.5) * prof).sum() / prof.sum())
        df = pd.DataFrame({"time_utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "T_out_C": np.round(Tf, 2),
                           "GHI_Wm2": np.round(G, 2), "T_flag": flag, "GHI_flag": gflag})
        df.to_csv(out / f"weather_{site}.csv", index=False)
        per_year = {}
        for y in range(Y0, Y1 + 1):
            m = idx.year == y
            per_year[str(y)] = dict(hours=int(m.sum()), T_observed=int((flag[m] == 0).sum()), T_interp_le2h=int((flag[m] == 1).sum()),
                                    T_donor=int((flag[m] == 2).sum()), T_fallback_linear=int((flag[m] == 3).sum()),
                                    T_mean_C=round(float(Tf[m].mean()), 2), T_min_C=round(float(Tf[m].min()), 1),
                                    GHI_qualityY=int((gflag[m] == 1).sum()), GHI_filled=int((gflag[m] == 2).sum()),
                                    GHI_annual_kWh_m2=round(float(G[m].sum() / 1000), 1))
        qc[site] = dict(stations={k: v for k, v in c["names"].items()}, T_station=c["T"], GHI_station=c["GHI"], donors=c["donors"],
                        T_missing_hours_2020_2025=n_missing, T_gap_runs=int(len(r)), T_gap_max_h=int(lens.max()),
                        T_quality_Y_hours=int((Tq.values == "Y").sum()),
                        donor_validation_synthetic_night_gaps=val, GHI_centroid_hour_utc_after_shift=round(centroid, 3),
                        solar_noon_utc_approx=round(12 - c["lon"] / 15, 3), per_year=per_year)
        log("weather", site, "missing T hours", n_missing, "GHI centroid", round(centroid, 3))
    json.dump(qc, open(out / "weather_qc.json", "w"), indent=1, default=str)
