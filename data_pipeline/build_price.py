import json

import numpy as np
import pandas as pd

from data_pipeline.common import SPOT_RAW, SWEDEN_RAW, copy_static, log

YEARS = [2022, 2023, 2024, 2025]
VAT = 0.25
TAX_EX_VAT = {2022: 0.360, 2023: 0.392, 2024: 0.428, 2025: 0.439, 2026: 0.360}
NORTH_DEDUCTION = 0.096
NET_HIGH_EX_VAT = 0.536
NET_LOW_EX_VAT = 0.2144
MARKUP_EX_VAT = 0.084
ZONES = {"SE3": dict(site="karlstad", reduced_tax=False), "SE1": dict(site="lulea", reduced_tax=True)}
EXTRA_ZONES = {"SE2": dict(site=None, reduced_tax=True), "SE4": dict(site=None, reduced_tax=False)}
SPRING_DST_DAYS = {"03-27", "03-26", "03-31", "03-30"}


def outputs():
    return [f"price/price_{z}.csv" for z in ("SE1", "SE2", "SE3", "SE4")] + ["price/price_qc.json", "price/tariff_composition.json"]


def easter(y):
    a = y % 19
    b = y // 100
    c = y % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    mo = (h + l - 7 * m + 114) // 31
    da = ((h + l - 7 * m + 114) % 31) + 1
    return pd.Timestamp(y, mo, da)


def vattenfall_holidays(y):
    e = easter(y)
    return {pd.Timestamp(y, 1, 1), pd.Timestamp(y, 1, 6), e - pd.Timedelta(days=2), e + pd.Timedelta(days=1),
            pd.Timestamp(y, 12, 24), pd.Timestamp(y, 12, 25), pd.Timestamp(y, 12, 26), pd.Timestamp(y, 12, 31)}


def high_load(idx_utc):
    loc = idx_utc.tz_convert("Europe/Stockholm")
    day = loc.tz_localize(None).normalize()
    hol = set().union(*[vattenfall_holidays(y) for y in range(2021, 2027)])
    return ((loc.month.isin([1, 2, 3, 11, 12])) & (loc.weekday < 5) & (loc.hour >= 6) & (loc.hour < 22)
            & ~pd.Index(day).isin(list(hol)))


def build_zones(zones, raw, out):
    idx = pd.date_range("2022-01-01", "2026-01-01", freq="h", tz="UTC", inclusive="left")
    qc = {}
    for z, c in zones.items():
        r = pd.read_csv(raw / f"spot_{z}_raw.csv.gz")
        r["ts"] = pd.to_datetime(r.time_start, utc=True)
        r["te"] = pd.to_datetime(r.time_end, utc=True)
        r["dur_min"] = (r.te - r.ts).dt.total_seconds() / 60
        r["hour"] = r.ts.dt.floor("h")
        assert not r.ts.duplicated().any()
        g = r.groupby("hour").agg(spot=("SEK_per_kWh", "mean"), n=("SEK_per_kWh", "size"))
        expect = np.where(g.index >= pd.Timestamp("2025-09-30T22:00Z"), 4, 1)
        bad = g[g.n != expect]
        dst_end_label_artifacts = int(((r.dur_min != 60) & (r.dur_min != 15)).sum())
        s = g.spot.reindex(idx)
        hl = high_load(idx)
        net = np.where(hl, NET_HIGH_EX_VAT, NET_LOW_EX_VAT)
        tax = np.array([TAX_EX_VAT[y] - (NORTH_DEDUCTION if c["reduced_tax"] else 0.0)
                        for y in idx.tz_convert("Europe/Stockholm").year])
        retail = (1 + VAT) * (s.values + MARKUP_EX_VAT + net + tax)
        df = pd.DataFrame({"time_utc": idx.strftime("%Y-%m-%dT%H:%M:%SZ"), "spot_SEK_per_kWh": np.round(s.values, 5),
                           "network_energy_SEK_per_kWh_exVAT": np.round(net, 4), "energy_tax_SEK_per_kWh_exVAT": np.round(tax, 4),
                           "supplier_markup_SEK_per_kWh_exVAT": MARKUP_EX_VAT, "high_load_hour": hl.astype(int),
                           "retail_total_SEK_per_kWh": np.round(retail, 5)})
        df.to_csv(out / f"price_{z}.csv", index=False)
        per_year = {}
        for y in YEARS:
            m = idx.year == y
            sv = s.values[m]
            per_year[str(y)] = dict(hours=int(m.sum()), missing=int(np.isnan(sv).sum()), spot_mean=round(float(np.nanmean(sv)), 4),
                                    spot_min=round(float(np.nanmin(sv)), 4), spot_max=round(float(np.nanmax(sv)), 4),
                                    negative_hours=int((sv < 0).sum()), retail_mean=round(float(np.nanmean(retail[m])), 4),
                                    high_load_hours=int(hl[m].sum()))
        q15 = r[r.dur_min == 15]
        qc[z] = dict(site=c["site"], raw_intervals=int(len(r)), first=str(r.ts.min()), last_end=str(r.te.max()),
                     quarter_hour_intervals=int(len(q15)), quarter_hour_from=str(q15.ts.min()) if len(q15) else None,
                     hours_with_wrong_interval_count=int(len(bad)), provider_dst_end_label_artifacts=dst_end_label_artifacts,
                     missing_hours=int(s.isna().sum()), per_year=per_year)
        log("price", z, "missing hours", qc[z]["missing_hours"], "wrong-count hours", qc[z]["hours_with_wrong_interval_count"])
    return json.loads(json.dumps(qc))


def extra_zone_notes(qc, raw, out):
    for z in EXTRA_ZONES:
        r = pd.read_csv(raw / f"spot_{z}_raw.csv.gz")
        per_day = r.groupby(r.time_start.str[:10]).size()
        short = [d for d in per_day[per_day == 23].index if d[5:] not in SPRING_DST_DAYS]
        px = pd.read_csv(out / f"price_{z}.csv")
        qc[z]["missing_hours_utc"] = px.time_utc[px.spot_SEK_per_kWh.isna()].tolist()
        qc[z]["provider_short_days_non_dst"] = short
        qc[z]["note"] = ("fetched 2022-01-01..2025-12-31 local days, so 2025-12-31T23:00Z is not covered "
                         "(SE1/SE3 were fetched one day further)")
        qc[z]["tariff_assumption"] = ("reduced (northern) energy tax" if z == "SE2" else "standard energy tax") + \
            "; network tariff and markup as SE1/SE3 (Vattenfall T4 2025 list); no site attached"
    return qc


def run(raw=SPOT_RAW, out=SWEDEN_RAW / "price"):
    out.mkdir(parents=True, exist_ok=True)
    qc = build_zones(ZONES, raw, out)
    qc.update(extra_zone_notes(build_zones(EXTRA_ZONES, raw, out), raw, out))
    (out / "price_qc.json").write_text(json.dumps(qc, indent=1))
    copy_static("tariff_composition.json", out / "tariff_composition.json")
