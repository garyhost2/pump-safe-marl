import datetime as dt
import gzip
import json
import math
import random

import holidays
import numpy as np
import OpenDHW as OD
import pandas as pd
from OpenDHW import OpenDHW as odhw

from data_pipeline.common import SWEDEN_RAW, log

S_STEP = 60
FACTOR = 1.2
RHO, CP = OD.OpenDHW.rho, OD.OpenDHW.cp
YEARS = list(range(2020, 2026))
HOUSES = [
    dict(house=1, occupants=1, L40_per_person_day=70.0),
    dict(house=2, occupants=2, L40_per_person_day=55.0),
    dict(house=3, occupants=2, L40_per_person_day=66.0),
    dict(house=4, occupants=3, L40_per_person_day=61.0),
    dict(house=5, occupants=4, L40_per_person_day=50.0),
    dict(house=6, occupants=4, L40_per_person_day=61.0),
    dict(house=7, occupants=5, L40_per_person_day=48.0),
    dict(house=8, occupants=3, L40_per_person_day=72.0),
]


def outputs():
    files = [f"draws/draws_house{h['house']}_{y}.csv" for h in HOUSES for y in YEARS]
    files += [f"draws/draws_hourly_{y}.csv" for y in YEARS]
    return files + ["draws/draws_1min_events.csv.gz", "draws/draws_meta.json"]


def yearly_probabilities(initial_day, p_off_day, p_work_day, s_step, holidays_doy, ndays):
    p_final = []
    tsd = int(24 * 3600 / s_step)
    for day in range(ndays):
        current_day = (day + initial_day) % 7
        is_off_day = current_day in (5, 6) or (day + 1) in holidays_doy
        p_day = p_off_day if is_off_day else p_work_day
        arg = math.pi * (2 / 365 * day - 1 / 4)
        probability_season = 1 + 0.1 * np.cos(arg)
        for step in range(tsd):
            p_final.append(p_day[step] * probability_season)
    return np.array(p_final)


def distribute(timeseries_df, cats_series, ndays):
    s_step = S_STEP
    drawoff_steps = int(cats_series['drawoff_duration_min'] * 60 / s_step)
    V_curr, V_max, drawoffs = 0, cats_series['mean_vol_per_year'], []
    while V_curr <= V_max:
        drawoff = odhw.generate_single_drawoff_inside_boundaries(cats_series, s_step)
        drawoffs.append(drawoff)
        V_curr += drawoff / 3600 * s_step * drawoff_steps
    p_norm_integral = list(timeseries_df['p_norm_integral'])
    min_rand, max_rand = min(p_norm_integral), max(p_norm_integral)
    p_drawoffs = [random.uniform(min_rand, max_rand) for _ in range(len(drawoffs))]
    p_drawoffs.sort()
    p_norm_integral.sort()
    n = int(ndays * 24 * 3600 / s_step)
    water_LperH_cat = [0] * n
    water_LperH = list(timeseries_df['Water_LperH'])
    max_flow_rate = cats_series['max_flow_rate_per_drawoff_LperH']
    drawoff_count = 0
    for time_step, p_current_sum in enumerate(p_norm_integral):
        if water_LperH[time_step] >= max_flow_rate:
            continue
        if drawoff_count >= len(drawoffs):
            break
        while p_drawoffs[drawoff_count] < p_current_sum:
            drawoff = drawoffs[drawoff_count]
            delta = [drawoff] * drawoff_steps
            ok = True
            for i in range(drawoff_steps):
                if time_step + i >= n or water_LperH[time_step + i] + delta[i] > max_flow_rate:
                    ok = False
                    break
            if ok:
                for i in range(drawoff_steps):
                    water_LperH[time_step + i] += delta[i]
                    water_LperH_cat[time_step + i] += delta[i]
                drawoff_count += 1
            else:
                break
            if drawoff_count >= len(drawoffs):
                break
    timeseries_df['Water_LperH'] = water_LperH
    return timeseries_df, drawoff_count, len(drawoffs)


def categories(vol_per_day, pinned_order):
    cats = odhw.get_data_drawoff_categories(s_step=S_STEP, categories=4, mean_drawoff_vol_per_day=vol_per_day, building_type='SFH')
    if pinned_order:
        cats = cats.sort_values(by=['drawoff_duration_min', 'mean_flow_rate_per_drawoff_LperH'], ascending=False, kind='mergesort')
    return cats


def profile(vol_per_day, ndays, initial_day, holidays_doy, pinned_order=True):
    cats = categories(vol_per_day, pinned_order)
    cats['mean_vol_per_year'] = cats['mean_vol_per_day'] * ndays
    p_we = odhw.generate_daily_probability_step_function(mode='off-day', s_step=S_STEP, building_type='SFH')
    p_wd = odhw.generate_daily_probability_step_function(mode='work-day', s_step=S_STEP, building_type='SFH')
    p_wd_w, p_we_w, _ = odhw.shift_weekend_weekday(p_work_day=p_wd, p_off_day=p_we, factor=FACTOR)
    p_final = yearly_probabilities(initial_day, p_we_w, p_wd_w, S_STEP, holidays_doy, ndays)
    df = pd.DataFrame({'p_norm_integral': odhw.normalize_and_sum_list(lst=p_final)})
    df['Water_LperH'] = [0] * len(df)
    placed = {}
    for i in range(len(cats)):
        df, nplaced, ngen = distribute(df, cats.iloc[i], ndays)
        placed[int(cats.iloc[i]['mean_flow_rate_per_drawoff_LperH'])] = (nplaced, ngen)
    return np.asarray(df['Water_LperH'], dtype=float) / 60.0, placed


def selftest():
    for seed, vol, ini in [(1, 180, 0), (7, 95, 3)]:
        random.seed(seed)
        np.random.seed(seed)
        ref = odhw.generate_dhw_profile(s_step=60, categories=4, mean_drawoff_vol_per_day=vol, occupancy=1, holidays=[1, 359],
                                        building_type='SFH', weekend_weekday_factor=FACTOR, initial_day=ini)
        random.seed(seed)
        np.random.seed(seed)
        mine, _ = profile(vol, 365, ini, [1, 359], pinned_order=False)
        diff = np.abs(ref['Water_LperH'].values - mine * 60.0).max()
        log("draws selftest seed", seed, "max |diff| L/h =", diff)
        assert diff < 1e-9
    return True


def se_holidays(year):
    try:
        return holidays.SE(years=year, include_sundays=False)
    except TypeError:
        return holidays.SE(years=year)


def se_offdays(year):
    days = set(se_holidays(year).keys())
    days.add(dt.date(year, 12, 24))
    days.add(dt.date(year, 12, 31))
    days.add([dt.date(year, 6, d) for d in range(19, 26) if dt.date(year, 6, d).weekday() == 4][0])
    return sorted(days)


def local_to_utc_index(year, ndays):
    utc = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="min", tz="UTC", inclusive="left")
    loc = utc.tz_convert("Europe/Stockholm").tz_localize(None)
    idx = ((loc - pd.Timestamp(f"{year}-01-01")) / pd.Timedelta(minutes=1)).astype(np.int64).values
    n = ndays * 1440
    wrapped = idx >= n
    idx = np.where(wrapped, idx - n, idx)
    return utc, idx, int(wrapped.sum())


def run(pkg=SWEDEN_RAW):
    out = pkg / "draws"
    out.mkdir(parents=True, exist_ok=True)
    selftest()
    kwh_per_L = RHO * CP * 30 / 3.6e6
    meta = dict(generator="OpenDHW 0.2.9 (PyPI), ported yearly functions with ndays", s_step_s=60, categories=4,
                building_type="SFH", weekend_weekday_factor=FACTOR, max_flow_L_per_h=1200,
                kWh_10to40_per_litre=kwh_per_L, houses=HOUSES, runs=[])
    hourly = {y: {} for y in YEARS}
    ev_rows = []
    for H in HOUSES:
        for y in YEARS:
            seed = 100000 + 1000 * H['house'] + y
            ndays = 366 if (y % 4 == 0) else 365
            ini = dt.date(y, 1, 1).weekday()
            off = se_offdays(y)
            doy = [d.timetuple().tm_yday for d in off]
            random.seed(seed)
            np.random.seed(seed)
            vol_day = H['occupants'] * H['L40_per_person_day']
            prof, placed = profile(vol_day, ndays, ini, doy)
            utc, idx, nwrap = local_to_utc_index(y, ndays)
            L = prof[idx]
            s = pd.Series(L, index=utc)
            m5 = s.resample('5min').sum()
            df = pd.DataFrame({'time_utc': m5.index.strftime('%Y-%m-%dT%H:%M:%SZ'), 'litres_40C': np.round(m5.values, 3),
                               'kWh_10to40': np.round(m5.values * kwh_per_L, 5)})
            df.to_csv(out / f"draws_house{H['house']}_{y}.csv", index=False, float_format='%g')
            hourly[y][f"house{H['house']}_litres_40C"] = s.resample('h').sum().round(3)
            nz = np.nonzero(L)[0]
            ev_rows.append(pd.DataFrame({'house': H['house'], 'year': y, 'time_utc': utc[nz].strftime('%Y-%m-%dT%H:%M:%SZ'),
                                         'litres_40C': np.round(L[nz], 3)}))
            meta['runs'].append(dict(house=H['house'], year=y, seed=seed, python_random_seed=seed, numpy_seed=seed,
                                     ndays=ndays, initial_weekday=ini, offdays=[str(d) for d in off],
                                     target_L40_per_day=vol_day, profile_L40_total=round(float(prof.sum()), 1),
                                     utc_file_L40_total=round(float(L.sum()), 1), utc_file_L40_mean_per_day=round(float(L.sum()) / ndays, 2),
                                     minutes_wrapped_from_profile_start=nwrap,
                                     drawoffs_placed_vs_generated={str(k): v for k, v in placed.items()},
                                     kWh_10to40_total=round(float(L.sum() * kwh_per_L), 1)))
        log("draws house", H['house'], "done")
    for y in YEARS:
        hdf = pd.DataFrame(hourly[y])
        hdf.insert(0, 'time_utc', hdf.index.strftime('%Y-%m-%dT%H:%M:%SZ'))
        hdf.to_csv(out / f"draws_hourly_{y}.csv", index=False, float_format='%g')
    with gzip.GzipFile(out / "draws_1min_events.csv.gz", "wb", mtime=0) as fh:
        fh.write(pd.concat(ev_rows).to_csv(index=False, float_format='%g').encode())
    json.dump(meta, open(out / "draws_meta.json", 'w'), indent=1)
