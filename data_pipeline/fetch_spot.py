import concurrent.futures as cf
import datetime as dt
import time

import pandas as pd

from data_pipeline.common import SPOT_RAW, log, session
from data_pipeline.sources import SPOT_SPANS, SPOT_URL

COLUMNS = ["time_start", "time_end", "SEK_per_kWh", "EUR_per_kWh", "EXR"]


def raw_path(zone):
    return SPOT_RAW / f"spot_{zone}_raw.csv.gz"


def get(s, day, zone):
    for k in range(4):
        try:
            r = s.get(SPOT_URL.format(y=day.year, m=day.month, d=day.day, z=zone), timeout=30)
            if r.status_code == 200:
                return [(x["time_start"], x["time_end"], x["SEK_per_kWh"], x["EUR_per_kWh"], x["EXR"]) for x in r.json()]
            if r.status_code == 404:
                return None
        except Exception:
            pass
        time.sleep(2 * (k + 1))
    return None


def fetch_zone(zone):
    d0, d1 = (dt.date.fromisoformat(x) for x in SPOT_SPANS[zone])
    days = [d0 + dt.timedelta(days=i) for i in range((d1 - d0).days + 1)]
    s = session()
    rows, missing = [], []
    with cf.ThreadPoolExecutor(4) as ex:
        for day, res in zip(days, ex.map(lambda d: get(s, d, zone), days)):
            if res is None:
                missing.append(str(day))
            else:
                rows += res
    df = pd.DataFrame(rows, columns=COLUMNS)
    SPOT_RAW.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path(zone), index=False, compression={"method": "gzip", "mtime": 0})
    log("spot", zone, len(df), "intervals, missing days:", len(missing), missing[:10])
    return missing


def run(zones=("SE3", "SE1", "SE2", "SE4"), refresh=False):
    fetched = False
    for z in zones:
        if raw_path(z).exists() and not refresh:
            continue
        missing = fetch_zone(z)
        if missing:
            log(f"WARNING spot {z}: {len(missing)} days could not be fetched; delete {raw_path(z)} and run again to retry")
        fetched = True
    return fetched
