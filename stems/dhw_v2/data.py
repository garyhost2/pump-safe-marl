from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from stems.dhw_v2.tank import CV

__all__ = ["HouseData", "DistrictData", "load_texas", "load_package", "intra_hour_weights",
           "ALLOWED_DT_MIN", "PRICE_AREAS", "load_area_spot", "apply_price_area"]

ALLOWED_DT_MIN = (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60)
REPO = Path(__file__).resolve().parents[2]


def intra_hour_weights(n_sub: int, profile: str = "uniform") -> np.ndarray:
    if profile == "uniform":
        return np.full(n_sub, 1.0 / n_sub)
    if profile == "front":
        w = np.zeros(n_sub)
        w[0] = 1.0
        return w
    raise ValueError(f"unknown intra-hour profile {profile!r}")


@dataclass
class HouseData:
    id: str
    draw_kwh: np.ndarray
    draw_dt_min: int
    t_cold_c: np.ndarray
    building: Optional[Dict[str, object]] = None
    tank_alt: Optional[Dict[str, float]] = None
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class DistrictData:
    name: str
    source: str
    time_utc: np.ndarray
    hour_of_day: np.ndarray
    day_of_week: np.ndarray
    t_out_c: np.ndarray
    ghi_w_m2: np.ndarray
    price: np.ndarray
    currency: str
    t_ground_c: float
    houses: List[HouseData]
    meta: Dict[str, object] = field(default_factory=dict)
    parent: Optional["DistrictData"] = field(default=None, repr=False, compare=False)
    parent_offset: int = 0

    @property
    def T(self) -> int:
        return len(self.t_out_c)

    @property
    def B(self) -> int:
        return len(self.houses)

    def subset(self, houses: Optional[Sequence[int]] = None, hours: Optional[int] = None) -> "DistrictData":
        idx = list(range(self.B)) if houses is None else list(houses)
        T = self.T if hours is None else int(hours)
        hs = []
        for i in idx:
            h = self.houses[i]
            k = 60 // h.draw_dt_min
            b = dict(h.building) if h.building else None
            if b is not None and "heating_demand_kwh" in b:
                b["heating_demand_kwh"] = np.asarray(b["heating_demand_kwh"])[:T]
            hs.append(HouseData(h.id, h.draw_kwh[:T * k], h.draw_dt_min, h.t_cold_c[:T], b,
                                h.tank_alt, dict(h.meta)))
        return DistrictData(self.name, self.source, self.time_utc[:T], self.hour_of_day[:T],
                            self.day_of_week[:T], self.t_out_c[:T], self.ghi_w_m2[:T],
                            self.price[:T], self.currency, self.t_ground_c, hs, dict(self.meta),
                            parent=self.parent if self.parent is not None else self,
                            parent_offset=self.parent_offset)

    def hourly_draw_kwh(self) -> np.ndarray:
        out = np.zeros((self.T, self.B))
        for j, h in enumerate(self.houses):
            k = 60 // h.draw_dt_min
            out[:, j] = h.draw_kwh[: self.T * k].reshape(self.T, k).sum(axis=1)
        return out

    def draws_substep(self, dt_min: int, profile: str = "uniform") -> np.ndarray:
        if 60 % dt_min:
            raise ValueError("dt_min must divide 60")
        S = 60 // dt_min
        out = np.zeros((self.T, S, self.B))
        for j, h in enumerate(self.houses):
            k = 60 // h.draw_dt_min
            d = h.draw_kwh[: self.T * k].reshape(self.T, k)
            if k == S:
                out[:, :, j] = d
            elif k > S:
                if k % S:
                    raise ValueError(f"draw resolution {h.draw_dt_min} min does not nest in {dt_min} min")
                out[:, :, j] = d.reshape(self.T, S, k // S).sum(axis=2)
            else:
                if S % k:
                    raise ValueError(f"plant step {dt_min} min does not nest in {h.draw_dt_min} min")
                w = intra_hour_weights(S // k, profile)
                out[:, :, j] = (d[:, :, None] * w[None, None, :]).reshape(self.T, S)
        return out


_TANKS_JSON = Path(__file__).resolve().parents[1] / "data" / "tx_travis_8b_dhw_tanks.json"


def _read_csv_columns(path: Path) -> Dict[str, np.ndarray]:
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        rows = [r for r in rd if r]
    cols = {}
    for j, name in enumerate(head):
        try:
            cols[name] = np.array([float(r[j]) for r in rows])
        except ValueError:
            cols[name] = np.array([r[j] for r in rows])
    return cols


def load_texas(schema: Optional[str] = None, steps: int = 8759, *, t_cold_c: float = 10.0,
               start: str = "2018-01-01T00:00:00-06:00") -> DistrictData:
    schema = Path(schema) if schema else REPO / "citylearn_schemas" / "tx_travis_8b" / "schema.json"
    s = json.loads(schema.read_text())
    root = Path(s["root_directory"])
    inc = [(k, v) for k, v in s["buildings"].items() if v.get("include")]
    w = _read_csv_columns(root / inc[0][1]["weather"])
    t_out = w["outdoor_dry_bulb_temperature"][:steps].astype(float)
    ghi = (w["direct_solar_irradiance"] + w["diffuse_solar_irradiance"])[:steps].astype(float)
    pr = _read_csv_columns(Path(inc[0][1]["pricing"]))["electricity_pricing"][:steps].astype(float)
    T = len(t_out)
    t0 = datetime.fromisoformat(start)
    times = [t0 + timedelta(hours=i) for i in range(T)]
    time_utc = np.array([t.astimezone(timezone.utc).replace(tzinfo=None) for t in times],
                        dtype="datetime64[s]")
    hod = np.array([t.hour for t in times])
    dow = np.array([t.weekday() for t in times])
    tanks = json.loads(_TANKS_JSON.read_text()) if _TANKS_JSON.is_file() else {}
    houses = []
    for name, b in inc:
        e = _read_csv_columns(root / b["energy_simulation"])
        draw = e["dhw_demand"][:T].astype(float)
        heat = e["heating_demand"][:T].astype(float)
        bld = {"model": "demand", "heating_demand_kwh": heat, "T_set_c": 21.0,
               "T_comfort_min_c": 20.0, "source": "CityLearn tx_travis_8b heating_demand (ResStock)",
               "heating_curve": {"curve_c": 35.0, "T_supply_min_c": 20.0, "T_supply_max_c": 50.0,
                                 "seasonal_stop_c": 99.0}}
        houses.append(HouseData(id=name, draw_kwh=draw, draw_dt_min=60,
                                t_cold_c=np.full(T, float(t_cold_c)), building=bld,
                                tank_alt=tanks.get(name),
                                meta={"energy_simulation": b["energy_simulation"]}))
    return DistrictData(
        name="tx_travis_8b", source=f"citylearn:{schema}", time_utc=time_utc, hour_of_day=hod,
        day_of_week=dow, t_out_c=t_out, ghi_w_m2=ghi, price=pr, currency="USD",
        t_ground_c=float(np.mean(t_out) + 1.0), houses=houses,
        meta={"sources": {"weather": "CityLearn tx_travis_county_neighborhood weather.csv",
                          "price": "citylearn_challenge_2022_phase_all pricing.csv",
                          "draws": "CityLearn dhw_demand (hourly, thermal kWh)",
                          "buildings": "CityLearn heating_demand (demand mode, no RC)"},
              "t_cold_c": "10 degC constant (v1 convention)",
              "t_ground_c": "annual mean outdoor + 1 K", "time": "ASSUMPTION: CST from 2018-01-01"})


def _parse_time(xs) -> List[datetime]:
    out = []
    for x in xs:
        t = datetime.fromisoformat(str(x).replace("Z", "+00:00"))
        if t.tzinfo is None:
            raise ValueError(f"naive timestamp {x!r}: the format requires an explicit offset")
        out.append(t)
    return out


def _check_step(times: List[datetime], seconds: int, what: str) -> None:
    d = np.diff(np.array([t.timestamp() for t in times]))
    if len(d) and not np.all(d == seconds):
        bad = int(np.flatnonzero(d != seconds)[0])
        raise ValueError(f"{what}: interval {seconds} s expected, row {bad + 1} differs ({d[bad]} s)")


def load_package(path, *, houses: Optional[Sequence[str]] = None) -> DistrictData:
    root = Path(path)
    man = json.loads((root / "manifest.json").read_text())
    if man.get("format") != "stems-dhw-v2":
        raise ValueError("manifest.format must be 'stems-dhw-v2'")
    w = _read_csv_columns(root / man.get("weather", "weather.csv"))
    p = _read_csv_columns(root / man.get("price", "price.csv"))
    tw = _parse_time(w["time"])
    tp = _parse_time(p["time"])
    _check_step(tw, 3600, "weather.csv")
    if [t.timestamp() for t in tw] != [t.timestamp() for t in tp]:
        raise ValueError("weather.csv and price.csv time columns differ")
    T = len(tw)
    missing = [k for k in ("weather", "price", "draws", "buildings") if k not in man.get("sources", {})]
    if missing:
        raise ValueError(f"manifest.sources lacks {missing}")
    if (root / "cold_water.csv").is_file():
        cw = _read_csv_columns(root / "cold_water.csv")["T_cold"].astype(float)
    else:
        cw = np.full(T, float(man.get("t_cold_c", 8.0)))
    out_houses = []
    for hd in man["houses"]:
        if houses is not None and hd["id"] not in houses:
            continue
        bpath = root / hd["building"] if hd.get("building") else None
        bld = json.loads(bpath.read_text()) if bpath else None
        d = _read_csv_columns(root / hd["draws"])
        td = _parse_time(d["time"])
        if td[0].timestamp() != tw[0].timestamp():
            raise ValueError(f"{hd['draws']}: does not start at the weather start")
        step = int(round(td[1].timestamp() - td[0].timestamp())) if len(td) > 1 else 3600
        if step % 60 or (step // 60) not in ALLOWED_DT_MIN:
            raise ValueError(f"{hd['draws']}: interval {step} s not allowed")
        _check_step(td, step, hd["draws"])
        dt_min = step // 60
        n = T * 60 // dt_min
        if len(td) != n:
            raise ValueError(f"{hd['draws']}: {len(td)} rows, {n} expected")
        k = 60 // dt_min
        tc_sub = np.repeat(cw, k)
        if "energy_kwh" in d:
            e = d["energy_kwh"].astype(float)
        elif "volume_l" in d:
            t_draw = d["t_draw_c"].astype(float) if "t_draw_c" in d else \
                np.full(n, float((bld or {}).get("t_draw_c", 40.0)))
            e = d["volume_l"].astype(float) * CV * np.maximum(t_draw - tc_sub, 0.0)
        else:
            raise ValueError(f"{hd['draws']}: needs volume_l or energy_kwh")
        if np.any(e < 0):
            raise ValueError(f"{hd['draws']}: negative draw")
        out_houses.append(HouseData(id=hd["id"], draw_kwh=e, draw_dt_min=dt_min, t_cold_c=cw.copy(),
                                    building=bld, meta={"draws": hd["draws"]}))
    loc = [t for t in tw]
    hod = np.array([t.hour for t in loc])
    dow = np.array([t.weekday() for t in loc])
    time_utc = np.array([t.astimezone(timezone.utc).replace(tzinfo=None) for t in tw], dtype="datetime64[s]")
    t_out = w["T_out"].astype(float)
    tg = float(man["t_ground_c"]) if "t_ground_c" in man else float(np.mean(t_out) + 1.0)
    return DistrictData(name=man.get("name", root.name), source=f"package:{root}", time_utc=time_utc,
                        hour_of_day=hod, day_of_week=dow, t_out_c=t_out,
                        ghi_w_m2=np.maximum(w["GHI"].astype(float), 0.0), price=p["price"].astype(float),
                        currency=man.get("currency", "SEK"), t_ground_c=tg, houses=out_houses,
                        meta={k: man[k] for k in ("sources", "price_includes", "location") if k in man})


PRICE_AREAS = ("SE1", "SE2", "SE3", "SE4")
from stems.paths import SWEDEN_RAW as _SWEDEN_RAW
PRICE_DIRS = (_SWEDEN_RAW / "price",)
_SPOT_CACHE: Dict[str, tuple] = {}


def load_area_spot(area: str, price_dir=None):
    if area not in PRICE_AREAS:
        raise ValueError(f"unknown price area {area!r}; one of {PRICE_AREAS}")
    dirs = (Path(price_dir),) if price_dir is not None else PRICE_DIRS
    for d in dirs:
        f = d / f"price_{area}.csv"
        if f.is_file():
            break
    else:
        raise FileNotFoundError(f"price_{area}.csv not found in {[str(d) for d in dirs]}")
    key = str(f)
    if key not in _SPOT_CACHE:
        with open(f, newline="") as fh:
            rd = csv.reader(fh)
            head = next(rd)
            it, isp = head.index("time_utc"), head.index("spot_SEK_per_kWh")
            rows = [(r[it], r[isp]) for r in rd if r]
        t = np.array([r[0].rstrip("Z") for r in rows], dtype="datetime64[s]")
        sp = np.array([float(r[1]) if r[1] not in ("", "nan", "NaN") else np.nan for r in rows])
        bad = np.isnan(sp)
        if bad.any():
            idx = np.where(~bad, np.arange(len(sp)), -1)
            last = np.maximum.accumulate(idx)
            first_ok = int(np.argmax(~bad))
            sp = sp[np.where(last >= 0, last, first_ok)]
        _SPOT_CACHE[key] = (t, sp, bad)
    return _SPOT_CACHE[key]


def apply_price_area(data: DistrictData, area: str, *, own_area: Optional[str] = None, vat: float = 0.25,
                     price_dir=None) -> DistrictData:
    own = own_area or (data.meta.get("location") or {}).get("bidding_zone")
    if own is None:
        raise ValueError("the data has no location.bidding_zone; pass own_area")
    out = DistrictData(data.name, data.source, data.time_utc, data.hour_of_day, data.day_of_week,
                       data.t_out_c, data.ghi_w_m2, data.price.copy(), data.currency, data.t_ground_c,
                       data.houses, dict(data.meta), parent=data.parent, parent_offset=data.parent_offset)
    filled = 0
    if area != own:
        t = np.asarray(data.time_utc, dtype="datetime64[s]")
        sp = []
        for a in (area, own):
            ta, sa, bad = load_area_spot(a, price_dir)
            i = np.searchsorted(ta, t)
            if np.any(i >= len(ta)) or np.any(ta[np.minimum(i, len(ta) - 1)] != t):
                raise ValueError(f"price_{a}.csv does not cover the data's hours")
            sp.append(sa[i])
            filled += int(bad[i].sum())
        out.price = data.price + (1.0 + vat) * (sp[0] - sp[1])
    out.meta["price_area"] = {"area": area, "own_area": own, "filled_spot_hours": filled,
                              "rule": "price + (1+VAT)*(spot_area - spot_own); site tariff and tax kept"}
    out.name = f"{data.name}@{area}"
    return out
