import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from data_pipeline.common import SWEDEN_PACKAGES, SWEDEN_RAW, log

SITES = {"karlstad": {"zone": "SE3", "lat": 59.40, "lon": 13.50, "name": "Karlstad"},
         "lulea": {"zone": "SE1", "lat": 65.58, "lon": 22.15, "name": "Lulea"}}
YEARS = (2022, 2023, 2024, 2025)
TZ = ZoneInfo("Europe/Stockholm")
RHO_CP = 0.98 * 4.18 / 3600.0
SOLAR_APERTURE_FRAC = 0.2
INTERNAL_GAINS_W_M2 = 3.0
CURVE = {"curve_c": 40.0, "T_supply_min_c": 20.0, "T_supply_max_c": 55.0, "seasonal_stop_c": 17.0}
RXT12_MAX_KW = 12.0


def _read(path):
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        rows = list(rd)
    return {k: [r[k] for r in rows] for k in rows[0]}


def _utc(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _local_iso(t_utc):
    return t_utc.astimezone(TZ).isoformat()


def _year_slice(times, year):
    a = [i for i, t in enumerate(times) if t.year == year]
    return a[0], a[-1] + 1


def building_json(h, site):
    p = h["iso13790_5R1C"]
    win = float(h.get("window_area_m2", 0.0))
    s = h["sites"][site]
    return {
        "model": "5R1C",
        "source": (f"Boverket BETSI {h['betsi_building_id']} (era {h['era']}, built {h['construction_year']}) "
                   f"via stems_sweden_data_v1 houses.json iso13790_5R1C"),
        "A_f_m2": float(p["A_f_m2"]), "H_ve_W_K": float(p["H_ve_W_per_K"]),
        "H_tr_w_W_K": float(p["H_tr_w_W_per_K"]), "H_tr_op_W_K": float(p["H_tr_op_W_per_K"]),
        "C_m_J_K": float(p["C_m_J_per_K"]), "A_m_m2": float(p["A_m_m2"]),
        "solar_aperture_m2": round(SOLAR_APERTURE_FRAC * win, 3),
        "internal_gains_W_m2": INTERNAL_GAINS_W_M2,
        "T_set_c": 21.0, "T_comfort_min_c": 20.0, "heating_curve": dict(CURVE), "t_draw_c": 40.0,
        "occupants": h.get("occupants"),
        "H_total_W_K": float(h["H_total_W_per_K"]), "DVUT_C": float(s["DVUT_C"]),
        "design_heat_load_kW": float(s["design_heat_load_kW"]),
        "ventilation_system": h.get("ventilation_system"),
        "assumptions": {"solar_aperture_m2": f"{SOLAR_APERTURE_FRAC} x window_area_m2 (not in the package)",
                        "internal_gains_W_m2": "dhw-v2 default, not in the package",
                        "heating_curve": "Thermia factory curve 40, max 55 degC, seasonal stop 17 degC",
                        "FVP": "exhaust-air heat-pump heat recovery not modelled (package limitation 3)"},
    }


def convert(src: Path, out: Path):
    src, out = Path(src), Path(out)
    houses = json.loads((src / "houses" / "houses.json").read_text())["houses"]
    ground = json.loads((src / "ground" / "ground_sites.json").read_text())
    occ = {}
    meta = json.loads((src / "draws" / "draws_meta.json").read_text())
    for hm in (meta.get("houses") or []):
        if isinstance(hm, dict) and "house" in hm:
            occ[int(hm["house"])] = hm.get("occupants")
    made = []
    for site, info in SITES.items():
        w = _read(src / "weather" / f"weather_{site}.csv")
        p = _read(src / "price" / f"price_{info['zone']}.csv")
        cw = _read(src / "houses" / f"cold_water_{site}.csv")
        tcold_m = {int(m): float(v) for m, v in zip(cw["month"], cw["T_cold_used_C"])}
        tw = [_utc(x) for x in w["time_utc"]]
        tp = [_utc(x) for x in p["time_utc"]]
        g = ground[site]
        bore = {"H_m": 200.0, "k_s_w_mk": float(g["bedrock_thermal_conductivity_W_per_mK"]["value"])}
        for year in YEARS:
            name = f"{site}_{year}"
            d = out / name
            (d / "draws").mkdir(parents=True, exist_ok=True)
            (d / "buildings").mkdir(parents=True, exist_ok=True)
            a, b = _year_slice(tw, year)
            pa, pb = _year_slice(tp, year)
            times = tw[a:b]
            assert [t.timestamp() for t in times] == [t.timestamp() for t in tp[pa:pb]], name
            T = len(times)
            loc = [_local_iso(t) for t in times]
            tcold = np.array([tcold_m[t.month] for t in times])
            with open(d / "weather.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["time", "T_out", "GHI"])
                for i in range(T):
                    wr.writerow([loc[i], w["T_out_C"][a + i], w["GHI_Wm2"][a + i]])
            with open(d / "price.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["time", "price", "spot"])
                for i in range(T):
                    wr.writerow([loc[i], p["retail_total_SEK_per_kWh"][pa + i], p["spot_SEK_per_kWh"][pa + i]])
            with open(d / "cold_water.csv", "w", newline="") as fh:
                wr = csv.writer(fh)
                wr.writerow(["time", "T_cold"])
                for i in range(T):
                    wr.writerow([loc[i], f"{tcold[i]:.2f}"])
            hl = []
            for h in houses:
                hid = f"H{int(h['house']):02d}"
                dr = _read(src / "draws" / f"draws_house{int(h['house'])}_{year}.csv")
                td = [_utc(x) for x in dr["time_utc"]]
                assert td[0] == times[0] and len(td) == T * 12, (name, hid, len(td), T)
                lit = np.array(dr["litres_40C"], float)
                tc5 = np.repeat(tcold, 12)
                e = lit * RHO_CP * np.maximum(40.0 - tc5, 0.0)
                with open(d / "draws" / f"{hid}.csv", "w", newline="") as fh:
                    wr = csv.writer(fh)
                    wr.writerow(["time", "energy_kwh", "volume_l"])
                    for i in range(len(td)):
                        wr.writerow([_local_iso(td[i]), f"{e[i]:.6g}", dr["litres_40C"][i]])
                bj = building_json(h, site)
                bj["occupants"] = bj["occupants"] or occ.get(int(h["house"]))
                (d / "buildings" / f"{hid}.json").write_text(json.dumps(bj, indent=1))
                hl.append({"id": hid, "draws": f"draws/{hid}.csv", "building": f"buildings/{hid}.json",
                           "betsi_id": h["betsi_building_id"]})
            man = {
                "format": "stems-dhw-v2", "version": 1, "name": name,
                "location": {"name": info["name"], "lat": info["lat"], "lon": info["lon"],
                             "bidding_zone": info["zone"]},
                "currency": "SEK",
                "price_includes": ("retail_total = 1.25 x (spot + supplier markup + network energy (Vattenfall T4 2025 "
                                   "list, all years) + energy tax); VAT incl.; fixed charges and power tariffs excluded"),
                "t_ground_c": float(g["mean_borehole_temperature_for_200m_C"]),
                "borehole": bore,
                "houses": hl,
                "sources": {"weather": "SMHI open data (CC BY 4.0 SE), stems_sweden_data_v1 weather_%s.csv" % site,
                            "price": "elprisetjustnu.se day-ahead + tariff composition, stems_sweden_data_v1 price_%s.csv" % info["zone"],
                            "draws": "OpenDHW 0.2.9 synthetic 5-min draws, stems_sweden_data_v1 draws_house<i>_%d.csv" % year,
                            "buildings": "Boverket BETSI (8 real houses), ISO 13790 5R1C, stems_sweden_data_v1 houses.json",
                            "cold_water": "Burch & Christensen (2007) monthly correlation, cold_water_%s.csv" % site,
                            "ground": "ground_sites.json (SGU national values; not site-specific conductivity)"},
                "conversion": {"script": "experiments/dhw_v2_matrix/convert_sweden.py", "year": "UTC calendar year",
                               "time": "written in Europe/Stockholm offset (local hour/day for controllers)",
                               "draw_energy": "litres_40C x 0.98 x 4.18/3600 x (40 - T_cold(month))",
                               "solar_aperture_frac": SOLAR_APERTURE_FRAC, "internal_gains_W_m2": INTERNAL_GAINS_W_M2,
                               "heating_curve": CURVE,
                               "t_ground_c": "mean_borehole_temperature_for_200m_C",
                               "borehole_rho_c": "plant default 2.16 MJ/m3K (package has none)"},
                "hours": T,
            }
            (d / "manifest.json").write_text(json.dumps(man, indent=1))
            made.append(d)
            log("package", name, T, "h")
    return made


def validate(out: Path, src: Path):
    from stems.dhw_v2.building import make_building
    from stems.dhw_v2.data import load_package
    res = {}
    for d in sorted(p for p in Path(out).iterdir() if (p / "manifest.json").is_file()):
        man = json.loads((d / "manifest.json").read_text())
        site, year = d.name.split("_")
        data = load_package(d)
        r = {"hours": int(len(data.t_out_c)), "houses": [h.id for h in data.houses],
             "price_mean": float(np.mean(data.price)), "price_max": float(np.max(data.price)),
             "t_out_min": float(np.min(data.t_out_c)), "t_out_mean": float(np.mean(data.t_out_c)),
             "local_hour0": int(data.hour_of_day[0])}

        tc = np.array(_read(d / "cold_water.csv")["T_cold"], float)
        dr_rows = []
        for h in data.houses:
            k = int(h.id[1:])
            s = _read(src / "draws" / f"draws_house{k}_{year}.csv")
            lit = np.array(s["litres_40C"], float)
            ref = lit * RHO_CP * np.maximum(40.0 - np.repeat(tc, 12), 0.0)
            dr_rows.append({"house": h.id, "draw_kwh": float(h.draw_kwh.sum()),
                            "max_abs_err_kwh": float(np.max(np.abs(h.draw_kwh - ref))),
                            "kwh_10to40_source": float(np.sum(np.array(s["kWh_10to40"], float)))})
        r["draws"] = dr_rows

        brows = []
        for h in data.houses:
            bj = dict(h.building)
            ss = dict(bj, internal_gains_W_m2=0.0, solar_aperture_m2=0.0)
            ss.pop("internal_gains_W", None)
            b = make_building(ss)
            for _ in range(2000):
                q = b.need_w(21.0, bj["DVUT_C"], 0.0)
                b.step(q, bj["DVUT_C"], 0.0)
            steady_kw = b.need_w(21.0, bj["DVUT_C"], 0.0) / 1000.0
            b2 = make_building(bj)
            need = np.zeros(len(data.t_out_c))
            for t in range(len(need)):
                q = b2.need_w(21.0, data.t_out_c[t], data.ghi_w_m2[t])
                need[t] = q / 1000.0
                b2.step(q, data.t_out_c[t], data.ghi_w_m2[t])
            brows.append({"house": h.id, "steady_dvut_kw": steady_kw, "design_kw": bj["design_heat_load_kW"],
                          "H_total_x_dT_kw": bj["H_total_W_K"] * (21.0 - bj["DVUT_C"]) / 1000.0,
                          "ideal_sh_kwh": float(need.sum()), "ideal_sh_peak_kw": float(need.max()),
                          "hours_over_12kw": int((need > RXT12_MAX_KW).sum())})
        r["buildings"] = brows
        res[d.name] = r
        log("validated", d.name)

    res["energy_balance"] = energy_balance(sorted(p for p in Path(out).iterdir() if p.name == "lulea_2024")[0])
    (Path(out) / "validation.json").write_text(json.dumps(res, indent=1))
    return res


def energy_balance(d: Path, hours: int = 336):
    from stems.dhw_v2.controllers import ReactiveController
    from stems.dhw_v2.data import load_package
    from stems.dhw_v2.env import DHWDistrictEnv, EnvConfig
    from stems.dhw_v2.tank import CV
    data = load_package(d)
    env = DHWDistrictEnv(data, EnvConfig(hours=hours))
    env.reset()
    ctl = ReactiveController(env.B)
    vol = np.array([t.volume_l for t in env._tanks])
    N = env.plant.T.shape[1]
    tc0 = float(data.houses[0].t_cold_c[0])

    def stored():
        return (vol[:, None] / N * CV * (env.plant.T - tc0)).sum(axis=1)
    e0 = stored()
    while True:
        _, _, term, _, info = env.step_command(ctl.act(env))
        ctl.update(info["record"])
        if term:
            break
    A = env.arrays()
    e1 = stored()
    heat, served, loss = A["heat_dhw_kwh"].sum(0), A["served_kwh"].sum(0), A["loss_kwh"].sum(0)
    resid = heat - served - loss - (e1 - e0)
    return {"package": d.name, "hours": hours, "t_cold_constant_in_window": bool(np.ptp(data.houses[0].t_cold_c[:hours]) == 0),
            "heat_dhw_kwh": heat.tolist(), "served_kwh": served.tolist(), "loss_kwh": loss.tolist(),
            "delta_stored_kwh": (e1 - e0).tolist(), "residual_kwh": resid.tolist(),
            "max_abs_residual_rel": float(np.max(np.abs(resid) / np.maximum(heat, 1e-9)))}


def outputs():
    files = []
    for site in SITES:
        for year in YEARS:
            d = f"{site}_{year}"
            files += [f"{d}/manifest.json", f"{d}/weather.csv", f"{d}/price.csv", f"{d}/cold_water.csv"]
            files += [f"{d}/draws/H{i:02d}.csv" for i in range(1, 9)]
            files += [f"{d}/buildings/H{i:02d}.json" for i in range(1, 9)]
    return files + ["validation.json"]


def run(src=SWEDEN_RAW, out=SWEDEN_PACKAGES, with_validation=True):
    convert(src, out)
    if with_validation:
        validate(out, Path(src))
