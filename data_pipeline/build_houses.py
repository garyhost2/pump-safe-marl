import io
import json
import math
import zipfile

import numpy as np
import pandas as pd

from data_pipeline.common import SOURCES, SWEDEN_RAW, log

RHO_C_AIR = 1200.0
C_M_TABULA = 45.0
T_IN_DESIGN = 21.0
SITES = {"karlstad": dict(dvut_name="Karlstad", zone="SE3"), "lulea": dict(dvut_name="Luleå", zone="SE1")}
SELECTION_RULE = (
    "surveyed houses are selected (no synthetic averaging): per era, the house closest (relative\n"
    "squared distance) to the weighted era quantile of UA_tb/Atemp and the weighted era median Atemp, among\n"
    "houses with a valid PFT ventilation measurement, measured indoor temperature 15-27 C, Atemp 90-250 m2\n"
    "and a ventilation system without heat recovery (self-draught, exhaust F, exhaust-air heat pump FVP, FT).\n"
    "Allocation: -1960: P50; 1961-75: P25, P75; 1976-85: P25, P75; 1986-95: P50; 1996-2005: P25, P75."
)


def outputs():
    return ["houses/houses.json", "houses/betsi_era_statistics.json", "houses/cold_water_karlstad.csv",
            "houses/cold_water_karlstad_params.json", "houses/cold_water_lulea.csv", "houses/cold_water_lulea_params.json"]

def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", ".").str.replace("\xa0", "").str.strip(), errors="coerce")


def rd(z, n):
    b = z.read("betsi/" + n)
    for enc in ["utf-8-sig", "cp1252", "latin1"]:
        try:
            s = b.decode(enc); break
        except UnicodeDecodeError:
            pass
    sep = ";" if s[:2000].count(";") > s[:2000].count(",") else ","
    return pd.read_csv(io.StringIO(s), sep=sep, low_memory=False)


def wq(v, w, q):
    v = np.asarray(v, float); w = np.asarray(w, float); ok = ~np.isnan(v) & ~np.isnan(w)
    v, w = v[ok], w[ok]; i = np.argsort(v); v, w = v[i], w[i]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, c, v))


def load_betsi():
    z = zipfile.ZipFile(SOURCES / "betsi-v2.zip")
    K, A, L, F, X, O, U, FE = [rd(z, n) for n in ["FB_Klimat.csv", "FA.csv", "LUFTKVALITET.csv", "FF.csv", "FX.csv",
                                                  "FB_OVRIGT.csv", "Uppräkningstal.csv", "FE.csv"]]
    cols = {"Byggnad": "b", "Totalt UA-värde": "UA", "Totalt UA-värde inkl köldbryggor": "UA_tb", "Total omslutningsarea": "A_env",
            "Total fönsterarea": "A_win", "FB97 UA-värde fönster inkl köldbryggor": "UA_win_tb", "FB48 UA-värde dörrar inkl köldbryggor": "UA_door_tb",
            "FB48 Summa dörrarea": "A_door", "FB88 Summa fönsterarea N": "A_win_N", "FB91 Summa fönsterarea S": "A_win_S",
            "FB94 Summa fönsterarea V": "A_win_W", "FB97 Summa fönsterarea O": "A_win_E", "Total ytterväggsarea": "A_wall",
            "FB52 UA-värde ytterväggar inkl köldbryggor": "UA_wall_tb", "FB56 UA-värde horizontellt vindsbjälklag inkl köldbryggor": "UA_attic_tb",
            "FB62 UA-värde snedtak inkl köldbryggor": "UA_sloped_roof_tb", "FB33 UA-värde platta på mark/källargolv inkl köldbryggor": "UA_slab_tb",
            "FB30 UA-värde krypgrundsbjälklag inkl köldbryggor": "UA_crawl_tb", "FB41 UA-värde källarbjälklag inkl köldbryggor": "UA_cellar_floor_tb",
            "FB47 UA-värde källarväggar under mark inkl köldbryggor": "UA_cellar_wall_below_tb", "FB47 UA-värde källarväggar ovan mark inkl köldbryggor": "UA_cellar_wall_above_tb",
            "FB58 UA-värde stödbensväggar inkl köldbryggor": "UA_kneewall_tb"}
    S = K[K.Hustyp == "S"][list(cols)].rename(columns=cols)
    for c in S.columns[1:]:
        S[c] = num(S[c])
    a = A[["Byggnad", "F14ATempt", "FA16AntalPlanOvanMark"]].rename(columns={"Byggnad": "b", "F14ATempt": "Atemp", "FA16AntalPlanOvanMark": "floors_above_ground"})
    for c in ["Atemp", "floors_above_ground"]:
        a[c] = num(a[c])
    u = U[["Byggnad", "ald", "Byggnader_totalvikt"]].rename(columns={"Byggnad": "b", "ald": "era", "Byggnader_totalvikt": "w"}); u["w"] = num(u.w)
    x = X[["Byggnad", "FXByggår"]].rename(columns={"Byggnad": "b", "FXByggår": "construction_year"}); x["construction_year"] = num(x.construction_year)
    l = L[["Byggnad", "LGH 1_bostadsvolym m3", "LGH 1_Totalventilations-flöde m3 h", "LGH 1_specifikt flöde h-1 medel för bostad", "LGH 1_medel-temp °C bostad"]].copy()
    l.columns = ["b", "dwelling_volume_m3", "vent_flow_m3h", "air_change_h", "T_in_measured_C"]
    for c in l.columns[1:]:
        l[c] = num(l[c])
    f = F[["Byggnad", "FF1 Typ av ventilationssystem"]].rename(columns={"Byggnad": "b", "FF1 Typ av ventilationssystem": "ventilation_system"})
    o = O[["Byggnad", "FB7 Grundkonstruktion"]].rename(columns={"Byggnad": "b", "FB7 Grundkonstruktion": "foundation"})
    fe = FE[["Byggnad", "FE27 Temperatur i varmvattenberedare", "FE37 Temperatur i toppen av tanken", "FE37 Temperatur i botten av tanken"]].copy()
    fe.columns = ["b", "T_heater_setting_C", "T_tank_top_C", "T_tank_bottom_C"]
    for c in fe.columns[1:]:
        fe[c] = num(fe[c])
    D = (S.merge(a, on="b", how="left").merge(u, on="b", how="left").merge(x, on="b", how="left").merge(l.drop_duplicates("b"), on="b", how="left")
         .merge(f, on="b", how="left").merge(o, on="b", how="left").merge(fe, on="b", how="left"))
    D["UA_per_Atemp"] = D.UA_tb / D.Atemp
    return D


def burch_christensen(T_hourly):
    m = T_hourly.groupby(T_hourly.index.month).mean()
    Tavg_F = T_hourly.mean() * 9 / 5 + 32
    dT_F = (m.max() - m.min()) * 9 / 5
    ratio = 0.4 + 0.01 * (Tavg_F - 44)
    lag = 35 - 1.0 * (Tavg_F - 44)
    mid = [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    out = []
    for mo, day in enumerate(mid, start=1):
        TF = (Tavg_F + 6) + ratio * (dT_F / 2) * math.sin(math.radians(0.986 * (day - 15 - lag) - 90))
        out.append(dict(month=mo, T_cold_model_C=round((TF - 32) * 5 / 9, 2), T_air_monthly_mean_C=round(float(m.loc[mo]), 2)))
    return out, dict(T_air_annual_mean_C=round(float(T_hourly.mean()), 2), dT_monthly_max_K=round(float(m.max() - m.min()), 2),
                     ratio=round(ratio, 4), lag_days=round(lag, 2))


def run(pkg=SWEDEN_RAW):
    (pkg / "houses").mkdir(parents=True, exist_ok=True)
    D = load_betsi()
    era_names = {"1 -60": "-1960", "2 61-75": "1961-1975", "3 76-85": "1976-1985", "4 86-95": "1986-1995", "5 96-05": "1996-2005"}
    stats = {}
    for e, g in D.groupby("era"):
        gv = g[g.vent_flow_m3h > 0]; gt = g[g.T_in_measured_C.between(15, 27)]
        stats[era_names[e]] = dict(
            n_houses=int(len(g)), stock_weight_sum=round(float(g.w.sum())),
            Atemp_m2_P25_P50_P75=[round(wq(g.Atemp, g.w, q), 1) for q in (.25, .5, .75)],
            UA_tb_W_per_K_P25_P50_P75=[round(wq(g.UA_tb, g.w, q), 1) for q in (.25, .5, .75)],
            UA_tb_per_Atemp_W_per_m2K_P25_P50_P75=[round(wq(g.UA_per_Atemp, g.w, q), 3) for q in (.25, .5, .75)],
            vent_flow_m3h_P50=round(wq(gv.vent_flow_m3h, gv.w, .5), 1), n_vent_measured=int(len(gv)),
            air_change_h_P50=round(wq(gv.air_change_h, gv.w, .5), 3),
            T_in_measured_C_P50=round(wq(gt.T_in_measured_C, gt.w, .5), 2),
            ventilation_system_weighted_share={k: round(float(v), 3) for k, v in (g.groupby("ventilation_system").w.sum() / g.w.sum()).items()})
    tank = D[D.T_tank_top_C.between(20, 95)]
    tank_ctx = dict(n=int(len(tank)), T_tank_top_C_P10_P50_P90=[round(wq(tank.T_tank_top_C, tank.w, q), 1) for q in (.1, .5, .9)],
                    share_top_below_60C=round(float(tank.w[tank.T_tank_top_C < 60].sum() / tank.w.sum()), 3),
                    note="BETSI FE37: measured temperature at the top of the hot-water tank in surveyed smahus (weighted). Context only.")
    alloc = {"1 -60": [.5], "2 61-75": [.25, .75], "3 76-85": [.25, .75], "4 86-95": [.5], "5 96-05": [.25, .75]}
    cand = D[(D.vent_flow_m3h > 0) & D.T_in_measured_C.between(15, 27) & D.Atemp.between(90, 250)
             & D.ventilation_system.isin(["Självdrag", "F-system", "FVP-system (värmepump)", "FT-system"])]
    picks = []
    for e, qs in alloc.items():
        g = D[D.era == e]; c = cand[cand.era == e]; a50 = wq(g.Atemp, g.w, .5)
        for q in qs:
            tgt = wq(g.UA_per_Atemp, g.w, q)
            dist = ((c.UA_per_Atemp - tgt) / tgt) ** 2 + ((c.Atemp - a50) / a50) ** 2
            k = dist.idxmin(); picks.append((e, q, k)); c = c.drop(k)
    dv = pd.read_csv(SOURCES / "dvut_1991_2020.csv", sep=";", skiprows=1, encoding="utf-8-sig")
    weather = {s: pd.read_csv(f"{pkg}/weather/weather_{s}.csv", parse_dates=["time_utc"]).set_index("time_utc").T_out_C for s in SITES}
    houses = []
    vent_en = {"Självdrag": "natural (self-draught)", "F-system": "mechanical exhaust (F)", "FVP-system (värmepump)": "exhaust with exhaust-air heat pump (FVP)", "FT-system": "mechanical supply+exhaust, no recovery (FT)"}
    found_en = {"Platta på mark": "slab on ground", "Krypgrund": "crawl space", "Källare": "basement"}
    for i, (e, q, k) in enumerate(picks, start=1):
        r = D.loc[k]
        H_ve = RHO_C_AIR * r.vent_flow_m3h / 3600.0
        H_tot = r.UA_tb + H_ve
        C = C_M_TABULA * 3600.0 * r.Atemp
        tau_h = C / H_tot / 3600.0
        tau_d = int(min(30, max(1, round(tau_h / 24))))
        Htr_w = float(np.nansum([r.UA_win_tb, r.UA_door_tb]))
        site_vals = {}
        for s, c in SITES.items():
            row = dv[dv.Ort == c["dvut_name"]].iloc[0]
            dvut = float(str(row[f"{tau_d}-dygn"]).replace(",", "."))
            site_vals[s] = dict(DVUT_C=dvut, design_heat_load_kW=round(H_tot * (T_IN_DESIGN - dvut) / 1000, 2),
                                heat_loss_degree_hours_kWh_per_year_mean_2020_2025=round(float(
                                    (H_tot * np.clip(T_IN_DESIGN - weather[s], 0, None)).sum() / 1000 / 6), 0))
        houses.append(dict(
            house=i, draws_file_pattern=f"draws/draws_house{i}_<year>.csv",
            betsi_building_id=r.b, era=era_names[e], era_quantile_target_UA_per_Atemp=q, construction_year=int(r.construction_year),
            Atemp_m2=float(r.Atemp), floors_above_ground=None if pd.isna(r.floors_above_ground) else int(r.floors_above_ground),
            foundation=found_en.get(r.foundation, r.foundation), ventilation_system=vent_en.get(r.ventilation_system, r.ventilation_system),
            UA_total_W_per_K=round(float(r.UA_tb), 2), UA_total_excl_thermal_bridges_W_per_K=round(float(r.UA), 2),
            UA_per_Atemp_W_per_m2K=round(float(r.UA_per_Atemp), 4),
            UA_components_W_per_K={k2: (None if pd.isna(r[k2]) else round(float(r[k2]), 2)) for k2 in
                                   ["UA_wall_tb", "UA_win_tb", "UA_door_tb", "UA_attic_tb", "UA_sloped_roof_tb", "UA_kneewall_tb", "UA_slab_tb", "UA_crawl_tb",
                                    "UA_cellar_floor_tb", "UA_cellar_wall_below_tb", "UA_cellar_wall_above_tb"]},
            envelope_area_m2=float(r.A_env), window_area_m2=float(r.A_win),
            window_area_by_orientation_m2=dict(N=float(r.A_win_N), E=float(r.A_win_E), S=float(r.A_win_S), W=float(r.A_win_W)),
            ventilation_flow_measured_m3h=float(r.vent_flow_m3h), air_change_measured_h=None if pd.isna(r.air_change_h) else float(r.air_change_h),
            H_ve_W_per_K=round(H_ve, 2), H_total_W_per_K=round(H_tot, 2), T_in_measured_C=float(r.T_in_measured_C),
            C_thermal_J_per_K=round(C, 0), time_constant_h=round(tau_h, 1), time_constant_d_for_DVUT=tau_d,
            iso13790_5R1C=dict(H_tr_w_W_per_K=round(Htr_w, 2), H_tr_op_W_per_K=round(float(r.UA_tb) - Htr_w, 2), H_ve_W_per_K=round(H_ve, 2),
                               A_m_m2=round(2.5 * float(r.Atemp), 1), C_m_J_per_K=round(C, 0), A_f_m2=float(r.Atemp)),
            sites=site_vals))
    for s in SITES:
        rows, par = burch_christensen(weather[s])
        cw = pd.DataFrame(rows); cw["T_cold_used_C"] = cw.T_cold_model_C.clip(lower=2.0)
        cw.to_csv(f"{pkg}/houses/cold_water_{s}.csv", index=False)
        par["clip_lower_C"] = 2.0
        json.dump(par, open(f"{pkg}/houses/cold_water_{s}_params.json", "w"), indent=1)
    meta = dict(
        source="Boverket BETSI open database v2 (betsi-v2.zip), https://www.boverket.se/sv/om-boverket/oppna-data/betsi-oppna-data/databasen-fran-betsi/ ; cite 'Boverket, BETSI'",
        selection_rule=SELECTION_RULE,
        conventions=dict(UA_total="BETSI 'Totalt UA-värde inkl köldbryggor' (= 1.20 x element UA sum in every BETSI house: flat +20 % thermal-bridge allowance)",
                         H_ve="rho_a*c_a*q_measured/3600 with rho_a*c_a = 1200 J/(m3K) (ISO 13790); q = BETSI passive-tracer (PFT) total outdoor-air flow, includes infiltration",
                         C_thermal="TABULA standard internal heat capacity 45 Wh/(m2K) x Atemp (Loga & Diefenbach 2013, TABULA calculation method); ISO 13790 Table 12 classes light 110 / medium 165 kJ/(m2K) give a sensitivity range",
                         design_heat_load=f"H_total x ({T_IN_DESIGN} C - DVUT); DVUT from Boverket/SMHI DVUT 1991-2020 at the column = time constant in days; no internal or solar gains; no DHW",
                         heat_loss_degree_hours="H_total x sum(max(0, 21 C - T_out)) / 6 years from weather/weather_<site>.csv 2020-2025; upper bound on space-heating need (no gains)",
                         iso13790_5R1C="H_tr_w = windows + doors (incl thermal bridges); H_tr_op = remainder of UA_total; A_m = 2.5 A_f (ISO 13790 Table 12 'medium')",
                         FVP_note="Houses with an exhaust-air heat pump (FVP) are listed with that system; if modelled with a ground-source heat pump the exhaust flow stays, its heat recovery does not"),
        T_in_design_C=T_IN_DESIGN, houses=houses, betsi_dhw_tank_context=tank_ctx)
    json.dump(meta, open(f"{pkg}/houses/houses.json", "w"), indent=1, ensure_ascii=False)
    json.dump(stats, open(f"{pkg}/houses/betsi_era_statistics.json", "w"), indent=1, ensure_ascii=False)
    for h in houses:
        log("house", h["house"], h["betsi_building_id"], h["era"], h["Atemp_m2"], h["H_total_W_per_K"], h["time_constant_h"])
