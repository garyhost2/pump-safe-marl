import json

import pandas as pd

from data_pipeline.common import SWEDEN_RAW, log

SRC = {
    "SGU_S1616": "SGU (publication year not verified here) 'Geoenergi - Geologisk information för geoenergianläggningar', SGU-rapport, https://resource.sgu.se/produkter/sgurapp/s1616-rapport.pdf : ground temperature down to some tens of metres normally ~3-10 C in Sweden; below that it rises 15-30 C/km; energy-well depth usually 100-300 m",
    "SGU_geoinfo": "SGU document 'Geologisk information som stöd till utförande av ...' (keep.eu project attachment 62310): granite usually 3-4 W/mK, gabbro 2-2.5 W/mK; mean 3.3 W/mK from SGU mineral-composition calculations; ground temperature ~9-10 C in the south to <3 C in the north; in southern Sweden groundwater temperature at tens of metres ~ annual mean outdoor temperature",
    "Geoguiden": "Borrföretagen, Geoguiden (2025), https://borrforetagen.se/wp-content/uploads/2025/04/Geoguiden.pdf : thermal conductivity of Swedish bedrock usually 2-4 W/mK (6,500 rock samples)",
    "SGU_brunnsarkiv_via_brunnskartan": "brunnskartan.se statistics computed from SGU Brunnsarkivet (updated 2026-09-16): 522,228 energy wells; median depth of energy wells drilled 2021-2026 = 200 m, half between 170 and 240 m (secondary aggregation of SGU data, not an SGU publication)",
}


def outputs():
    return ["ground/ground_sites.json"]


def run(pkg=SWEDEN_RAW):
    (pkg / "ground").mkdir(parents=True, exist_ok=True)
    out = {}
    for site in ["karlstad", "lulea"]:
        T = pd.read_csv(f"{pkg}/weather/weather_{site}.csv").T_out_C
        T0 = round(float(T.mean()), 2)
        out[site] = dict(
            site=site,
            undisturbed_ground_temperature_near_surface_C=T0,
            undisturbed_ground_temperature_method="annual mean of weather_<site>.csv T_out_C 2020-2025 (SGU simplification: groundwater temperature at tens of metres ~ annual mean air temperature; stated for southern Sweden, used here for both sites)",
            geothermal_gradient_K_per_km=dict(low=15, high=30, used=22.5, source="SGU_S1616"),
            mean_borehole_temperature_for_200m_C=round(T0 + 0.0225 * 100, 2),
            mean_borehole_temperature_note="T0 + gradient x H/2 with H = 200 m and 22.5 K/km (mid of SGU range); range 15-30 K/km gives T0+1.5 .. T0+3.0 K",
            bedrock_thermal_conductivity_W_per_mK=dict(value=3.3, low=2.0, high=4.0, granite_typical=[3.0, 4.0],
                                                      source="SGU_geoinfo (mean 3.3) and Geoguiden (2-4)", site_specific=False),
            volumetric_heat_capacity_J_per_m3K=dict(value=None, note="not retrieved from a verified source; set in the g-function model and declare as an assumption"),
            typical_borehole_depth_m=dict(range=[100, 300], median_recent_energy_wells=200, iqr_recent=[170, 240],
                                          sources=["SGU_S1616", "SGU_brunnsarkiv_via_brunnskartan"],
                                          note="national statistic for all energy wells (mostly single-family GSHP), not sized for a specific 6-12 kW unit"),
        )
    out["sources"] = SRC
    json.dump(out, open(f"{pkg}/ground/ground_sites.json", "w"), indent=1, ensure_ascii=False)
    log("ground", {s: v["undisturbed_ground_temperature_near_surface_C"] for s, v in out.items() if s != "sources"})
