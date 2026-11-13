# STEMS Paper 2 — Swedish data package v1 (stems_sweden_data_v1)

Built 2026-10-26 for the hot-water / legionella paper (Thermia RXT 12 model of record). Sites: **Karlstad** (Thermia's home, bidding zone SE3) and **Luleå** (cold-climate sensitivity, SE1). No file on branch dhw-v2 existed (`docs/DHW_V2_DATA_FORMAT.md` not found: branch dhw-v2 does not exist), so the formats below are defined here.

## Coverage at a glance

| Item | Span | Resolution | Gaps |
|---|---|---|---|
| weather_karlstad.csv | 2020-01-01T00Z – 2025-12-31T23Z (6 years) | 1 h | T: 872 of 52,608 h missing (872 filled: 104 by ≤2 h interpolation, 768 by donor regression); GHI: 0 missing |
| weather_lulea.csv | 2020-01-01T00Z – 2025-12-31T23Z (6 years) | 1 h | T: 209 of 52,608 h missing (209 filled: 150 by ≤2 h interpolation, 59 by donor regression); GHI: 0 missing |
| price_SE3.csv | 2022-01-01T00Z – 2025-12-31T23Z (4 years) | 1 h | 0 missing; 15-min products averaged from 2025-09-30 22:00:00+00:00 |
| price_SE1.csv | 2022-01-01T00Z – 2025-12-31T23Z (4 years) | 1 h | 0 missing; 15-min products averaged from 2025-09-30 22:00:00+00:00 |
| draws_house<i>_<year>.csv | 8 houses × 2020–2025 (48 house-years) | 5 min (+1-min events, hourly) | none (synthetic) |
| houses/houses.json | 8 real BETSI houses, eras −1960 … 1996–2005 | static | no post-2005 house (BETSI ends 2005) |
| ground/ground_sites.json | both sites | static | national conductivity, not site-specific |

**Joint span for weather + price + draws: 2022–2025 (4 full UTC years).** Weather and draws additionally cover 2020–2021 (usable as held-out weather years with a price series from another source).

## File formats (all times UTC, ISO 8601 with `Z`; a row's timestamp is the START of its interval)

### weather/weather_<site>.csv
`time_utc, T_out_C, GHI_Wm2, T_flag, GHI_flag`
- `T_out_C`: SMHI parameter 1 (air temperature, instantaneous once per hour) at `time_utc`.
- `GHI_Wm2`: SMHI parameter 11 (global irradiance, hourly mean, station radiation network) averaged over `[time_utc, time_utc+1h)`. SMHI stamps the mean at the END of the hour; the package shifts it by −1 h. Check: after the shift the mean diurnal centroid is karlstad 11.137 h UTC vs solar noon 11.111 h, lulea 10.53 h UTC vs solar noon 10.525 h (equation-of-time averages out over the year).
- `T_flag`: 0 observed, 1 linear interpolation (gap ≤2 h), 2 donor-station regression, 3 linear fallback (not used). `GHI_flag`: 0 SMHI quality G, 1 SMHI quality Y (kept as delivered), 2 filled (none).

Gap filling of temperature. Gaps > 2 h are filled from the first nearby SMHI station that covers the whole gap, by linear regression `T = a + b·T_donor` fitted on overlapping observed hours within ±15 days. Karlstad gaps are mostly 11-h night blocks (19→06 UTC), where interpolation is poor. Validation on 300 synthetic 11-h night gaps per site (masked observed data):

| site | stations (T / GHI / donors) | donor RMSE K | donor bias K | linear-interp RMSE K | max abs donor error K |
|---|---|---|---|---|---|
| karlstad | Karlstad Flygplats 93220 / Karlstad Sol 93235 / Sunne A 93520, Arvika A 92410 | 1.488 | 0.003 | 2.634 | 6.9 |
| lulea | Lulea-Kallax Flygplats 162860 / Lulea Sol 162015 / Boden 161940, Rodkallen A 162790 | 1.597 | -0.137 | 2.151 | 7.55 |

Per-year counts (observed / ≤2 h interp / donor-filled hours; annual mean T; annual GHI):

| site | year | observed | interp | donor | T mean °C | T min °C | GHI kWh/m² | GHI quality-Y h |
|---|---|---|---|---|---|---|---|---|
| karlstad | 2020 | 8692 | 13 | 79 | 8.36 | -9.7 | 1032.6 | 0 |
| karlstad | 2021 | 8726 | 18 | 16 | 6.6 | -17.5 | 1023.9 | 28 |
| karlstad | 2022 | 8592 | 29 | 139 | 7.3 | -18.5 | 1069.0 | 40 |
| karlstad | 2023 | 8646 | 23 | 91 | 6.58 | -17.6 | 1045.2 | 0 |
| karlstad | 2024 | 8714 | 10 | 60 | 7.17 | -24.9 | 1003.1 | 25 |
| karlstad | 2025 | 8366 | 11 | 383 | 7.83 | -21.8 | 1014.7 | 1 |
| lulea | 2020 | 8756 | 28 | 0 | 4.96 | -21.4 | 943.0 | 4 |
| lulea | 2021 | 8698 | 43 | 19 | 2.69 | -29.1 | 929.5 | 3 |
| lulea | 2022 | 8734 | 26 | 0 | 3.65 | -27.2 | 927.9 | 7 |
| lulea | 2023 | 8706 | 28 | 26 | 2.88 | -24.3 | 935.6 | 1 |
| lulea | 2024 | 8766 | 15 | 3 | 3.7 | -36.4 | 934.6 | 3 |
| lulea | 2025 | 8739 | 10 | 11 | 4.53 | -24.5 | 926.8 | 1 |

### price/price_<zone>.csv
`time_utc, spot_SEK_per_kWh, network_energy_SEK_per_kWh_exVAT, energy_tax_SEK_per_kWh_exVAT, supplier_markup_SEK_per_kWh_exVAT, high_load_hour, retail_total_SEK_per_kWh`

`retail_total = 1.25 × (spot + supplier_markup + network_energy(t) + energy_tax(year))`, VAT included, per kWh only. Fixed charges are in `price/tariff_composition.json`, which lists every component with its source. `high_load_hour` is 1 for weekdays 06–22 local time in Jan–Mar and Nov–Dec, excluding Vattenfall's listed holidays. Energy tax follows the local calendar year; SE1/Luleå gets the northern deduction of 9.6 öre/kWh.

| zone | year | spot mean SEK/kWh | spot min | spot max | negative h | retail mean SEK/kWh | high-load h |
|---|---|---|---|---|---|---|---|
| SE3 | 2022 | 1.3788 | -0.0223 | 8.5605 | 29 | 2.6243 | 1696 |
| SE3 | 2023 | 0.5903 | -0.6911 | 3.7499 | 428 | 1.6779 | 1680 |
| SE3 | 2024 | 0.4088 | -0.6897 | 8.1571 | 652 | 1.4937 | 1632 |
| SE3 | 2025 | 0.5125 | -0.2785 | 4.5527 | 345 | 1.6366 | 1616 |
| SE1 | 2022 | 0.6339 | -0.0223 | 6.425 | 29 | 1.5733 | 1696 |
| SE1 | 2023 | 0.4563 | -0.6911 | 3.7499 | 433 | 1.3904 | 1680 |
| SE1 | 2024 | 0.2857 | -0.5715 | 5.8945 | 627 | 1.2198 | 1632 |
| SE1 | 2025 | 0.185 | -0.8172 | 3.0107 | 520 | 1.1071 | 1616 |

Provider artifact (harmless): on each autumn DST day the first local 02:00 interval carries a wrong end label (03:00+01:00); start labels are unique and complete; 4 such rows per zone. **Not covered: 2020–2021 spot prices.** No source with an open licence was available: elprisetjustnu starts 2021-11-01; Energy-Charts serves SE1/SE3 only for private/internal use; ENTSO-E needs a token, which we do not have; Nord Pool is licensed.

### draws/
- `draws_house<i>_<year>.csv` (5 min): `time_utc, litres_40C, kWh_10to40`. `litres_40C` = litres of mixed water at 40 °C delivered at the taps in the 5-min interval. `kWh_10to40` = litres × 0.98 kg/L × 4.18 kJ/(kg K) × 30 K / 3600 (OpenDHW's ρ and c_p; reference cold water 10 °C). For site energy use `houses/cold_water_<site>.csv`: kWh = litres × 0.98·4.18/3600 × (40 − T_cold).
- `draws_hourly_<year>.csv`: hourly litres_40C, one column per house.
- `draws_1min_events.csv.gz`: every non-zero 1-min step (`house, year, time_utc, litres_40C`), i.e. the native OpenDHW resolution.
- `draws_meta.json`: seeds (Python `random` and NumPy both seeded with `100000 + 1000·house + year`), off-days, targets, totals, and draw-offs placed vs generated (all placed in all 48 runs).

Generator: OpenDHW 0.2.9 (RWTH-EBC, MIT licence; Python port of DHWcalc, Uni Kassel / IEA SHC Task 26). DHWcalc defaults: 4 categories, 60 s step, SFH, weekend factor 1.2, ±10 % seasonal cosine, 1200 L/h maximum flow. OpenDHW hard-codes 365 days, so the two affected functions were ported with a day-count argument (`scripts/gen_draws.py`). The self-test reproduces `OpenDHW.generate_dhw_profile` to within 1.1e-13 L/h on every minute for two seeds; the only difference is float rounding in a unit conversion. Off days: Saturdays, Sundays, Swedish public holidays (python-holidays `SE`), plus midsommarafton, julafton and nyårsafton. Profiles are built on Europe/Stockholm local clock time and then mapped to UTC (the spring-forward hour is skipped, the fall-back hour repeated, and the last local hour of the year wraps to the first).

Volumes. Energimyndigheten measured hot and cold water in 44 households in 2007-2008, 35 småhus and 9 apartments (Energimyndigheten page 'Mätningar av varm- och kallvattenförbrukning', saved as `sources/energimyndigheten_vattenmatningar.html`; full report ER 2009:26 'Mätning av kall- och varmvattenanvändning i 44 hushåll'). The hearing slides (`sources/vattenmatningar_hearing-2009-09-30.pdf`, 'Del 3: 35 småhus + 9 lägenheter') give for the småhus: hot water 17-77 L/person/day (mean 42) and **energy for water heating 328-1524 kWh/person/yr (mean 781)**. 781 kWh/p/yr at 10→40 °C corresponds to ≈61 L/p/d of 40 °C water. Sveby Brukarindata gives 14 m³/person/yr for new småhus (≈800 kWh). The per-person volumes below are design choices bracketing that mean (48–72 L/p/d, i.e. ≈598–897 kWh/p/yr at 10→40 °C):

| house | occupants | L40/person/day | target L40/day | realised 2023 L40/day | kWh_10to40 2023 |
|---|---|---|---|---|---|
| 1 | 1 | 70.0 | 70.0 | 70.31 | 876.0 |
| 2 | 2 | 55.0 | 110.0 | 110.13 | 1372.2 |
| 3 | 2 | 66.0 | 132.0 | 132.29 | 1648.3 |
| 4 | 3 | 61.0 | 183.0 | 183.23 | 2283.1 |
| 5 | 4 | 50.0 | 200.0 | 200.44 | 2497.4 |
| 6 | 4 | 61.0 | 244.0 | 244.28 | 3043.7 |
| 7 | 5 | 48.0 | 240.0 | 240.25 | 2993.4 |
| 8 | 3 | 72.0 | 216.0 | 216.36 | 2695.8 |

### houses/
- `houses.json`: 8 real single-family houses from Boverket's BETSI survey database (826 småhus surveyed 2007–2008, stratified by era, survey-weighted). Per house: BETSI id, construction year, Atemp, UA incl. thermal bridges (total and by element), window areas by orientation, PFT-measured ventilation flow, H_ve, measured indoor temperature, thermal capacitance C (TABULA 45 Wh/(m²K)·Atemp), time constant, ISO 13790 5R1C parameters, and per-site DVUT, design heat load and an uncorrected heat-loss estimate.
- `betsi_era_statistics.json`: survey-weighted P25/P50/P75 by era (Atemp, UA, UA/Atemp, ventilation, indoor temperature, ventilation-system shares).
- `cold_water_<site>.csv`: monthly cold-water temperature from the Burch & Christensen (2007) mains correlation (EnergyPlus `Site:WaterMainsTemperature`, Correlation), fed with this package's 2020–2025 SMHI air temperatures; lower clip 2 °C; parameters in `cold_water_<site>_params.json`.

| house | BETSI id | era | year | Atemp m² | UA_tb W/K | H_ve W/K | H_total W/K | τ h | vent. system | DVUT Karlstad °C | P_design Karlstad kW | DVUT Luleå °C | P_design Luleå kW |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | SH1152 | -1960 | 1924 | 164.0 | 257.97 | 51.33 | 309.3 | 23.9 | mechanical exhaust (F) | -16.5 | 11.6 | -25.4 | 14.35 |
| 2 | SH2059 | 1961-1975 | 1973 | 153.0 | 148.8 | 11.67 | 160.47 | 42.9 | natural (self-draught) | -15.3 | 5.82 | -24.5 | 7.3 |
| 3 | SH2174 | 1961-1975 | 1965 | 138.0 | 233.04 | 29.33 | 262.37 | 23.7 | natural (self-draught) | -16.5 | 9.84 | -25.4 | 12.17 |
| 4 | SH3146 | 1976-1985 | 1978 | 140.0 | 120.3 | 14.67 | 134.96 | 46.7 | natural (self-draught) | -15.3 | 4.9 | -24.5 | 6.14 |
| 5 | SH3130 | 1976-1985 | 1981 | 144.0 | 179.24 | 73.33 | 252.58 | 25.7 | natural (self-draught) | -16.5 | 9.47 | -25.4 | 11.72 |
| 6 | SH4028 | 1986-1995 | 1986 | 120.0 | 102.35 | 36.0 | 138.35 | 39.0 | exhaust with exhaust-air heat pump (FVP) | -15.3 | 5.02 | -24.5 | 6.29 |
| 7 | SH5146 | 1996-2005 | 1998 | 133.0 | 103.44 | 35.33 | 138.77 | 43.1 | exhaust with exhaust-air heat pump (FVP) | -15.3 | 5.04 | -24.5 | 6.31 |
| 8 | SH5007 | 1996-2005 | 2001 | 140.0 | 152.28 | 30.33 | 182.61 | 34.5 | exhaust with exhaust-air heat pump (FVP) | -16.5 | 6.85 | -25.4 | 8.47 |

Design heat load = H_total × (21 °C − DVUT), with no internal or solar gains and no DHW. DVUT is from Boverket/SMHI *DVUT 1991–2020*, read at the column for the house time constant in days (1–2 d here).

| month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T_cold karlstad °C | 7.6 | 7.0 | 7.3 | 8.6 | 10.3 | 12.2 | 13.7 | 14.3 | 13.9 | 12.7 | 10.9 | 9.0 |
| T_cold lulea °C | 3.7 | 2.8 | 2.9 | 4.2 | 6.2 | 8.5 | 10.3 | 11.3 | 11.2 | 9.9 | 7.8 | 5.6 |

### ground/ground_sites.json
Undisturbed near-surface ground temperature = site annual mean air temperature 2020–2025, following SGU's simplification that groundwater at tens of metres sits near the annual mean air temperature: karlstad 7.31 °C (mean over a 200 m borehole ≈ 9.56 °C with 22.5 K/km), lulea 3.73 °C (mean over a 200 m borehole ≈ 5.98 °C with 22.5 K/km). SGU gives a gradient of 15–30 K/km and ground temperatures of ~3–10 °C nationally. Bedrock conductivity is 3.3 W/(m K) (SGU mean; national range 2–4; granite 3–4), **not site-specific**. Typical energy-well depth is 100–300 m (SGU); recent wells have a median of 200 m (IQR 170–240) per SGU Brunnsarkivet statistics. Volumetric heat capacity is not provided (no verified source retrieved).

## Sources and licences
| data | source | licence / terms |
|---|---|---|
| weather | SMHI open data, meteorological observations, corrected archive (`sources/smhi_download_urls.txt`) | CC BY 4.0 SE — cite SMHI and state modifications (gap filling, GHI time shift) |
| spot price | elprisetjustnu.se API (ENTSO-E day-ahead, provider FX conversion); raw intervals in `sources/spot_raw/` | provider: free and open for any use, attribution requested; upstream Nord Pool/SDAC copyright notices apply — check before public redistribution of the raw series |
| energy tax | Skatteverket (2023); Energimarknadsbyrån (2025); Vattenfall (2024); secondary tables for 2022/2026 | facts |
| network tariff | Vattenfall Eldistribution price lists 2025 (and 2026 reference) | facts |
| supplier markup | Vattenfall rörligt-pris history page | facts |
| houses | Boverket BETSI open database v2 (`sources/betsi-v2.zip`, `sources/beskrivning-_betsi.pdf`) | copy, distribute and process allowed; cite 'Boverket, BETSI' |
| DVUT | Boverket/SMHI DVUT 1991–2020 (`sources/dvut_1991_2020.csv`) | copy, distribute and process allowed; cite Boverket and SMHI |
| draws | generated with OpenDHW 0.2.9 (MIT) | package output |
| DHW volumes | Energimyndigheten ER 2009:26; Sveby Brukarindata bostäder | facts |
| cold water | Burch & Christensen (2007) correlation as in EnergyPlus | method |
| ground | SGU report 'Geoenergi – geologisk information för geoenergianläggningar'; SGU geoenergy information sheet; Borrföretagen Geoguiden (2025); brunnskartan.se (SGU Brunnsarkivet statistics) | facts |

## Limitations and results against us (read before use)
1. Spot prices cover 4 years (2022–2025), not 6: the weather years 2020–2021 have no openly licensed price series. 2022 is the crisis year (SE3 spot mean 1.38 SEK/kWh, 2.3–3.4× the 2023–2025 means). A 4-year average is dominated by it; report per year.
2. The network tariff for 2022–2024 is the 2025 Vattenfall T4 list, not each year's own list. Vattenfall is not the local DSO in Karlstad or Luleå. Power (kW) tariffs are not modelled.
3. Houses are BETSI houses surveyed in 30 municipalities, not houses located in Karlstad or Luleå. The same 8 envelopes are used at both sites. There is no post-2005 house. The 3 houses with exhaust-air heat pumps (FVP; houses 6, 7, 8) keep their measured exhaust flows, but the FVP heat recovery is not modelled.
4. Two Luleå design loads exceed the RXT 12's 12 kW maximum (house 1: 14.35 kW, house 3: 12.17 kW, at DVUT and without gains). These houses need the auxiliary heater, or should be excluded at Luleå, or the 12 kW limit becomes binding by design.
5. Thermal capacitance is the TABULA standard value, not measured. Time constants of 24–47 h are short for Swedish houses. The ISO 13790 'heavy' class (260 kJ/(m²K)) would roughly 1.6× them and change the DVUT column.
6. Cold-water temperature is a correlation, not a measurement, and is not validated against Swedish data. Karlstad and Luleå draw on surface/infiltrated river water.
7. DHWcalc/OpenDHW fixes each year's total volume by construction (realised within 0.4 L/day of target), so inter-annual variability comes from timing only. Bath counts are nearly constant. The probability shapes are German (DHWcalc) defaults, not Swedish measurements.
8. Ground conductivity and temperature are national values plus a simplification. SGU's site map (Kartvisaren Geoenergi) was not queried.
9. BETSI's thermal-bridge allowance is a flat +20 % on UA. A context value from BETSI tank measurements: top-of-tank temperature P10/P50/P90 = [35.2, 67.4, 84.7] °C (n = 20, weighted share below 60 °C 0.292); small sample, context only.

## Reproduce
`scripts/build_weather.py` (SMHI CSVs → weather), `scripts/fetch_spot.py` + `scripts/build_price.py`, `scripts/gen_draws.py` (conda env with OpenDHW 0.2.9, holidays 0.107; `--selftest` first), `scripts/build_houses.py`, `scripts/build_ground.py`. `MANIFEST.sha256` lists file hashes.
