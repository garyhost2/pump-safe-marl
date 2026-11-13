# DHW v2 data package format (`stems-dhw-v2`, version 1)

Branch `dhw-v2`. Reader: `stems/dhw_v2/data.py::load_package(path)`. The Texas CityLearn
schema (`tx_travis_8b`) goes through `load_texas()` and produces **the same in-memory
structure** (`DistrictData`, below), so every plant, filter, controller and the env wrapper
are data-source agnostic.

## Directory layout

```
<package>/
  manifest.json            required
  weather.csv              required, hourly
  price.csv                required, hourly
  draws/<house_id>.csv     required, one per house, 1-60 min resolution
  buildings/<house_id>.json  required for Swedish houses (RC parameters)
  cold_water.csv           optional, hourly mains-water temperature
```

All CSV files: comma separated, UTF-8, one header row, `.` as decimal separator, no index
column, no missing values, no gaps. Extra columns are ignored.

## Time convention (all files)

* Column `time`: ISO 8601 timestamp with an explicit UTC offset or `Z`
  (`2019-01-01T00:00:00+01:00` or `2018-12-31T23:00:00Z`). Naive timestamps are rejected.
* Each row describes the interval **starting** at `time` (hour-beginning convention); a value
  is the mean (temperatures, irradiance, price) or the total (draws) over that interval.
* `weather.csv` and `price.csv` must have identical `time` columns: consecutive hours, length
  `T` (8760 or 8784 for a year; any length >= 168 is accepted). Daylight-saving changes are
  handled by the offsets: the series is strictly 3600 s apart in UTC.
* Every `draws/<id>.csv` must start at the same instant as `weather.csv`, have a constant
  interval `dt_draw` that divides 60 min (1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30 or 60 min),
  and cover exactly `T` hours (`T * 60 / dt_draw` rows).

## `manifest.json`

```json
{
  "format": "stems-dhw-v2",
  "version": 1,
  "name": "se_stockholm_2019_example",
  "location": {"name": "Stockholm", "lat": 59.33, "lon": 18.06},
  "currency": "SEK",                 // "SEK" | "EUR" | "USD"; price.csv is currency per kWh
  "price_includes": "spot + grid fee + energy tax + VAT",   // free text, reported verbatim
  "t_cold_c": 8.0,                   // constant mains temperature, used if no cold_water.csv
  "t_ground_c": 8.0,                 // undisturbed ground temperature for the borehole (optional)
  "houses": [
    {"id": "H01", "draws": "draws/H01.csv", "building": "buildings/H01.json"}
  ],
  "sources": {"weather": "SMHI open data, station ...", "price": "...", "draws": "OpenDHW ...",
              "buildings": "TABULA/EPISCOPE SE ..."}
}
```

`sources` is free text but required (every key above); it is copied into every run record.

## `weather.csv`

| column  | unit  | meaning                                             |
|---------|-------|-----------------------------------------------------|
| `time`  |       | interval start                                      |
| `T_out` | degC  | outdoor dry-bulb temperature, hourly mean           |
| `GHI`   | W/m2  | global horizontal irradiance, hourly mean (>= 0)    |

## `price.csv`

| column  | unit           | meaning                                       |
|---------|----------------|-----------------------------------------------|
| `time`  |                | interval start                                |
| `price` | currency / kWh | electricity price paid for the hour (import)  |

## `draws/<house_id>.csv`

One of the two quantity columns is required:

| column       | unit | meaning                                                              |
|--------------|------|----------------------------------------------------------------------|
| `time`       |      | interval start                                                       |
| `volume_l`   | L    | volume of **mixed** water delivered at the tap in the interval       |
| `energy_kwh` | kWh  | heat content of the delivered water above mains temperature          |
| `t_draw_c`   | degC | optional, per-row delivery (mixed) temperature for `volume_l`        |

If `volume_l` is given, energy is computed as
`energy_kwh = volume_l * 1.0 kg/L * 4.18 kJ/(kg K) * (t_draw_c - t_cold_c) / 3600`, with
`t_draw_c` from the row, else from the house JSON (`"t_draw_c"`), else 40 degC, and `t_cold_c`
from `cold_water.csv` (hourly) or the manifest constant. OpenDHW and DHWcalc write volumes at a
fixed mixed temperature: put that temperature in the house JSON.

The plant serves a draw from the top of the store through a thermostatic mixing valve: water
leaves the store at the top-node temperature, so the stored volume drawn is
`energy_kwh / (c * (T_top - t_cold))`; water colder than `t_use_c` (40 degC) is not delivered
and counts as unmet.

## `buildings/<house_id>.json`

```json
{
  "model": "5R1C",                 // "5R1C" (ISO 13790 simple hourly) | "1R1C"
  "source": "TABULA SE.N.SFH.02.Gen (1961-1975), ...",   // required, free text
  "A_f_m2": 140.0,                 // conditioned floor area
  "H_ve_W_K": 55.0,                // ventilation heat transfer coefficient
  "H_tr_w_W_K": 60.0,              // windows and doors (no thermal mass)
  "H_tr_op_W_K": 140.0,            // opaque envelope incl. ground
  "C_m_J_K": 23.1e6,               // internal heat capacity
  "A_m_m2": 350.0,                 // optional, effective mass area, default 2.5 * A_f
  "A_t_m2": 630.0,                 // optional, area of all surfaces facing the zone, default 4.5 * A_f
  "solar_aperture_m2": 3.0,        // effective aperture so that phi_sol = aperture * GHI
  "internal_gains_W": 400.0,       // constant internal gains (or "internal_gains_W_m2")
  "T_set_c": 21.0,                 // heating set point
  "T_comfort_min_c": 20.0,         // optional, comfort floor for reporting/filter, default T_set - 1
  "heating_curve": {"curve_c": 40.0, "T_supply_min_c": 20.0, "T_supply_max_c": 55.0,
                    "seasonal_stop_c": 17.0},
  "t_draw_c": 40.0,                // optional, delivery temperature of volume_l draws
  "occupants": 3                   // optional, reported only
}
```

`1R1C` replaces the five conductances by `H_W_K` (total heat loss coefficient) and `C_J_K`.
`heating_curve.curve_c` is Thermia's curve value: the supply temperature at 0 degC outdoor; the
curve is the straight line through (20 degC outdoor, 20 degC supply) and (0 degC, `curve_c`),
clipped to [`T_supply_min_c`, `T_supply_max_c`]; no space heating above `seasonal_stop_c`.

## `cold_water.csv` (optional)

`time`, `T_cold` (degC), hourly, same index as `weather.csv`.

## In-memory structure (`stems.dhw_v2.data.DistrictData`)

```
DistrictData
  name: str
  source: str                 # "package:<path>" or "citylearn:tx_travis_8b"
  time_utc: np.ndarray        # (T,) datetime64[s], hour starts
  hour_of_day: np.ndarray     # (T,) int, local time
  day_of_week: np.ndarray     # (T,) int, 0 = Monday, local time
  t_out_c: np.ndarray         # (T,)
  ghi_w_m2: np.ndarray        # (T,)
  price: np.ndarray           # (T,) currency/kWh
  currency: str
  t_ground_c: float           # undisturbed ground temperature
  meta: dict                  # sources, price_includes, ...
  houses: list[HouseData]
HouseData
  id: str
  draw_kwh: np.ndarray        # (T * 60 / draw_dt_min,) thermal kWh per native interval
  draw_dt_min: int            # native draw resolution (60 for CityLearn)
  t_cold_c: np.ndarray        # (T,)
  building: dict | None       # RC parameters as in buildings/<id>.json, or
                              # {"model": "demand", "heating_demand_kwh": (T,)} (CityLearn)
  tank_alt: dict | None       # optional alternative tank (CityLearn autosized capacity, loss)
  meta: dict
DistrictData.draws_substep(dt_min, profile) -> (T, 60/dt_min, B) kWh per plant sub-step
```

`draws_substep` aggregates native draws to the plant step when they are finer than the plant
step, and distributes them when they are coarser, using the documented intra-hour profile
(`uniform` by default; `front` = whole interval at its first sub-step, used as a stress test).

## Texas path (`load_texas`)

`citylearn_schemas/tx_travis_8b/schema.json`, the 8 included buildings, 8759 h:
`dhw_demand` (kWh thermal, hourly), `heating_demand` (kWh thermal, hourly; becomes
`building = {"model": "demand", ...}`), `outdoor_dry_bulb_temperature`, GHI = direct + diffuse
irradiance columns of `weather.csv` (a stated approximation), `electricity_pricing` (USD/kWh),
`t_cold_c` = 10 degC constant (as in the v1 runs), `t_ground_c` = annual mean outdoor + 1 K.
CityLearn autosized DHW tank capacity and loss coefficient go to `tank_alt` (selectable tank).
