# Data

The repository does not store data. `python -m data_pipeline` downloads the public sources, builds every file
the code reads, and compares the result with the reference copy the paper results were produced on.

```
python -m data_pipeline                  # everything (Sweden + CityLearn)
python -m data_pipeline --only sweden    # Swedish package and dhw-v2 packages only
python -m data_pipeline --only citylearn # CityLearn datasets and schemas only
python -m data_pipeline --check          # compare what is on disk, download and build nothing
```

Other options: `--refresh` downloads the sources again even when a cached copy exists, `--force` rebuilds
every step, `--no-validation` skips the dhw-v2 package validation run, `--schemas DIR` writes the CityLearn
schemas somewhere other than `citylearn_schemas/`, and `--reference DIR` (or `THERMIA_REFERENCE_DATA`)
points at a reference data tree so that files which differ are also diffed value by value.

## Where the data goes

All locations come from `stems/paths.py`.

| path | content |
|---|---|
| `THERMIA_DATA` | root; default `<repo>/data` (git-ignored). Set the environment variable to put the data elsewhere |
| `THERMIA_DATA/sweden/stems_sweden_data_v1/` | the Swedish data package (weather, price, houses, ground, draws) and its `sources/` |
| `THERMIA_DATA/sweden/packages/<site>_<year>/` | the same data converted to the dhw-v2 `load_package` format, one folder per site-year |
| `THERMIA_DATA/sweden/downloads/smhi/` | raw SMHI station files (cache; not part of the package) |
| `THERMIA_DATA/citylearn/datasets/<name>/` | the two CityLearn datasets the code uses |
| `<repo>/citylearn_schemas/tx_travis_8b/schema.json`, `.../tx_travis_8b_ev/` | generated CityLearn schemas (git-ignored; the EV charger CSVs are versioned) |
| `THERMIA_DATA/.pipeline_state.json` | what the pipeline last built, used to skip steps on the next run |

## Running time and disk use

Measured on the development machine (24 cores, fast connection), first run into an empty folder:

| step | time |
|---|---|
| SMHI and Boverket downloads (8 + 3 files) | 5 s |
| spot prices (4 zones × 1,461-1,462 daily requests, 4 in parallel) | 50-60 s |
| weather, price, houses, ground | 4 s |
| hot-water draws (48 house-years) | 40 s |
| dhw-v2 packages incl. validation | 32 s |
| CityLearn tarball (streamed) and extraction | 40 s |
| schemas | < 1 s |
| **total** | **about 3 minutes** (170 s measured on 2026-11-13) |

Peak memory is about 0.7 GB.

Disk, about 520 MB in total:

| folder | size |
|---|---|
| `sweden/stems_sweden_data_v1/` (of which `draws/` 131 MB) | 143 MB |
| `sweden/packages/` | 212 MB |
| `sweden/downloads/smhi/` (raw station files, can be deleted after the build) | 79 MB |
| `citylearn/datasets/` | 84 MB |
| generated schemas and charger CSVs | 2 MB |

The CityLearn tarball is streamed and never written to disk.

A second run with everything in place skips every step and only re-hashes the files (about 3 s).

## Steps

Each build step lists its output files. A step is skipped when all of them exist and each matches either
the reference hash or what the previous run of the pipeline wrote, and no step before it changed anything in
this run. Downloads are cached: a source file that is already on disk is not fetched again unless `--refresh`
is given.

1. **Fetch SMHI and Boverket sources** (`fetch_sources.py`). Eight SMHI station files and three Boverket files.
2. **Fetch spot prices** (`fetch_spot.py`). One request per day and bidding zone to elprisetjustnu.se,
   2022-01-01 to 2026-01-01 for SE1/SE3 and to 2025-12-31 for SE2/SE4 (the spans the reference was fetched
   with). Raw intervals are kept in `sources/spot_raw/spot_<zone>_raw.csv.gz`.
3. **Build weather** (`build_weather.py`). Hourly outdoor temperature and global irradiance 2020-2025 UTC for
   Karlstad and Luleå. Temperature gaps of up to 2 h are interpolated, longer gaps are filled by a linear
   regression on a nearby donor station fitted within ±15 days. SMHI stamps the irradiance mean at the end of
   the hour; it is moved to the start. `weather_qc.json` records gap counts and a donor-fill validation on 300
   synthetic night gaps (seeded).
4. **Build price** (`build_price.py`). Hourly spot price for SE1-SE4 2022-2025 (the 15-minute products from
   2025-10-01 are averaged per hour) and the retail price
   `1.25 × (spot + supplier markup + network energy + energy tax)`. The tariff components are fixed values
   documented in `price/tariff_composition.json`, which is copied from `data_pipeline/static/` (it is a
   hand-compiled table, not computed). SE2 uses the reduced northern energy tax, SE4 the standard one.
5. **Build houses** (`build_houses.py`). Picks 8 real single-family houses from Boverket's BETSI survey by a
   fixed rule (per construction era, the house closest to a weighted quantile of UA/Atemp and the era's median
   Atemp), derives ISO 13790 5R1C parameters and the design heat load at the DVUT temperature, and computes a
   monthly cold-water temperature from the site's air temperature (Burch & Christensen 2007).
6. **Build ground** (`build_ground.py`). Undisturbed ground temperature = mean air temperature 2020-2025, plus
   national SGU values for conductivity, gradient and borehole depth.
7. **Generate hot-water draws** (`gen_draws.py`). Synthetic 5-minute tap-water draws for the 8 houses and
   2020-2025 with OpenDHW 0.2.9, seeded per house-year, on Swedish local time and off days. A self-test first
   checks that the ported yearly functions reproduce OpenDHW's own profile.
8. **Copy the package README** (`static/stems_sweden_data_v1_README.md` to `stems_sweden_data_v1/README.md`). It
   documents every column, the per-year statistics and the limitations of the package. Its "Reproduce" section
   names the original script files; they are now the modules of `data_pipeline/`.
9. **Convert to dhw-v2 packages** (`convert_sweden.py`). One folder per site and UTC year 2022-2025 with
   `weather.csv`, `price.csv` (retail and spot), `cold_water.csv`, `draws/H01..H08.csv` (draw energy from
   litres and the month's cold-water temperature), `buildings/H01..H08.json` and `manifest.json`, all
   timestamps written in Europe/Stockholm time. Then `validation.json`: each package is loaded with
   `stems.dhw_v2.data.load_package`, draw energy and the 5R1C steady state are checked, an ideal-heating year is
   simulated per house, and a two-week energy balance of the hot-water store is run on `lulea_2024`. The
   manifest's `conversion.script` field keeps the original name `experiments/dhw_v2_matrix/convert_sweden.py`
   so that the files stay identical to the reference.
10. **Fetch CityLearn datasets** (`citylearn_data.py`). Streams one tarball of the pinned CityLearn fork commit
    and keeps only `data/datasets/tx_travis_county_neighborhood` and
    `data/datasets/citylearn_challenge_2022_phase_all`. This avoids CityLearn's own downloader, which walks the
    GitHub contents API file by file and hits its rate limit. The step is skipped when both `schema.json` files
    exist.
11. **Generate CityLearn schemas** (`tools/citylearn_setup/setup_citylearn_8b.py`, `setup_citylearn_ev.py`).
    `tx_travis_8b`: the 8 Travis County buildings with three controllable devices, priced and carbon-weighted
    with the 2022 challenge files. `tx_travis_8b_ev`: the same plus 6 synthetic EV chargers (seed 17). The
    schemas contain absolute paths to the dataset folder, so they are compared after replacing those paths with
    placeholders. `setup_citylearn_mixed.py` (CityLearn 2020 climate zones) is not part of the pipeline; it needs
    the `citylearn_challenge_2020_climate_zone_*` datasets, which it looks for in
    `THERMIA_DATA/citylearn/datasets/` before falling back to CityLearn's downloader.

## Comparison with the reference

`data_pipeline/expected_hashes.json` holds the SHA-256 of every output file of the reference data (474 files:
the v1 package, its sources, all dhw-v2 packages, the two CityLearn datasets and the generated schemas).
`.gz` files are hashed after decompression, because gzip stores a timestamp. Schemas are hashed after their
machine-specific paths are replaced by `{CITYLEARN_DATASETS}` and `{SCHEMAS}`. After a run, or with `--check`,
the pipeline prints per folder how many files are identical, different or missing, and lists each file that
is not identical (with the largest absolute numeric difference for CSV files when `--reference` is given).
Differences are reported, never fatal; the exit code is 1 only when a file is missing or a manual step is
needed.

`python -m data_pipeline.expected <reference tree>` rewrites `expected_hashes.json` from a reference tree laid
out like `THERMIA_DATA` with an extra `schemas/` folder.

Why a fresh build can differ from the reference:

- SMHI's corrected archive is revised after quality control. A changed hour in 2020-2025 changes
  `weather_*.csv` and, through the annual mean temperature, `houses.json`, `cold_water_*`, `ground_sites.json`
  and the dhw-v2 packages. Outputs are cut to 2020-2025, so data after 2025 changes nothing.
- elprisetjustnu.se may correct past prices. Outputs are cut to 2022-2025. The SE2 series of the reference has
  four days on which the provider delivered 23 instead of 24 hours (listed in `price_qc.json`); if the provider
  repairs them, `price_SE2.csv` will differ.
- `validation.json` is produced by running `stems.dhw_v2` from this repository; when that code changes, the
  file changes even though the data are identical.

Result of a clean run on 2026-11-13 (empty `THERMIA_DATA`, all sources downloaded fresh): 470 of the 474
files are identical to the reference. Weather, prices, houses, ground, all 160 dhw-v2 package files, both
CityLearn datasets and both schemas are bit for bit the same. Four files differ, all by small amounts:

| file | difference | cause |
|---|---|---|
| `houses/betsi_era_statistics.json` | 7 of the survey-weighted era quantiles, by at most 0.6 (e.g. median Atemp of the -1960 era 152.0 instead of 152.1 m²) | the weighted quantile sorts values with `np.argsort`; houses with equal values are ordered differently by the NumPy version the reference was built with, which changes the cumulative weights at ties. The house selection in `houses.json` uses the same function and is identical |
| `draws/draws_house7_2022.csv`, `draws/draws_house8_2024.csv` | one 5-minute row each, `kWh_10to40` 0.97289 instead of 0.9729 (1e-5) | the litres of that row sit exactly on a rounding boundary (28.5 L → 0.972895 kWh); the 5-minute sum comes out a few ulp lower with this pandas version, so it rounds down. `litres_40C` is identical |
| `sweden/packages/validation.json` | the two-week store energy balance (largest absolute difference 0.006 kWh over two weeks) and the source-side draw totals of 4 packages (1e-5 kWh) | produced by running this repository's `stems.dhw_v2`, which has changed since the reference was made, and by the float summation of the source draw energies; not a data difference |

Two details are pinned in the code so that the draws reproduce the reference on any machine: OpenDHW orders
the draw categories with an unstable sort (`short` and `medium` both last one minute, and their order depends
on the NumPy version), so `gen_draws.py` orders them by duration, then flow rate, both descending; and newer
python-holidays versions count every Sunday as a Swedish holiday, so `gen_draws.py` asks for the list without
Sundays (Sundays are off days anyway, so only `draws_meta.json` would change).

## Sources and licences

| data | source | URL | licence / attribution |
|---|---|---|---|
| weather | SMHI open data, meteorological observations, corrected archive; stations Karlstad Flygplats 93220, Karlstad Sol 93235, Sunne A 93520, Arvika A 92410, Luleå-Kallax Flygplats 162860, Luleå Sol 162015, Boden 161940, Rödkallen A 162790 | `https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/<1 or 11>/station/<id>/period/corrected-archive/data.csv` | CC BY 4.0. Cite "SMHI" and state the modifications (gap filling, irradiance time shift) |
| spot prices SE1-SE4 | elprisetjustnu.se API (ENTSO-E day-ahead prices converted to SEK by the provider) | `https://www.elprisetjustnu.se/api/v1/prices/<yyyy>/<mm>-<dd>_<zone>.json` | free and open; attribution requested: "Elpriser tillhandahålls av Elpriset just nu.se". Upstream Nord Pool / SDAC notices apply to redistribution of the raw series |
| tariff components | Skatteverket, Energimarknadsbyrån, Vattenfall Eldistribution price lists, Vattenfall supplier markup | listed in `price/tariff_composition.json` | facts |
| houses | Boverket, BETSI open database v2 (826 single-family houses surveyed 2007-2008) | `https://www.boverket.se/contentassets/bfeba7eb66e24db8aa01ee42b5eafe29/betsi-v2.zip` (description: `.../beskrivning-_betsi.pdf`; page `https://www.boverket.se/sv/om-boverket/oppna-data/betsi-oppna-data/databasen-fran-betsi/`) | may be copied, distributed and processed; cite "Boverket, BETSI" |
| design winter temperature | Boverket / SMHI, DVUT 1991-2020 | `https://www.boverket.se/contentassets/78ea1170505245b8b105e163b56dff27/dvut_1991_2020.csv` (page `https://www.boverket.se/sv/om-boverket/oppna-data/dvut-1991-2020/`) | may be copied, distributed and processed; cite Boverket and SMHI |
| hot-water draws | generated with OpenDHW 0.2.9 (RWTH-EBC; Python port of DHWcalc, Uni Kassel) | PyPI `OpenDHW==0.2.9` | OpenDHW is MIT; the draws are output of this pipeline. Volumes follow Energimyndigheten ER 2009:26 and Sveby |
| cold water | Burch & Christensen (2007) correlation as in EnergyPlus `Site:WaterMainsTemperature` | | method |
| ground | SGU report S1616, SGU geoenergy information, Borrföretagen Geoguiden (2025), brunnskartan.se statistics of SGU Brunnsarkivet | in `ground/ground_sites.json` | facts |
| CityLearn datasets | CityLearn (Intelligent Environments Lab, UT Austin), fork `garyhost2/CityLearn` at commit `7f62933f` | `https://github.com/garyhost2/CityLearn/archive/7f62933fea157aaed45c90bca43ef72a8aaf99b4.tar.gz` | CityLearn is MIT. `tx_travis_county_neighborhood` is built from NREL ResStock / End-Use Load Profiles (published by NREL under CC BY 4.0); cite CityLearn (Nweye et al.) and NREL. Check the dataset's own README before redistributing |

The BETSI, DVUT and SMHI downloads are fetched automatically; all three were checked on 2026-11-13 and the
BETSI zip, its description PDF and the DVUT CSV are byte-identical to the reference copies. If a download
fails, the pipeline stops that step and prints the URL, the page that links it, and the exact path to save the
file to; after saving it by hand, run the pipeline again.

The Energimyndigheten pages cited for the draw volumes (`energimyndigheten_vattenmatningar.html`,
`vattenmatningar_hearing-2009-09-30.pdf` in the reference `sources/`) are citations only; no step reads them,
so the pipeline does not download them.
