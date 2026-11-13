# What a validity audit finds in CityLearn's own datasets

`tools/citylearn_validity_audit.py` asks five questions of a CityLearn
experiment before anyone reports a number from it: can the constraint you
imposed ever bind, does each observation move when the actions move, which
scheduled-arrival channels were constant over the window, does each device with
capacity actually actuate, and what is the smallest non-zero rate the experiment
can express. It imports nothing from this project and takes any
dataset name or schema path.

This file is the audit run against every dataset CityLearn ships, on a stock
`citylearn==3.0.2` install with no STEMS code importable. It is evidence for
two claims in the paper: that the indoor-temperature defect of §3 is not a
property of our scenario, and that the unreachable-cap failure of §6 is the
normal condition rather than our mistake.

## Method

```bash
pip install citylearn==3.0.2 'nrel-pysam'      # PySAM only for the three county datasets
python tools/citylearn_validity_audit.py --dataset DATASET --steps 120 --settle 30 \
    --out audit.md --json audit.json
```

Each dataset is audited over the first 120 steps. Bindingness is measured on a
zero-action rollout. Responsiveness perturbs one action dimension at a time and
compares the whole observation vector against that baseline, discarding the
first 30 steps so that the LSTM lookback buffer has filled — `simulate_dynamics`
is off until it does, and comparing before then would hide a real response.
Dimensions that admit a negative command are driven with an alternating
half-amplitude command rather than a constant maximum, because a constant
maximum saturates a storage device within a step or two and then every channel
downstream of it looks frozen.

Coverage: 24 of the 32 shipped datasets. The other eight use the flat-entity
("tables") action interface, which nests `Dict` spaces and needs a different
action container; the audit reports that and exits rather than guessing. Three
county datasets need PySAM for PV sizing.

## Results

| dataset | buildings | with dynamics | horizon | peak kW (120 steps) | action dimensions | responsiveness | frozen indoor temp. | other frozen channels | constant arrival channels |
|---|---|---|---|---|---|---|---|---|---|
| `baeda_3dem` | 4 | 4 | 2928 | 64.9 | cooling_device, cooling_storage, dhw_storage | **FAIL** | 4 | cooling_device_efficiency x4; indoor_dry_bulb_temperature_cooling_delta x4 | — |
| `ca_alameda_county_neighborhood` | 100 | 100 | 8760 | 319 | cooling_or_heating_device, dhw_storage, electrical_storage | **FAIL** | 100 | comfort_band x100; dhw_electricity_consumption x96; hvac_mode x100; indoor_dry_bulb_temperature_cooling_delta x100; indoor_dry_bulb_temperature_heating_delta x100 | — |
| `citylearn_challenge_2020_climate_zone_1` | 9 | 0 | 8760 | 238 | cooling_storage, dhw_storage, electrical_storage | **FAIL** | 9 | — | — |
| `citylearn_challenge_2020_climate_zone_2` | 9 | 0 | 8760 | 252 | cooling_storage, dhw_storage, electrical_storage | **FAIL** | 9 | — | — |
| `citylearn_challenge_2020_climate_zone_3` | 9 | 0 | 8760 | 232 | cooling_storage, dhw_storage, electrical_storage | **FAIL** | 9 | — | — |
| `citylearn_challenge_2020_climate_zone_4` | 9 | 0 | 8760 | 236 | cooling_storage, dhw_storage, electrical_storage | **FAIL** | 9 | — | — |
| `citylearn_challenge_2021` | 9 | 0 | 35040 | 207 | cooling_storage, dhw_storage, electrical_storage | **FAIL** | 9 | — | — |
| `citylearn_challenge_2022_phase_1` | 5 | 0 | 8760 | 13 | electrical_storage | pass | 0 | — | — |
| `citylearn_challenge_2022_phase_2` | 5 | 0 | 8760 | 14.5 | electrical_storage | pass | 0 | — | — |
| `citylearn_challenge_2022_phase_3` | 7 | 0 | 8760 | 15.8 | electrical_storage | pass | 0 | — | — |
| `citylearn_challenge_2022_phase_all` | 17 | 0 | 8760 | 37.7 | electrical_storage | pass | 0 | — | — |
| `citylearn_challenge_2022_phase_all_plus_evs` | 17 | 0 | 8760 | 37.6 | deferrable_appliance_deferrable_appliance_1, electric_vehicle_storage_charger_10_1, electric_vehicle_storage_charger_12_1, electric_vehicle_storage_charger_15_1, electric_vehicle_storage_charger_15_2, electric_vehicle_storage_charger_1_1, electric_vehicle_storage_charger_4_1, electric_vehicle_storage_charger_5_1, electric_vehicle_storage_charger_7_1, electrical_storage | **FAIL** | 0 | deferrable_appliance_deferrable_appliance_1_deadline_missed x1 | 49 |
| `citylearn_challenge_2023_phase_1` | 3 | 3 | 720 | 9.77 | cooling_device, dhw_storage, electrical_storage | **FAIL** | 3 | — | — |
| `citylearn_challenge_2023_phase_2_local_evaluation` | 3 | 3 | 720 | 9.77 | cooling_device, dhw_storage, electrical_storage | **FAIL** | 3 | — | — |
| `citylearn_challenge_2023_phase_2_online_evaluation_1` | 3 | 3 | 2208 | 9.32 | cooling_device, dhw_storage, electrical_storage | **FAIL** | 3 | — | — |
| `citylearn_challenge_2023_phase_2_online_evaluation_2` | 3 | 3 | 2208 | 9.32 | cooling_device, dhw_storage, electrical_storage | **FAIL** | 3 | — | — |
| `citylearn_challenge_2023_phase_2_online_evaluation_3` | 3 | 3 | 2208 | 9.32 | cooling_device, dhw_storage, electrical_storage | **FAIL** | 3 | — | — |
| `citylearn_challenge_2023_phase_3_1` | 6 | 6 | 2208 | 16.3 | cooling_device, dhw_storage, electrical_storage | **FAIL** | 6 | — | — |
| `citylearn_challenge_2023_phase_3_2` | 6 | 6 | 2208 | 16.3 | cooling_device, dhw_storage, electrical_storage | **FAIL** | 6 | — | — |
| `citylearn_challenge_2023_phase_3_3` | 6 | 6 | 2208 | 16.3 | cooling_device, dhw_storage, electrical_storage | **FAIL** | 6 | — | — |
| `quebec_neighborhood_with_demand_response_set_points` | 20 | 20 | 2160 | 138 | heating_device | **FAIL** | 20 | — | — |
| `quebec_neighborhood_without_demand_response_set_points` | 20 | 20 | 2160 | 138 | heating_device | **FAIL** | 20 | — | — |
| `tx_travis_county_neighborhood` | 100 | 100 | 8760 | 574 | cooling_or_heating_device, dhw_storage, electrical_storage | **FAIL** | 100 | comfort_band x100; dhw_electricity_consumption x61; hvac_mode x100; indoor_dry_bulb_temperature_cooling_delta x100; indoor_dry_bulb_temperature_heating_delta x100 | — |
| `vt_chittenden_county_neighborhood` | 47 | 47 | 8760 | 656 | cooling_or_heating_device, dhw_storage, electrical_storage | **FAIL** | 47 | comfort_band x47; dhw_electricity_consumption x36; hvac_mode x47; indoor_dry_bulb_temperature_cooling_delta x47; indoor_dry_bulb_temperature_heating_delta x47 | — |

`results/citylearn_dataset_audit.csv` carries the same table with the full
per-channel counts.

## What it says

**The indoor-temperature observation is frozen in every dataset whose buildings
simulate thermal dynamics.** Fourteen of the 24 have at least one
`DynamicsBuilding`, and in all fourteen the count of buildings whose
`indoor_dry_bulb_temperature` never responds to any action equals the count of
buildings with dynamics: 324 building-instances in total, no exceptions. The
agent cannot see the effect of its own HVAC command on any of them.

**It is the same off-by-one in all of them.**
`tools/indoor_temperature_offset_probe.py` compares the observation against the
simulator's predicted series, the dataset's uncontrolled series and the cooling
set point, each at four indices around the current step:

```
citylearn_challenge_2023_phase_2_local_evaluation  level=1.0  compared=114
  observation == simulated series at time_step+0: 1.00
  observation == dataset series   at time_step+0: 1.00

tx_travis_county_neighborhood                      level=1.0  compared=3700
  observation == simulated series at time_step+0: 1.00
  observation == dataset series   at time_step+0: 1.00
```

`step()` advances the clock after the energy balance is written, so
`time_step + 0` has not been simulated yet and still carries the dataset's
uncontrolled value — which is why the two series agree there. The simulated
value sits one index behind, at `time_step - 1`, where the observation never
looks. The observed range is identical at command 0.0 and 1.0 while the
simulated range moves (on Travis County, −3.40..26.33 °C at zero command
against −15.94..77.75 °C at full), so the simulator is computing a response and
the observation is not reporting it. This is the defect filed upstream in
`docs/upstream/ISSUE_indoor_temperature.md`, reproduced here on 100 buildings
and a second action interface (`cooling_or_heating_device`) rather than the
three-building challenge set it was found on.

**Where the channel is frozen by design, the audit still says something worth
hearing.** The five 2020 and 2021 challenge datasets have no dynamics buildings
at all: their indoor temperature is a dataset series and no action could move it
(45 further building-instances). That is not a bug, but any comfort term reading
that channel on those datasets is scoring the dataset rather than the
controller, and a paper reporting comfort there is reporting a constant.

**The four datasets that pass expose only a battery.** `electrical_storage` is
the sole action dimension in 2022 phases 1, 2, 3 and all — there is no thermal
command, so there is nothing for the defect to break. The audit's verdict
tracks the presence of thermal control exactly, which is the behaviour you want
from a check you intend to trust.

**The EV dataset's failure needed splitting, not excusing.**
`citylearn_challenge_2022_phase_all_plus_evs` originally failed on 50 channels,
49 of which are electric-vehicle arrival metadata — `departure_time`,
`required_soc_departure`, `battery_capacity`, charger connection and incoming
states. No action moves those: they are properties of the car that arrives. They
are not weather either, so folding them into the exogenous count would hide
them from an EV-aware experiment that conditions on them. They now get their own
verdict, which reports them as constant over the audited window without calling
it a responsiveness failure.

That leaves one channel in the responsiveness verdict, and it is a real
question: `deferrable_appliance_1_deadline_missed`. A missed deadline is an
outcome of control, not a property of an arrival, so it stays in scope
deliberately. It may well be constant here because no deadline fell inside 120
steps — which is the resolution limit of the audit, not a defect — but that is a
thing to check rather than classify away.

**Measured peaks span 9.3 to 656 kW, and the 300 kW cap of prior work is above
almost all of them.** Over the audited 120 steps the district peak is 9.3 kW on
the 2023 three-building sets and 656 kW on the 47-building Vermont set. These
are lower bounds on the annual peak — 120 steps is five days — but the ordering
is the point: a 300 kW district cap is unreachable on 21 of the 24 datasets even
before accounting for the full horizon, so a violation rate measured under it is
structurally zero on nearly every dataset in the library. That is the §6 failure
mode, and it is not specific to us.

## What the audit does not establish

- Non-responsiveness is not wrongness. The audit says a channel does not move;
  only the offset probe says the simulator computed a different value. That
  probe was run on `citylearn_challenge_2023_phase_2_local_evaluation` and
  `tx_travis_county_neighborhood`, not on all fourteen.
- The `comfort_band`, `hvac_mode` and `dhw_electricity_consumption` channels
  flagged in the three county datasets were not investigated individually. The
  first two are plausibly dataset series that belong in the exogenous list.
  `dhw_electricity_consumption` is frozen in 61 of 100 Travis County buildings
  and 36 of 47 Vermont ones, which is a partial pattern and needs its own look.
- The two Quebec datasets draw a `warn` on device actuation: no storage device
  with non-zero capacity exists, so the actuation check has nothing to measure
  there.
- Peaks are 120-step figures from a zero-action rollout, not annual peaks, and
  the zero-action trajectory is not the same as the dataset's no-control
  baseline on every dataset.
