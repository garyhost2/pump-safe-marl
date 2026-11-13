# Porting STEMS from CityLearn 2.6.0b1 to CityLearn 3.0.2

CityLearn 3.0.2 here means the user's fork, branch `fix/indoor-temperature-observation-stale`,
commit `7f62933` (source in `citylearn-pr/`, byte-identical to the copy installed in `.venv`).
Line numbers below are in that tree unless they are marked "2.6". The 2.6.0b1 source was read
from the stems repo's old virtual environment.

Ten tests failed after the upgrade. Each one is traced below to a specific difference, and the
section says what was changed in STEMS. Nine of the ten came from a real change in CityLearn. One
(the MPC cap test) had nothing to do with CityLearn: it came from a different LP solver version,
and the test itself was measuring the wrong thing.

## 1. Battery: standby loss, depth-of-discharge floor, efficiency evaluation

`stems/battery.py::BatteryModel` reproduces CityLearn's `Battery.charge` in closed form. Its claim
is that it is exact. After the upgrade it was off by up to 1.0e-3 in state of charge inside the
band and by 1.8e-2 next to the depth-of-discharge floor. That broke
`test_model_is_exact_inside_the_band` and `test_model_errs_toward_the_bound_being_protected`.
Five things changed in CityLearn:

1. **Standby loss moved, and it is clamped.** In 2.6, `energy_init` applied the loss when it was
   read: `soc[t-1] * capacity * (1 - loss)` (2.6 `energy_model.py:688-692`). In 3.0.2,
   `StorageDevice.next_time_step` writes `soc[t] = clip(soc[t-1] * (1 - loss), 1 - DoD, 1)`
   when the step advances (`energy_model.py:831-842`). `energy_init` then reads `soc[t]` with
   no further loss (`energy_model.py:776-787`). Standby loss can no longer pull the battery below
   its depth-of-discharge floor. `loss_coefficient` is now also scaled by the step length
   (`energy_model.py:757-760`); on hourly data that factor is 1.
2. **The storage floor is `1 - DoD`, not 0.** `StorageDevice._charge` discharges down to
   `max(_minimum_energy(), ...)` (`energy_model.py:873-874, 913-917`). `Battery._minimum_soc` is
   `1 - depth_of_discharge` (`energy_model.py:1233-1234`). `reset()` starts at
   `max(initial_soc, 1 - DoD)` (`energy_model.py:951`). In 2.6 the floor was 0 and the DoD limit
   was worked out from `soc[t-1]` before standby loss (2.6 `energy_model.py:1070-1074`).
3. **Efficiency is evaluated at the energy actually moved.** In 2.6, the efficiency curve was read
   at `min(|requested|, max_power)` (2.6 `energy_model.py:1067, 1076`). In 3.0.2,
   `Battery.charge` makes two fixed-point passes (`energy_model.py:1199-1226`). The efficiency is
   read at the energy after limiting. The capacity limit is divided by `sqrt(eta)`
   (`energy_model.py:1206-1208`) and the DoD limit is multiplied by it (`energy_model.py:1219-1223`).
   This matters whenever the capacity limit or the DoD limit binds. In that case the efficiency is
   read at the small energy that is actually moved, not at the large energy requested, and so it
   comes out different.
4. **Curve lookups.** `get_max_input_power` and `get_current_efficiency` now use
   `np.searchsorted(..., side='left') - 1`, clamped to the last segment
   (`energy_model.py:1247-1295`). The efficiency input is normalised power, clipped to `[0, 1]`.
   Inside `[0, 1]` the result is the same as before. Outside it, 2.6 extrapolated from the first
   segment.
5. **The action is clipped** to `[-1, 1]` in `Building.update_electrical_storage`
   (`building.py:1701`).

**Change.** `BatteryModel._step` now implements the 3.0.2 equations above: the clamped decay, the
`1 - DoD` floor, the two-pass efficiency, and the `searchsorted` lookup. `BatteryModel` takes a
`depth_of_discharge` argument, which `from_citylearn` and `EVFleetModel.from_citylearn` read from
the simulator. Its default of 1.0 keeps `BatteryModel.linear` and every hand-built test battery
exactly as they were.

**Result.** The original tolerances hold (1e-4). On the 167-step random rollout, the model now
matches the simulator to 9.5e-8 at every step with state of charge below 0.95, the DoD floor
included. That is float32 storage resolution. Before the upgrade, steps at the floor had to be
excluded from the test. The remaining error, up to 8.5e-4, is only above 0.95. It comes from
capacity fade: CityLearn limits the charge to `degraded_capacity` (`energy_model.py:1206`), and
the stateless model uses the nameplate capacity. 2.6 behaved the same way, and the test excluded
that region then as well. It is not new.

**A consequence tests had to follow.** `test_standing_still_at_the_lower_bound_needs_a_charge`
assumed that a battery sitting at 0.1 drifts below 0.1 through standby loss. In 3.0.2 that holds
only when the DoD floor is below 0.1. Buildings with DoD = 0.9 are held at 0.1 by the clamp in
(1), so idling is already safe for them. Building 344233 has DoD = 0.85, so its floor is 0.15 and
it can never be at 0.1 at all. The test was renamed
`..._unless_the_floor_holds_it` and now asserts both cases.

## 2. EV charger: below the minimum power, the charger now gives nothing

2.6 raised any positive request below `min_charging_power` up to the minimum:
`applied = max(min(requested, max), min)` (2.6 `electric_vehicle_charger.py:320-335`). 3.0.2
applies a dead band instead. A request below the minimum gives zero power
(`ChargerSimulation._deadband_power_kw`, `electric_vehicle_charger.py:299-345`, used at
`:355-384`). The EV battery also gets the floor from section 1. In this schema the EVs have
DoD = 0.85.

**Change.** `EVFleetModel.applied_kw` now returns 0 when `a * p_max < p_min`. The fleet battery
carries the vehicle's DoD.

**Tests.** In `test_fleet_model_matches_the_simulator`, the state-of-charge error (< 2e-4) and the
draw error (< 2e-2) both pass at their original bounds. The test also asserted that a 0.05 command
(0.35 kW) draws the 1.4 kW minimum. That is the 2.6 behaviour, so the assertion now checks the
simulator's 0 kWh instead.

`test_lp_shield_meets_every_deadline_under_a_cap_that_binds` counted cap breaches above
`cap + 1e-6` kW. The fleet model is now exact, so the LP shield fills the cap exactly. The
overshoot is now at most 3.8e-6 kW. That is one float32 ulp at 54 kW, and it comes from CityLearn
storing `net_electricity_consumption` as float32. With a margin of 1e-5 there are no breaches at
all. The test now counts breaches above `8 * spacing(float32(cap))`, about 3e-5 kW, which is the
resolution of the quantity it reads. The model tolerances were not touched, and no deadline is
missed.

## 3. First-step booking (t0 double count): fixed in CityLearn 3

`Building.update_variables` still books the ideal thermal load at `t = 0` during `reset()`
(`building.py:2830-2868`). However, `CityLearnEnv.step` now calls
`Building.clear_step_electric_loads_for_action()` before the first `apply_actions`
(`internal/runtime.py:217-219`, `building.py:1510-1519`). That call clears the cooling, heating,
DHW and non-shiftable loads. It has been in CityLearn since release 3.0.0.

**Change.** `stems.environment.citylearn_native_fixes()` detects the fix: the method exists, and
the runtime source calls it. When the fix is present, `patch_t0_double_count` installs nothing.
`citylearn_patches` no longer lists `t0_thermal_double_count`, and the new
`citylearn_native_fixes` property lists it instead.

**Tests.** The two "unpatched is wrong" tests now assert the correct behaviour. An unpatched first
step on the scenario that used to crash now runs, and it books the heating load once. A full DHW
charge at `t = 0` draws exactly the nameplate power on every heater, with or without the flag.

## 4. DHW storage action scaled by the wrong tank: fixed in CityLearn 3

`Building.update_dhw_storage` now scales by `dhw_storage.capacity` (`building.py:1651-1680`). In
2.6 it used `heating_storage.capacity` (2.6 `building.py:1562`). This defect did not cause any of
the ten failures, because the STEMS patch was still being installed and does the same thing. It
was handled in the same way anyway. The patch is detected as unnecessary and not installed.

A 72-step random rollout with the old STEMS patch forced on gives exactly the same observations as
CityLearn's native code: the largest difference is 0.0. `test_thermal.py` now asserts that the fix
is recorded as native.

## 5. Indoor-temperature observation: the fork fixes it, and the old patch then went stale

Upstream 3.0.2 still writes the LSTM prediction to `indoor_dry_bulb_temperature[self.time_step]`,
which is the step the action was applied to. The observation is built one step later from
`series[t + 1]` (`internal/building_ops.py:320-322`), so the agent sees the uncontrolled dataset
value. The fork writes to `min(time_step + 1, n - 1)` instead (`building.py:3276-3291`).

STEMS's `endogenous_obs_from_simulated_hour` patch replaced the observation with `series[t]`.
That was correct on 2.6. On the fork, `series[t]` is the temperature **before** the action, so the
patch made the observation one step stale. Every temperature the agent saw was the previous hour's.
This was the failure in `test_unpatched_temperature_is_the_uncontrolled_next_hour`. It was also
the cause of the comfort failures in section 6.

**Change.** `citylearn_native_fixes()` detects the fork's write to `self.time_step + 1`. When it
finds it, `STEMSEnvironment.temperature_write_offset` is 1, and the patch reads
`series[min(t + offset, n - 1)]`. That is the same value as the raw observation at every step
except the last one. On the final transition CityLearn does not advance the buildings
(`internal/runtime.py:892-911`) and returns the previous state. That second defect is still present
in 3.0.2, so the patch is kept for it. The other endogenous observations (state of charge, net
consumption, heating consumption) are already read at `endogenous_t = t - 1` by CityLearn itself
(`internal/building_ops.py:130, 261, 339-357`). For them, too, the patch only changes the final
transition.

**Run record.** `citylearn_patches` still lists `endogenous_obs_from_simulated_hour`, because the
patch is still installed. `citylearn_native_fixes` lists `indoor_temperature_observation`,
`t0_thermal_double_count` and `dhw_storage_capacity`. `experiments/runner.py` writes both lists to
`meta`. `experiments/hp_report.py` accepts a run as not blind to the heat pump when it has either
the patch or the native fix, and prints both lists.

**Tests.** `test_unpatched_temperature_is_the_simulated_next_hour` asserts the fork's behaviour.
The unpatched observation equals the simulated `series[t + 1]`, and it differs from the
uncontrolled dataset by more than 5 K under full winter cooling. It is stale only on the final
transition. The patched test and `test_metrics_match_citylearn_ground_truth` now read the
simulated series at `t + offset`, so they compare against the temperature that the hour-`t`
action produced. That is what both tests meant before the upgrade.

Detection reads source text with `inspect.getsource`. If the source cannot be read, every fix is
reported as absent, and the 2.6 patches are installed as before.

## 6. Comfort identification: a port issue, now fixed

`identify_from_rollout` fits `dT[t] = a (T_out - T_in) + b Phi + c` to the observations from an
excitation rollout. With the stale observation from section 5, `T_in[t+1]` was the temperature
before the action at `t`. The regression therefore paired each heat input with the change from the
previous step. On building 0 it returned `b = -3.7e-5` K/W, which means more heat cools the house,
and `identify` correctly refused it.

With the observation read at the right step, the same 48-step rollout gives a physical fit on all
eight buildings: R² from 0.24 (worst building) to 0.53 (median), RMSE up to 6.6 K, and time
constants from 1.2 h to 3.5 h. A 672-step rollout gives R² from 0.30 to 0.45, RMSE up to 8.9 K,
and time constants from 2.3 h to 5.7 h. Both comfort tests pass unchanged.

The fit is still poor against a 2 K comfort tolerance, which is what
`test_the_envelope_identified_on_citylearn_is_reported_not_trusted` documents. The dataset route
(`identify_from_citylearn`) still refuses, for the reason it always did: the dataset gives ideal
load, not delivered power against a free-running temperature. That is not a port issue.

## 7. MPC district cap: a solver change, not CityLearn

`test_the_district_cap_binds_when_it_is_tight` uses a toy battery with no CityLearn in the loop.
It passes in the old environment (cvxpy 1.7.5, scipy 1.17.1). It fails in this one (cvxpy 1.6.0,
scipy 1.13.1). The LP has several optimal plans with the same cost (52.6), and the two HiGHS
builds pick different ones.

The test turned the plan into an import as `(c - d) * 2.5 kW`. But `c` is a fraction of the
reachable state-of-charge range at that step, not of nameplate power. Near the top of the band the
reachable range is capacity-limited: at a state of charge of 0.8, `gc = 0.2`, not 0.25. The plan
HiGHS now picks puts `c = 0.5` there. The exact plant draws 1.0 kWh for that, but the test counted
1.25 kWh. The controller's own import variable, and the plant, both peak at exactly 11.0 kW.

**Change.** Test only. It now computes the import from the exact plant: `accepted_kwh` of
`action_for_soc` along the planned state-of-charge trajectory. The 1e-3 kW tolerance is unchanged.
The new form passes on both solver versions.

## 8. Would CityLearn-based numbers change?

Yes. Any result produced on 2.6 should be re-run before it is compared with one produced on 3.0.2:

* **Battery trajectories.** These change because of section 1: the floor clamp, the efficiency
  read at the limited energy, and the DoD floor. The same random 167-step action sequence on the
  same schema gives state-of-charge differences of up to 0.15 between versions (mean 0.0055). The
  largest ones are the 2.6 battery starting at 0 below its floor.
* **EV results.** These change wherever a command fell below the charger minimum (section 2), and
  through the EV DoD floor.
* **Indoor temperature.** Under the same actions, the LSTM trajectory differs from 2.6 by up to
  0.11 K over 167 steps, once the one-step storage shift is accounted for.
* **First step and DHW.** These do not change relative to patched 2.6 runs: CityLearn's own fixes
  give the same numbers as the STEMS patches.
* **Runs made on the fork before this port.** These had a one-step-stale temperature observation
  (section 5). Any comfort, reward or controller result from those runs is affected and should be
  re-run.
* **Model-based arms.** These now use exact plant models again (sections 1 and 2), so their
  safety and feasibility margins are what they were designed to be.
