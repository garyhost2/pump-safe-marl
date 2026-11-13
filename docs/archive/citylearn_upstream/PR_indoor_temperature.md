# PR title

`fix: write the simulated indoor temperature to the observed time step`

Branch: `fix/indoor-temperature-observation-stale`

---

## Description

For buildings with simulated dynamics, the `indoor_dry_bulb_temperature` observation is the
dataset's uncontrolled value rather than the simulated one, so an agent cannot observe the
effect of its own HVAC action.

The one-line change is an index, but the question worth deciding is not `+1` — it is what the
temporal contract between `apply_actions`, the dynamics model, and observation construction
is meant to be. That is set out first, because it is what determines which fix is right.

## Issue

Fixes #<ISSUE_NUMBER>

## The temporal contract, as the runtime currently implements it

Within one `step(a_t)`, `self.time_step == t` for the whole first half:

| stage | `internal/runtime.py` | index in play |
|---|---|---|
| `apply_actions` — devices consume, dynamics predict | 223 | `t` |
| `update_variables` | 251 | `t` |
| reward observations assembled, reward computed | 274–293 | **`t`** |
| `next_time_step()` — advance | 298 | `t → t+1` |
| `next_observations` assembled, returned to the agent | 398 | **`t+1`** |

Two things follow. First, every `energy_simulation` series uses one indexing convention:
entry `k` is the value *at* step `k`, and the observation builder reads `self.time_step`
uniformly for all variables. Second, there are two consumers of the temperature at two
different indices — the reward before the advance, the agent after it.

Now take the dynamics model's own definition, from its docstring and `get_dynamics_input`:
temperature history is supplied for `t - lookback - 1 … t - 1`, every other input —
including the demand just applied by `apply_actions` — for `t - lookback … t`. The quantity
predicted is therefore the temperature *resulting from* interval `t`: the state at the end
of that interval, which under the array's own convention is the temperature at `t+1`.

Writing it to index `t` stores `T(t+1)` in slot `t`. For a dynamics building the series is
off by one against its own name and against every other series in `energy_simulation`.

The two consumers then disagree, and both behaviours are measurable:

- the **agent** reads index `t+1` after the advance, which no prediction was ever written
  to — it receives the untouched dataset value, and its observation does not respond to its
  own action at all (0.000 °C between idle and full cooling);
- the **reward** reads index `t` before the advance, so it does pick up the prediction — but
  one index ahead of every other variable in the same observation vector.

So the defect is not only that the observation is stale. It is that the array violates the
indexing convention the rest of the runtime relies on, and the inconsistency is visible to
one consumer and invisible to the other.

## Changes

**`citylearn/building.py`** — `LSTMDynamicsBuilding.update_indoor_dry_bulb_temperature`

Restore the invariant: the prediction describes the state at `t+1`, so it is written at
`t+1`. The comment above the write still documents the pre-v2.4.0 contract ("this function
is called after advancing to next timestep") — before the reorder the write executed after
the advance, so `self.time_step` *was* the advanced index. This change reproduces that
exactly rather than introducing a new convention.

```diff
-        self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] = indoor_dry_bulb_temperature.item()
+        write_time_step = min(
+            self.time_step + 1,
+            self.energy_simulation.indoor_dry_bulb_temperature.shape[0] - 1,
+        )
+        self.energy_simulation.indoor_dry_bulb_temperature[write_time_step] = indoor_dry_bulb_temperature.item()
```

The index is clamped so the final step of an episode cannot write out of bounds. The
stale comment is replaced with one that states the current ordering.

## Why `t+1`, and not make the observation consume the prediction at `t`

The alternative — leave the write where it is and have the observation read the previous
index — is the one I would expect to be asked about, so here is why I did not take it.

1. The observation builder samples **one** index, `self.time_step`, for every variable.
   Special-casing the indoor temperature would make it the only component of the agent's
   state vector drawn from a different step than weather, pricing, storage SOC and
   occupancy. The agent would receive a vector mixing two time indices.
2. It would leave the stored series itself inconsistent. Everything that reads
   `energy_simulation.indoor_dry_bulb_temperature` by index afterwards — the comfort cost
   functions, logging, the occupant model — would continue to see `T(t+1)` in slot `t`. The
   observation would be patched; the data would still be off by one.
3. It does not restore the previous behaviour, and `t+1` does. Before the v2.4.0 reorder the
   write executed after the advance, so `self.time_step` already *was* the advanced index.
   The stale comment is the surviving documentation of that contract.

Point 2 is the one I weight most: the bug is in the data, and fixing it at the observation
boundary would leave every other consumer reading a misaligned series.

## Behavioural consequences, including one I could not test

Measured on `citylearn_challenge_2023_phase_2_local_evaluation`, 80 steps, constant action,
comparing the value each consumer sees under idle versus full cooling:

| consumer | index read | before | after |
|---|---|---|---|
| agent observation | `t+1`, post-advance | 0.000 °C | **26.642 °C** |
| reward function | `t`, pre-advance | 26.642 °C | 26.642 °C |

The reward was **not** blind before this change and is not blind after it — worth stating
plainly, since the obvious worry about moving the write is that it breaks the reward. What
does change is which step the reward's temperature refers to: previously the value predicted
for the interval just simulated, now the realised temperature at the start of it. In RL
terms the comfort term moves from `R(s_{t+1})` to `R(s_t, a_t)`. **This PR deliberately does
not move the reward sampling point** — where reward observations are taken relative to the
advance is your design decision, and if you would rather the reward keep reading the
post-action value, the clean way is to assemble reward observations after
`next_time_step()`, not to keep the series misaligned. I am happy to prepare that as a
separate PR.

One consequence I have **not** been able to exercise: `OccupantInteractionBuilding.apply_actions`
calls `update_set_points()` *after* the dynamics write within the same `apply_actions`, and
`LogisticRegressionOccupantInteractionBuilding.update_set_points` reads
`indoor_dry_bulb_temperature[self.time_step]` and `[self.time_step - 1]`
(`building.py:3516-3517`). Under this change the occupant model reads the realised
temperature of the previous interval rather than the one predicted for the current one. Two
shipped datasets use that class —
`quebec_neighborhood_with_demand_response_set_points` and
`quebec_neighborhood_without_demand_response_set_points`. The dataset my tests run on
instantiates `LSTMDynamicsBuilding` with no occupant, so that path is not covered here. If
you would like a test pinning the occupant behaviour before merging, say so and I will add
one; I did not want to assert a correct value for a model I have not benchmarked.

**`tests/unit/test_indoor_temperature_observation.py`** — new, two tests:

- `test_indoor_temperature_observation_responds_to_cooling_action` — the observation must
  differ between an idle and a full-power cooling action, and be lower under cooling.
- `test_indoor_temperature_observation_is_not_the_uncontrolled_series` — the observation
  must not equal `indoor_dry_bulb_temperature_without_control` on every controlled step.

Both step past the LSTM lookback window before collecting, since `simulate_dynamics` only
turns on once that buffer has filled.

## Verification

| | observed indoor temp, cooling idle | observed, cooling full | separation |
|---|---|---|---|
| before | 20.00–26.46 °C | 20.00–26.46 °C | **0.000 °C** |
| after | 23.10–38.86 °C | 9.10–25.46 °C | **26.642 °C** |

After the change the observed range tracks the simulated range. Measured on
`citylearn_challenge_2023_phase_2_local_evaluation`, 36 steps, 108 controlled
building-steps.

Your test suite, run three times on the same machine and environment, changing one thing
at a time:

| tree | result |
|---|---|
| unmodified `master` | 1 failed, 478 passed, 5 skipped |
| `master` + the fix, **no new tests** | 1 failed, **478 passed**, 5 skipped |
| `master` + the fix + the 2 new tests | 1 failed, **480 passed**, 5 skipped |

**Rows 1 and 2 are identical**, which is the regression claim stated separately from the
tests: with the fix applied and nothing added, not one existing test changes state. Row 3
then adds exactly the two new tests.

The single failure is the same test in all three runs —
`tests/unit/test_business_as_usual_baseline.py::test_cli_default_agent_is_business_as_usual`
— and it **fails on unmodified `master`**, so it is pre-existing and unrelated to this
change. Flagging it here so a red test on the branch is not mistaken for one I caused.
`5 skipped` is unchanged throughout.

The new tests fail on `master` with:

```
AssertionError: the observed indoor dry-bulb temperature is identical with the cooling
device idle and at full power, so the simulated dynamics are invisible to the agent
AssertionError: the observation equalled the uncontrolled dataset value on all 72
controlled building-steps
```

## Screenshots

n/a.

## Checklist

- [x] I have tested the changes locally and they work as intended.
- [ ] I have updated the documentation, if applicable. — no documented behaviour changes; the corrected inline comment is part of the diff.
- [x] I have added new tests, if applicable.
- [x] I have added any required dependencies to the `requirements.txt` file, if applicable. — none needed.
- [x] I have followed the project's code style and conventions.

## Additional notes

This section is kept short because the reasoning now sits under *The temporal contract*
above, where it belongs.

Scope note: this PR deliberately does **not** touch `update_dhw_storage`. That behaviour
no longer reproduces on `master` — it scales by `self.dhw_storage.capacity` there — and it
was reported independently in #156, with #157 still open proposing a wider fix across the
heating and cooling storage updates.
