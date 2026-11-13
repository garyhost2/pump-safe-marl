---
name: Bug report
about: Create a report to help us improve
title: "[BUG] indoor_dry_bulb_temperature observation is the uncontrolled dataset value, so agents cannot observe their own HVAC actions"
labels: bug
assignees: kingsleynweye
---

## Issue Description

For buildings with simulated dynamics, the `indoor_dry_bulb_temperature` observation
returned by `step()` is the dataset's uncontrolled value rather than the simulated one. The
agent therefore cannot observe the effect of its own HVAC action, and any policy input or
comfort term computed from the returned observation is blind to the thermal dynamics.

The cause is an ordering mismatch introduced when `step()` was reordered in v2.4.0.
`LSTMDynamicsBuilding.update_indoor_dry_bulb_temperature` writes its prediction to
`self.time_step` (`citylearn/building.py`):

```python
# update temperature
# this function is called after advancing to next timestep
# so the cooling demand update and this temperature update are set at the same time step
self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] = indoor_dry_bulb_temperature.item()
```

That comment describes the pre-v2.4.0 contract. Up to v2.3.1 the order was
`next_time_step` → `apply_actions`, so `self.time_step` was already the step whose
observation would be returned. Since v2.4.0 the order is `apply_actions` →
`update_variables` → `next_time_step` (`citylearn/internal/runtime.py`, lines 223, 251 and
298 on `master`), so `apply_actions` runs while `self.time_step` is still the step the
action was applied to, and the observation is built one step later — reading an index the
dynamics model never wrote to.

It is worth being precise about what is *not* affected, because the asymmetry is probably why
this has gone unnoticed. Reward observations are assembled at `runtime.py:274`, **before**
`next_time_step()` at line 298, so they read index `t` — the index the prediction is actually
written to. The reward therefore does see the simulated temperature: on the dataset below it
separates by 26.642 °C between idle and full cooling. CityLearn's own KPIs likewise read the
simulated series and are unaffected.

So every reported metric is correct, and only the observation handed to the agent — assembled
at index `t+1` after the advance — is stale. A user comparing KPIs across agents sees nothing
wrong; only the agent is blind.

## Expected Behavior

The `indoor_dry_bulb_temperature` observation reflects the simulated indoor temperature, so
it responds to the agent's cooling/heating action.

## Actual Behavior

The observation is identical regardless of the action, and equals the uncontrolled dataset
series on every controlled step.

Running 60 steps on `citylearn_challenge_2023_phase_2_local_evaluation`, over 144
dynamics-controlled building-steps:

```
cooling action 0: observed == simulated[t] on 0.00 | observed == uncontrolled dataset[t+1] on 1.00
                  simulated 23.1..38.9 C, observed 20.0..26.5 C
cooling action 1: observed == simulated[t] on 0.00 | observed == uncontrolled dataset[t+1] on 1.00
                  simulated  8.4..25.5 C, observed 20.0..26.5 C
```

The simulated temperature responds to the action as it should — its range moves from
23.1–38.9 °C at zero cooling to 8.4–25.5 °C at full cooling. The observed value does not
move at all: `20.0..26.5 °C` in both cases, to the digit.

## Steps to Reproduce

```python
import numpy as np
from citylearn.citylearn import CityLearnEnv

OBS = 'indoor_dry_bulb_temperature'

def observed(level, steps=36):
    env = CityLearnEnv('citylearn_challenge_2023_phase_2_local_evaluation', central_agent=False, random_seed=0)
    env.reset()
    out = []

    for _ in range(steps):
        actions = [[
            level if n == 'cooling_device' else (-level if n == 'cooling_or_heating_device' else 0.0)
            for n in b.active_actions
        ] for b in env.buildings]
        observations = env.step(actions)[0]

        for b, o in zip(env.buildings, observations):
            # simulate_dynamics only turns on once the LSTM lookback buffer has filled
            if b.simulate_dynamics and OBS in b.active_observations:
                out.append(o[list(b.active_observations).index(OBS)])

    return np.asarray(out)

idle, full = observed(0.0), observed(1.0)
print('identical across actions:', np.allclose(idle, full))   # -> True on 3.0.2
```

On 3.0.2 this prints `True`: full-power cooling and an idle device produce the same
observation, to floating-point equality.

## Environment

- CityLearn version: **3.0.2** (also reproduced on 2.6.0b1)
- Dataset name/link: `citylearn_challenge_2023_phase_2_local_evaluation` (shipped)
- Operating System: Ubuntu 24.04
- Python version: 3.11.17

## Possible Solution

Write the prediction to the step the observation will be built for:

```python
write_time_step = min(self.time_step + 1, self.energy_simulation.indoor_dry_bulb_temperature.shape[0] - 1)
self.energy_simulation.indoor_dry_bulb_temperature[write_time_step] = indoor_dry_bulb_temperature.item()
```

With that change the observed range tracks the simulated one (23.10–38.86 °C idle versus
9.10–25.46 °C at full cooling, a 26.6 °C separation), and the full test suite is unchanged
at 478 passed / 5 skipped with the same single pre-existing failure in
`test_business_as_usual_baseline.py`.

A PR with this change and two regression tests follows.

There is a second, equally valid resolution — leaving the write where it is and having the
observation read the simulated value instead — which may fit better with how
`update_variables` indexes demand under the current ordering. I have not assumed which you
prefer; happy to rework the PR.

## Additional Notes

Two related points, for completeness and to avoid wasting your time:

- I also looked at the `dhw_storage` action being scaled by `heating_storage.capacity`.
  That one no longer reproduces on `master`: `update_dhw_storage` now scales by
  `self.dhw_storage.capacity`, and the DHW tank charges correctly on 3.0.2. It was
  reported independently in #156, and #157 proposes a fix that additionally covers
  `update_heating_storage` and `update_cooling_storage`; as of writing that PR is still
  open, so the behaviour appears to have been corrected separately. Nothing needed here —
  noting it only so the scope of this report reads as deliberate.
- I initially suspected a third defect, where `update_variables` books thermal-device
  consumption at `time_step == 0` during `reset()` and reduces `available_nominal_power`
  for the first action. **This is not a defect on 3.0.2**:
  `clear_step_electric_loads_for_action()` is called at `time_step == 0` before
  `apply_actions` (`runtime.py:219`), and instrumenting the first step confirms the device
  sees its full nameplate (`available_nominal_power` 4.1096 kW of 4.1096 kW nominal) rather
  than the post-`reset()` value of 3.8479 kW. Mentioning it only so it is on record as
  checked and dismissed.
