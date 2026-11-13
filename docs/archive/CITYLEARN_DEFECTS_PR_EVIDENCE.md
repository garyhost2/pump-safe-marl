# CityLearn defects — verification before filing upstream

*Verified 2026-10-07 against the latest release.* This document exists to answer one
question before anything is filed: **are these defects real, still present, and not
already reported?** The answer differs per defect, so the plan changes.

## Bottom line

| # | Defect | Present at 3.0.2? | Already reported upstream? | Action |
|---|---|---|---|---|
| 1 | `indoor_dry_bulb_temperature` observation is the uncontrolled dataset value | **Yes** | **No report found** | **File** — the only one |
| 2 | `dhw_storage` action scaled by `heating_storage.capacity` | **No — fixed** | **Yes:** issue #156, PR #157 | **Do not file.** Drop |
| 3 | Thermal-device load booked at `t = 0` before any action | **No — already mitigated** | n/a | **Do not file. Retracted** |

The draft in `docs/citylearn_v2.4_defects.md` was written against 2.6.0b1 and lists all
three. Only **defect 1** survives on 3.0.2. Filing the other two would report one bug
someone else already found and one that upstream already mitigated — which would cost
credibility on the one that is real. **Reduce the report to defect 1.**

> **Retraction, 2026-10-07.** An earlier version of this document, and the session that
> produced it, claimed defect 3 survived at 3.0.2. That was wrong. The original probe
> measured device consumption *after `reset()`* and **inferred** that the first action
> would be limited. It is not: `clear_step_electric_loads_for_action()` is called at
> `time_step == 0` before `apply_actions` (`citylearn/internal/runtime.py:219`), and
> instrumenting the first step shows the device sees its full nameplate. See §Defect 3.

---

## How this was verified

Deliberately adversarial to our own claim: nothing from this repository is importable, so
no STEMS patch can mask or manufacture a result.

| element | value |
|---|---|
| installs | `pip install citylearn==2.6.0b1` and `citylearn==3.0.2`, each in its own fresh conda env |
| interpreter | Python 3.11.17, numpy 1.26.4 |
| dataset | `citylearn_challenge_2023_phase_2_local_evaluation` (shipped with CityLearn) |
| cache | a separate empty `XDG_CACHE_HOME` per env, downloaded fresh |
| path hygiene | run with `env -u PYTHONPATH` from a scratch directory; the script imports only `citylearn` and `numpy` |
| script | `experiments/citylearn_defects.py`, copied out of the repo and run standalone |

`master` is at `__version__ = '3.0.2'`; PyPI's latest is also 3.0.2. The pinned 2.6.0b1 is
four releases behind (`2.6.0b2`, `3.0.0`, `3.0.1`, `3.0.2`).

**Step order is unchanged at HEAD**, which is why two of the three survive. In
`citylearn/internal/runtime.py` the step is `apply_actions` (line 223) →
`update_variables` (251) → `next_time_step` (298) — the same ordering introduced in 2.4.0
that the original analysis identified as the root cause.

---

## Defect 1 — the indoor temperature observation is blind to the agent's own actions

**Still present at 3.0.2. No upstream report found.** This is the one worth filing.

### Evidence

Identical output on 2.6.0b1 and 3.0.2, 144 controlled building-steps:

```
cooling action 0: observed == simulated[t] on 0.00 | observed == uncontrolled dataset[t+1] on 1.00
                  simulated 23.1..38.9 C, observed 20.0..26.5 C
cooling action 1: observed == simulated[t] on 0.00 | observed == uncontrolled dataset[t+1] on 1.00
                  simulated  8.4..25.5 C, observed 20.0..26.5 C
```

Read the two rows together. The simulated temperature responds to the action exactly as it
should — the range moves from 23.1–38.9 °C at zero cooling to 8.4–25.5 °C at full cooling.
**The observed value does not move at all**: `20.0..26.5 °C` in both cases, to the digit.
It matches the simulated series on **0%** of steps and the uncontrolled dataset series on
**100%** of them.

So an agent driving the HVAC cannot observe the effect of its own action on indoor
temperature. Any reward, policy input, comfort metric or termination condition built on
that observation is reading a fixed exogenous series.

### Cause at HEAD

1. `LSTMDynamicsBuilding.apply_actions` (`citylearn/building.py:3176`) calls
   `self.update_indoor_dry_bulb_temperature()` at line 3181.
2. That method writes the prediction into the *current* index:
   `self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] = ...`
   (`building.py:3281`).
3. The step then advances (`runtime.py:298`), and the observation builder reads the series
   at the **new** `time_step`, which still holds the dataset's uncontrolled value.

The write and the read are one index apart. The method's own documentation describes the
pre-2.4 ordering, where the write landed on the index the observation would later read.

### Scope, stated honestly

CityLearn's KPIs read the simulated series and are therefore **unaffected**. The defect is
confined to what the agent observes — which is precisely why it can go unnoticed while
silently invalidating any learned thermal controller.

### Suggested fix

Not proposed as a patch, because the correct resolution is an upstream design choice
between two options, and the second changes semantics for anyone who adapted to the
current behaviour:

- write the prediction to the index the observation will read, or
- have the observation source read the simulated value rather than the raw series.

The report should present the measurement and let the maintainers choose.

---

## Defect 2 — already fixed upstream, and already reported. Do not file.

`update_dhw_storage` at HEAD uses the correct device:

```python
energy = normalized_capacity_action_to_energy_kwh(
    action, self.dhw_storage.capacity, ...)   # was heating_storage.capacity
```

Confirmed empirically — the same probe that returns all-zero SOC on 2.6.0b1 returns a
charging tank on 3.0.2:

| version | `dhw_storage` SOC over 8 steps at action 0.8 |
|---|---|
| 2.6.0b1 | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` |
| 3.0.2 | `[0.782, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]` |

**Prior report:** issue **#156** "[BUG] DHW Storage is never used" (opened 2025-12-11) and
PR **#157** "Fix DHW, heating, and cooling storage update bugs" (2025-12-12). Their
diagnosis matches ours line for line, and goes further: PR #157 also fixes
`update_heating_storage` (which used `cooling_storage.capacity`) and a missing time-step
scaling factor in `update_cooling_storage` — two faults our analysis did not find.

If anything is owed here it is a confirming comment on #156 noting independent
reproduction on 2.6.0b1, not a new issue.

---

## Defect 3 — RETRACTED: not a defect on 3.0.2

**Upstream already mitigates this. Do not file.**

The booking itself is real — after `reset()`, at `time_step = 0`, before any `step()`,
three devices report non-zero consumption (table below). What does **not** happen is the
consequence we claimed. `citylearn/internal/runtime.py:219` calls
`clear_step_electric_loads_for_action()` at `time_step == 0` immediately before
`apply_actions`, and that method zeroes the cooling, heating, DHW and non-shiftable-load
consumption at the current step. Its docstring states the purpose exactly: *"Clear
reset-populated electric loads before applying control at the current step."*

Instrumenting `update_energy_from_cooling_device` on the first step of Building_1
(nominal 4.1096 kW):

| moment | `electricity_consumption[0]` | `available_nominal_power` |
|---|---|---|
| after `reset()`, before `step()` | 0.2618 kW | 3.8479 kW |
| **at the first action** | **0.0** | **4.1096 kW = full nameplate** |

No headroom is lost, so the premise of the claim fails. The original probe — and §[3] of
`experiments/citylearn_defects.py` — reports the post-`reset()` state and *infers* the
limit arithmetically; on 3.0.2 that inference is invalid.

**Lesson for the repository's own probe:** `experiments/citylearn_defects.py` §[3] should
either be rewritten to instrument the first step, or labelled as measuring the booking
rather than its effect. As written it will keep implying a defect that no longer exists.

<details>
<summary>Original (superseded) evidence for the booking</summary>

### Evidence

The cleanest form of the defect: call `reset()`, then inspect the devices **before calling
`step()` even once**. Identical on 2.6.0b1 and 3.0.2:

```
after reset(), time_step=0; BEFORE any step() is called:

building    device     nominal kW   consumption[0] kW  headroom kW
Building_1  cooling          4.11              0.2618         3.85   <-- booked before any action
Building_1  dhw              4.86              0.0593         4.80   <-- booked before any action
Building_2  cooling          2.25              0.0000         2.25
Building_2  dhw              3.68              0.0000         3.68
Building_3  cooling          2.78              0.1411         2.64   <-- booked before any action
Building_3  dhw              6.29              0.0000         6.29
```

No action has been applied, yet three devices already report consumption. That consumption
is subtracted from the nameplate when the first step computes each device's available
output, so the agent's first action is served by a device that has already spent part of
its capacity.

### Cause at HEAD

`Building.update_variables` (`citylearn/building.py:2827`) contains a
`if self.time_step == 0:` branch (line 2830) that *sets* cooling, heating and DHW device
consumption from the dataset's ideal load. It is reached during environment
initialisation — demonstrated by the table above, which is taken after `reset()` and
before any `step()`.

The first `step()` then limits each device to `get_max_output_power(...)` over a nominal
power reduced by that already-booked consumption.

### Consequence

Two modes, in increasing order of visibility:

- **Silent:** storage charging on the first step is clipped to the reduced headroom.
- **Loud:** when the first-step demand exceeds the remaining headroom, the heating and
  cooling paths raise `demand is greater than ... max output`. On this repository's Travis
  County winter schema a building needing 3.84 kW of a 4.96 kW heat pump at `t = 0` is left
  1.12 kW of headroom and asserts.

The loud mode above was observed on a locally generated schema, not on a shipped dataset,
and is reported here as context rather than as part of the reproduction. **The booked
consumption shown in the table is the reproducible claim**, and it is enough: it is
observable on a stock install with a shipped dataset in four lines of code.

CityLearn's own source notes the same double count was previously fixed for the battery,
which suggests the thermal devices were simply missed.

---

</details>

## Affected versions

| version | D1 | D2 | D3 | basis |
|---|---|---|---|---|
| ≤ 2.3.1 | ok | ok | ok | source reading of tagged releases — **not re-verified here** |
| 2.4.0 – 2.5.0 | broken | broken | broken | source reading — **not re-verified here** |
| 2.6.0b1 | **broken** | **broken** | **broken** | measured, this document |
| 3.0.2 / master | **broken** | fixed | **mitigated** (cleared before the first action) | measured, this document |

Only the last two rows are measurements. The first two come from the earlier analysis in
`docs/citylearn_v2.4_defects.md` and should either be re-verified before publication or
stated as a source reading.

---

## Reproducing this

```bash
pip install citylearn==3.0.2
python experiments/citylearn_defects.py     # imports only citylearn and numpy
```

The `reset()`-only probe for defect 3 is four lines:

```python
from citylearn.citylearn import CityLearnEnv
env = CityLearnEnv("citylearn_challenge_2023_phase_2_local_evaluation", central_agent=False)
env.reset()
print(env.buildings[0].cooling_device.nominal_power,
      env.buildings[0].cooling_device.electricity_consumption[0])
```

## Recommended filing

**One issue plus one PR, covering defect 1 only.**

- Issue text: `UPSTREAM_ISSUE.md` (bug-report template, with a self-contained repro).
- PR body: `UPSTREAM_PR.md` (their PR template).
- Patch: `citylearn_fix.patch` — branch `fix/indoor-temperature-observation-stale`,
  one-line behavioural change plus two regression tests.
- Verified: suite goes 478 -> 480 passed, 5 skipped, with the same single pre-existing
  failure (`test_business_as_usual_baseline.py`) on both unmodified `master` and the branch.

Defect 2 is a duplicate of #156/#157 and already fixed; defect 3 is retracted. Both are
mentioned in the issue's Additional Notes so the maintainers can see they were checked and
dismissed rather than missed.
