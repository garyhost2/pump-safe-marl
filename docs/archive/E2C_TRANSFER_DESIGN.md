# E2C — constrained transfer across climate zones: pre-registered design

*Written 2026-10-07, before the grid is run.* Positioned against
`PRIOR_ART_TRANSFER_RL.md`: the transfer literature for building RL and the
parameter-efficient-adaptation literature for RL are **both unconstrained**, so the
dependent variable here is constraint satisfaction under transfer, not cost.

## The question

> When a controller carrying a hard safety guarantee is transferred to a new climate and
> building stock, does the guarantee survive — and which component must be re-identified
> for it to survive: the policy, or the safety layer?

**Why it is answerable here and not elsewhere.** The guarantee sweep established that the
battery band is hard *because the barrier inverts the plant exactly*: 0 breaches in 52 488
in-band cases with the exact inverse against 4 829 (9.20%) with a mis-calibrated
surrogate. Transfer changes the plant by construction. The standard recipe in the
literature — share a trunk, adapt per building — transfers the **policy** while silently
leaving the **shield** calibrated to the source plant.

## Vehicle

CityLearn 2020 challenge climate zones 1–4, 8 buildings each (commercial and multi-family;
`--hvac-control power --no-heat-pump --allow-missing-obs`), `year-split`: train
`[0, 4379]`, evaluate `[4380, 8759]`.

### Caps, measured not inherited

Full-year uncontrolled district peaks, measured 2026-10-07 (`results/zone_peaks`), with
the cap at 80% of the measured peak — the same fraction fixed for E1, carried as a stated
design choice:

| zone | district peak (kW) | district cap at 80% | worst single-building peak (kW) | building cap at 80% |
|---|---|---|---|---|
| cl2020_zone1 | 578.299 | **462.5** | 167.062 | **133.5** |
| cl2020_zone2 | 560.184 | **448.0** | 142.828 | **114.5** |
| cl2020_zone3 | 593.883 | **475.0** | 177.930 | **142.5** |
| cl2020_zone4 | 528.828 | **423.0** | 143.132 | **114.5** |

For reference the Texas stock peaks at 41.8822 kW, so these buildings are 12.6–14.2×
larger and **their caps cannot be inherited from Texas in either direction**.

**The stored `results/mixed_v1` is unusable for any constraint claim.** It runs at
`cap2000-500`, against measured peaks of 528–594 kW. Re-measured here: at that cap the
grid violation rate is `0.0` and exceedance `0.0 kWh` in **all four zones** — the identical
structural-zero defect E0 found in the Texas grid, confirmed rather than assumed.

**Closed 2026-10-07: the per-building caps are now measured.** `MetricsCalculator` records
`peak_building_import_kw`, the largest import any single building reaches, and the four
idle probes were re-run (`results/zone_peaks`). The method is validated against the
reference stock, where it reproduces both pinned caps to the digit: Texas measures a
district peak of 41.882 kW and a worst-building peak of 14.317 kW, which at 80% give
exactly the 33.5 kW and 11.5 kW this project already runs at.

**Inheriting the Texas ratio would have been wrong in the loose direction.** The
building-to-district peak ratio is 0.289, 0.255, 0.300 and 0.271 across the four zones
against 0.342 in Texas, so scaling 11.5/33.5 onto these district caps would set
per-building limits of about 159, 154, 163 and 145 kW against the measured 133.5, 114.5,
142.5 and 114.5 — **14% to 35% too loose**. A per-building constraint set that way would
under-bind, which is the structural-zero failure of §6 arriving through a ratio instead of
through a round number.

## Arms: the 2×2 the literature does not run

Crossing **policy adaptation** with **shield calibration**. Both factors are manipulated;
the second is the one no cited paper varies.

| | shield kept at **source** calibration | shield **re-identified** on target |
|---|---|---|
| **policy zero-shot** | A1 — cheapest transfer; predicted unsafe | A2 — isolates shield re-identification |
| **policy adapted** | A3 — isolates policy adaptation | A4 — the full method |

Plus three controls:

- **C1 from-scratch on target** — the per-zone ceiling; what transfer must beat on sample cost.
- **C2 rule + exact barrier on target** — the non-learned reference, as in E1.
- **C3 idle on target** — the uncontrolled baseline and the structural-zero check.

For "policy adapted" the adaptation scheme is an **instrument, not a contribution**: full
finetuning as the primary, with low-rank adaptation as a secondary cheaper variant, cited
to LoRASA (arXiv 2502.05573), which already does per-agent low-rank adaptation on a shared
MAPPO backbone. **Do not claim low-rank adaptation in RL as novel.**

## Hypotheses, declared in advance

- **H1 (the point of the study).** Shield re-identification dominates policy adaptation for
  constraint satisfaction: residual(A2) < residual(A3), and A1 is the worst arm on safety.
- **H2.** Policy adaptation dominates shield re-identification for cost: cost(A3) < cost(A2).
- **H3.** The interaction is the headline: A4 is the only arm that is both safe and cheap,
  so the two components must be transferred separately and cannot be collapsed.

H1 and H2 pulling in opposite directions is what makes the 2×2 worth running; if both
factors moved the same metric the design would be redundant.

## Statistics

**Unit of replication: the climate zone.** $|S| = 4$, seeds averaged within. That is
honest but weak: $t_{0.975,3} = 3.182$, and with the cost seed-SD of 5.2% measured in E1,
cost contrasts will almost certainly be inconclusive — E1 already showed ~75 runs would be
needed for the learner-vs-rule cost effect. **Do not promise a cost result.** The safety
contrast is the one with a chance, because the predicted effect (0% vs ~9% breaches) is
one to two orders of magnitude, not a few percent.

Primary endpoints, in order: SOC-band violation rate; district-cap violation rate and
exceedance energy; then cost. Report integer event counts beside every rate, as E1 did —
at these magnitudes a rate is a count, and a difference of one event is not a difference.

## Sizing, measured

Pilot: zone 1, `rl+calibrated`, 1 seed, 3 episodes, year-split at the binding 462.5 kW cap
— **121 s wall, 2.34 GB peak RSS**, completed `ok`.

Extrapolating linearly to E1's 15 episodes: ≈10 min per learning run. A 4 zones × 7 arms ×
5 seeds grid is ~100 learning runs ≈ 17 core-hours; at 16 workers (16 × 2.4 GB = 38 GB of
the ~54 GB available) that is **roughly 1.5 hours wall**. Memory, not cores, is the
binding constraint — do not reuse E1's `--workers 20` here, since 20 × 2.4 GB = 48 GB
leaves no headroom.

## Blocking issue found by the pilot — resolve before the grid

The pilot returned `verified: null`, where E1's `rl+calibrated` was `verified=True` on all
five seeds. The actuator evidence explains why:

```
battery: n_charge = 7,  n_rest = 39399
dhw:     n_charge = 0,  n_rest = 39365
hvac:    null
```

**The policy is barely actuating the storage at all** — 7 charge events in 39 406
building-steps, and the hot-water tank never. Its reported `soc_violation_rate` of `0.0` is
therefore worth nothing: it is the `idle` artefact from E1 §7.3, where no control gives a
perfect band record because the battery is never moved. A barrier that never has to act
cannot demonstrate that it works.

Compounding it, `barrier_intervention_rate = 0.9908` — the barrier is intervening on
essentially every step, which is not the behaviour of a shield guarding an occasionally
unsafe policy.

Three episodes is far too few to call the policy broken, and the pilot's KPIs are valid
only for sizing. But **the grid must not run until `rl+calibrated` reaches
`verified=True` on at least one climate zone.** Otherwise every safety number this design
produces is unfalsifiable in exactly the way the E0 cap finding was. Diagnose in this
order: episode budget; whether `--allow-missing-obs` zero-filling is starving the actor of
the observations it needs; and whether the barrier's near-total intervention rate is
itself suppressing the actions.

## Scope note

This is **not** a bid to out-generalise Building2Building (arXiv 2607.16534, 6000
EnergyPlus environments). Four climate zones are a vehicle for a constrained-transfer
question, not a generality claim, and the writing must say so. The contribution, if it
survives, is the dependent variable — constraint satisfaction under transfer — not the
breadth of the testbed.
