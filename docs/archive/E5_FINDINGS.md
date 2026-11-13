# E5 — the EV fleet and the multi-agent learners under a binding cap

Two district caps, treated as a factor rather than a constant, on the eight-building
Travis County scenario with chargers (`citylearn_schemas/tx_travis_8b_ev`).

## Why two caps

Measuring this scenario's uncontrolled peak the way E1–E3 do returns **41.8822 kW** —
identical to the non-EV reference to four decimals, because the `idle` arm charges nothing.
It misses all 1 713 departures and leaves 40 314 kWh of shortfall. An 80% cap taken from
that trajectory would be calibrated against a year containing none of the load E5 exists to
study, which is the §6 error in a new place. Serving the fleet with the rule-based arm peaks
at **81.2699 kW** instead, missing 0.8% of departures.

- **65.0 kW** = 80% of the serving peak. Binds, deadline set mostly feasible.
- **33.5 kW** = 80% of the idle peak, the E1–E3 operating point. 2.4× below the serving
  peak, so the fleet runs saturated.

Both measured in `results/e5_peaks` with an unreachable 2 000 kW cap. Reproduce with
`bash experiments/drive_e5.sh`.

## Part A — allocation rules (28 runs per cap)

`python -m experiments.diagnostics.summarise_e5_rules`

**The blank column is the result.** `infeasible_hour_rate` is empty, with zero scored hours,
for all six myopic rules at both caps. Before the fix described in FRAMEWORK_GUARANTEES S5.1
it would have printed `0.000` and read as a perfect score. Only `lp` reports, from 335
scored hours: **0** at 65 kW and **0.0433** (causal) / **0.0224** (replay) at 33.5 kW. The
deadline set genuinely becomes infeasible when headroom halves, and exactly one rule in the
library is able to say so.

**At 65 kW the comparison measures nothing.** Every coordinating rule serves all departures
and touches the cap by 0.0–0.6 kWh. **At 33.5 kW the same rules separate**, breaching by
14.7–40.0 kWh and missing 3.7–24.6% of departures (58–175 kWh unserved).

**The uncoordinated arms show the opposite sacrifice.** Under the 33.5 kW cap `independent`
misses no departures and breaches by 775.9 kWh; `noguard` breaches by 941.6 kWh while
missing 1.49%; both peak at 78–82 kW against a 33.5 kW cap. `static` is the mirror image:
it never breaches and silently misses 24.6% of departures under the causal forecast. Same
conflict, opposite resolution — which is what `cap_resolution` now records per step.

## Part B — learning and multi-agent arms (17 records per cap)

Five seeds for each learner, one for the deterministic arms.
`python -m experiments.diagnostics.summarise_e5_learners`

| cap | arm | cost | total viol | cap viol | peak kW | EV missed | EV short kWh |
|---|---|---|---|---|---|---|---|
| 65.0 | `rbc+calibrated` | **15 307** | 2.28e-4 | 2.28e-4 | 65.4 | 0.0000 | 0 |
| 65.0 | `rl+calibrated` | 19 606 | 4.57e-5 | 4.57e-5 | 62.7 | 0.0009 | 1 |
| 65.0 | `dmappo` | 20 146 | 4.57e-5 | 4.57e-5 | 63.4 | 0.0014 | 0 |
| 65.0 | `mappo-cc` | 20 754 | 1.43e-4 | 1.37e-4 | 64.5 | 0.0000 | 0 |
| 33.5 | `rbc+calibrated` | **14 906** | 0.0262 | 0.0187 | 40.8 | **0.0000** | 0 |
| 33.5 | `rl+calibrated` | 17 737 | 0.00933 | 0.00439 | 37.8 | 0.0388 | 58 |
| 33.5 | `dmappo` | 18 343 | 0.00916 | 0.00402 | 37.3 | 0.0342 | 30 |
| 33.5 | `mappo-cc` | 18 277 | 0.00932 | 0.00411 | 40.0 | 0.0478 | 77 |

`idle` is omitted from the discussion: it is the cheapest arm in the table at 11 842 and
misses every departure, which is the reminder that cost without the service constraint is
not a ranking.

**The rule and the learners resolve the conflict in opposite directions, and only under the
binding cap.** At 33.5 kW `rbc+calibrated` serves **every** departure and breaches the cap
**4.3× more often** than the learners (0.0187 against 0.0040–0.0044). The learners hold the
cap and drop **3.4–4.8% of departures**, 30–77 kWh unserved. Neither is wrong; they are
different priority orderings, and neither arm declares one. At 65 kW the distinction nearly
vanishes — every arm serves the fleet and the cap rates are 4.6e-5 to 2.3e-4 — so a paper
reporting only the generous cap would conclude the arms behave alike.

**Learning costs 19–36% more than the rule, in both regimes.** 19 606–20 754 against 15 307
at 65 kW; 17 737–18 343 against 14 906 at 33.5 kW. The E2 cost null (§7.1, mean +2.10%,
CI [−5.34, +9.54]) becomes an unambiguous deficit once the fleet is present. Part of that
gap is purchased service — the learners miss departures the rule serves — but not at 65 kW,
where every arm serves the fleet and the learners are still 28–36% dearer.

**No multi-agent advantage.** Single-agent `rl+calibrated` is the cheapest learner at both
caps (17 737 and 19 606), ahead of `dmappo` and `mappo-cc`, and its safety and EV service
are within the spread of theirs. On this scenario the centralised critic buys nothing.

## What this does not establish

- One scenario, one climate, one synthetic EV schedule set. The deterministic arms are a
  single run each; only the learners carry five seeds.
- The cap factor has two levels, so "as headroom falls the rule shifts toward the cap and
  the learner toward the deadline" is a direction between two points, not a measured slope.
- `infeasible_hour_rate` is reported only where a rule solves the joint deadline set. That
  is now honest, but it means the myopic rules remain uncompared on feasibility — the metric
  does not exist for them, rather than existing and reading zero.
