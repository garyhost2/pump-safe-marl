# E1 — the year-split gate

`results/e1_split`, scenario key `tx_travis_8b__year-split__refn8__cap33.5-11.5`.
23 records, 23 `ok`, 0 errors, 0 failed actuator checks. Train `[0, 4379]`,
evaluate `[4380, 8759]`. Caps 33.5 / 11.5 kW. 7 arms, 5 seeds on the four
learning arms, 15 episodes. Ran in 23.2 min on 20 workers.

Preceded by the full suite on this machine: **658 passed, 0 failed, 0 skipped**.
That was the real check on `505de3a`, which changed the cap defaults and left
about 500 tests unverified.

## Q1. Does `rl+calibrated` still report a zero violation rate now the cap binds?

**No. It is 0.0001884, and the decomposition says exactly where it comes from.**

| arm | SOC band | district cap | per-building power | total |
|---|---|---|---|---|
| time-of-use rule | 0.3131 | 0.008678 | 0.0002855 | 0.3211 |
| rule + exact barrier | 0 | 0.007993 | 0.0002284 | 0.008078 |
| learner, penalty only | 0.0596 | 0.0003197 | 9.135e-05 | 0.05997 |
| learner + uniform-rate barrier | 0.00133 | 9.135e-05 | 0.0001199 | 0.001541 |
| learner + linear barrier | 0.004995 | 9.135e-05 | 9.135e-05 | 0.005172 |
| learner + exact barrier | 0 | 9.135e-05 | 9.705e-05 | 0.0001884 |

The SOC-band rate for `rl+calibrated` is **exactly 0** — on held-out data, under a
cap that now binds. The exact battery barrier survives the gate intact. The whole
of the new 0.0001884 total is the cap dimension
(9.135e-05 district + 9.705e-05 per-building).

This is E0 finding 3 landing as predicted. In-sample, every arm's district-cap
violation rate was **0.0000 for all seven arms** — not because they complied, but
because a full-year uncontrolled rollout peaks at 41.88 kW against the old 300 kW
default, so 0 of 8759 hours could exceed it. The stored `0.0000` for
`rl+calibrated` on all five seeds was structurally zero in that dimension. It was
never evidence of cap compliance.

## Q2. How large is the drop from the in-sample numbers?

**Smaller than expected in money, and the safety claim is what actually moves.**

Only rates and per-hour quantities are comparable: the in-sample grid evaluates
8759 steps (train window = eval window), the split evaluates 4379.

| arm | cost/h in-sample | cost/h held-out | change |
|---|---|---|---|
| idle | 2.6237 | 2.7043 | +3.07% |
| time-of-use rule | 2.1433 | 2.2098 | +3.10% |
| rule + exact barrier | 2.1209 | 2.1852 | +3.03% |
| learner, penalty only | 2.4598 | 2.5057 | +1.86% |
| learner + uniform-rate barrier | 2.2886 | 2.3438 | +2.41% |
| learner + linear barrier | 2.3071 | 2.4008 | +4.06% |
| learner + exact barrier | 2.1158 | 2.1483 | +1.54% |

Every arm is 1.5–4.1% dearer per hour out of sample, and `rl+calibrated` has the
smallest gap of any arm at +1.54%. The stored cost numbers broadly survive the
window change.

**This number is confounded and must not be quoted as a pure generalisation
cost.** The held-out window is hours 4380–8759 — the second half of the calendar
year — so it differs from the full-year in-sample window in seasonal composition
as well as in being held out. Separating the two would need the same window
evaluated by a policy trained on it, which this grid does not contain.

## What the binding cap exposes: the rule, not the learner

| arm | cap violation rate | exceedance over the half-year | peak import |
|---|---|---|---|
| time-of-use rule | 0.008678 | 303.7 kWh | 57.2 kW |
| rule + exact barrier | 0.007993 | 277.4 kWh | 51.3 kW |
| learner, penalty only | 0.0003197 | 1.9 kWh | 33.7 kW |
| learner + uniform-rate barrier | 9.135e-05 | 3.2 kWh | 35.6 kW |
| learner + linear barrier | 9.135e-05 | 4.1 kWh | 35.9 kW |
| learner + exact barrier | 9.135e-05 | 2.3 kWh | 34.1 kW |

`rbc+calibrated` holds the battery band perfectly (SOC rate 0) and still breaches
the district cap in 0.80% of hours for
277 kWh, peaking at
51.3 kW against a 33.5 kW cap — about
88x
the shielded learner's rate. The exact barrier is a battery-band barrier; it was
never a cap shield. On this housing-only schema the arms that comply are the ones
whose policy happens to keep the district peak low. Whether that is the shield
acting or the learned policy's lower peak is E3's question, not this grid's.

Note the design tolerance: the district cap is **soft by construction** — its
margin is the 95th percentile of one-step forecast error, so roughly 5%
exceedance is intended. The shielded learner's observed
0.009% sits far inside that, and the rule's
0.80% also does. These are not design-tolerance failures;
they are differences between arms that were previously invisible.

## Does learning beat the rule on held-out data?

Not at a standard worth publishing. `rl+calibrated` costs
9407 against `rbc+calibrated`'s
9569 — an advantage of
162
(1.69%).
The learner's across-seed SD is 491 (5.2% of its mean),
three times the advantage. The advantage is inside training noise.

It also cannot be tested here: the unit of replication is the scenario, and this
grid is one scenario, so `aggregate` reports every contrast as `n=1` with no CI
and no p-value. Seeds are training noise, not replication. E2's five building
subsets are what make this contrast testable.

`rl+calibrated` does win on the secondary KPIs it should win on: discomfort
0.01395 vs 0.03806,
average daily peak 17.4 vs 25.7 kW,
emissions 5228 vs 5758.

## Caveats that travel with these numbers

- **The 80% cap fraction is a design choice**, not a sourced engineering limit
  (confirmed as the intended basis for E1). The peaks it multiplies are measurements.
- **One scenario, so no inference.** Every contrast is `n=1`.
- The per-hour cost comparison mixes generalisation with seasonal composition (above).
- **Actuator evidence thinned on the unshielded arms**: `rl` and `rl+basic` are
  2 of 5 seeds `verified` here against 3 of 5 and 4 of 5 in-sample, the rest
  "insufficient evidence". `rl+calibrated` and `rl+linear` are 5 of 5.
- Everything in the `RESUME.md` "carry these into anything published" list still
  applies, in particular that the learning-arm numbers in `docs/REPORT_2026-10.md`
  are superseded by this grid for the cap dimension.
