# Overnight results, 2026-10-08/09

> **SUPERSEDED — this is a dated snapshot, not the current state of the work.**
> It is kept because it is the record of what was believed on 2026-10-08/09, and
> because several of its headline claims were later refuted by our own checks. Read
> `TECHNICAL_REPORT.md` for anything current. In particular, three claims below do not
> survive: the 0.718 headline in §1 is **not** like-for-like (it runs
> $\lambda_{\text{indoor}} = 0$ while every baseline runs 0.4; the matched number is
> 0.817); the comparison to the STEMS safety column is **withdrawn** (our zero covers
> the state-of-charge band alone, since the two power caps cannot bind on this stock);
> and the three-subset replication described below as "running" completed long ago —
> its results, and a later four-stock replication, are in the report. Section numbers
> here do not correspond to the report's.

Everything below is measured on the Travis County eight-building year, evaluated
in-sample over the same 8760 steps the policy trained on — matching what STEMS report as
a twelve-month table — with their metric definitions (`stems/stems_table.py`, from
Section III-A item 3 of arXiv:2510.14112) and their normalisation to an **unshielded**
rule. Five seeds per learned point unless stated.

## 1. Headline

The best configuration is behaviour-cloning warm start with the comfort weight at zero,
fifty episodes:

| column | ours | STEMS | margin |
|---|---|---|---|
| cost | **0.718** | 0.792 | −9.3% |
| avg daily peak | **0.632** | 0.821 | −23.0% |
| ramping | **0.674** | 0.883 | −23.7% |
| SOC + equipment violations | **0.000** | 0.056 | zero |
| discomfort | 0.450 | 0.132 | we lose |

Per-seed cost ratios [0.7089, 0.7103, 0.7113, 0.7215, 0.739], mean 0.7182,
sd 0.0127; **5 of 5 seeds** below their 0.792. The same
configuration trained cold gives 0.7489 (sd 0.0324,
4 of 5 below), so the warm start is worth 0.031 of cost **and cuts the
seed spread by 2.5×** — which for a controller anyone would deploy matters as much as
the mean.

0.718 is below every figure in their cost column, not only theirs: MADDPG 0.795,
D-MAPPO 0.803, MADCQ 0.814, SAC 0.824, MetaEMS 0.836, MARLISA 0.847, MPC 0.872.

## 2. The frontier, not a point

| arm | λ | ξ | cost | emission | peak | consumption | ramping | discomfort | safety viol % |
|---|---|---|---|---|---|---|---|---|---|
| `rl+calibrated` | 0 | 0 | 0.749 | 0.757 | 0.562 | 0.756 | 0.644 | 0.453 | 0.000 |
| `rl+calibrated` | 0.1 | 0 | 0.778 | 0.789 | 0.567 | 0.789 | 0.657 | 0.404 | 0.000 |
| `rl+calibrated` | 0.2 | 0 | 0.830 | 0.822 | 0.591 | 0.825 | 0.642 | 0.396 | 0.000 |
| `rl+calibrated` | 0.4 | 0 | 0.896 | 0.864 | 0.642 | 0.866 | 0.675 | 0.407 | 0.000 |
| `rl+calibrated` | 0.8 | 0 | 0.916 | 0.867 | 0.630 | 0.866 | 0.643 | 0.404 | 0.000 |
| `rl+calibrated` | 0.4 | 0.6 | 0.909 | 0.868 | 0.644 | 0.868 | 0.673 | 0.398 | 0.000 |
| `rl+calibrated+bc` | 0 | 0 | 0.718 | 0.768 | 0.632 | 0.766 | 0.674 | 0.450 | 0.000 |
| `rl+calibrated+bc` | 0.4 | 0 | 0.817 | 0.847 | 0.740 | 0.849 | 0.753 | 0.420 | 0.000 |
| **STEMS (published)** | — | — | 0.792 | 0.824 | 0.821 | 0.805 | 0.883 | 0.132 | 5.600 |

Cost is monotone in the comfort weight across a 22% span. Discomfort is not: it improves
to λ=0.2 and then reverses, so its whole range is 0.057 and the second half of the sweep
buys nothing. **Every row beats the STEMS row on peak, on ramping, and on the safety
union.** λ ≤ 0.1 also beats it on cost, emission and consumption.

Each row of their Table I is one operating point with no trade-off shown. These are
points on the curve those rows sit on.

## 3. Three terms that cannot do what they are credited with

**3.1 Discomfort measures the weather file and the building sample.** On the unpatched
simulator the metric equals the raw dataset exactly — 0.429329832 against 0.429329832,
the same 30084/70072, difference 0.00e+00 — because the indoor temperature observation
never leaves the input data. Across the hundred-building stock the per-building rate
spans 0.000–0.856; their 0.130 sits at the **3.9th percentile** of a random eight-building
draw and our subset at 0.429 above the 95th. And the λ sweep above moves discomfort by at
most 0.057 across an eightfold change in its own weight. Three independent lines, one
conclusion: that column does not measure control.

**3.2 The safety column is smaller than an event it contains.** Their definition is a
union including the thermal comfort bound; the STEMS row reports discomfort 0.132 and
safety 0.056. Our MPC row makes the arithmetic concrete: a comfort-inclusive union of
44.55%, exactly 100× its discomfort of 0.4455, because its other terms are zero.

**3.3 The renewable term is inert.** `r_renew = ξ·min(solar/non-shiftable load, 1)` reads
two exogenous channels on the pre-action state, so on a fixed window it is a function of
the calendar hour. Measured over 1500 hours, idle, the rule and MPC each accumulated
**2495.417184178** — spread 0.000e+00. It cannot change which policy is best. Turning it
on moved cost the wrong way at both budgets (0.885→0.919 at 15 episodes, 0.896→0.909 at
50), which is the optimiser being perturbed by a rescaled reward.

## 4. Explainability: three results

**Exact Shapley over the safety mechanism.** The E3 grid is a complete 2³ factorial, so
the values are closed-form, not sampled. Credit splits 50.5 / 44.7 / 4.8 per cent across
Lagrangian, projection and exact plant inversion, every interval excluding zero. Exact
calibration has **no main effect at all** — added alone it leaves the violation rate
bit-identical at 0.710094 — yet contributes −0.202432 when the projection is present. Its
entire value is interaction, which one-factor-at-a-time ablation cannot see.

**The shield as a labelling oracle.** It labels every control step for free: 70,072
building-steps, 22.1% interventions, no human annotation. Two conditions reproduce its
decisions at **0.9368 balanced accuracy** on a time-split held-out half-year. The battery
threshold it recovers is **0.101815** against a configured `SOC_min` of 0.10 — the
barrier's own parameter, read back out of behaviour.

**Graph attribution.** Cutting every edge into a building moves its action 5.65e-02;
occluding one observation channel moves it 2.12e-01. **Isolating a building from the
entire graph matters 3.8× less than one observation.** Four channels clear that bar
alone. This agrees with the architecture grid, where uniform mean pooling cost only 0.9%.

## 5. What this does not establish

* **In-sample.** Trained and evaluated on the same year — matched to their table, not
  flattering it, but not a generalisation claim. A replication across three independent
  eight-building subsets is running. **[Completed; see the report's §8.4 and §8.3f.]**
* **The cost column does not reproduce for anyone.** All eight published baselines sit
  below our measurement of the same method. Budget explains part of it (our own arm goes
  1.011 → 0.896 → 0.718) and not all. The cause is unidentified; the paper gives no
  demand-charge rate, which is why `stems/stems_table.py` refuses to invent one.
* **The caps do double duty** in this codebase — constraint threshold and reward
  normaliser — so matching their constraint set also collapsed the stability terms from
  21% of the objective to 0.27%. Stated, not hidden.
* **Our SAC is weak** (also +91% above the oracle in the architecture grid) and is not
  used to argue their SAC figure is wrong.


## 6. Correction: "exactly zero" is true of avoidable violations, not of all of them

An earlier commit said the replication held the state-of-charge and equipment union at
exactly zero on every stock. On three of the four it did. On subset 2 the warm-started
arm retains **1.43e-05**, identical across all three seeds.

All of it is the *unavoidable* class -- states from which no action can prevent the
breach. The avoidable rate is **exactly 0.0e+00 on all four stocks**, which is the claim
that was meant and the one the shield is responsible for:

| stock | avoidable | unavoidable |
|---|---|---|
| reference | 0.00e+00 | 0.00e+00 |
| subset 1 | 0.00e+00 | 0.00e+00 |
| subset 2 | 0.00e+00 | 1.43e-05 |
| subset 3 | 0.00e+00 | 0.00e+00 |
| unshielded rule, reference | 3.16e-01 | 0.00e+00 |
| unshielded rule, subset 2 | 3.12e-01 | 1.57e-04 |

On the stock where unavoidable states exist at all, the learner enters them **11 times
less often** than the rule does (1.43e-05 against 1.57e-04). Avoiding the states from
which a breach is forced is a property of the policy; refusing the breach once there is
a property of the barrier, and only the second is something a shield can promise.

## 7. A collector bug caught by looking at the figure

The frontier collector grouped points by arm, weights and budget but not by building
subset, so the three-subset replication merged into the reference row -- n went from 5 to
14 and the reference cost read 0.801 instead of 0.718, divided by a rule that never
controlled those buildings. The headline numbers in this document were read directly
from the record files and are unaffected, but the generated table would have shipped the
contaminated figure.

The subset is now part of a point's identity, each stock is normalised to its own rule,
and a stock with no rule baseline is skipped with a printed warning rather than divided
by the wrong one. The defect was visible because a plotted point sat where no measurement
was.
