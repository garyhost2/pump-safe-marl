# "Zero violations" is not a result: safety in building-energy RL is a property of the filter's plant model

**Draft for WSED.** Every number below is measured and traceable to a script in the
repository. Sections marked ⏳ are waiting on grids running at the time of writing.

---

## Abstract

Safe reinforcement-learning controllers for building energy are routinely reported with a
constraint-violation rate, and a rate of zero is read as evidence that the method works. We
show, on a controller we built ourselves, that a reported zero can arise for at least three
reasons that have nothing to do with the method: the constraint could not bind under the
benchmark's configuration, the metric was unreachable in the implementation, or the
guarantee belonged to the fidelity of the plant model inside the safety filter rather than
to the learning algorithm. We measure each: a district import cap set at 300 kW against a
neighbourhood whose uncontrolled peak is 41.88 kW, exceeded in 0 of 8759 hours; a fleet
metric structurally 0.000 for six of seven rules because of an absent dictionary key; and an
exhaustive 52 488-case sweep in which the same shield with the same policy yields 0 band
breaches with an exact plant inverse and 4 829 (9.20%) with a plausible surrogate. We also
report a defect in the CityLearn simulator, present in the current release, that makes the
indoor-temperature observation independent of the agent's own HVAC action while leaving
every reported metric correct — which is plausibly why it has survived four minor releases —
and we supply a verified fix. We re-run our own headline grid under a cap that binds, on
held-out data, and restate the result: not 0% violations, but 0% in the state-of-charge band
and 1.9e-4 overall. Replicating across building subsets then separates what survives from
what does not: the cost advantage over a rule-based controller appears to vanish at the
15-episode budget our grids first used (mean +2.10%, 95% CI [−5.34, +9.54]) and returns at
50 episodes on the identical grid (mean **−9.78%**, CI [−15.86, −3.71], cheaper in all five
subsets), where the learner also dominates the rule on every other term of its objective —
50% less discomfort, 23% lower peak, 27% less ramping. The null was our own instance of the
failure this paper is about: a comparison that could not have come out otherwise. Meanwhile
while varying only the plant model inside the filter moves the avoidable violation rate by
21.4× against a linear filter and 24.8× against a uniform one, in the same direction in
every subset (t = 6.11 and t = 5.05, n = 5). In one subset an approximate filter is 38×
*worse* than no filter at all. The constructive counterpart is the strongest measurement
here: bolting the *exact* filter unchanged onto four published policies we did not write,
four independent building stocks and a second dataset gives **nineteen conditions in
which the band-violation rate is exactly 0.000000% in fifteen, from starting points of
40.2–94.6%, at a mean cost effect of −0.004 that is indistinguishable from zero
(p = 0.672); of three significant cells, two favour the filter**. The four exceptions all
sit on one building stock at an identical 0.001427%, a residual predicted from a battery
datasheet and then reproduced to six decimal places by four unrelated learners — the guarantee is a
property of the filter and travels with it. It also *replaces* the machinery above it:
across 15 runs and 575 episodes the PPO-Lagrangian's three multipliers are numerically
identical, reach exactly zero by episode 5 and never revive, so **90.4% of training in a
controller presented as constrained RL is unconstrained PPO**. We close with a reporting
checklist that would have caught every failure above, including two mirror-image ones in
which a comfort column computed from a temperature that was never measured reported first
a perfect record and then a total one.

---

## 1. Introduction

The claim "our controller satisfied the constraints" is the load-bearing claim of safe
reinforcement learning. In building energy it is usually reported as a single
violation rate, and a rate of zero is taken at face value.

This paper is an argument that the number is uninterpretable on its own, made by
falsifying our own prior result. We are not auditing someone else's work: the structurally
zero metric, the unreachable metric and the mis-calibrated barrier are all from our own
codebase, and we found them only because we built an exact plant inverse and could compare
against it.

Our contributions:

1. **A simulator defect** in CityLearn ≥ 2.4.0, present in 3.0.2, that renders the indoor
   temperature observation independent of the agent's HVAC action (§3). A fix with
   regression tests is supplied.
2. **A measurement** that the hardness of a control-barrier guarantee is a property of the
   plant model inside the filter, not of the architecture: 0 breaches of 52 488 with the
   exact inverse, 9.20% with a surrogate (§4).
3. **A coupled-feasibility result**: vehicles that are each individually schedulable can be
   jointly infeasible, with a sharp transition and an exactly-reported shortfall (§5).
4. **A falsified headline**: our previously reported 0% safety-violation rate was
   structurally zero; re-measured under a binding cap on held-out data it is 1.9e-4, with
   the state-of-charge component still exactly 0 (§6).
5. **A transfer result**: the same exact filter, unmodified, drives band violations to
   exactly zero across a 4x4 policy x stock matrix and a second dataset — nineteen
   conditions, 40.2–94.6% → 0.000000% in fifteen, mean cost effect −0.004 at p = 0.672,
   with the four exceptions predicted from a battery datasheet (§8.2).
6. **A self-deactivating safety layer**: behind that projection, a PPO-Lagrangian's
   multipliers are identical across channels and exactly zero for 90.4% of training, so
   the constrained-RL component is absent from the algorithm it is credited to (§8.3).
7. **A reporting checklist** that makes each failure mode visible (§9).

---

## 2. Setup

Eight residential buildings in Travis County, Texas (NREL ResStock, 2018 weather), each
with battery, photovoltaics, a heat pump and a hot-water tank, simulated in CityLearn. The
control problem is a constrained Markov decision process: maximise discounted return subject
to expected discounted constraint costs, with $B=8$ buildings, hourly steps, and a learned
policy sharing a graph-convolution and transformer encoder across buildings, trained with
PPO-Lagrangian.

Four constraints are in play — a battery state-of-charge band, a shared district import cap,
vehicle departure deadlines, and thermal comfort. They differ in enforceability, and keeping
that difference explicit is the point of §4–§6.

Expectation constraints of the CMDP form are satisfied *in expectation, discounted, over the
run*. That is a much weaker promise than "never violated at any step", which is what a
battery warranty or a service transformer cares about. The per-step promise is delivered by
a projection — a safety filter — and §4 measures exactly what that filter's promise rests on.

---

## 3. A simulator defect that makes learned thermal control unobservable

For buildings with simulated dynamics, the `indoor_dry_bulb_temperature` observation
returned by CityLearn's `step()` is the dataset's **uncontrolled** value, not the simulated
one.

Since v2.4.0 the environment applies actions before advancing the time step
(`apply_actions` → `update_variables` → `next_time_step`), but the dynamics model still
writes its prediction to the step the action was applied to, while the observation is built
one step later. The source comment above that write still documents the pre-v2.4.0 contract.

Measured on the shipped `citylearn_challenge_2023_phase_2_local_evaluation` dataset over 144
dynamics-controlled building-steps:

| cooling action | observation == simulated | observation == uncontrolled | simulated range | observed range |
|---|---|---|---|---|
| 0 (idle) | 0.00 | 1.00 | 23.1–38.9 °C | 20.0–26.5 °C |
| 1 (full) | 0.00 | 1.00 | 8.4–25.5 °C | 20.0–26.5 °C |

The simulated temperature responds to the action as it should; the observed value is
identical to the digit in both cases. An agent driving the heat pump cannot see the effect
of its own action, so any policy input, reward term or comfort metric built on that
observation is blind to the dynamics.

CityLearn's own KPIs read the simulated series and are unaffected, which is why this
survives unnoticed. The implication for the field is direct: published CityLearn results in
which a learned policy drives thermal equipment, or in which comfort enters the reward, were
computed against a blind observation.

**Fix and verification.** Writing the prediction to the step the observation is built for
restores the dynamics to the agent. After the fix, the observed range tracks the simulated
one — 23.10–38.86 °C idle against 9.10–25.46 °C at full cooling, a separation of 26.642 °C
where it was previously 0.000 °C. Upstream's own test suite is unchanged (478 passed on
master, 480 with our two added regression tests, the same single pre-existing unrelated
failure on both). Reported upstream with a patch.

---

## 4. The guarantee is a property of the plant model, not the architecture

Let $x_b \in [0,1]$ be the state of charge of building $b$'s battery and
$[x_{\min}, x_{\max}] = [0.1, 0.9]$ the band. The simulator's battery is nonlinear:
round-trip efficiency depends on power, usable capacity depends on power, and there is a
standby loss.

The safety filter computes the set of admissible actions
$\mathcal{A}^{\text{safe}}(s) = \{a : f_b(x, a) \in [x_{\min}, x_{\max}]\}$ and clips the
policy's request into it. Because $f_b(x, \cdot)$ is monotone in the action, this set is an
*interval*, and both endpoints are roots of a monotone scalar equation — so they are found
by bisection on the simulator's own map, with no linearisation. With 24 iterations from a
bracket of width 2 the endpoint error is $2^{-23} \approx 1.19 \times 10^{-7}$ in action
units.

The measurement that matters is what happens when the $f$ inside the filter is *not* the
plant's. Exhaustive sweep over the eight autosized batteries
($C_b = [10.8, 6.6, 5.0, 9.7, 13.5, 5.4, 16.2, 16.0]$ kWh, true
$r_b = [0.267, 0.244, 0.500, 0.516, 0.244, 0.533, 0.178, 0.208]$ per step), 81 commanded
actions × 81 in-band start states × 8 buildings:

| plant model inside the filter | cases | breaches | rate | max below $x_{\min}$ | max above $x_{\max}$ |
|---|---|---|---|---|---|
| exact inverse of the plant | 52 488 | **0** | 0.0000 | 0.00000 | 0.00000 |
| uniform $r_b = 0.1$ surrogate | 52 488 | 4 829 | 0.0920 | 0.10000 | 0.09390 |
| surrogate at half the true $r_b$ | 52 488 | 1 223 | 0.0233 | 0.10000 | 0.05998 |

Same shield, same policy, same constraint — three different answers. The uniform surrogate
understates the true per-step rate by 1.8×–5.3× on these devices and drives the floor to
$x = 0$ in 9.2% of states.

**This is the paper's central methodological point.** A reported "we used a control barrier
function and observed zero violations" is uninterpretable unless the paper states what $f$
was inside the filter and how well it matches the plant. The guarantee is a property of the
calibration.

What is *not* promised: starting from outside the band, the filter returns the best one-step
recovery. Over the full $[0,1]$ sweep the exact inverse shows 972/65 448 = 1.49% breaches,
all inherited from the initial state, with recovery measured (from $x_0 = 1.00$, 5 of 8
buildings are still outside after one step and 0 of 8 after two).

---

## 5. Coupled feasibility: individually schedulable, jointly impossible

Vehicles behind a shared cap are not independent. A joint linear programme over all
connected vehicles and the whole remaining horizon admits a per-vehicle shortfall slack
penalised by big-$M$, so it is always feasible and *reports* rather than failing; a
max-share term spreads shortfall instead of dumping it on one vehicle.

Three vehicles, each owing 0.4 state of charge of a 50 kWh battery, deadlines at 2/3/4
steps, $p^{\max} = 10$ kW. **Each vehicle is individually reachable at every cap in this
table**; only the joint set fails:

| $P_{\text{cap}}$ [kW] | feasible | shortfall [kWh] | worst share |
|---|---|---|---|
| 30.0 | True | 0.000 | 0.0000 |
| 20.0 | True | 0.000 | 0.0000 |
| 17.5 | True | 0.000 | 0.0000 |
| 15.0 | False | 5.000 | 0.0833 |
| 10.0 | False | 20.000 | 0.3333 |
| 5.0 | False | 40.000 | 0.6667 |

The transition is sharp at 17.5 kW and below it the reported shortfall equals the missing
energy exactly. The deadline is therefore **hard conditional on feasibility** — not a
constraint a solver can always satisfy, so one it must always report on.

The district cap itself is **soft by construction**: the filter plans against
$P_{\text{cap}} - m_t$ where $m_t$ is the 95th percentile of recent one-step forecast error,
so roughly 5% exceedance is the design, not a defect. With the margin the residual is
≤ 0.66 kW on a 30 kW cap (2.2%) on 0.48% of steps; without it, 5.82% of steps and 16.31 kWh.
Robustness depends on error *structure*, not size: a persistent bias six times the margin
lifts exceedance only to 1.53%, while scaling the innovation by 8 drives the worst residual
to 22.381 kW.

---

## 6. A reported zero that measured nothing

Our own stored result reported **0.0000 safety violations on all five seeds** for the
shielded learner. It was an artefact, in two compounding ways.

**The cap could not bind.** A full-year uncontrolled rollout peaks at **41.8822 kW**. The
district cap default was **300 kW**. The cap was therefore exceeded in **0 of 8759 hours**,
and every cap-violation column was structurally zero — not compliance, arithmetic.

**The evaluation was in-sample.** Train and evaluation windows were both $[0, 8759]$.

We re-pinned the caps to 80% of the measured uncontrolled peak (33.5 kW district, 11.5 kW
per building) and re-ran the identical arms, seeds and episode budget on a held-out window
(train $[0,4379]$, evaluate $[4380,8759]$). 23 records, 23 `ok`.

The total is the rate of building-hours violating **any** constraint — a union, not a sum
(`any_violation = soc | power | grid`, then averaged). A building-hour that breaches two
constraints is counted once, so the total is at or below the column sum, and the gap is
itself a measurement: the number of co-occurring breaches. It is reported here rather than
left for a reader to trip over.

| arm | SOC band | district cap | per-building | **total (union)** | overlap [bldg-h] |
|---|---|---|---|---|---|
| time-of-use rule | 0.313085 | 0.008678 | 0.000285 | 0.321078 | 34.0 |
| rule + exact barrier | **0** | 0.007993 | 0.000228 | 0.008078 | 5.0 |
| learner, penalty only | 0.059603 | 0.000320 | 0.000091 | 0.059974 | 1.4 |
| learner + uniform-rate barrier | 0.001330 | 0.000091 | 0.000120 | 0.001541 | 0.0 |
| learner + linear barrier | 0.004995 | 0.000091 | 0.000091 | 0.005172 | 0.2 |
| **learner + exact barrier** | **0** | 0.000091 | 0.000097 | **0.000188** | 0.0 |

Overlap is $(\sum \text{components} - \text{union}) \times TB$ with $T = 4379$, $B = 8$,
i.e. the count of building-hours breaching more than one constraint. The unshielded rule
has 34 such hours; the shielded learner has none, because it has no state-of-charge
breaches to overlap with.

The headline becomes **0% in the state-of-charge band, 1.9e-4 overall**. The band component
is exactly 0 across all five seeds — 0 breaches in 175 160 building-hours — on data the
policy never trained on, under a cap that now binds. The entire non-zero remainder is the
cap dimension, which was previously unmeasurable.

**Read the small numbers as counts.** With $T = 4379$ and $B = 8$, the shielded learner's
cap rate of $9.135\times10^{-5}$ is $[0,0,0,1,1]$ hours across the five seeds — one hour in
two runs of five. All three barrier variants share that identical profile and differ only in
the state-of-charge dimension (0 vs 46.6 vs 175.0 building-hours), confirming the barrier is
a battery-band mechanism and not a cap mechanism.

**A caution on the `idle` row.** With no control the battery never moves, so its band rate
is a perfect 0 for a trivial reason. A violation rate measures whether a controller manages
a device safely *only if it manages it at all* — which is the fourth way a zero can mislead.

---

## 7. Replication across building subsets (E2)

Five building subsets, seven arms, five seeds each, 115 completed records. Every subset
runs under its own district cap, re-measured from its own uncontrolled peak rather than
inheriting the reference scenario's: caps of 46.0, 41.0, 39.0, 42.5 and 35.0 kW against
measured peaks of 57.7, 51.5, 49.1, 53.5 and 43.8 kW. This raises the replication count
from one scenario to five, which is what permits any interval or test at all. The unit of
replication is the subset; seeds are averaged within it, because seeds measure retraining
noise on a fixed problem and do not reduce between-scenario variance.

**To be clear about what this is**: the policy trains and evaluates on the same subset.
These are scenarios, not held-out buildings. No run in this work evaluates a policy on a
building it never saw; that remains the most valuable missing evidence.

### 7.1 The cost advantage is real, and the budget that hid it is the lesson

At the 15-episode budget every grid in this paper originally used, the shielded learner's
operating cost against the calibrated rule comes out at **−8.36, +5.13, +3.37, +3.64 and
+6.71 per cent** across the five subsets: mean +2.10%, 95% CI [−5.34, +9.54], t = +0.78. The
sign is unstable and the interval spans zero. We first reported that as a null — no cost
advantage for learning.

It was an artifact of the budget. Re-running the identical grid — same subsets, same
per-subset caps, same seeds, same held-out window — and changing **only** the training budget
to 50 episodes gives **−15.58, −4.16, −9.86, −13.58 and −5.74 per cent**: mean **−9.78%**,
95% CI **[−15.86, −3.71]**, t = **−4.47**. The learner is cheaper in every subset and the
interval excludes zero. The rule is deterministic and its cost is bit-identical across the
two runs — 11830.279296875, 8651.533203125, 10663.603515625, 11982.935546875 and
8351.5859375 on subsets 1 to 5, in both grids — so nothing but the budget changed.

| budget | per-subset cost difference (%) | mean | 95% CI | t |
|---|---|---|---|---|
| 15 episodes | −8.36, +5.13, +3.37, +3.64, +6.71 | +2.10% | [−5.34, +9.54] | +0.78 |
| **50 episodes** | −15.58, −4.16, −9.86, −13.58, −5.74 | **−9.78%** | **[−15.86, −3.71]** | **−4.47** |

A separate sweep (`results/learning_budget`) had the diagnosis in it all along: on a
different scenario the learner sits 18.8% above the rule at 10 episodes, 10.8% at 30 and
3.7% at 50, with training reward still improving monotonically at 60. Fifteen episodes is
on the steep part of that curve. A null measured there is not a statement about the method.

**And at 50 episodes the learner dominates the rule on every term of its own objective**,
not only on cost:

| | calibrated rule | shielded learner at 50 ep |
|---|---|---|
| cost | 10 300 | **9 224** (−10.4%) |
| discomfort rate | 0.0552 | **0.0276** (−50.0%) |
| peak import | 54.59 kW | **42.05 kW** (−23.0%) |
| ramping | 5.416 | **3.965** (−26.8%) |
| emissions | 6 214 | **5 368** (−13.6%) |
| total violation rate | 2.91e-3 | **4.01e-4** (7.3× lower) |
| state-of-charge violations | 5.71e-6 | 5.71e-6 (equal) |

The reward is a weighted sum of exactly these terms — price × import, district peak,
per-building load, ramping, comfort, renewables, EV service, with weights
μ = 1.0, α_grid = 0.5, α_build = 0.3, β_ramp = 0.2, λ_indoor = 0.4 — so reporting the cost
column alone was also a single-axis projection of a six-term objective. Both errors pointed
the same way, and both flattered the rule.

**This is the paper's own failure mode, committed by us, three times.** §6 is about a
violation rate that could not be non-zero; §7.1 as first written was a cost comparison that
could not have come out in the learner's favour, because the policy was stopped a third of
the way into its learning curve and then scored on one of the six things it was optimising.
We leave the null in the record above rather than quietly replacing it, because the shape
of the mistake is the contribution.

The third instance is the largest and we report it against ourselves. Our reproduction of
the target paper's comparison table concluded, for six weeks and in several drafts, that
**no cost figure reproduced, for any of the eight rows**. That verdict came from reading
the target's "converges within 15 epochs" as fifteen of our episodes. Their epoch is about
three minutes on a current GPU; our episode is one 8760-step simulated year, 285–1160 s of
single-threaded CPU and 8759 gradient steps. Nothing licensed the identification, and the
15-episode grid said so on its own terms: 30 of 30 seeds still had a positive reward slope
at the cutoff. A 50-episode grid, run for §8 of the companion report and never compared
against the published table, reproduces three of the six learners —

| | ours, 50 ep | published | error |
|---|---|---|---|
| MARLISA | 0.847 | 0.847 | +0.0 % |
| D-MAPPO | 0.817 | 0.803 | +1.8 % |
| MADCQ | 0.833 | 0.814 | +2.3 % |

— on operating cost, emission and electricity consumption simultaneously. The matching
number had in fact been printed in two different tables of our own report, two hundred
lines apart, once as "ours" and once as "published".

The three that still fail fail for a reason worth separating from the paper being
reproduced. Replaying each saved policy for one deterministic year and recording what it
commands, the three that reproduce issue a saturated action on 0.0, 2.4 and 10.2 per cent
of building-steps; the three that do not, on 86.6, 92.4 and 97.4 per cent. Our soft
actor-critic policy loss was $\alpha\log\pi - Q$ with no penalty on the pre-squash Gaussian
parameters, which the reference implementation carries at weight $10^{-3}$. Without it
nothing bounds the pre-tanh mean, and a $\tanh$ at $\pm 1$ has Jacobian $1-a^2 = 0$: the
policy cannot come back. One of the three additionally took 64 gradient steps per episode
against 8759 for every other learner, and emitted a battery command of exactly $-1.000$
(s.d. $9\times10^{-6}$) at all 70 072 building-steps of the evaluation year.

The general point is the one a reproduction study is for. **Three of our six failures to
reproduce were ours**: one budget misreading and two implementation omissions, each of
which produced a plausible, stable, publishable-looking number that was wrong. None
crashed. The remaining disagreements are the ones worth discussing, and they are a much
shorter list than the one we started with.

What this does **not** license is a claim about converged policies in general. 50 episodes
is where the curve flattens on one scenario family; we have not run 100. The result is that
the advantage exists and is large at a budget a practitioner would actually use — not that
we have found its ceiling.

### 7.2 What replicates is the calibration ladder

Holding the policy class, the seed count and the scenario fixed, and varying **only the
plant model inside the safety filter**, the avoidable violation rate moves by one to three
orders of magnitude in the same direction in almost every subset:

| plant model in the filter | subset 1 | subset 2 | subset 3 | subset 4 | subset 5 | mean |
|---|---|---|---|---|---|---|
| none (no filter) | 7.69e-2 | 5.37e-4 | 3.50e-2 | 2.08e-2 | 2.67e-2 | 3.20e-2 |
| uniform rate | 1.14e-2 | 2.04e-2 | 1.36e-2 | 2.23e-3 | 4.11e-4 | 9.60e-3 |
| linear | 8.54e-3 | 9.00e-3 | 6.56e-3 | 4.74e-3 | 5.77e-4 | 5.88e-3 |
| **exact inverse** | **7.42e-5** | **5.60e-4** | **5.60e-4** | **7.71e-4** | **1.71e-5** | **3.96e-4** |

![Avoidable violation rate against the plant model used inside the safety filter, on five
building subsets. Policy class, seed count and scenario are held fixed; only the filter's
plant model varies. Five seeds per point; the rate is over the held-out half-year.](figures/fig_calibration_ladder.png)

Paired across the five scenarios on the log rate, all three comparisons now exclude zero:
exact versus linear **21.4× geometric mean, t = 6.11**, 95% CI on log₁₀ [0.73, 1.94]; exact
versus uniform **24.8×, t = 5.05**, CI [0.63, 2.16]; exact versus no filter **76.5×,
t = 3.22**, CI [0.26, 3.50]. At three subsets the uniform comparison sat at p = 0.0516 and
we could not claim it; at five it is separated. The exact inverse attains the lowest rate in
four of the five subsets.

Three observations matter more than the fold-changes. First, **an approximate filter can be
worse than no filter at all**: in subset 2 the uniform-rate and linear filters sit at 2.04e-2
and 9.00e-3 against 5.37e-4 with no filter — 38× and 17× worse than leaving the learner
unshielded. A safety filter is not a monotone improvement; it is only as good as the model
it inverts. Second, the unshielded learner's own rate ranges over 5.37e-4 to 7.69e-2 across
subsets, a factor of **143**. Any single-scenario comparison of "with shield" against
"without" is sampling from that spread, and subset 2 is the draw where the unshielded
learner matches the exact filter outright. Third, the ordering of the three *approximate*
models is not stable either — uniform beats linear in subsets 4 and 5 and loses in 1, 2 and
3 — so "a better plant model is safer" holds between exact and approximate, not within the
approximations.

### 7.3 The barrier, not the policy, is doing the work

The exact-inverse filter intervenes on **47–73%** of control steps (0.469, 0.581, 0.730,
0.682, 0.701 across the subsets). The learner proposes actions outside the admissible set
most of the time and is corrected. We state this plainly because it is the mechanism: the
reported safety of the system is a property of the filter, and attributing it to learning
would be the same category of error as §6.

Two things follow that E1 could only suggest from one scenario. The total violation-rate
contrast between the shielded learner and the calibrated rule is **−2.51e-3, CI
[−4.36e-3, −6.63e-4]** — the one safety comparison in this work whose interval excludes
zero. And it is the *rule* that fails under a binding cap, in all five subsets:
`rbc+calibrated` holds the battery band essentially perfectly and still breaches the district
cap in every subset, 11 to 122 kWh of exceedance, peaking at 47.5–58.8 kW against caps of
35–46 kW, while the shielded learner peaks at 32.8–45.6 kW and stays under. The exact
barrier is a battery-band barrier and was never a cap shield. What keeps the learner under
the cap is answered in §8, and not in the shield's favour.

### 7.4 The violation metric is not degenerate, and "unavoidable" is trajectory-dependent

A reasonable objection to any violation rate is that a policy which does nothing would score
perfectly. It does not: the `idle` arm violates the battery band on **100% of steps in all
five subsets**. Inaction strands the state of charge outside the band and cannot recover it.

Subset 2 contains exactly one building-hour, rate 2.8545e-5, that is **identical across all
six controlling arms and all five seeds** — 22 runs: rule, learner and every filter variant
alike. It is structurally infeasible and no controller is responsible for it. In the same
subset the `idle` arm's unavoidable rate is 0.125, four thousand times higher: one of eight
buildings is infeasible for the entire horizon once nothing acts, and `idle`'s total of 1.000
decomposes into 0.875 avoidable and 0.125 unavoidable. So the unavoidable set is not a fixed
property of the scenario but a function of the trajectory the controller has already
produced. A raw violation rate sums two quantities that answer different questions, and §9
includes the split as a reporting requirement.

---

## 8. Separating the mechanism from the calibration (E3)

A 4×3 grid on the reference scenario: constraint mechanism (none, Lagrangian penalty,
projection, both) × the battery model inside the filter (uniform rate, linear, exact
inverse), five seeds each, 60 records, all under the binding 33.5 kW district cap. §7
establishes that the plant model matters; this grid asks which part of the controller it
matters *through*, and settles the question §6 could not: is the shielded learner's cap
compliance the shield acting, or simply a lower peak?

| mechanism | battery model | SOC rate | cap rate | total | cost | peak [kW] | barrier intervenes |
|---|---|---|---|---|---|---|---|
| none | uniform / linear / exact | 0.710 | 0 | 0.710 | 10 015 | 30.7 | 0.000 |
| Lagrangian | uniform / linear / exact | 0.0596 | 3.20e-4 | 0.0600 | 10 972 | 33.7 | 0.000 |
| projection | uniform | 0.202 | 1.83e-4 | 0.203 | 9 905 | 36.6 | 0.471 |
| projection | linear | 0.0272 | 1.37e-4 | 0.0274 | 9 957 | 36.4 | 0.771 |
| projection | **exact** | **0** | 1.37e-4 | **2.17e-4** | **9 376** | 34.6 | 0.664 |
| both | uniform | 1.33e-3 | 9.13e-5 | 1.54e-3 | 10 264 | 35.6 | 0.081 |
| both | linear | 5.00e-3 | 9.13e-5 | 5.17e-3 | 10 513 | 35.9 | 0.727 |
| both | **exact** | **0** | 9.13e-5 | **1.88e-4** | 9 407 | 34.1 | 0.699 |

**The projection does the work; the penalty does not.** Against no mechanism at all
(0.710), the projection with an exact inverse reaches 2.17e-4 — a factor of 3 300. The
Lagrangian penalty alone reaches 0.0600: twelve times better than nothing, and 276 times
worse than the projection. Adding the penalty on top of the projection moves the total from
2.17e-4 to 1.88e-4, which is one violating building-hour and well inside the across-seed
spread, while costing 31 currency units of operating cost. On this evidence the penalty is
not what makes the system safe.

**Calibration is a property of the projection, not of the controller.** With no mechanism
and with the Lagrangian, the three battery models give **bit-identical runs** — the same
per-seed costs, rates and peaks to every digit, because nothing in those arms inverts a
plant model. The calibration ladder of §7.2 therefore is not a statement about safe RL in
general; it is a statement about the projection step, and it is inert wherever that step is
absent. We report the identity rather than hiding it: it is also the wiring check for this
grid, since any difference between those arms would have meant the calibration argument was
leaking somewhere it should not.

**The shield is not what keeps the learner under the cap — it pushes against it.** This is
the question E1 raised and could not answer, and the answer is the opposite of the
comfortable one. The *unshielded* learner has the **lowest peak of any arm, 30.7 kW against
the 33.5 kW cap, and breaches the cap in zero steps**. Every arm with a projection peaks
*above* the cap (34.1–36.6 kW) and breaches it (9.13e-5 to 1.83e-4), and the Lagrangian arm
does too (33.7 kW, 3.20e-4). Holding the battery inside its band means charging when the
band requires it rather than when the district has headroom, and that costs cap margin. So
§6's compliant learner was not being held under the cap by its shield; it was a policy whose
own peak was low, operating a shield that works on a different constraint entirely and makes
the cap slightly worse.

### 8.1 The guarantee is invariant to the architecture; the cost is not

E3 varies the mechanism with the policy class held fixed. The complementary question is
what happens when the mechanism is held fixed and the *learner* is varied as widely as the
library allows. Nine arms were run on the reference subset at 50 episodes, five seeds each,
every one carrying the identical exact-inverse barrier: four PPO variants (projection only,
projection with an intervention penalty, mean pooling, a residual policy over the rule),
SAC, MADDPG, the rule, and a receding-horizon MPC oracle for the bound.

**Eight rows, not nine — and the missing one is a defect worth stating.** A ninth arm,
`rl+calibrated+own`, carries a *forced-charge* penalty of 0.22. It produced results
bit-identical to plain `rl+calibrated` on all five seeds, in cost, peak and violation
rate alike. The cause is structural rather than statistical: that penalty is applied as
`if forced_charge_penalty and fleet_shield is not None`, and it debits
`fleet_shield.last["forced_kw"]` — a quantity only the electric-vehicle fleet shield
produces. This stock has no electric vehicles, so the fleet shield is never built, the
guard never passes, and the arm *is* `rl+calibrated` under another name. It is dropped
from the table rather than listed as an independent data point, and pinned by a test so
it cannot quietly return. This is the third term in this codebase found to be inert by
measurement rather than by reading the code, after the renewable-utilisation term and the
two non-binding power caps.

| arm | cost | above oracle | discomfort | peak kW | total viol. | **SOC viol.** |
|---|---|---|---|---|---|---|
| MPC oracle | 7 498 | — | 0.0381 | 34.1 | 7.14e-4 | **0** |
| projection + intervention penalty | 8 058 | +7.5% | **0.0109** | **33.4** | 2.00e-4 | **0** |
| residual over rule | 8 178 | +9.1% | 0.0231 | 46.9 | 3.63e-3 | **0** |
| projection only | 8 248 | +10.0% | 0.0128 | 38.8 | 2.00e-4 | **0** |
| projection, mean pooling | 8 324 | +11.0% | 0.0111 | 36.7 | 2.06e-4 | **0** |
| MADDPG | 9 399 | +25.4% | 0.0362 | 37.3 | 1.23e-3 | **0** |
| rule + barrier | 9 569 | +27.6% | 0.0381 | 51.3 | 8.08e-3 | **0** |
| SAC | 14 324 | +91.0% | 0.2512 | 44.1 | 5.48e-3 | **0** |

![Operating cost above the MPC oracle (left) against the state-of-charge violation rate
(right), for the eight distinct controllers sharing one exact-inverse barrier. Reference subset, 50
episodes, binding 33.5 kW cap, five seeds per learned arm. Cost spans a factor of twelve;
the band rate is zero in every arm.](figures/fig_architecture_invariance.png)

**The last column does not move.** Excess cost spans a factor of twelve across PPO variants,
a residual policy, an off-policy actor-critic and a multi-agent method — and the
state-of-charge violation rate is exactly zero in all of them, including the arm that is 91%
worse than the oracle and leaves occupants outside the comfort band a quarter of the time.
A learner cannot earn this guarantee and cannot damage it. It belongs to the model the filter
inverts, which is §4's claim demonstrated across architectures rather than asserted on one.

Three things follow for practice. **Adding the penalty alongside the projection is the best
architecture here** — cheaper, more comfortable and lower-peaking than the projection alone —
which reverses what E3 concluded at 15 episodes and is one more instance of a mechanism
comparison that changes sign with budget. **The residual policy inherits what the rule is bad
at**: marginally cheaper than learning from scratch, but carrying the rule's peak (46.9 kW
against 38.8) and eighteen times its violation rate. And **the choice of algorithm family
dominates every architectural detail we varied**: the gap from the best PPO variant to SAC
is 83 percentage points, while pooling and residual structure move it by one or two.

That is the sharpest form of this paper's argument. A safety filter is a guarantee about
**the constraint it inverts** — here the state-of-charge band, where the exact inverse
achieves zero across five seeds — and it is silent, or mildly counterproductive, on every
other constraint in the system. Reporting a single "safety violation rate" over a union of
constraints lets a perfect record on one dimension absorb a degradation on another.

### 8.2 The constructive counterpart: an exact filter is free, and it is not ours

Everything above is a negative result about reported zeros. The same instrument gives a
positive one, and it is the strongest measurement in this paper: **when the filter's plant
model is exact, the guarantee transfers to policies and building stocks the filter was
never tuned on, at no reliable cost.**

We bolt the same control-barrier projection — the exact battery inverse of §4, unchanged —
onto four published policies we did not write, four independent building stocks, and a
second CityLearn dataset. Every arm trains for 50 episodes under the identical objective
and budget, and each condition is normalised to its own stock's unshielded rule. Five
seeds on the reference stock, three elsewhere; **nineteen counted conditions**.

| condition | D-MAPPO | SAC | MADDPG | MetaEMS |
|---|---|---|---|---|
| reference | +0.015 | **−0.068** | +0.062 | −0.005 |
| draw 1 | −0.002 | −0.032 | +0.004 | −0.004 |
| draw 2 | +0.029 | **−0.135** | +0.035 | −0.007 |
| draw 3 | +0.001 | +0.017 | +0.010 | −0.002 |
| 2022 dataset, 17 buildings | **+0.014** | +0.001 | −0.012 | *excluded* |

*Cost effect of adding the filter, normalised to each condition's own rule. Bold =
significant at 5 %.*

| | result |
|---|---|
| conditions with violations **exactly 0.000000 %** | **15 of 19** |
| the four exceptions | every policy on draw 2, each at **identically 0.001427 %** |
| starting violation rates | 40.2 % to 94.6 % |
| mean cost effect | **−0.004**, range $[-0.135, +0.062]$ |
| mean effect against zero | $t = -0.431$, $p = 0.672$ |
| significant cells | 3 of 19 — **two in the filter's favour** |

One of the twenty measured cells is excluded, under a rule stated before the result: a
cell counts only if the *unshielded* policy actually operates the battery (mean
equivalent full cycles $> 1$, non-zero cost spread across seeds). MetaEMS on the 2022
stock fails both — 0.000 cycles, three bit-identical seeds — so its headline
100.0 % → 0.000000 % measures the filter charging a dead policy's battery into the band,
not a filter improving a controller. It would have been the largest effect in this paper.

**The four exceptions are the strongest evidence here, and they were predicted.** Draw 2
retains one violating building-step per seed under *every* policy — D-MAPPO, SAC, MADDPG
and MetaEMS, which share no code and no training procedure — at exactly the same rate,
0.001427 %, with a spread across policies of 0.0 to machine precision. The prediction came
from the battery datasheet before three of those four policies had been run: draw 2
contains a 14.0 kWh cell limited to 1.28 kW at an efficiency of 0.9601, so its largest
one-hour gain is $P_{\text{nom}}\eta/C = 1.28 \times 0.9601 / 14.0 = 0.0878$, and every episode starts at $\mathrm{SoC} = 0$, below the
0.1 bound. That battery cannot reach the set in one step whatever commands it is given.
The worst battery of each other stock can (0.1642, 0.1302, 0.1214) — and none of those
twelve cells shows any residual at all.

This is what a control-barrier function actually certifies: **forward invariance**, not
reachability from an infeasible initial condition. The residual belongs to the plant, and
four unrelated learners agreeing on it to six decimal places is a sharper demonstration of
that than any of the thirteen zeros.

**What the filter does not buy.** The cost effect is a null — not a demonstrated zero —
at these seed counts, and we report the interval rather than the point. One condition does
cost significantly (+0.014 on the 2022 set), and a deployment should expect to pay
something rather than nothing. Both arms on that dataset also sit above its rule-based
controller, so D-MAPPO does not beat a rule there at all; that is a fact about the learner,
not the filter.

### 8.3 The constrained-RL layer above the filter deactivates itself

§8.1 found that architecture does not move the guarantee. The mechanism turns out to be
blunter than "the architecture does not matter", and it is a fifth instance of this paper's
pattern — a component that measures nothing while appearing to work.

Our controller carries a PPO-Lagrangian with three constraint channels (state-of-charge
band, per-building power cap, district import cap) and a PID multiplier law. Reading the
saved multiplier traces over **15 runs and 575 episodes**:

| property | result |
|---|---|
| all three multipliers numerically identical, every episode | **15 of 15 runs** |
| λ reaches exactly 0 and never revives | **15 of 15 runs**, by episode 4 or 5 |
| maximum realised violation rate in any episode | **0.000000** |
| episodes trained with every multiplier at zero | **520 / 575 = 90.4 %** |

The projection is the cause. The PID law drives $\lambda_k$ on the constraint error
$\varepsilon_k = \bar c_k - d_k$; the projection makes the realised cost $\bar c_k$
identically zero on every channel at every update, so $\varepsilon_k < 0$ always, the
integral term clips at zero, and every multiplier returns zero. The three channels are
*numerically identical* for the same reason: all three costs are the same number.

**For 90 % of training, a controller presented as constrained RL is unconstrained PPO.**
This is not an argument that PPO-Lagrangian is useless; it is an argument that it is
redundant *behind a projection enforcing the same constraints*, which is the configuration
the safe-RL literature most often builds. A paper reporting this stack would credit the
Lagrangian for a guarantee the projection supplied, and nothing in the training curves
would contradict it.

---

## 9. A checklist for reporting constraint satisfaction

Each item would have caught one of the failures above, and each is cheap.

1. **Can the constraint bind?** Report the uncontrolled extremum of the constrained quantity
   next to the limit. If no admissible trajectory reaches it, the violation rate measures
   nothing. *(Ours: 41.88 kW peak against a 300 kW cap.)*
2. **Is the metric reachable?** Verify the violation counter can take a non-zero value for
   every arm. *(Ours: `infeasible_hour_rate` was structurally 0.000 for six of seven rules
   because the key it reads was only written by the seventh. The fix is not to write a
   zero: the rules that never solve the joint deadline set now emit no verdict, the metric
   is `None` where nothing was scored, and the number of scored hours is reported beside
   it, so "did not fail" and "never checked" are distinguishable.)*
3. **What model is inside the safety filter, and how close is it to the plant?** Report the
   projection error. *(Ours: 0% vs 9.20% breaches on the same sweep.)*
4. **Is the controller actuating the device at all?** A device never moved has a perfect
   constraint record. Report actuation counts beside violation rates.
5. **Is the constraint hard or quantile-calibrated?** If the margin is a 95th percentile,
   ~5% exceedance is the design, and the paper must say so.
6. **In-sample or held out?** State train and evaluation windows explicitly.
7. **Report counts, not only rates**, when events are rare. A rate of $9\times10^{-5}$ over
   4379 hours is one hour.
8. **When two constraints conflict, record which one you gave up.** Jointly infeasible
   limits do not announce themselves: each code path quietly picks a winner. *(Ours: the
   dead band rounds an urgent charger up past a cap that the allocator had just respected,
   while the LP holds the cap and accepts a charging shortfall — opposite resolutions of the
   same conflict, and neither was in the record. Both now report the breach in kilowatts and
   name the direction they chose.)*
9. **Was the quantity the metric scores actually measured?** A comfort, emission or
   occupancy column computed from an absent or unsimulated signal still prints a number,
   and the number is usually at one extreme — which reads as a strong result rather than
   as missing data. *(Ours, in both directions on the same column.
   `citylearn_challenge_2022_phase_all` carries an indoor-temperature column in every
   building file but runs no thermal simulation, so the observation is NaN; both band
   comparisons are False for NaN and the discomfort proportion came out **0.000**, a
   perfect comfort record for a temperature that was never simulated. Leaving the
   observation inactive instead zero-fills it, every building sits at 0 °C, and the same
   column reports **1.000**. Both are fabrications. Both are now `None`, and the
   comfort-inclusive safety union goes `None` with them rather than silently collapsing
   onto the state-of-charge-plus-equipment union.)*
10. **Does each term of a composite objective change anything?** Hold the rest fixed and
    vary the term; if no reported column moves, it is inert and should not be credited.
    *(Ours: four such terms. A renewable-utilisation weight under which three different
    controllers accumulate the identical 2495.417184178; two power caps that no trajectory
    reaches within a factor of twenty; a forced-charge penalty of 0.22 that debits a
    quantity only an electric-vehicle fleet shield produces, on a stock with no electric
    vehicles — bit-identical results with and without it on all five seeds; and the
    Lagrangian of §8.3, zero for 90 % of training.)*

---

## 10. Limitations

- The 80% cap fraction is a design choice, not a sourced engineering limit; the peaks it
  multiplies are measurements.
- Comfort is scored, not enforced, and rests on CityLearn's learned thermal model, which
  moves an indoor temperature by 4–13 °C in one hour under a −0.25 cooling action. Every
  comfort number here is labelled accordingly.
- No held-out-building evaluation exists in this work (§7).
- The calibration-ladder result (§7.2) rests on five scenarios from a single climate, all
  eight-building subsets of one Travis County dataset. The paired tests at n = 5 separate
  the exact inverse from both approximate filters (t = 6.11 against linear, t = 5.05 against
  uniform, intervals on the log rate excluding zero), where at n = 3 the uniform comparison
  did not clear 0.05. Five subsets of one neighbourhood is still one climate, one building
  stock and one tariff, and the comparison against *no* filter is the weakest of the three:
  its geometric mean is larger (76.5×) but its direction is not universal — in subset 2 the
  unshielded learner matches the exact filter outright.
- The finding that an approximate filter can be worse than no filter at all rests on **one**
  subset (§7.2). It is a counterexample to monotonicity, not an estimated rate.
- **Only §7.1 has been re-run at 50 episodes.** E1, E3, E5 and the original grid of §6 are
  all still at 15, which §7.1 shows is on the steep part of the learning curve, so every
  cost column outside §7.1 should be read as a lower bound on what the learner achieves.
  Safety results are unaffected: they are enforced by the projection and already at zero.
- 50 episodes is where the curve flattens on this scenario family; we have not run 100, so
  the −9.78% is an advantage at a usable budget, not a converged ceiling.
- The mechanism grid (§8) is a single scenario with five seeds, so every number in it is
  descriptive and no interval is reported. The across-seed spread is substantial — the
  unshielded arm's total ranges 0.548 to 0.848 over its five seeds — and the comparisons we
  draw from it are the order-of-magnitude ones, not the differences of a few violating
  building-hours. In particular, projection-with-penalty against projection-alone
  (1.88e-4 vs 2.17e-4) is one building-hour and we claim no difference there.
- Vehicle schedules are synthetic; buildings, weather and devices are CityLearn's.
- Baselines are no-control and one rule family. No model-predictive or alternative learning
  method has been run on these scenarios under a binding cap.
- Six constraint breaches in our own framework are documented and pinned by tests rather
  than fixed; the sharpest is checklist item 2.
- The per-hour cost comparison between the in-sample and held-out grids is confounded with
  seasonal composition, because the held-out window is the second half of the calendar year.

## 11. Conclusion

A constraint-violation rate is a measurement, and like any measurement it can be invalid.
We found three distinct ways our own reported zero was invalid, and the instrument that
exposed all three was an exact plant inverse we could compare everything else against.

That instrument then produced the paper's one positive result. Holding the policy class and
the seeds fixed and varying only the plant model inside the safety filter moves the
avoidable violation rate by 21.4× against a linear filter and 24.8× against a uniform one,
in the same direction in all five scenarios we ran. The cost advantage we set out to
demonstrate is also real — −9.78% against the calibrated rule, cheaper in all five subsets —
but we nearly published the opposite, because at a 15-episode budget the same grid gives
+2.10% with the sign unstable. We had written the null up before we checked the budget. The safety a shielded
learner reports is, to first order, a statement about how well the shield's model of the
plant matches the plant. In one scenario an approximate filter was 38× worse than having no
filter at all, which is worth stating in the strong form: a safety filter is not a monotone
improvement, and reporting one without reporting the fidelity of the model it inverts
describes the wrong object.

The mechanism grid sharpens that into the form we would most want a reader to carry away. A
filter is a guarantee about the constraint it inverts and nothing else: ours drives the
state-of-charge band to zero violations across five seeds while *raising* the district peak
above a cap that the unshielded learner, which peaks lowest of any arm, never breaches at
all. The safety we were reporting and the constraint we were advertising were not the same
constraint. A single violation rate taken over a union of constraints is exactly the
statistic that lets one hide inside the other.

That has a practical consequence for transfer, which is where we think this goes next. The
guarantee is a property of plant calibration, and transfer changes the plant by
construction — so moving a policy to a new building stock while leaving the shield
calibrated to the old one is the common recipe and, on this evidence, the one that quietly
removes the guarantee.

The checklist in §9 is the cheap part; building something you can falsify yourself is the
expensive part, and it is what we would ask of a reviewer in return.
