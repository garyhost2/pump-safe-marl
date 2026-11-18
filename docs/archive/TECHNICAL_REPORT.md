# Safe multi-agent control for building energy: a reproduction of STEMS Table I, a cost–comfort frontier, and three attributions

**Vehicle.** CityLearn 2.6.0b1, Travis County (Texas) neighbourhood, 8 buildings,
8760 hourly steps (Aug 2018 – Aug 2019). All runs on a 24-core workstation; no GPU
(see §11.3).

**Target.** H. Zhang, D. Wu, A. Zinflou, B. Boulet, *Spatial-Temporal Enhanced Safe
Multi-Agent Coordination for Building Energy Management*, arXiv:2510.14112v1.

Every number below is reproduced by a committed script named in §12. Numbers are quoted
from the stored records, not from run logs.

---

## 1. Problem formulation

Let $N=8$ buildings be indexed $i$, over a horizon $T=8759$ control steps of one hour.
Each building exposes an observation $o_i^t \in \mathbb{R}^{30}$ and accepts an action
$a_i^t \in [-1,1]^3$ — electrical storage, domestic hot-water storage, and the
heating/cooling device.

The problem is a **constrained Markov decision process** (CMDP). Writing $\pi_\theta$ for
the joint policy and $c_k$ for the $k$-th constraint cost,

$$
\max_{\theta}\; \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=0}^{T} \gamma^t\, r^t\right]
\quad\text{s.t.}\quad
\mathbb{E}_{\pi_\theta}\!\left[\frac{1}{T}\sum_{t=0}^{T} c_k^t\right] \le d_k,
\quad k = 1,\dots,K .
$$

with $\gamma = 0.99$ and $d_k = 0.05$ (`cost_limit`). The constraint channels are derived
from the configuration, not hand-set — `constraint_channel_names()` returns

$$
K = 3:\qquad \texttt{soc\_band},\quad \texttt{building\_power\_cap},\quad \texttt{district\_import\_cap}
$$

rising to $K=4$ (`battery_degradation`) when a degradation arm is active. Per step,

$$
c^t_{\text{soc},i} = \mathbb{1}\!\left[\mathrm{SOC}_i^t < \underline{s} \;\vee\; \mathrm{SOC}_i^t > \bar{s}\right],
\qquad \underline{s}=0.1,\ \bar{s}=0.9,
$$
$$
c^t_{\text{pow},i} = \mathbb{1}\!\left[\,|e_i^t| > P_{\text{build}}\,\right],
\qquad
c^t_{\text{grid},i} = \mathbb{1}\!\Big[\textstyle\sum_j \max(e_j^t,0) > P_{\text{grid}}\Big],
$$

where $e_i^t$ is net electricity consumption of building $i$ (kW).

---

## 2. Architecture

### 2.1 Graph construction

The adjacency is built from **measured per-building device characteristics** $f_i$
(not from coordinates — CityLearn's `Building` exposes no latitude/longitude). With
$D^2_{ij} = \lVert f_i - f_j\rVert_2^2$,

$$
w_{ij} = \exp\!\left(-\frac{D^2_{ij}}{2\sigma_f^2}\right),\qquad w_{ii} := 0,
$$

and $\sigma_f$ fixed by the **median heuristic**
$\sigma_f = \sqrt{\operatorname{median}_{i\neq j}(D^2_{ij})/2}$ rather than a constant,
so the bandwidth follows the data. Measured off-diagonal range on this stock:
$w_{ij}\in[0.0407,\,0.8384]$.

The self-loop is **structurally zero**. This matters for §10.3: a building's own state
reaches its actor through the temporal branch, not through $A$.

The ablation `mean_pool` sets $w_{ij}=1$ for $i \ne j$ — uniform pooling, the control the
feature graph must beat.

### 2.2 Spatial–temporal encoder

Matching the target paper's stated dimensions exactly (3 GCN layers × 64 hidden;
temporal attention 4 heads × 32):

$$
h = \mathrm{GCN}_{3\times 64}(x, A) \in \mathbb{R}^{N\times 64},
\qquad
z = \mathrm{Transformer}_{4\times 32}(H_{t-23:t}) \in \mathbb{R}^{N\times 32},
$$
$$
\boxed{\;r = W_s h + W_t z \in \mathbb{R}^{N \times 64}\;}
$$

an **additive fusion** of a linear projection of each branch ($W_s$ without bias, $W_t$
with). The history window is 24 hours and is *primed* at reset, never zero-padded.

Matching the encoder deliberately: if we beat their numbers with a larger network the
comparison is worthless. Any difference is then attributable to the objective, the
safety layer or the training budget.

### 2.3 Policy and critics

Actors are **shared** across buildings (indexed $b \bmod |\text{actors}|$) — the paper's
"model sharing". Each actor emits a diagonal Gaussian in a **pre-tanh** variable $u$:

$$
u_i \sim \mathcal{N}\!\big(\mu(r_i),\ \operatorname{diag}\sigma(r_i)^2\big),
\qquad a_i = \tanh(u_i).
$$

Observations are normalised by a running normaliser that is **fitted once and then
frozen** (`obs_normalizer.count == 0` guard), so the input scale does not drift during
training.

---

## 3. The objective

`stems/reward.py`, per building $i$, with $e_i$ the *next*-step net consumption and
$p$ the price:

$$
\begin{aligned}
r_i &= \underbrace{-\mu\, p\, \max(e_i, 0)}_{\text{cost}}
\;\underbrace{-\,\alpha_{\text{grid}}\Big(\tfrac{\sum_j \max(e_j,0)}{P_{\text{grid}}}\Big)^{2}}_{\text{district peak}}
\;\underbrace{-\,\alpha_{\text{build}} \tfrac{|e_i|}{P_{\text{build}}}}_{\text{building load}} \\[4pt]
&\quad \underbrace{-\,\beta_{\text{ramp}} \tfrac{|e_i - e_i^{\text{prev}}|}{P_{\text{build}}}}_{\text{ramping}}
\;\underbrace{-\,\lambda_{\text{indoor}}\,\phi(T_i)}_{\text{comfort}}
\;\underbrace{+\,\xi \min\!\big(\tfrac{\text{solar}_i}{\text{load}_i},1\big)}_{\text{renewables}}
\;-\; r^{\text{EV}}_i .
\end{aligned}
$$

Defaults: $\mu=1.0$, $\alpha_{\text{grid}}=0.5$, $\alpha_{\text{build}}=0.3$,
$\beta_{\text{ramp}}=0.2$, $\lambda_{\text{indoor}}=0.4$, $\xi=0.0$,
$T_{\text{ref}}=22^\circ$C with a $\pm2^\circ$C dead-band.

### 3.1 The caps do double duty — a measured consequence

$P_{\text{grid}}$ and $P_{\text{build}}$ are **both** the constraint thresholds of §1
**and** the normalisers of three reward terms. Loosening them to match the target
paper's constraint set (which has no district cap) therefore also changes the objective.
Measured by accumulating each term separately over a 2000-step rollout of the calibrated
rule:

| term | binding caps (33.5 / 11.5 kW) | non-binding caps (2000 / 500 kW) |
|---|---|---|
| cost | 45.91 % | **58.23 %** |
| comfort | 32.82 % | **41.49 %** |
| district peak | 12.04 % | 0.00 % |
| building load | 6.49 % | 0.19 % |
| ramping | 2.74 % | 0.08 % |

The three stability terms collapse from **21 %** of the objective to **0.27 %**. In the
reproduction runs the objective is therefore *cost versus comfort and essentially
nothing else*. This is stated rather than hidden, and it is why §8.2 sweeps
$\lambda_{\text{indoor}}$.

---

## 4. Safety machinery

Two mechanisms, separable and separately measured (§10.1).

### 4.1 Control-barrier projection with an exact plant inverse

Given the state of charge $s$ and the band $[\underline{s},\bar{s}]$, the shield computes
the **exact** admissible action interval by inverting the battery model:

$$
\mathcal{A}(s) = \{\,a : \underline{s} \le g(s,a) \le \bar{s}\,\},
\qquad a^{\text{safe}} = \Pi_{\mathcal{A}(s)}\big(a^{\text{nom}}\big),
$$

where $g$ is the plant's SOC transition. Because $g$ is monotone in $a$ but not
analytically invertible, the endpoints are found by **24-step bisection** on a scalar —
not a learned approximation:

$$
a_{\text{lo}} = \inf\{a : g(s,a) \ge \underline{s}\},\qquad
a_{\text{hi}} = \sup\{a : g(s,a) \le \bar{s}\}.
$$

The **calibration ladder** replaces $g$ with successively cruder surrogates —
`none`, `uniform`, `linear`, `exact` — and is the axis the target paper does not have.

### 4.2 PPO-Lagrangian with a PID multiplier

The advantage is combined across constraint channels as

$$
A_{\text{eff}} = \frac{A - \sum_{k} \lambda_k A^{c}_k}{1 + \sum_k \lambda_k},
$$

the denominator keeping $A_{\text{eff}}$ invariant to positive rescaling. Multipliers are
driven by a **PID** law on the constraint error $\varepsilon_k = \bar c_k - d_k$:

$$
I_k \leftarrow \mathrm{clip}\big(I_k + k_i \varepsilon_k,\ 0,\ \lambda_{\max}\big),
\qquad
\lambda_k = \mathrm{clip}\big(k_p \varepsilon_k + I_k + k_d [\bar c_k - \bar c_k^{\text{prev}}]^{+},\ 0,\ \lambda_{\max}\big)
$$

with $k_p=1.0$, $k_i=0.3$, $k_d=0.0$, $\lambda_{\max}=10$. FOCOPS is implemented as an
alternative family.

**Credit assignment.** The buffer stores the *pre-projection* action and its log-prob, so
PPO treats the projection as part of the environment. This is self-consistent but the
projection is many-to-one.

---

## 5. Behaviour-cloning warm start

The one mechanism built new in this work. Let $\mathcal{D}=\{(o^t, H^t, a^t_{\text{teacher}})\}$
be rollouts of the calibrated rule. The encoder and actors are fitted by maximum
likelihood **in the pre-tanh variable**, scoring only controlled dimensions $\mathcal{C}$:

$$
\mathcal{L}_{\text{BC}}(\theta) = -\frac{1}{|\mathcal{D}|\,N}\sum_{t}\sum_{i}
\sum_{c \in \mathcal{C}} \log \mathcal{N}\!\Big(\operatorname{atanh}(a^t_{\text{teacher},i,c})
\;\Big|\; \mu_c(r_i^t), \sigma_c(r_i^t)\Big).
$$

Three choices, each with a reason:

1. **Clone the commanded action, not the shielded one.** The shield is active when the
   student runs; cloning the post-projection action would teach the policy to pre-undo a
   correction that is about to be applied to it again.
2. **Critics stay at initialisation.** A critic warmed on the teacher's returns biases the
   first advantage estimates toward the teacher rather than merely starting from it.
3. **Reset the spread afterwards.** This is load-bearing and was found by measurement.

### 5.1 The log-std collapse

Maximum-likelihood fitting of a near-deterministic teacher drives $\log\sigma$ to its
floor. Measured: $\log\sigma = -2.423$ ($\sigma \approx 0.089$). PPO's KL early-stop
(`target_kl = 0.02`) then fires on the first minibatch:

| | fitted spread kept | spread reset to $\log\sigma=-1$ |
|---|---|---|
| entropy, episode 1 | $-2.979$ | $+1.228$ |
| approximate KL | $1.619$ | $0.0078$ |
| clip fraction | $0.400$ | $0.097$ |
| **gradient steps / episode** | **2** | **110** |

Without the reset the warm-started arm is a cloned policy that then *barely trains* — it
would have read as "behaviour cloning does not help" rather than "the warm start broke
the optimiser". The fitted value is recorded alongside the reset one so the collapse
stays visible in the run record.

---

## 6. Evaluation metrics, as the target paper defines them

Implemented in `stems/stems_table.py` from Section III-A item 3 of arXiv:2510.14112.

$$
\text{Discomfort } D = \frac{1}{T N}\sum_{t=1}^{T}\sum_{i=1}^{N}
\mathbb{1}\big[T_{i,t} < 20^\circ\mathrm{C} \;\vee\; T_{i,t} > 26^\circ\mathrm{C}\big]
$$

— a **fixed band, no occupancy mask, no set point**, and explicitly *not* normalised.

$$
\text{Ramping} = \sum_{t=2}^{T} \big| E^t - E^{t-1} \big|,
\qquad
\text{Avg. daily peak} = \frac{1}{365}\sum_{d=1}^{365} \max_{t \in d} E^t
$$

where $E^t$ is total district load. The daily peak is a plain mean of **signed** daily
maxima (not floored at zero).

Five columns — cost, emission, average daily peak, electricity consumption, ramping —
are normalised to a rule-based baseline. Since the paper's Rule-Based row reports a
safety violation rate of **0.351**, that baseline is an **unshielded** rule; we therefore
normalise to `rbc`, not to `rbc+calibrated`. Normalising to the shielded rule would
divide by a stronger baseline and silently shrink every improvement.

The paper gives **no demand-charge rate**, so `stems_cost` is an energy cost only and the
module refuses to invent one.

---

## 7. Reproduction of Table I

Same scenario, same metric definitions, same normalisation, 5 seeds per learned method,
evaluated in-sample on the training year as their twelve-month table reports. **The
comparison methods carry no safety layer of ours**, which is what makes them comparable
to a published table whose baselines violate at 15.5–33.0 %; §7.2 records what happened
when they did.

**7.0 The headline, and a budget error of ours that hid it for six weeks.** Every
reproduction claim in §7.1–§7.6 below was measured at **15 episodes**, because STEMS
states that training "converges within 15 epochs". That reading was wrong. Their epoch is
about three minutes on an RTX 5090; our episode is one 8760-step simulated year, which
costs 285–1160 s of single-threaded CPU and 8759 gradient steps. Nothing licenses setting
one equal to the other, and the 15-episode grid itself says so: **30 of 30 seeds still had
a positive reward slope at the cutoff** (§7.3), which is the signature of a budget that
ended before the method did, not of a method that disagrees with its paper.

A 50-episode grid under identical conditions had already been run for §8 and was never
compared against the published table. It reproduces:

| method | our cost | published | err | our emission | published | our electricity | published |
|---|---|---|---|---|---|---|---|
| **MARLISA** | **0.847** | **0.847** | **+0.0 %** | 0.872 | 0.875 | 0.880 | 0.838 |
| **D-MAPPO** | 0.817 | 0.803 | +1.8 % | 0.828 | 0.834 | 0.832 | 0.816 |
| **MADCQ** | 0.833 | 0.814 | +2.3 % | 0.856 | 0.837 | 0.863 | 0.823 |
| MPC | 0.979 | 0.872 | +12.2 % | 0.957 | 0.914 | 0.947 | 0.975 |
| MADDPG | 1.047 | 0.795 | +31.7 % | 0.996 | 0.817 | 0.996 | 0.856 |
| MetaEMS | 1.327 | 0.836 | +58.7 % | 1.163 | 0.804 | 1.147 | 0.835 |
| SAC | 1.573 | 0.824 | +90.9 % | 1.392 | 0.867 | 1.388 | 0.813 |

![**(a)** Our 50-episode reimplementations against the published Table I cost column,
normalised to the same unshielded rule; bars are the mean ± s.d. over 5 seeds (MPC is
deterministic and is omitted here, being a non-learner for which budget is not an
excuse). **(b)** The same six methods against the fraction of building-steps on which
their saved policy commands a saturated action, measured by replaying each policy for
one deterministic year
(`experiments/diagnostics/action_profile.py`).]({{artifact:art_d834b8c5-e984-42c7-80e2-f4ca2dcca881}})

Three of six learning methods land within 2.3 % of their published cost, and do so on
emission and electricity consumption at the same time. MARLISA's five seeds are 0.8285,
0.8368, 0.8432, 0.8617, 0.8668 — mean **0.8474**, sd 0.0146, against a published 0.847.
All seven rows are unshielded (`barrier="none", mechanism="none"`, intervention rate 0.000
on every seed), normalised to the same unshielded rule, same eight buildings, same
patches, `lambda_indoor` 0.4 throughout; the three grids differ in no condition except
episode count.

Peak and ramping do **not** track the published column, and the direction splits by
controller rather than uniformly: the five arms that converge — MPC, MADDPG, MARLISA,
MADCQ, D-MAPPO — come in 17–27 % below the published avg-daily-peak and 11–27 % below
published ramping, while SAC and MetaEMS, the two arms §7.0a shows are not trained, come
in 20.0 % and 28.1 % *above* on peak and 22.3 % and 13.3 % above on ramping. An earlier
draft of this paragraph called the offset uniform and unexplainable by controller
quality; the same table it cited contradicts that, and the claim is withdrawn.

**7.0a Why the other three fail, measured rather than asserted.** The reward traces
separate three causes that the cost column alone cannot:

| arm | gradient steps / episode | reward ep1 → ep50 | slope at ep50 | reading |
|---|---|---|---|---|
| MARLISA | 8759 | −13 604 → −2 879 | +14.8 | converged |
| MADCQ | 8759 | −9 758 → −3 257 | −0.7 | converged |
| D-MAPPO | 1370 | −9 552 → −3 442 | +24.3 | converged |
| MADDPG | 8759 | −13 365 → −5 985 | **+39.6** | undertrained |
| MetaEMS | **64** | −14 692 → −7 773 | +15.7 | gradient-starved |
| SAC | 8759 | −14 171 → **−7 845** | +8.1 | converged to a bad optimum |

*MetaEMS* is a defect in our implementation and is reported as one. `MetaEMSAgent.update`
runs `B × inner_steps = 8 × 8 = 64` gradient steps and is called once per episode, so over
50 episodes it takes **3 200** steps against every other learner's 437 950 — 0.7 % of the
budget it is being compared at. Its saved policy is still at initialisation scale
(mean-head weight magnitude 0.032 against SAC's 0.098, log-std bias −0.012, i.e. unit
variance) and its entropy temperature is frozen at the initial 0.2 because the inner SAC
is built with `autotune_alpha=False`. It also barely operates the battery: 61.9
battery-equivalent full cycles against 233–377 for every other arm, failing the
actuator-responsiveness check on 4 of 5 seeds. Its published row is not evidence against
the paper.

*MADDPG* is undertrained rather than broken: +39.6 reward per episode at the cutoff, and
it moved 1.269 → 1.047 on budget alone between the two grids.

*Our first explanation of SAC's saturation was wrong, and the experiment that refuted it
found the real one.* We hypothesised the missing pre-squash regulariser of the reference
soft actor-critic — the policy loss is $(\alpha\log\pi - Q)$ with no penalty on the
pre-tanh Gaussian parameters, so nothing bounds the mean, and a saturated $\tanh$ has
vanishing Jacobian. Implemented, swept over four orders of magnitude at a 10-episode
budget, one seed:

| `policy_reg` | mean saturation | cost |
|---|---|---|
| none (50 ep) | 0.974 | 1.573 |
| $10^{-3}$ | 0.987 | 1.569 |
| $10^{-2}$ | 0.985 | 1.561 |
| $10^{-1}$ | 0.986 | 1.562 |
| $1$ | 0.979 | 1.612 |

Nothing. **The hypothesis is refuted** and the term is kept only because it is correct
on its own merits (it defaults to inert). The sweep is reported rather than deleted
because it is what licensed looking elsewhere.

*What is actually wrong is the critic, and it is measurable.*
`experiments/diagnostics/critic_sensitivity.py` freezes a trained policy's deterministic
action, sweeps the battery channel across its full range, and records the twin-critic
minimum. For the 50-episode seed-0 policies:

| arm | $\lvert Q\rvert$ | $Q$ span over the action range | relative sensitivity | argmax | published-cost error |
|---|---|---|---|---|---|
| Single-agent SAC | 71 422 | 4.58 | $6.4\times10^{-5}$ | $+1.00$, a corner | +90.9 % |
| MARLISA | 58.0 | 0.216 | $3.7\times10^{-3}$ | $-0.85$, interior | +0.0 % |

SAC's critic is **1231× larger in magnitude and 58× less action-sensitive** than the
critic of the arm that reproduces.

*Value scaling was the obvious next hypothesis. It is also refuted.* `DMAPPOAgent`
carries a return normaliser whose docstring predicts this failure mode in advance;
`SingleAgentSAC` had none. Adding it (`--sac-scale-returns`) collapses the magnitude by
78× and changes nothing that matters:

Every column below is **seed 0**, because the critic being probed is one saved network
and a five-seed mean does not have a critic. The seed-0 costs therefore differ slightly
from the five-seed means quoted in §7.0 (SAC 1.557 here against 1.573 there, MARLISA
0.862 against 0.847); an earlier draft of this table mixed the two, pairing seed-0
critic measurements with five-seed costs.

| variant | episodes | $\lvert Q\rvert$ | $Q$ span | relative sensitivity | argmax | saturation | cost, seed 0 |
|---|---|---|---|---|---|---|---|
| MARLISA — reproduces | 50 | 58.0 | 0.216 | $3.7\times10^{-3}$ | $-0.85$ interior | 0.024 | 0.862 |
| independent SAC | 15 | 2093 | 0.434 | $2.1\times10^{-4}$ | corner, 8 of 8 | — | 1.590 |
| single-agent SAC | 50 | 71 422 | 4.578 | $6.4\times10^{-5}$ | $+1.00$ corner | 0.974 | 1.557 |
| single-agent SAC, scaled | 12 | 913 | 0.054 | $6.0\times10^{-5}$ | $-1.00$ corner | 0.975 | 1.609 |

**What survives two refutations is one quantity.** Relative action sensitivity separates
the arm that reproduces from every arm that does not, by 18× at the narrowest, and it is
the one number neither intervention moved: the pre-squash penalty left it alone across
four orders of magnitude, and return scaling left it at $6\times10^{-5}$ while cutting
$\lvert Q\rvert$ by 78×. Magnitude is not the problem; the policy parameterisation is not
the problem.

The argmax column tracks it exactly: **every failing variant's critic is maximised at a
corner — all eight of independent SAC's sub-critics, at $[+1, -1, -1, +1, -1, -1, -1,
-1]$ — while the reproducing one's sits at $-0.85$, inside the range.** An earlier draft
of this paragraph reported independent SAC's optimum as "interior, $-0.50$" and used it
to refute the corner reading. That $-0.50$ was the *mean* over eight sub-critics sitting
at opposite corners, a value none of them has; `critic_sensitivity.py` now reports the
fraction at a corner and keeps the per-agent list so the average cannot be quoted alone.

**The causal direction is still not established, and that decides what to try next.** The
reading consistent with all four rows is a loop rather than a one-way cause: a policy that
only ever executes saturated actions fills its buffer with saturated actions, so its
critic never observes the interior, cannot learn action dependence there, and returns no
gradient that would pull the policy back in. That would explain why two interventions on
the *loss* — a penalty on the policy's parameters, a rescaling of the critic's target —
changed neither the sensitivity nor the cost. If it is right, the experiment that breaks
it is an exploration intervention: `start_steps` is 256 against 437 950 gradient steps.
That has not been run and nothing is claimed for it.

MetaEMS, whose fix is the gradient budget rather than the penalty, did move: 92.4 %
saturated to 78.4 % over five episodes.

*MPC is not a learner, so budget is no excuse for its +12.2 %, and it has a different and
complete explanation: the forecaster, which their paper does not specify.* Running our own
MPC twice on the identical scenario and changing nothing but the forecast — the causal
forecaster it normally uses, against a perfect-foresight tape recorded by rolling the same
environment once under the zero action — brackets their published row on every column:

| MPC variant | cost | emission | avg. daily peak | electricity | ramping |
|---|---|---|---|---|---|
| ours, causal forecaster | 0.979 | 0.957 | 0.816 | 0.947 | 0.852 |
| **published** | **0.872** | **0.914** | **0.983** | **0.975** | **0.981** |
| ours, perfect foresight | 0.770 | 0.788 | 0.739 | 0.787 | 0.676 |

Perfect foresight is worth −0.209 on cost, and their value sits 0.107 **below** our
causal version and 0.102 **above** our oracle — within a point of the midpoint.

**The bracket holds on two of the five columns, and the other three say something
different.** Cost (0.770 < 0.872 < 0.979) and emission (0.788 < 0.914 < 0.957) are
bracketed. Avg. daily peak (published 0.983 against our 0.816 and 0.739), electricity
consumption (0.975 against 0.947 and 0.787) and ramping (0.981 against 0.852 and 0.676)
are not: their MPC is worse than *both* of our forecaster settings on all three, and no
choice of forecaster inside this implementation reaches it. An earlier version of this
paragraph claimed the ordering held on electricity, which the table above contradicts.

We therefore do not report MPC as a failure to reproduce **on cost**: a receding-horizon
controller's cost is a function of its forecast, that function spans the published value
in our hands, and the paper gives no forecaster to match. Its peak and ramping columns
remain unexplained, and sit with the general pattern of §7.0 — the controllers we run
flatten the district profile more than the published ones do. The oracle row is **not** a result — it is an upper bound built from information no
controller has, and it is reported only to establish the bracket.

*SAC* is the only one of the three that is neither starved nor still climbing. It spends
437 950 gradient steps, flattens by episode 5, and settles at a reward 2.7× worse than
MARLISA, which is also a soft actor-critic evaluated deterministically with an annealed
temperature (α = 0.019 for SAC, 0.0084 for MARLISA). Its deficit is energy, not comfort:
1.388× the rule's electricity against MARLISA's 0.880, at a comparable discomfort. A
decentralised variant with per-building networks (§7.6) did not close it. This one we
have diagnosed to a behaviour but not yet to a cause, and it is marked open.

**The 15-episode grid**, kept in full because §7.2–§7.6 are measured on it and because
the 15-to-50 difference is itself the finding. Read it against §7.0, not on its own. Its
discomfort column is the fixed 20–26 °C band, which §9.1 withdraws in favour of the
occupied set-point band their paper defines.

| method | source | cost | emission | peak | consumption | ramping | discomfort | safety % |
|---|---|---|---|---|---|---|---|---|
| Rule-Based | measured | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.446 | 31.63 |
| Rule-Based | published | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.130 | 35.10 |
| MPC | measured | 0.979 | 0.957 | 0.816 | 0.947 | 0.852 | 0.446 | 0.45 |
| MPC | published | 0.872 | 0.914 | 0.983 | 0.975 | 0.981 | 0.654 | 33.00 |
| Single-Agent SAC | measured | 1.581 | 1.365 | 1.065 | 1.355 | 1.048 | 0.427 | 81.38 |
| Single-Agent SAC | published | 0.824 | 0.867 | 0.856 | 0.813 | 0.925 | 0.485 | 22.30 |
| MADDPG | measured | 1.269 | 1.192 | 0.867 | 1.205 | 0.959 | 0.418 | 70.53 |
| MADDPG | published | 0.795 | 0.817 | 0.854 | 0.856 | 0.936 | 0.357 | 19.70 |
| MetaEMS | measured | 1.310 | 1.147 | 0.978 | 1.136 | 0.973 | 0.435 | 94.84 |
| MetaEMS | published | 0.836 | 0.804 | 0.884 | 0.835 | 0.907 | 0.396 | 23.10 |
| MARLISA | measured | 0.962 | 0.946 | 0.718 | 0.955 | 0.838 | 0.412 | 57.64 |
| MARLISA | published | 0.847 | 0.875 | 0.826 | 0.838 | 0.956 | 0.237 | 21.40 |
| MADCQ | measured | 0.879 | 0.907 | 0.767 | 0.917 | 0.892 | 0.424 | 0.00 |
| MADCQ | published | 0.814 | 0.837 | 0.862 | 0.823 | 0.906 | 0.325 | 15.50 |
| D-MAPPO | measured | 0.998 | 0.927 | 0.667 | 0.928 | 0.700 | 0.405 | 67.61 |
| D-MAPPO | published | 0.803 | 0.834 | 0.841 | 0.816 | 0.912 | 0.152 | 19.80 |
| **ours** (shielded) | measured | 1.011 | 0.933 | 0.710 | 0.933 | 0.721 | 0.405 | 0.00 |
| STEMS | published | 0.792 | 0.824 | 0.821 | 0.805 | 0.883 | 0.132 | 5.60 |

![Our unshielded reimplementations against STEMS Table I. Left: every method we
reimplement costs more than published; the rule-based normaliser and the no-control
reference bracket the range. Right: their seven baselines occupy a narrow violation band
while ours split by algorithm class. 15 episodes, 5 seeds per learned method (MPC is
deterministic and runs once), normalised to the unshielded
rule.]({{artifact:art_ff06f675-850c-4696-8742-67a6ce8c0966}})

**7.1 The baseline reproduces.** Unshielded rule: **31.63 %** against their **35.10 %**,
on a metric neither implementation tuned for. That is close enough to believe the same
baseline is running in both places, and it is the only row of the table that reproduces.

**7.2 A defect in our harness, and its correction.** An earlier version of this section
reported that no cost figure reproduced, eight for eight, and offered the cost definition,
the tariff or the strength of the normalising rule as candidate causes. That table was
measuring the wrong thing. Every comparison method was declared with
`barrier="calibrated"` and therefore ran *inside our own safety projection*; only the rule
ran unshielded. Two consequences. The safety column read 0.00 for all eight rows, which
deletes the comparison their table exists to make. And the shield was intervening at rates
from 0.65 % (MADCQ) to 96.9 % (MetaEMS), so those rows were not measurements of those
algorithms at all.

The rank correlation between a method's intervention rate and its gap to the published
cost was Spearman *ρ* = 0.929 (*p* = 0.0025, n = 7), which suggested the shield was also
the cause of the cost gap. **It is not.** Re-running all seven with
`barrier="none", mechanism="none"` moves cost by a mean of +0.012, range [−0.021, +0.063],
with mixed signs — MARLISA improves, SAC worsens:

| method | shield fired | published | shielded | unshielded | shield effect | gap |
|---|---|---|---|---|---|---|
| MADCQ | 0.65 % | 0.814 | 0.881 | 0.879 | −0.002 | +0.065 |
| MPC | 0.00 % | 0.872 | 0.979 | 0.979 | −0.000 | +0.107 |
| MARLISA | 62.73 % | 0.847 | 0.983 | 0.962 | −0.021 | +0.115 |
| D-MAPPO | 64.58 % | 0.803 | 0.985 | 0.998 | +0.014 | +0.195 |
| MADDPG | 79.06 % | 0.795 | 1.233 | 1.269 | +0.037 | +0.474 |
| MetaEMS | 96.86 % | 0.836 | 1.315 | 1.310 | −0.005 | +0.474 |
| SAC | 85.45 % | 0.824 | 1.518 | 1.581 | +0.063 | +0.757 |

Against gaps of +0.065 to +0.757, an effect of ±0.06 explains none of it. The correlation
was a confound with the causation reversed: a policy overridden on 85 % of steps is a
policy whose raw proposals are bad, so the intervention rate *measures* a weak controller
rather than producing one. It was computed across arms of differing quality, never across
a manipulation of the shield itself — which is the experiment that tests it. A second
claim fell with it: MetaEMS's 36.9 battery-equivalent full cycles were read as the shield
clamping it into near-inaction, but unshielded it does 37.7, so that is a property of the
policy.

Both the shielded and the unshielded arms are kept. Sharing one shield across methods is a
valid controlled comparison of algorithms; it is simply not a reproduction of this table.

**7.3 What does not reproduce at 15 episodes, and what that turned out to mean.**
*Superseded by §7.0: at 50 episodes MARLISA, D-MAPPO and MADCQ reproduce to within 2.3 %.
The paragraph is kept because its diagnosis was correct and was not acted on — it
identifies the budget as the cause, measures the headroom, and asks for the matched-budget
grid that §7.0 reports.* At 15 episodes our reimplementations of all seven
comparison methods are more expensive than published, 7 for 7, by +0.065 to +0.757, and
the shield is not why. The evidence says they are undertrained rather than broken: across
the six learning methods at 5 seeds each, **30 of 30 seeds have a positive episode-reward
slope at the 15-episode cutoff and not one curve has flattened** (MARLISA +378 per
episode, MADCQ +368, D-MAPPO +313, MADDPG +249, SAC +176, MetaEMS +151). Our own arm moved
1.011 → 0.896 on budget alone, so the headroom is measured rather than assumed. This is a
limitation of our reproduction. It is not evidence that the published figures are wrong.

It also exposes a fairness defect in our own comparison, which §8.3 must be read against:
the headline 0.718 is a 50-episode warm-started run while every baseline here is 15
episodes and cold. A matched-budget grid is required before claiming to beat these rows.

**7.4 The safety column does not reproduce either, and fails in both directions.** Their
seven baselines occupy a narrow 15.5–33.0 %. Ours span 0.0–94.8 % and split by algorithm
class. The value-based and model-based methods come out far *safer* than published — MADCQ
0.00 % against 15.50 %, MPC 0.45 % against 33.00 % — while the five policy-gradient and
meta-learning methods come out far more dangerous: MARLISA 57.64 % against 21.40 %,
D-MAPPO 67.61 % against 19.80 %, MADDPG 70.53 % against 19.70 %, SAC 81.38 % against
22.30 %, MetaEMS 94.84 % against 23.10 %. Two directions rule out a scale factor on the
metric. A published MPC that violates a state-of-charge bound on a third of its steps,
when ours carrying that bound inside its own program violates on 0.45 %, is the single
hardest number in their table to account for.

**7.5 Five explanations tested and set aside.**

1. *The "12-Month Average Results" caption.* A mean of twelve monthly ratios is not the
   ratio of the annual totals — the latter is consumption-weighted, so on a
   cooling-dominated Texas stock it is close to a summer-only measurement. Both estimators
   were computed from a single rollout per arm through `MetricsCalculator.slice_steps`, so
   the same code produces each. They differ by at most **0.034** on any column and
   **0.008** on cost. Not the explanation.
2. *The building sample.* Their eight buildings are "randomly selected from the Travis
   County dataset" and are never listed, and our own replication puts the cost ratio at
   0.718, 0.732, 0.771 and 0.783 across four draws — a spread of 0.065 against the 0.080
   separating their best and worst published row, so the draw alone can reorder their
   table. The stock they describe may also not be drawable from the dataset they name:
   they specify 5 residential, 2 commercial and 1 mixed-use, while CityLearn's
   `tx_travis_county_neighborhood` holds exactly 100 building files, every one
   `resstock-amy2018-2021-release-1-*`. NREL ResStock is a residential stock model, and
   none of the 25 CityLearn datasets carries a commercial-sounding name.
3. *The rule-based normaliser.* Every ratio in their table is divided by a rule-based
   controller, and CityLearn ships one — `BasicRBC` — whose behaviour we found to be
   defective once the HVAC action is enabled (§11, and `docs/CITYLEARN_DATASET_AUDIT.md`): on the eight Travis buildings it
   draws **3.15×** the electricity of doing nothing and ends up *more* uncomfortable than
   doing nothing (0.771 against 0.400). A pathological denominator inflates every
   numerator's ratio, so the conjecture was that their rule is this one and that the gap
   is arithmetic rather than methodological. **Measured and ruled out.** CityLearn's own
   reference SAC, run in stock CityLearn with the HVAC action active and scored against
   `BasicRBC` by the same metric code, costs **0.153 and 0.162** on two seeds (8.4 h and
   7.1 h of single-threaded training for 15 episodes each). They publish 0.824 for
   Single-Agent SAC. A denominator that puts the library's own SAC five times below their
   figure cannot be the denominator they used, so the baseline gap is not the normaliser.
   The reference run also reproduces the `BasicRBC` defect independently — its optimised
   variant lands at 0.991, i.e. the two rules are interchangeable as denominators — and
   it violates the state-of-charge bound on 44.5 % and 56.6 % of steps, which is itself
   outside their 15.5–33.0 % band for every baseline.

   **A second reference implementation agrees.** CityLearn's own MARLISA, 15 episodes,
   9.7 h of single-threaded training, scored by the same code against the same
   denominator, lands at **0.191** against their published 0.847 for MARLISA. So two
   independent reference implementations of two different methods, over three runs, sit
   **4.4× to 5.4× below** the figures STEMS publish for those same methods:

   | method | CityLearn reference, measured | STEMS published | ratio |
   |---|---|---|---|
   | Single-Agent SAC | 0.153, 0.162 | 0.824 | 5.2× |
   | MARLISA | 0.191 | 0.847 | 4.4× |

   A denominator cannot be 4–5× wrong in one direction for two unrelated methods and
   still be the denominator that produced their table. This closes the conjecture.
4. *HVAC action semantics.* Our grids run `hvac_control="setpoint"`, under which doing
   nothing costs 22 981 at discomfort 0.446. Under `"power"` the same idle controller
   costs 13 767 at discomfort 0.717, because a zero action means "command the neutral set
   point" in one convention and "no HVAC power" in the other — a 40 % shift in the cost
   scale from a convention the paper does not state.

**7.6 A fifth explanation, tested and set aside: we implemented the wrong method.**

Two of our seven baselines do not implement what STEMS describe. Their *Single-Agent SAC*
row is "Independent Soft Actor-Critic reinforcement learning for **each building** without
coordination", while our `sac` is a **centralised** controller over a 240-dimensional joint
observation and a 24-dimensional joint action — its own docstring says so. (Our `MetaEMS`
is likewise Reptile where they specify MAML; that one is not retested here.)

So we built `isac`: $B$ disjoint SAC agents sharing no parameters, replay buffers,
gradients or observation statistics, each seeing only $o_b$, acting only through $a_b$ and
maximising only $r_b$. The per-building agent is the same `SingleAgentSAC` with
`num_buildings=1`, so the two arms differ in decomposition and in nothing else. 15
episodes, 5 seeds, unshielded, on the reference stock:

| arm | cost | peak | ramping | emission | consumption | discomfort | SoC viol. | cycles |
|---|---|---|---|---|---|---|---|---|
| `isac` — independent, as described | 1.563 | 1.024 | 1.079 | 1.386 | 1.381 | 0.430 | 68.2 % | 385.1 |
| `sac` — centralised, what we had | 1.581 | 1.065 | 1.048 | 1.365 | 1.355 | 0.427 | 81.4 % | 236.9 |
| **STEMS published** | **0.824** | 0.821 | 0.883 | 0.824 | 0.805 | 0.132 | 22.3 % | — |

**Implementing the method they describe moves cost from 1.581 to 1.563.** The gap to 0.824
is 1.9×, and correcting the decomposition closes about 2 % of it. The independent learner
is the more active of the two (385 against 237 equivalent full cycles) and marginally
cheaper, so this is a working implementation rather than a broken one — it simply does not
reproduce their number either. Both arms are kept: `sac` is a legitimate centralised
reference point whose state and action spaces grow with $B$, and `isac` is the row their
table actually describes.

The shield behaves on it exactly as everywhere else: **68.2 % → 0.000000 %** violations at
a cost effect of **−0.050** ($p = 0.232$, a null in the favourable direction). This arm is
at 15 episodes rather than 50, so it is reported here and not folded into the
seventeen-condition matrix of §8.3h.

---

**7.7 Their own method, reimplemented from their own paper.** Everything above
concerns the eight *comparison* rows. The ninth — STEMS itself, 0.792 — we had never
attempted, and said so. `stems/paper_stems.py` now implements their Algorithms 1 and 2
and nothing else. It deliberately does not reuse `stems/agent.py`: that grew a
multi-Lagrangian, a behaviour-cloning warm start, a residual action head and a
centralised critic, none of which appear in their paper, and §9.4 measures the first of
them switching itself off for 90.4 % of training.

Every quantity their §III-A4 states is matched rather than chosen: the edge kernel
$w_{ij} = \alpha e^{-d_{ij}^2/2\sigma_d^2} + \beta e^{-\lVert f_i - f_j\rVert^2/2\sigma_f^2}$
of their eq. 11, a 3-layer 64-unit graph convolution with their
$\sqrt{w_{ij}/d_id_j}$ propagation (eq. 12), 4-head 32-dimensional temporal attention
(eq. 13–14), the fusion $r_i = W_s h_i^{(L)} + W_t z_i + b$ (eq. 15), a **per-building**
actor and critic of two 128-unit layers (eq. 22–23), Adam at $3\times10^{-4}$,
$\gamma = 0.99$, $\xi = 0.6$, and their three control barriers with the action projected
before execution **and before storage** (Alg. 1; Alg. 2 line 8).

Three quantities their paper does not fix are resolved explicitly rather than silently,
and each is a place a reproduction can legitimately differ:

1. **The policy-gradient form.** Algorithm 2 line 11 is
   $\theta \leftarrow \theta + \alpha_\pi \nabla_\theta J(\theta)$ — vanilla advantage
   policy gradient, one step per episode. One step per 8760-step episode cannot train
   anything in 15 epochs, so minibatching is implied and unstated. The default is a
   single on-policy pass; `clip_eps` switches to the standard clipped PPO surrogate.
   **Their paper states no surrogate form for any method** — D-MAPPO is described only
   as "proximal policy optimization with soft penalty constraints", which names its
   constraint handling, not its objective — so the clipped variant is our choice of a
   standard objective for an underspecified line, not something read off their text.
   An earlier draft attributed it to their description of D-MAPPO; that attribution was
   unsupported and is withdrawn. Both readings are run, and the difference is not
   cosmetic: at 12 episodes on the reference stock the literal reading trains
   *backwards* (reward −7273 → −7807) while the clipped one improves monotonically
   (−7273 → −6319).
2. **The advantage estimator.** "Compute advantage functions" (line 10). GAE at
   $\lambda = 0.95$.
3. **The exploration noise** $\epsilon_t$ of eq. 22, which is not specified. A Gaussian
   policy whose mean is $\pi(r)$ with a learned state-independent log-standard-deviation
   supplies both the noise and the log-density line 11 needs, and is the same
   parameterisation our D-MAPPO baseline uses — so the two differ in exactly the
   components the paper says they differ in.

Storing the projected action rather than the proposal is theirs, not ours, and it has a
consequence worth naming: the policy gradient is taken with respect to the log-density of
an action the *shield* chose, so the actor is trained on behaviour it did not propose.

Seven tests cover the parts that can be wrong without crashing — the time-window gather's
axis order (the natural indexing transposes it, and the error is invisible whenever the
window length equals the building count, so the tests use 7 and 3), that the stored action
is the projected one, that the shared encoder actually receives gradient, both readings of
line 11, save/load determinism, and the same known-answer optimum check the other nine
controllers get.

---

## 8. The cost–comfort frontier (50 episodes)

Every row of Table I is one operating point reported without a trade-off. These are
points on the curve those rows sit on. All at 5 seeds, reference stock.

| $\lambda_{\text{indoor}}$ | $\xi$ | cost | emission | peak | consumption | ramping | discomfort | safety % |
|---|---|---|---|---|---|---|---|---|
| 0.0 | 0 | **0.749** | 0.757 | **0.562** | 0.756 | 0.644 | 0.453 | 0.000 |
| 0.1 | 0 | 0.778 | 0.789 | 0.567 | 0.789 | 0.657 | 0.404 | 0.000 |
| 0.2 | 0 | 0.830 | 0.822 | 0.591 | 0.825 | **0.642** | **0.396** | 0.000 |
| 0.4 | 0 | 0.896 | 0.864 | 0.642 | 0.866 | 0.675 | 0.407 | 0.000 |
| 0.8 | 0 | 0.916 | 0.867 | 0.630 | 0.866 | 0.643 | 0.404 | 0.000 |
| 0.4 | 0.6 | 0.909 | 0.868 | 0.644 | 0.868 | 0.673 | 0.398 | 0.000 |
| STEMS | — | 0.792 | 0.824 | 0.821 | 0.805 | 0.883 | 0.132 | 5.600 |

**8.1 Cost is monotone in the comfort weight over a 22 % span.** Discomfort is not: it
improves to $\lambda=0.2$ and then reverses, so its entire range is **0.057**. Past
$\lambda = 0.2$ the comfort weight is a tax with no return, and the repository default of
0.4 sits beyond it. All five points lie between 0.396 and 0.453, **straddling the 0.4293
the raw dataset gives for these eight buildings with no controller at all**.

**8.2 Every point beats the STEMS row on peak (0.562–0.642 vs 0.821) and on ramping
(0.642–0.675 vs 0.883).** $\lambda \le 0.1$ also beats it on cost, emission and
consumption.

**The safety column is not one of these comparisons, and an earlier version of this
section wrongly presented it as one** ("beats it on the safety union, 0 vs 0.056"). Our
union is nominally $\text{SoC} \cup \text{per-building power} \cup \text{district grid}$,
but on this stock the latter two cannot bind: across **all 139 reference-stock records**
the district peak never exceeds **5.08 %** of the 2000 kW cap and the per-building peak
never exceeds **4.72 %** of the 500 kW cap, so both columns are structurally zero and
`safety_violation_rate` equals `soc_violation_rate` to the last bit in **139 of 139**
records. Our zero is therefore a zero on **one** constraint — the state-of-charge band,
which is exactly the constraint the projection inverts, so it is the one place a hard
guarantee is expected rather than impressive. Setting it against their 0.056 compares a
single-constraint rate to a union, and §9.2 shows their union is in any case smaller than
an event it contains. The honest statement is in §8.3e: the projection drives *SoC*
violations to zero, on policies that are not ours, at near-zero cost.

### 8.3 Best configuration

> **This row is not a like-for-like comparison with the baselines, on two counts.**
>
> *Objective.* Every baseline in §7 runs the default $\lambda_{\text{indoor}} = 0.4$. This
> row runs $\lambda_{\text{indoor}} = 0$ — it is told not to pay for comfort. On a cost
> ranking that is a different contest, and quoting 0.718 against baselines that bought
> comfort is the same category of error as §7.2's shield.
>
> *Budget.* It runs 50 episodes with a warm start; several baselines are at 15.
>
> **The matched comparison is §8.3a, and it is the one to quote.** At
> $\lambda_{\text{indoor}} = 0.4$ and 50 episodes the same controller reaches **0.817 at
> discomfort 0.420**, against the best measured baseline's 0.879 at 0.424 — a 7.0 % margin
> at essentially identical comfort. Against STEMS's published 0.792 the comparison is
> genuinely ambiguous, because their $\lambda$ is not stated: our matched point sits above
> it, our $\lambda = 0$ point below.

Warm start + $\lambda_{\text{indoor}}=0$, 50 episodes:

| column | ours | STEMS | margin |
|---|---|---|---|
| cost | **0.7182** | 0.792 | −9.3 % |
| emission | **0.7683** | 0.824 | −6.8 % |
| avg daily peak | **0.6317** | 0.821 | −23.0 % |
| consumption | **0.7664** | 0.805 | −4.8 % |
| ramping | **0.6744** | 0.883 | −23.6 % |
| SoC violations | **0.000** | — | not comparable (§8.2) |
| discomfort | 0.4497 | 0.132 | we lose (§9.1) |

The safety row is deliberately left without a margin. Our 0.000 is a state-of-charge rate
— the other two members of our union cannot bind on this stock (§8.2) — while their 0.056
is a union whose own definition their table contradicts (§9.2). The two numbers do not
measure the same event, so no margin between them means anything.

Per-seed cost ratios $\{0.7089, 0.7103, 0.7113, 0.7215, 0.7390\}$, $\mathrm{sd}=0.0127$,
**5/5 seeds below 0.792**. Trained cold the same configuration gives $0.7489$,
$\mathrm{sd}=0.0324$, 4/5 below. The warm start is worth **0.031 of cost and a 2.5×
reduction in seed spread** — for a deployable controller the second matters as much as the
first.

The warm-started curve is **non-monotone** in $\lambda$ (0.718, 0.814, 0.765, 0.817), so
its points are plotted unconnected: joining them would assert an ordering the
measurements do not support.

### 8.3a Matched-objective comparison — the number to quote

Every controller below optimises the **same** six-term objective
($\lambda_{\text{indoor}} = 0.4$), is evaluated on the **same** reference stock, and is
scored by the **same** metric code. Baselines carry no safety layer of ours (§7.2).

| controller | $\lambda_{\text{indoor}}$ | episodes | seeds | cost | discomfort |
|---|---|---|---|---|---|
| **ours, warm start** | 0.4 | 50 | 5 | **0.817** | 0.420 |
| ours, cold | 0.4 | 50 | 10 | 0.903 | 0.403 |
| MADCQ | 0.4 | 15 | 5 | 0.879 | 0.424 |
| MARLISA | 0.4 | 15 | 5 | 0.962 | 0.412 |
| D-MAPPO | 0.4 | 15 | 5 | 0.998 | 0.405 |
| MADDPG | 0.4 | 50 | 4 | 1.060 | 0.414 |
| MetaEMS | 0.4 | 50 | 5 | 1.327 | 0.423 |
| Single-Agent SAC | 0.4 | 50 | 5 | 1.573 | 0.408 |

> **Superseded within hours of being written — read §8.3b.** The 7 % margin claimed
> here measured our 50-episode controller against MADCQ at **15 episodes**. When
> D-MAPPO finished its 50-episode run it reached 0.817, tying us to four decimals.
> The table above is correct for the budgets it lists; the margin drawn from it was
> not.

### 8.3b Matched objective *and* matched budget — the margin is gone

Same objective, same 50-episode budget, same reference stock, 5 seeds each:

| | cost | emission | peak | consumption | ramping | discomfort | violations |
|---|---|---|---|---|---|---|---|
| ours (shielded, warm start) | 0.817 | 0.847 | 0.740 | 0.849 | 0.753 | 0.420 | **0.00 %** |
| D-MAPPO (unshielded) | 0.817 | **0.828** | **0.613** | **0.832** | **0.668** | **0.410** | 61.92 % |

Cost 0.8171 against 0.8173, Welch $t = -0.008$, $p = 0.994$. Not a narrow win — a tie
to four decimals. D-MAPPO's seed spread is also 3.8× tighter (sd 0.0186 against 0.0699;
one of our seeds returned 0.9413 and inflates the mean).

**That outlier is not a training failure, and it says something about the objective.**
Seed 0 reaches a final episode reward of $-3334$ against seed 3's $-3315$ — within
0.6 % — while costing 22 % more. It got there by trading cost ($+0.168$) and consumption
($+0.067$) for peak ($-0.086$) and ramping ($-0.094$). It is a legitimate optimum of the
objective we wrote, not a seed that failed to converge. Across our five seeds the final
reward spans 99 ($\approx 3\,\%$) while cost spans 0.168 ($\approx 21\,\%$); for D-MAPPO
the same spans are 142 and 0.046 ($\approx 6\,\%$). **The six-term reward does not pin
down the column we report.** With $n = 5$ no reward-cost correlation is significant
($p > 0.1$ for every arm, signs inconsistent), so this is stated as the level-set
observation it is rather than as a correlation.

**D-MAPPO beats us on every performance column**: emission, peak, consumption, ramping
and comfort. The single column we win, we win completely — 0.00 % violations against
61.92 %.

**What this leaves as the contribution, stated narrowly.** Our safety layer eliminates
constraint violations at no cost penalty. That is the claim safe RL exists to make, and
we can demonstrate it on five seeds. What it is *not* is evidence that the
GCN-transformer encoder or the multi-Lagrangian buys anything: against independent PPO
at matched objective and budget, they buy nothing here.

**The experiment that decides the architecture's worth** is D-MAPPO carrying our shield
at the same budget (`results/dmappo_shielded_ep50`). If it reaches ~0.8 at zero
violations while keeping its better peak and ramping, the contribution collapses to the
shield alone — which attaches to any policy and needs neither our encoder nor our
Lagrangian. If it loses something our architecture preserves, that difference is the
contribution.

**Caveat on the other two rows.** MADCQ and MARLISA were still training at 50 episodes
when this was written, and both were still improving. Either may also land at or below
0.817.

### 8.3c The shield transfers; the architecture does not

The experiment §8.3b named has run. Three controllers, same objective
($\lambda_{\text{indoor}} = 0.4$), same 50-episode budget, same reference stock, 5 seeds
each:

| | cost | emission | peak | consumption | ramping | discomfort | violations |
|---|---|---|---|---|---|---|---|
| ours (shield + GCN-transformer + multi-Lagrangian + warm start) | 0.817 | 0.847 | 0.740 | 0.849 | 0.753 | 0.420 | **0.00 %** |
| **independent PPO + our shield, nothing else** | 0.833 | **0.844** | **0.615** | 0.853 | **0.670** | **0.409** | **0.00 %** |
| independent PPO, unshielded | 0.817 | 0.828 | 0.613 | 0.832 | 0.668 | 0.410 | 61.92 % |

**Read the bottom two rows first.** Attaching our safety projection to an off-the-shelf
independent PPO moves its cost from 0.8173 to 0.8326 — **+1.9 %** — and takes its
violation rate from **61.92 % to exactly zero**. That is the whole safe-RL claim,
demonstrated on a policy we did not design, and it is the result this project should be
built on.

**Now read the top two.** Our full architecture does not beat independent PPO carrying
only our shield. Cost 0.817 against 0.833 is not a significant difference
($t = -0.494$, $p = 0.647$), and on every other column the simpler controller is equal
or better: peak 0.615 against 0.740, ramping 0.670 against 0.753, discomfort 0.409
against 0.420, emission 0.844 against 0.847. Its seed spread is **11.6× tighter**
(sd 0.0060 against 0.0699).

**The conclusion is negative and it is the honest one.** The GCN-transformer encoder,
the multi-Lagrangian and the behaviour-cloning warm start — the parts of this work that
took the most effort — add nothing measurable here, and cost peak and ramping. The
safety projection is the contribution, it is policy-agnostic, and it is nearly free. A
reader is better served by "a control-barrier projection with an exact battery inverse
removes 62 points of violation for 1.9 % of cost, on any policy" than by an architecture
that does not outperform the baseline it is bolted to.

This also reframes §10.4 and §10.5. The graph carries half the Shapley attribution of
*our* controller's action — but a controller without any graph at all matches it. A
component can dominate an attribution and still not be necessary: attribution measures
what a trained network *uses*, not what the task *requires*.

### 8.3d The complete matched-budget table

The 50-episode grid finished: 31 runs, 12 h 20 min. Every arm below runs 50 episodes at
$\lambda_{\text{indoor}} = 0.4$ on the reference stock, 5 seeds, scored by the same code.

![Zero violations at about 0.82 cost is reachable three ways; the architecture is not
what gets you there.](figures/fig_safety_frontier.png)

| controller | cost | sd | emission | peak | consumption | ramping | discomfort | violations |
|---|---|---|---|---|---|---|---|---|
| ours (shield + GCN + Lagrangian + BC) | **0.817** | 0.0699 | 0.847 | 0.740 | 0.849 | 0.753 | 0.420 | **0.00 %** |
| D-MAPPO (independent PPO) | **0.817** | 0.0186 | 0.828 | 0.613 | 0.832 | 0.668 | 0.410 | 61.92 % |
| MADCQ | 0.833 | 0.0109 | 0.856 | 0.694 | 0.863 | 0.804 | 0.431 | **0.00 %** |
| independent PPO + our shield | 0.833 | **0.0060** | 0.844 | 0.615 | 0.853 | 0.670 | 0.409 | **0.00 %** |
| MARLISA | 0.847 | 0.0163 | 0.872 | 0.638 | 0.880 | 0.747 | 0.423 | 62.72 % |
| MADDPG | 1.047 | 0.0471 | 0.996 | 0.696 | 0.996 | 0.810 | 0.415 | 71.71 % |
| MetaEMS | 1.327 | 0.0472 | 1.163 | 1.133 | 1.147 | 1.028 | 0.423 | 93.59 % |
| Single-Agent SAC | 1.573 | 0.0396 | 1.392 | 1.027 | 1.388 | 1.131 | 0.408 | 69.30 % |

**Three independent routes reach exactly zero violations at 0.817–0.833**: our full
architecture, our shield bolted to independent PPO, and MADCQ — which gets there with no
shield at all, by maximising over a constrained action set. Their cost spread is 0.016,
smaller than our own arm's seed standard deviation of 0.0699.

**So "zero violations at ~0.82" is a property of the problem, not of our method.** What
differs between the three is reliability, and we are the worst of them: seed sd 0.0699
against MADCQ's 0.0109 and shielded PPO's 0.0060, an 11.6× spread.

**What the shield is worth, stated as the measurement it is.** Applied to a policy we did
not write, it moves cost 0.8173 → 0.8326 (+1.9 %) and violations 61.92 % → 0.00 %. That
transfer is the result; the architecture around it is not.

**None of these reaches the published 0.792.** The best figure anywhere in this table is
0.817, and the two cheapest rows achieve it at opposite ends of the safety axis.

### 8.3e The result that transfers: free safety on four policies we did not write

§8.3c showed the shield working on one policy. This is the same experiment on four,
each an independent implementation of a published method, none of them ours.

![The shield removes 62 to 94 points of violation for a mean cost of +0.1 per
cent.](figures/fig_shield_transfer.png)

| policy | cost, no shield | cost, shielded | $\Delta$ | $p$ | violations, no shield | violations, shielded |
|---|---|---|---|---|---|---|
| Single-Agent SAC | 1.573 | 1.505 | **−0.068** | 0.050 | 69.3 % | **0.000 %** |
| MADDPG | 1.047 | 1.109 | +0.062 | 0.203 | 71.7 % | **0.000 %** |
| MetaEMS | 1.327 | 1.321 | −0.005 | 0.883 | 93.6 % | **0.000 %** |
| D-MAPPO | 0.817 | 0.833 | +0.015 | 0.144 | 61.9 % | **0.000 %** |
| **mean** | | | **+0.001** | | | |

**Violations fall to exactly zero on four of four policies**, from between 61.9 % and
93.6 %. The mean cost of doing so is **+0.001** — one tenth of one per cent — with a
range of $[-0.068, +0.062]$ and no individual effect significant at the 5 % level except
SAC's, where the shield *helps*.

**No pattern in which policies are helped and which are taxed.** An earlier draft of this
section claimed the shield helps policies that sit above the rule and taxes those below,
on the grounds that it blocks an expensive mistake a poor policy makes often. **The table
refutes that.** Three arms sit above the rule — SAC (1.573), MetaEMS (1.327) and MADDPG
(1.047) — and the shield makes the first two cheaper while taxing the third by the
largest positive margin in the set. The effects do not order by cost either:
$-0.068$, $-0.005$, $+0.062$, $+0.015$ against costs $1.573$, $1.327$, $1.047$, $0.817$.

What the data supports is weaker and should be stated as such: **only SAC's effect
approaches significance** ($p = 0.050$, and there the shield helps); the other three have
$p = 0.14$, $0.20$ and $0.88$ — indistinguishable from zero. With four policies and these
intervals there is no mechanism to report, only the aggregate: the shield is free on
average and no individual cost is reliably moved.

**This is the project's result.** It is a property of the safety layer alone, it holds
across four unrelated algorithms, it requires no access to the policy's internals or its
training, and it costs nothing on average. Everything else this work built — the
GCN-transformer encoder, the multi-Lagrangian, the behaviour-cloning warm start — is not
needed for it (§8.3c, §8.3d).

### 8.3f The same result on three further building stocks

§8.3e varied the policy and held the buildings fixed. This varies the buildings and holds
the policy fixed: D-MAPPO, 50 episodes, on three further 8-building draws from the Travis
County stock, each normalised to **its own** unshielded rule.

| stock | seeds | cost, no shield | cost, shielded | $\Delta$ | $p$ | violations, no shield | violations, shielded |
|---|---|---|---|---|---|---|---|
| reference | 5 | 0.817 | 0.833 | +0.015 | 0.144 | 61.9 % | **0.000000 %** |
| subset 1 | 3 | 0.770 | 0.768 | −0.002 | 0.962 | 40.2 % | **0.000000 %** |
| subset 2 | 3 | 0.786 | 0.815 | +0.029 | 0.134 | 55.4 % | 0.001427 % |
| subset 3 | 3 | 0.810 | 0.811 | +0.001 | 0.985 | 58.4 % | **0.000000 %** |
| **mean** | | | | **+0.011** | | | |

**Taken with §8.3e, the claim rests on seven independent conditions** — four policies on
one stock and one policy on four stocks. The cost effect across all seven has mean
**+0.005**, range $[-0.068, +0.062]$, and no individual effect reaches significance except
SAC's $p = 0.050$, where the shield helps. The violation rate falls to **exactly
0.000000 % in six of the seven**, from starting points between 40.2 % and 93.6 %.

**The seventh is not zero, and the exception is worth more than the rule.** Subset 2
retains **one** violating building-step per seed — the same one in all three seeds,
$1/70\,072 = 0.001427\,\%$ — and it is the *first* step of one building. Every episode
starts at $\mathrm{SoC}=0$, below the 0.1 bound, so the projection begins **outside** the
set it enforces. Whether it can return within one step is a question about the actuator,
not the controller: the largest one-hour gain is $P_{\text{nom}}\eta/\text{capacity}$,
which for building 2 of subset 2 is $1.28 \times 0.9601 / 14.0 = \mathbf{0.0878} < 0.1$. That
battery **cannot** reach the band in an hour. The worst building of every other stock can
— 0.1642 reference, 0.1302 subset 1, 0.1214 subset 3 — which is precisely why those six
conditions read 0.000000 %.

A control-barrier function certifies **forward invariance**: once inside the set, never
outside. It promises nothing about *reaching* the set from an infeasible initial condition
faster than the plant physically allows. So this step is a reachability limit of a 1.28 kW
battery, not a failure of the filter — and the codebase's own classifier agrees
independently, booking it under the *unavoidable* class (§8.4: 1.43e−05 unavoidable,
0.00e+00 avoidable, against the same stock's rule at 1.57e−04 unavoidable). It is the only
non-zero entry in 35 seeds.

Two limits worth stating. All four stocks are drawn from the same Travis County dataset
and the same year, so this is replication across building samples, not across climates;
the cross-dataset transfer is §8.3g. And the three subsets carry 3 seeds rather than 5.

### 8.3g Transfer to a second CityLearn dataset

§8.3f's stated limitation is that all four stocks are draws from one dataset and one year.
The pre-registered remedy was the CityLearn 2020 climate zones 1–4. **That experiment
cannot measure what it was designed to measure.** Of the seven datasets in the normaliser
sweep (`results/rbc_normaliser_sweep`, written up in `docs/CITYLEARN_DATASET_AUDIT.md`),
**five ship a mean electricity price of exactly 0.0000** — climate zones 1–4
and the 2021 challenge set, i.e. precisely the pre-registered targets. Every cost ratio on
them is $0/0$. Only `baeda_3dem` (0.05114) and `citylearn_challenge_2022_phase_all`
(0.27314) carry a tariff at all.

The 2022 set is the better test in every other respect too: a different climate, a
different dataset generation, **17 buildings instead of 8**, and `electrical_storage` as
its only active action — which is exactly the constraint our projection inverts, so the
comparison isolates the filter rather than confounding it with thermal control.

Two obstacles were real and are recorded because both are the kind of defect §9 is about.
The dataset leaves `indoor_dry_bulb_temperature` **inactive** in its schema although every
building CSV carries the column; and it has **no set-point observations at all**, so
CityLearn's dynamic comfort band does not exist for it. Activating the temperature yields
NaN from the first step — the dataset runs no thermal simulation — and NaN fails both band
comparisons, so the discomfort proportion came out **0.000**, a perfect comfort record for
a quantity never simulated. Leaving it inactive zero-fills it, every building sits at
0 °C, and the same column reports **1.000**. Both numbers are fabrications; both are now
reported as unavailable (§9.5).

What remains is real, and is what the claim needs. The unshielded rule on this stock:

| column | value |
|---|---|
| cost | 25 042.41 |
| avg. daily peak | 27.26 kW |
| ramping | 6.25 |
| electricity consumption | 102 301.13 |
| **SoC violations** | **61.36 %** |
| comfort columns | unavailable (no measured indoor temperature) |

The 61.36 % matters: the constraint is genuinely violated here, so there is something for
the shield to do. The district peak reaches 63.9 kW against a deliberately non-binding
2000 kW cap, so as in §8.2 this measures the state-of-charge band alone.

**Result.** D-MAPPO shielded against unshielded, 50 episodes, 3 seeds each, normalised to
the rule above:

| arm | cost | sd | SoC violations | peak | ramping |
|---|---|---|---|---|---|
| D-MAPPO, no shield | 1.141 | 0.0038 | 90.54 % | 0.827 | 0.771 |
| D-MAPPO + our shield | 1.155 | 0.0007 | **0.000000 %** | **0.807** | **0.728** |

**Violations go to exactly zero on all three seeds — 0.000000, 0.000000, 0.000000 —
from 90.3 %, 90.6 % and 90.7 %.** The filter is unchanged from §4.1; nothing about it was
tuned for this dataset, this climate, or seventeen buildings.

Two honest qualifications, both of which make this the most informative of the eight
conditions rather than the least.

*The cost is real here, and it is the first condition where that is true.* $\Delta =
+0.014$ with $p = 0.020$ — on the other seven the effect was a null. On this stock the
shield costs about 1.4 % of the rule's bill, and a deployment should expect to pay
something rather than nothing. It is also worth noting that *both* arms sit above the rule
(1.141 and 1.155): D-MAPPO does not beat a rule-based controller on this dataset at all,
which is a fact about the learner and the stock, not about the filter.

*It is not only free of violations; it is better on the physical columns.* Peak drops
0.827 → 0.807 and ramping 0.771 → 0.728. A projection that refuses a charge command near
the band edge also refuses the sharpest ramps, so this is a plausible mechanism rather
than a coincidence — but it is one condition and we do not generalise from it.

Reproduce with
`--schema citylearn_schemas/transfer_2022/schema.json --allow-missing-obs`.

### 8.3h The full matrix — nineteen conditions

The four policies were run on all four Travis stocks and on the second dataset, giving a
**4 × 5 policy × condition matrix** at 50 episodes. Each cell is normalised to its own
condition's unshielded rule; 5 seeds on the reference stock, 3 elsewhere; 20 cells
measured, one excluded (below), **19 counted**.

**Cost effect of adding the projection** (bold = significant at 5 %):

| condition | D-MAPPO | SAC | MADDPG | MetaEMS |
|---|---|---|---|---|
| reference | +0.015 | **−0.068** | +0.062 | −0.005 |
| draw 1 | −0.002 | −0.032 | +0.004 | −0.004 |
| draw 2 | +0.029 | **−0.135** | +0.035 | −0.007 |
| draw 3 | +0.001 | +0.017 | +0.010 | −0.002 |
| 2022 dataset, 17 buildings | **+0.014** | +0.001 | −0.012 | *excluded* |

**Violations and the aggregate:**

| | result |
|---|---|
| conditions with violations **exactly 0.000000 %** | **15 of 19** |
| the four exceptions | every policy on draw 2, each at **identically 0.001427 %** |
| starting violation rates | 40.2 % to 94.6 % |
| mean cost effect | **−0.004**, range $[-0.135, +0.062]$ |
| mean effect against zero | $t = -0.431$, $p = 0.672$ |
| significant cells | 3 of 19 — **two in the shield's favour**, one against |

**One cell is excluded, and the rule for excluding it is stated before the result.** A
controller that never moves the battery has a perfect constraint record for free, so a
cell only counts if the *unshielded* policy is actually operating the device. The screen
is two measurements already in every record: mean battery-equivalent full cycles $> 1$,
and non-zero spread in cost across seeds. Applied to all twenty measured cells, exactly
one fails:

| cell | cycles, unshielded | cost spread across seeds |
|---|---|---|
| 2022 set / MetaEMS | **0.000** | **0.000000** |
| every other cell | 27.6 – 545.4 | 168 – 4996 |

On the 2022 stock MetaEMS collapses to a do-nothing controller: the battery never moves,
all three seeds are bit-identical, and its state of charge therefore sits at 0 — below the
0.1 bound — for the entire episode. That is why it reports **100.0 %** violations
unshielded and 0.000000 % shielded, the largest apparent effect anywhere in this report.
It measures the projection charging a dead policy's battery into the band, not a filter
improving a controller. The same policy is *not* inert on the four Travis stocks (27.6 to
109.5 cycles), so this is a property of that policy on that dataset. Screened by
`verify_report_claims`, which fails if any counted cell goes inert or if this one stops
being inert.

**The four exceptions are the strongest evidence in the table, not a blemish.** Every one
of the four policies on draw 2 — D-MAPPO, SAC, MADDPG and MetaEMS, which share no code and
no training procedure — lands on *exactly* the same residual, 0.001427 %: one
building-step in 70 072, with a spread across policies of 0.0 to machine precision.
§8.3f derived from the battery's datasheet that this must happen: draw 2 contains a
14.0 kWh battery limited to 1.28 kW at an efficiency of 0.9601, so its largest one-hour
gain is $P_{\text{nom}}\eta/C = 1.28 \times 0.9601 / 14.0 = 0.0878$, and an episode beginning at $\mathrm{SoC} = 0$ cannot
reach the 0.1 bound in the first step whatever the controller does. A prediction made from
$P_{\text{nom}}\eta/C$ alone, confirmed to six decimal places by four unrelated policies —
and absent from all fifteen cells whose worst battery clears the bound. The residual
belongs to the plant, not to the control.

**What the matrix says, stated conservatively.** Across nineteen conditions spanning four
policies, four building stocks and two datasets, a control-barrier projection with an
exact battery inverse removes between 40 and 94 percentage points of constraint violation
and leaves cost statistically unchanged ($p = 0.672$ on the mean effect). Three cells reach
significance; in two of them the shield makes the controller *cheaper*. This is a claim
about the projection, and §8.3c and §9.4 show the rest of our architecture contributes
nothing on top of it.

![Left: state-of-charge violations before and after the projection, one row per condition,
log scale with zero drawn at the axis floor; orange marks the four cells that retain the
single unreachable first step, all on draw 2. Right: the change in cost each condition
paid, with 95 % intervals, normalised to that condition's own rule; filled marks are
significant at 5 %.](figures/fig_shield_matrix.png)

### 8.4 Replication on three further building stocks

Same configuration retrained on three independent 8-building draws from the same
100-building neighbourhood, 3 seeds each, **each normalised to its own unshielded rule**:

| stock | cost | peak | ramping | discomfort | avoidable viol. | unavoidable viol. |
|---|---|---|---|---|---|---|
| reference ($n{=}5$) | 0.718 | 0.632 | 0.674 | 0.450 | 0.00e+00 | 0.00e+00 |
| subset 1 | 0.783 | 0.780 | 0.772 | 0.255 | 0.00e+00 | 0.00e+00 |
| subset 2 | 0.732 | 0.743 | 0.748 | 0.232 | 0.00e+00 | **1.43e−05** |
| subset 3 | 0.771 | 0.754 | 0.770 | 0.384 | 0.00e+00 | 0.00e+00 |
| rule, reference | 1.000 | 1.000 | 1.000 | 0.446 | 3.16e−01 | 0.00e+00 |
| rule, subset 2 | 1.000 | 1.000 | 1.000 | — | 3.12e−01 | 1.57e−04 |

All four stocks come in below 0.792 on cost, below 0.821 on peak and below 0.883 on
ramping. Held-out stocks average **0.762** (range 0.732–0.783), so the reference draw is
somewhat favourable and **0.76 is the honest figure to quote for an unseen stock**.

**Avoidable violations are exactly zero on all four.** Subset 2 retains 1.43e−05 of the
*unavoidable* class — states from which no action can prevent the breach — against the
rule's 1.57e−04 on the same stock, i.e. the learner **enters forced-breach states 11×
less often**. Avoiding such states is the policy's doing; refusing the breach once there
is the barrier's, and only the second is something a shield can promise.

**Discomfort runs 0.232 → 0.450 across the four stocks under an identical controller with
identical weights.** See §9.1.

---

## 9. Four terms that cannot do what they are credited with — and two claims withdrawn

> This section held six items until v2 of the paper arrived. Two of them (§9.1, §9.2)
> rested on misreadings of v1 that v2 settles, and both are withdrawn in place rather
> than deleted. Four stand: the renewable term, the multi-Lagrangian, and the two
> fabricated comfort columns of §9.5 — plus the non-binding caps of §8.2 and the
> forced-charge penalty recorded in the paper draft.

### 9.1 WITHDRAWN — we measured the wrong comfort metric, and the conclusion was an artefact of it

> **This section previously claimed that the discomfort column "measures the weather file
> and the building sample" — that it was a statistic of the input data rather than of the
> controller. That claim is withdrawn.** It rested on scoring comfort as the fraction of
> *all* building-steps outside a fixed 20–26 °C band. That is not STEMS's metric. v2 of
> the paper (Section III-A item 3) states theirs verbatim:
>
> > *"Proportion of **occupied** timesteps when indoor temperature deviates **from
> > setpoint** by more than 2 °C"*
>
> $$D = \frac{1}{T_{\text{occ}}}\sum_t\sum_i \mathbb{1}(O_{i,t}>0)\cdot
>   \mathbb{1}(|T_{i,t}-T^{\text{set}}_{i,t}|>2)$$
>
> Under the fixed band every arm scored ≈ 0.42 and the column looked inert — which is
> exactly what produced the withdrawn conclusion. Under *their* definition, computed from
> the same rollouts, it is one of the most responsive columns we have.

| | fixed band (what we reported) | set-point band (their definition) |
|---|---|---|
| spread across seven controllers | 0.405 – 0.446 | **0.0042 – 0.2809 (66×)** |
| spread across the comfort weight $\lambda \in \{0, 0.1, 0.2, 0.8\}$ | ≤ 0.057 | 0.0291 → 0.0112, monotone |

The comfort weight works, and in the right direction: raising it from 0 to 0.8 cuts
discomfort by 2.6×. The metric responds to the controller by a factor of 66. Neither
statement is compatible with "it measures the weather file".

**What survives from the old section is a genuine CityLearn defect, and only that.** The
unpatched indoor-temperature observation reads the *uncontrolled dataset* series rather
than the simulated one: over the same horizon the unpatched rollout and the raw CSVs
agree to every digit (0.429329832, 30084/70072, difference 0.00e+00). That is a real bug
in the simulator, it is why `hvac_control='setpoint'` is *unavailable* on stock CityLearn,
and we patch it. It does not license any claim about what the comfort metric measures once
the metric is computed correctly.

**The real puzzle the correction exposes.** Scored their way, *every* controller we run is
far more comfortable than its published counterpart — including the rule, which has no
learning in it at all:

| | ours | STEMS published |
|---|---|---|
| Rule-Based | 0.030 | 0.130 |
| MPC | 0.030 | 0.654 |
| MARLISA | 0.006 | 0.237 |
| MADCQ | 0.005 | 0.325 |
| D-MAPPO | 0.011 | 0.152 |

A learner being more comfortable than theirs is a result; *the rule* being 4× more
comfortable than theirs is an environment difference, because the rule is a fixed
heuristic with no tuning. Comfort is bought with energy, so a setup that holds the band
this tightly should cost more — which is the direction our cost column misses in (§7).
This is now the leading hypothesis for the reproduction gap and it is being tested
directly (§7.7).

### 9.2 WITHDRAWN — v2 removes comfort from the safety union, so there is no contradiction

> **This section previously claimed that STEMS's safety column is mathematically
> impossible** — that a union containing the comfort bound cannot report 0.056 while the
> comfort rate it contains is 0.132. That claim is withdrawn. It rested on v1 reading as
> though the safety-violation rate included thermal comfort. **v2 is unambiguous and it
> does not.** Section II-D defines exactly three safety constraints —
>
> $$h_{\text{battery}} = \min(\mathrm{SOC}_{\max}-\mathrm{SOC},\ \mathrm{SOC}-\mathrm{SOC}_{\min}),
>   \quad h_{\text{power}} = P^i_{\max}-|e^i|,
>   \quad h_{\text{grid}} = P_{\text{grid,max}}-\textstyle\sum_i \max(0,e^i)$$
>
> — and Section III-A item 3 defines the metric as the "proportion of timesteps violating
> safety constraints", with no comfort member. Their 0.056 against 0.132 is therefore
> entirely consistent.

Our own metric code still reports both readings, because our shield's union and theirs
need not coincide; the comparable column is `stems_safety_violations_soc_equipment`, and
that is the one used everywhere in §7 and §8. The arithmetic property — that a
comfort-inclusive union cannot fall below the comfort rate — is still pinned as a test,
now against our reading only.

### 9.3 The renewable-utilisation term is provably inert

The target paper specifies $\xi = 0.6$. The term is

$$
r^{\text{renew}}_i(t) = \xi \cdot \min\!\left(\frac{\text{solar}_i(t)}{\text{load}_i(t)},\,1\right),
$$

evaluated on the **conditioning** observation — the state *before* the action — where
`solar_generation` is weather-driven and `non_shiftable_load` is by definition not
shiftable. Both are exogenous, so on a fixed evaluation window
$r^{\text{renew}}_i(t) = f_i(t)$, a function of the calendar hour alone. Then for **any**
two policies $\pi, \pi'$,

$$
\sum_{t}\sum_i r^{\text{renew}}_i(t)\Big|_{\pi} \;=\; \sum_t \sum_i f_i(t) \;=\; \sum_{t}\sum_i r^{\text{renew}}_i(t)\Big|_{\pi'},
$$

so the term adds the same constant to every policy's return and **cannot change the
argmax**.

**Verified over 1500 hours** with three radically different controllers:

| controller | $\sum r^{\text{renew}}$ |
|---|---|
| idle (does nothing) | 2495.417184178 |
| rbc + calibrated | 2495.417184178 |
| MPC | 2495.417184178 |
| **spread** | **0.000e+00** |

What remains is a rescaling of the reward, which perturbs the optimiser. Measured, it
perturbs it the *wrong* way at both budgets: $0.885 \to 0.919$ at 15 episodes,
$0.896 \to 0.909$ at 50.

### 9.4 The multi-Lagrangian switches itself off — and the shield is what switches it

§8.3c reports that our architecture adds nothing measurable on top of the projection.
This is the mechanism for one of its three components, and it is not subtle.

The PPO-Lagrangian of §4.2 carries three constraint channels — `soc_band`,
`building_power_cap`, `district_import_cap`. Reading the saved multiplier traces over
**15 runs** (`rl+calibrated` at 15 and 50 episodes, `rl+calibrated+bc` at 50, five seeds
each, 575 episodes in total):

| property | result |
|---|---|
| all three multipliers numerically identical, every episode, every run | **15 of 15 runs** |
| $\lambda$ reaches exactly 0 and never revives | **15 of 15 runs**, by episode 4 or 5 |
| maximum `violating_step_rate` in any episode of any run | **0.000000** |
| episodes trained with every multiplier at zero | **520 / 575 = 90.4 %** |

The trajectory is the same every time: $\lambda$ starts at 0.100, decays 0.100 → 0.035 →
0.020 → 0.005 → **0.000**, and stays there for the rest of training.

![Left: the state-of-charge multiplier over training, all fifteen runs (thin) and their
mean (heavy). It decays 0.100 → 0.035 → 0.020 → 0.005 → 0.000 and never revives. Right:
peak and final multiplier on each of the three constraint channels — the three are the
same number in 15 of 15 runs, because the projection drives all three realised costs to
zero and the PID law therefore sees the same error on each.](figures/fig_lagrangian_deactivates.png)

The cause is the projection. The PID law drives $\lambda_k$ on the constraint error
$\varepsilon_k = \bar c_k - d_k$, and the projection makes the realised cost $\bar c_k$
identically zero on every channel at every update — `violating_step_rate` is 0.000000 in
every one of the 575 episodes. So $\varepsilon_k < 0$ always, the integral term clips at
zero, and $\lambda_k = \mathrm{clip}(\cdot, 0, \lambda_{\max})$ returns 0. The three
multipliers are *numerically identical* for the same reason: all three costs are the same
number, zero, so all three errors are the same number.

Two consequences, both of which bear on how this work should be described.

1. **For 90 % of training the agent is unconstrained PPO.** The "safe RL" component is
   active for four episodes and then mathematically absent. §8.3c found that independent
   PPO plus our shield matches our full stack; this says those two systems are not merely
   equivalent in outcome, they are running close to the same algorithm.
2. **There is no "multi" in the multi-Lagrangian on this stock.** Two of the three
   channels are the power caps that §8.2 shows cannot bind here, and the third is already
   hard-enforced by the projection. A Lagrangian layered on top of an exact projection has
   nothing left to price.

This does not say a PPO-Lagrangian is useless — it says it is redundant *behind a
projection that already enforces the same constraints*, which is exactly the configuration
we built. The honest reading is that the shield is doing the work and the constrained-RL
machinery above it is ornamental, which is the same conclusion §8.3c reached from the
outside.

### 9.5 Two comfort columns computed from a temperature that was never measured

The fifth and sixth members of this section's pattern, found while porting to a second
dataset (§8.3g), and the sharpest because they fail in *opposite directions on the same
column*.

`citylearn_challenge_2022_phase_all` carries an `indoor_dry_bulb_temperature` column in
every building file but runs no thermal simulation — its only active action is the
battery. Switch the observation on and the patched endogenous observation is **NaN** from
the first step. Both comparisons in
$\text{breach} = (T < 20) \lor (T > 26)$ are False for NaN, so the discomfort proportion
evaluates to **0.000**: a *perfect* comfort record for a building whose temperature was
never simulated. Leave the observation off instead and it is zero-filled, every building
sits at 0 °C, and the same expression returns **1.000**.

Neither number touches data. Both are now `None`, guarded on the observation being both
present and finite, and the comfort-inclusive safety union goes `None` with them rather
than silently collapsing onto the state-of-charge-plus-equipment union and concealing that
a member is missing.

The general form is the one worth carrying away, because it is what the other four members
of this section share: **a term that reads nothing still prints a number, and the number
is usually at an extreme** — 0.000 or 1.000, a perfect score or a total failure — which is
exactly where a reader is least likely to suspect missing data and most likely to read a
result.

---

## 10. Attribution

### 10.1 Exact Shapley over the safety mechanism

The usual objection to decomposing credit — that a Shapley value needs every subset of
players — does not apply, because the E3 grid **is** the full factorial. Three binary
mechanisms, $L$ (Lagrangian), $P$ (projection), $C$ (exact plant inverse), give
$2^3 = 8$ coalitions, all run at 5 seeds. Hence

$$
\phi_i = \sum_{S \subseteq \mathcal{N}\setminus\{i\}}
\frac{|S|!\,(n-|S|-1)!}{n!}\Big[v(S\cup\{i\}) - v(S)\Big]
$$

in closed form. Efficiency $\sum_i \phi_i = v(\mathcal{N}) - v(\varnothing)$ is
**asserted per seed**, so a mis-joined coalition cannot pass silently.

Measured coalition values (safety violation rate):

| coalition | $v$ | | coalition | $v$ |
|---|---|---|---|---|
| $\varnothing$ | 0.710094 | | $C$ | 0.710094 |
| $L$ | 0.059974 | | $L,C$ | 0.059974 |
| $P$ | 0.202649 | | $P,C$ | 0.000217 |
| $L,P$ | 0.001541 | | $L,P,C$ | 0.000188 |

Shapley values, paired over seeds:

| mechanism | $\phi$ | 95 % CI | share |
|---|---|---|---|
| Lagrangian | −0.3586 | [−0.4588, −0.2583] | 50.5 % |
| projection | −0.3171 | [−0.3607, −0.2736] | 44.7 % |
| exact inverse | −0.0342 | [−0.0375, −0.0308] | 4.8 % |

**The finding an ablation table cannot produce.** $C$ has **no main effect at all**:
$v(C) = v(\varnothing) = 0.710094$ bit-identically, because there is no projection for it
to invert. One-factor-at-a-time ablation — what this literature publishes — scores it
worthless. Its four marginal contributions are

$$
0,\quad 0,\quad -0.001353,\quad \mathbf{-0.202432},
$$

the last when the projection is present alone. **Its entire value is interaction.**
(For calibration: the game's single largest marginal is $-0.709877$, adding $P$ to $C$.)

The headline ordering flips with it: under the uniform inverse the Lagrangian beats the
projection $3.4\times$ (0.0600 vs 0.2026); under the exact inverse the projection beats
the Lagrangian $276\times$ (0.000217 vs 0.0600).

**On cost the signs separate**: Lagrangian $+548.9$ (CI [+158.8, +939.1]), projection
$-782.8$, exact inverse $-373.6$. The mechanism that buys the most safety is the one that
costs money — an operator-actionable statement.

### 10.2 The shield as a labelling oracle

A CBF shield labels every control step for free: it either passed the command or
corrected it. Over the reference year that is **70 072 labelled building-steps, 22.08 %
corrections**, with no human annotation. Define

$$
y^t_i = \mathbb{1}\Big[\;\big\lVert a^{\text{nom},t}_i - a^{\text{safe},t}_i \big\rVert_\infty > 10^{-9}\Big]
$$

(the projection is exact arithmetic, so an untouched action is bit-identical) and fit a
shallow decision tree $\hat y = g(o)$. **The split is on time, not at random** —
consecutive hours are dependent and a random split lets the tree memorise a step's
neighbours.

Fidelity on the held-out second half-year:

| depth | balanced accuracy | precision | recall |
|---|---|---|---|
| 1 | 0.7771 | — | — |
| **2** | **0.9368** | 0.795 | 0.943 |
| 3 | 0.9368 | 0.795 | 0.943 | 
| 4 | 0.9578 | 0.918 | 0.939 |
| 5 | 0.9696 | 0.868 | 0.981 |

Depth 3 scores *identically* to depth 2 and every one of its extra splits is degenerate
(both children predict the same class), so the effective explanation is **two
conditions**:

```
if dhw_storage_soc <= 0.998481
    and electrical_storage_soc <= 0.101815  ->  INTERVENE   (n=2087, purity 0.984)
    otherwise                               ->  allow       (n=24753, purity 0.926)
else
    and electricity_pricing <= 0.310000     ->  INTERVENE   (n=6737, purity 0.924)
    otherwise                               ->  allow       (n=1455, purity 1.000)
```

**The battery threshold it recovers is 0.101815 against a configured
$\underline{s} = 0.10$** — the barrier's own parameter, read back out of behaviour to
within 0.002, without being told the bound exists. That is a ground-truth check on the
method: when pointed at something whose answer we do not know, there is now a reason to
trust it.

Two reporting defects found and fixed: sklearn's `export_text` rounds thresholds to two
decimals, printing the 0.998481 split as "> 1.00" and reading as a state of charge above
full (`dhw_storage_soc` never exceeds 1.0 in 70 072 samples — the anomaly was the
printer); and the labels belong to whichever controller generated them, so what is
described is the **joint policy-and-shield boundary**, not the shield's intrinsic one.

### 10.3 Graph attribution — occlusion says the graph is minor, Shapley says it is dominant

> **This section's original conclusion was wrong and is kept here with its correction.**
> It read "the spatial half is doing less than one observation", on the occlusion evidence
> below. §10.4 re-runs occlusion on the warm-started policy and puts an axiomatic
> attribution beside it on that policy's own visited states; the graph ranks **first of
> 31** there rather than fifth, because its contribution is almost entirely interaction
> and a one-at-a-time probe cannot see it. (The table below is a separate, earlier run —
> different policy, 300 states — so its figures are not the ones §10.4 compares against.)
> The occlusion numbers are correct for what they measure; the importance claim drawn
> from them was not.

The adjacency is an explicit tensor, so the coordination claim is directly testable: cut
an edge, hold everything else fixed, measure how far the trained policy's committed
action moves. On a 50-episode policy over 300 visited states:

| perturbation | mean $|\Delta a|$ |
|---|---|
| cut nothing (null) | **0.00e+00** |
| one neighbour edge, mean | 6.10e−03 |
| strongest single edge | 1.32e−02 |
| **cut every edge into $i$** | **5.65e−02** |
| **occlude one observation** (`heating_electricity_consumption`) | **2.12e−01** |

Isolating a building from the entire graph moves its action $3.8\times$ less than
occluding a single observation channel, and four channels clear that bar alone: heating
electricity 0.212, occupant count 0.111, outdoor temperature 0.083, price 0.078. **Read
this as a statement about main effects, not about importance** — §10.4.

Cutting all edges at once is reported separately from the *sum* of single-edge cuts,
because summing would assume a linearity the encoder does not have. And the adjacency has
no self-loops (§2.1), so cutting $A_{ii}$ is a no-op **by construction** — reported only
as a check that it measures as one, not as evidence a building ignores itself.

### 10.4 Shapley over the state — the attribution that reverses §10.3

Occlusion answers "what does this feature do *alone*?". That is a main effect, and it is
not importance. The Shapley value is the unique attribution satisfying efficiency,
symmetry, nullity and additivity, so it is the right instrument for the question §10.3
was actually asking.

**The game.** Players are the 30 observation channels **plus one extra player, *neighbour
coupling*, which owns every off-diagonal entry of the adjacency.** Putting the graph into
the same game as the observations is what makes the comparison meaningful: "how much does
coordination contribute?" and "how much does price contribute?" become two numbers on one
scale, in units of the action, instead of two incomparable sensitivity measures. For
building $i$,

$$v_i(S) \;=\; \mathbb{E}_{x}\!\left[\, a_i(x_S) \,\right],$$

the mean committed battery action, where channels outside $S$ take the values they hold in
an independently drawn background state (the interventional convention) and an absent
neighbour-coupling player means the off-diagonal adjacency is zeroed. A masked channel is
masked in **every** building and across the **whole history window** — mask it only on the
target building and the value leaks back through a neighbour or through the transformer's
memory, and the attribution is silently wrong.

**Estimator and its guarantee.** Permutation sampling: draw a permutation, add players one
at a time, record each marginal. Within a permutation the marginals telescope, so
**efficiency holds exactly, by construction, not approximately** — the 31 values sum to
$v(\text{all}) - v(\varnothing)$ to float precision. Measured residual $2.2\times10^{-8}$,
asserted in code against a float32 tolerance, so an estimator bug fails loudly instead of
producing a plausible-looking saliency map. Intervals come from the spread across
permutations. One forward pass scores all eight buildings, so the eight games are solved
together.

**Result.** The occlusion column below is **not** the table in §10.3: that one was
measured on a different policy over 300 states. To make the comparison apples-to-apples,
occlusion was re-run on *this* policy over the *same* 192 visited states as the Shapley
(`results/graph_attribution_bc`), so the only thing that differs between the two columns
is the attribution method:

| | occlusion (§10.3) | Shapley |
|---|---|---|
| graph rank | **5 of 31** | **1 of 31** |
| graph vs. the top observation | $2.04\times$ *smaller* | $7.2\times$ *larger* |
| graph share of total $|\phi|$ | — | **50.1 %** |

Replicated across seeds 0, 1 and 2: rank 1 of 31 every time, share 50.1 / 46.4 / 43.6 %,
every interval excluding zero, efficiency residual $1.9$–$2.2\times10^{-8}$.

> **Read the share against §10.5.** A randomly initialised network of this architecture
> also ranks the graph first, at 38.9 %. First place is substantially architectural; what
> training adds is the eleven-point share increase and a 751× increase in magnitude.

The efficiency decomposition sharpens it. Graph $\phi = -0.0202$; all 31 players sum to
$-0.0168$; so the thirty observation channels together contribute $+0.0033$. **The graph
accounts for more than the entire net shift in the battery action, with the observations
partially cancelling it.**

**Why the two disagree.** Remove the graph while every observation is still present and
the building reconstructs most of what it lost from its own channels, so the action barely
moves and occlusion records a small number. Average the graph's marginal contribution over
*all* coalitions — including the many in which observations are absent — and the graph is
carrying the signal. Its value is almost entirely interaction, which a one-at-a-time probe
is structurally blind to.

This is the second interaction-dominated player in this system. In §10.1 exact calibration
has **no main effect at all** — $v(C) = v(\varnothing)$ bit-identical — yet holds 4.8 % of
the value, every bit of it interaction with the projection. The pattern is worth stating
on its own: **in this controller the components that matter are the ones a one-at-a-time
ablation cannot see.** Any paper that justifies its graph with an ablation is at risk of
the error §10.3 made.

It also bears on scaling. A per-building independent controller would be discarding the
single largest contributor to the action, so a design that admits new buildings without
retraining must keep the graph and make it count-invariant, not drop it.

### 10.5 Sanity control — how much of §10.4 is learned, and how much is the architecture?

An attribution that reports the same thing for a trained and an untrained network is
describing the architecture, not what the controller learned. That is the
model-randomisation test of Adebayo et al. (2018), and §10.4 has to survive it before its
number can be quoted. The control re-runs the identical game on a **randomly initialised**
network of the same architecture, on the **same states collected from the trained policy**,
so the only thing that changes is the weights.

| | trained | random init | ratio |
|---|---|---|---|
| graph $\|\phi\|$ | 0.032768 | 0.000044 | **751×** |
| total $\|\phi\|$ over all 31 players | 0.065413 | 0.000112 | 584× |
| top observation $\|\phi\|$ | 0.004538 | 0.000008 | 605× |
| action span $v(\text{all}) - v(\varnothing)$ | −0.016834 | +0.000031 | — |
| **graph share of total** | **50.1 %** | **38.9 %** | — |
| **graph rank** | **1 of 31** | **1 of 31** | — |

**The result is split, and the honest reading is the mixed one.**

*The magnitude test passes decisively.* Every attribution on the trained network is
584–751× larger than on the untrained one, and the untrained network barely responds to
its input at all — its whole action span is $3\times10^{-5}$ against the trained network's
$1.7\times10^{-2}$. The attribution is measuring something that training put there.

*The rank test does not pass.* **The graph ranks first of 31 on a randomly initialised
network too**, holding 38.9 % of the attribution mass. So first place is substantially a
property of the architecture: a GCN mixes its neighbours, and at any weights that mixing
moves the output more than any single input channel does. Training raises the graph's
share from 38.9 % to 50.1 % — an eleven-point increment — and the absolute magnitude by
roughly 750×.

**What survives, stated precisely.** "The graph carries half the attribution" must always
be quoted against "an untrained network of the same architecture carries 39 % of it". The
learned part of the §10.4 result is the eleven-point share increase and the 751× magnitude,
not the first-place ranking.

**What is untouched by this control.** The §10.4 comparison against occlusion is a
statement about two attribution *methods* applied to one fixed model, so re-randomising
the model cannot bear on it. Occlusion ranks the graph fifth and Shapley ranks it first on
the same trained network and the same 192 states; that disagreement, and its cause in
interaction effects, stands exactly as reported.

This agrees with the architecture grid, where replacing the feature graph with uniform
mean pooling cost only **0.9 %** (8324 vs 8248).

---

## 11. Threats to validity

**11.1 In-sample evaluation.** Trained and evaluated on the same 8760-step year. This is
*matched* to the target paper's twelve-month table, not flattering relative to it, but it
is not a generalisation claim. §8.4 is replication (retraining per stock), **not
transfer**: a policy trained on stock A and deployed unmodified on stock B is a harder
question that `run_one` has no code path for.

**11.2 The cost column reproduces for three of six learners, not all six** (§7.0).
MARLISA +0.0 %, D-MAPPO +1.8 %, MADCQ +2.3 % at a 50-episode budget. MADDPG is
undertrained (+39.6 reward/episode at the cutoff), MetaEMS is starved by our own code
(64 gradient steps per episode against 437 950 total for the others), and SAC converges
to a bad optimum we have localised to energy use but not explained. MPC is +12.2 % and is
not a learner, so budget cannot be its excuse. An earlier version of this limitation read
"reproduces for nobody" and was an artefact of equating their "15 epochs" with 15 of our
year-long episodes.

**11.3 Hardware.** `STEMSEnvironment.step` costs **9.894 ms** for 8 buildings, so a
full-year episode is ~87 s of single-threaded Python with no GPU path; an actor
forward+backward on a 256 batch is **0.798 ms** on one thread. The simulator, not the
network, is the bottleneck, so 24 independent CPU runs beat one GPU for this workload.

**11.4 Our SAC is weak** (+91 % above the oracle in the architecture grid) and is **not**
used to argue the published SAC figure is wrong.

**11.5 The caps change the objective** (§3.1).

**11.6 Shielded credit assignment** (§4.2): the projection is many-to-one and intervenes
on 22–73 % of steps, so the policy spends much of its time in the projection's null space
where the cost gradient is flat.

---

## 12. Reproduction

```bash
source scripts/env.sh

# Table I at their budget, all eight baselines
python -m experiments.ablation --seasons year --subsets ref --episodes 15 \
    --grid-cap 2000 --building-cap 500 --seeds 0 1 2 3 4 \
    --arms rbc mpc sac maddpg metaems marlisa madcq dmappo rl+calibrated \
    --out results/t1_repro_ep15
python -m experiments.diagnostics.stems_table_compare --results results/t1_repro_ep15

# the frontier (restartable; resumes rather than repeats)
bash experiments/drive_overnight.sh
python -m experiments.diagnostics.frontier --root results/overnight \
    --baseline-dir results/t1_repro_ep15 --out results/frontier.csv
python -m experiments.diagnostics.fig_frontier --csv results/frontier.csv --episodes 50

# the three attributions
python -m experiments.diagnostics.shapley_mechanism --results results/e3_mech
python -m experiments.diagnostics.shield_oracle --arm rbc+calibrated --season year
python -m experiments.diagnostics.graph_attribution --model <run>_model --states 300

# the comfort column
python -m experiments.diagnostics.comfort_defect --arms rbc+calibrated --season year
python -m experiments.diagnostics.comfort_is_the_dataset

pytest -q            # 681 passed
```

**Grid hygiene, learned the hard way.** `experiments/ablation.py` uses
`get_context("spawn").Pool(..., maxtasksperchild=1)`, so **every run re-imports from
disk**. Editing any module under `stems/` or `experiments/` while a grid is in flight
splits that grid across code versions; check `meta.code.fingerprint` (not `meta.code.git`,
which changes on documentation-only commits) and verify by reproducing one deterministic
arm. Also: `setsid`/`nohup` does **not** detach work here — each shell gets its own PID
namespace — so long jobs must run as the foreground process of a long-lived session, and
interrupting one background cell sends SIGTERM to a group that can include another's
children.
