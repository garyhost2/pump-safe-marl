# Related work and open-source code: what exists, what we can claim

A survey run to answer three questions that could change what we publish, not to
compile a bibliography. Each section states the question, what was found, and what it
costs or buys us. Claims that rest on our own measurements point at the file that
produced them.

Date: 2026-10-09. Target paper: STEMS, arXiv:2510.14112.

---

## 1. Is the attribution contribution novel?

**The claim we were preparing.** Shapley values over the state of a GNN-based RL
controller, with the adjacency entered into the game as a player alongside the
observation channels, so that "how much does coordination contribute?" and "how much
does price contribute?" are answered on one scale in units of the action (§10.4 of
the technical report).

**What exists.** The three ingredients are each well covered, and separately:

* **Shapley for GNNs.** GraphSVX (Duval & Malliaros, arXiv:2104.10482) computes
  Shapley explanations for graph neural networks, decomposing predictions over node
  features and graph nodes. It targets node and graph *classification*, not a control
  policy, and the adjacency is explained as a set of node players rather than entered
  as a single coalition member.
* **Shapley for RL.** There is a theoretical framework for explaining RL with Shapley
  values, and applied work using SHAP on an RL policy for industrial process control
  (arXiv:2512.18317). These attribute over *state features*. None of them has a graph.
* **Shapley in multi-agent RL.** SHAQ (arXiv:2105.15013), Shapley counterfactual
  credits (arXiv:2106.00285) and a Shapley-based MARL framework applied to energy
  networks (arXiv:2402.15324) all use Shapley over **agents**, for credit assignment
  in the learning algorithm. That is a different game with different players: they
  divide a *return* among agents, we divide an *action* among inputs.

**Assessment.** The narrow combination — the adjacency as a player in a Shapley game
over an RL control policy's action — did not surface. The honest framing is
"combination of established tools", not "new method". The Shapley machinery is
standard and we should present it as such; what is ours is the *game formulation*
(graph as a coalition member) and the result it produces.

**What actually carries more weight.** The methodological finding is stronger than the
novelty claim. On the same policy and the same visited states, occlusion ranks the
graph 5th of 31 while Shapley ranks it 1st (§10.4) — and the model-randomisation
control (§10.5) shows that first place is substantially architectural, with the learned
part being an eleven-point share increase and a 751× magnitude. A result that comes
with its own refutation of the naive reading is worth more than a novelty claim.

---

## 2. Is the benchmark's normaliser sound?

**Why it matters.** Every CityLearn result is a ratio to a rule-based controller. The
convention is explicit in the CityLearn paper (arXiv:2012.10504): a score of 0.88 means
the controller performed 12 % better than the benchmark RBC. The whole literature's
0.8–0.9 scores inherit whatever that reference point is.

**What the RBC is.** The Challenge rule charges storage at night and discharges by day
*on hour of day alone* — designed for a **storage-only** action space. CityLearn 2.x
added `cooling_or_heating_device` as a controllable action; `BasicRBC` handles it by
setting the heat pump to 40–80 % of nominal power by clock time, every hour of the
year, with no reference to indoor temperature or demand.

**What we measured** (`experiments/diagnostics/rbc_normaliser_sweep.py`, 7 shipped
datasets, `BaselineAgent` vs `BasicRBC`, one deterministic episode each):

| dataset | buildings | HVAC action | mean tariff | idle/RBC kWh | RBC peak cut |
|---|---|---|---|---|---|
| baeda_3dem | 4 | no | 0.051 | 0.730 | 9.0 % |
| 2020 climate zone 1 | 9 | no | **0.000** | 0.996 | 13.0 % |
| 2020 climate zone 2 | 9 | no | **0.000** | 0.988 | 17.5 % |
| 2020 climate zone 3 | 9 | no | **0.000** | 1.000 | 17.8 % |
| 2020 climate zone 4 | 9 | no | **0.000** | 0.982 | 20.9 % |
| 2021 | 9 | no | **0.000** | 0.990 | **−4.1 %** |
| 2022 phase_all | 17 | no | 0.273 | 0.846 | **−2.7 %** |

Three things follow.

1. **The RBC consumes more electricity than doing nothing on 7 of 7 datasets**, by 0 to
   37 %. On 5 of 7 it buys peak reduction in exchange (13–21 % on the 2020 zones), which
   is a defensible load-shifting trade. On 2021 and 2022 phase_all it does **not** — it
   spends extra energy *and* raises the peak.
2. **5 of 7 datasets ship a zero electricity tariff.** Mean price is exactly 0.0000 on
   all four 2020 climate zones and on 2021. Any cost-based score on those datasets is
   identically zero.
3. **0 of 7 shipped datasets enables an HVAC action.** So the pathology below does *not*
   touch the published Challenge leaderboards.

**Where it does bite.** With the HVAC action enabled — which our schema does, and which
STEMS describes for its buildings — `BasicRBC` costs **63,409** against **21,740** for no
control on the eight Travis County buildings: **2.92×**. With the same action inactive
the ratio is 1.06×. On that scale, doing nothing scores **0.343**.

**The conditional about the target paper — tested, and refuted.** The hypothesis was
that if their Rule-Based behaves like `BasicRBC` with HVAC active, every row of their
table (0.792–0.872) would be worse than switching the controller off, since idle scores
0.343 on that scale. It was supported circumstantially: their rule reports 35.1 %
violations, `BasicRBC` measures 37.0 %, our own unshielded rule 31.63 %.

It is wrong. **CityLearn's own SAC, run in stock CityLearn with the HVAC action active
for 15 episodes, scores 0.153 against `BasicRBC`** — beating the normaliser 6.5× and
beating idle by more than 2×. STEMS publish 0.824 for SAC. Their normaliser therefore
cannot be `BasicRBC`-with-HVAC; against that controller a competent SAC scores 0.153,
not 0.824. Their rule must be a sensible one, like ours (18,773 against idle's 22,981),
and the gap between our baselines and their figures is **not** the normaliser.

**What survives, now measured twice.** `BasicRBC` with the HVAC action enabled is
genuinely defective, and the comfort column shows the mechanism: it is **more
uncomfortable than doing nothing** (0.771 against 0.400) while costing 2.92× more. It
heats at 80 % of nominal power at night and cools at 80 % by day, on a clock, in a
cooling-dominated climate — wasting energy *and* pushing the building out of the comfort
band. That is a defect in a widely used reference controller, independent of any claim
about the target paper, and subject to the scope limit above: no shipped dataset enables
that action.

**Framing precedent.** Lin, *The Neural Hype and Comparisons Against Weak Baselines*
(ACM SIGIR Forum), is the standard citation for the failure mode: a field's reported
improvements measured against a reference nobody re-examined.

---

## 3. Can the target paper's baselines be reproduced?

**Reference implementations exist.** CityLearn ships the agents its own authors wrote —
`citylearn.agents.sac.SAC`, `citylearn.agents.marlisa.MARLISA`, plus RBC variants and a
no-control `BaselineAgent`. The 2021 Challenge report describes SAC and MARLISA as the
officially implemented baselines. `experiments/diagnostics/reference_baselines.py` runs
them in stock CityLearn and scores them with the same `stems_table_metrics` that scores
our arms, so the question "our code or their numbers?" is answerable with reference code.

**Two of our own arms do not implement the methods named.**

* **Single-Agent SAC.** They specify "Independent Soft Actor-Critic for **each building**
  without coordination" — eight agents, 30-dim observation, 3-dim action. Our arm is a
  **centralised** SAC over a 240-dim observation and a 24-dim joint action; its docstring
  says so and says it exists to demonstrate how that scaling behaves. It was built to
  answer the centralisation question and reused as the Table I row because the names
  matched. This is a sufficient explanation for our worst row (1.573 against 0.824).
* **MetaEMS.** Ours implements Reptile; they specify MAML. Reptile is a first-order
  approximation of MAML, so this is a weaker objection, but it should be stated.

MADDPG (CTDE, Lowe et al. 2017) and MARLISA (Vázquez-Canteli et al. 2020) do match.

**Two of their rows cannot be reproduced by anyone.** The paper's baseline section
describes six methods: Rule-Based, MPC, Single-Agent SAC, MADDPG, MetaEMS, MARLISA.
Table I has eight. The strings **"MADCQ" and "D-MAPPO" appear nowhere in the paper's
text** — not in the prose, not in the 26-item reference list. References [20] (QMIX,
Rashid et al.) and [21] (MAPPO, Yu et al.) exist and could plausibly underpin the two
undescribed rows, but the paper never says so. Those rows have a number and a name and
nothing else.

**One of the baselines is the authors' own prior work.** Reference [17], MetaEMS, is
Zhang, Wu & Boulet (arXiv:2210.12590); STEMS is Zhang, Wu, Zinflou & Boulet.

**Their dataset may not contain the stock they describe.** They specify 8 buildings as
"5 residential, 2 commercial, 1 mixed-use" randomly selected from the Travis County
dataset. CityLearn's `tx_travis_county_neighborhood` holds exactly 100 building files,
every one `resstock-amy2018-2021-release-1-*` — NREL ResStock is a residential stock
model, and no CityLearn dataset carries a commercial-sounding name. Separately, the stock
dataset **fails to instantiate at all** in CityLearn 2.6.0b1:
`AssertionError: demand is greater than heating_device max output`.

---

## 4. Open-source code inventory

| what | where | use to us |
|---|---|---|
| CityLearn itself | `intelligent-environments-lab/CityLearn`, v2.6.0b1 installed | reference SAC/MARLISA/RBC agents; 25 datasets |
| CityLearn v2 paper | Nweye et al., *J. Building Performance Simulation* (2024) | the environment STEMS cites as [14] |
| CityLearn base paper | arXiv:2012.10504 | the RBC normalisation convention, quoted verbatim in §2 |
| 2021 Challenge winner | arXiv:2212.01939, *Winning the CityLearn Challenge: Adaptive Optimization* | the winner was **not** RL; names SAC/MARLISA as the official baselines |
| 2022 Challenge overview | PMLR v220, Nweye et al. | leaderboard context |
| 2023 Challenge winner | `TheLeprechaun25/CHESCA` | community-based hierarchical control |
| 2022 community solutions | `leomuckley/city-learn-2022` | several DRL implementations |
| multi-KPI CityLearn benchmark | arXiv:2602.19223 | independent characterisation of MARL on CityLearn |
| multi-objective CityLearn benchmarking | arXiv:2408.15170 | KPI definitions |
| GraphSVX | arXiv:2104.10482 | nearest prior art for §10.4 |
| SHAQ / Shapley MARL | arXiv:2105.15013, arXiv:2402.15324 | Shapley over *agents*, the contrast case |
| weak-baselines framing | Lin, ACM SIGIR Forum | §2's framing precedent |

---

## 5. What this changes

**Strongest results we hold**, in order of how well they would survive review:

1. **Occlusion and Shapley invert each other on the same model** (§10.4), with the
   model-randomisation control bounding how much is learned (§10.5). A methods finding
   with its own caveat attached.
2. **The normaliser result** (§2 above): measured, reproducible, and with a published
   framing precedent — provided it is stated with the HVAC condition attached.
3. **The matched-objective comparison** (§8.3a): 0.817 against the best baseline's 0.879
   at near-identical discomfort. Clean, like-for-like, and modest.

**Claims to retire.** "We beat STEMS at 0.718" — that point runs
$\lambda_{\text{indoor}} = 0$ while every baseline runs 0.4, and against their published
number the comparison is ambiguous because their $\lambda$ is unstated.

**The honest shape of the paper** is not "we beat STEMS". It is a reproduction study:
what happens when you try to reproduce a published building-energy RL table, what the
benchmark's reference point actually is, and what two attribution methods say about the
same controller. Three of our own findings were wrong when first stated and are recorded
with their corrections; that is the asset, not an embarrassment.
