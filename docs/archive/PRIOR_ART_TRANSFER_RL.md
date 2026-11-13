# Prior art: cross-climate transfer and parameter-efficient adaptation in RL

*Searched 2026-10-07.* 14 OpenAlex queries + 9 arXiv queries → 1785 records → **1366**
after dedup and abstract filtering → all 1366 screened against two explicit targets →
413 scored ≥3, 216 scored ≥4.

**Evidence tiers**, following the convention in `docs/LITERATURE.md`:

| tier | what it means | count |
|---|---|---|
| **full text** | PDF retrieved and searched | 2 |
| **abstract** | title + abstract read from the source API | 8 |
| **screened** | abstract classified by the screening pass only | 1356 |

Every "nobody has done X" below is **an absence from this search**, not a proven absence.

---

## Verdict first

Your three candidate directions are each **already occupied**, two of them by work from
the last eight months. But the search also locates a gap that none of them touch, and it
is one our own E1 measurement is unusually well placed to exploit.

| your idea | status | who |
|---|---|---|
| train/test across other climates, not just Texas | **occupied, and it is CityLearn's founding task** | CityLearn Challenge 2020 onward |
| a benchmark of many buildings/climates for transfer | **occupied, 3 months ago, at far larger scale** | Building2Building (6000 envs) |
| freeze some layers / LoRA-style adaptation in RL | **occupied** | LoRASA (Feb 2025) |
| LoRA + safety | **occupied** | SLowRL (Mar 2026) |
| pretrain then finetune per house | **occupied** | TSFT (Aug 2026), MetaEMS (2022) |
| **does a transferred *safe* controller stay safe?** | **no work found in 1366 records** | — |

---

## Target A: cross-building / cross-climate transfer in building-energy RL

**This is a crowded field, and climate transfer is its origin story.** The CityLearn
Challenge series (2020–2023) is described by its own organisers as having evolved *"from
testing control policy transferability across climate zones to incorporating real-world
datasets, battery management, and grid resilience scenarios"* (abstract, 10.1145/3679240.3734667).
Climate-zone transfer was the premise of the 2020 competition.

### The one that matters most: Building2Building (arXiv 2607.16534, July 2026) — *full text read*

A large-scale EnergyPlus HVAC suite built explicitly to study *"transfer, multi-task
learning, and meta-learning in RL"*, Gymnasium-compatible, with a parametric building
generator producing **6000 environments spanning multiple building types, climates and
HVAC configurations**, and defined benchmark tasks isolating *goal adaptation, dynamics
adaptation, action-space shifts, and cross-domain transfer*. It frames environment
attributes as the context of a contextual MDP.

This is, at much larger scale and with better-grounded physics, the benchmark your
multi-climate plan was going to construct. Building a 4-climate CityLearn version to
compete with it on generality is not a winning move.

**But — and this is the opening — searching its full text for
`constraint|safety|safe RL|shield|barrier|violation|comfort bound` returns *zero* hits.**
B2B measures whether a policy still *performs* under distribution shift. It does not ask
whether a policy still *satisfies constraints* under distribution shift. There is no
shield, no barrier, no violation metric.

### The rest of track A (abstract tier)

- **MetaEMS** (arXiv 2210.12590): meta-RL with group-level and building-level adaptation,
  for fast adaptation to unseen buildings. Note your repo already has a `metaems` arm.
- **Personalised Federated RL** (arXiv 2412.20946): PPO/TRPO federated across buildings on
  a customised CityLearn, cost and emissions objectives. Privacy-motivated, unconstrained.
- **Context-Aware Meta-RL for diverse indoor HVAC** (2025): transformer context encoding
  plus meta-RL across diverse building configurations and disturbances.
- **Heterogeneous transfer learning for DRL building control** (2024, Building Simulation,
  10.1007/s12273-024-1109-6): transductive and inductive transfer for scalability.
- Plus a long tail of 2020–2026 "transfer learning for DRL HVAC" papers, including online
  transfer learning across two real office buildings.

**Established in track A:** that cross-building transfer works, that finetuning and
meta-RL beat from-scratch on sample efficiency, and that a shared policy with per-building
adaptation is the standard recipe. **Not established:** anything about constraint
satisfaction under transfer.

---

## Target B: parameter-efficient adaptation in RL

Occupied, and recently. LoRA is no longer novel in RL.

### LoRASA (arXiv 2502.05573, Feb 2025) — *full text read*

Treats **each agent's policy as a specialised "task" finetuned from a shared backbone**,
appending small low-rank adaptation matrices **to each layer of the shared policy**,
implemented on top of **MAPPO** and A2PO, evaluated on SMAC and MAMuJoCo. Ablations cover
adapter rank, placement and timing.

Read that against your architecture: STEMS shares one actor across buildings and uses
MAPPO with a centralised critic. "A LoRA adapter per house on a shared trunk" **is
LoRASA with buildings as agents.** The method exists, the ablations exist.

Its full text also returns **zero** hits for `constraint|safety|shield|barrier|violation`,
and zero for `climate`.

### The others (abstract tier)

- **SLowRL** (arXiv 2603.17092, Mar 2026) — the closest neighbour to a "safe LoRA" idea:
  LoRA plus **training-time safety enforcement via a recovery policy**, finetuning a
  sim-trained locomotion policy directly on a Unitree Go2. 46.5% less finetuning time,
  near-zero safety violations vs PPO, and **rank-1 adaptation alone sufficed** to recover
  pretrained performance. Single embodiment, sim-to-real; not a population of
  heterogeneous plants, and safety enforced by a learned recovery policy rather than an
  exact plant inverse.
- **LoRA for critic learning** (arXiv 2604.18978, Apr 2026): LoRA as a structural
  regulariser on the critic, freezing random base matrices. Relevant because STEMS has
  both reward and cost critics.
- **TSFT** (arXiv 2608.17180, Aug 2026): pretrain one policy, then finetune several for
  task specialisation, allocating a finetuning budget across task regions by integer
  linear programming. This is "pretrain then finetune per house", generalised and solved.
- **A Tutorial on Meta-Reinforcement Learning** (10.1561/2200000080): the reference for
  the few-shot adaptation framing.

---

## The gap, stated as a research question

Both literatures are **unconstrained**. Track A asks whether performance survives
transfer; track B asks how to adapt cheaply. Neither asks:

> When a controller carrying a *hard* safety guarantee is transferred to a new building or
> climate, does the guarantee survive — and which component has to be re-identified for it
> to survive, the policy or the safety layer?

**Why we are unusually well placed to answer it.** E1 and the guarantee sweep already
measured the mechanism this question turns on. The battery band is hard *because the
barrier inverts the plant exactly*: 0 breaches in 52 488 in-band cases with the exact
inverse, against 4 829 (9.20%) with a uniform-rate surrogate that understates the true
per-step rate by 1.8×–5.3×. The guarantee is a property of **plant calibration**, not of
the architecture.

Transfer changes the plant by construction — capacity, nominal power, efficiency curves and
measured peaks all differ per building and per climate. So the literature's standard recipe
(share a trunk, adapt per building) silently transfers a **policy** while leaving the
**shield** calibrated to the source plant. Our own numbers say that is the difference
between a 0% and a ~9% violation rate. Nobody in 1366 screened records appears to have
measured it.

### The design that follows

A 2×2 that no cited paper runs, because it requires both a transfer protocol *and* an
exact-inverse barrier:

| | shield kept at source calibration | shield re-identified on target |
|---|---|---|
| **policy zero-shot** | cheapest, and predicted to be unsafe | isolates shield re-identification alone |
| **policy adapted** (full finetune / low-rank) | isolates policy adaptation alone | the full method |

Measured on both axes that matter: cost (what the literature reports) **and** constraint
residual (what it does not). The honest hypothesis is that **shield re-identification
dominates policy adaptation for safety, while policy adaptation dominates for cost** — and
if that holds it is a clean, useful and previously unmeasured result.

### Secondary opening, from our own audit

S5.2 in `docs/FRAMEWORK_GUARANTEES.md`: when a deadline constraint and a shared-resource
cap are jointly infeasible, the myopic rules favour the deadline and break the cap while
the LP holds the cap and misses the deadline — *"with no record of the choice anywhere."*
The LP does report its shortfall; what is undeclared is the **arbitration rule**. A
principled precedence-and-disclosure scheme, with the infeasibility certificate as a
first-class output, is a second defensible contribution and is independent of the transfer
work.

---

## What this means for the plan

1. **Do not build a rival multi-climate benchmark.** B2B exists at 6000 environments.
   Use the 4 CityLearn 2020 climate zones as a *vehicle* for the constrained-transfer
   question, not as a generality claim.
2. **Do not claim LoRA-in-RL as a contribution.** Cite LoRASA and use it as the adaptation
   baseline. If low-rank adaptation appears in our arms, it is an *instrument*, not the novelty.
3. **Make constraint satisfaction under transfer the dependent variable.** It is the axis
   both literatures leave empty and the one our instrumentation already measures well.
4. **Re-measure the cap per climate zone.** `results/mixed_v1` currently carries
   `cap2000-500`, i.e. 2000/500 kW — non-binding, the same structural-zero defect E0 found
   in the Texas grid. Any multi-climate run that inherits it measures nothing.
5. **Read SLowRL in full before writing anything.** It is the nearest neighbour and will
   dictate how the claim must be positioned.

## Artifacts

- `lit_ranked.csv` — 413 records scoring ≥3, with target tag, score, year, citations, DOI
  and a one-line note each.
- `lit_key_abstracts.json` — verbatim abstracts of the 7 decisive arXiv papers.
