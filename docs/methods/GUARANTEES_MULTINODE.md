# Guarantees for the multi-node safety filter

*Companion to `docs/FRAMEWORK_GUARANTEES.md`. Written 2026-10 on branch `audit/2026-10`.
Three results: a finite-time recovery bound for the state-of-charge band (M3), a
cycle-phase augmented barrier for the hot-water ceiling (M4), and an attempted
two-node domination argument (M5) — which **does not go through**, so M5's claim says
so in its own statement and M6 specifies the fallback.*

`FRAMEWORK_GUARANTEES.md` §S1 states what the projection certifies: **forward
invariance** of the state-of-charge band, measured at 0 breaches in 52 488 in-band
cases, and hard *because* the inverse is the plant's own. It says nothing about
reaching the band, and our episodes all start outside it. M3 is the missing half.

Assumptions are stated in M2, separately from the conclusions, each labelled
**measured** / **chosen** / **unsourced** in the convention of `docs/FRAMEWORK.md` §7.
Every number below comes from `experiments/diagnostics/multinode_guarantees.py` or from
a stored run record; M7 says how to reproduce each one. Where this document contradicts
a claim already in the repository, the contradiction is stated in the claim, not in a
footnote — M3.6 corrects two things.

---

## M1. Notation

Battery symbols are `FRAMEWORK_GUARANTEES.md` §S0's; hot-water symbols are
`stems/legionella.py`'s docstring table, unchanged. Only the node symbols are new.

| symbol | code | meaning | unit |
|---|---|---|---|
| `x_b` | `soc`, `electrical_storage_soc` | state of charge of building `b`'s battery | — |
| `x_min`, `x_max` | `CBFConfig.SOC_min/SOC_max` | the band the metric reports against, 0.1 / 0.9 | — |
| `lo_b`, `hi_b` | `CBFShield.enforced_soc_bounds()` | the band the projection actually enforces, margins included | — |
| `a_b` | action column `elec_idx` | commanded battery action | — (in [−1, 1]) |
| `a_max` | `action_scale` | action bound the projection respects, 1.0 | — |
| `C_b`, `p_nom_b` | `capacity`, `nominal_power` | battery energy capacity and nominal power | kWh, kW |
| `loss_b` | `loss_coefficient` | standing-loss fraction per step | per step |
| `eta_b` | `efficiency` | nameplate battery efficiency, *not* the per-step one | — |
| `g_b(·)` | `capacity_power_curve` | available power as a fraction of `p_nom_b`, versus state of charge | — |
| `eta_b(·)` | `power_efficiency_curve` | efficiency versus normalised power | — |
| `r_b` | `soc_rate` | `p_nom_b · dt / C_b`, the *datasheet* rate | per step |
| `Delta_b(x)` | — | the plant's realised one-step gain at `a = +a_max` | per step |
| `rho_b(x)` | — | the plant's realised one-step drop at `a = −a_max` | per step |
| `N` | — | worst-case steps to enter the band from outside it | steps |
| `C` | `capacity_kwh` | hot-water tank rated energy capacity | kWh |
| `E`, `soc` | `E`, `soc` | stored energy, `E / C` | kWh, — |
| `P_hp` | `heat_source_kw` | heat-source thermal power limit | kW |
| `loss` | `loss_coefficient` | tank standing-loss fraction per step | per step |
| `draw` | `draw_kwh` | hot-water draw demanded in the step | kWh |
| `T_cold`, `T_rated` | `t_cold_c`, `t_rated_c` | inlet temperature; temperature at which `C` is defined | degC |
| `T_normal`, `T_legionella` | `t_normal_c`, `t_legionella_c` | normal set point; disinfection set point | degC |
| `soc_legionella` | `soc_legionella` | `(T_legionella − T_cold)/(T_rated − T_cold)` | — |
| `period_hours` | `period_hours` | length of the disinfection window | h |
| **new:** `q` | — | disinfection phase, `q ∈ Q = {N, D}` | — |
| **new:** `soc_hi(q)` | — | phase-dependent band ceiling | — |
| **new:** `soc_lo` | — | comfort floor, `soc_of_temperature(45)` | — |
| **new:** `soc_top`, `soc_bot` | — | node states of charge, top (draw, comfort) and bottom (hygiene) | — |
| **new:** `C_top`, `C_bot` | — | node energy capacities, `C_top + C_bot = C` | kWh |
| **new:** `beta` | — | `draw / C_node`, the draw as a multiple of one node's energy | — |
| **new:** `kappa` | — | inter-node conduction, fraction of the node difference exchanged per step | per step |

`dt = 1 h` throughout, so kW and kWh/step coincide. A *residual* is the amount by which
a realised quantity violated a constraint, in that constraint's own units.

---

## M2. Assumptions

Stated here once; the claims in M3–M5 cite them by number and add nothing.

**A1 (measured).** The one-step battery map used in every proof below is
`BatteryModel._step`, which reproduces `citylearn.energy_model.Battery` to a maximum
forward error of 2.98e−08 state of charge over 938 unsaturated steps
(`docs/FRAMEWORK.md` §3, `experiments/plant_projection_error.py`). On saturated steps
the model errs toward the bound being protected
(`tests/test_battery_model.py::test_model_errs_toward_the_bound_being_protected`).

**A2 (measured).** The realised one-step gain `Delta_b(x)` at `a = +a_max` is
non-increasing in `x` on `[0, lo_b]` for **all 32 devices** of the four Travis stocks,
checked on a 201-point grid per device. This is the monotonicity M3's closed form
needs. It is a property of these devices, not a theorem: `g_b` is decreasing and the
standing loss grows with `x`, but `eta_b(·)` rises slightly as the available power
falls, so the sum is not monotone by construction.

**A3 (measured).** Device parameters are read off the live simulator, not off the
schema: `citylearn_schemas/tx_travis_8b/schema.json` declares
`electrical_storage: {autosize: true}` for all 100 buildings, so `C_b`, `p_nom_b`,
`eta_b`, `loss_b` and both curves exist only after the environment is built. M3.5 lists
what was read.

**A4 (measured).** Every battery in all four stocks has `initial_soc = 0.0`, so every
episode starts at `x_b = 0`, below `x_min = 0.1`. This is the whole reason M3 exists.

**A5 (chosen).** The band and its margins are `CBFConfig(SOC_min=0.1, SOC_max=0.9)` and
`SafetyConfig(soc_margin=0.03, soc_tolerance=1e-3, anticipatory=True,
invariance_horizon=1)`. None is fitted to anything.

**A6 (chosen).** `soc` is linear in store temperature — a **fully mixed** tank
(`LegionellaSpec.soc_of_temperature`). `stems/legionella.py` states the direction of the
error: a real cylinder stratifies, so the energy needed to bring the whole volume to
`T_legionella` is *understated* whenever a draw has left a cold bottom layer. M5 is the
attempt to remove this assumption; it is also where the attempt fails.

**A7 (unsourced).** The temperatures. `T_cold = 10`, `T_rated = 60`, comfort band
45–55 degC, `T_legionella = 60` degC are the values this document is written against,
and all of them are in `LegionellaSpec.OPEN_QUESTIONS` — required arguments with no
default, with no sourced value anywhere in this repository. Under that mapping
`soc_of_temperature` gives 45 degC → 0.70, 55 degC → 0.90, 57.5 degC → 0.95,
60 degC → 1.00, which are the levels M4 uses.

**A8 (unsourced).** `P_hp`, the heat-source thermal power limit, likewise. The numbers
quoted in M4.4 and M5.4 use CityLearn's *autosized `ElectricHeater` nominal power*
(3.94–7.77 kW on these 32 tanks) as a stand-in. That is the **optimistic** end by
construction: `stems/legionella.py` records that CityLearn's hot-water device is sized
so it can always serve the draw directly, which is exactly the property a real
heat-pump source lacks. Any conclusion below that depends on `P_hp` is therefore an
upper bound on what a real unit would achieve, and is labelled where it is used.

**A9 (chosen).** The two-node split in M5: two nodes of equal energy capacity
(`C_top = C_bot = C/2`), plug-flow displacement (a draw leaves the top, cold inlet
enters the bottom, water moves upward), linear inter-node conduction with coefficient
`kappa`, and **the heat source injecting into the bottom node**. The last is the
standard arrangement for a heat-pump water heater with a wrapped or bottom-mounted
condenser, and it is the assumption M5's failure turns on; M5.5 states what changes if
the source is an upper element instead.

**A10 (unsourced).** `kappa`, and any buoyancy / de-stratification rule. No value for
either exists anywhere in this repository, and M5 shows the proof depends on both.

**A11 (chosen).** In M4 the disinfection phase `q` is a **controlled** variable with a
guarded transition relation, not a timer. If the ceiling instead snapped back at a
fixed hour, the argument in M4 would not apply and the switched-system problem it
avoids would return; M4.3 is explicit about this.

---

## M3. Claim 1 — finite-time recovery into the band

### M3.1 The plant, and what the projection does outside the band

Write the charging branch of A1's map at the commanded bound. With
`e(x) = C·x·(1 − loss)` the energy after the standing loss,

```
  p(x)       = p_nom · g(x(1 − loss))                 available power, kW
  eta_eff(x) = eta_curve( min(a_max, g(x(1−loss))) )  efficiency at the realised power
  x+(x)      = [ e(x) + min( p(x)·dt, p_nom·dt, C − e(x) ) · sqrt(eta_eff(x)) ] / C
  Delta(x)   = x+(x) − x
             = −loss·x + (dt/C)·min( p_nom·g(x(1−loss)), p_nom, (C − e(x))/dt )
                                  · sqrt(eta_eff(x))
```

The square root is the plant's, not a modelling choice: CityLearn's `StorageDevice`
applies `round_trip_efficiency = efficiency**0.5` to the charged energy
(`citylearn/energy_model.py`), and `BatteryModel._step` applies `root = sqrt(eta)` to
match it.

**Lemma 1 (the projection commands the bound while the band is unreachable).**
`BatteryModel.safe_interval` sets `a_lo = a_hi = +a_max` whenever
`x+(x) < lo` — the `np.where(top < lo, a_max, ...)` branch — so the clip in
`CBFShield.project` returns exactly `+a_max` regardless of what the policy asked for.
Once `x+(x) >= lo`, `a_lo` is the bisected smallest action whose successor lands on
`lo`, and the clip raises any smaller action to it, so the successor state is in the
band. ∎

**Lemma 2 (the recovery command survives both power guards).** `_apply_power_guard`
clips the battery column into `[min(a_lo, a), a]` per building and, district-wide,
to `max(a·s, keep)` with `keep = min(max(a_lo, 0), a)`; neither can return a value
below `a_lo`. **Measured**: with a 500 kW base load per house against a 1 kW
per-building *and* district cap — both guards saturated — and the policy commanding
`a = −1`, `CBFShield.project` returns exactly `a_lo` for all 8 buildings on all four
stocks, never below it (`multinode_guarantees.py guard`). ∎

Lemma 2 is worth reading twice: it means the recovery floor **overrides the import
cap**, and that priority ordering is implicit in the code rather than declared
anywhere. It is the same unstated ordering `FRAMEWORK_GUARANTEES.md` §S5.2 complains
about on the fleet path, with the opposite sign.

### M3.2 The bound

> **Claim 1 (entry in N steps, then invariance).** Under A1, A2 and A5, for a device
> with capacity `C`, nominal power `p_nom`, efficiency `eta`, standing loss `loss` and
> action bound `a_max`, starting from any `x_0 ∈ [0, 1]`, the projection of
> `CBFShield.project` drives the state into the enforced band `[lo, hi]` within
>
> ```
>        ⎧  ceil( (lo − x_0) / r_min ),   x_0 < lo,    r_min  = Delta(lo)
>   N =  ⎨  ceil( (x_0 − hi) / rho_min ), x_0 > hi,    rho_min = min_{x ∈ [hi, x_0]} rho(x)
>        ⎩  0,                            otherwise
> ```
>
> steps, and `FRAMEWORK_GUARANTEES.md` §S1's invariance holds from that step on. With
> flat device curves (`g ≡ 1`, `eta_eff ≡ eta`) the charging case is the closed form in
> device parameters,
>
> ```
>   N = ceil(  (lo − x_0) · C  /  ( a_max · p_nom · dt · sqrt(eta) − loss · lo · C )  ),
> ```
>
> and the band is **never** entered — `N = ∞`, a fixed point strictly below the floor —
> when `a_max · p_nom · dt · sqrt(eta) / C <= loss · lo`.

**Proof.** Charging case. By Lemma 1 the commanded action is `+a_max` at every step
while `x+(x) < lo`, so the realised trajectory is `x_{k+1} = x_k + Delta(x_k)`. By A2
`Delta` is non-increasing on `[0, lo]`, hence `Delta(x) >= Delta(lo) = r_min` for every
`x ∈ [x_0, lo]`, so `x_k >= x_0 + k·r_min` for as long as `x_k <= lo`. Therefore
`x_k >= lo` for every `k >= (lo − x_0)/r_min`, and the first such integer is
`ceil((lo − x_0)/r_min)`. At the first step where `x+(x_k) >= lo`, Lemma 1's second
branch lands the successor in the band exactly, which can only be earlier. Invariance
from that step is §S1. The flat-curve form is `Delta(x) = a_max·p_nom·dt·sqrt(eta)/C −
loss·x` evaluated at `x = lo`. For the non-recovery condition, `Delta` is affine and
decreasing in `x` with a single zero at
`x* = a_max·p_nom·dt·sqrt(eta)/(loss·C)`; the forced-charge recursion converges
monotonically to `x*` from below, so when `x* <= lo` — equivalently
`a_max·p_nom·dt·sqrt(eta)/C <= loss·lo` — the trajectory is bounded above by a fixed
point at or below the floor and the band is never entered at any `N`.

Discharging case. Symmetrically, `safe_interval` returns `a_lo = a_hi = −a_max` while
`x−(x) > hi`, the drop is
`rho(x) = loss·x + (dt/C)·min(p_nom·g(x(1−loss)), p_nom)/sqrt(eta_eff(x))` — note the
*division*: energy leaving the store is `|discharge|/sqrt(eta)`, so discharging is
faster per commanded kWh than charging — and the same argument applies with `rho_min`
in place of `r_min`. `rho` is **not** monotone in `x` on all devices, so `rho_min` must
be taken as the minimum over the traversed interval rather than at an endpoint; on all
32 devices that minimum is attained at `x = x_0` (the capacity-power curve collapses to
`g(1) ≈ 0.28`, so a full battery discharges slowest), which makes the discharging bound
conservative: the state accelerates as it falls. ∎

**Corollary 1.1 (what the reported metric counts).** `stems_soc_violations_pct`
(`stems/stems_table.py`) counts post-step states outside `[x_min, x_max]`, not outside
the enforced `[lo, hi]`. From `x_0 = 0` the number of violating steps per device per
episode is therefore

```
  V <= ceil( x_min / Delta(x_min) ) − 1,    and  V = 0  whenever  Delta(x_min) >= x_min.
```

The inequality is the bound (the trajectory may gain more than `Delta(x_min)` per step);
on all 32 devices measured in M3.4 it holds with equality, because the gain varies by
less than 1.4 % across `[0, x_min]`.

### M3.3 The device parameters, as read

Read from the live simulator (A3), for the worst device of each stock — "worst" meaning
the smallest realised one-step gain from `x = 0`:

| stock | building index | `C_b` kWh | `p_nom_b` kW | `eta_b` | `p_nom·eta/C` (as quoted) | plant `Delta(0)` | plant `Delta(0.1)` |
|---|---|---|---|---|---|---|---|
| reference | 6 | 16.2 | 2.88 | 0.9237 | **0.1642** | 0.16197 | 0.16066 |
| subset 1 | 4 | 21.6 | 2.88 | 0.9764 | **0.1302** | 0.13007 | 0.12849 |
| subset 2 | 2 | **14.0** | **1.28** | **0.9601** | **0.0878** | 0.08733 | 0.08614 |
| subset 3 | 1 | 20.0 | 2.50 | 0.9709 | **0.1214** | 0.12108 | 0.11960 |

**The quoted parameters are confirmed exactly.** Subset 2's worst device is
`C = 14.0` kWh, `p_nom = 1.28` kW, `eta = 0.9601`, giving `p_nom·eta/C = 0.087776`, and
the three other stocks' worst devices give 0.164210, 0.130188 and 0.121367 — the 0.1642,
0.1302 and 0.1214 of `docs/TECHNICAL_REPORT.md` §8.3f and `docs/WSED_PAPER_DRAFT.md`.
What differs is what that product *is*; see M3.6.

### M3.4 Numerical verification

`multinode_guarantees.py recover` rolls the forced-charge recursion of the exact plant
from `x_0 = 0` on every device and compares with Claim 1.

| stock | worst device `Delta(0.1)` | `N` to `x_min`, bound | simulated | violating steps, predicted | measured `stems_soc_violations_pct` |
|---|---|---|---|---|---|
| reference | 0.16066 | 1 | 1 | 0 | **0.000000 %** (20 records) |
| subset 1 | 0.12849 | 1 | 1 | 0 | **0.000000 %** (12 records) |
| subset 2 | 0.08614 | **2** | **2** | **1** | **0.001427 %** (12 records) |
| subset 3 | 0.11960 | 1 | 1 | 0 | **0.000000 %** (12 records) |

* **32 of 32 devices**: simulated entry step equals the bound, predicted violating
  steps equal simulated, gain monotone on `[0, lo]`. No device outside the bound.
* **Subset 2 reproduces the residual exactly.** One violating building-step per episode
  on one building, `1/70 072 = 0.001427 %` of `8 × 8759` building-steps — the number in
  all twelve stored shielded records on that stock
  (`results/shield_transfer_subsets/.../subset2n8/...`: D-MAPPO, SAC, MADDPG and MetaEMS
  × 3 seeds, every one `0.001427`). The other two subsets' twenty-four records are all
  `0.0`, as are the reference stock's twenty (`results/shielded_ep50` and
  `results/dmappo_shielded_ep50`, the same four policies × 5 seeds). The rule-based arm
  on the same four stocks sits at 29.5–31.6 %.
* **The first step is the violating one**, as reported: `x_1 = 0.08733 < 0.1`,
  `x_2 = 0.1737 >= 0.1`.
* **Discharging branch.** From `x_0 = 1.0`, entry into `[x_min, x_max]` takes 1–3
  simulated steps against a bound of 1–3, none outside; on the reference stock 5 of 8
  devices are still above `x_max` after one step and 0 of 8 after two — which reproduces
  `FRAMEWORK_GUARANTEES.md` §S1's recovery sentence independently. Entry into the
  *enforced* `hi` takes 2–4 steps against a bound of 3–5, i.e. the discharging bound is
  conservative by one to two steps, as the proof says it must be.

### M3.5 Reading the bound for this fleet

The enforced floor is not 0.1. With A5's margins,
`lo_b = 0.1 + 0.03 + 0.001 + 0.5·r_b`, which is 0.1767–0.3977 across the 32 devices, so
the device whose datasheet rate is largest has the *highest* floor to climb to. For
subset 2's worst device `lo = 0.1767` and entry into the enforced band takes **3** steps
(bound 3, simulated 3) against **2** steps to the reported `x_min`. Both are Claim 1 at
different targets, and a reader of §S1 should not conflate them: the projection is
**outside its own enforced set for three steps** on that device, while the metric books
one violation.

The non-recovery condition is far from binding here: `loss_b ∈ [0.0016, 0.0086]`, so
`loss_b · lo_b <= 0.0034` against gains of 0.086–0.505.

### M3.6 Two corrections to claims already in the repository

**(a) `p_nom·eta/C` is not the plant's one-step gain, and the prose states it as if it
were.** `docs/TECHNICAL_REPORT.md` §8.3f/§8.4, `docs/WSED_PAPER_DRAFT.md` and
`RESUME.md` all write "the largest one-hour gain is `P_nom·eta/C`". The plant applies
`sqrt(eta_curve(p/p_nom))`, not `eta`, and derates the available power by
`g(x)` — at `x = 0`, `g = 0.988` on that device — and evaluates the efficiency curve at
the realised normalised power (0.9875), where it reads 0.9355 rather than the nameplate
0.9601. The three effects nearly cancel: the datasheet product is 0.087776 and the
plant realises **0.087331**. Across all 32 devices the datasheet product **overstates**
the realised gain, by a factor 0.9845–0.9991 (largest absolute shortfall 0.0069 soc).

  *Consequence for the published claims: none.* On no device does the comparison with
  `x_min = 0.1` flip, so `ceil(x_min / rate)` is the same integer under either
  arithmetic on all 32 devices, and every "cannot reach the band in an hour" conclusion
  stands. But `p_nom·eta/C` should be labelled a **datasheet proxy** rather than "the
  largest one-hour gain", and it is not a safe bound in general: because it overstates
  the rate it *understates* `N`, so a device whose proxy sits just above `x_min` and
  whose realised gain sits just below would be predicted to recover in one step and
  would not. `experiments/diagnostics/verify_report_claims.py` already pins this product
  — with a comment recording that the prose once wrote it with `eta = 1.0` — so the pin
  should be read as pinning the proxy, not the physics.

**(b) The projection is outside its own enforced band for longer than the metric says.**
See M3.5: 3 steps against the reported 1 violating step on subset 2's worst device. No
published number is wrong; the published number answers a different question.

---

## M4. Claim 2 — the cycle-phase augmented barrier

### M4.1 Why augment rather than switch

The hot-water comfort band is 45–55 degC and the weekly disinfection cycle needs
60 degC, so the ceiling cannot be a constant. The usual treatment is a switched system:
one barrier per mode, plus a dwell-time condition so the state has time to re-enter the
new mode's set after a switch. The dwell time is the part that is hard to justify — it
has to be derived from the slowest uncontrolled dynamics (here the tank's standing loss,
which is 13 to 88 hours; M4.4), and a policy that may switch faster than that breaks
the guarantee with no warning.

Augmenting the state removes the need for it. One set, one barrier, invariance by a
single induction, and the number of transitions is irrelevant.

### M4.2 The augmented state and the barrier

```
  z = (soc, q, k) ∈ [0,1] × Q × N,          Q = {N, D}
```

`soc` is the store level (A6), `q` the disinfection phase, `k = steps_since_cycle` the
window counter `LegionellaCycleBarrier` already maintains. The ceiling is a function of
the phase and the floor is not:

```
  soc_hi(N) = soc_of_temperature(55)  = 0.90          soc_lo = soc_of_temperature(45) = 0.70
  soc_hi(D) = soc_legionella          = 1.00
  h(z) = min( soc − soc_lo,  soc_hi(q) − soc )
  S    = { z : h(z) >= 0 } = ( [soc_lo, 0.90] × {N} )  ∪  ( [soc_lo, 1.00] × {D} )
```

Since `soc_hi(D) > soc_hi(N)`, the two slices nest: `S_N ⊂ S_D`. That asymmetry is the
whole argument — entry into the cycle is a *relaxation* and is free; the exit is the
transition that can strand the state outside the set it is returning to.

Tank dynamics, from `ShadowTank.step` with the action `u ∈ [0, 1]` and the draw `draw`
exogenous:

```
  soc+ = clip(  soc·(1 − loss)  +  ( u·P_hp·dt − draw ) / C ,  0,  1 )
```

The heat source can only add heat: there is no active cooling, so the state can be
driven **up** at will and comes **down** only through the standing loss and the draw.

### M4.3 The admissibility condition

> **Claim 2.** Under A6, A7, A8 and A11, let the transition relation be guarded by
>
> ```
>   q_t → q_{t+1} is admissible  ⟺  soc_t ∈ [soc_lo, soc_hi(q_t)] ∩ [soc_lo, soc_hi(q_{t+1})]
>                                ⟺  soc_t <= min( soc_hi(q_t), soc_hi(q_{t+1}) )  (the floor is common)
> ```
>
> and let the action be projected at every step into
> `U(soc, q) = { u ∈ [0, 1] : soc+(soc, u) ∈ [soc_lo, soc_hi(q)] }`. Then `S` is forward
> invariant for the augmented system, with **no dwell-time condition and no bound on the
> number of transitions**, provided `U(soc, q)` is non-empty at every step — which holds
> for the ceiling unconditionally (`u = 0` never raises the state) and for the floor at
> every state of `S` when
>
> ```
>   draw_t  <=  P_hp · dt  −  C · loss · soc_lo .
> ```
>
> Concretely, in temperatures: **entry** `N → D` is admissible from anywhere in the
> comfort band, because 55 <= 60; **exit** `D → N` is admissible only at a tank
> temperature **at or below 55 degC**. A request to exit above 55 degC is refused.

**Proof.** Induction on `t`. Suppose `z_t ∈ S`, i.e. `soc_t ∈ [soc_lo, soc_hi(q_t)]`.

*Step with no transition.* `q_{t+1} = q_t`, and the projection puts `u_t ∈ U(soc_t, q_t)`
whose definition is exactly `soc_{t+1} ∈ [soc_lo, soc_hi(q_t)]`. So `z_{t+1} ∈ S`
provided `U` is non-empty. For the ceiling, `u = 0` gives
`soc+ = soc(1 − loss) − draw/C <= soc <= soc_hi(q_t)`, so the ceiling is always
enforceable by withholding heat — no active cooling is needed and the ceiling never
makes `U` empty. For the floor, the largest reachable successor is
`soc(1 − loss) + (P_hp·dt − draw)/C`, and requiring it to be at least `soc_lo` at the
worst case `soc = soc_lo` gives the displayed draw condition.

*Step with a transition.* The guard gives `soc_t ∈ [soc_lo, soc_hi(q_{t+1})]`, so
`(soc_t, q_{t+1}) ∈ S` **at the transition instant**; the step then proceeds as above
in the new phase. The guard is evaluated on the pre-transition state, so no reachability
argument is needed and no time has to elapse: the state never leaves `S`, and therefore
there is nothing for a dwell time to repair. ∎

**What makes the dwell time unnecessary, stated plainly.** A switched argument needs one
because it allows a switch from a state that is feasible in the old mode and infeasible
in the new one, and then has to wait for the state to come back. The guard forbids
exactly that transition. The price is that the phase is **not freely controllable**: the
controller proposes, the guard disposes.

**On an inadmissible request.** The transition is **refused**: `q` stays at its current
value, the relaxed ceiling therefore stays in force, the barrier keeps enforcing `S_D`,
and the refusal is recorded (the same way `LegionellaCycleBarrier` records
`missed_windows`). Refusal — not clipping the state, not accepting and reporting a
violation — is the only option that keeps `S` invariant, because the state is already
above the post-transition ceiling and nothing in the actuator set can bring it down
within the step. The deadline is not jeopardised by a refusal: `observe()` resets the
window when the *level* is reached, independently of the phase, so a cycle counts as
completed at the moment `soc >= soc_legionella` even if the phase cannot yet return.

### M4.4 What the refusal costs, on this data

Because the only downward motion is loss plus draw, the exit lag with no draw is the
smallest `m` with `1.00·(1 − loss)^m <= 0.90`, i.e.
`m = ceil( ln(0.9) / ln(1 − loss) )`. The 32 tanks of these four stocks carry
`loss ∈ [0.0012, 0.0084]` (read from `b.dhw_storage.loss_coefficient`), giving

```
  loss = 0.0084  ->  13 steps      loss = 0.0012  ->  88 steps
```

So after a cycle completes the relaxed ceiling stays in force for **13 to 88 hours**
unless a draw pulls the tank down sooner, and during that window the comfort ceiling is
not enforced. That is a real and quantified cost of the no-dwell-time construction: it
converts a dwell-time *assumption* into a measurable period of relaxed ceiling, which
is the trade this claim makes.

The floor condition of Claim 2 requires `P_hp >= draw_max/dt + C·loss·soc_lo`, which on
these 32 tanks is **3.76–7.42 kW**. Evaluated with A8's optimistic stand-in (CityLearn's
autosized heater, 3.94–7.77 kW) it is satisfied on **31 of 32** devices and fails on one
(subset 3: 4.644 kWh available against a 4.647 kWh peak draw hour — short by 0.003 kWh,
one hour of the year). With a real heat-pump source, whose compressor is the reason the
tank exists, the condition would fail far more often; `P_hp` is unsourced (A8) and this
document does not invent one, so the honest statement is: **the floor of Claim 2 is
conditional on a draw bound that is satisfiable only with a source sized like
CityLearn's.** The residual when it fails is the floor shortfall `soc_lo − soc` in state
of charge — equivalently `(45 degC − T_tank)` once converted — and, when the store is
emptied outright, `unmet_kwh`, which `ShadowTank.report` already carries as its headline.
Neither is measured here: no run in this repository enforces a hot-water ceiling or
floor (M4.5b), so there is nothing to measure yet.

### M4.5 Two premises corrected

**(a) `dhw_soc_cap = 0.95` does not forbid reaching the disinfection temperature; it
forbids *requiring* it.** `soc_cap` enters `DeadlineStorageBarrier.required_soc`, which
clips the *required level*, and the barrier's `project` only ever raises an action to a
floor. So with `ThermalConfig.dhw_soc_cap = 0.95` the readiness barrier cannot demand
more than 57.5 degC, and a deadline barrier built with that cap could never certify a
60 degC cycle — but nothing stops a controller from heating the store to 1.00.
`LegionellaCycleBarrier` already passes `soc_cap = 1.0` and so escapes the clip. The
correct statement of the defect is therefore: *the requirement is clipped below the
disinfection level, not the state.*

**(b) There is no ceiling on the hot-water store today, phase-dependent or otherwise.**
`FlexibleLoad.has_band` requires `plant` to expose `safe_interval`, which only
`BatteryModel` does; `FlexibilityPortfolio.from_environment` gives the tank loads a
deadline and `band=None`, and `stems/flexibility.py` says so in its docstring ("a
tank's state of charge is bounded by its own dynamics rather than by a configured
band"). Claim 2 therefore adds a constraint where none exists, and there is no
before/after comparison to make. The missing mechanism is exactly one method —
`safe_interval` on the tank plant, with `hi` taken from `soc_hi(q)` — after which
`project_band` enforces the augmented barrier unchanged.

---

## M5. Claim 3 — two-node domination: **the proof does not go through**

The aim was to keep the scalar structure: enforce only the node that binds first in each
mode — the top node for comfort, the bottom node for hygiene — and prove the other
follows, using monotonicity of plug-flow displacement with inter-node conduction. The
attempt is in M5.1–M5.2 and most of it works. It then fails, on two of the three cases
it was attacked with (M5.3–M5.5), for reasons that are not repairable by a tighter
argument.

### M5.1 The model

Two nodes of equal energy capacity (A9), states `soc_top`, `soc_bot`, cold inlet at
`soc = 0`. Per step, as four operators:

```
  (L) standing loss        soc_i  <-  soc_i · (1 − loss)
  (W) draw, beta = draw/C_node, plug flow upward with cold inlet at the bottom:
        beta <= 1:  soc_top <- (1−beta)·soc_top + beta·soc_bot
                    soc_bot <- (1−beta)·soc_bot
        beta  > 1:  the front crosses a whole node within one step — see M5.4
  (H) injection at the bottom (A9):  soc_bot <- soc_bot + u·P_hp·dt / C_bot
  (K) conduction, top to bottom:     soc_i   <-  soc_i ∓ kappa·(soc_top − soc_bot)
```

The domination argument rests entirely on one ordering invariant:

```
  (STRAT)   soc_top >= soc_bot .
```

### M5.2 What goes through

**Lemma 3 (the passive operators preserve STRAT).** `(L)` scales both nodes by the same
factor, so it preserves the ordering and the sign of the difference. `(W)` with
`beta <= 1` gives
`soc_top+ − soc_bot+ = (1−beta)(soc_top − soc_bot) + beta·soc_bot >= 0` whenever
`soc_top >= soc_bot >= 0`. `(K)` maps the difference to `(1 − 2·kappa)(soc_top −
soc_bot)`, which is non-negative and no larger in magnitude iff `kappa <= 1/2`. So loss,
a sub-node draw and conduction each preserve STRAT. ∎

**Lemma 4 (domination, given STRAT).** In phase `D`, if the enforced bottom node
satisfies `soc_bot >= soc_legionella`, then `soc_top >= soc_bot >= soc_legionella`, so
the top node is disinfected too and *a fortiori* above the comfort floor — the hygiene
constraint on the coldest node implies everything above it. In phase `N`, the top node
carries the comfort band and the bottom node carries no active constraint (hygiene is a
weekly deadline, not a per-step band), so enforcing the top node is enforcing everything
active. ∎

Lemmas 3 and 4 are the claim, and they are true. Note what Lemma 4 is *not*: it is not
"one node dominates the other". It is "in each mode the enforced node dominates the
constraint active in that mode" — comfort on the top says nothing whatever about the
bottom, which is fine only as long as nothing is required of the bottom in phase `N`.

### M5.3 Attack 1 — injection breaks STRAT in one step

`(H)` raises the bottom node alone. Starting from a stratified tank at
`soc_top = soc_bot = 0.7` and charging, one step gives `soc_bot > soc_top`: the model
places colder water above warmer, which is not a tank. STRAT fails on the **normal
operating condition of a bottom-injected heat-pump water heater**, not on a corner case,
and with STRAT goes Lemma 4: a model state with `soc_bot = 1.00` and `soc_top = 0.50`
satisfies the enforced hygiene constraint while leaving the served node at 35 degC.

The repair is a buoyancy operator: whenever `(H)` would invert the order, mix the two
nodes to a common temperature, conserving energy. That restores STRAT — but a mixing
rule that fires whenever the source is on makes the tank **fully mixed exactly while it
is charging**, i.e. collapses the two-node model to `ShadowTank` precisely in the regime
the second node was introduced to resolve. The modelling gain that motivated two nodes —
A6's stated error, that a cold bottom layer makes the fully mixed model understate the
energy needed to disinfect — is deleted by the operator that saves the proof. One can
have the stratified model or the scalar domination argument, not both, and no choice of
`kappa` changes that (A10: neither `kappa` nor the mixing rule has a sourced value
anywhere in this repository).

### M5.4 Attack 2 — the cold front crosses a node inside one step (measured, on every building)

The `beta <= 1` branch of `(W)` is the convex combination Lemma 3 needs. At `beta > 1`
the whole top-node volume is displaced within the step — its final content comes from
the bottom node, and for `beta > 2` from the inlet itself — so the front's position
inside a node is not resolved by the discretisation at all, the one-step map is no
longer that convex combination, and the result depends on the operator-splitting
order — `(W)` then `(H)` injects into a node the draw has already flushed, `(H)` then
`(W)` draws out the heat just injected. Two defensible conventions, two different node
states, two different verdicts on the same command.

This is not hypothetical at a one-hour step. From
`multinode_guarantees.py tank`, with the tank split into two equal nodes
(`C_node = C/2`), on the 32 building-tanks of the four stocks:

| quantity | value |
|---|---|
| devices with `beta_max >= 1` | **32 of 32** |
| `beta_max` range | 1.040 – 1.957 |
| share of that building's draw hours with `beta >= 1` | 0.07 % – 2.90 % |
| building-hours with `beta >= 1` | 454 of 80 835 draw hours |
| largest single-hour draw as a fraction of the **whole** tank | **0.978** |

A single hour's draw replaces up to 98 % of one tank's stored energy. At that
resolution the tank is being flushed, not stratified, and a two-node discretisation has
no front to track. The failure is rare per hour and certain per year: every one of the
32 devices enters the regime, the worst on 2.9 % of its draw hours.

### M5.5 Attack 3 — the source at the top

If the disinfection boost is an **upper** electric element rather than a bottom
condenser — which is how a legionella boost is usually implemented, and is the variant
`LegionellaSpec.t_heat_pump_max_c` exists to describe — then STRAT is preserved
(the top rises, the bottom does not) and Lemma 4 fails in the other direction: the
served node reaches 60 degC while the bottom node stays cold, and it is the bottom node
that harbours the organism. Enforcing the node that binds first for *comfort* certifies
nothing about hygiene. This case does not break the mathematics; it breaks the
assignment of constraints to nodes, and it is the physical reason Legionella controls
target the bottom of the cylinder.

### M5.6 Verdict

> **Claim 3 (as attempted, and it fails).** Enforcing only the binding node implies the
> other **only** under (i) STRAT, `soc_top >= soc_bot`, and (ii) `beta <= 1`, the draw
> not exceeding one node's energy per step. Under A9's bottom injection, (i) is violated
> by the source itself at every charging step unless a buoyancy-mixing operator is
> added, and that operator collapses the two-node model to the fully mixed one while the
> source is on — removing the stratification the second node was introduced to
> represent. (ii) is violated on **all 32** building-tanks of the four stocks, on up to
> 2.90 % of a building's draw hours, with single-hour draws reaching 0.978 of the whole
> tank. **The domination argument therefore does not support a scalar projection on this
> testbed at a one-hour step.** It would hold on a sub-node time step with a buoyancy
> rule and a sourced `kappa`, none of which exists here.

A third obstacle, independent of the mathematics: **the state does not exist.** The
observation vector carries one `dhw_storage_soc` (`stems/observations.py`), and
CityLearn's `StorageTank` has one. A two-node barrier would be enforced against node
temperatures no instrument in this testbed reports, estimated by the very model whose
stratification assumption is in question — the same category of error as
`FRAMEWORK_GUARANTEES.md` §S4's reason for gating the comfort barrier off, and it
should be gated off for the same reason until node measurements exist.

---

## M6. The fallback: per-node barriers, and what they cost

Since Claim 3 fails, the node constraints have to be enforced per node rather than
through one dominating node:

```
  minimise    || a − a_policy ||^2
  subject to  soc_i+(soc, a, draw)  ∈  [ soc_lo_i, soc_hi_i(q) ]    for i ∈ {top, bot}
              a ∈ [0, a_max]^m
```

Two cases, and they cost very different things. The distinction matters because the
first is cheaper than the brief assumed.

**One actuator per store (`m = 1`): the exact inverse survives.** With a single heat
input and the state and draw known at the step, each node's successor is monotone
non-decreasing in `u` — the bottom node directly, the top node through displacement —
so each node's admissible set is an *interval* in `u`, obtainable by the same
`_bisect_toward_zero` call `BatteryModel.safe_interval` already uses, and the
projection is the intersection of two such intervals. No quadratic programme is needed,
the one-step inverse stays exact, and the cost is two bisections instead of one. What
*is* new is that the intersection can be **empty**: the bottom node may need heat that
raises the top node past its ceiling, which no action resolves. Emptiness is the
feasibility verdict this path owes and must report rather than silently mid-point, which
is what `safe_interval` currently does when its own bounds cross
(`np.where(crossed, mid, ...)`).

**More than one actuator, or nodes solved jointly with the cap: the QP, and exactness
goes.** A heat pump plus a resistive element is `m = 2` — the configuration
`LegionellaSpec.t_heat_pump_max_c` and `cop_legionella` already describe — and so is any
formulation that enforces the node bands jointly with the district import cap instead of
sequentially. There the admissible set is no longer an interval and a programme is
required. What that costs, precisely:

1. **Exactness, which is the guarantee.** §S1 is "hard *because* the inverse is exact,
   and for no other reason", with the cost of getting it wrong measured in the same
   table: a uniform `r = 0.1` surrogate breaches the band in 9.2 % of states and drives
   the floor to `x = 0`. The node map is **piecewise** affine in the actions, not
   affine: it saturates at `soc = 1` (spill) and `soc = 0`, the served draw is a
   `min(draw, available)`, and M5.3's buoyancy rule is a switch on whether injection
   inverts the node order. The exact one-step feasible set is therefore a *union* of
   polyhedra, and a single QP is a convexification of it — a relaxation on some pieces
   and a restriction on others. That reintroduces model error in the protected
   direction, which is the one thing the present construction does not have. The
   0/52 488 result does not transfer and would have to be re-measured per site; it must
   not be reported as zero by analogy.
2. **The closed form.** Today the projection's output is a deterministic function of the
   state, auditable by replay with no external dependency. A programme's output is not:
   reproducing a run requires pinning the solver and its version, and the shield acquires
   failure modes it does not currently have — infeasible, iteration limit, tolerance — in
   which it returns no action at all. A safety filter that can fail to answer needs a
   declared fallback for that case, which is a design decision rather than a refactor.
3. **Cost in the loop.** One solve per building per step is 70 072 solves per episode per
   arm. `stems/cbf.py` records that the fleet linear programme is currently the only
   optimisation in the control loop; this would make it the second, on every building.
4. **The minimum-intervention property survives**, provided the objective is
   `||a − a_policy||^2`: that is what `_bisect_toward_zero`'s smallest-magnitude
   convention buys today, and the programme reproduces it by construction rather than by
   bracketing around zero.

**What is gained, stated fairly.** A programme returns a feasibility verdict and a dual
price per node. `FRAMEWORK_GUARANTEES.md` §S5.4 records that the CBF path has a working
infeasibility detector (`CBFShield.feasibility_report`) that no live code calls; this
formulation makes the verdict a by-product of the projection rather than a separate
uncalled computation. That is a real improvement in exactly the place the audit found a
silent breach — and the interval-intersection fallback above can report it too, at no
cost in exactness.

---

## M7. What this adds to the ledger

| statement | status | where |
|---|---|---|
| Entry into the enforced band from any `x_0`, in `N = ceil(dist / rate_min)` steps, then §S1 invariance | **proved**, verified on 32/32 devices, exact on the charging branch and conservative by 1–2 steps on the discharging branch | M3 |
| The one violating building-step on subset 2 is predicted from the device parameters, not fitted | **measured**: predicted 1 = `0.001427 %`, observed `0.001427 %` in all 12 shielded records | M3.4 |
| `p_nom·eta/C` is the plant's largest one-hour gain | **corrected**: it is a datasheet proxy that overstates the realised gain on 32/32 devices; conclusions unaffected | M3.6a |
| The recovery command overrides the per-building and district import caps | **measured**, and the priority ordering is undeclared | M3.1 |
| The augmented set is forward invariant with no dwell-time condition, given the guard | **proved**, conditional on a draw bound | M4.3 |
| The exit guard holds the relaxed ceiling for 13–88 h after a cycle on these tanks | **measured** from the tanks' loss coefficients | M4.4 |
| `dhw_soc_cap` forbids reaching 60 degC | **corrected**: it clips the *required level*, not the state | M4.5a |
| A ceiling on the hot-water store exists today | **corrected**: it does not; `band=None` for every tank load | M4.5b |
| Enforcing the binding node implies the other | **fails**: STRAT broken by bottom injection, and `beta > 1` on 32/32 devices | M5.6 |
| A two-node barrier can be enforced against the observed state | **false**: one `dhw_storage_soc` exists; node temperatures do not | M5.6 |
| Per-node enforcement needs a quadratic programme | **partly corrected**: with one actuator per store the per-node sets are intervals and the exact inverse survives (two bisections); a programme is needed only for multiple actuators or joint cap enforcement | M6 |
| The fallback's residual once a programme is needed | **unknown until measured**; it must not be inherited from §S1 | M6 |

Not addressed here: the district cap (§S2), vehicle deadlines (§S3), comfort (§S4), and
every "does not exist" row of §S6. Claim 1 is about one device's band and says nothing
about the coupled set; Claims 2 and 3 are about one tank.

---

## M8. Reproducing the numbers

```sh
source scripts/env.sh                       # XDG_CACHE_HOME, or the schemas do not resolve
python experiments/diagnostics/multinode_guarantees.py                 # all four sections
python experiments/diagnostics/multinode_guarantees.py recover tank    # named sections
python experiments/diagnostics/multinode_guarantees.py --json out.json
```

Sections: `recover` (M3.3, M3.4, M3.5), `down` (M3.4's discharging branch), `guard`
(Lemma 2), `tank` (M4.4's draw bound and M5.4's `beta`). Each builds four CityLearn
environments to read the **autosized** device parameters (A3), which takes a few minutes
and no GPU; nothing is trained. The measured violation rates in M3.4 are read from
`results/shield_transfer_subsets/*/*.json`, key `eval.stems_soc_violations_pct`.

