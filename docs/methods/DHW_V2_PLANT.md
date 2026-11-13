# DHW v2 plant, filter and environment (branch `dhw-v2`, v2.1)

**Code.** `stems/dhw_v2/`: `heatpump`, `tank`, `calibration`, `borehole`, `building`, `data`, `plant`,
`filter`, `controllers`, `identify`, `env`.

**Tests.** `tests/test_dhw_v2.py`.

**Data format.** `docs/DHW_V2_DATA_FORMAT.md`.

**Smoke runs.**
* `experiments/dhw_v2/smoke_year.py` (45 runs).
* `experiments/dhw_v2/smoke_results/`: one JSON per run, `SMOKE_TABLE.md` and `validate_store.json`.
* The v2.0 results are kept in `smoke_results/v2.0/`.

**Theory.** `docs/DHW_THEORY.md` on branch `dhw-theory` (T1–T5). Section 6 says where v2 implements
each statement and where it differs.

**v1 code.** The v1 modules (`dhw_plant`, `dhw_stratified`, `dhw_barrier`, `heatpump_hplib`,
`experiments/dhw_paper/`) are untouched; they are the "before" of the before/after comparison.

**Running.** `stems/__init__` imports torch, so use the repo venv: `.venv/stems/bin/python`,
`PYTHONPATH=.`, `XDG_CACHE_HOME=<repo>/.citylearn_cache`. pygfunction 2.3.1 is in `requirements.txt`.

Every number that is an ASSUMPTION rather than a sourced value is marked as such.

## 0. What changed in v2.1 (defaults frozen here)

| item | v2.0 | v2.1 default | why |
|---|---|---|---|
| store model | N = 2 | **N = 10** (N = 2 sensitivity, N = 20 convergence) | V40: N = 10 gives 218 L against the certified 227 L (-4 %); N = 2 gives 164 L (-28 %) |
| DHW charging speed | full speed | **q = f q_max with f = 0.4579**, calibrated to the certified EN 16147 heat-up of 1:55 h; the same law at 65 C (nothing certifies the legionella speed) | full speed heats the store in 0.88 h |
| comfort certificate | this hour + optional one-hour terminal condition | **24-h reserve set** (robust control-invariant set by backward recursion, evaluated as a max-charge rollout); its first step is theory T2's one-hour-ahead certificate | recursive feasibility |
| margin | clipped at 0, two-sided coverage claim | **unclipped**, one-sided bound `freq <= alpha + (range + eta)/(eta T)` (T3) | the two-sided claim was false (theory §8b) |
| disinfection trigger | just in time (d_hat + m, actual capacity) | **robust (T4)**: conformal hour-of-day bounds at beta = 0.02, one-hour lookahead, retry reserve R = 2, worst-case capacity over the brine band; checked every hour of the window | completion guarantee |
| hold definition | every node >= 60 C (mesh-dependent) | **bottom 20 % of the volume, volume-weighted mean >= 60 C at every sub-step of a clock hour** (NZ G12/AS2 §3.5.1(c) reading; HSE HSG274). Whole-volume mean and all-nodes reported as secondary | theory Remark 4.5 |
| Thermia logic | start 45 / stop 52 C, program due after 7 days | **start 47 / stop 53 C** (start above the 45 C floor), sensor 0.6 H, program due at the start of the run; legionella stop on the bottom-20 % mean (sensor position); all parameters swept | the 45 C start = floor artefact |
| brine band (worst case) | 2 K | **3 K** | the realised 24-h drop reached 2.8 K |
| phase guard | none in v2 | **none in v2.** A forced N→D entry is always admitted (`dis = proposal or trigger`). The floor half that theory T1 found vacuous does not exist here, so T4's assumption "the guard admits D when the trigger fires" holds trivially. v1's `DisinfectionBarrier` is left as is (the before) | theory T1 |

## 1. Heat pump: Thermia Calibra RXT 12 400 V (`heatpump.py`)

**Map.** `stems/thermia_rxt12.py`, fitted to the 21 certified KEYMARK 012-C700403 points. The form is
`1/COP = b0 + b1 L + b2 L^2 + b3 Q + b4/Q + b5 s^3`. Leave-one-out RMS error 4.18 %, max 9.07 %
(`docs/THERMIA_RXT12_MAP.md`).

**Limits.**
* The electric-input cap is 4.7 + 0.2 kW (datasheet).
* Cycling below q_min follows EN 14825.
* The supply goes up to 70 C on the compressor alone (Thermia, 2026-10-19, said about hot water).
  Since branch paper-b/thermia-free (fix-list HW1) the 3/6/9 kW auxiliary element tops up SPACE HEATING at
  COP 1 when the compressor's heat is short of the indoor minimum (`PlantConfig.aux_heater`, default
  `auto_min`; `off` reproduces the earlier plant). DHW and the legionella program stay compressor-only.
  Every Sweden number produced before this fix has the heating shortfall of HW1 in it.
* At B0, q_max = 11.15 / 10.39 / 10.01 / 9.82 kW at 35 / 55 / 65 / 70 C supply.

**Extrapolation flags.**
* `extrap_supply`: supply above 55 C, i.e. all disinfection energy.
* `extrap_source`: brine outside 0–10 C, which is all of Texas.
* Both are reported per hour and summed per run.

**Conservative electric bound.** `p_el(q) <= P0 + kappa q`, proved in the module docstring and tested
on a grid.

**DHW speed law (`calibration.py`).** `q = f q_max(T_src, T_sup)`, with `f` calibrated as follows:
* The certified EN 16147 heat-up (store at 10 C, B0, 55 C supply, standing loss on) takes 1:55 h to
  reach the heat content of the V40 starting state. That state is a uniform 52.3 C for N = 10.
  ASSUMPTION: the heat-up ends at the state from which the V40 tapping starts.
* The result is f = 0.4579: 4.76 kW at B0/W55, COP 3.29 (the certified XL-cycle COP is 3.06,
  including standby). N = 20 gives 0.448.
* N = 2 cannot be calibrated: its V40 start state needs 57.2 C, above the 55 C supply. The speed is a
  property of the unit, so the N = 10 value is used for every N.
* `PlantConfig.dis_speed_frac = None` means the same law at 65 C. That is a statement that no data
  allow full speed there, not a sourced value.

## 2. Store (`tank.py`)

* **Volume.** 184 L (datasheet; KEYMARK EN 16147). TWS external stratified charging.
* **Nodes.** N equal-volume nodes, node 0 = top. N = 2 reproduces v1's closed form to 1e-9; energy is
  conserved exactly for N = 2–40.
* **Standing loss.** `UA (T - T_room)` per node, UA = 1.71 W/K. This comes from the certified 34 W
  standby, read as electric input: (Pes - PSB) x COP_reheat = 52.7 W at 50.9 C / 20 C (ASSUMPTIONS:
  mean store at 50.9 C, 20 C ambient). The alternatives are 3.41 and 1.10 W/K. The CityLearn loss
  convention and CityLearn tank sizes can still be selected.
* **Sensor.** At 0.6 H by default (`EnvConfig.sensor_height`; ASSUMPTION). That is node 3 for N = 10.
* **V40 check** (`calibration.v40`; tapping at 10 L/min, the rate is an ASSUMPTION; start temperature
  chosen so the mean outlet is 50.9 C): 164 L at N = 2, 218 L at N = 10, 247 L at N = 50, against the
  certified 227 L.

## 3. Sub-hourly plant (`plant.py`)

**Integration.** `dt_min` from 1 to 5 min (default 5) under hourly commands.

**Sub-step order.**
1. Loss.
2. Draw.
3. Record the minima: top node, V40, coldest node, bottom-20 % mean, volume mean, and the delivered
   water temperature (the coldest node the draw reached).
4. DHW charge.
5. Space heating.

**Shared compressor.** DHW runs first in the hour for `60 u_d` minutes, then space heating runs at the
modulated rate in the remaining time.

**Native Thermia thermostat.** Unfiltered only; evaluated every sub-step.

**Hold rules (`PlantConfig.hold_rule`).** In every rule the metric must be >= 60 C at every post-draw
sub-step of a clock hour.
* `bottom20` (PRIMARY): volume-weighted mean of the bottom 20 % of the volume. That is the bottom node
  for N = 2, the bottom 2 nodes for N = 10 and the bottom 4 nodes for N = 20.
* `whole_mean`: the volume mean of the store.
* `all_nodes`: the v2.0 rule; mesh-dependent.

Every run reports week completion under all three rules.

**dt check** (two weeks, houses 0–1, dt = 1 against dt = 5 min): electricity 152 vs 152 kWh, 0 vs 0
violation hours.

## 4. Brine (`borehole.py`) and space heating (`building.py`)

These are unchanged from v2.0:
* one pygfunction borehole per house, with the parameter table and sources in the module docstring;
* the v1 correlation is selectable;
* 5R1C / 1R1C / demand buildings and Thermia's heating curve;
* `DEFAULT_SE_SFH` is a PLACEHOLDER;
* Texas houses use demand mode (CityLearn `heating_demand`) with no seasonal stop.

The realised maximum 24-h drop of the DHW brine inlet is reported per run as `brine_max_24h_drop_k`.
It reached 2.8 K on Texas, inside the filter's 3 K band.

## 5. Controllers (`controllers.py`)

**ThermiaLogic** (an earlier reading of Thermia's logic; NOT confirmed as what Thermia does today: the
thresholds, sensor height and 65 C supply below were invented here, see fix-list HW3. The agreed default is
`ThermiaWeekly`, scored by `stems.dhw_v2.hygiene.ThermiaRuleScorer`).
* DHW: start/stop hysteresis on the tank sensor. Defaults are start 47 C and stop 53 C (ASSUMPTIONS,
  start above the floor). Compressor only, DHW priority.
* Legionella program, as modelled here:
  * it is due 7 days after the last completed program, and also at the start of the run;
  * it starts together with the next normal DHW start;
  * it heats at 65 C;
  * it stops when the bottom-20 % mean reaches 60 C (the sensor position).
* Thermia gave no hold time. The 1-h hold the paper counts is our definition, applied to the
  program's outcome.
* Modes: `native` (sub-step) and `hourly` (simulated on the plant twin with the causal draw forecast,
  proposing a fixed fraction; used with the filter).
* Sensitivity runs: start 45/46/48, stop 52/55, sensor 0.5/0.75 H.

**ReactiveController.** The v1 rule, with a weekly timer at Sunday 02:00. `timer_dow=None` disables
the timer.

**PriceAwareController.** Ported from v1.

**NullController (new).** It never charges and never asks for a disinfection; space heating is set to
the set point. This is the worst-case proposer for the filter: every DHW kWh is a filter intervention.
That is where an RL agent that exploits the filter would sit.

## 6. The joint filter (`filter.py`) and its theory

There is ONE QP over each unit's `(u_d, u_s)`, coupled by an optional district cap.

**Hard constraints.**
* The box.
* Time sharing: `u_d + u_s <= 1`.
* House and district electric caps, through the chord upper bound. That bound is exact for the oracle
  model but only as good as the model for the identified one.

**Soft constraints.** The minimum-slack LP uses weights 4/2/1 for DHW comfort, disinfection and indoor
comfort. The projection is exact.

**T2 + reserve set (comfort).**
* Certified draws: `D_k = max(d_hat(t+k) + m, 0)`.
* Worst-case DHW capacity: `f q_max(T_src - 3 K, 55)`.
* `u_max` is the largest DHW fraction the house cap allows.
* Backward recursion: `C_H` = all states, and `C_k` = the states from which hour t+k at `u_max` keeps
  the top node >= 45 C at every sub-step and ends in `C_{k+1}`.
* By monotonicity in the state and in u (property-tested), this is the max-charge rollout. It is
  evaluated with a full-store table F; `test_reserve_tail_equals_brute_force_rollout` checks it.
* The filter requires
  1. the current hour certified at `D_cert = max(d_hat + m, 0)`, and
  2. the end-of-hour state in `C_1`.
* The first step of (2) is exactly theory T2(b): a constraint on the end of hour t given the quantile
  draw of hour t+1, which is announced an hour ahead. In the sub-hourly plant the post-draw state of
  the first sub-step of hour t+1 does not depend on `u_{t+1}`, and the later sub-steps are helped by
  the charge (theory Theorem 2′).
* `reserve_h = 1` is the T2 one-hour-ahead certificate; `reserve_h = 0` is the v2.0 one-hour
  certificate.

**Recursive feasibility.**
* Suppose hour t was certified, its draw stayed within its quantile, the next certified draws do not
  grow, the brine stays in its band and the cap does not tighten. Then hour t+1 is certifiable at
  `u_max`. The one new hour t+1+H is covered once the rollout refills the store and the full store
  survives that hour. That always holds for a unit that is not power-limited.
* The measured falsifier is `recursive_feasibility_failures`: an hour that starts >= 45 C, yet no
  admissible u certifies it, although the previous hour was certified and within quantile.
* Hours that START below 45 C (`inherited`) follow a draw exceedance in the previous hour. They are
  outside every certificate (theory Theorem 1′: they cannot occur while draws stay within the bound).

**T3 (margin).**
* `m <- m + eta (1{e > m} - alpha)`, unclipped, with `e = D - d_hat`.
* A violation on a certified hour implies an exceedance in hour t or in the t-1 plan for t (T3(iv)).
* `freq(e > m) <= alpha + (range + eta)/(eta T)` (one-sided; `exceed_freq_upper_bound`).
* The per-step bound `v <= (delta - m)^+` holds for the oracle model with the certified intra-hour
  shape.

**T4 (disinfection).**
* Conformal hour-of-day draw bound: rank `ceil((n+1)(1-beta))` among the n past draws at that hour of
  day; +inf with too little history, which makes the trigger fire at once, i.e. a timer.
* Rule: force the cycle at hour k if `L_k - 1 < n*(x', k+1) + R`, where x' is the state after this
  hour idle under its bound, and n* is the plan length at full output at 65 C, worst-case capacity,
  until a clock hour meets the hold rule. It is checked every hour while the hold is owed.
* Once forced, the cycle runs at full output at 65 C until the hold.
* Guarantee: completion inside the window if every draw from firing to the planned hold is within its
  bound and the capacity is within its band.
* Comparison triggers: `jit`, `recourse`, `quantile` (interpolated q0.9, no lookahead) and `none`
  (the proposer's timer or Thermia's program).

**T1 / guard.** There is no phase guard in v2 (section 0).

**Identified model (6c).**
* Fitted from four weeks of Thermia-logic logs, with noise of 0.2 K and 1 %.
* UA from idle hours.
* Delivered heat from the store balance.
* `q_max = g0 + g1 T_src + g2 T_sup`, from logged DHW rate divided by the known speed setting.
  v2.1.0 had double-counted the speed; fixed in this commit.
* `1/COP = h0 + h1 L`.
* Texas houses 0–1: UA = 1.73 W/K (plant 1.71); g = (12.26, 0.228, -0.029), against the map's
  (12.48, 0.242, -0.038).

## 7. Environment (`env.py`): interface

**Construction and stepping.**
* `env = DHWDistrictEnv(data, EnvConfig(...))`
* `obs, info = env.reset(seed)`; `obs` is (B, 24) float32, with names and meanings in `OBS_NAMES`.
* `obs, reward, terminated, truncated, info = env.step(a)`, with `a` shaped (B, 3) in [0, 1] or a dict
  per agent: `[u_d, u_s, u_dis >= 0.5]`.
* `env.step_command(cmd)` serves rule controllers; a native `thermostat` is allowed only when the
  filter is off.

**Reward.**
`-(price * elec + w_comfort * h_below_floor + w_sh * K h below T_min + w_dis * missed week
+ k_correction * price * |executed - proposed|_el)`. Defaults 1 / 0.5 / 5 / 1. `k_correction = 0`
gives the unpenalised arm.

**info.**
* `proposed`, `executed`, `filter` (l_c, l_dis, l_s, slacks, intervened, D_cert, D_next_cert, d_hat,
  m, infeasible_c, reserve_infeasible, power_limited, must, H_need, trigger_infeasible,
  delta_el_kwh, elec_bound_kw).
* `delta`, `m`, `bound`, `violation`, `v`, `declared_infeasible`, `inherited`, `announced_prev`,
  `explained_prev`, `unexplained`, `missed_week`.
* `record`: the plant hour, including `extrap_supply`, `extrap_source`, `hold`, `hold_bottom20`,
  `hold_whole_mean`, `hold_all_nodes`, `bot20_min_c`, `delivered_min_c`, `delivered_below_floor`.

`env.summary()` gives every run metric (section 9 columns); `env.spec()` gives the specification.

## 8. Data interface

Unchanged: `load_texas()` / `load_package(dir)` return `DistrictData`
(`docs/DHW_V2_DATA_FORMAT.md`).

## 9. Smoke year v2.1 (Texas houses 0–1, dt = 5 min)

PROVISIONAL throughout:
* Texas weather and draws, two houses, one year;
* in-sample: β, R and α were chosen with the theory on the same Texas year;
* every operating hour is source-extrapolated;
* the identified model is fitted on January of the same year.

Wall time per year, with 8 parallel processes:
* 24–72 s unfiltered (full-year runs; `wall s` column);
* 117–457 s with the filter (the reserve rollout plus the T4 trigger rolled every owed hour). The three
  identified-model reruns ran on 3 workers, not 8.
* power-limited rollouts never refill, so the 4-week stress runs take about 7 min with H = 24.

The whole 45-run matrix took about 28 minutes on 8 workers.

Smoke runs v2.1, tx_travis_8b houses 0-1 (sums over houses; PROVISIONAL: Texas data, 2 houses, one year, in-sample, every hour source-extrapolated). 'viol h' = hours whose intra-hour top-node minimum fell below 45 C; 'inherited' = of those, hours that STARTED below 45 C; 'RF fail' = recursive-feasibility failures (hour started >= 45 C but could not be certified although the previous hour was certified and within its quantile); 'unexpl.' = violations with no draw exceedance in hour t or in the t-1 plan for t, not announced at t-1, not declared infeasible, not inherited (T3(iv)); 'announced' = violations announced one hour earlier (T2); 'deliv<45 h' = hours in which water delivered at the tap was below 45 C; 'weeks b20/mean/all-nodes' = calendar weeks with a 1-h hold under the bottom-20 % mean (primary), whole-volume mean, all-nodes rules; 'bound P' = one-sided T3 bound alpha + (range+eta)/(eta T) on P(d>m).

| run | hours | houses | wall s | elec kWh | viol h | inherited | RF fail | unexpl. | announced | deliv<45 h | min<45 h | unmet kWh | weeks b20 | weeks mean | weeks all-nodes | max gap h | dis-supply h | kWh@extrap sup | SH K h<Tmin | interv. rate | P(d>m) | bound P | brine 24h drop K | h over cap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| null__oracle | 8759 | 2 | 425 | 1860 | 622 | 304 | 0 | 0 | 0 | 346 | 114.7 | 50.6 | 104/104 | 104 | 100 | 332 | 220 | 368 | 7 | 0.314 | 0.0525 | 0.0699 | 2.5 | - |
| null__oracle_N2 | 8759 | 2 | 256 | 1889 | 862 | 426 | 0 | 0 | 0 | 527 | 174.7 | 53.1 | 104/104 | 104 | 104 | 333 | 150 | 320 | 2 | 0.466 | 0.0525 | 0.0699 | 2.4 | - |
| null__oracle_N20 | 8759 | 2 | 344 | 1842 | 580 | 284 | 0 | 0 | 0 | 312 | 108.8 | 64.6 | 104/104 | 104 | 93 | 332 | 233 | 377 | 12 | 0.289 | 0.0525 | 0.0699 | 2.6 | - |
| null__oracle_a001 | 8759 | 2 | 457 | 1902 | 30 | 15 | 0 | 0 | 0 | 15 | 4.2 | 2.2 | 104/104 | 104 | 103 | 331 | 190 | 385 | 7 | 0.642 | 0.0019 | 0.0031 | 2.5 | - |
| null__oracle_a01 | 8759 | 2 | 414 | 1864 | 232 | 114 | 0 | 0 | 0 | 124 | 44.5 | 30.1 | 104/104 | 104 | 99 | 332 | 197 | 389 | 6 | 0.543 | 0.0142 | 0.0299 | 2.5 | - |
| null__oracle_onehour | 8759 | 2 | 287 | 1861 | 675 | 331 | 47 | 0 | 0 | 379 | 125.9 | 51.7 | 104/104 | 104 | 101 | 331 | 217 | 363 | 7 | 0.279 | 0.0525 | 0.0699 | 2.5 | - |
| null__oracle_trig_jit | 8759 | 2 | 202 | 1849 | 636 | 311 | 0 | 0 | 0 | 360 | 115.6 | 54.4 | 104/104 | 104 | 101 | 333 | 180 | 340 | 1 | 0.318 | 0.0525 | 0.0699 | 2.5 | - |
| null__oracle_trig_quantile | 8759 | 2 | 199 | 1850 | 642 | 314 | 0 | 0 | 0 | 361 | 116.2 | 52.7 | 104/104 | 104 | 101 | 333 | 181 | 341 | 1 | 0.317 | 0.0525 | 0.0699 | 2.5 | - |
| null__oracle_trig_recourse | 8759 | 2 | 187 | 1849 | 642 | 314 | 0 | 0 | 0 | 360 | 117.4 | 54.3 | 104/104 | 104 | 100 | 333 | 174 | 339 | 1 | 0.316 | 0.0525 | 0.0699 | 2.5 | - |
| null__oracle_trig_robust_R0 | 8759 | 2 | 299 | 1847 | 657 | 321 | 0 | 0 | 0 | 370 | 118.8 | 50.7 | 104/104 | 104 | 101 | 335 | 162 | 325 | 2 | 0.317 | 0.0525 | 0.0699 | 2.5 | - |
| null__oracle_trig_robust_b010 | 8759 | 2 | 239 | 1852 | 644 | 315 | 0 | 0 | 0 | 353 | 116.0 | 51.0 | 104/104 | 104 | 100 | 333 | 188 | 345 | 2 | 0.315 | 0.0525 | 0.0699 | 2.5 | - |
| null__stress_lookahead1_4w | 672 | 2 | 13 | 355 | 156 | 78 | 0 | 0 | 9 | 77 | 35.8 | 77.2 | 8/8 | 8 | 7 | 169 | 29 | 38 | 532 | 0.542 | 0.1042 | 0.5325 | 1.7 | 0/0 |
| null__stress_onehour_4w | 672 | 2 | 7 | 352 | 166 | 84 | 3 | 0 | 8 | 86 | 40.0 | 88.1 | 8/8 | 8 | 7 | 169 | 29 | 38 | 529 | 0.525 | 0.1042 | 0.5325 | 1.7 | 0/0 |
| null__stress_reserve24_4w | 672 | 2 | 418 | 355 | 156 | 78 | 0 | 0 | 9 | 77 | 35.8 | 77.2 | 8/8 | 8 | 7 | 169 | 29 | 39 | 532 | 0.542 | 0.1042 | 0.5325 | 1.7 | 0/0 |
| reactive__identified | 8759 | 2 | 157 | 1944 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 310 | 212 | 263 | 1 | 0.003 | 0.0525 | 0.0699 | 2.7 | - |
| reactive__identified_cap3 | 8759 | 2 | 157 | 1944 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 310 | 212 | 263 | 1 | 0.003 | 0.0525 | 0.0699 | 2.7 | 0/0 |
| reactive__off | 8759 | 2 | 35 | 1944 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 168 | 208 | 262 | 8 | - | 0.0525 | - | 2.8 | - |
| reactive__off_N2 | 8759 | 2 | 48 | 1925 | 218 | 106 | - | - | - | 135 | 47.9 | 20.1 | 104/104 | 104 | 104 | 168 | 208 | 279 | 7 | - | 0.0525 | - | 2.4 | - |
| reactive__off_N20 | 8759 | 2 | 50 | 1947 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 168 | 208 | 259 | 9 | - | 0.0525 | - | 2.8 | - |
| reactive__off_dt1_2w | 336 | 2 | 4 | 152 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 4/4 | 4 | 4 | 166 | 8 | 6 | 1 | - | 0.1101 | - | 2.0 | - |
| reactive__off_dt5_2w | 336 | 2 | 2 | 152 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 4/4 | 4 | 4 | 166 | 8 | 6 | 1 | - | 0.1101 | - | 2.0 | - |
| reactive__off_fullspeed | 8759 | 2 | 35 | 2242 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 168 | 208 | 278 | 8 | - | 0.0525 | - | 2.7 | - |
| reactive__off_v1plant | 8759 | 2 | 24 | 2225 | 218 | 106 | - | - | - | 135 | 46.8 | 20.1 | 104/104 | 104 | 104 | 168 | 208 | 2225 | 14 | - | 0.0525 | - | 5.8 | - |
| reactive__oracle | 8759 | 2 | 251 | 1945 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 310 | 212 | 263 | 4 | 0.003 | 0.0525 | 0.0699 | 2.7 | - |
| reactive__oracle_N2 | 8759 | 2 | 280 | 1926 | 212 | 103 | 0 | 0 | 0 | 134 | 45.3 | 16.6 | 104/104 | 104 | 104 | 310 | 212 | 280 | 2 | 0.020 | 0.0525 | 0.0699 | 2.4 | - |
| reactive__oracle_a001 | 8759 | 2 | 284 | 1945 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 310 | 212 | 263 | 4 | 0.003 | 0.0019 | 0.0031 | 2.7 | - |
| reactive__oracle_a01 | 8759 | 2 | 295 | 1945 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 310 | 212 | 263 | 4 | 0.003 | 0.0142 | 0.0299 | 2.7 | - |
| reactive__oracle_cap3 | 8759 | 2 | 275 | 1945 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 310 | 212 | 263 | 4 | 0.003 | 0.0525 | 0.0699 | 2.7 | 0/0 |
| reactive__oracle_trig_timer | 8759 | 2 | 117 | 1943 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0 | 0.0 | 104/104 | 104 | 104 | 168 | 208 | 262 | 0 | 0.001 | 0.0525 | 0.0699 | 2.7 | - |
| thermia_hourly__identified | 8759 | 2 | 206 | 1925 | 10 | 5 | 0 | 0 | 0 | 6 | 1.7 | 1.0 | 104/104 | 104 | 95 | 324 | 308 | 475 | 1 | 0.187 | 0.0525 | 0.0699 | 2.5 | - |
| thermia_hourly__off | 8759 | 2 | 72 | 1908 | 18 | 9 | - | - | - | 10 | 2.5 | 1.8 | 73/104 | 96 | 67 | 1239 | 247 | 436 | 0 | - | 0.0525 | - | 2.5 | - |
| thermia_hourly__oracle | 8759 | 2 | 335 | 1921 | 29 | 14 | 0 | 0 | 0 | 12 | 3.5 | 2.3 | 104/104 | 104 | 95 | 301 | 301 | 482 | 2 | 0.006 | 0.0525 | 0.0699 | 2.5 | - |
| thermia_hourly__oracle_N2 | 8759 | 2 | 325 | 1920 | 451 | 223 | 0 | 0 | 0 | 271 | 95.9 | 33.7 | 104/104 | 104 | 104 | 330 | 255 | 434 | 2 | 0.159 | 0.0525 | 0.0699 | 2.4 | - |
| thermia_hourly__oracle_trig_thermia | 8759 | 2 | 138 | 1909 | 28 | 14 | 0 | 0 | 0 | 14 | 4.1 | 3.3 | 78/104 | 97 | 69 | 710 | 221 | 440 | 0 | 0.002 | 0.0525 | 0.0699 | 2.4 | - |
| thermia_native__off | 8759 | 2 | 49 | 1904 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 48/104 | 93 | 31 | 1436 | 190 | 408 | 0 | - | 0.0525 | - | 2.7 | - |
| thermia_native__off_N2 | 8759 | 2 | 51 | 1909 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 50/104 | 81 | 50 | 1219 | 177 | 396 | 0 | - | 0.0525 | - | 2.4 | - |
| thermia_native__off_N20 | 8759 | 2 | 41 | 1903 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 47/104 | 89 | 26 | 1425 | 185 | 415 | 0 | - | 0.0525 | - | 2.5 | - |
| thermia_native__off_fullspeed | 8759 | 2 | 48 | 2183 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 41/104 | 88 | 30 | 1400 | 135 | 468 | 0 | - | 0.0525 | - | 2.4 | - |
| thermia_native__off_sensor050 | 8759 | 2 | 29 | 1914 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 55/104 | 96 | 36 | 1056 | 177 | 400 | 0 | - | 0.0525 | - | 2.4 | - |
| thermia_native__off_sensor075 | 8759 | 2 | 42 | 1891 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 45/104 | 97 | 35 | 1060 | 186 | 415 | 0 | - | 0.0525 | - | 2.4 | - |
| thermia_native__off_start45 | 8759 | 2 | 35 | 1899 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 52/104 | 92 | 35 | 1093 | 191 | 420 | 0 | - | 0.0525 | - | 2.5 | - |
| thermia_native__off_start46 | 8759 | 2 | 45 | 1901 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 57/104 | 96 | 44 | 877 | 186 | 411 | 0 | - | 0.0525 | - | 2.4 | - |
| thermia_native__off_start48 | 8759 | 2 | 30 | 1909 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 61/104 | 94 | 50 | 1229 | 185 | 409 | 0 | - | 0.0525 | - | 2.4 | - |
| thermia_native__off_stop52 | 8759 | 2 | 31 | 1902 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 49/104 | 90 | 38 | 1056 | 184 | 411 | 0 | - | 0.0525 | - | 2.7 | - |
| thermia_native__off_stop55 | 8759 | 2 | 29 | 1916 | 0 | 0 | - | - | - | 0 | 0.0 | 0.0 | 51/104 | 92 | 37 | 1043 | 190 | 431 | 1 | - | 0.0525 | - | 2.6 | - |

Reading (all provisional):

1. **Core, N = 10.**
   * Reactive, with and without the filter: 0 comfort-violation hours, 0 hours of delivered water
     below 45 C, 104/104 weeks with the primary hold. The filter intervenes in 0.3 % of hours.
   * Thermia logic native: 0 comfort violations with the 47 C start, but only 48/104 weeks with the
     primary hold (93 under the whole-volume mean, 31 under all-nodes). Its programs (50 per house)
     all ran and completed, then cooled or were drawn below 60 C within the hour.
   * Thermia hourly with the filter: 104/104 weeks (robust trigger), 29 violation hours (14 inherited),
     12 h of delivered water below 45 C, intervention rate 0.006.
2. **Recursive feasibility (item 3).** The reserve set removes the failures it is meant to remove.
   * Null proposer, filter only: 0 recursive-feasibility failures with the 24-h reserve against 47
     with the one-hour certificate.
   * Power-limited stress (draws x2, 0.8 kW cap, 4 weeks): 0 with H = 24 or with the one-hour terminal
     condition, against 3 for the plain one-hour certificate.
   * Against the lead's expectation, inherited violations do NOT drop to about 0: about half of all
     violations are inherited in every filtered run.
   * By the certificate (oracle model, certified intra-hour shape), a certified hour whose draw
     stays within its quantile ends with the top at or above 45 C. So each inherited hour must follow
     an hour that exceeded its certified draw, or one that was declared infeasible. The run logs
     classify the violating hours themselves (0 unexplained); the predecessor of each inherited hour
     was not tabulated separately.
   * The reserve set prevents getting stuck under within-quantile draws; it cannot undo an exceedance.
3. **Against us: the filter is weaker than a simple rule.**
   * The null proposer with the oracle filter at α = 0.05 has 622 violation hours (346 h of delivered
     water below 45 C, 50.6 kWh unmet). Reactive alone has 0.
   * The certificate charges exactly enough for `d_hat + m`. As theory T3 says, the margin bites only
     below the base exceedance rate: 232 h at α = 0.01 and 30 h at α = 0.001 (η = 0.5).
   * At α = 0.001 the cost is +42 kWh/yr and an intervention rate of 0.64. An agent that leaves DHW
     to the filter will sit near this policy, so the RL track needs α <= 0.001 or the correction
     penalty.
   * P(d > m) stayed inside the one-sided bound at every α (0.0525 <= 0.0699, 0.0142 <= 0.0299,
     0.0019 <= 0.0031).
4. **Hold definition (item a).** With the primary bottom-20 % mean:
   * Thermia native completes 48 / 47 / 50 weeks of 104 at N = 10 / 20 / 2.
   * Null + filter completes 104 / 104 / 104.
   * The all-nodes rule is mesh-dependent: Thermia 31 / 26 / 50, null 100 / 93 / 104.
   * The whole-volume mean: Thermia 93 / 89 / 81.
5. **Triggers (null proposer, so the filter does all disinfection).** Every trigger completed 104/104
   under the primary hold on Texas, so this year does not discriminate between them.
   * Disinfection-supply hours: robust β = 0.02, R = 2: 220; β = 0.1: 188; R = 0: 162; jit 180;
     quantile 181; recourse 174.
   * The proposer's own timer (Reactive, trigger none) completes 104/104.
   * Thermia's own program under the filter (trigger none) completes 78/104; the robust trigger
     brings it to 104/104.
6. **Thermia-logic assumptions (native).**
   * Comfort violations are 0 in every variant.
   * Primary-hold completion ranges over 45–61/104 across start 45–48, stop 52–55 and sensor
     0.5–0.75 H; it is 41 at full speed.
   * "Thermia's program, as modelled, meets the 1-h bottom-20 % hold in about half the weeks"
     survives the sensitivity. The 45 C start artefact is gone: 0 violations.
7. **N convergence.**
   * Reactive off: 1944 vs 1947 kWh at N = 10 vs 20, both 0 violations.
   * Null + filter: 622 (N = 10) vs 580 (N = 20) violation hours, against 862 at N = 2.
   * N = 2 adds 218 violation hours to Reactive (0 at N = 10).
8. **Speed law.** The calibrated speed lowers electricity: Reactive 1944 vs 2242 kWh at full speed,
   Thermia 1904 vs 2183. The lower compressor speed gives a better COP on the map.
9. **Caps and the identified model.**
   * 3 kW house / 5 kW district caps: 0 hours over for both the oracle and the identified model.
   * The identified electric bound underestimates the plant: its maximum is 1.77 kW against an actual
     maximum of 2.96 kWh/h (the maxima need not fall in the same hour). The cap is therefore not
     certified under the identified model.
   * The identified model otherwise behaves like the oracle: Reactive identical, Thermia hourly
     10 vs 29 violation hours.

## 10. Extrapolated or assumed (summary)

**Extrapolated.**
* The map above 55 C supply: all disinfection energy, 13–25 % of electricity in these runs.
* Source temperatures outside 0–10 C: all of Texas.

**Assumed.**
* Thermia start/stop/sensor/legionella supply (swept).
* The heat-up end state used for the speed calibration.
* The same speed at 65 C.
* The UA conventions.
* Borehole geometry and ground properties.
* The 3 K brine band (checked against the realised 2.8 K).
* The placeholder Swedish archetype.
* The Texas CST timeline.
* GHI = direct + diffuse.
* The V40 tapping rate.

**Definitions chosen, not results.**
* The hold rule (bottom-20 % mean, after G12/AS2 and HSG274).
* The 45 C top-node comfort floor.
* The weekly calendar window.
