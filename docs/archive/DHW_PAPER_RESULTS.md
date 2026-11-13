# Hot-water comfort and weekly disinfection on a stratified two-node store

*Results for the hot-water paper with Manal Rahal (Thermia). Branch `dhw-paper`, cut from
`dhw-band` at 4dea6fe; commits aeb6dc8 (tank physics), 6c1d0cd (triggers and runner) and
the commit that adds this document. Eight Travis County houses (CityLearn `tx_travis_8b`,
8,759 hourly steps), one heat pump per house scaled to the Thermia Calibra Eco 8.*

*Every table between `<!-- table:… -->` markers is rewritten from the run records in
`results/dhw_paper/` by `experiments/dhw_paper/make_tables.py`; no table number is typed by
hand. 200 year-long runs: 152 in the main matrix, 18 filtered no-disinfection references,
30 on power-limited units.*

## 0. Summary

1. **The committed two-node plant could not disinfect, and its fix is physics, not an
   algorithm.** It charged only the top node, so the bottom node was never heated
   (conduction κ = 0 in every run here). Under a plain weekly timer, 0 of 416 house-weeks
   met the hold definition (every node ≥ 60 °C through one hour) and the bottom node's
   weekly peak averaged 11.3 °C. With external stratified charging (the heat-pump loop
   takes water from the bottom and returns it at the level its buoyancy puts it), the same
   timer and controller met the hold in 405 of 416 weeks and unmet hot water fell from
   58.64 to 0.00 kWh/yr. A store that is actually full costs more to keep: 4,892 instead of
   3,794 kWh/yr of electricity (Table 0).
2. **Completion depends on what schedules the cycle.** Under the time-of-use tariff, the
   price-aware controller requests the disinfection itself in a cheap hour after 120 h of
   each calendar week, and completes all 416 windows with every trigger. The trigger never
   acts, so its three trigger runs are identical. When nothing else schedules the cycle (the
   reactive controller; the price-aware controller under a flat price, where no hour is
   cheap), the trigger decides. The original just-in-time trigger misses 33 windows (36 for
   price-aware, flat), the committed recourse plan 24 (24), the quantile recourse plan 6 (6),
   and the conventional weekly timer, due at hour 2 and held until completion, none.
3. **The filter's certificate held in all 92 filtered runs given the hour's draw, and it is
   narrower than "comfort".** No step ended outside the band without the filter having
   reported it as infeasible beforehand. The model reproduced the plant to 0.0 kWh in every
   run. On the Calibra Eco 8 no step was infeasible, so the top node never ended an hour
   below 45 °C. Without the filter it did so in 165–185 building-hours/yr under TOU, and in
   1,172–1,288 for the price-aware controller under the flat price. On power-limited units,
   every violation was a step reported as infeasible (Table 4). The certificate covers the
   end of the hour only. Between the draw and the charge, the top node was below 45 °C in
   about as many hours with the filter as without it (687–865 against 667–827
   building-hours, TOU). Given a causal forecast of the draw instead of the draw, there is no
   certificate, and 150–177 building-hours ended below 45 °C (TOU).
4. **The weather forecast does not matter for this unit.** The brine temperature follows a
   24-hour mean of the outdoor temperature. In 16 of 24 matched configuration pairs, a noisy
   forecast instead of persistence changed nothing. In the other 8 (price-aware, flat price)
   it changed completion by at most 2 windows and electricity by at most 3.7 kWh/yr, because
   forecast noise occasionally makes an hour look cheap. That is noise, not information.
5. **On this unit, disinfection costs little energy.** Take the year's electricity increase
   over the matching no-disinfection run and divide it by the completed windows. That gives
   0.20–0.29 kWh per completed disinfection with the reactive controller and 0.26 kWh with
   the price-aware one under TOU. Most of the extra heat is drawn off afterwards as hot
   water. What remains is the COP penalty of the 65 °C supply and the extra standing loss of
   a hotter store. The reliable triggers cost slightly more per cycle (timer 0.29, quantile
   recourse 0.26) than the unreliable ones (just-in-time 0.20, recourse 0.21).
6. **What the paper must not claim.**
   1. The quantile recourse plan was introduced after the hold definition broke the
      committed recourse plan, and it was designed on this same year. Its 6 missed windows
      are in-sample.
   2. The price-aware controller's lower cost under TOU (1,140.38 against 1,350.60 USD/yr
      for the reactive timer) comes from buying heat in cheap hours. It is a property of the
      controller, not of any disinfection trigger.
   3. Table 0 measures a correction of the physics, not an improvement by a method.
   4. Everything is one deterministic year of eight houses. Differences are exact
      differences on that year, not estimates with sampling error, and nothing here is
      held out.

## 1. Tank physics: what was wrong, what changed

### 1.1 The committed store

`stems/dhw_plant.py` as committed on `dhw-band`: two nodes, a draw taken from the top node
first, and a charge that went into the top node only. The bottom node gained heat only by
conduction between the nodes, which is zero in every run here. Two further defects were
open:

1. `DHWPlant.max_charge_thermal_kwh` read the sink temperature *before* the step's draw,
   while the step charged at the sink *after* it. It now takes the step's draw and reads
   the sink the step charges at. `charge_limit_kwh` equals the heat the step accepts (tests
   pin both).
2. The standing loss was applied after the charge, which capped every post-step node at
   soc 1 − loss (59.9 °C for a 60 °C store). A conventional single-sensor stop rule ("stop
   when the top reads 60 °C") therefore never fired. In variant A below, the weekly cycle
   never ends and the top is held at 60 °C all year (70,056 of 70,072 building-hours above
   55 °C). `loss_first=True`, the default for stratified tanks, applies the loss first, and
   `t_max_c` lets a store hold more than its rated temperature.

The committed top-node path is unchanged by default. The 126 legacy DHW tests pass
unchanged, except the three that called the method with its old signature.

### 1.2 The stratified store (`TankSpec(charging="stratified")`, commit aeb6dc8)

* **Order within an hour:** standing loss, then the draw, then the charge.
* **Draw (plug flow):** hot water leaves at the top and mains water at 10 °C enters at the
  bottom. Water below `t_use_c` = 40 °C is not delivered and counts as unmet.
* **Charge (external, stratified):** the heat pump takes water from the bottom of the
  store and heats it in one pass to the supply temperature (55 °C, or 65 °C during
  disinfection). It returns the water at the level its buoyancy puts it: on top if it is
  not cooler than the top node, otherwise into the bottom node. The charge front moves
  down, and a full charge brings every node to the supply temperature. The unit stops when
  the store is full, so nothing is spilled.
* **Discretisation:** within a step the column moves as a single plug and the nodes are
  re-mixed at the end of the step, in closed form (`stems/dhw_physics.py`). This is less
  diffusive than a stirred-tank update. A draw of up to half the stored heat leaves the top
  node at its temperature: a 55 °C store after a draw of 44 % of its heat has its top at
  55.0 °C and its bottom at 15.4 °C.
* **Geometry:** two nodes of equal volume, no conduction between them. The store may reach
  65 °C (`t_max_c`); soc 1 corresponds to 60 °C over a 10 °C cold-water temperature.
* **Twin and filter:** `StratifiedStoreModel` (`stems/dhw_stratified.py`) is the filter's
  model. It is built from the same functions as the plant, so it is an exact twin.
  `tests/test_dhw_barrier.py::test_the_model_reproduces_the_plant_bit_for_bit` (12
  parametrisations, `np.array_equal` on energy, unmet and spill) passes, and in all 200 runs
  the model-vs-plant error was 0.0 kWh. Exactness against a twin is not a check of physics,
  which is why the physics has its own tests.
* **Tests:** `tests/test_dhw_stratified.py`, 30 tests: energy conservation, monotone charge
  front, full charge, buoyancy, usable water, both defects, the bit-for-bit twin, monotone
  successors, the exact inverse, the hold, the hplib rows and the triggers.

### 1.3 Before and after on the year

Variants A–E share the draws, the heat pump and the controller. The controller keeps the
store full and runs a weekly timer at hour 2 of each calendar week. That timer heats to the
disinfection target and stops when the top node reads ≥ 60 °C at the end of a step (the
single-sensor rule of a conventional anti-legionella program). It is a test harness, not
a proposal: it restarts the unit almost every hour, hence the starts.
`experiments/dhw_paper/before_after.py`.

* A, committed: top-node charging, top-first draw, loss after the charge, store capped at
  60 °C, every stored kWh counted as usable.
* B: A with the loss first and a 65 °C store (defect 2 fixed).
* C: B with plug-flow draws and the 40 °C usable-water limit.
* D: external stratified charging, with C's draw and loss order.
* E: D with the committed usable-water definition (every kWh above 10 °C usable).

<!-- table:0 -->
| variant | unmet (kWh) | electricity (kWh) | cost (USD) | starts | top < 45 °C (building-h) | top > 55 °C (building-h) | weeks, committed convention | weeks, hold definition | mean weekly bottom peak (°C) |
|---|---|---|---|---|---|---|---|---|---|
| A committed | 58.64 | 3793.6 | 1094.77 | 68912 | 0 | 70056 | 416 | 0 | 11.3 |
| B +loss first, 65 degC store | 109.26 | 3374.5 | 971.65 | 66007 | 0 | 3378 | 416 | 0 | 11.2 |
| C +plug-flow draw, 40 degC usable | 117.16 | 3368.9 | 970.14 | 65976 | 0 | 3415 | 416 | 0 | 10.9 |
| D stratified charging | 0.00 | 4892.4 | 1385.17 | 65581 | 0 | 9974 | 416 | 405 | 65.0 |
| E D with all kWh usable | 0.00 | 4892.4 | 1385.17 | 65581 | 0 | 9974 | 416 | 405 | 65.0 |
<!-- /table:0 -->

Energy balance of A against D:

<!-- table:0_balance -->
| variant | heat delivered (kWh) | hot water served (kWh) | standing loss (kWh) | change in stored heat (kWh) | seasonal COP |
|---|---|---|---|---|---|
| A committed | 16774.5 | 14907.1 | 1896.3 | -28.9 | 4.42 |
| D stratified charging | 18369.8 | 14965.7 | 3403.8 | +0.3 | 3.75 |

Electricity difference D − A: +1098.8 kWh = 360.8 kWh for the extra heat at A's seasonal COP + 738.0 kWh because D's COP is evaluated at the supply temperature it produces (55/65 °C) and A's at the top node's pre-charge temperature.
<!-- /table:0_balance -->

Reading the table:

* **Only stratified charging heats the bottom.** A, B and C never heat the bottom node
  (mean weekly peak about 11 °C), so no week meets the hold. Every week still "completes"
  under the committed convention, which looked only at the charged node.
* **D holds both nodes at 55 °C instead of one.** It serves all the hot water (0.00 kWh
  unmet). It loses about 80 % more heat at standstill, and its COP is evaluated at the
  temperature it produces. Its weekly bottom peak is 65 °C.
* **Why D misses 11 weeks.** The single-sensor stop rule ends the cycle as soon as the top
  reads 60 °C. That is before the hold is complete whenever the bottom is not yet at 60 °C,
  or when the next hour's draw cools it, and the rule does not retry.
* **E equals D.** With stratified charging and this controller, no water below 40 °C is
  ever drawn.
* **A loses less hot water than B for a controller reason.** Because A's cycle never ends,
  A's top node stays at 60 °C all year. Once B's cycle ends, the top returns to 55 °C.

![Two weeks in house 0: the committed plant never heats the bottom node; stratified
charging does (dashed line 60 °C, shaded band 45–55 °C).](../results/dhw_paper/fig_trace.png)

![Before/after on the year, per variant.](../results/dhw_paper/fig_before_after.png)

## 2. Definitions used by the controllers, the filter and the metrics

Each quantity has one definition, in `stems/dhw_stratified.py`. The controllers, the filter
(`DisinfectionBarrier`) and the metric code (`experiments/dhw_paper/run_matrix.py`) all
import it.

| quantity | definition | where |
|---|---|---|
| comfort | top node between 45 and 55 °C at the end of each hour. A violation is a building-hour outside the band; its size is counted in K·h below 45 °C and above 55 °C separately. Physical, whatever the phase: hours spent above 55 °C *because* of a disinfection count | `COMFORT_C` |
| disinfection (hold) | **every node ≥ 60 °C throughout one hour**: at the start of the hour, after the hour's standing loss and draw (the coldest point of the hour, since the charge only adds heat), and therefore at its end | `hold_met`, `DISINFECT_C` |
| weekly window | the fixed calendar week: 168 h windows from hour 0 of the year, 52 per house, 416 in total. The last 23 h of the year are not counted | `DisinfectionBarrier.observe` |
| completed window | a window with at least one hold-met hour. Headline metric: missed windows out of 416 | same |
| gap (secondary) | hours between consecutive hold-met hours, per house, reported as the longest gap and as the number of gaps over 168 h (the stricter rolling reading of "weekly") | `run_matrix.py` |
| unmet hot water | draw energy that cannot be delivered at ≥ 40 °C | `TankSpec.t_use_c` |
| supply temperature | 65 °C while the week's disinfection is owed and the phase is D, 55 °C otherwise; the same for every controller | `DisinfectionBarrier.supply` |
| filter bands | phase N: top 45–55 °C, bottom ≤ 55 °C. Phase D: top 45–65 °C, bottom ≤ 65 °C. Strict guard: the phase changes only from a state inside both phases' bands | `PhaseBands.from_temperatures` |
| kWh per completed disinfection | (run's electricity − electricity of the matching no-disinfection run, same plant, controller, price, weather and filter setting) / completed windows. A net figure: heat put in for the cycle and drawn off later is not counted | `make_tables.py` |

### 2.1 What the standards and guidance require (read, not recalled)

| source (read) | what it says, in brief | used for |
|---|---|---|
| HSE, *HSG274 Part 2* (copy retrieved from cambridgesafety.co.uk) | the base of a calorifier is usually much cooler than its top, so "arrangements should therefore be made to heat the whole water content each day"; the whole content, including the base, at 60 °C for one hour; a calorifier out of use for over a week brought to 60 °C for 1 h before use | which nodes (all, including the bottom) and for how long (1 h) |
| NZ MBIE, *Acceptable Solution G12/AS2* §3.5.1(c) (G12 3rd ed., amendment 7) | solar water heaters may be controlled so that "all of the stored water is heated to 60°C or higher, once a week", for not less than 1 h, sensed in the bottom 20 % of the volume | the weekly window; all stored water; 1 h; checked at the bottom |
| HSE web guidance (hse.gov.uk/healthservices/legionella.htm) | where temperature is the control: store hot water above 60 °C, distribute it above 50 °C | context |
| US CDC, *Controlling Legionella in Potable Water Systems* (toolkit page) | store hot water above 60 °C; circulating hot water not below 49 °C | context |
| ANSI/ASHRAE Standard 188-2018 (scope, ashrae.org preview); Georgia DCA Q&A on 188 | the standard does not cover single-family residential buildings and gives no target temperatures | **no rule for these houses** |
| Boverket, BBR (PBL kunskapsbanken, "Legionella", 6 §) and "Om legionella" | binding: installations designed so that ≥ 50 °C can be reached at the tap, circulation ≥ 50 °C. Guidance: 60 °C recommended in water heaters; growth possible where the whole volume is not heated above 50 °C | Swedish context (Thermia) |
| NASEM, *Management of Legionella in Water Systems* (2019, NCBI Bookshelf NBK555108) | electric water heaters are thermally stratified, cooler at the bottom; a bottom temperature below 40 °C is reported as a factor for Legionella positivity | why the bottom node decides the hold |

**Not read at the primary source, so no rule is taken from them:**

* **DVGW W 551** (paywalled). Secondary sources (biostream.de, schell.eu) report reduced
  requirements for small systems in one- and two-family houses, and that a 2023 draft
  advises against routine thermal disinfection.
* **WHO, *Legionella and the prevention of legionellosis* (2007).** A letter in PMC2094925
  reports it as asking for the entire tank to reach 60 °C at least daily.
* **CIBSE TM13.** Not retrieved.

**What follows for these eight single-family houses.** None of the sources read *mandates*
a weekly cycle in a single-family house. The most specific weekly rule for a stratified
store is G12/AS2 §3.5.1(c): all stored water at ≥ 60 °C for ≥ 1 h once a week, checked at
the bottom. HSG274's daily rule has the same shape: the whole content including the base,
60 °C, one hour. The paper's definition is therefore **every node at ≥ 60 °C through one
full hour, once per calendar week**. On a two-node model "every node" means the bottom node
in practice, which is where both sources put the check. The daily HSG274 variant was not
run; it is the same machinery with a 24 h window.

## 3. Heat pump and source

`stems/heatpump_hplib.py`, replacing the unsourced `ThermiaSpec` fixture.

* **Shape of COP and capacity:** hplib 1.9's generic *regulated* brine/water regression
  (group 2; Tjaden, Hoops, Roesken, Zenodo doi:10.5281/zenodo.5521597). It is a
  least-squares fit to Heat Pump Keymark data, with COP and electric power linear in
  (T_brine, T_supply) and T_amb = −7 °C as hplib evaluates brine units. Coefficients are
  copied from `hplib_database.csv`, rows `Generic`.
* **Scale:** the Thermia Calibra Eco 8 (datasheet range 2–8 kW). P_el,ref = 2.237 kW, so
  that the thermal output at B0/W35 is 8.0 kW, *assuming* the nameplate refers to B0/W35.
  That gives 6.20 kW at B0/W55 and 5.39 kW at B5/W65, with COP 4.67 (B0/W35), 2.70
  (B0/W55), 3.20 (B5/W55), 3.69 (B10/W55) and 2.21 (B5/W65).
* **Limits:** modulation floor 25 % (2 of 8 kW); no output above 65 °C supply (datasheet
  maximum); no resistive element for hot water. Starts are counted, not priced or
  constrained.
* **Source:** brine temperature from hplib's correlation (WP Monitor, Fraunhofer ISE 2014),
  T_brine = −0.0003 t³ + 0.0086 t² + 0.3047 t + 5.0647, with t the trailing 24-h mean
  outdoor temperature, clipped to [−15, 25] °C.
* **Seasonal COP** in the matrix: 3.72–3.81.

## 4. Assumptions

| item | value | status |
|---|---|---|
| houses, horizon | 8 houses, 8,759 h | CityLearn v2.6.0b1 `tx_travis_county_neighborhood`, schema `tx_travis_8b` |
| hot-water draw | schema `dhw_demand`, kWh/h per house; mean daily 1.48–7.77 kWh, 14,965.7 kWh/yr in total | dataset |
| store capacity | 6.56–11.20 kWh per house (schema `dhw_storage` capacity), about 113–193 L over our 50 K span | dataset; litres follow from our convention |
| standing loss | coefficient 0.0024–0.0083 per hour × stored energy (CityLearn convention: proportional to stored energy, not to T − T_room) | dataset; convention flagged |
| cold water | 10 °C | assumption, unsourced |
| rated / maximum / usable | soc 1 = 60 °C; store up to 65 °C; water usable at ≥ 40 °C | 40 °C matches the Thermia datasheet's "40 °C hot water" volume basis |
| initial state | both nodes at 55 °C | chosen |
| nodes | 2, equal volume, no conduction | chosen; a 3-node sensitivity was not run |
| supply | 55 °C normally, 65 °C in phase D while owed | chosen; question 5 for Manal |
| tariff (TOU) | schema `electricity_pricing`: 0.21–0.22 USD/kWh outside 17–21 h, 0.40–0.54 at 17–21 h | dataset |
| tariff (flat) | annual time-average of the TOU series, 0.27314 USD/kWh | chosen |
| weather | schema outdoor dry-bulb temperature, −7.8 to 42.8 °C (mean 20.7) | dataset |
| weather forecast "off" | persistence of the current brine temperature over the 24 h horizon | chosen; question 7 for Manal |
| weather forecast "on" | actual outdoor temperature + i.i.d. N(0, (0.0324 K × lead h)²) noise, mapped through the brine model; 5 seeds. 0.0324 K/h fits the RMSE of CityLearn's own predicted outdoor temperature (0.175, 0.379, 0.787 K at 6, 12, 24 h) | chosen; the only stochastic element |
| draw forecasts | causal hour-of-day mean (committed recourse); causal hour-of-day 0.9-quantile (quantile recourse). The just-in-time reserve and the recourse safety stock are each house's whole-year 0.9-quantile hourly draw (not causal) | chosen |

## 5. The benchmark matrix

Manal's axes, crossed: controller × disinfection trigger × filter × price × weather
forecast, plus the filter given a forecast of the draw instead of the draw.
`experiments/dhw_paper/run_matrix.py --grid main | base_filter | power_limited`.

**Controllers** (`stems/dhw_stratified.py`):

* **Reactive:** a thermostat with hysteresis. On when the bottom sensor is below 40 °C or
  the top below 47 °C; off when the bottom reaches 50 °C. It never asks for disinfection.
* **Price-aware:** the effective heat price is c = price / COP(source, 55 °C) over the next
  24 h, the source path being the forecast (weather on) or persistence (off). An hour is
  *cheap* when c is at least 5 % below the 24-h mean. Cheap hours use thresholds that fill
  the store (on below 50 °C bottom or 47 °C top, off at 53 °C bottom); other hours coast
  (30, 47, 40 °C). After 120 h of the window it requests the disinfection in a cheap hour.
  Under the flat price and persistence, c is constant, so no hour is cheap and the
  controller coasts throughout. This follows from the rule as stated; it was not tuned.

**Disinfection triggers** (`DisinfectionBarrier`; each forces full output at the 65 °C
supply):

* **None:** no cycle (the reference).
* **Weekly timer:** due at hour 2 of each calendar week and held until the window
  completes, as a conventional fixed program.
* **Just-in-time:** the original schedule of `stems.deadline`. It forces the cycle when the
  hours left are no more than ceil(gap / net rate) + 1, where the gap is the heat that
  brings every node to 60 °C and the net rate is the compressor's output at 65 °C minus the
  reserve and the standing loss.
* **Recourse (committed):** rolls the model forward at full output against the causal mean
  draw forecast, plus a lump safety stock at the first step, and forces the cycle when the
  hours left are no more than the plan.
* **Recourse, quantile plan:** the same, against the causal hour-of-day 0.9-quantile draw,
  with no lump stock. It is a revision made after the hold definition: under the hold, the
  committed design plans two steps, like just-in-time, and a draw in the hold hour misses
  the window.

**Filter:** off, or on (strict phase guard, plus projection of the controller's request
onto the charges that keep the end-of-hour state inside the phase's bands). The filter is
exact for the draw it is given. With the step's draw known, it certifies that every band
violation is announced beforehand as an infeasible step. With a causal forecast of the
draw, it certifies nothing.

**Price:** TOU, or flat. **Weather forecast:** off (persistence), or on (5 seeds). It is
run for every configuration that reads a source forecast: the price-aware controller, and
the recourse triggers, which use the controller's source path.

**What is stochastic:** only the weather-forecast noise. Draws, prices and weather are one
deterministic year. Weather-on rows are seed means; the tables print [min, max] wherever
the seeds differ.

## 6. Results

### 6.1 Completion first

![Missed weekly windows by controller and trigger (n = 416 house-weeks per
run).](../results/dhw_paper/fig_completion.png)

**Table 1. Time-of-use price, weather forecast off, step's draw known to the filter.**

<!-- table:1 -->
| controller | disinfection trigger | filter | missed windows (of 416) | completion | top < 45 °C (building-h) | below 45 °C (K·h) | top > 55 °C (building-h) | above 55 °C (K·h) | unmet (kWh) | electricity (kWh) | cost (USD) | starts | kWh per completed disinfection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Price-aware | no disinfection | off | 416 | 0.0 % | 174 | 751.7 | 0 | 0 | 39.84 | 4740.0 | 1119.69 | 10737 | - |
| Price-aware | no disinfection | on | 416 | 0.0 % | 0 | 0.0 | 0 | 0 | 22.34 | 4744.9 | 1121.40 | 11226 | - |
| Price-aware | just-in-time | off | 0 | 100.0 % | 165 | 719.1 | 8389 | 51382 | 38.28 | 4847.3 | 1140.38 | 11046 | 0.26 |
| Price-aware | just-in-time | on | 0 | 100.0 % | 0 | 0.0 | 8392 | 51402 | 20.69 | 4852.3 | 1142.32 | 11483 | 0.26 |
| Price-aware | recourse (committed) | off | 0 | 100.0 % | 165 | 719.1 | 8389 | 51382 | 38.28 | 4847.3 | 1140.38 | 11046 | 0.26 |
| Price-aware | recourse (committed) | on | 0 | 100.0 % | 0 | 0.0 | 8392 | 51402 | 20.69 | 4852.3 | 1142.32 | 11483 | 0.26 |
| Price-aware | recourse, quantile plan | off | 0 | 100.0 % | 165 | 719.1 | 8389 | 51382 | 38.28 | 4847.3 | 1140.38 | 11046 | 0.26 |
| Price-aware | recourse, quantile plan | on | 0 | 100.0 % | 0 | 0.0 | 8392 | 51402 | 20.69 | 4852.3 | 1142.32 | 11483 | 0.26 |
| Reactive | no disinfection | off | 416 | 0.0 % | 185 | 1119.3 | 0 | 0 | 55.88 | 4698.8 | 1335.09 | 6856 | - |
| Reactive | no disinfection | on | 416 | 0.0 % | 0 | 0.0 | 0 | 0 | 35.02 | 4704.5 | 1336.42 | 7305 | - |
| Reactive | weekly timer | off | 0 | 100.0 % | 166 | 1007.0 | 7516 | 46054 | 46.17 | 4821.0 | 1350.60 | 7349 | 0.29 |
| Reactive | just-in-time | off | 33 | 92.1 % | 172 | 1057.1 | 4284 | 27256 | 51.39 | 4774.7 | 1347.38 | 7076 | 0.20 |
| Reactive | just-in-time | on | 33 | 92.1 % | 0 | 0.0 | 4284 | 27256 | 31.80 | 4779.9 | 1348.87 | 7492 | 0.20 |
| Reactive | recourse (committed) | off | 24 | 94.2 % | 173 | 1044.0 | 4621 | 29010 | 52.69 | 4781.9 | 1356.29 | 7103 | 0.21 |
| Reactive | recourse (committed) | on | 24 | 94.2 % | 0 | 0.0 | 4621 | 29010 | 32.67 | 4787.5 | 1358.54 | 7494 | 0.21 |
| Reactive | recourse, quantile plan | off | 6 | 98.6 % | 167 | 1037.3 | 6013 | 37528 | 52.67 | 4806.4 | 1352.97 | 7303 | 0.26 |
| Reactive | recourse, quantile plan | on | 6 | 98.6 % | 0 | 0.0 | 6013 | 37528 | 32.04 | 4812.1 | 1354.66 | 7708 | 0.26 |
<!-- /table:1 -->

**Table 2. Flat price, weather forecast off.**

<!-- table:2 -->
| controller | disinfection trigger | filter | missed windows (of 416) | completion | top < 45 °C (building-h) | below 45 °C (K·h) | top > 55 °C (building-h) | above 55 °C (K·h) | unmet (kWh) | electricity (kWh) | cost (USD) | starts | kWh per completed disinfection |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Price-aware | no disinfection | off | 416 | 0.0 % | 1288 | 4349.2 | 0 | 0 | 214.63 | 4607.9 | 1258.58 | 4851 | - |
| Price-aware | no disinfection | on | 416 | 0.0 % | 0 | 0.0 | 0 | 0 | 114.00 | 4634.5 | 1265.87 | 5118 | - |
| Price-aware | just-in-time | off | 36 | 91.3 % | 1247 | 4126.7 | 4105 | 26939 | 201.86 | 4698.1 | 1283.24 | 5087 | 0.24 |
| Price-aware | just-in-time | on | 36 | 91.3 % | 0 | 0.0 | 4104 | 26929 | 103.25 | 4723.2 | 1290.09 | 5332 | 0.23 |
| Price-aware | recourse (committed) | off | 24 | 94.2 % | 1238 | 4061.9 | 4397 | 28563 | 204.64 | 4708.1 | 1285.95 | 5115 | 0.26 |
| Price-aware | recourse (committed) | on | 24 | 94.2 % | 0 | 0.0 | 4397 | 28560 | 105.92 | 4733.9 | 1293.00 | 5354 | 0.25 |
| Price-aware | recourse, quantile plan | off | 6 | 98.6 % | 1172 | 3977.1 | 5664 | 37030 | 189.20 | 4734.9 | 1293.28 | 5297 | 0.31 |
| Price-aware | recourse, quantile plan | on | 6 | 98.6 % | 0 | 0.0 | 5664 | 37030 | 102.19 | 4760.2 | 1300.18 | 5534 | 0.31 |
| Reactive | no disinfection | off | 416 | 0.0 % | 185 | 1119.3 | 0 | 0 | 55.88 | 4698.8 | 1283.42 | 6856 | - |
| Reactive | no disinfection | on | 416 | 0.0 % | 0 | 0.0 | 0 | 0 | 35.02 | 4704.5 | 1284.97 | 7305 | - |
| Reactive | weekly timer | off | 0 | 100.0 % | 166 | 1007.0 | 7516 | 46054 | 46.17 | 4821.0 | 1316.80 | 7349 | 0.29 |
| Reactive | just-in-time | off | 33 | 92.1 % | 172 | 1057.1 | 4284 | 27256 | 51.39 | 4774.7 | 1304.14 | 7076 | 0.20 |
| Reactive | just-in-time | on | 33 | 92.1 % | 0 | 0.0 | 4284 | 27256 | 31.80 | 4779.9 | 1305.58 | 7492 | 0.20 |
| Reactive | recourse (committed) | off | 24 | 94.2 % | 173 | 1044.0 | 4621 | 29010 | 52.69 | 4781.9 | 1306.10 | 7103 | 0.21 |
| Reactive | recourse (committed) | on | 24 | 94.2 % | 0 | 0.0 | 4621 | 29010 | 32.67 | 4787.5 | 1307.64 | 7494 | 0.21 |
| Reactive | recourse, quantile plan | off | 6 | 98.6 % | 167 | 1037.3 | 6013 | 37528 | 52.67 | 4806.4 | 1312.81 | 7303 | 0.26 |
| Reactive | recourse, quantile plan | on | 6 | 98.6 % | 0 | 0.0 | 6013 | 37528 | 32.04 | 4812.1 | 1314.38 | 7708 | 0.26 |
<!-- /table:2 -->

How to read Tables 1–2:

* **Completion.** Under TOU, the price-aware controller schedules every cycle itself. With
  the reactive controller (either price) and the price-aware one under the flat price,
  completion is the trigger's: just-in-time 33–36 missed windows, committed recourse 24,
  quantile recourse 6, timer 0. The quantile plan's 6 misses are window 0 in houses 1 and 6,
  where the causal forecaster has no history yet, and two windows each in houses 2 and 4.
  These are the two lowest-draw houses, where every hour passes the 0.9-quantile test and
  the hold is left to the last 2 h. On the rolling reading, the timer and the price-aware controller under TOU never leave
  more than 168 h between holds (169 h once, with the filter). With the reactive
  controller, just-in-time leaves 197 gaps over 168 h, recourse 180 and the quantile plan
  63, the longest 335–336 h.
* **Comfort.** The disinfection itself raises the store temperature, which is why each
  trigger has slightly fewer hours below 45 °C than its no-disinfection reference. The
  filter takes the end-of-hour count to 0 in every configuration on this unit.
* **Hours above 55 °C.** These are mostly the disinfection and the cool-down after it, so
  more reliable triggers spend more hours above 55 °C. The filter does not change them
  (8,389 → 8,392 for the price-aware controller under TOU). The filter's own phase labels
  would report 0 such hours outside phase D, because the strict guard keeps the phase at D
  until the store is back inside both bands. That is relabelling, so the physical count is
  the one reported.
* **Electricity, cost, starts.** Disinfecting adds 76–122 kWh/yr to the reactive runs and
  107 kWh/yr to the price-aware runs under TOU. The price-aware controller starts the unit
  about 1.5× as often as the reactive one (10,737–11,483 against 6,856–7,708 starts/yr for
  8 houses), and the filter adds 391–489 starts/yr under TOU.

### 6.2 Manal's axes as differences

Each row is one configuration with one axis switched and everything else the same.
Weather-on rows are means over 5 seeds.

**Weather forecast on minus off.**

<!-- table:delta_weather -->
| controller | trigger | filter | price | Δ missed windows | Δ top < 45 °C (building-h) | Δ unmet (kWh) | Δ electricity (kWh) | Δ cost (USD) | Δ starts |
|---|---|---|---|---|---|---|---|---|---|
| Price-aware | just-in-time | off | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Price-aware | no disinfection | off | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Price-aware | recourse (committed) | off | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Price-aware | recourse, quantile plan | off | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Price-aware | just-in-time | off | flat | +0.0 | -15 | +4.21 | -0.8 | -0.22 | +65 |
| Price-aware | no disinfection | off | flat | +0.0 | -6 | +3.94 | -1.0 | -0.29 | +74 |
| Price-aware | recourse (committed) | off | flat | -2.0 | -15 | +6.05 | -3.7 | -1.02 | +68 |
| Price-aware | recourse, quantile plan | off | flat | +0.0 | -10 | +0.65 | -0.0 | -0.01 | +65 |
| Price-aware | just-in-time | on | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Price-aware | no disinfection | on | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Price-aware | recourse (committed) | on | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Price-aware | recourse, quantile plan | on | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Price-aware | just-in-time | on | flat | +0.0 | +0 | +2.69 | -0.1 | -0.02 | +68 |
| Price-aware | no disinfection | on | flat | +0.0 | +0 | +1.48 | -0.2 | -0.04 | +78 |
| Price-aware | recourse (committed) | on | flat | -2.0 | +0 | +4.97 | -3.2 | -0.86 | +69 |
| Price-aware | recourse, quantile plan | on | flat | +0.0 | +0 | -2.96 | +1.0 | +0.28 | +69 |
| Reactive | recourse (committed) | off | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Reactive | recourse, quantile plan | off | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Reactive | recourse (committed) | off | flat | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Reactive | recourse, quantile plan | off | flat | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Reactive | recourse (committed) | on | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Reactive | recourse, quantile plan | on | dynamic | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Reactive | recourse (committed) | on | flat | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
| Reactive | recourse, quantile plan | on | flat | +0.0 | +0 | +0.00 | +0.0 | +0.00 | +0 |
<!-- /table:delta_weather -->

**Flat minus TOU price.**

<!-- table:delta_price -->
| controller | trigger | filter | weather | Δ missed windows | Δ top < 45 °C (building-h) | Δ unmet (kWh) | Δ electricity (kWh) | Δ cost (USD) | Δ starts |
|---|---|---|---|---|---|---|---|---|---|
| Price-aware | just-in-time | off | off | +36.0 | +1082 | +163.58 | -149.2 | +142.86 | -5959 |
| Price-aware | no disinfection | off | off | +0.0 | +1114 | +174.79 | -132.1 | +138.89 | -5886 |
| Price-aware | recourse (committed) | off | off | +24.0 | +1073 | +166.36 | -139.3 | +145.57 | -5931 |
| Price-aware | recourse, quantile plan | off | off | +6.0 | +1007 | +150.93 | -112.4 | +152.90 | -5749 |
| Price-aware | just-in-time | off | on | +36.0 | +1067 | +167.79 | -150.0 | +142.64 | -5894 |
| Price-aware | no disinfection | off | on | +0.0 | +1108 | +178.73 | -133.2 | +138.61 | -5812 |
| Price-aware | recourse (committed) | off | on | +22.0 | +1058 | +172.41 | -143.0 | +144.55 | -5863 |
| Price-aware | recourse, quantile plan | off | on | +6.0 | +997 | +151.57 | -112.4 | +152.89 | -5684 |
| Price-aware | just-in-time | on | off | +36.0 | +0 | +82.56 | -129.1 | +147.77 | -6151 |
| Price-aware | no disinfection | on | off | +0.0 | +0 | +91.66 | -110.4 | +144.46 | -6108 |
| Price-aware | recourse (committed) | on | off | +24.0 | +0 | +85.23 | -118.4 | +150.68 | -6129 |
| Price-aware | recourse, quantile plan | on | off | +6.0 | +0 | +81.50 | -92.1 | +157.86 | -5949 |
| Price-aware | just-in-time | on | on | +36.0 | +0 | +85.24 | -129.1 | +147.74 | -6083 |
| Price-aware | no disinfection | on | on | +0.0 | +0 | +93.14 | -110.6 | +144.42 | -6030 |
| Price-aware | recourse (committed) | on | on | +22.0 | +0 | +90.20 | -121.5 | +149.82 | -6060 |
| Price-aware | recourse, quantile plan | on | on | +6.0 | +0 | +78.54 | -91.1 | +158.14 | -5880 |
| Reactive | just-in-time | off | off | +0.0 | +0 | +0.00 | +0.0 | -43.24 | +0 |
| Reactive | no disinfection | off | off | +0.0 | +0 | +0.00 | +0.0 | -51.67 | +0 |
| Reactive | recourse (committed) | off | off | +0.0 | +0 | +0.00 | +0.0 | -50.19 | +0 |
| Reactive | recourse, quantile plan | off | off | +0.0 | +0 | +0.00 | +0.0 | -40.17 | +0 |
| Reactive | weekly timer | off | off | +0.0 | +0 | +0.00 | +0.0 | -33.80 | +0 |
| Reactive | recourse (committed) | off | on | +0.0 | +0 | +0.00 | +0.0 | -50.19 | +0 |
| Reactive | recourse, quantile plan | off | on | +0.0 | +0 | +0.00 | +0.0 | -40.17 | +0 |
| Reactive | just-in-time | on | off | +0.0 | +0 | +0.00 | +0.0 | -43.29 | +0 |
| Reactive | no disinfection | on | off | +0.0 | +0 | +0.00 | +0.0 | -51.45 | +0 |
| Reactive | recourse (committed) | on | off | +0.0 | +0 | +0.00 | +0.0 | -50.90 | +0 |
| Reactive | recourse, quantile plan | on | off | +0.0 | +0 | +0.00 | +0.0 | -40.28 | +0 |
| Reactive | recourse (committed) | on | on | +0.0 | +0 | +0.00 | +0.0 | -50.90 | +0 |
| Reactive | recourse, quantile plan | on | on | +0.0 | +0 | +0.00 | +0.0 | -40.28 | +0 |
<!-- /table:delta_price -->

**Filter on minus off (step's draw known).**

<!-- table:delta_filter -->
| controller | trigger | price | weather | Δ missed windows | Δ top < 45 °C (building-h) | Δ unmet (kWh) | Δ electricity (kWh) | Δ cost (USD) | Δ starts |
|---|---|---|---|---|---|---|---|---|---|
| Price-aware | just-in-time | dynamic | off | +0.0 | -165 | -17.59 | +4.9 | +1.94 | +437 |
| Price-aware | no disinfection | dynamic | off | +0.0 | -174 | -17.50 | +4.9 | +1.71 | +489 |
| Price-aware | recourse (committed) | dynamic | off | +0.0 | -165 | -17.59 | +4.9 | +1.94 | +437 |
| Price-aware | recourse, quantile plan | dynamic | off | +0.0 | -165 | -17.59 | +4.9 | +1.94 | +437 |
| Price-aware | just-in-time | dynamic | on | +0.0 | -165 | -17.59 | +4.9 | +1.94 | +437 |
| Price-aware | no disinfection | dynamic | on | +0.0 | -174 | -17.50 | +4.9 | +1.71 | +489 |
| Price-aware | recourse (committed) | dynamic | on | +0.0 | -165 | -17.59 | +4.9 | +1.94 | +437 |
| Price-aware | recourse, quantile plan | dynamic | on | +0.0 | -165 | -17.59 | +4.9 | +1.94 | +437 |
| Price-aware | just-in-time | flat | off | +0.0 | -1247 | -98.61 | +25.1 | +6.85 | +245 |
| Price-aware | no disinfection | flat | off | +0.0 | -1288 | -100.63 | +26.7 | +7.28 | +267 |
| Price-aware | recourse (committed) | flat | off | +0.0 | -1238 | -98.72 | +25.8 | +7.06 | +239 |
| Price-aware | recourse, quantile plan | flat | off | +0.0 | -1172 | -87.01 | +25.3 | +6.90 | +237 |
| Price-aware | just-in-time | flat | on | +0.0 | -1232 | -100.13 | +25.8 | +7.05 | +248 |
| Price-aware | no disinfection | flat | on | +0.0 | -1282 | -103.09 | +27.5 | +7.52 | +271 |
| Price-aware | recourse (committed) | flat | on | +0.0 | -1223 | -99.80 | +26.4 | +7.21 | +240 |
| Price-aware | recourse, quantile plan | flat | on | +0.0 | -1162 | -90.62 | +26.3 | +7.18 | +241 |
| Reactive | just-in-time | dynamic | off | +0.0 | -172 | -19.59 | +5.3 | +1.49 | +416 |
| Reactive | no disinfection | dynamic | off | +0.0 | -185 | -20.85 | +5.7 | +1.33 | +449 |
| Reactive | recourse (committed) | dynamic | off | +0.0 | -173 | -20.01 | +5.6 | +2.25 | +391 |
| Reactive | recourse, quantile plan | dynamic | off | +0.0 | -167 | -20.63 | +5.7 | +1.68 | +405 |
| Reactive | recourse (committed) | dynamic | on | +0.0 | -173 | -20.01 | +5.6 | +2.25 | +391 |
| Reactive | recourse, quantile plan | dynamic | on | +0.0 | -167 | -20.63 | +5.7 | +1.68 | +405 |
| Reactive | just-in-time | flat | off | +0.0 | -172 | -19.59 | +5.3 | +1.44 | +416 |
| Reactive | no disinfection | flat | off | +0.0 | -185 | -20.85 | +5.7 | +1.55 | +449 |
| Reactive | recourse (committed) | flat | off | +0.0 | -173 | -20.01 | +5.6 | +1.53 | +391 |
| Reactive | recourse, quantile plan | flat | off | +0.0 | -167 | -20.63 | +5.7 | +1.57 | +405 |
| Reactive | recourse (committed) | flat | on | +0.0 | -173 | -20.01 | +5.6 | +1.53 | +391 |
| Reactive | recourse, quantile plan | flat | on | +0.0 | -167 | -20.63 | +5.7 | +1.57 | +405 |
<!-- /table:delta_filter -->

* **Weather.** Zero for the reactive controller and for the price-aware controller under
  TOU. Under the flat price, the only change is forecast noise making the odd hour look
  cheap. A ground-source unit's brine temperature follows a 24-h mean of the outdoor
  temperature, so a 24-h forecast has little to add.
* **Price.** The reactive controller's physics is identical under both tariffs, and its
  bill is 34–52 USD/yr lower under the flat price, because it also buys at the evening
  peak. Under the flat price, the price-aware controller loses its cheap hours. It coasts,
  uses 91–150 kWh/yr less electricity and pays 139–158 USD/yr more. It has 997–1,114 more
  building-hours below 45 °C without the filter, and its completion falls back on the
  trigger.
* **Filter.** Under TOU it buys 165–185 fewer building-hours below 45 °C and 17.5–20.9 kWh/yr
  less unmet water, for 4.9–5.7 kWh/yr of electricity, 1.33–2.25 USD/yr and 391–489 extra
  starts. For the coasting price-aware controller under the flat price, it buys 1,162–1,288
  fewer building-hours below 45 °C and 87–103 kWh/yr less unmet water, for 25.1–27.5 kWh/yr. On
  this unit it never changes completion.

### 6.3 Electricity per completed disinfection

![Missed windows against net electricity per completed disinfection (filter off; with the
filter on, completion is identical and the electricity per disinfection moves by at most
0.01 kWh).](../results/dhw_paper/fig_tradeoff.png)

The net figure is small because the heat put in for the cycle is mostly drawn off as hot
water in the following hours, which displaces later charging. What the cycle really costs
is the lower COP of the 65 °C supply and the extra standing loss while the store is hot.
The gross electricity spent in phase D is not reported. It depends on how long each policy
labels steps as D, and the filter's guard prolongs D by construction. Among the reactive
triggers, reliability costs energy: the timer (0 missed) costs 0.29 kWh per completed
cycle, the quantile plan (6) 0.26, recourse (24) 0.21 and just-in-time (33) 0.20. Under TOU
the price-aware controller reaches 0 missed windows at 0.26 kWh.

### 6.4 The filter without the step's draw

**Table 3. The filter given a causal forecast of the step's draw (weather off).** There is
no certificate, so the unreported-violation column is not defined (–).

<!-- table:3 -->
dynamic price:

| controller | trigger | filter's draw | missed windows | top < 45 °C (building-h) | unmet (kWh) | electricity (kWh) | cost (USD) | unreported band violations |
|---|---|---|---|---|---|---|---|---|
| Price-aware | just-in-time | forecast | 0 | 170 | 40.05 | 4845.9 | 1138.21 | - |
| Price-aware | just-in-time | known | 0 | 0 | 20.69 | 4852.3 | 1142.32 | 0 |
| Price-aware | recourse (committed) | forecast | 0 | 170 | 40.05 | 4845.9 | 1138.21 | - |
| Price-aware | recourse (committed) | known | 0 | 0 | 20.69 | 4852.3 | 1142.32 | 0 |
| Price-aware | recourse, quantile plan | forecast | 0 | 150 | 38.55 | 4846.7 | 1142.03 | - |
| Price-aware | recourse, quantile plan | known | 0 | 0 | 20.69 | 4852.3 | 1142.32 | 0 |
| Reactive | just-in-time | forecast | 36 | 176 | 49.71 | 4773.2 | 1347.08 | - |
| Reactive | just-in-time | known | 33 | 0 | 31.80 | 4779.9 | 1348.87 | 0 |
| Reactive | recourse (committed) | forecast | 25 | 177 | 51.08 | 4782.6 | 1356.93 | - |
| Reactive | recourse (committed) | known | 24 | 0 | 32.67 | 4787.5 | 1358.54 | 0 |
| Reactive | recourse, quantile plan | forecast | 6 | 165 | 49.97 | 4806.4 | 1353.22 | - |
| Reactive | recourse, quantile plan | known | 6 | 0 | 32.04 | 4812.1 | 1354.66 | 0 |

flat price:

| controller | trigger | filter's draw | missed windows | top < 45 °C (building-h) | unmet (kWh) | electricity (kWh) | cost (USD) | unreported band violations |
|---|---|---|---|---|---|---|---|---|
| Price-aware | just-in-time | forecast | 43 | 1274 | 205.92 | 4690.7 | 1281.19 | - |
| Price-aware | just-in-time | known | 36 | 0 | 103.25 | 4723.2 | 1290.09 | 0 |
| Price-aware | recourse (committed) | forecast | 27 | 1265 | 204.54 | 4702.2 | 1284.33 | - |
| Price-aware | recourse (committed) | known | 24 | 0 | 105.92 | 4733.9 | 1293.00 | 0 |
| Price-aware | recourse, quantile plan | forecast | 6 | 1044 | 178.27 | 4737.7 | 1294.05 | - |
| Price-aware | recourse, quantile plan | known | 6 | 0 | 102.19 | 4760.2 | 1300.18 | 0 |
| Reactive | just-in-time | forecast | 36 | 176 | 49.71 | 4773.2 | 1303.75 | - |
| Reactive | just-in-time | known | 33 | 0 | 31.80 | 4779.9 | 1305.58 | 0 |
| Reactive | recourse (committed) | forecast | 25 | 177 | 51.08 | 4782.6 | 1306.30 | - |
| Reactive | recourse (committed) | known | 24 | 0 | 32.67 | 4787.5 | 1307.64 | 0 |
| Reactive | recourse, quantile plan | forecast | 6 | 165 | 49.97 | 4806.4 | 1312.80 | - |
| Reactive | recourse, quantile plan | known | 6 | 0 | 32.04 | 4812.1 | 1314.38 | 0 |
<!-- /table:3 -->

With a forecast draw, the filter projects against the wrong next state. End-of-hour
violations return close to the unfiltered level (150–177 building-hours under TOU,
1,044–1,274 for the price-aware controller under the flat price). Completion is unchanged
or slightly worse: just-in-time 36 instead of 33, recourse 25 instead of 24 (TOU, reactive).
The guarantee is only as good as the draw the filter is given. In a real installation the
hour's draw is not known in advance, so the deployable claim is the forecast-draw row, not
the known-draw row.

### 6.5 Power-limited units

**Table 4. Units sized by the design note's rule** (mean daily demand / 4 h at B0/W35,
×1 = `rule1`: 0.37–1.94 kW; ×2 = `rule2`: 0.74–3.89 kW; same hplib shape and 25 % floor;
TOU price, weather off).

<!-- table:4 -->
rule1:

| controller | disinfection trigger | filter | missed windows (of 416) | completion | top < 45 °C (building-h) | below 45 °C (K·h) | top > 55 °C (building-h) | above 55 °C (K·h) | unmet (kWh) | electricity (kWh) | cost (USD) | starts | kWh per completed disinfection | steps reported infeasible | unreported band violations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Price-aware | no disinfection | off | 416 | 0.0 % | 537 | 2165.9 | 0 | 0 | 68.23 | 4716.6 | 1079.87 | 9399 | - | - | - |
| Price-aware | no disinfection | on | 416 | 0.0 % | 349 | 1296.4 | 0 | 0 | 52.50 | 4721.8 | 1087.49 | 9578 | - | 349 | 0 |
| Price-aware | just-in-time | off | 0 | 100.0 % | 481 | 1931.7 | 9048 | 55326 | 62.34 | 4841.1 | 1105.23 | 9529 | 0.30 | - | - |
| Price-aware | just-in-time | on | 0 | 100.0 % | 304 | 1094.6 | 9052 | 55353 | 46.65 | 4845.1 | 1112.23 | 9675 | 0.30 | 304 | 0 |
| Price-aware | recourse (committed) | off | 0 | 100.0 % | 489 | 1957.8 | 9023 | 55227 | 62.34 | 4838.9 | 1104.37 | 9529 | 0.29 | - | - |
| Price-aware | recourse (committed) | on | 0 | 100.0 % | 309 | 1103.0 | 9053 | 55440 | 46.65 | 4843.6 | 1111.64 | 9669 | 0.29 | 309 | 0 |
| Price-aware | recourse, quantile plan | off | 0 | 100.0 % | 479 | 1914.2 | 9166 | 55776 | 61.93 | 4856.0 | 1111.63 | 9492 | 0.34 | - | - |
| Price-aware | recourse, quantile plan | on | 0 | 100.0 % | 297 | 1072.4 | 9200 | 56001 | 47.28 | 4859.4 | 1118.38 | 9636 | 0.33 | 297 | 0 |
| Reactive | no disinfection | off | 416 | 0.0 % | 626 | 2908.9 | 0 | 0 | 88.64 | 4672.5 | 1310.74 | 6045 | - | - | - |
| Reactive | no disinfection | on | 416 | 0.0 % | 459 | 1824.8 | 0 | 0 | 66.67 | 4679.2 | 1313.86 | 6106 | - | 459 | 0 |
| Reactive | weekly timer | off | 0 | 100.0 % | 554 | 2632.9 | 8751 | 53383 | 84.13 | 4824.4 | 1331.94 | 6296 | 0.37 | - | - |
| Reactive | just-in-time | off | 29 | 93.0 % | 585 | 2715.2 | 7310 | 44237 | 83.45 | 4859.9 | 1389.56 | 6026 | 0.48 | - | - |
| Reactive | just-in-time | on | 27 | 93.5 % | 428 | 1686.8 | 7268 | 44020 | 62.57 | 4861.4 | 1392.26 | 6077 | 0.47 | 428 | 0 |
| Reactive | recourse (committed) | off | 39 | 90.6 % | 576 | 2582.0 | 5927 | 36431 | 83.51 | 4810.0 | 1378.00 | 6088 | 0.36 | - | - |
| Reactive | recourse (committed) | on | 40 | 90.4 % | 433 | 1682.0 | 5896 | 36313 | 62.70 | 4815.6 | 1381.15 | 6137 | 0.36 | 433 | 0 |
| Reactive | recourse, quantile plan | off | 0 | 100.0 % | 556 | 2517.3 | 8542 | 52243 | 83.77 | 4852.7 | 1351.94 | 6270 | 0.43 | - | - |
| Reactive | recourse, quantile plan | on | 0 | 100.0 % | 407 | 1594.5 | 8541 | 52199 | 65.57 | 4855.4 | 1351.98 | 6309 | 0.42 | 407 | 0 |

rule2:

| controller | disinfection trigger | filter | missed windows (of 416) | completion | top < 45 °C (building-h) | below 45 °C (K·h) | top > 55 °C (building-h) | above 55 °C (K·h) | unmet (kWh) | electricity (kWh) | cost (USD) | starts | kWh per completed disinfection | steps reported infeasible | unreported band violations |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Price-aware | no disinfection | off | 416 | 0.0 % | 223 | 939.8 | 0 | 0 | 44.65 | 4734.8 | 1102.62 | 10646 | - | - | - |
| Price-aware | no disinfection | on | 416 | 0.0 % | 51 | 182.7 | 0 | 0 | 26.90 | 4740.1 | 1105.71 | 10956 | - | 51 | 0 |
| Price-aware | just-in-time | off | 0 | 100.0 % | 201 | 851.8 | 8567 | 52799 | 41.75 | 4844.3 | 1123.69 | 10895 | 0.26 | - | - |
| Price-aware | just-in-time | on | 0 | 100.0 % | 41 | 127.1 | 8571 | 52812 | 23.81 | 4849.4 | 1126.75 | 11177 | 0.26 | 41 | 0 |
| Price-aware | recourse (committed) | off | 0 | 100.0 % | 201 | 851.8 | 8567 | 52799 | 41.75 | 4844.3 | 1123.69 | 10895 | 0.26 | - | - |
| Price-aware | recourse (committed) | on | 0 | 100.0 % | 41 | 127.1 | 8571 | 52812 | 23.81 | 4849.4 | 1126.75 | 11177 | 0.26 | 41 | 0 |
| Price-aware | recourse, quantile plan | off | 0 | 100.0 % | 201 | 851.8 | 8567 | 52799 | 41.75 | 4844.3 | 1123.69 | 10895 | 0.26 | - | - |
| Price-aware | recourse, quantile plan | on | 0 | 100.0 % | 41 | 127.1 | 8571 | 52812 | 23.81 | 4849.4 | 1126.75 | 11177 | 0.26 | 41 | 0 |
| Reactive | no disinfection | off | 416 | 0.0 % | 258 | 1485.4 | 0 | 0 | 61.38 | 4691.6 | 1328.50 | 6785 | - | - | - |
| Reactive | no disinfection | on | 416 | 0.0 % | 78 | 305.2 | 0 | 0 | 41.72 | 4697.3 | 1331.53 | 7030 | - | 78 | 0 |
| Reactive | weekly timer | off | 0 | 100.0 % | 228 | 1326.5 | 7833 | 48222 | 49.69 | 4821.3 | 1346.43 | 7214 | 0.31 | - | - |
| Reactive | just-in-time | off | 58 | 86.1 % | 223 | 1310.3 | 4670 | 29630 | 51.76 | 4785.4 | 1352.88 | 6928 | 0.26 | - | - |
| Reactive | just-in-time | on | 61 | 85.3 % | 63 | 246.8 | 4649 | 29492 | 36.95 | 4789.7 | 1356.00 | 7149 | 0.26 | 63 | 0 |
| Reactive | recourse (committed) | off | 31 | 92.5 % | 225 | 1278.2 | 5089 | 31979 | 53.31 | 4797.1 | 1367.33 | 6939 | 0.27 | - | - |
| Reactive | recourse (committed) | on | 31 | 92.5 % | 62 | 248.6 | 5077 | 31903 | 37.77 | 4802.2 | 1370.17 | 7153 | 0.27 | 62 | 0 |
| Reactive | recourse, quantile plan | off | 5 | 98.8 % | 227 | 1306.8 | 6472 | 40534 | 55.26 | 4810.8 | 1358.60 | 7160 | 0.29 | - | - |
| Reactive | recourse, quantile plan | on | 5 | 98.8 % | 68 | 265.6 | 6456 | 40443 | 36.88 | 4815.4 | 1362.41 | 7388 | 0.29 | 68 | 0 |
<!-- /table:4 -->

* **What the certificate means on a small unit.** The filter cannot make heat. On a
  power-limited unit some steps are infeasible (the floor is out of reach even at full
  output). The filter reports them, and every end-of-hour violation in Table 4 is such a
  reported step: building-hours below 45 °C equal the steps reported infeasible in all 16
  filtered power-limited runs, and none is unreported.
* **Completion.** The price-aware controller still completes every window. The reactive
  triggers do not order consistently with unit size: just-in-time misses 29 windows on
  `rule1`, 58 on `rule2` and 33 on the Calibra Eco 8. This was not diagnosed. A bigger unit
  shortens the planned cycle and moves it closer to the deadline, which is a plausible
  mechanism but untested.

## 7. Limits

* **One year, eight houses, no held-out data.** Nothing here estimates performance on
  other houses or years.
* **Texas draws and weather, German brine correlation.** Travis County is warm (mean
  20.7 °C), so brine temperatures are high and the 65 °C supply is easy. Swedish
  conditions would lower both the COP and the capacity at 65 °C.
* **Two nodes.** The hold on the bottom node is a statement about the mean temperature of
  the bottom half. Thermoclines within a node are not resolved. A 3-node sensitivity was
  not run.
* **The certificate is end-of-hour only.** Within the hour, after the draw and before the
  charge, the top node falls below 45 °C in 667–865 building-hours/yr (TOU), with or
  without the filter.
* **Cold start.** The causal draw forecasts have no history in the first window.
* **CityLearn's loss convention** (proportional to stored energy) and the 10 °C cold-water
  temperature are not from Thermia.
* **Starts are counted but neither priced nor limited.**
* **MPC and DRL controllers are not included.** The two rule controllers are baselines,
  not a claim about the best achievable policy.

## 8. Open questions for Manal

1. **Charging geometry:** the Calibra Eco's integrated tank heats through a coil over the
   whole tank height. The model here charges externally and stratified (from the bottom,
   returned at the buoyancy level). Which applies to the units the paper describes? A
   full-height coil heats the column more uniformly, which changes comfort and the hold.
2. **Thermia's own anti-legionella program:** which sensor decides completion (top,
   bottom or both), the target temperature and hold time, the schedule (weekday, hour,
   interval), and whether the auxiliary heater supplies the last degrees.
3. **The 65 °C supply at low brine temperature:** does the compressor deliver it at B0 and
   below on the Calibra Eco, or does the auxiliary heater take over? The model assumes the
   compressor alone reaches 65 °C.
4. **Nameplate rating point:** is "8 kW" quoted at B0/W35 (assumed here) or at another
   point? Every capacity and COP above scales with it.
5. **Normal DHW mode:** is a 55 °C supply right, and what start/stop temperatures does
   Thermia use for hot water?
6. **Which rule to cite:** for a Swedish single-family house, BBR binds only ≥ 50 °C at
   the tap. Is a weekly ≥ 60 °C hold of the whole store what Thermia ships, or what the
   paper should assume?
7. **What "weather forecast" means for a ground-source unit:** here it is a forecast of
   the brine temperature (persistence when off). For hot water alone it makes no
   difference. Is the intended axis a weather-compensated space-heating curve interacting
   with hot water instead?
8. **Data:** would a Swedish climate and dataset be preferred to Travis County, and are
   measured draw profiles from Thermia installations available?
9. **Tariff:** is the flat price as the annual mean of the TOU tariff the right
   comparison, or should it be a Swedish tariff (fixed, or hourly spot)?
10. **Compressor protection:** are there limits on minimum run time or starts per hour?
    Starts are counted, not constrained.

## 9. Reproduction

`scripts/grid_env.sh` is a local helper and is not committed. It sets
`PYTHONPATH=<this worktree>`, puts the `stems` virtualenv's `bin/` first on `PATH`, points
`XDG_CACHE_HOME` at the CityLearn dataset cache, and sets `OMP_NUM_THREADS=1`. hplib is not
a dependency: its generic coefficients are copied as constants into
`stems/heatpump_hplib.py`. `series_8759.npz` (5.6 MB) is not committed; `cache_series.py`
regenerates it from the schema.

```
cd stems-dhwpaper && source scripts/grid_env.sh
python -B experiments/dhw_paper/cache_series.py       # results/dhw_paper/series_8759.npz
python -B experiments/dhw_paper/run_matrix.py --grid main --workers 3            # 152 runs
python -B experiments/dhw_paper/run_matrix.py --grid base_filter --workers 3     # 18 runs
python -B experiments/dhw_paper/run_matrix.py --grid power_limited --workers 3   # 30 runs
python -B experiments/dhw_paper/before_after.py       # Table 0
python -B experiments/dhw_paper/trace_week.py --house 0
python -B experiments/dhw_paper/make_tables.py        # tables, figures, and this doc's tables
python -B experiments/dhw_paper/figcheck_discrimination.py   # the overlap check flags known-bad layouts
python -B -m pytest -q tests/test_dhw_stratified.py tests/test_dhw_plant.py tests/test_dhw_barrier.py
```

`make_tables.py` and `trace_week.py` run a programmatic overlap check on every figure
before saving it (`experiments/dhw_paper/figcheck.py`). The check looks for text boxes
overlapping each other, and for data markers inside a text or legend box, at the 300 dpi
the figures are saved at. All four figures pass.
`figcheck_discrimination.py` re-renders the three layouts that had been corrected by eye
(single-line completion titles, the trace legend over the band label, the trade-off legend
over a data point) and confirms the check flags each one. Results:
`results/dhw_paper/figure_checks.json`.

Each year-long run takes about 45 s on one core (median of the main grid), so the 152-run
main grid needs about 40 min on 3 workers.
