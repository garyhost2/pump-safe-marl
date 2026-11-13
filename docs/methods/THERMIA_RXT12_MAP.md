# Thermia Calibra RXT 12 — performance map from certified test data

Step 1 of the Thermia rebuild (decisions of 2026-10-19: model of record = Calibra RXT 12 400 V,
R290, built-in 184 L tank; performance map; datasheet now, Thermia simulator later).
Code `stems/thermia_rxt12.py`, data `stems/data/thermia_rxt12_keymark.json`, tests
`tests/test_thermia_rxt12.py` (13 pass), figure `docs/figures/rxt12_map_validation.png`.

## 1. This replaces the previous model, and why

`stems/heatpump_thermia.py` modelled the wrong product (Calibra Eco 8, R452B, 2-8 kW) with a
Carnot fit whose every parameter was a required argument with no sourced value. The unit we
told Andreas about - ground source, R290, inverter, supply up to 70 C - is the Calibra RXT.

## 2. Source of every number

Heat Pump KEYMARK report **012-C700403 / Issue 1**, certification body RISE CERT, tests by
Universitaet Stuttgart IGTE to EN 14511:2022, EN 14825:2022, EN 16147:2017+A1:2022. Block used:
"Thermia Calibra RXT 12 400V | Heating + DHW + low temp", brine/water and water/water. Plus the
Thermia datasheet (250908_CalibraRXT_EN). Both PDFs and their text are in `docs/thermia_sources/`.

21 operating points: 11 brine (source 0 C) and 10 water (source 10 C), supply 24-55 C, 2.70-13.85 kW.
The supply temperature of each EN 14825 part-load point is fixed by the standard; it was checked
against independent test reports (OST WPZ brine/water results: B0/W34, W30; B0/W55, W52, W42,
W36, W30; a TUV report: A(-7)/W34, A2/W30, A7/W27, A12/W24) rather than written from memory.
Colder- and warmer-climate points are excluded because their supply temperatures were not verified.

## 3. Model

* temperatures enter through lift only, `L = T_supply - T_source_eff`;
  `T_source_eff = T_source + dw` for the water/water points, `dw` fitted (1.95 K) - water at
  10 C transfers heat better than ethanol brine at 0 C. **Assumption:** with a single certified
  brine temperature, the brine source-temperature slope is carried by lift, not measured.
* `Qmax = a0 + a1*T_source_eff + a2*T_supply` fitted to the four full-load points, then capped so
  electric input stays within the datasheet's 4.7 kW compressor + 0.2 kW pumps; no compressor
  output above 70 C (KEYMARK WTOL, datasheet heating-circuit maximum).
* modulation floor `Qmin = c0 + c1*T_source_eff` from the four EN 14825 D(+12) points, where the
  unit could not modulate down to the demand: 2.93 kW at brine 0 C (datasheet: "3-12 kW").
* `1/COP = b0 + b1 L + b2 L^2 + b3 Q + b4/Q + b5 s^3`, `s = Q/Qmax`. The `s^3` term carries the
  efficiency collapse near full speed the certificate shows (W10/W34: 12.25 kW at COP 5.73;
  W10/W35: 13.85 kW at COP 4.73).
* below the floor the unit cycles, with the EN 14825 degradation form and the mean certified
  `Cdh` = 0.9705; above `Qmax` the 3/6/9 kW element covers the deficit at COP 1; standby 17 W (PSB).

The form was chosen by leave-one-out error among nine candidates (EnergyPlus multiplicative,
Carnot-efficiency, several linear-EIR forms). Without the speed term and the fluid offset every
form stalled near 4.5 % LOO RMS with the same failing point, W10/W35 at full load.

## 4. Accuracy

| | in-sample RMS | in-sample max | leave-one-out RMS | leave-one-out max |
|---|---|---|---|---|
| all 21 points | 2.55 % | 4.77 % | 4.18 % | 9.07 % |
| brine points (the application fluid) | 2.16 % | 4.05 % | 3.25 % | 5.98 % |

Per-point errors: `results/thermia_rxt12/fit_report.json`.

## 5. What this model does NOT know - read before using it for disinfection

1. **No certified point above 55 C supply.** Between 55 C and 70 C - the whole disinfection
   range - COP and capacity are extrapolated by the lift form, and `operate()` returns
   `extrapolated=True`. The map offers **9.8 kW at COP 2.27 at B0/W70**: near-nominal capacity
   at 70 C, which is exactly what Andreas warned against assuming. The electric-input cap does
   not bind there (4.32 kW of 4.9 kW). **No disinfection cost or completion result computed with
   this map is publishable until Thermia simulator points replace the 55-70 C region.**
2. **Brine below 0 C is extrapolated.** A Swedish borehole runs roughly -3 to +5 C in winter.
3. **Source dependence of brine performance is inferred, not measured** (Section 3).
4. Compressor starts under cycling are counted once per call; the true number per hour depends
   on the tank and controller, which steps 3-4 add.
5. COP follows the EN 14511 convention (includes the pumps' share), not compressor-only.

## 6. Questions for Manal (Thermia simulator)

1. Capacity and electric input at B0 and B-5 for supply 55, 60, 65, 70 C, at minimum, mid and
   maximum compressor speed. This one table replaces the extrapolated region.
2. Is maximum compressor speed limited at high supply or low brine temperature (envelope)?
3. How the built-in 184 L tank is charged (TWS): flow path, where the hot water enters.
4. Where the immersion element sits in the tank, and when the controller enables it.
5. ~~Settings of the anti-legionella function~~ - **answered, see Section 7.**

## 7. Answer from Thermia (Manal Rahal, 2026-10-19): anti-legionella

* The new propane (R290) models, the RXT included, **reach 70 C with the compressor alone**.
  The old models needed the electric heater.
* Old models: the legionella program starts together with a normal hot-water heating start
  (to make it efficient), once one week has passed without it running.
* New models: the interval is a **user setting** (1 week, 10 days, ...), chosen by national
  rules and energy prices.
* **Modelling assumption agreed with Thermia: once a week, on a fixed schedule, water heated
  to 60 C.**

Consequences for this model and the hot-water work:

1. The disinfection cycle runs on the compressor; the element is not part of it. This
   **refutes** the earlier repo assumption (heat pump to 55 C, element above) and with it the
   "71.4 % of cycle electricity is resistive" figure, which was a property of that assumption,
   not of the RXT.
2. It confirms the compressor can *operate* up to 70 C. It does **not** confirm the map's
   capacity and COP there (9.8 kW at COP 2.27 at B0/W70) - those remain extrapolated until
   question 1 is answered.
3. The old-model rule (start with a normal heating cycle once a week has passed) is Thermia's
   own scheduling logic and becomes the natural baseline for our scheduler; the weekly fixed
   schedule is the agreed default.
4. Thermia gave no hold time. "Whole tank at or above 60 C for at least 1 h per week" stays
   **our** definition and must be stated as such.
