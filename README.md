# pump-safe-marl

Safe multi-agent control of ground-source heat pumps and hot-water tanks under a shared power limit.

Eight Swedish houses, each with a Thermia Calibra RXT 12 ground-source heat pump and its 184 L hot-water
tank, share one grid connection. Controllers (rules, model-predictive control and multi-agent
reinforcement learning) try to heat the houses and the water at the lowest electricity cost. A safety
filter sits behind every controller and keeps each house and the whole street inside its power limit,
keeps the indoor temperature in its band, and keeps the hot water hot enough to be safe from Legionella.

> **Status: research code.** The heat-pump model is not yet valid above 55 °C supply (no certified
> data there) and the tank is still charged directly instead of through Thermia's coil. Every number
> the code prints carries a "pre-fix, not quotable" label until both are fixed.

---

## Install (from nothing)

You need about **3 GB of free disk** and an internet connection. No administrator rights, no Python,
no git and no graphics card are needed. The installer brings everything it uses.

### 1. Get the code

- **With git:** `git clone https://github.com/garyhost2/pump-safe-marl.git`
- **Without git:** on the GitHub page click **Code → Download ZIP**, then unzip it.

The repository is private: you must be signed in to GitHub with an account that has been given access.

### 2. Run the installer

**Linux or macOS.** Open a terminal in the folder and run:

```bash
./install.sh
```

**Windows.** Open PowerShell in the folder (in Explorer: Shift + right-click → *Open PowerShell window
here*) and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

The installer does four things, and prints each step:

| step | what happens | first time |
|---|---|---|
| 1 | installs [uv](https://docs.astral.sh/uv/), a small tool that manages Python, into your home folder | seconds |
| 2 | uv installs its own Python 3.11 and the exact package versions in `uv.lock` into `.venv/` | 2–5 min |
| 3 | downloads the public data and builds every file the code reads into `data/` | 2–4 min |
| 4 | runs a quick check | 1 min |

Nothing is installed outside this folder and your home folder's `.local/bin` (`uv`) and uv's cache.
To skip the data download, run `./install.sh --no-data` and fetch it later with `./run.sh data`.

### 3. Check it works

```bash
./run.sh smoke
```

This runs the eight houses in Karlstad for one week with a rule-based controller behind the safety
filter, and prints the electricity use, the cost and the number of limit violations it cannot explain
(it should be 0). It takes about 10 seconds.

On Windows, use `.\run.ps1` instead of `./run.sh` everywhere.

---

## Everyday commands

| command | what it does | time |
|---|---|---|
| `./run.sh help` | list the commands | – |
| `./run.sh check` | check the installation | 1 min |
| `./run.sh smoke` | one short heat-pump run | 10 s |
| `./run.sh test` | the full test suite (944 tests) | 5–10 min |
| `./run.sh data` | download and build the data, skipping what is already correct | 2–4 min |
| `./run.sh data --check` | only verify the data on disk against the reference hashes | seconds |
| `./run.sh run <script> [args]` | run any script of the repository | – |

---

## Running the experiments

Every experiment is a script under `experiments/`. Run any of them through `./run.sh run`, which puts
the repository on Python's path and uses the installed environment. Add `--help` to see a script's
options.

**Short runs of the heat-pump plant** (one controller, one site-year window, one hygiene rule):

```bash
./run.sh run experiments/paper_b/run.py --specs '[{"name":"rule_se4","set":"karlstad:2025@SE4","start":"monday","hours":672,"ctl":"reactive","regime":"weekly_bottom20_60_1h"}]' --out outputs/paper_b --workers 4
```

- `set` is `<site>:<year>` with site `karlstad` or `lulea` and year 2022–2025, optionally `@SE1`…`@SE4`
  to price it in another Swedish bidding zone.
- `ctl` is a controller: `reactive`, `price_aware`, `preheat`, `mpc_band`, `thermia`, `thermia_weekly`, `null`.
- `regime` is a hygiene rule: `none`, `weekly_bottom20_60_1h`, `daily_whole_60_1h` or `w551_70_3min`.

**Baselines** (rules and MPC behind the same filter):

```bash
./run.sh run experiments/dhw_v2_rl/baselines_v3.py --sets karlstad:2025 lulea:2025 --out outputs/baselines --workers 4
```

**Multi-agent reinforcement learning** (one agent per house, shared network):

```bash
./run.sh run experiments/dhw_v2_rl/train_v3.py --variant A --seed 0 --steps 500000 --out outputs/rl_v3
./run.sh run experiments/dhw_v2_rl/v4.py --help
```

**Certified-test check** of the heat-pump and tank model (EN 16147 tapping cycle and standby):

```bash
./run.sh run experiments/paper_b/en16147_check.py --out outputs/en16147.json
```

A whole year for eight houses takes a few minutes per controller on one core; training runs take
hours. `--workers` sets how many runs go in parallel (each needs about 1–2 GB of memory).

---

## What is in the repository

| path | content |
|---|---|
| `stems/dhw_v2/` | the heat-pump plant: RXT 12 performance map, stratified tank, borehole, house thermal model, safety filter, controllers, hygiene rules |
| `stems/dhw_v2_rl/` | multi-agent learners (PPO, SAC, behaviour cloning, DAgger, IQL) and the MPC baselines |
| `stems/thermia_rxt12.py` | the heat-pump map fitted to the 21 certified KEYMARK test points |
| `stems/` (rest) | the CityLearn-based district setting: environment wrapper, batteries, EV fleet, district filter, MAPPO, rewards, metrics |
| `experiments/` | one script per experiment and the scripts that make tables and figures |
| `tests/` | the test suite |
| `data_pipeline/` | downloads the public data and rebuilds every data file, with hash checks |
| `tools/citylearn_setup/` | builds the CityLearn schemas (run automatically by the data step) |
| `citylearn_schemas/` | EV charger schedules and sizing data; the schema files are generated |
| `docs/methods/` | how the plant, heat-pump map, filter and its guarantees work |
| `docs/DATA.md` | every data source, its licence, and how it is built |
| `docs/archive/` | earlier reports and findings, kept for reference |
| `cli.py`, `run.sh`, `run.ps1`, `install.sh`, `install.ps1` | the commands above |

The code has no comments by design; the explanations are in `docs/`.

---

## Data

The repository stores no data. `./run.sh data` downloads it from the original public sources and
checks every file against the hashes of the copy the results were produced on:

- **Weather:** SMHI open data (CC BY 4.0), Karlstad and Luleå, 2020–2025.
- **Electricity prices:** day-ahead spot prices for SE1–SE4 from elprisetjustnu.se (ENTSO-E data).
- **Houses:** eight real single-family houses from Boverket's BETSI survey.
- **Hot-water draws:** generated with OpenDHW (a Python port of DHWcalc).
- **CityLearn datasets:** Travis County (Texas) and the 2022 challenge, from the pinned CityLearn release.

Details, licences and what can differ when a source revises its history are in [docs/DATA.md](docs/DATA.md).
The data goes into `data/`; set the environment variable `THERMIA_DATA` to put it somewhere else.

---

## CityLearn

The district parts run on CityLearn 3.0.2 from
[garyhost2/CityLearn](https://github.com/garyhost2/CityLearn) at commit `7f62933`, which adds the fix
that writes the simulated indoor temperature to the observed time step. The installer gets it
automatically. What changed when moving from CityLearn 2.6 is in
[docs/CITYLEARN3_PORT.md](docs/CITYLEARN3_PORT.md).

CityLearn 3 changed the battery and EV-charger physics, so results of the CityLearn-based (Texas and
California) experiments differ from runs made on CityLearn 2.6: re-run them here rather than compare
with older numbers. The Swedish heat-pump simulator does not use CityLearn and is not affected.

---

## Troubleshooting

- **`./install.sh: Permission denied`:** run `sh install.sh` instead.
- **Windows says scripts are disabled:** use the `powershell -ExecutionPolicy Bypass -File .\install.ps1`
  line above; it changes nothing on your system.
- **A data source is unreachable:** the data step stops, prints the exact URL and where to save the
  file by hand; run `./run.sh data` again afterwards.
- **Start again from scratch:** delete the `.venv/` and `data/` folders and run the installer again.
- **Behind a proxy:** set `HTTPS_PROXY` before running the installer.

---

## Licence

MIT, see [LICENSE](LICENSE). The data keeps the licences of its sources (see [docs/DATA.md](docs/DATA.md)).
