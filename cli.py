import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
os.environ.setdefault("PYTHONPATH", str(REPO))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

HELP = """
Usage:  ./run.sh <command>        (Windows: .\\run.ps1 <command>)

  check                 quick check that the installation works (about 1 minute)
  smoke                 one short heat-pump run: 8 houses, Karlstad, one week (about 2 minutes)
  test [pytest args]    the full test suite (about 20-40 minutes)
  data [--check]        download and build the data, or only verify it
  run <script> [args]   run any script of the repository, e.g.
                        ./run.sh run experiments/paper_b/run.py --help
  help                  this text
"""


def check():
    import numpy
    import torch
    import citylearn
    from stems.paths import DATA, SWEDEN_PACKAGES, CITYLEARN_DATA
    from stems.thermia_rxt12 import CalibraRXT12
    print(f"Python {sys.version.split()[0]}  numpy {numpy.__version__}  torch {torch.__version__}  CityLearn {citylearn.__version__}")
    hp = CalibraRXT12()
    print(f"Heat-pump map loaded: {type(hp).__name__}")
    print(f"Data folder: {DATA}")
    for name, p in (("Swedish site-years", SWEDEN_PACKAGES), ("CityLearn datasets", CITYLEARN_DATA)):
        print(f"  {name:20s} {'found' if p.is_dir() else 'MISSING (run ./run.sh data)'}  {p}")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/test_thermia_rxt12.py", "tests/test_dhw_v2.py"], cwd=REPO)
    if r.returncode != 0:
        sys.exit(r.returncode)
    print("\nInstallation OK.")


def smoke():
    out = REPO / "outputs" / "smoke"
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_reactive.json").unlink(missing_ok=True)
    spec =[{"name": "smoke_reactive", "set": "karlstad:2025", "start": "monday", "hours": 168, "ctl": "reactive"}]
    t0 = time.time()
    r = subprocess.run([sys.executable, "experiments/paper_b/run.py", "--specs", json.dumps(spec),
                        "--out", str(out), "--workers", "1"], cwd=REPO)
    if r.returncode != 0:
        sys.exit(r.returncode)
    res = json.loads((out / "smoke_reactive.json").read_text())
    s = res["summary"]
    m = res["metrics"]
    print("\nSmoke run: 8 houses, Karlstad 2025, one week, rule-based controller behind the safety filter")
    print(f"  electricity      {sum(s['elec_kwh']):8.1f} kWh")
    print(f"  cost             {sum(s['cost']):8.1f} {res['currency']}")
    print(f"  coldest indoor   {min(s['t_in_min_c']):8.1f} C")
    print(f"  unexplained      {int(m.get('violations_unexplained') or 0)} limit violations")
    print(f"  (pre-fix numbers, not for quoting)")
    print(f"\nFinished in {time.time() - t0:.0f} s. Full record: {out / 'smoke_reactive.json'}")


def main(argv):
    cmd = argv[0] if argv else "help"
    rest = argv[1:]
    if cmd in ("help", "-h", "--help"):
        print(HELP)
    elif cmd == "check":
        check()
    elif cmd == "smoke":
        smoke()
    elif cmd == "test":
        sys.exit(subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *rest], cwd=REPO).returncode)
    elif cmd == "data":
        sys.exit(subprocess.run([sys.executable, "-m", "data_pipeline", *rest], cwd=REPO).returncode)
    elif cmd == "run":
        if not rest:
            print("Usage: ./run.sh run <script.py> [args]")
            sys.exit(2)
        script = str((REPO / rest[0]).resolve())
        sys.argv = [script, *rest[1:]]
        runpy.run_path(script, run_name="__main__")
    else:
        print(f"Unknown command: {cmd}\n{HELP}")
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
