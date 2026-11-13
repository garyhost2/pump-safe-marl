import sys
from pathlib import Path

pkg = Path(sys.argv[1])
src = (pkg / "scripts" / "fetch_spot.py").read_text()
assert 'ZONES = ["SE3", "SE1"]' in src
src = src.replace('ZONES = ["SE3", "SE1"]', 'ZONES = ["SE2", "SE4"]')
sys.argv = ["fetch_spot.py", str(pkg / "sources" / "spot_raw")]
exec(compile(src, "fetch_spot.py", "exec"), {"__name__": "__main__"})
