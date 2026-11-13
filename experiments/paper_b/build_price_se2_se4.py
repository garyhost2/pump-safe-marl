import json
import shutil
import sys
import tempfile
from pathlib import Path

pkg = Path(sys.argv[1])
src = (pkg / "scripts" / "build_price.py").read_text()
old = 'ZONES = {"SE3": dict(site="karlstad", reduced_tax=False), "SE1": dict(site="lulea", reduced_tax=True)}'
assert old in src
src = src.replace(old, 'ZONES = {"SE2": dict(site=None, reduced_tax=True), "SE4": dict(site=None, reduced_tax=False)}')
tmp = Path(tempfile.mkdtemp(prefix="price_se2_se4_"))
sys.argv = ["build_price.py", str(pkg / "sources" / "spot_raw"), str(tmp)]
exec(compile(src, "build_price.py", "exec"), {"__name__": "__main__"})

qc_path = pkg / "price" / "price_qc.json"
qc = json.loads(qc_path.read_text())
new = json.loads((tmp / "price_qc.json").read_text())
import pandas as pd
for z in ("SE2", "SE4"):
    raw = pd.read_csv(pkg / "sources" / "spot_raw" / f"spot_{z}_raw.csv.gz")
    per_day = raw.groupby(raw.time_start.str[:10]).size()
    dst = {"03-27", "03-26", "03-31", "03-30"}
    short = [d for d in per_day[per_day == 23].index if d[5:] not in dst]
    px = pd.read_csv(tmp / f"price_{z}.csv")
    new[z]["missing_hours_utc"] = px.time_utc[px.spot_SEK_per_kWh.isna()].tolist()
    new[z]["provider_short_days_non_dst"] = short
    new[z]["note"] = ("fetched 2022-01-01..2025-12-31 local days, so 2025-12-31T23:00Z is not covered "
                      "(SE1/SE3 were fetched one day further)")
    new[z]["tariff_assumption"] = ("reduced (northern) energy tax" if z == "SE2" else "standard energy tax") + \
        "; network tariff and markup as SE1/SE3 (Vattenfall T4 2025 list); no site attached"
    qc[z] = new[z]
    shutil.move(str(tmp / f"price_{z}.csv"), str(pkg / "price" / f"price_{z}.csv"))
qc_path.write_text(json.dumps(qc, indent=1))
shutil.rmtree(tmp)
print("price_qc.json zones:", sorted(qc))
