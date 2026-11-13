import argparse
import datetime as dt
from pathlib import Path

from data_pipeline.compare import EXPECTED_FILE, write_expected
from data_pipeline.sources import CITYLEARN_DATASETS

V1 = "sweden/stems_sweden_data_v1"
V1_DIRS = ["weather", "price", "houses", "ground", "draws"]
SOURCE_FILES = ["betsi-v2.zip", "beskrivning-_betsi.pdf", "dvut_1991_2020.csv"]
SCHEMA_DIRS = ["tx_travis_8b", "tx_travis_8b_ev"]


def files_under(root, rel):
    base = Path(root) / rel
    return sorted(str(p.relative_to(root)) for p in base.rglob("*") if p.is_file() or (p.is_symlink() and p.resolve().is_file()))


def collect(root):
    root = Path(root)
    groups = {}
    groups["sweden_v1"] = [k for d in V1_DIRS for k in files_under(root, f"{V1}/{d}")] + [f"{V1}/README.md"]
    groups["sweden_sources"] = [f"{V1}/sources/{n}" for n in SOURCE_FILES] + files_under(root, f"{V1}/sources/spot_raw")
    groups["sweden_packages"] = files_under(root, "sweden/packages")
    groups["citylearn_datasets"] = [k for n in CITYLEARN_DATASETS for k in files_under(root, f"citylearn/datasets/{n}")]
    groups["citylearn_schemas"] = ["schemas/" + str(Path(k).relative_to("schemas")) for d in SCHEMA_DIRS
                                   for k in files_under(root, f"schemas/{d}")]
    return groups


def main():
    ap = argparse.ArgumentParser(description="Write data_pipeline/expected_hashes.json from a reference data tree")
    ap.add_argument("reference", help="folder laid out like THERMIA_DATA, plus schemas/ for the CityLearn schemas")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    note = a.note or f"computed {dt.date.today().isoformat()} from the reference data tree"
    n = write_expected(Path(a.reference), collect(a.reference), note)
    print(f"wrote {n} hashes to {EXPECTED_FILE}")


if __name__ == "__main__":
    main()
