import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from data_pipeline import build_ground, build_houses, build_price, build_weather, convert_sweden, gen_draws
from data_pipeline import citylearn_data, fetch_sources, fetch_spot
from data_pipeline.common import (CITYLEARN_DATASETS_DIR, DATA, SWEDEN_RAW, ManualStepRequired,
                                  copy_static, log)
from data_pipeline.compare import compare, digest, load_expected, locate, print_table
from data_pipeline.sources import CITYLEARN_DATASETS
from stems.paths import SCHEMAS

STATE_FILE = DATA / ".pipeline_state.json"
V1 = "sweden/stems_sweden_data_v1/"
PACKAGES = "sweden/packages/"


def v1_keys(module):
    return [V1 + k for k in module.outputs()]


def readme_step():
    copy_static("stems_sweden_data_v1_README.md", SWEDEN_RAW / "README.md")


def schema_keys():
    return [k for k in load_expected()["files"] if k.startswith("schemas/")]


def steps(schemas_dir, with_validation):
    sweden = [
        ("fetch SMHI and Boverket sources", None, fetch_sources.run),
        ("fetch spot prices (elprisetjustnu.se)", None, fetch_spot.run),
        ("build weather", v1_keys(build_weather), build_weather.run),
        ("build price", v1_keys(build_price), build_price.run),
        ("build houses", v1_keys(build_houses), build_houses.run),
        ("build ground", v1_keys(build_ground), build_ground.run),
        ("generate hot-water draws", v1_keys(gen_draws), gen_draws.run),
        ("copy package README", [V1 + "README.md"], readme_step),
        ("convert to dhw-v2 packages", [PACKAGES + k for k in convert_sweden.outputs()
                                        if with_validation or k != "validation.json"],
         lambda: convert_sweden.run(with_validation=with_validation)),
    ]
    citylearn = [
        ("fetch CityLearn datasets", "datasets", citylearn_data.fetch_datasets),
        ("generate CityLearn schemas", schema_keys(), lambda: citylearn_data.build_schemas(schemas_dir)),
    ]
    return {"sweden": sweden, "citylearn": citylearn}


def load_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1))


def code_version():
    h = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parent.glob("*.py")) + sorted(citylearn_data.SETUP_DIR.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def up_to_date(name, keys, schemas_dir, state):
    if keys is None:
        return False
    if keys == "datasets":
        return all((CITYLEARN_DATASETS_DIR / n / "schema.json").is_file() for n in CITYLEARN_DATASETS)
    expected = load_expected()["files"]
    entry = state.get(name, {})
    recorded = entry.get("files", {}) if entry.get("code") == code_version() else {}
    for key in keys:
        path = locate(key, DATA, schemas_dir)
        if not path.is_file():
            return False
        d = digest(key, path)
        if d != expected.get(key, {}).get("sha256") and d != recorded.get(key):
            return False
    return True


def snapshot(keys, schemas_dir):
    if keys == "datasets":
        return {n: (CITYLEARN_DATASETS_DIR / n / "schema.json").is_file() for n in CITYLEARN_DATASETS}
    out = {}
    for key in keys:
        path = locate(key, DATA, schemas_dir)
        out[key] = digest(key, path) if path.is_file() else None
    return out


def record(name, keys, schemas_dir, state):
    if not isinstance(keys, list):
        return
    files = {k: digest(k, locate(k, DATA, schemas_dir)) for k in keys if locate(k, DATA, schemas_dir).is_file()}
    state[name] = {"code": code_version(), "files": files}
    save_state(state)


def keys_for(only):
    keys = list(load_expected()["files"])
    if only == "sweden":
        return {k for k in keys if k.startswith("sweden/")}
    if only == "citylearn":
        return {k for k in keys if not k.startswith("sweden/")}
    return set(keys)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m data_pipeline",
                                 description="Fetch and build the data this repository needs, then compare it with the reference hashes")
    ap.add_argument("--only", choices=["sweden", "citylearn"], help="run one half of the pipeline")
    ap.add_argument("--check", action="store_true", help="only compare the files on disk with data_pipeline/expected_hashes.json")
    ap.add_argument("--refresh", action="store_true", help="download the sources again even when a cached copy exists")
    ap.add_argument("--force", action="store_true", help="rebuild every step even when its outputs already match")
    ap.add_argument("--no-validation", action="store_true", help="skip the dhw-v2 package validation run (validation.json)")
    ap.add_argument("--schemas", default=str(SCHEMAS), help="where the CityLearn schemas are written (default: %(default)s)")
    ap.add_argument("--reference", default=os.environ.get("THERMIA_REFERENCE_DATA"),
                    help="a reference data tree (THERMIA_DATA layout plus schemas/); when given, files that differ are diffed against it")
    a = ap.parse_args(argv)
    schemas_dir = Path(a.schemas).expanduser().resolve()
    log("THERMIA_DATA =", DATA)
    log("schemas      =", schemas_dir)
    t0 = time.time()
    failures = []
    if not a.check:
        plan = steps(schemas_dir, not a.no_validation)
        halves = [a.only] if a.only else ["sweden", "citylearn"]
        state = load_state()
        for half in halves:
            upstream_changed = False
            for name, keys, fn in plan[half]:
                is_fetch = keys is None
                if not is_fetch and not a.force and not upstream_changed and up_to_date(name, keys, schemas_dir, state):
                    log(f"skip  {name} (outputs present and unchanged)")
                    continue
                before = None if is_fetch else snapshot(keys, schemas_dir)
                log(f"start {name}")
                t = time.time()
                try:
                    result = fn(refresh=a.refresh) if is_fetch else fn()
                except ManualStepRequired as exc:
                    failures.append((name, str(exc)))
                    log(f"STOP  {name}: manual step required")
                    break
                if is_fetch:
                    upstream_changed = upstream_changed or bool(result)
                else:
                    record(name, keys, schemas_dir, state)
                    upstream_changed = upstream_changed or snapshot(keys, schemas_dir) != before
                log(f"done  {name} in {time.time() - t:.0f} s")
    rows = compare(DATA, schemas_dir, keys_for(a.only), Path(a.reference) if a.reference else None)
    total = print_table(rows)
    log(f"finished in {time.time() - t0:.0f} s")
    for name, text in failures:
        print(f"\nMANUAL STEP NEEDED ({name}):\n{text}\n", file=sys.stderr)
    if failures or total["missing"]:
        return 1
    return 0
