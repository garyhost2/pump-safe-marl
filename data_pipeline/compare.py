import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from data_pipeline.common import sha256_file, sha256_gz_content

EXPECTED_FILE = Path(__file__).resolve().parent / "expected_hashes.json"
SCHEMA_PREFIX = "schemas/"


def load_expected():
    return json.loads(EXPECTED_FILE.read_text())


def locate(key, data_root, schemas_root):
    if key.startswith(SCHEMA_PREFIX):
        return Path(schemas_root) / key[len(SCHEMA_PREFIX):]
    return Path(data_root) / key


def normalise_path_string(value):
    if not value.startswith("/"):
        return value
    if "/datasets/" in value:
        return "{CITYLEARN_DATASETS}/" + value.rsplit("/datasets/", 1)[1]
    parts = value.split("/")
    return "{SCHEMAS}/" + "/".join(parts[-2:])


def normalise_json(obj):
    if isinstance(obj, dict):
        return {k: normalise_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalise_json(v) for v in obj]
    if isinstance(obj, str):
        return normalise_path_string(obj)
    return obj


def normalised_schema_text(path):
    return json.dumps(normalise_json(json.loads(Path(path).read_text())), indent=2)


def digest(key, path):
    if key.startswith(SCHEMA_PREFIX) and key.endswith("schema.json"):
        return hashlib.sha256(normalised_schema_text(path).encode()).hexdigest()
    if key.endswith(".gz"):
        return sha256_gz_content(path)
    return sha256_file(path)


def digest_kind(key):
    if key.startswith(SCHEMA_PREFIX) and key.endswith("schema.json"):
        return "sha256 of the schema with machine paths replaced by {CITYLEARN_DATASETS} and {SCHEMAS}"
    if key.endswith(".gz"):
        return "sha256 of the decompressed content"
    return "sha256 of the file"


def csv_difference(new, ref):
    a = pd.read_csv(new, low_memory=False)
    b = pd.read_csv(ref, low_memory=False)
    if list(a.columns) != list(b.columns):
        return f"columns differ: {sorted(set(a.columns) ^ set(b.columns))}"
    notes = []
    if len(a) != len(b):
        notes.append(f"rows {len(a)} vs reference {len(b)}")
        n = min(len(a), len(b))
        a, b = a.iloc[:n], b.iloc[:n]
    worst, worst_col, rows = 0.0, None, np.zeros(len(a), dtype=bool)
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            xv, yv = x.to_numpy(float), y.to_numpy(float)
            nan_mismatch = np.isnan(xv) != np.isnan(yv)
            d = np.where(np.isnan(xv) | np.isnan(yv), 0.0, np.abs(xv - yv))
            rows |= (d > 0) | nan_mismatch
            if nan_mismatch.any():
                notes.append(f"{int(nan_mismatch.sum())} NaN mismatches in {c}")
            if d.max(initial=0.0) > worst:
                worst, worst_col = float(d.max()), c
        else:
            neq = x.astype(str).to_numpy() != y.astype(str).to_numpy()
            rows |= neq
            if neq.any():
                notes.append(f"{int(neq.sum())} text cells differ in {c}")
    head = f"max |diff| {worst:.6g}" + (f" ({worst_col})" if worst_col else "") + f", {int(rows.sum())} rows differ"
    return "; ".join([head] + notes)


def json_difference(new, ref, is_schema):
    load = (lambda p: normalise_json(json.loads(Path(p).read_text()))) if is_schema else (lambda p: json.loads(Path(p).read_text()))
    a, b = load(new), load(ref)
    leaves, worst = [], 0.0

    def walk(x, y, where):
        nonlocal worst
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                if k not in x or k not in y:
                    leaves.append(f"{where}/{k}")
                else:
                    walk(x[k], y[k], f"{where}/{k}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{where}[{i}]")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if x != y:
                leaves.append(where)
                worst = max(worst, abs(x - y))
        elif x != y:
            leaves.append(where)

    walk(a, b, "")
    text = f"{len(leaves)} values differ"
    if worst:
        text += f", max numeric |diff| {worst:.6g}"
    if leaves:
        text += f", first: {leaves[0]}"
    return text


def explain(key, new, ref):
    try:
        if key.endswith(".csv") or key.endswith(".csv.gz"):
            return csv_difference(new, ref)
        if key.endswith(".json"):
            return json_difference(new, ref, key.startswith(SCHEMA_PREFIX))
    except Exception as exc:
        return f"could not diff: {exc!r}"
    return ""


def compare(data_root, schemas_root, keys=None, reference=None):
    expected = load_expected()["files"]
    rows = []
    for key, info in expected.items():
        if keys is not None and key not in keys:
            continue
        path = locate(key, data_root, schemas_root)
        if not path.is_file():
            rows.append((key, "missing", ""))
            continue
        if digest(key, path) == info["sha256"]:
            rows.append((key, "identical", ""))
            continue
        detail = ""
        if reference is not None:
            ref = locate(key, Path(reference), Path(reference) / "schemas")
            detail = explain(key, path, ref) if ref.is_file() else "(no reference copy to diff against)"
        rows.append((key, "different", detail))
    return rows


def group_of(key):
    parts = key.split("/")
    if key.startswith("sweden/packages/"):
        return "sweden/packages/" + ("validation.json" if parts[2] == "validation.json" else parts[2])
    if key.startswith("citylearn/datasets/"):
        return "/".join(parts[:3])
    if key.startswith(SCHEMA_PREFIX):
        return "/".join(parts[:2])
    return "/".join(parts[:-1]) or key


def print_table(rows, max_listed=60):
    groups = OrderedDict()
    for key, status, _ in rows:
        g = groups.setdefault(group_of(key), {"identical": 0, "different": 0, "missing": 0})
        g[status] += 1
    width = max([len(g) for g in groups] + [10])
    print()
    print(f"{'group':<{width}}  identical  different  missing")
    for g, c in groups.items():
        print(f"{g:<{width}}  {c['identical']:>9}  {c['different']:>9}  {c['missing']:>7}")
    total = {s: sum(1 for r in rows if r[1] == s) for s in ("identical", "different", "missing")}
    print(f"{'TOTAL':<{width}}  {total['identical']:>9}  {total['different']:>9}  {total['missing']:>7}")
    absent = {g for g, c in groups.items() if c["identical"] + c["different"] == 0}
    odd = [r for r in rows if r[1] == "different" or (r[1] == "missing" and group_of(r[0]) not in absent)]
    if odd:
        print()
        print("files that are not identical to the reference:")
        for key, status, detail in odd[:max_listed]:
            print(f"  {status:<9} {key}" + (f"  [{detail}]" if detail else ""))
        if len(odd) > max_listed:
            print(f"  ... and {len(odd) - max_listed} more")
    print()
    return total


def write_expected(reference_root, keys_by_group, note):
    files = OrderedDict()
    for group, keys in keys_by_group.items():
        for key in keys:
            path = locate(key, reference_root, Path(reference_root) / "schemas")
            files[key] = {"sha256": digest(key, path), "bytes": path.stat().st_size, "group": group}
    kinds = sorted({digest_kind(k) for k in files})
    out = OrderedDict(note=note, digests=kinds, files=files)
    EXPECTED_FILE.write_text(json.dumps(out, indent=1) + "\n")
    return len(files)
