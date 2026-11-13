import gzip
import hashlib
import shutil
import time
from pathlib import Path

import requests

from data_pipeline.sources import USER_AGENT
from stems import paths

REPO = paths.REPO
DATA = paths.DATA
SWEDEN = paths.SWEDEN
SWEDEN_RAW = paths.SWEDEN_RAW
SWEDEN_PACKAGES = paths.SWEDEN_PACKAGES
CITYLEARN_DATA = paths.CITYLEARN_DATA
CITYLEARN_DATASETS_DIR = CITYLEARN_DATA / "datasets"
SOURCES = SWEDEN_RAW / "sources"
SPOT_RAW = SOURCES / "spot_raw"
SMHI_RAW = SWEDEN / "downloads" / "smhi"
STATIC = Path(__file__).resolve().parent / "static"


class ManualStepRequired(RuntimeError):
    pass


def log(*parts):
    print("[data]", *parts, flush=True)


def session():
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def download(url, dest, manual_hint=None, tries=4, timeout=120):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last = None
    for k in range(tries):
        try:
            with session().get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.replace(dest)
            return dest
        except Exception as exc:
            last = exc
            time.sleep(3 * (k + 1))
    if tmp.exists():
        tmp.unlink()
    lines = [f"Could not download {url}", f"  error: {last!r}", f"  save the file by hand as: {dest}"]
    if manual_hint:
        lines.append(f"  the file is linked from: {manual_hint}")
    lines.append("  then run the pipeline again.")
    raise ManualStepRequired("\n".join(lines))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_gz_content(path):
    h = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_static(name, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(STATIC / name, dest)
