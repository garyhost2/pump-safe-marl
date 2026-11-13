import importlib.util
import shutil
import tarfile
from pathlib import Path, PurePosixPath

from data_pipeline.common import CITYLEARN_DATASETS_DIR, REPO, ManualStepRequired, log, session
from data_pipeline.sources import CITYLEARN_COMMIT, CITYLEARN_DATASETS, CITYLEARN_TARBALL

SETUP_DIR = REPO / "tools" / "citylearn_setup"


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, SETUP_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wanted_member(name):
    parts = PurePosixPath(name).parts
    if len(parts) < 5 or parts[1:3] != ("data", "datasets") or parts[3] not in CITYLEARN_DATASETS:
        return None
    if any(p in ("..", "") for p in parts):
        return None
    return PurePosixPath(*parts[3:])


def fetch_datasets(dest=CITYLEARN_DATASETS_DIR):
    dest = Path(dest)
    staging = dest.parent / "datasets.part"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    log("CityLearn tarball", CITYLEARN_TARBALL, "(streamed, only the datasets are kept)")
    count = 0
    try:
        with session().get(CITYLEARN_TARBALL, stream=True, timeout=300) as r:
            r.raise_for_status()
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
                for member in tar:
                    rel = wanted_member(member.name)
                    if rel is None or not member.isfile():
                        continue
                    target = staging / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, open(target, "wb") as out:
                        shutil.copyfileobj(src, out)
                    count += 1
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise ManualStepRequired(
            f"Could not stream {CITYLEARN_TARBALL}: {exc!r}\n"
            f"  download that file by hand, then extract data/datasets/<name> for {', '.join(CITYLEARN_DATASETS)}\n"
            f"  so that {dest}/<name>/schema.json exists, e.g.\n"
            f"  tar -xzf {CITYLEARN_COMMIT}.tar.gz --strip-components=3 -C {dest} "
            + " ".join(f"CityLearn-{CITYLEARN_COMMIT}/data/datasets/{n}" for n in CITYLEARN_DATASETS))
    missing = [n for n in CITYLEARN_DATASETS if not (staging / n / "schema.json").is_file()]
    if missing:
        shutil.rmtree(staging, ignore_errors=True)
        raise ManualStepRequired(f"The CityLearn tarball did not contain {missing}")
    dest.mkdir(parents=True, exist_ok=True)
    for n in CITYLEARN_DATASETS:
        if (dest / n).exists():
            shutil.rmtree(dest / n)
        (staging / n).rename(dest / n)
    shutil.rmtree(staging)
    log("CityLearn datasets extracted:", count, "files")


def build_schemas(schemas_dir, datasets=CITYLEARN_DATASETS_DIR):
    schemas_dir = Path(schemas_dir)
    base = schemas_dir / "tx_travis_8b" / "schema.json"
    load_tool("setup_citylearn_8b").generate(base, datasets)
    log("schema", base)
    ev_path, _ = load_tool("setup_citylearn_ev").generate(schemas_dir / "tx_travis_8b_ev", base)
    log("schema", ev_path)
