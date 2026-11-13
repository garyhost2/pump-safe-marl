import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("THERMIA_DATA", REPO / "data")).expanduser().resolve()
SWEDEN = DATA / "sweden"
SWEDEN_RAW = SWEDEN / "stems_sweden_data_v1"
SWEDEN_PACKAGES = SWEDEN / "packages"
CITYLEARN_DATA = DATA / "citylearn"
SCHEMAS = REPO / "citylearn_schemas"
