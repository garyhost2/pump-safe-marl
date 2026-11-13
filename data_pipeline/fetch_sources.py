import datetime as dt

from data_pipeline.common import SMHI_RAW, SOURCES, ManualStepRequired, download, log, sha256_file
from data_pipeline.sources import FILES, SMHI_STATIONS, SMHI_URL


def smhi_path(param, station):
    return SMHI_RAW / f"p{param}_{station}_corrected-archive.csv"


def fetch_smhi(refresh=False):
    lines = [f"# SMHI open data (CC BY 4.0 SE). Downloaded {dt.date.today().isoformat()} (corrected archive)"]
    fetched = False
    for param, station, label in SMHI_STATIONS:
        url = SMHI_URL.format(param=param, station=station)
        dest = smhi_path(param, station)
        lines.append(f"{url}  # {label}")
        if dest.exists() and not refresh:
            continue
        log("SMHI", param, station, label)
        download(url, dest, manual_hint="https://www.smhi.se/data/meteorologi/ladda-ner-meteorologiska-observationer")
        fetched = True
    if fetched:
        SOURCES.mkdir(parents=True, exist_ok=True)
        (SOURCES / "smhi_download_urls.txt").write_text("\n".join(lines) + "\n")
    return fetched


def fetch_boverket(refresh=False):
    problems = []
    fetched = False
    for name, info in FILES.items():
        dest = SOURCES / name
        if dest.exists() and not refresh:
            continue
        log("Boverket", name)
        try:
            download(info["url"], dest, manual_hint=info["page"])
        except ManualStepRequired as exc:
            problems.append(str(exc))
            continue
        fetched = True
        got = sha256_file(dest)
        if got != info["sha256"]:
            log(f"WARNING {name}: sha256 {got} differs from the reference {info['sha256']}; Boverket may have republished it")
    if problems:
        raise ManualStepRequired("\n\n".join(problems))
    return fetched


def run(refresh=False):
    a = fetch_smhi(refresh)
    b = fetch_boverket(refresh)
    return a or b
