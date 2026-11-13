SMHI_URL = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/{param}/station/{station}/period/corrected-archive/data.csv"

SMHI_STATIONS = [
    (1, 93220, "Karlstad Flygplats, air temperature"),
    (1, 93520, "Sunne A, donor"),
    (1, 92410, "Arvika A, donor"),
    (11, 93235, "Karlstad Sol, global irradiance"),
    (1, 162860, "Lulea-Kallax Flygplats, air temperature"),
    (1, 161940, "Boden, donor"),
    (1, 162790, "Rodkallen A, donor"),
    (11, 162015, "Lulea Sol, global irradiance"),
]

SPOT_URL = "https://www.elprisetjustnu.se/api/v1/prices/{y}/{m:02d}-{d:02d}_{z}.json"

SPOT_SPANS = {
    "SE3": ("2022-01-01", "2026-01-01"),
    "SE1": ("2022-01-01", "2026-01-01"),
    "SE2": ("2022-01-01", "2025-12-31"),
    "SE4": ("2022-01-01", "2025-12-31"),
}

BOVERKET_BETSI_PAGE = "https://www.boverket.se/sv/om-boverket/oppna-data/betsi-oppna-data/databasen-fran-betsi/"
BOVERKET_DVUT_PAGE = "https://www.boverket.se/sv/om-boverket/oppna-data/dvut-1991-2020/"

FILES = {
    "betsi-v2.zip": dict(
        url="https://www.boverket.se/contentassets/bfeba7eb66e24db8aa01ee42b5eafe29/betsi-v2.zip",
        page=BOVERKET_BETSI_PAGE,
        sha256="f9124daf3d25ce3089cfb989efcaa0642b79ed60b60943a197effcc635c0201e",
    ),
    "beskrivning-_betsi.pdf": dict(
        url="https://www.boverket.se/contentassets/bfeba7eb66e24db8aa01ee42b5eafe29/beskrivning-_betsi.pdf",
        page=BOVERKET_BETSI_PAGE,
        sha256="10da932c02bfb4b694f5f0c081107ed288864fb6679ee2f2af104ba09c2f0c67",
    ),
    "dvut_1991_2020.csv": dict(
        url="https://www.boverket.se/contentassets/78ea1170505245b8b105e163b56dff27/dvut_1991_2020.csv",
        page=BOVERKET_DVUT_PAGE,
        sha256="a203c31bebce0527c52f930f93c3a963e2c7d8452b068ee91158c10b9cd68fdc",
    ),
}

CITYLEARN_COMMIT = "7f62933fea157aaed45c90bca43ef72a8aaf99b4"
CITYLEARN_TARBALL = f"https://github.com/garyhost2/CityLearn/archive/{CITYLEARN_COMMIT}.tar.gz"
CITYLEARN_DATASETS = ["tx_travis_county_neighborhood", "citylearn_challenge_2022_phase_all"]

USER_AGENT = "thermia-safe-marl data_pipeline (research; contact via repository)"
