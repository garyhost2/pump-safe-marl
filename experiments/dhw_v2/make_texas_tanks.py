#!/usr/bin/env python3
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def main():
    from citylearn.citylearn import CityLearnEnv
    schema = REPO / "citylearn_schemas" / "tx_travis_8b" / "schema.json"
    env = CityLearnEnv(str(schema), random_seed=0)
    out = {}
    for b in env.buildings:
        st = b.dhw_storage
        out[b.name] = {"capacity_kwh": float(st.capacity), "loss_coefficient": float(st.loss_coefficient),
                       "source": f"CityLearnEnv({schema.name}, random_seed=0) dhw_storage autosize"}
    p = REPO / "stems" / "data" / "tx_travis_8b_dhw_tanks.json"
    p.write_text(json.dumps(out, indent=1))
    print(p, out)


if __name__ == "__main__":
    main()
