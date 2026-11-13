#!/usr/bin/env python
import sys

import numpy as np
from citylearn.citylearn import CityLearnEnv

OBSERVATION = "indoor_dry_bulb_temperature"
OFFSETS = (-2, -1, 0, 1)


def actions(env, level):
    return [[level if n == "cooling_device" else
             -level if n == "cooling_or_heating_device" else 0.0
             for n in b.active_actions] for b in env.buildings]


def main(dataset, steps, level):
    env = CityLearnEnv(dataset, central_agent=False, random_seed=0)
    names = [[n for n, on in b.observation_metadata.items() if on] for b in env.buildings]
    env.reset()
    hits, compared = {}, 0
    for _ in range(steps):
        obs = env.step(actions(env, level))[0]
        for i, b in enumerate(env.buildings):
            if not getattr(b, "simulate_dynamics", False) or OBSERVATION not in names[i]:
                continue
            o = float(obs[i][names[i].index(OBSERVATION)])
            compared += 1
            es = b.energy_simulation
            for lab, series in (("simulated", np.asarray(b.indoor_dry_bulb_temperature)),
                                ("dataset", np.asarray(es.indoor_dry_bulb_temperature)),
                                ("cooling set point",
                                 np.asarray(es.indoor_dry_bulb_temperature_cooling_set_point))):
                for off in OFFSETS:
                    j = b.time_step + off
                    if 0 <= j < len(series) and abs(o - float(series[j])) < 1e-6:
                        hits[(lab, off)] = hits.get((lab, off), 0) + 1
    print(f"{dataset}  level={level}  controlled building-steps compared: {compared}")
    if not compared:
        print("  no building reached simulate_dynamics; raise STEPS")
        return 2
    if not hits:
        print("  the observation matches no series at any tested index")
    for (lab, off), c in sorted(hits.items(), key=lambda kv: -kv[1]):
        print(f"  observation == {lab} series at time_step{off:+d}: {c / compared:.2f}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    sys.exit(main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3])))
