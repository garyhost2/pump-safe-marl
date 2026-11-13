from citylearn.citylearn import CityLearnEnv
import citylearn, numpy as np
print("citylearn", citylearn.__version__)
env = CityLearnEnv("citylearn_challenge_2023_phase_2_local_evaluation", central_agent=False)
env.reset()
b = env.buildings[0]
dev = b.cooling_device
print(f"building {b.name}, cooling_device nominal={dev.nominal_power:.4f} kW")
print(f"  AFTER reset(), before step: consumption[0]={dev.electricity_consumption[0]:.4f}  "
      f"available_nominal_power={dev.available_nominal_power:.4f}")
seen = {}
orig = type(b)._ops_service.__get__(b).apply_actions if False else None
import citylearn.building as B
real_update = B.Building.update_energy_from_cooling_device
def spy(self):
    if self.time_step == 0 and self.name == b.name:
        seen['avail_at_first_action'] = self.cooling_device.available_nominal_power
        seen['consumption_at_first_action'] = self.cooling_device.electricity_consumption[0]
    return real_update(self)
B.Building.update_energy_from_cooling_device = spy
acts = [[0.0]*len(bb.active_actions) for bb in env.buildings]
env.step(acts)
B.Building.update_energy_from_cooling_device = real_update
print(f"  AT the first action:        consumption[0]={seen.get('consumption_at_first_action')}  "
      f"available_nominal_power={seen.get('avail_at_first_action')}")
