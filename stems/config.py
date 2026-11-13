from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

BASE_CONSTRAINT_CHANNELS: Tuple[str, ...] = (
    "soc_band",
    "building_power_cap",
    "district_import_cap",
)

DEGRADATION_CONSTRAINT_CHANNEL = "battery_degradation"


def constraint_channel_names(config: "STEMSConfig") -> Tuple[str, ...]:
    names = list(BASE_CONSTRAINT_CHANNELS)
    if (config.degradation.mode != "none"
            and config.degradation.limit_kwh_per_episode is not None):
        names.append(DEGRADATION_CONSTRAINT_CHANNEL)
    return tuple(names)


@dataclass
class GraphConfig:

    mode: str = "feature"
    alpha: float = 0.5
    beta: float = 0.5
    sigma_d: float = 1.0
    sigma_f: Optional[float] = None


@dataclass
class GCNConfig:
    num_layers: int = 3
    hidden_dim: int = 64


@dataclass
class TransformerConfig:
    num_heads: int = 4
    embed_dim: int = 32
    window_size: int = 24


@dataclass
class FusionConfig:
    output_dim: int = 64


@dataclass
class ActorCriticConfig:
    hidden_dim: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    share_parameters: bool = False
    policy_reg: float = 0.0
    metaems_inner_steps: int = 8
    device: str = "cpu"
    stems_clip_eps: Optional[float] = None
    stems_epochs: int = 1
    sac_scale_returns: bool = False


@dataclass
class RewardConfig:
    mu: float = 1.0
    alpha_grid: float = 0.5
    alpha_build: float = 0.3
    beta_ramp: float = 0.2
    lambda_indoor: float = 0.4
    xi: float = 0.0
    T_ref: float = 22.0
    T_comfort_threshold: float = 2.0

    ev_service: float = 25.0
    ev_shaping: float = 0.5

    scale_grid_kw: Optional[float] = None
    scale_building_kw: Optional[float] = None

    def resolve_scales(self, p_grid_max: float, p_building_max: float) -> Tuple[float, float]:
        grid = self.scale_grid_kw if self.scale_grid_kw is not None else p_grid_max
        building = (self.scale_building_kw if self.scale_building_kw is not None
                    else p_building_max)
        if not grid > 0.0 or not building > 0.0:
            raise ValueError(f"reward scales must be positive, got grid={grid}, "
                             f"building={building}")
        return float(grid), float(building)


@dataclass
class CBFConfig:
    SOC_min: float = 0.1
    SOC_max: float = 0.9
    P_grid_max: float = 300.0
    P_building_max: float = 80.0
    gamma_cbf: float = 1.0


@dataclass
class SafetyConfig:
    feasibility_qp: bool = True
    anticipatory: bool = True
    invariance_horizon: int = 1
    robust_margins: bool = True
    soc_margin: float = 0.03
    soc_tolerance: float = 1e-3
    power_derate: float = 0.05


@dataclass
class TrainingConfig:
    episodes: int = 50
    batch_size: int = 512
    buffer_capacity: int = 100_000
    exploration_noise: float = 0.1
    action_scale: float = 1.0

    update_epochs: int = 10
    minibatch_size: int = 64
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.02
    entropy_coef: float = 0.01
    scale_rewards: bool = True

    actor_target: str = "raw"

    residual_scale: float = 0.5
    residual_log_std: float = -1.2

    intervention_penalty: float = 0.0

    forced_charge_penalty: float = 0.0

    bc_episodes: int = 0
    bc_epochs: int = 0
    bc_lr: float = 1e-3
    bc_batch: int = 256
    bc_reset_log_std: Optional[float] = -1.0


@dataclass
class LagrangianConfig:

    num_constraints: int = 3
    cost_limit: float = 0.05
    lambda_init: float = 0.1
    lambda_max: float = 10.0

    lambda_lr: float = 0.005

    use_pid: bool = True
    pid_kp: float = 1.0
    pid_ki: float = 0.3
    pid_kd: float = 0.0

    enabled: bool = True
    algorithm: str = "ppo-lagrangian"
    focops_temperature: float = 1.5
    focops_kl_limit: Optional[float] = None


@dataclass
class HeatPumpConfig:
    enabled: bool = False
    cop_heating_mild: float = 3.5
    cop_heating_cold: float = 2.0
    cop_cooling: float = 4.0
    cold_snap_temp: float = 0.0
    rated_power_kw: float = 5.0
    heating_setpoint: float = 20.0
    cooling_setpoint: float = 22.0


@dataclass
class ThermalConfig:
    dhw_readiness: bool = True
    preheat_horizon: int = 2
    dhw_margin: float = 0.05
    dhw_soc_cap: float = 0.95

    forecast_alpha: float = 0.2
    forecast_warmup: int = 24
    forecast_temp_gain: float = 0.02

    weather_anticipation: bool = True
    weather_gain: float = 0.15
    weather_horizon: int = 3

    cop_aware_power: bool = True


@dataclass
class ComfortConfig:

    enabled: bool = False
    tolerance_k: float = 2.0
    occupied_only: bool = True
    enforce_cooling: bool = True
    design_t_out_heating_c: float = -10.0
    design_t_out_cooling_c: float = 35.0


@dataclass
class DegradationConfig:

    mode: str = "none"
    calendar: bool = False
    dod_exponent: float = 0.0
    reversal_threshold: float = 0.05
    price_eur_per_kwh: float = 0.0
    limit_kwh_per_episode: Optional[float] = None
    fallback_capacity_loss_coefficient: float = 1e-5
    calendar_loss_per_second: Optional[float] = None


@dataclass
class STEMSConfig:
    graph: GraphConfig = field(default_factory=GraphConfig)
    gcn: GCNConfig = field(default_factory=GCNConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    actor_critic: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    cbf: CBFConfig = field(default_factory=CBFConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    lagrangian: LagrangianConfig = field(default_factory=LagrangianConfig)
    heat_pump: HeatPumpConfig = field(default_factory=HeatPumpConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    comfort: ComfortConfig = field(default_factory=ComfortConfig)
    degradation: DegradationConfig = field(default_factory=DegradationConfig)

    def tricks_active(self) -> List[str]:
        active = ["feasibility_qp", "real_battery_dynamics"]
        if self.safety.anticipatory:
            active.append("anticipatory")
        if self.safety.robust_margins:
            active.append("robust_margins")
        if self.lagrangian.use_pid:
            active.append("pid_lagrangian")
        if self.thermal.dhw_readiness:
            active.append(f"dhw_readiness(L={self.thermal.preheat_horizon})")
        if self.thermal.weather_anticipation:
            active.append("weather_anticipation")
        if self.thermal.cop_aware_power:
            active.append("cop_aware_power")
        if self.comfort.enabled:
            active.append(f"hard_comfort(theta={self.comfort.tolerance_k}K)")
        if self.degradation.mode != "none":
            active.append(f"degradation({self.degradation.mode}"
                          + (", constrained" if self.degradation.limit_kwh_per_episode
                             is not None else ", reported") + ")")
        return active
