from stems.config import STEMSConfig
from stems.deadline import DeadlineRequirement, DeadlineStorageBarrier
from stems.environment import STEMSEnvironment
from stems.flexibility import FlexibilityPortfolio, FlexibleLoad
from stems.protocols import (Environment, EVProvider, PlantModel, PlantProvider,
                             missing_capabilities)
from stems.replay import CSVReplayEnvironment
from stems.graph import BuildingGraph
from stems.legionella import (LegionellaCycleBarrier, LegionellaSpec,
                              LegionellaStack, ShadowTank)
from stems.encoder import STEncoder
from stems.cbf import CBFShield
from stems.reward import STEMSReward
from stems.agent import STEMSAgent
from stems.metrics import MetricsCalculator
from stems.utils import ReplayBuffer, EpisodeBuffer, HistoryBuffer
from stems.forecast import CausalForecaster, OracleForecaster
from stems.mpc import StorageMPC
from stems.mappo import MAPPOCentralisedCritic
from stems.baselines import (
    RuleBasedAgent,
    SingleAgentSAC,
    IndependentSAC,
    DMAPPOAgent,
    MADDPGAgent,
    MARLISAAgent,
    MADCQAgent,
    MetaEMSAgent,
)

CONTROLLERS_REACHABLE_FROM_ARMS = {
    "RuleBasedAgent": ("rbc", "rbc+calibrated"),
    "STEMSAgent": ("rl", "rl+calibrated"),
    "SingleAgentSAC": ("sac",),
    "IndependentSAC": ("isac",),
    "DMAPPOAgent": ("dmappo",),
    "StorageMPC": ("mpc", "mpc-oracle"),
    "MADDPGAgent": ("maddpg",),
    "MARLISAAgent": ("marlisa",),
    "MADCQAgent": ("madcq",),
    "MetaEMSAgent": ("metaems",),
    "MAPPOCentralisedCritic": ("mappo-cc",),
}

FRAMEWORK_EXPORTS = {
    "DeadlineRequirement",
    "DeadlineStorageBarrier",
    "FlexibleLoad",
    "FlexibilityPortfolio",
    "PlantModel",
    "Environment",
    "PlantProvider",
    "EVProvider",
    "CSVReplayEnvironment",
    "LegionellaCycleBarrier",
    "LegionellaSpec",
    "LegionellaStack",
    "ShadowTank",
    "missing_capabilities",
}

__all__ = [
    "STEMSConfig",
    "STEMSEnvironment",
    *sorted(FRAMEWORK_EXPORTS),
    "FRAMEWORK_EXPORTS",
    "BuildingGraph",
    "STEncoder",
    "CBFShield",
    "STEMSReward",
    "STEMSAgent",
    "MetricsCalculator",
    "ReplayBuffer",
    "EpisodeBuffer",
    "HistoryBuffer",
    "CausalForecaster",
    "OracleForecaster",
    "StorageMPC",
    "MAPPOCentralisedCritic",
    "RuleBasedAgent",
    "SingleAgentSAC",
    "IndependentSAC",
    "DMAPPOAgent",
    "MADDPGAgent",
    "MARLISAAgent",
    "MADCQAgent",
    "MetaEMSAgent",
    "CONTROLLERS_REACHABLE_FROM_ARMS",
]
