"""GPU-native model fitting and differentiable dynamics for the teacher."""

from .dynamics import MLP, ResidualDynamics
from .gradients import freeze_parameters
from .replay import StartBatch, TransitionReplay, WindowBatch
from .trainer import WMTrainerConfig, WorldModelTrainer

__all__ = ["MLP", "ResidualDynamics", "freeze_parameters", "StartBatch",
           "TransitionReplay", "WindowBatch", "WMTrainerConfig", "WorldModelTrainer"]
