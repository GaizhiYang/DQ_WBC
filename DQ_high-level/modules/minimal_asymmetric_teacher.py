"""A 67-input actor alongside the existing 1102-input privileged teacher critic.

PPO stores the packed teacher observation; only robot state and the selected
grasp enter this actor. The standalone actor accepts those 67 values directly.
"""
import torch
from torch import nn
from skrl.models.torch import GaussianMixin, Model


PACKED_OBS_DIM = 1102
ACTOR_OBS_DIM = 67
ACTOR_INDICES = tuple(range(1030, 1082)) + tuple(range(1093, 1102)) + tuple(range(1087, 1093))


def select_actor_observation(states):
    """Return robot61 followed by grasp6, in the deployment contract order."""
    if states.ndim != 2 or states.shape[-1] != PACKED_OBS_DIM:
        raise ValueError("Minimal asymmetric policy requires packed [batch, 1102] observations")
    return torch.cat((states[:, 1030:1082], states[:, 1093:1102], states[:, 1087:1093]), dim=-1)


class DeploymentActor(nn.Module):
    """Gaussian action means from already standardized [batch,67] inputs."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(67, 512), nn.ELU(),
            nn.Linear(512, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 9),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.dim() != 2 or states.size(-1) != 67:
            raise ValueError("Minimal deployment actor requires [batch, 67] observations")
        return self.net(states)


class MinimalAsymmetricTeacherPolicy(GaussianMixin, Model):
    """Same distribution interface as KarlTeacherPolicy, without its GT inputs."""
    def __init__(self, observation_space, action_space, device="cpu", use_tanh=False,
                 clip_actions=False, deterministic=False):
        Model.__init__(self, observation_space, action_space, device)
        if self.num_observations != PACKED_OBS_DIM or self.num_actions != 9:
            raise ValueError("Minimal asymmetric policy expects packed 1102 observations and 9 actions")
        if use_tanh or clip_actions:
            raise ValueError("Minimal asymmetric policy supports the default unsquashed Gaussian with environment action clipping only")
        GaussianMixin.__init__(
            self, False, True, -20, 2, "sum", deterministic=deterministic,
        )
        self.actor = DeploymentActor()
        self.log_std_parameter = nn.Parameter(torch.zeros(9))
        self.to(self.device)

    def compute(self, inputs, role):
        return self.actor(select_actor_observation(inputs["states"])), self.log_std_parameter, {}
