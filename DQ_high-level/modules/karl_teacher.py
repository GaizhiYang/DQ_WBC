"""Original privileged teacher MLPs with physical selected grasp replacing GFM."""
import torch
from torch import nn
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin


def _build(model, output_dim):
    if model.num_observations != 1102 or model.num_actions != 9:
        raise ValueError("KARL teacher expects packed 1102 observations and 9 actions")
    model.feature_encoder = nn.Sequential(nn.Linear(1024, 512), nn.ELU(), nn.Linear(512, 128))
    model.net = nn.Sequential(nn.Linear(model.num_observations - 1024 + 128, 512), nn.ELU(),
                              nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 128), nn.ELU(),
                              nn.Linear(128, output_dim))
    # Retained also in Value for parity with the original implementation;
    # the critic's log_std parameter is unused by the value computation.
    model.log_std_parameter = nn.Parameter(torch.zeros(model.num_actions))
    model.to(model.device)


def _features(model, states):
    encoded = model.feature_encoder(states[..., :1024])
    # Identical backbone ordering: state, previous action, object code, grasp.
    return torch.cat((states[..., 1024:-15], states[..., -9:], encoded,
                      states[..., -15:-9]), -1)


class KarlTeacherPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device="cpu", use_tanh=False,
                 clip_actions=False, deterministic=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, True, -20, 2, "sum",
                              transform_func=torch.distributions.transforms.TanhTransform() if use_tanh else None,
                              deterministic=deterministic)
        _build(self, self.num_actions)

    def compute(self, inputs, role):
        return self.net(_features(self, inputs["states"])), self.log_std_parameter, {}


class KarlTeacherValue(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device="cpu"):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)
        _build(self, 1)

    def compute(self, inputs, role):
        return self.net(_features(self, inputs["states"])), {}
