"""Reference deterministic decoder with explicit real-next-observation moments.

Network/action-noise semantics follow clab.algorithms.supermimic.networks.Policy.
The empirical-moment equations follow rsl_rl.modules.EmpiricalNormalization
(ETH Zurich/NVIDIA/Preferred Networks; BSD-3-Clause, attribution in licenses).
DQ retains its continuous sigmoid gripper execution contract.
"""

import math
import torch
from torch import nn
from skrl.models.torch import Model

from .normalization import run_mlp


class EmpiricalObservationNorm(nn.Module):
    """No forward-time mutation; statistics change only from real NEXT rows."""

    def __init__(self, size, eps=1e-2):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("_mean", torch.zeros(1, size))
        self.register_buffer("_var", torch.ones(1, size))
        self.register_buffer("_std", torch.ones(1, size))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, value):
        return (value-self._mean)/(self._std+self.eps)

    @torch.no_grad()
    def update(self, value):
        if not self.training or not len(value):
            return
        value = value.detach().float()
        if value.ndim != 2 or value.shape[-1] != self._mean.shape[-1] or not bool(torch.isfinite(value).all()):
            raise ValueError("Actor normalization requires finite real NEXT observations")
        total = self.count + len(value)
        rate = len(value)/total
        var, mean = value.var(0, unbiased=False, keepdim=True), value.mean(0, keepdim=True)
        delta = mean-self._mean
        updated_mean = self._mean+rate*delta
        updated_var = self._var+rate*(var-self._var+delta*(mean-updated_mean))
        if not bool(torch.isfinite(updated_mean).all() & torch.isfinite(updated_var).all()):
            raise FloatingPointError("Actor empirical moments overflowed")
        self._mean.copy_(updated_mean)
        self._var.copy_(updated_var.clamp_min(0))
        self._std.copy_(self._var.sqrt())
        self.count.copy_(total)


class ReferenceTeacherPolicy(Model):
    """Flat deterministic ELU policy; fixed exploration is outside its mean."""

    reference_policy = True
    deterministic = True
    transform_func = None
    _g_clip_actions = False
    _g_clip_log_std = False

    def __init__(self, observation_space, action_space, device, *,
                 hidden=(2048, 1024, 512, 256), amp="bf16", action_noise=.1,
                 running_normalization=True, last_gain=.01):
        Model.__init__(self, observation_space, action_space, device)
        if self.num_actions != 9 or not hidden or min(hidden) < 1:
            raise ValueError("Reference teacher requires nine actions and positive hidden widths")
        if amp not in ("none", "bf16") or not math.isfinite(action_noise) or action_noise < 0:
            raise ValueError("Invalid reference policy precision/exploration")
        if not math.isfinite(last_gain) or last_gain <= 0:
            raise ValueError("Reference policy last_gain must be positive and finite")
        self.amp, self.running_normalization = amp, bool(running_normalization)
        self.hidden, self.last_gain = tuple(hidden), float(last_gain)
        self.obs_norm = EmpiricalObservationNorm(self.num_observations)
        layers, width = [], self.num_observations
        for output in hidden:
            layers.extend((nn.Linear(width, output), nn.ELU()))
            width = output
        layers.append(nn.Linear(width, self.num_actions))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.mul_(self.last_gain)
            self.net[-1].bias.zero_()
        self.register_buffer("exploration_std", torch.full((self.num_actions,), float(action_noise)))
        # Diagnostic interface only. This is a buffer, never an optimized std.
        self.register_buffer("log_std_parameter", torch.full((self.num_actions,), math.log(max(action_noise, 1e-8))))
        self.to(device)

    def forward(self, states):
        inputs = self.obs_norm(states) if self.running_normalization else states
        return run_mlp(self.net, inputs, self.amp)

    def compute(self, inputs, role="policy"):
        return self.forward(inputs["states"]), self.log_std_parameter, {}

    @staticmethod
    def mean_to_action(mean):
        return torch.cat((mean[..., :6], mean[..., 6:7].sigmoid(), mean[..., 7:]), -1)

    def act(self, inputs, role="policy"):
        mean = self.forward(inputs["states"])
        latent = mean
        if inputs.get("explore", False):
            latent = mean + self.exploration_std*torch.randn_like(mean)
        actions = self.mean_to_action(latent)
        return actions, None, {"mean_actions": self.mean_to_action(mean), "latent_actions": latent}

    @torch.no_grad()
    def observe_real_next(self, observations):
        if self.running_normalization:
            self.obs_norm.update(observations)
