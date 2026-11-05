"""Fixed scales and raw sensor noise for the high-level grasping policy.

The noise magnitudes here are task-specific choices, not a copy of the
SuperMimic sensor table. Only independently measured high-level channels are
perturbed. Object features, controls, EKF forecasts, candidate grasps, masks and
previous actions are unchanged. In particular this module must not be applied
to the pretrained low-level policy or to world-model physical target labels.

Use the same noise module for real-policy observations and reconstructed
imagined-policy observations. Apply noise before FixedScale, and call eval()
to disable it during evaluation. Neither module learns running statistics.
"""

import math

import torch
from torch import nn


def _width(size):
    if isinstance(size, int):
        width = size
    elif isinstance(size, (tuple, list, torch.Size)):
        width = math.prod(size)
    elif getattr(size, "shape", None) is not None:
        # Accept Box-like spaces without importing either gym implementation.
        width = math.prod(size.shape)
    else:
        raise ValueError("size must be a positive integer, shape or Box-like space")
    if not isinstance(width, int) or width < 1:
        raise ValueError("size must contain a positive number of scalar features")
    return width


def _vector(value, width, device, name, *, positive):
    vector = torch.as_tensor(value, device=device, dtype=torch.float32).detach().clone()
    if vector.ndim == 0:
        vector = vector.expand(width).clone()
    if vector.shape != (width,):
        raise ValueError("{} must be scalar or have shape ({},)".format(name, width))
    valid = torch.isfinite(vector) & ((vector > 0) if positive else (vector >= 0))
    if not bool(valid.all()):
        raise ValueError("{} must be finite and {}".format(name, "positive" if positive else "nonnegative"))
    return vector


def _check_input(x, width, device):
    if x.ndim < 1 or x.shape[-1] != width:
        raise ValueError("Expected observation final dimension {}".format(width))
    if not x.is_floating_point():
        raise TypeError("Policy observations must be floating-point tensors")
    if x.device != device:
        raise ValueError("Move policy-input buffers and observations to the same device")


def _check_version(state_dict, prefix, marker, expected, error_msgs):
    value = state_dict.get(prefix + marker)
    if (not torch.is_tensor(value) or value.numel() != 1
            or value.dtype != torch.int64 or int(value.item()) != expected):
        error_msgs.append("{}{} is missing or incompatible; refusing an implicit legacy "
                          "running-statistics/noise checkpoint conversion".format(prefix, marker))


def teacher_observation_vectors(feature_dim=1024, ndof=19, horizon_count=2,
                                candidates=30, velocity_obs=False, device=None,
                                dtype=torch.float32):
    """Return ``(scale, noise_half_width)`` matching teacher observation order.

    Default width is 1284, or 1289 with five velocity observations. Angular
    scales are pi for object/EE/goal/candidate angles and joint positions;
    other scales are one. Joint velocities are already multiplied by .05 in
    the teacher observation. Their noise half-width .005 therefore corresponds
    to .1 rad/s before that mapping.

    With zero future horizons the entire forecast block, including its valid
    bit, is absent. This follows the adapter rather than reserving empty slots.
    """
    counts = (feature_dim, ndof, horizon_count, candidates)
    if any(not isinstance(value, int) for value in counts) or ndof < 1 or min(feature_dim, horizon_count, candidates) < 0:
        raise ValueError("Feature/horizon/candidate counts must be nonnegative integers and ndof >= 1")
    blocks = []

    def add(width, scale=1., noise=0.):
        blocks.append((torch.full((width,), scale, device=device, dtype=dtype),
                       torch.full((width,), noise, device=device, dtype=dtype)))

    add(feature_dim)
    add(3, noise=.005)  # object position
    add(3, math.pi, .01)  # object RPY
    add(3, noise=.005)  # EE position
    add(3, math.pi, .01)  # EE RPY
    add(ndof, math.pi, .005)
    add(ndof - 1, noise=.005)  # qd is already scaled by .05
    add(3)  # commands
    add(3)  # accumulated EE goal position
    add(3, math.pi)  # accumulated EE goal RPY
    if velocity_obs:
        add(5, noise=.01)
    for _ in range(horizon_count):
        add(3)
        add(3, math.pi)
    if horizon_count:
        add(1)  # forecast validity
    add(candidates * 3)
    add(candidates * 3, math.pi)
    add(9)  # actually executed previous high-level action
    return tuple(torch.cat(parts, dim=0) for parts in zip(*blocks))


class FixedScale(nn.Module):
    """skrl-compatible fixed normalization, without centering or clipping."""

    version = 1

    def __init__(self, size, device=None, scale=None):
        super().__init__()
        self.size = _width(size)
        self.register_buffer("scale", _vector(1. if scale is None else scale,
                                              self.size, device, "scale", positive=True))
        self.register_buffer("_fixed_scale_version", torch.tensor(self.version, device=self.scale.device,
                                                                  dtype=torch.int64))

    @property
    def device(self):
        return self.scale.device

    def forward(self, x, train=False, inverse=False, no_grad=True):
        # train is accepted for skrl API compatibility; there are no statistics
        # to update, and module.train() likewise does not change normalization.
        _check_input(x, self.size, self.device)

        def compute():
            factor = self.scale.to(dtype=x.dtype)
            return x * factor if inverse else x / factor

        if no_grad:
            with torch.no_grad():
                return compute()
        return compute()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _check_version(state_dict, prefix, "_fixed_scale_version", self.version, error_msgs)
        scale = state_dict.get(prefix + "scale")
        if torch.is_tensor(scale) and (not bool(torch.isfinite(scale).all()) or not bool((scale > 0).all())):
            error_msgs.append(prefix + "scale must be finite and positive")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)


class HighLevelObservationNoise(nn.Module):
    """Stateless independent uniform raw-observation noise, disabled by eval()."""

    version = 1

    def __init__(self, size, device=None, noise_half_width=None, noise_scale=1.):
        super().__init__()
        self.size = _width(size)
        self.register_buffer("noise_half_width", _vector(0. if noise_half_width is None else noise_half_width,
                    self.size, device, "noise_half_width", positive=False))
        strength = torch.as_tensor(noise_scale, device=self.noise_half_width.device,
                                   dtype=torch.float32).detach().clone()
        if strength.ndim != 0 or not bool(torch.isfinite(strength)) or not bool(strength >= 0):
            raise ValueError("noise_scale must be a finite nonnegative scalar")
        self.register_buffer("noise_scale", strength)
        self.register_buffer("_observation_noise_version", torch.tensor(self.version,
                    device=self.noise_half_width.device, dtype=torch.int64))

    @property
    def device(self):
        return self.noise_half_width.device

    def forward(self, x):
        _check_input(x, self.size, self.device)
        if not self.training:
            return x
        half_width = self.noise_half_width.to(dtype=x.dtype) * self.noise_scale.to(dtype=x.dtype)
        return x + (torch.rand_like(x) * 2 - 1) * half_width

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        _check_version(state_dict, prefix, "_observation_noise_version", self.version, error_msgs)
        for name in ("noise_half_width", "noise_scale"):
            value = state_dict.get(prefix + name)
            if torch.is_tensor(value) and (not bool(torch.isfinite(value).all()) or not bool((value >= 0).all())):
                error_msgs.append(prefix + name + " must be finite and nonnegative")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
