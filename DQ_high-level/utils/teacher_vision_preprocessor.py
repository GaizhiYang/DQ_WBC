"""Teacher-compatible running statistics that leave camera pixels untouched."""

import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler


class FrozenRunningStandardScaler(RunningStandardScaler):
    """An original skrl scaler whose statistics can explicitly be frozen.

    ``freeze`` is independent of the module's train/eval mode: with freeze=True,
    even an explicit ``forward(..., train=True)`` does not update statistics.
    No new buffers are registered, so old teacher scaler checkpoints load with
    strict=True and retain their original keys and shapes. ``freeze`` is runtime
    configuration and should be recorded in the experiment configuration.
    """

    def __init__(self, size, epsilon=1e-8, clip_threshold=5.0, device=None, freeze=True):
        super().__init__(size=size, epsilon=epsilon, clip_threshold=clip_threshold, device=device)
        self.freeze = bool(freeze)

    def _compute(self, x, train=False, inverse=False):
        return super()._compute(x, train=train and not self.freeze, inverse=inverse)


class PrefixRunningStandardScaler(FrozenRunningStandardScaler):
    """Normalize only the privileged prefix of a packed teacher observation.

    ``size`` describes the complete input (63484 by default), but the stored
    running_mean/running_variance have only ``privileged_dim`` entries (1276).
    The image suffix passes through unchanged during both forward and inverse
    transforms. The inherited forward also preserves skrl's ``no_grad`` option.
    """

    def __init__(self, size=63484, privileged_dim=1276, epsilon=1e-8,
                 clip_threshold=5.0, device=None, freeze=True):
        privileged_dim = int(privileged_dim)
        if privileged_dim <= 0:
            raise ValueError("privileged_dim must be positive")
        super().__init__(size=privileged_dim, epsilon=epsilon,
                         clip_threshold=clip_threshold, device=device, freeze=freeze)
        self.input_size = int(self._get_space_size(size))
        self.privileged_dim = privileged_dim
        if self.input_size <= self.privileged_dim:
            raise ValueError("size must include an image suffix after privileged_dim")

    def _compute(self, x, train=False, inverse=False):
        if x.ndim not in (2, 3) or x.shape[-1] != self.input_size:
            raise ValueError(
                "Expected batched packed observations with final dimension "
                f"{self.input_size}, got {tuple(x.shape)}"
            )
        privileged = super()._compute(x[..., :self.privileged_dim], train=train, inverse=inverse)
        return torch.cat((privileged, x[..., self.privileged_dim:]), dim=-1)
