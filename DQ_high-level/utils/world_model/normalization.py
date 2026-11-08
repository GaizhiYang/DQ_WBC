"""Explicit real-data normalization, with no updates during model forwards.

Running moments follow supermimic_manipulation's networks.RunningNorm
algorithm (see that repository's LICENSE and REFERENCE_LICENSE). Buffers are
checkpointed; input gradients remain enabled when statistics are frozen.
"""

import math
import torch
from torch import nn


class RunningNorm(nn.Module):
    def __init__(self, dim, max_count=2e8, eps=1e-2):
        super().__init__()
        if int(dim) < 1 or not math.isfinite(max_count) or max_count <= 0:
            raise ValueError("Invalid running normalization dimension/count")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Running normalization eps must be positive")
        self.max_count, self.eps = float(max_count), float(eps)
        self.register_buffer("mean", torch.zeros(int(dim)))
        self.register_buffer("var", torch.ones(int(dim)))
        self.register_buffer("count", torch.zeros(()))

    @torch.no_grad()
    def update(self, values):
        if values.ndim < 2 or values.shape[-1] != self.mean.numel():
            raise ValueError("RunningNorm update has incorrect feature dimensions")
        values = values.detach().reshape(-1, values.shape[-1]).float()
        if not values.shape[0]:
            return
        if not bool(torch.isfinite(values).all()):
            raise ValueError("RunningNorm requires finite real training samples")
        if bool(self.count >= self.max_count):
            return
        batch_count = values.shape[0]
        mean, var = values.mean(0), values.var(0, unbiased=False)
        total = self.count + batch_count
        delta = mean - self.mean
        combined_mean = self.mean + delta * batch_count / total
        combined_var = (self.var * self.count + var * batch_count
                        + delta.square() * self.count * batch_count / total) / total
        if not bool(torch.isfinite(combined_mean).all() & torch.isfinite(combined_var).all()):
            raise FloatingPointError("RunningNorm moments overflowed; statistics unchanged")
        self.mean.copy_(combined_mean)
        self.var.copy_(combined_var.clamp_min(0))
        self.count.copy_(total)

    def std(self):
        return (self.var + self.eps ** 2).sqrt()

    def forward(self, values):
        return (values - self.mean) / self.std()

    def denormalize(self, values):
        return values * self.std() + self.mean


def run_mlp(network, inputs, amp="none"):
    """Autocast only neural layers; all geometric integration remains FP32."""
    if amp == "none":
        return network(inputs)
    if amp != "bf16":
        raise ValueError("World-model amp must be none or bf16")
    with torch.autocast(device_type=inputs.device.type, dtype=torch.bfloat16):
        return network(inputs).float()
