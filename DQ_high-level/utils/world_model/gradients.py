"""Parameter freezing without cutting differentiable model/critic inputs."""

from contextlib import contextmanager

import torch


def tensors_finite(tensors):
    """Reduce on each device before reading a scalar; never copy model tensors."""
    checks = {}
    for tensor in tensors:
        if torch.is_tensor(tensor):
            checks.setdefault(tensor.device, []).append(torch.isfinite(tensor.detach()).all())
    return all(bool(torch.stack(group).all()) for group in checks.values())


def optimizer_finite(optimizer):
    """Check Adam moments and tensor step counters without moving them off GPU."""
    return tensors_finite(value for state in optimizer.state.values() for value in state.values())


@contextmanager
def freeze_parameters(*modules):
    """Keep autograd enabled while temporarily freezing the given parameters.

    ``torch.no_grad`` and detached model outputs are intentionally not used:
    actor BPTT requires the dynamics and terminal value input Jacobians.
    Original flags and train/eval modes are restored even if the body raises.
    """
    parameters = []
    seen = set()
    modes = []
    seen_modules = set()
    for module in modules:
        for child in module.modules():
            if id(child) not in seen_modules:
                seen_modules.add(id(child))
                modes.append((child, child.training))
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
        module.eval()
    try:
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)
        for module, training in modes:
            module.training = training
