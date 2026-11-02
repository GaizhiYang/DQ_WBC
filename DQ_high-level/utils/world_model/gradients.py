"""Parameter freezing without cutting differentiable model/critic inputs."""

from contextlib import contextmanager


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
