"""Twenty-millisecond dynamics inside a frozen, eight-step controller loop.

The network learns physical residuals and foot contacts only. Policy history,
gait, FK, command accumulation and the nominal support trajectory belong to
the adapter. In particular none of the controller's large history tensors is
an output of the learned dynamics.
"""

import math

import torch
from torch import nn

from .dynamics import MLP


def _validate_duration(duration, expected, label):
    if torch.is_tensor(duration):
        valid = torch.isfinite(duration) & ((duration - expected).abs() <= 1e-7)
        if not bool(valid.all()):
            raise ValueError("{} dynamics requires dt={}".format(label, expected))
    elif not math.isclose(float(duration), expected, abs_tol=1e-8):
        raise ValueError("{} dynamics requires dt={}".format(label, expected))


class HierarchicalDynamics(nn.Module):
    low_steps = 8

    def __init__(self, spec, hidden=(512, 512, 512), contact_hidden=(128, 128),
                 nominal_dt=0.16, device=None, term_hidden=None):
        super().__init__()
        # The adapter owns the frozen low policy. Never register its parameters
        # as part of this model/optimizer or change its training mode with ours.
        object.__setattr__(self, "spec", spec)
        self.low_dt = float(getattr(spec, "low_dt", 0.02))
        self.nominal_dt = float(nominal_dt)
        if not math.isclose(self.low_dt, 0.02, abs_tol=1e-8):
            raise ValueError("Hierarchical dynamics requires 20 ms model steps")
        if not math.isclose(self.nominal_dt, self.low_steps * self.low_dt,
                            abs_tol=1e-8):
            raise ValueError("A high-level action must span exactly eight 20 ms steps")
        context_dim = getattr(spec, "context_feature_dim", spec.context_dim)
        action_features = getattr(spec, "low_action_feature_dim", spec.low_action_dim)
        input_dim = spec.low_feature_dim + action_features + context_dim + 1
        self.dynamics = MLP(input_dim, hidden, spec.delta_dim, zero_last=True)
        self.contact_head = MLP(input_dim, contact_hidden, 4)
        # Success, failure and timeout use the task's separate rules. There is
        # deliberately no learned mixed success/failure survival probability.
        self.term_head = None
        if device is not None:
            self.to(device)

    def low_step(self, state, action, context, dt=0.02, return_logits=False):
        if action.shape[-1] != self.spec.low_action_dim:
            raise ValueError("low_step requires recorded low-level actuator commands")
        _validate_duration(dt, self.low_dt, "Low-level")
        duration = torch.as_tensor(dt, dtype=state.dtype, device=state.device)
        if duration.ndim == 0:
            duration = duration.expand(*state.shape[:-1], 1)
        elif duration.shape == state.shape[:-1]:
            duration = duration.unsqueeze(-1)
        duration = duration.expand(*state.shape[:-1], 1)
        conditions = self.spec.normalize_context(context)
        inputs = torch.cat((self.spec.low_features(state),
                            self.spec.low_action_features(state, action),
                            conditions, duration / self.low_dt), dim=-1)
        delta = self.dynamics(inputs)
        logits = self.contact_head(inputs)
        predicted = self.spec.integrate_low(state, delta, action, context, duration)
        # Match the frozen controller's binary sensor input. This threshold has
        # no gradient; continuous dynamics still carries action/state gradients.
        predicted = self.spec.with_contacts(predicted, (logits > 0).to(state.dtype))
        return (predicted, logits) if return_logits else predicted

    def forward(self, state, action, context, dt=0.16):
        _validate_duration(dt, self.nominal_dt, "High-level")
        previous = state
        state = self.spec.prepare_high(state, action)
        for _ in range(self.low_steps):
            state, low_action = self.spec.low_control(state, context)
            state = self.low_step(state, low_action, context, self.low_dt)
        return self.spec.finish_high(previous, state, action, context)

    def step(self, state, action, context, dt=0.16):
        return self(state, action, context, dt)

    def normalized_sq_error(self, predicted, target):
        return self.spec.error_vector(predicted, target).square().mean(-1)
