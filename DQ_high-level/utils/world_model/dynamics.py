"""Residual high-level dynamics with an adapter-owned geometric state layout.

The MLP and zero-initialized residual head are adapted from
SuperMimic-Reproduced ``world_model/dynamics.py`` (MIT). Geometry, command
integration, causal observation history and input normalization belong to the
DQ task adapter, not to this generic network. See THIRD_PARTY_LICENSES.md.
"""

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, zero_last=False, layer_norm=False, activation=nn.SiLU):
        super().__init__()
        layers = []
        width = int(in_dim)
        for size in hidden:
            layers.append(nn.Linear(width, int(size)))
            if layer_norm:
                layers.append(nn.LayerNorm(int(size)))
            layers.append(activation())
            width = int(size)
        last = nn.Linear(width, int(out_dim))
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


class ResidualDynamics(nn.Module):
    """F(state, action, physical context, actual dt) -> next physical state.

    The spec implements ``features``, ``integrate`` and ``error_vector`` and
    supplies dimensions. Optional ``normalize_context`` may change the context
    width when ``context_feature_dim`` is specified. A learned termination head
    is built only if the spec exposes termination features. It models natural
    task termination, not timeout; timeout is handled by the task clock.
    """

    def __init__(self, spec, hidden=(512, 512, 512), term_hidden=(64, 64),
                 nominal_dt=0.15, device=None):
        super().__init__()
        self.spec = spec
        self.nominal_dt = float(nominal_dt)
        if self.nominal_dt <= 0:
            raise ValueError("nominal_dt must be positive")
        context_width = getattr(spec, "context_feature_dim", spec.context_dim)
        self.dynamics = MLP(spec.feature_dim + spec.action_dim + context_width + 1,
                            hidden, spec.delta_dim, zero_last=True)
        self.term_head = (MLP(spec.termination_feature_dim, term_hidden, 1)
                          if getattr(spec, "termination_feature_dim", 0) else None)
        if device is not None:
            self.to(device)

    def forward(self, state, action, context, dt):
        duration = torch.as_tensor(dt, dtype=state.dtype, device=state.device)
        if duration.ndim == 0:
            duration = duration.expand(*state.shape[:-1], 1)
        elif duration.shape == state.shape[:-1]:
            duration = duration.unsqueeze(-1)
        duration = duration.expand(*state.shape[:-1], 1)
        normalization = getattr(self.spec, "normalize_context", None)
        conditions = normalization(context) if normalization is not None else context
        features = self.spec.features(state)
        action_encoder = getattr(self.spec, "dynamics_action", None)
        physical_action = action_encoder(state, action) if action_encoder is not None else action
        # Known actuator clipping/deadzones belong outside the learned MLP.
        # A saturated real command must not acquire a fictitious derivative
        # simply because a neural network sees its unconstrained raw value.
        inputs = torch.cat((features, physical_action, conditions, duration / self.nominal_dt), -1)
        delta = self.dynamics(inputs)
        return self.spec.integrate(state, delta, action, context, duration)

    def step(self, state, action, context, dt):
        return self(state, action, context, dt)

    def termination_logits(self, next_state, context):
        if self.term_head is None:
            raise RuntimeError("This state adapter does not define termination features")
        return self.term_head(self.spec.termination_features(next_state, context)).squeeze(-1)

    def termination_probability(self, next_state, context):
        return self.termination_logits(next_state, context).sigmoid()

    def normalized_sq_error(self, predicted, target):
        return self.spec.error_vector(predicted, target).square().mean(-1)
