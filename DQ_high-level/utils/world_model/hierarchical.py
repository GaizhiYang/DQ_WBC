"""Twenty-millisecond dynamics inside a frozen, eight-step controller loop.

The network learns physical increments (or legacy residuals) and foot contacts only. Policy history,
gait, FK, command accumulation and the nominal support trajectory belong to
the adapter. In particular none of the controller's large history tensors is
an output of the learned dynamics.
"""

import math

import torch
from torch import nn

from .dynamics import MLP
from .normalization import RunningNorm, run_mlp


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
                 nominal_dt=0.16, device=None, term_hidden=None,
                 layer_norm=False, running_normalization=False, amp="none",
                 dynamics_target="residual"):
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
        if amp not in ("none", "bf16"):
            raise ValueError("World-model amp must be none or bf16")
        self.amp = amp
        if dynamics_target not in ("residual", "physical_increment"):
            raise ValueError("dynamics_target must be residual or physical_increment")
        if dynamics_target == "physical_increment" and (not running_normalization or spec.delta_dim != 62):
            raise ValueError("Physical increments require running normalization and the 62-output layout")
        self.dynamics_target = dynamics_target
        self.running_normalization = bool(running_normalization)
        self.in_norm = RunningNorm(input_dim) if running_normalization else None
        self.out_norm = RunningNorm(spec.delta_dim, eps=1e-4) if running_normalization else None
        if dynamics_target == "physical_increment":
            from .physical_increments import physical_increment_error_floor
            self.register_buffer("error_floor", torch.tensor(physical_increment_error_floor(), dtype=torch.float32))
        self.dynamics = MLP(input_dim, hidden, spec.delta_dim, zero_last=True,
                            layer_norm=layer_norm)
        self.contact_head = MLP(input_dim, contact_hidden, 4)
        # Success, failure and timeout use the task's separate rules. There is
        # deliberately no learned mixed success/failure survival probability.
        self.term_head = None
        if device is not None:
            self.to(device)

    def _inputs(self, state, action, context, dt):
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
        return inputs, duration

    @torch.no_grad()
    def observe_transition(self, state, action, next_state, context, dt,
                           train_mask=None, *, training_mask=None):
        """Observe each newly collected valid training row exactly once.

        The replay's add() return mask excludes held-out and invalid rows.
        Fitting, validation and imagination must never call this method.
        """
        if not self.running_normalization:
            return
        if training_mask is not None:
            if train_mask is not None:
                raise ValueError("Specify only one training mask")
            train_mask = training_mask
        if train_mask is None:
            raise ValueError("Real-transition normalization requires an explicit training mask")
        mask = torch.as_tensor(train_mask, dtype=torch.bool, device=state.device)
        if mask.shape != state.shape[:1]:
            raise ValueError("Training mask must select real transition rows")
        state, action = state[mask], action[mask]
        next_state, context = next_state[mask], context[mask]
        if not len(state):
            return
        if torch.is_tensor(dt) and dt.ndim and dt.shape[0] == len(mask):
            dt = dt[mask]
        inputs, duration = self._inputs(state, action, context, dt)
        if self.dynamics_target == "physical_increment":
            from .physical_increments import physical_increment_targets
            targets = physical_increment_targets(self.spec, state, next_state)
        else:
            from .residual_targets import residual_targets
            targets = residual_targets(self.spec, state, next_state, duration)
        if not bool(torch.isfinite(inputs).all() & torch.isfinite(targets).all()):
            raise ValueError("Normalization requires finite real features and residual targets")
        self.in_norm.update(inputs)
        self.out_norm.update(targets)

    def low_step(self, state, action, context, dt=0.02, return_logits=False):
        inputs, duration = self._inputs(state, action, context, dt)
        if self.in_norm is not None:
            inputs = self.in_norm(inputs)
        delta = run_mlp(self.dynamics, inputs, self.amp)
        if self.out_norm is not None:
            delta = self.out_norm.denormalize(delta)
        logits = run_mlp(self.contact_head, inputs, self.amp)
        if self.dynamics_target == "physical_increment":
            from .physical_increments import integrate_physical_increments
            predicted = integrate_physical_increments(self.spec, state, delta, action, context, duration)
        else:
            predicted = self.spec.integrate_low(state, delta, action, context, duration)
        # Match the frozen controller's binary sensor input. This threshold has
        # no gradient; continuous dynamics still carries action/state gradients.
        predicted = self.spec.with_contacts(predicted, (logits > 0).to(state.dtype))
        return (predicted, logits) if return_logits else predicted

    def forward(self, state, action, context, dt=0.16, *, return_task=False, active=None):
        _validate_duration(dt, self.nominal_dt, "High-level")
        previous = state
        state = self.spec.prepare_high(state, action)
        for _ in range(self.low_steps):
            state, low_action = self.spec.low_control(state, context)
            state = self.low_step(state, low_action, context, self.low_dt)
        if return_task:
            return self.spec.finish_high(previous, state, action, context,
                                         return_task=True, active=active)
        return self.spec.finish_high(previous, state, action, context)

    def step(self, state, action, context, dt=0.16):
        return self(state, action, context, dt)

    def step_with_task(self, state, action, context, dt=0.16, *, active=None):
        """Share endpoint geometry and task rules with reward/termination."""
        return self(state, action, context, dt, return_task=True, active=active)

    def normalized_sq_error(self, predicted, target):
        return self.spec.error_vector(predicted, target).square().mean(-1)

    def reference_error_vector(self, predicted, target, pre_state):
        if self.dynamics_target != "physical_increment":
            raise ValueError("Reference physical loss requires physical_increment dynamics")
        from .physical_increments import physical_endpoint_error
        return physical_endpoint_error(self.spec, predicted, target, pre_state)

    def reference_error_scale(self):
        if self.dynamics_target != "physical_increment":
            raise ValueError("Reference error scale is defined only for physical increments")
        return torch.maximum(self.out_norm.std(), self.error_floor)

    def coordinate_metadata(self):
        if self.dynamics_target == "physical_increment":
            from .physical_increments import physical_increment_metadata
            return physical_increment_metadata()
        return {"dynamics_target": "residual", "version": 1,
                "frame": "PRE robot heading", "units": "fixed-scale semi-implicit residual"}
