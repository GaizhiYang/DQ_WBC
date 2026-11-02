"""Short closed-loop BPTT as an auxiliary update of the real PPO Actor.

The survival-weighted objective and short-horizon rollout follow
SuperMimic-Reproduced ``imagination/{objective,rollout,policy_trainer}.py``
(MIT; see THIRD_PARTY_LICENSES.md). This adaptation uses the *real PPO*
Critic, conditional binary gripper commands, and a separate full-policy KL
accept/reject check. It does not train the Critic on imagined experience.

All rollout, gradient and optimizer tensors stay on the input CUDA device.
Only scalar diagnostics are copied to the host once an update is finished.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import inspect

import torch
from torch import nn

from .gradients import freeze_parameters


@dataclass
class ImaginationConfig:
    horizon: int = 3
    gamma: float = 0.99
    lr: float = 1e-5
    strength: float = 0.05  # Scales the Adam learning rate, not its normalized gradient.
    max_grad_norm: float = 0.5
    max_kl: float = 0.005
    use_tail_value: bool = True
    bootstrap_timeouts: bool = False  # Match the real teacher's finite-task PPO target.
    dt: float = 0.15
    gripper_index: int = 6


@dataclass
class ImaginedRollout:
    states: torch.Tensor       # [H+1, batch, state_dim]
    actions: torch.Tensor      # [H, batch, 9]
    rewards: torch.Tensor      # [H, batch]
    termination: torch.Tensor  # [H, batch], natural termination, not timeout
    survival: torch.Tensor     # [H+1, batch], stops at timeout after its reward
    timeouts: torch.Tensor     # [H, batch]
    timeout_values: torch.Tensor  # [H, batch], bootstrap at a time limit
    tail_values: torch.Tensor  # [batch]


def imagination_objective(rollout, gamma, weights=None):
    """Return weighted mean of short rewards and differentiable bootstraps.

    A terminal transition retains its reward (c_t rather than c_{t+1}).
    Natural termination and, by default, the finite task's time limit both
    suppress bootstrap. Explicit bootstrap_timeouts=True instead evaluates
    its actual endpoint. Neither case continues the episode after a timeout.
    Trust weights are fixed conditions, not Actor-optimizable rewards.
    """
    horizon = rollout.rewards.shape[0]
    discounts = gamma ** torch.arange(horizon, device=rollout.rewards.device,
                                      dtype=rollout.rewards.dtype)
    returns = (discounts[:, None] * rollout.survival[:-1]
               * rollout.rewards).sum(0)
    timeout_survival = (rollout.survival[:-1]
                        * (1.0 - rollout.termination) * rollout.timeouts)
    returns = returns + (gamma * discounts[:, None] * timeout_survival
                         * rollout.timeout_values).sum(0)
    returns = returns + gamma ** horizon * rollout.survival[-1] * rollout.tail_values
    if weights is not None:
        returns = returns * weights.detach().reshape_as(returns)
    return returns.mean()


def diagonal_gaussian_kl(old_mean, old_log_std, new_mean, new_log_std):
    """KL(old || new), summed over *all nine* Gaussian action coordinates."""
    variance_ratio = (2.0 * (old_log_std - new_log_std)).exp()
    mean_term = (old_mean - new_mean).square() * (-2.0 * new_log_std).exp()
    return (new_log_std - old_log_std + 0.5 * (variance_ratio + mean_term - 1.0)).sum(-1)


def _role_option(module, name, role, default):
    option = getattr(module, name, default)
    if isinstance(option, dict):
        return option.get(role, option.get("", default))
    return option


@contextmanager
def _evaluation(module):
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, mode in modes:
            child.training = mode


class ImaginationTrainer:
    """Alternate with PPO; caller controls warmup and model reliability gates.

    ``adapter`` reconstructs causal observations/rewards from complete model
    states. Optional ``timeout(next_state, context)`` must indicate a time
    limit only. Learned natural termination belongs to ``model``. No reset
    state or true future observation enters a predicted rollout.

    The caller can pass a fixed [H, batch] raw gripper-command sequence. Otherwise a
    no-gradient reference rollout samples it before building the BPTT graph.
    Its clipped raw magnitude is retained in previous-action observations and
    rewards; the dynamics adapter alone maps its sign to physical open/close.
    Continuous actions use the same fixed reparameterization noise. Shared
    Actor parameters may still indirectly change the gripper distribution;
    the KL test consequently covers every Gaussian action coordinate.
    """

    def __init__(self, model, adapter, actor, critic, config=None,
                 state_preprocessor=None, value_preprocessor=None):
        self.model, self.adapter = model, adapter
        self.actor, self.critic = actor, critic
        self.cfg = config or ImaginationConfig()
        self.state_preprocessor = state_preprocessor
        self.value_preprocessor = value_preprocessor
        self._preprocessor_options = {}
        cfg = self.cfg
        if cfg.horizon < 1 or cfg.dt <= 0 or not 0 < cfg.gamma <= 1:
            raise ValueError("BPTT requires a positive horizon/dt and gamma in (0, 1]")
        if cfg.lr <= 0 or cfg.strength < 0 or cfg.max_grad_norm <= 0 or cfg.max_kl <= 0:
            raise ValueError("Invalid BPTT optimizer strength, learning rate or limits")
        if cfg.gripper_index != 6:
            raise ValueError("The teacher binary gripper action is at index 6")
        if getattr(actor, "transform_func", None) is not None:
            raise ValueError("This BPTT adapter requires the teacher's unsquashed Gaussian policy")
        if getattr(actor, "deterministic", False):
            raise ValueError("BPTT expects the stochastic Gaussian training Actor")
        actor_ids = {id(parameter) for parameter in actor.parameters()}
        for frozen in (model, critic):
            if actor_ids.intersection(id(parameter) for parameter in frozen.parameters()):
                raise ValueError("BPTT requires separate Actor, Critic and world-model parameters")
        # Scaling a loss by beta alone largely cancels inside Adam's adaptive
        # normalization. Scale its actual parameter step via learning rate.
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr * cfg.strength)
        self.accepted_updates = 0
        self.rejected_updates = 0

    def _preprocess(self, preprocessor, values, inverse=False):
        if preprocessor is None:
            return values
        key = id(preprocessor)
        if key not in self._preprocessor_options:
            function = preprocessor.forward if isinstance(preprocessor, nn.Module) else preprocessor
            parameters = inspect.signature(function).parameters
            self._preprocessor_options[key] = ("no_grad" in parameters or any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()))
        options = {"train": False}
        if inverse:
            options["inverse"] = True
        # Installed skrl defaults to no_grad=True even with train=False.
        # Both scaler directions must explicitly preserve the input Jacobian.
        if self._preprocessor_options[key]:
            options["no_grad"] = False
        result = preprocessor(values, **options)
        if torch.is_grad_enabled() and values.requires_grad and not result.requires_grad:
            raise RuntimeError("BPTT preprocessor detached its inputs; enable input gradients without updating statistics")
        return result

    def _normalize(self, observations):
        return self._preprocess(self.state_preprocessor, observations)

    def _policy(self, observations):
        mean, log_std, _ = self.actor.compute({"states": self._normalize(observations)}, role="policy")
        if mean.shape[-1] != 9:
            raise ValueError("Teacher BPTT requires eight continuous actions plus gripper at index 6")
        if _role_option(self.actor, "_g_clip_log_std", "policy", True):
            low = _role_option(self.actor, "_g_log_std_min", "policy", -20.0)
            high = _role_option(self.actor, "_g_log_std_max", "policy", 2.0)
            log_std = log_std.clamp(low, high)
        return mean, log_std.expand_as(mean)

    def _value(self, state, context):
        if not self.cfg.use_tail_value:
            return torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        observation = self.adapter.observation(state, context)
        value, _ = self.critic.compute({"states": self._normalize(observation)}, role="value")
        value = self._preprocess(self.value_preprocessor, value, inverse=True)
        return value.reshape(state.shape[0])

    def _action(self, mean, log_std, noise, mode):
        action = mean + log_std.exp() * noise
        index = self.cfg.gripper_index
        # Keep the fixed raw command, not a +/-1 surrogate: previous-action
        # observations and action penalties depend on its actual magnitude.
        # The adapter handles its binary physical effect without a gradient.
        action = torch.cat((action[:, :index], mode.detach().unsqueeze(-1),
                            action[:, index + 1:]), -1)
        if _role_option(self.actor, "_g_clip_actions", "policy", False):
            action = torch.clamp(action, min=self.actor.clip_actions_min,
                                 max=self.actor.clip_actions_max)
        # Replay stores env.actions after the outer simulator clamp. Keep the
        # learned-model action distribution identical when clipActions is set.
        if hasattr(self.adapter, "clip_actions"):
            action = action.clamp(-self.adapter.clip_actions, self.adapter.clip_actions)
        return action

    @staticmethod
    def _dt_at(dt, step, horizon, batch_size):
        if torch.is_tensor(dt) and dt.ndim >= 2 and dt.shape[:2] == (horizon, batch_size):
            return dt[step]
        return dt

    def _reference_modes(self, state, context, noise, dt):
        modes = []
        with torch.no_grad():
            for step in range(self.cfg.horizon):
                mean, log_std = self._policy(self.adapter.observation(state, context))
                gripper = mean[:, self.cfg.gripper_index] + log_std[:, self.cfg.gripper_index].exp() * noise[step, :, self.cfg.gripper_index]
                action = self._action(mean, log_std, noise[step], gripper)
                modes.append(action[:, self.cfg.gripper_index])
                duration = self._dt_at(dt, step, self.cfg.horizon, state.shape[0])
                state = self.model.step(state, action, context, duration)
        return torch.stack(modes)

    @torch.enable_grad()
    def imagine(self, start_state, context, gripper_modes=None, noise=None, dt=None):
        """Build a closed-loop graph; callers freeze model/Critic parameters.

        Public for diagnostics and tests. ``update`` establishes freezing and
        preprocessing modes automatically. States before the start are fixed;
        every subsequent state and reconstructed observation stays connected.
        ``gripper_modes`` retains its existing API name but contains fixed raw
        command values, not sign-normalized +/-1 values.
        """
        state, context = start_state.detach(), context.detach()
        horizon, batch_size = self.cfg.horizon, state.shape[0]
        dt = self.cfg.dt if dt is None else dt
        if noise is None:
            noise = torch.randn(horizon, batch_size, 9, device=state.device, dtype=state.dtype)
        else:
            noise = noise.detach().to(device=state.device, dtype=state.dtype)
            if noise.shape != (horizon, batch_size, 9):
                raise ValueError("BPTT noise must have shape [horizon, batch, 9]")
        if gripper_modes is None:
            gripper_modes = self._reference_modes(state, context, noise, dt)
        else:
            gripper_modes = gripper_modes.detach().to(device=state.device, dtype=state.dtype)
            if gripper_modes.shape == (batch_size,):
                gripper_modes = gripper_modes.unsqueeze(0).expand(horizon, -1)
            if gripper_modes.shape != (horizon, batch_size):
                raise ValueError("gripper_modes must have shape [batch] or [horizon, batch]")
        survival = torch.ones(batch_size, device=state.device, dtype=state.dtype)
        states, survivals = [state], [survival]
        actions, rewards, termination, timeouts, timeout_values = [], [], [], [], []
        timeout_fn = getattr(self.adapter, "timeout", getattr(self.adapter, "timeouts", None))
        for step in range(horizon):
            mean, log_std = self._policy(self.adapter.observation(state, context))
            action = self._action(mean, log_std, noise[step], gripper_modes[step])
            duration = self._dt_at(dt, step, horizon, batch_size)
            next_state = self.model.step(state, action, context, duration)
            reward = self.adapter.reward(state, next_state, action, context).reshape(batch_size)
            if getattr(self.model, "term_head", True) is None:
                done = torch.zeros_like(survival)
            else:
                done = self.model.termination_probability(next_state, context).reshape(batch_size).clamp(0, 1)
            timeout = (torch.zeros_like(survival) if timeout_fn is None else
                       timeout_fn(next_state, context).to(dtype=state.dtype).reshape(batch_size).detach())
            # Evaluate the correct endpoint, not the hypothetical post-reset state.
            timeout_value = (self._value(next_state, context)
                             if timeout_fn is not None and self.cfg.bootstrap_timeouts else
                             torch.zeros_like(survival))
            survival = survival * (1.0 - done) * (1.0 - timeout)
            states.append(next_state)
            actions.append(action)
            rewards.append(reward)
            termination.append(done)
            timeouts.append(timeout)
            timeout_values.append(timeout_value)
            survivals.append(survival)
            state = next_state
        return ImaginedRollout(torch.stack(states), torch.stack(actions), torch.stack(rewards),
                               torch.stack(termination), torch.stack(survivals), torch.stack(timeouts),
                               torch.stack(timeout_values), self._value(state, context))

    @torch.enable_grad()
    def update(self, start_state, context, weights=None, anchor_observations=None,
               gripper_modes=None, noise=None, dt=None):
        """One bounded auxiliary Adam update, with rollback of parameters AND Adam state."""
        if start_state.ndim != 2 or start_state.shape[0] == 0:
            raise ValueError("BPTT needs a nonempty [batch, state_dim] start tensor")
        if weights is not None:
            weights = weights.detach().to(device=start_state.device, dtype=start_state.dtype).reshape(-1)
            if weights.shape[0] != start_state.shape[0] or not bool(torch.isfinite(weights).all()) or bool((weights < 0).any()):
                raise ValueError("BPTT trust weights must be finite, nonnegative and match the batch")
        if self.cfg.strength == 0 or (weights is not None and not bool((weights > 0).any())):
            return {"accepted": 0.0, "rollback": 0.0, "skipped": 1.0, "kl": 0.0,
                    "actor_grad_norm": 0.0}
        frozen = [self.model, self.critic]
        frozen.extend(preprocessor for preprocessor in (self.state_preprocessor, self.value_preprocessor)
                      if isinstance(preprocessor, nn.Module))
        with freeze_parameters(*frozen), _evaluation(self.actor):
            if anchor_observations is None:
                anchor_observations = self.adapter.observation(start_state.detach(), context.detach())
            anchor_observations = anchor_observations.detach()
            with torch.no_grad():
                old_mean, old_log_std = self._policy(anchor_observations)
                old_mean, old_log_std = old_mean.clone(), old_log_std.clone()
            rollout = self.imagine(start_state, context, gripper_modes, noise, dt)
            objective = imagination_objective(rollout, self.cfg.gamma, weights)
            loss = -objective / self.cfg.horizon
            self.optimizer.zero_grad(set_to_none=True)
            if not loss.requires_grad:
                return {"accepted": 0.0, "rollback": 0.0, "skipped": 1.0,
                        "no_gradient": 1.0, "kl": 0.0, "actor_grad_norm": 0.0}
            if not bool(torch.isfinite(loss)):
                self.rejected_updates += 1
                return {"accepted": 0.0, "rollback": 0.0, "skipped": 1.0,
                        "nonfinite": 1.0, "kl": 0.0, "actor_grad_norm": 0.0}
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            if not bool(torch.isfinite(grad_norm)):
                self.optimizer.zero_grad(set_to_none=True)
                self.rejected_updates += 1
                return {"accepted": 0.0, "rollback": 0.0, "skipped": 1.0,
                        "nonfinite": 1.0, "kl": 0.0, "actor_grad_norm": float(grad_norm)}
            # Snapshot on the current GPU; rollback must include Adam moments/step.
            before_actor = {name: value.detach().clone() for name, value in self.actor.state_dict().items()}
            before_optimizer = deepcopy(self.optimizer.state_dict())
            self.optimizer.step()
            with torch.no_grad():
                new_mean, new_log_std = self._policy(anchor_observations)
                kl = diagonal_gaussian_kl(old_mean, old_log_std, new_mean, new_log_std).mean()
                accepted = bool(torch.isfinite(kl)) and float(kl) <= self.cfg.max_kl
            if accepted:
                self.accepted_updates += 1
            else:
                self.actor.load_state_dict(before_actor)
                self.optimizer.load_state_dict(before_optimizer)
                self.optimizer.zero_grad(set_to_none=True)
                self.rejected_updates += 1
            actual_dt = torch.as_tensor(self.cfg.dt if dt is None else dt, device=start_state.device)
            metrics = {
                "accepted": float(accepted), "rollback": float(not accepted), "skipped": 0.0,
                "kl": float(kl), "objective": float(objective.detach()),
                "actor_grad_norm": float(grad_norm),
                "reward_mean": float(rollout.rewards.detach().mean()),
                "survival_H": float(rollout.survival[-1].detach().mean()),
                "tail_value": float(rollout.tail_values.detach().mean()),
                "timeout_fraction": float(rollout.timeouts.detach().mean()),
                "trust_mean": 1.0 if weights is None else float(weights.mean()),
                "dt_mean": float(actual_dt.float().mean()),
                "horizon_seconds_mean": float(actual_dt.float().mean()) * self.cfg.horizon,
                "effective_lr": float(self.optimizer.param_groups[0]["lr"]),
            }
        return metrics

    def state_dict(self):
        return {"optimizer": self.optimizer.state_dict(),
                "accepted_updates": self.accepted_updates,
                "rejected_updates": self.rejected_updates}

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.accepted_updates = int(state.get("accepted_updates", 0))
        self.rejected_updates = int(state.get("rejected_updates", 0))
