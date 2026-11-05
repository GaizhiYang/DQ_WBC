"""Short closed-loop BPTT as an auxiliary update of the real PPO Actor.

The survival-weighted objective and short-horizon rollout follow
SuperMimic-Reproduced ``imagination/{objective,rollout,policy_trainer}.py``
(MIT; see THIRD_PARTY_LICENSES.md). The production path has its own privileged
Critic fitted on fixed imagined TD(lambda) targets. The real PPO Critic remains
available for legacy experiments and is never fitted on imagined transitions.
Continuous gripper commands and the finite task's hard endpoints are retained.
Imagined actions use the Actor mean and its differentiable action mapping.
The real PPO policy keeps
sampling Gaussian actions and exclusively learns its action standard deviations.

All rollout, gradient and optimizer tensors stay on the input CUDA device.
Only scalar diagnostics are copied to the host once an update is finished.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import inspect
import math

import torch
from torch import nn

from .gradients import freeze_parameters
from .imagined_critic import ImaginedCritic, ImaginedCriticTrainer


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
    value_source: str = "ppo"  # Legacy default; the production entry selects "imagined".
    actor_betas: tuple = (0.9, 0.999)
    grad_clip_value: float = None
    kl_rollback: bool = True
    chunk_size: int = 0  # Accumulate one batch's Actor gradient across smaller rollouts.
    critic_hidden: tuple = (256, 128)
    critic_lr: float = 2e-4
    critic_gae_lambda: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    critic_max_grad_norm: float = 1.0


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

    Every action coordinate is recomputed from the predicted observation at
    every step. The Actor's optional ``mean_to_action`` maps its latent Gaussian
    mean to executed commands, including differentiable gripper opening. No
    action coordinate is detached or replaced by a reference sequence. The
    rollout uses no sampled action noise. When enabled, the KL gate covers
    every latent Gaussian action coordinate, including gripper. Standard
    deviations are only used as detached values in the diagnostic, never to form
    imagined actions or to receive an auxiliary policy gradient.
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
        if (cfg.lr <= 0 or cfg.strength < 0 or cfg.max_grad_norm <= 0
                or (cfg.kl_rollback and cfg.max_kl <= 0)):
            raise ValueError("Invalid BPTT optimizer strength, learning rate or limits")
        if cfg.value_source not in ("ppo", "imagined"):
            raise ValueError("BPTT value_source must be ppo or imagined")
        if (len(cfg.actor_betas) != 2 or any(not 0 <= beta < 1 for beta in cfg.actor_betas)
                or cfg.chunk_size < 0 or int(cfg.chunk_size) != cfg.chunk_size
                or (cfg.grad_clip_value is not None and cfg.grad_clip_value <= 0)):
            raise ValueError("Invalid BPTT Adam, clipping or chunk settings")
        if any(not math.isfinite(float(value)) for value in
               (cfg.lr, cfg.strength, cfg.max_grad_norm, cfg.max_kl, cfg.critic_lr)):
            raise ValueError("BPTT optimization settings must be finite")
        if cfg.gripper_index != 6:
            raise ValueError("The teacher continuous gripper opening action is at index 6")
        if getattr(actor, "transform_func", None) is not None:
            raise ValueError("BPTT requires no global policy transform; use mean_to_action for coordinate transforms")
        if getattr(actor, "deterministic", False):
            raise ValueError("The real PPO training Actor must remain stochastic; BPTT uses its means directly")
        actor_ids = {id(parameter) for parameter in actor.parameters()}
        for frozen in (model, critic):
            if actor_ids.intersection(id(parameter) for parameter in frozen.parameters()):
                raise ValueError("BPTT requires separate Actor, Critic and world-model parameters")
        # Scaling a loss by beta alone largely cancels inside Adam's adaptive
        # normalization. Scale its actual parameter step via learning rate.
        # Preserve parameter groups for existing auxiliary checkpoints. The
        # mean-only objective leaves log_std.grad=None; zero_grad(set_to_none=True)
        # also clears residual PPO gradients. Adam skips these parameters even
        # if a legacy stochastic-BPTT checkpoint contains nonzero moments.
        self.optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr * cfg.strength,
                                          betas=tuple(cfg.actor_betas))
        self.imagined_critic = None
        self.imagined_critic_trainer = None
        if cfg.value_source == "imagined":
            if not callable(getattr(adapter, "critic_features", None)):
                raise ValueError("Imagined value learning requires adapter.critic_features")
            exemplar = next(actor.parameters())
            self.imagined_critic = ImaginedCritic(adapter.critic_feature_dim, cfg.critic_hidden).to(
                device=exemplar.device, dtype=exemplar.dtype)
            self.imagined_critic_trainer = ImaginedCriticTrainer(
                self.imagined_critic, adapter, lr=cfg.critic_lr, gamma=cfg.gamma,
                lam=cfg.critic_gae_lambda, epochs=cfg.critic_epochs,
                minibatches=cfg.critic_minibatches, max_grad_norm=cfg.critic_max_grad_norm,
                bootstrap_timeouts=cfg.bootstrap_timeouts)
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
            raise ValueError("Teacher BPTT requires nine continuous actions with gripper opening at index 6")
        if _role_option(self.actor, "_g_clip_log_std", "policy", True):
            low = _role_option(self.actor, "_g_log_std_min", "policy", -20.0)
            high = _role_option(self.actor, "_g_log_std_max", "policy", 2.0)
            log_std = log_std.clamp(low, high)
        return mean, log_std.detach().expand_as(mean)

    def _value(self, state, context):
        if not self.cfg.use_tail_value:
            return torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        if self.imagined_critic is not None:
            # These features have task-specific fixed scales. No PPO value
            # scaler or stochastic policy-observation transform belongs here.
            return self.imagined_critic(self.adapter.critic_features(state, context))
        observation = self.adapter.observation(state, context)
        value, _ = self.critic.compute({"states": self._normalize(observation)}, role="value")
        value = self._preprocess(self.value_preprocessor, value, inverse=True)
        return value.reshape(state.shape[0])

    def _action(self, mean):
        # The real policy and imagined mean commands share the same mapping.
        # In the teacher, sigmoid(mean[:, 6]) is the normalized gripper opening;
        # its derivative remains in the graph just like the other eight actions.
        mean_to_action = getattr(self.actor, "mean_to_action", None)
        action = mean if mean_to_action is None else mean_to_action(mean)
        if _role_option(self.actor, "_g_clip_actions", "policy", False):
            action = torch.clamp(action, min=self.actor.clip_actions_min,
                                 max=self.actor.clip_actions_max)
        # Replay stores env.actions after the outer simulator clamp. Apply the
        # same execution mapping to the mean commands when clipActions is set.
        if hasattr(self.adapter, "clip_actions"):
            action = action.clamp(-self.adapter.clip_actions, self.adapter.clip_actions)
        return action

    @staticmethod
    def _dt_at(dt, step, horizon, batch_size):
        if torch.is_tensor(dt) and dt.ndim >= 2 and dt.shape[:2] == (horizon, batch_size):
            return dt[step]
        return dt

    @staticmethod
    def _slice_dt(dt, begin, end, horizon, batch_size):
        if not torch.is_tensor(dt) or dt.ndim == 0:
            return dt
        if dt.ndim >= 2 and dt.shape[:2] == (horizon, batch_size):
            return dt[:, begin:end]
        if dt.shape[0] == batch_size:
            return dt[begin:end]
        return dt

    @torch.enable_grad()
    def imagine(self, start_state, context, *, dt=None):
        """Build a closed-loop graph; callers freeze model/Critic parameters.

        Public for diagnostics and tests. ``update`` establishes freezing and
        preprocessing modes automatically. States before the start are fixed;
        every subsequent state and reconstructed observation stays connected.
        All nine mean action coordinates, including gripper opening, remain
        differentiable through later predicted states and observations.
        There is no sampled action noise. An adapter may independently add
        known sensor noise through policy_observation; without that hook the
        mean-action rollout is deterministic at fixed states, context and dt.
        """
        state, context = start_state.detach(), context.detach()
        horizon, batch_size = self.cfg.horizon, state.shape[0]
        dt = self.cfg.dt if dt is None else dt
        survival = torch.ones(batch_size, device=state.device, dtype=state.dtype)
        states, survivals = [state], [survival]
        actions, rewards, termination, timeouts, timeout_values = [], [], [], [], []
        timeout_fn = getattr(self.adapter, "timeout", getattr(self.adapter, "timeouts", None))
        for step in range(horizon):
            observation = getattr(self.adapter, "policy_observation", self.adapter.observation)
            mean, _ = self._policy(observation(state, context))
            action = self._action(mean)
            duration = self._dt_at(dt, step, horizon, batch_size)
            next_state = self.model.step(state, action, context, duration)
            hard_absorbing = getattr(self.adapter, "hard_absorbing", False)
            if hard_absorbing:
                # Keep already-ended rows at their endpoint. Do not generate
                # imaginary post-success/failed episodes or their rewards.
                next_state = torch.where((survival > 0)[:, None], next_state, state)
            reward = self.adapter.reward(state, next_state, action, context).reshape(batch_size)
            if hard_absorbing:
                outcome = self.adapter.task_outcome(state, next_state, action, context)
                done = (outcome["failed"] | outcome["completed"]).to(state.dtype).detach()
                reward = torch.where(survival > 0, reward, torch.zeros_like(reward))
            elif getattr(self.model, "term_head", True) is None:
                done = torch.zeros_like(survival)
            else:
                done = self.model.termination_probability(next_state, context).reshape(batch_size).clamp(0, 1)
            timeout = (torch.zeros_like(survival) if timeout_fn is None else
                       timeout_fn(next_state, context).to(dtype=state.dtype).reshape(batch_size).detach())
            if hard_absorbing:
                timeout = outcome["timeout"].to(state.dtype).detach()
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
    def update(self, start_state, context, weights=None, anchor_observations=None, *, dt=None):
        """One mean-policy Adam update, followed by independent imagined value fitting.

        Chunking changes memory use, not the number of Actor optimizer steps or
        the full-batch objective. Every chunk uses the same pre-update Actor and
        Critic; fixed value targets are collected before any Critic fitting.
        """
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
        if self.imagined_critic is not None:
            frozen.append(self.imagined_critic)
        frozen.extend(preprocessor for preprocessor in (self.state_preprocessor, self.value_preprocessor)
                      if isinstance(preprocessor, nn.Module))
        critic_batches = []
        with freeze_parameters(*frozen), _evaluation(self.actor):
            if anchor_observations is None:
                anchor_observations = self.adapter.observation(start_state.detach(), context.detach())
            anchor_observations = anchor_observations.detach()
            with torch.no_grad():
                old_mean, old_log_std = self._policy(anchor_observations)
                old_mean, old_log_std = old_mean.clone(), old_log_std.clone()
            self.optimizer.zero_grad(set_to_none=True)
            batch_size = len(start_state)
            chunk_size = int(self.cfg.chunk_size) or batch_size
            aggregate = {name: start_state.new_zeros(()) for name in
                         ("objective", "reward_mean", "survival_H", "tail_value", "timeout_fraction")}
            has_gradient = False
            for begin in range(0, batch_size, chunk_size):
                end = min(begin + chunk_size, batch_size)
                fraction = float(end - begin) / batch_size
                chunk_dt = self._slice_dt(dt, begin, end, self.cfg.horizon, batch_size)
                rollout = self.imagine(start_state[begin:end], context[begin:end], dt=chunk_dt)
                chunk_weights = None if weights is None else weights[begin:end]
                objective = imagination_objective(rollout, self.cfg.gamma, chunk_weights)
                loss = -objective * fraction / self.cfg.horizon
                if not bool(torch.isfinite(loss)):
                    self.optimizer.zero_grad(set_to_none=True)
                    self.rejected_updates += 1
                    return {"accepted": 0.0, "rollback": 0.0, "skipped": 1.0,
                            "nonfinite": 1.0, "kl": 0.0, "actor_grad_norm": 0.0}
                if loss.requires_grad:
                    loss.backward()
                    has_gradient = True
                if self.imagined_critic_trainer is not None and self.cfg.use_tail_value:
                    critic_batches.append(self.imagined_critic_trainer.prepare(rollout, context[begin:end]))
                aggregate["objective"] += objective.detach() * fraction
                aggregate["reward_mean"] += rollout.rewards.detach().mean() * fraction
                aggregate["survival_H"] += rollout.survival[-1].detach().mean() * fraction
                aggregate["tail_value"] += rollout.tail_values.detach().mean() * fraction
                aggregate["timeout_fraction"] += rollout.timeouts.detach().mean() * fraction
                # Only detached Critic training tensors survive the chunk.
                del rollout, objective, loss
            if not has_gradient:
                return {"accepted": 0.0, "rollback": 0.0, "skipped": 1.0,
                        "no_gradient": 1.0, "kl": 0.0, "actor_grad_norm": 0.0}
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            if not bool(torch.isfinite(grad_norm)):
                self.optimizer.zero_grad(set_to_none=True)
                self.rejected_updates += 1
                return {"accepted": 0.0, "rollback": 0.0, "skipped": 1.0,
                        "nonfinite": 1.0, "kl": 0.0, "actor_grad_norm": float(grad_norm)}
            if self.cfg.grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.cfg.grad_clip_value)
            # Snapshot on the current GPU; rollback must include Adam moments/step.
            # Even without a KL gate, a numerically invalid optimizer step must
            # not leave a corrupted Actor or Adam state in real PPO training.
            before_actor = {name: value.detach().clone() for name, value in self.actor.state_dict().items()}
            before_optimizer = deepcopy(self.optimizer.state_dict())
            self.optimizer.step()
            with torch.no_grad():
                new_mean, new_log_std = self._policy(anchor_observations)
                kl = diagonal_gaussian_kl(old_mean, old_log_std, new_mean, new_log_std).mean()
                finite = bool(torch.isfinite(new_mean).all()) and all(
                    bool(torch.isfinite(parameter).all()) for parameter in self.actor.parameters())
                kl_allowed = (not self.cfg.kl_rollback or
                              (bool(torch.isfinite(kl)) and float(kl) <= self.cfg.max_kl))
                accepted = finite and kl_allowed
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
                "kl": float(kl), "objective": float(aggregate["objective"]),
                "kl_gate_enabled": float(self.cfg.kl_rollback),
                "actor_grad_norm": float(grad_norm),
                "reward_mean": float(aggregate["reward_mean"]),
                "survival_H": float(aggregate["survival_H"]),
                "tail_value": float(aggregate["tail_value"]),
                "timeout_fraction": float(aggregate["timeout_fraction"]),
                "trust_mean": 1.0 if weights is None else float(weights.mean()),
                "dt_mean": float(actual_dt.float().mean()),
                "horizon_seconds_mean": float(actual_dt.float().mean()) * self.cfg.horizon,
                "effective_lr": float(self.optimizer.param_groups[0]["lr"]),
                "imagined_value_source": float(self.imagined_critic is not None),
                "actor_chunks": float((batch_size + chunk_size - 1) // chunk_size),
            }
        # At this point Actor and model BPTT is over, and parameter flags have
        # been restored. The detached data cannot send a Critic update into
        # the policy, dynamics, low controller or the real PPO Critic.
        if critic_batches:
            metrics.update(self.imagined_critic_trainer.fit(critic_batches))
        return metrics

    def set_lr(self, actor_lr, critic_lr=None):
        """Set effective optimizer LRs; caller owns the training schedule."""
        if not math.isfinite(float(actor_lr)) or actor_lr < 0:
            raise ValueError("BPTT learning rate must be finite and nonnegative")
        for group in self.optimizer.param_groups:
            group["lr"] = float(actor_lr)
        if self.imagined_critic_trainer is not None:
            self.imagined_critic_trainer.set_lr(actor_lr if critic_lr is None else critic_lr)

    def state_dict(self):
        result = {"optimizer": self.optimizer.state_dict(),
                  "accepted_updates": self.accepted_updates,
                  "rejected_updates": self.rejected_updates}
        if self.imagined_critic_trainer is not None:
            result["imagined_critic"] = self.imagined_critic_trainer.state_dict()
        return result

    def load_state_dict(self, state):
        if self.imagined_critic_trainer is not None and "imagined_critic" not in state:
            raise ValueError("Imagined value training checkpoint is missing its independent Critic")
        if self.imagined_critic_trainer is None and "imagined_critic" in state:
            raise ValueError("An imagined Critic checkpoint cannot be resumed with value_source=ppo")
        self.optimizer.load_state_dict(state["optimizer"])
        self.accepted_updates = int(state.get("accepted_updates", 0))
        self.rejected_updates = int(state.get("rejected_updates", 0))
        if self.imagined_critic_trainer is not None:
            self.imagined_critic_trainer.load_state_dict(state["imagined_critic"])
