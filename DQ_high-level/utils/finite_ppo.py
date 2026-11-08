"""The repository's skrl PPO update with explicit numerical step rejection.

GAE, minibatch order, preprocessing, shared Adam and learning-rate scheduling
follow the vendored PPO. The clipped surrogate is evaluated in the log domain
before exp: an inactive overflowing branch must not create a NaN gradient.
No actions, advantages or genuine negative-advantage losses are silently capped.
"""

from copy import deepcopy
import math

import torch
from torch.nn import functional as F

from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR


class PPONumericalError(FloatingPointError):
    """A nonfinite value which must not reach a policy optimizer step."""


def _all_finite(value):
    """Check nested optimizer/model state with one host read per device."""
    groups = {}

    def visit(item):
        if torch.is_tensor(item):
            groups.setdefault(item.device, []).append(torch.isfinite(item).all())
        elif isinstance(item, dict):
            return all(visit(child) for child in item.values())
        elif isinstance(item, (tuple, list)):
            return all(visit(child) for child in item)
        elif isinstance(item, float):
            return math.isfinite(item)
        return True

    return visit(value) and all(bool(torch.stack(items).all()) for items in groups.values())


def clipped_policy_loss(log_ratio, advantages, ratio_clip):
    """Exact PPO surrogate, excluding the branch discarded by PPO clipping.

    For positive A the effective log ratio is min(log(r), log(1+eps)); for
    negative A it is max(log(r), log(1-eps)). A genuinely overflowing negative
    branch remains an error, rather than being changed into a capped objective.
    """
    if not 0 <= ratio_clip < 1:
        raise ValueError("FinitePPO requires ratio_clip in [0, 1)")
    if not _all_finite((log_ratio, advantages)):
        raise PPONumericalError("nonfinite_surrogate_input")
    effective = torch.where(advantages >= 0,
                            log_ratio.clamp_max(math.log1p(ratio_clip)),
                            log_ratio.clamp_min(math.log1p(-ratio_clip)))
    surrogate = advantages * effective.exp()
    if not _all_finite(surrogate):
        raise PPONumericalError("surrogate_overflow")
    return -surrogate.mean()


def _module_snapshot(module):
    return deepcopy(module.state_dict()) if isinstance(module, torch.nn.Module) else None


def _restore_module(module, snapshot):
    if snapshot is not None:
        module.load_state_dict(snapshot)


class FinitePPO(PPO):
    """Drop-in PPO with inspectable rejection reasons in last_ppo_numerics.

    An optional ``_report_progress(message)`` callback receives each rejection.
    Invalid initial parameters/Adam state raise immediately: there is no valid
    local snapshot from which this update could recover an already bad policy.

    Policies advertising ``supports_latent_ppo`` additionally replay their
    original Gaussian samples. Memory ``actions`` remains the executed action;
    ``latent_actions`` is the sampled Gaussian vector and ``log_prob`` its
    Gaussian density. For a fixed latent the transformation Jacobian cancels
    exactly in the PPO ratio, so no inverse opening or large Jacobian-density
    subtraction enters the update. Public ``act`` still returns the policy's
    execution density for compatibility with existing inference callers.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._latent_replay = bool(getattr(self.policy, "supports_latent_ppo", False))
        if self._latent_replay and self._random_timesteps:
            raise ValueError("Latent PPO requires random_timesteps=0; uniform random actions have no sampled Gaussian latent")
        self._current_latent_actions = None

    def init(self, trainer_cfg=None):
        super().init(trainer_cfg=trainer_cfg)
        if self._latent_replay and self.memory is not None:
            for memory in [self.memory, *self.secondary_memories]:
                memory.create_tensor(name="latent_actions", size=self.action_space, dtype=torch.float32)
            self._tensors_names.append("latent_actions")

    def act(self, states, timestep, timesteps):
        if not self._latent_replay:
            return super().act(states, timestep, timesteps)
        actions, execution_log_prob, outputs = self.policy.act(
            {"states": self._state_preprocessor(states), "latent_ppo": True}, role="policy")
        if "latent_actions" not in outputs or "latent_log_prob" not in outputs:
            raise ValueError("Policy advertised latent PPO support without returning its sample/density")
        self._current_latent_actions = outputs["latent_actions"].detach()
        # A fixed sampled latent has the same sigmoid Jacobian under both
        # policies. Cancel it analytically, before any float32 subtraction.
        self._current_log_prob = outputs["latent_log_prob"].detach()
        # Preserve the public action API; only PPO's internal likelihood
        # storage changes. The simulator still receives the same opening.
        return actions, execution_log_prob, outputs

    def record_transition(self, states, actions, rewards, next_states, terminated,
                          truncated, infos, timestep, timesteps):
        if not self._latent_replay:
            return super().record_transition(states, actions, rewards, next_states,
                terminated, truncated, infos, timestep, timesteps)
        latent = self._current_latent_actions
        if self.memory is not None and (latent is None or latent.shape != actions.shape):
            raise ValueError("Latent PPO transition must follow act() with its matching action batch")
        # Same bookkeeping and single memory write as the vendored PPO; an
        # additional add_samples call would incorrectly advance the ring twice.
        Agent.record_transition(self, states, actions, rewards, next_states,
                                terminated, truncated, infos, timestep, timesteps)
        if self.memory is not None:
            self._current_next_states = next_states
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)
            values, _, _ = self.value.act({"states": self._state_preprocessor(states)}, role="value")
            values = self._value_preprocessor(values, inverse=True)
            if self._time_limit_bootstrap:
                rewards += self._discount_factor * values * truncated
            for memory in [self.memory, *self.secondary_memories]:
                memory.add_samples(states=states, actions=actions, rewards=rewards, next_states=next_states,
                    terminated=terminated, truncated=truncated, log_prob=self._current_log_prob,
                    values=values, latent_actions=latent)
        self._current_latent_actions = None

    def _update(self, timestep, timesteps):
        stats = self.last_ppo_numerics = {
            "timestep": int(timestep), "attempted_minibatches": 0,
            "accepted_minibatches": 0, "rejected_minibatches": 0,
            "rejected_rollouts": 0, "rolled_back_steps": 0,
            "reason_counts": {}, "last_rejection": None,
            "nonfinite_kl_diagnostics": 0, "log_ratio_abs_max": 0.,
            "max_finite_grad_norm": 0.,
        }
        parameters = list(dict.fromkeys(parameter for group in self.optimizer.param_groups
                                        for parameter in group["params"]))

        def reject(reason, epoch=None, minibatch=None, *, rollout=False, rollback=False):
            self.optimizer.zero_grad(set_to_none=True)
            key = "rejected_rollouts" if rollout else "rejected_minibatches"
            stats[key] += 1
            stats["rolled_back_steps"] += int(rollback)
            stats["reason_counts"][reason] = stats["reason_counts"].get(reason, 0) + 1
            stats["last_rejection"] = {"reason": reason, "epoch": epoch, "minibatch": minibatch}
            callback = getattr(self, "_report_progress", None)
            if callable(callback):
                callback("PPO numerical rejection: {} (epoch={}, minibatch={}, rollback={})".format(
                    reason, epoch, minibatch, rollback))

        def record_numerics():
            for name, value in stats.items():
                if isinstance(value, (int, float)):
                    self.track_data("PPO numerics / " + name, value)
            for reason, count in stats["reason_counts"].items():
                self.track_data("PPO numerics / rejected_" + reason, count)

        initial_states = [("parameters", parameters), ("optimizer", self.optimizer.state_dict()),
                          ("policy_buffers", list(self.policy.buffers())),
                          ("value_buffers", list(self.value.buffers()))]
        initial_states.extend((label, module.state_dict()) for label, module in
                              (("state_preprocessor", self._state_preprocessor),
                               ("value_preprocessor", self._value_preprocessor))
                              if isinstance(module, torch.nn.Module))
        for label, value in initial_states:
            if not _all_finite(value):
                reject("initial_nonfinite_" + label, rollout=True)
                record_numerics()
                raise PPONumericalError("PPO entered with nonfinite {}; refusing to train or simulate".format(label))

        rewards = self.memory.get_tensor_by_name("rewards")
        dones = self.memory.get_tensor_by_name("terminated")
        values = self.memory.get_tensor_by_name("values")
        if not _all_finite((self._current_next_states, rewards, dones, values)):
            reject("nonfinite_rollout_input", rollout=True)
            record_numerics()
            return
        preprocessing = [(module, _module_snapshot(module)) for module in
                         (self._state_preprocessor, self._value_preprocessor)]
        with torch.no_grad():
            self.value.train(False)
            try:
                states = self._state_preprocessor(self._current_next_states.float())
                last_values, _, _ = self.value.act({"states": states}, role="value")
            finally:
                self.value.train(True)
            last_values = self._value_preprocessor(last_values, inverse=True)
        if not _all_finite((states, last_values)):
            for module, snapshot in preprocessing:
                _restore_module(module, snapshot)
            reject("nonfinite_bootstrap_value", rollout=True)
            record_numerics()
            return

        # Preserve the vendored finite-task GAE and advantage normalization.
        advantage = 0
        advantages = torch.zeros_like(rewards)
        not_dones = dones.logical_not()
        for index in reversed(range(rewards.shape[0])):
            next_values = values[index + 1] if index < rewards.shape[0] - 1 else last_values
            advantage = rewards[index] - values[index] + self._discount_factor * not_dones[index] * (
                next_values + self._lambda * advantage)
            advantages[index] = advantage
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if not _all_finite((returns, advantages)):
            reject("nonfinite_gae", rollout=True)
            record_numerics()
            return
        scaled_values = self._value_preprocessor(values, train=True)
        scaled_returns = self._value_preprocessor(returns, train=True)
        preprocessor_state = [module.state_dict() for module, _ in preprocessing
                              if isinstance(module, torch.nn.Module)]
        if not _all_finite((scaled_values, scaled_returns, preprocessor_state)):
            for module, snapshot in preprocessing:
                _restore_module(module, snapshot)
            reject("nonfinite_value_preprocessing", rollout=True)
            record_numerics()
            return
        self.memory.set_tensor_by_name("values", scaled_values)
        self.memory.set_tensor_by_name("returns", scaled_returns)
        self.memory.set_tensor_by_name("advantages", advantages)
        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        cumulative_policy_loss = cumulative_entropy_loss = cumulative_value_loss = 0.

        for epoch in range(self._learning_epochs):
            kl_divergences = []
            for minibatch, batch in enumerate(sampled_batches):
                stats["attempted_minibatches"] += 1
                if not _all_finite(batch):
                    reject("nonfinite_minibatch_input", epoch, minibatch)
                    continue
                sampled_states, sampled_actions, sampled_log_prob, sampled_values, sampled_returns, sampled_advantages = batch[:6]
                before_preprocessor = _module_snapshot(self._state_preprocessor)
                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                preprocessor_state = (self._state_preprocessor.state_dict()
                                      if isinstance(self._state_preprocessor, torch.nn.Module) else ())
                if not _all_finite((sampled_states, preprocessor_state)):
                    _restore_module(self._state_preprocessor, before_preprocessor)
                    reject("nonfinite_state_preprocessing", epoch, minibatch)
                    continue
                policy_inputs = {"states": sampled_states, "taken_actions": sampled_actions}
                if self._latent_replay:
                    if len(batch) != 7:
                        raise ValueError("Latent PPO memory is missing sampled latent actions")
                    policy_inputs.update(latent_ppo=True, taken_latent_actions=batch[6])
                _, next_log_prob, outputs = self.policy.act(policy_inputs, role="policy")
                if self._latent_replay:
                    if "latent_log_prob" not in outputs:
                        raise ValueError("Latent PPO replay did not return its Gaussian log probability")
                    next_log_prob = outputs["latent_log_prob"]
                log_ratio = next_log_prob - sampled_log_prob
                if not _all_finite((next_log_prob, log_ratio)):
                    reject("nonfinite_log_probability", epoch, minibatch)
                    continue
                stats["log_ratio_abs_max"] = max(stats["log_ratio_abs_max"], float(log_ratio.detach().abs().max()))
                with torch.no_grad():
                    kl_divergence = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    # +inf KL still correctly decreases KLAdaptiveLR's rate;
                    # it is a diagnostic, not an overflowing backward branch.
                    stats["nonfinite_kl_diagnostics"] += int(not bool(torch.isfinite(kl_divergence)))
                    kl_divergences.append(kl_divergence)
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break
                try:
                    policy_loss = clipped_policy_loss(log_ratio, sampled_advantages, self._ratio_clip)
                except PPONumericalError as error:
                    reject(str(error), epoch, minibatch)
                    continue
                entropy_loss = (-self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                                if self._entropy_loss_scale else 0)
                predicted_values, _, _ = self.value.act({"states": sampled_states}, role="value")
                if not _all_finite(predicted_values):
                    reject("nonfinite_value_prediction", epoch, minibatch)
                    continue
                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                        min=-self._value_clip, max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
                loss = policy_loss + entropy_loss + value_loss
                if not _all_finite((policy_loss, entropy_loss, value_loss, loss)):
                    reject("nonfinite_loss", epoch, minibatch)
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
                if not _all_finite(gradients):
                    reject("nonfinite_gradient", epoch, minibatch)
                    continue
                if self._grad_norm_clip > 0:
                    norm = torch.nn.utils.clip_grad_norm_(parameters, self._grad_norm_clip)
                    if not _all_finite((norm, gradients)):
                        reject("nonfinite_gradient_norm", epoch, minibatch)
                        continue
                    stats["max_finite_grad_norm"] = max(stats["max_finite_grad_norm"], float(norm))
                before_parameters = [parameter.detach().clone() for parameter in parameters]
                before_optimizer = deepcopy(self.optimizer.state_dict())
                self.optimizer.step()
                reason = ("nonfinite_parameter_after_step" if not _all_finite(parameters) else
                          "nonfinite_optimizer_after_step" if not _all_finite(self.optimizer.state_dict()) else None)
                if reason is not None:
                    with torch.no_grad():
                        for parameter, saved in zip(parameters, before_parameters):
                            parameter.copy_(saved)
                    self.optimizer.load_state_dict(before_optimizer)
                    reject(reason, epoch, minibatch, rollback=True)
                    continue
                stats["accepted_minibatches"] += 1
                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self._entropy_loss_scale:
                    cumulative_entropy_loss += entropy_loss.item()
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    # No valid likelihood in an epoch provides no KL evidence.
                    if kl_divergences:
                        self.scheduler.step(torch.tensor(kl_divergences).mean())
                else:
                    self.scheduler.step()

        denominator = self._learning_epochs * self._mini_batches
        self.track_data("Loss / Policy loss", cumulative_policy_loss / denominator)
        self.track_data("Loss / Value loss", cumulative_value_loss / denominator)
        if self._entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cumulative_entropy_loss / denominator)
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
        record_numerics()
