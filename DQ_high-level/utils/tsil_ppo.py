"""Joint PPO and Temporal Self-Imitation for the SGS + EKF teacher.

The positive-gap loss below is adapted from Yinsen Jia and Boyuan Chen's
TSIL ``core/training/algo/tsil/loss.py`` (commit 137e3ab, Apache-2.0).
Modified here for skrl models, configuration-based archives, frozen scalers,
and exactly equivalent logical-batch/microbatch gradient accumulation.
The PPO terms follow this repository's vendored skrl PPO implementation.
"""

import copy
import itertools
import warnings

import torch

from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR

from utils.tsil_normalization import FrozenObservationScaler, RewardTraceNormalizer, ValueScaler
from utils.tsil import TemporalReplay


def positive_gap_losses(log_prob, predicted_values, normalized_returns, efficiency,
                        positive_weight_std=None):
    """Author's SIL loss, preserving the gradient of its one-sided value term.

``positive_weight_std`` is shared by every microbatch of a logical batch. Actor
weights are detached; neither efficiency nor weight clipping affects Critic.
    """
    positive_gap = (normalized_returns - predicted_values).clamp_min(0)
    weights = positive_gap.detach() * efficiency
    if positive_weight_std is None:
        active_weights = weights[weights > 0]
        positive_weight_std = (active_weights.std(unbiased=False) if active_weights.numel()
                               else weights.new_zeros(()))
    normalized = torch.clamp(weights / (positive_weight_std + 1e-8), max=1.0)
    weights = torch.where(weights > 0, normalized, weights)
    policy_loss = -(weights * log_prob).mean()
    value_loss = 0.5 * positive_gap.square().mean()
    return policy_loss, value_loss, positive_gap, weights


def compute_gae(rewards, dones, values, last_values, gamma, lam):
    """Finite-task GAE: task-budget termination has the same zero tail as failure."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(last_values)
    for index in reversed(range(rewards.shape[0])):
        following = values[index + 1] if index + 1 < rewards.shape[0] else last_values
        alive = (~dones[index]).to(rewards.dtype)
        carry = rewards[index] - values[index] + gamma * alive * (following + lam * carry)
        advantages[index] = carry
    returns = advantages + values
    normalized = (advantages - advantages.mean()) / (advantages.std(unbiased=advantages.numel() > 1) + 1e-8)
    return returns, normalized


def clipped_value_loss(values, old_values, targets, clip_enabled, clip_range):
    """TSIL author's conservative PPO value loss, retaining DQ's outer scale.

Adapted from ``core/training/algo/utils.py::compute_value_loss`` (137e3ab).
Clipping limits improvements relative to old values; the unclipped branch must
still penalize predictions that move further away from their targets.
    """
    errors = (values - targets).square()
    if clip_enabled:
        clipped = old_values + (values - old_values).clamp(-clip_range, clip_range)
        errors = torch.maximum(errors, (clipped - targets).square())
    return errors.mean()


class TSILPPO(PPO):
    """One optimizer step jointly uses new PPO samples and fixed historical samples."""

    def __init__(self, models, memory=None, observation_space=None, action_space=None,
                 device=None, cfg=None, tsil_env=None):
        if tsil_env is None or getattr(tsil_env, "tsil_state", None) is None:
            raise ValueError("TSILPPO needs an environment with a TemporalState")
        configured = copy.deepcopy(cfg or {})
        settings = configured.get("tsil", {})
        self.tsil_env = tsil_env
        self.tsil_state = tsil_env.tsil_state
        self.time_indices = tuple(settings.get("time_indices", (-191, -190)))
        self.coefficient = float(settings.get("coefficient", .05))
        self.value_coefficient = float(settings.get("value_coefficient", .05))
        self.samples_per_trajectory = int(settings.get("samples_per_trajectory", 32))
        self.micro_batch_size = int(settings.get("micro_batch_size", 2048))
        self.screen_batch_size = int(settings.get("screen_batch_size", 4096))
        if min(self.micro_batch_size, self.screen_batch_size, self.samples_per_trajectory) < 1:
            raise ValueError("TSIL sampling and microbatch sizes must be positive")
        if self.coefficient < 0 or self.value_coefficient < 0:
            raise ValueError("TSIL loss coefficients cannot be negative")
        if configured.get("time_limit_bootstrap", False):
            raise ValueError("TSIL preserves DQ's finite task budget; timeout bootstrap must be disabled")
        if configured.get("rewards_shaper") is not None:
            raise ValueError("TSIL reward relabeling requires unshaped task reward components")
        if configured.get("random_timesteps", 0):
            raise ValueError("TSIL's old likelihoods require policy actions from the first rollout")
        if configured.get("kl_threshold", 0):
            raise ValueError("TSIL traverses the full fixed replay set each epoch; use KLAdaptiveLR, not early stopping")
        configured["state_preprocessor"] = FrozenObservationScaler
        configured["state_preprocessor_kwargs"] = dict(size=observation_space, device=device,
                                                          time_indices=self.time_indices)
        configured["value_preprocessor"] = ValueScaler
        configured["value_preprocessor_kwargs"] = dict(size=1, device=device)
        super().__init__(models=models, memory=memory, cfg=configured, observation_space=observation_space,
                         action_space=action_space, device=device)
        self.reward_normalizer = RewardTraceNormalizer(tsil_env.num_envs, self._discount_factor,
                                                       device=self.device)
        self.checkpoint_modules["tsil_state"] = self.tsil_state
        self.checkpoint_modules["reward_normalizer"] = self.reward_normalizer
        self.replay = None
        if self.memory is not None:
            self.replay = TemporalReplay(
                self.tsil_state, num_envs=tsil_env.num_envs,
                observation_size=self._state_preprocessor.running_mean.numel(),
                action_size=self.policy.num_actions, gamma=self._discount_factor,
                topk=int(settings.get("top_k", 5)), time_indices=self.time_indices,
                max_episode_steps=int(tsil_env.max_episode_length),
                static_prefix_size=int(settings.get("static_prefix_size", 1024)),
                success_reward_scale=float(getattr(tsil_env, "reward_scales", {}).get("pick_up", 50.)))
        self._command_gate = False
        self._diagnostic_steps = []
        self._episode_advantage_mass = {}
        self._last_replay_selection = None
        self._legacy_value_evaluation_only = False

    def _predict_values(self, normalized_states):
        """All online/replay value paths share the same differentiable adapter."""
        if self._legacy_value_evaluation_only:
            raise RuntimeError("Legacy TSIL checkpoint supports policy evaluation only; Critic use is unavailable")
        values, _, _ = self.value.act({"states": normalized_states}, role="value")
        return self._value_preprocessor.normalize_prediction(values)

    def record_transition(self, states, actions, rewards, next_states, terminated, truncated,
                          infos, timestep, timesteps):
        # Keep the existing user-facing reward statistics in original task units.
        Agent.record_transition(self, states, actions, rewards, next_states, terminated, truncated,
                                infos, timestep, timesteps)
        if self.memory is None:
            return
        info = infos.get("tsil")
        if info is None:
            raise ValueError("TSIL transition is missing per-environment reward and configuration metadata")
        if bool(truncated.any()):
            raise ValueError("Administrative truncations must abort the recorder, not enter DQ task memory")
        self._command_gate = bool(info["command_gate"])
        self._current_next_states = next_states
        with torch.no_grad():
            normalized_states = self._state_preprocessor(states)
            values = self._predict_values(normalized_states)
            values = self._value_preprocessor(values, inverse=True)
            scaled_rewards = self.reward_normalizer(rewards)
            self._state_preprocessor.observe(states)
            episode_start = info.get("elapsed_before")
            episode_start = None if episode_start is None else episode_start <= 0
            self.reward_normalizer.observe(rewards, terminated, episode_start)
        samples = dict(states=states, actions=actions, rewards=scaled_rewards, next_states=next_states,
                       terminated=terminated, truncated=truncated, log_prob=self._current_log_prob, values=values)
        self.memory.add_samples(**samples)
        for memory in self.secondary_memories:
            memory.add_samples(**samples)
        self.replay.record(states, actions, rewards, terminated, info)
        self._diagnostic_steps.append({key: info[key].detach().clone()
                                       for key in ("config_id", "episode_id", "elapsed", "success")})
        self._diagnostic_steps[-1]["done"] = terminated.detach().reshape(-1).clone()

    @torch.no_grad()
    def _screen_gap(self, raw_states, scaled_returns):
        states = self._state_preprocessor(raw_states.to(device=self.device, dtype=torch.float32))
        target = self._value_preprocessor(scaled_returns.to(device=self.device, dtype=torch.float32))
        values = self._predict_values(states)
        return (target - values).clamp_min(0).reshape(-1).cpu()

    @torch.no_grad()
    def _logical_weight_std(self, selection, indices):
        # Population variance in float64 over the WHOLE logical replay minibatch,
        # including samples from all microbatches, with no mean subtraction in
        # the eventual actor weight. No model mutation occurs between passes.
        count = torch.zeros((), dtype=torch.float64, device=self.device)
        total = torch.zeros_like(count)
        squares = torch.zeros_like(count)
        for chunk in indices.split(self.micro_batch_size):
            batch = selection.get_batch(chunk, device=self.device)
            states = self._state_preprocessor(batch["states"])
            target = self._value_preprocessor(batch["returns"])
            values = self._predict_values(states)
            weights = (target - values).clamp_min(0) * batch["efficiency"]
            active = weights[weights > 0].double()
            count += active.numel()
            total += active.sum()
            squares += active.square().sum()
        safe_count = count.clamp_min(1)
        return (squares / safe_count - (total / safe_count).square()).clamp_min(0).sqrt().float()

    def _backward_replay(self, selection, indices):
        """Accumulate one logical SIL loss without changing optimizer frequency."""
        metrics = dict(policy=0., value=0., gap=0., positive=0., logp=0., success=0.)
        if indices.numel() == 0:
            return metrics
        weight_std = self._logical_weight_std(selection, indices)
        logical_count = indices.numel()
        for chunk in indices.split(self.micro_batch_size):
            batch = selection.get_batch(chunk, device=self.device)
            states = self._state_preprocessor(batch["states"])
            target = self._value_preprocessor(batch["returns"])
            _, log_prob, _ = self.policy.act({"states": states, "taken_actions": batch["actions"]}, role="policy")
            values = self._predict_values(states)
            actor, critic, gap, weights = positive_gap_losses(
                log_prob, values, target, batch["efficiency"], weight_std)
            fraction = chunk.numel() / logical_count
            (self.coefficient * fraction * (actor + self.value_coefficient * critic)).backward()
            metrics["policy"] += fraction * actor.detach().item()
            metrics["value"] += fraction * critic.detach().item()
            metrics["gap"] += fraction * gap.detach().mean().item()
            metrics["positive"] += fraction * (gap.detach() > 0).float().mean().item()
            metrics["logp"] += fraction * log_prob.detach().mean().item()
            metrics["success"] += fraction * batch["success"].float().mean().item()
        return metrics

    @torch.no_grad()
    def _reference_log_likelihood(self, selection, indices):
        total = 0.
        count = 0
        for chunk in indices.split(self.micro_batch_size):
            if not chunk.numel():
                continue
            batch = selection.get_batch(chunk, device=self.device)
            _, log_prob, _ = self.policy.act(
                {"states": self._state_preprocessor(batch["states"]),
                 "taken_actions": batch["actions"]}, role="policy")
            total += log_prob.sum().item()
            count += log_prob.numel()
        return total / max(count, 1)

    def _record_advantage_diagnostics(self, advantages):
        """Attribute actual PPO positive advantage to whole, possibly crossing episodes."""
        if not self._diagnostic_steps:
            return
        env_count = self.tsil_env.num_envs
        ids = torch.stack([step["episode_id"].reshape(-1) for step in self._diagnostic_steps])
        keys = ids.long() * env_count + torch.arange(env_count, device=ids.device)
        unique, inverse = torch.unique(keys.reshape(-1), return_inverse=True)
        masses = torch.zeros(unique.numel(), device=advantages.device)
        masses.index_add_(0, inverse, advantages.reshape(-1).clamp_min(0))
        for key, mass in zip(unique.cpu().tolist(), masses.cpu().tolist()):
            self._episode_advantage_mass[key] = self._episode_advantage_mass.get(key, 0.) + mass
        aggregates = {}
        families = getattr(self.tsil_state, "family_ids", None)
        families = None if families is None else families.detach().cpu().tolist()
        for step, step_keys in zip(self._diagnostic_steps, keys):
            ended = step["done"].bool()
            rows = torch.stack((step_keys[ended].double(), step["config_id"].reshape(-1)[ended].double(),
                                step["success"].reshape(-1)[ended].double(),
                                step["elapsed"].reshape(-1)[ended].double()), dim=1).cpu().tolist()
            for key, config_id, success, elapsed in rows:
                mass = self._episode_advantage_mass.pop(int(key), 0.)
                family = ("full_task", "pregrasp", "near_goal")[families[int(config_id)]] if families is not None else "all"
                for label in {family, "all"}:
                    sums = aggregates.setdefault(label, [0., 0., 0.])
                    sums[0] += mass
                    sums[1] += mass * success
                    sums[2] += mass * success * (1 - elapsed / self.tsil_env.tsil_time_reference)
        active = set(keys[-1].cpu().tolist())
        self._episode_advantage_mass = {key: value for key, value in self._episode_advantage_mass.items()
                                        if key in active}
        for label, (total, success, fast) in aggregates.items():
            self.track_data("TSIL PPO / {}/positive_mass".format(label), total)
            if total > 0:
                self.track_data("TSIL PPO / {}/F_success".format(label), success / total)
                self.track_data("TSIL PPO / {}/F_fast".format(label), fast / total)
        self._diagnostic_steps.clear()

    def _update(self, timestep, timesteps):
        with torch.no_grad():
            last_values = self._predict_values(
                self._state_preprocessor(self._current_next_states.float()))
            last_values = self._value_preprocessor(last_values, inverse=True)
            old_values = self.memory.get_tensor_by_name("values")
            returns, advantages = compute_gae(self.memory.get_tensor_by_name("rewards"),
                                              self.memory.get_tensor_by_name("terminated"), old_values,
                                              last_values, self._discount_factor, self._lambda)
            self._value_preprocessor.update_returns(returns)
            self.memory.set_tensor_by_name("values", self._value_preprocessor(old_values))
            self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns))
            self.memory.set_tensor_by_name("advantages", advantages)
        self._record_advantage_diagnostics(advantages)
        selection = self.replay.prepare(
            self._screen_gap, reward_scale=self.reward_normalizer.active_scale.item(),
            command_gate=self._command_gate, transitions_per_trajectory=self.samples_per_trajectory,
            screen_batch_size=self.screen_batch_size)
        self._last_replay_selection = selection
        # Diagnostic cap only; ALL selected transitions still train in every
        # epoch. These exact historical success actions are measured twice.
        reference_indices = selection.success.reshape(-1).nonzero(as_tuple=False).flatten()[:1024]
        reference_before = self._reference_log_likelihood(selection, reference_indices)
        totals = dict(policy=0., value=0., entropy=0., replay_policy=0., replay_value=0.,
                      replay_gap=0., replay_positive=0., replay_logp=0., replay_success=0.)
        updates = 0
        replay_updates = 0
        flat_memory = {name: self.memory.get_tensor_by_name(name).flatten(0, 1)
                       for name in self._tensors_names}
        batch_count = min(self._mini_batches, flat_memory["states"].shape[0])
        for epoch in range(self._learning_epochs):
            # Exactly the configured number of optimizer steps; tensor_split
            # retains a remainder instead of dropping data or making a seventh
            # minibatch for an unusual num_envs setting.
            batches = torch.tensor_split(torch.randperm(flat_memory["states"].shape[0],
                                                        device=self.device), batch_count)
            replay_batches = torch.tensor_split(torch.randperm(len(selection)), batch_count)
            divergences = []
            for online_indices, replay_indices in zip(batches, replay_batches):
                batch = [flat_memory[name][online_indices] for name in self._tensors_names]
                states, actions, old_log_prob, old_values, targets, advantages = batch
                states = self._state_preprocessor(states)
                _, log_prob, _ = self.policy.act({"states": states, "taken_actions": actions}, role="policy")
                log_ratio = log_prob - old_log_prob
                with torch.no_grad():
                    divergences.append(((log_ratio.exp() - 1) - log_ratio).mean())
                ratio = log_ratio.exp()
                actor = -torch.minimum(advantages * ratio,
                                       advantages * ratio.clamp(1 - self._ratio_clip, 1 + self._ratio_clip)).mean()
                entropy = (-self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                           if self._entropy_loss_scale else actor.new_zeros(()))
                values = self._predict_values(states)
                critic = self._value_loss_scale * clipped_value_loss(
                    values, old_values, targets, self._clip_predicted_values, self._value_clip)
                self.optimizer.zero_grad()
                (actor + critic + entropy).backward()
                replay_metrics = self._backward_replay(selection, replay_indices)
                if self._grad_norm_clip > 0:
                    parameters = (self.policy.parameters() if self.policy is self.value else
                                  itertools.chain(self.policy.parameters(), self.value.parameters()))
                    torch.nn.utils.clip_grad_norm_(parameters, self._grad_norm_clip)
                self.optimizer.step()
                totals["policy"] += actor.detach().item()
                totals["value"] += critic.detach().item()
                totals["entropy"] += entropy.detach().item()
                updates += 1
                if replay_indices.numel():
                    replay_updates += 1
                    for key, value in replay_metrics.items():
                        totals["replay_" + key] += value
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    self.scheduler.step(torch.stack(divergences).mean())
                else:
                    self.scheduler.step()
        if reference_indices.numel():
            reference_after = self._reference_log_likelihood(selection, reference_indices)
            self.track_data("TSIL Reference / success_action_count", reference_indices.numel())
            self.track_data("TSIL Reference / logp_before", reference_before)
            self.track_data("TSIL Reference / logp_after", reference_after)
            self.track_data("TSIL Reference / logp_change", reference_after - reference_before)
        for name in ("policy", "value", "entropy"):
            self.track_data("Loss / {} loss".format(name.capitalize()), totals[name] / max(updates, 1))
        for name in ("policy", "value", "gap", "positive", "logp", "success"):
            self.track_data("TSIL Replay / " + name, totals["replay_" + name] / max(replay_updates, 1))
        for key, value in self.replay.stats.items():
            if isinstance(value, (int, float)):
                self.track_data("TSIL Archive / " + key, value)
        for key, value in selection.stats.items():
            if isinstance(value, (int, float)):
                self.track_data("TSIL Selection / " + key, value)
        self.track_data("TSIL / replay_transitions", len(selection))
        self.track_data("TSIL / reward_scale", self.reward_normalizer.active_scale.item())
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
        # Only next reset/rollout may use newly discovered goals and scales.
        improved = self.tsil_state.commit_targets()
        self.track_data("TSIL Targets / improved_configurations", improved)
        self.track_data("TSIL Targets / successful_configurations", self.tsil_state.seen_success.sum().item())
        if self.tsil_state.targets.numel():
            self.track_data("TSIL Targets / minimum_seconds", self.tsil_state.targets.min().item())
            self.track_data("TSIL Targets / mean_seconds", self.tsil_state.targets.mean().item())
        self._state_preprocessor.commit()
        ratio = self.reward_normalizer.commit()
        self._value_preprocessor.rescale_units(ratio)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        required = {"policy", "value", "state_preprocessor", "value_preprocessor",
                    "tsil_state", "reward_normalizer"}
        missing = required.difference(checkpoint)
        if missing:
            raise ValueError("Incomplete TSIL checkpoint: missing {}".format(sorted(missing)))
        scaler_state = checkpoint["value_preprocessor"]
        metadata = checkpoint.get("tsil_metadata", {})
        declared_actor_only = (metadata.get("critic_available", True) is False
                               or metadata.get("migrated_from_version") == 1)
        if declared_actor_only and self.memory is not None:
            raise ValueError("Legacy TSIL checkpoint supports policy evaluation only, not Critic training")
        missing_adapter = ValueScaler.ADAPTER_STATE_KEYS.difference(scaler_state)
        legacy_missing_adapter = (missing_adapter == ValueScaler.ADAPTER_STATE_KEYS
                                  and self.memory is None and metadata.get("version") == 1)
        if missing_adapter and not legacy_missing_adapter:
            raise ValueError("Incomplete TSIL value adapter: missing {}; legacy checkpoints support "
                             "policy evaluation only with version-1 metadata".format(sorted(missing_adapter)))
        if legacy_missing_adapter:
            # The old Actor and observation transform are unchanged. Its old
            # clipped Critic cannot be reconstructed with an affine adapter;
            # allow Actor-only evaluation explicitly and prevent value use.
            checkpoint["value_preprocessor"] = dict(scaler_state)
            defaults = self._value_preprocessor.state_dict()
            for name in ValueScaler.ADAPTER_STATE_KEYS:
                checkpoint["value_preprocessor"][name] = defaults[name].clone()
        legacy_evaluation = legacy_missing_adapter or declared_actor_only
        if legacy_evaluation:
            warnings.warn("Loading legacy TSIL checkpoint for policy evaluation only; its Critic "
                          "is unavailable and training resume is unsupported", RuntimeWarning)
        self._legacy_value_evaluation_only = legacy_evaluation
        for name, data in checkpoint.items():
            module = self.checkpoint_modules.get(name)
            if module is not None:
                module.load_state_dict(data)
                if hasattr(module, "eval"):
                    module.eval()
