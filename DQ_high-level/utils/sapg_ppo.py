"""SAPG leader/follower PPO for the Scout & Relay teacher.

Singla, Agarwal, Pathak, "SAPG: Split and Aggregate Policy Gradients", ICML 2024,
Sections 4.1-4.6 and Algorithm 1; design: doc/SGS_SAPG_scout_relay_design.md.

* All branches share the actor and critic backbones; each has its own learned
  condition vector and log-std (see Policy/Value in train_multistate_DQ_teacher.py).
* Followers (scouts) are updated with their own on-policy PPO loss only.
* The leader (branch 0) uses its own on-policy data plus an equal amount of
  follower data with the SAPG off-policy objective (Eq. 3/4, lambda = coef):
      L_off = E_follower[ mu * min(rho * A, clip(rho, 1 - eps, 1 + eps) * A) ],
      mu = pi_L_old / pi_follower (fixed), rho = pi_L / pi_L_old,
  which equals SAPG's clip of r = mu * rho to [mu (1 - eps), mu (1 + eps)] with
  the usual PPO sign handling. A is the old leader's one-step TD residual and the
  critic target is r + gamma * V_L_old(s') (Eq. 6/8). On-policy targets keep this
  repository's GAE; that is a deliberate port, not an exact SAPG reproduction.
* Per-sample weights equal those of the on-policy minibatch, so with on:off = 1:1
  the leader's on- and off-policy samples count the same (SAPG lambda = 1).

Diagnostics: per-branch KL/std/loss, the batch IS effective sample size, and a
per-configuration ESS (hypothesis H1: follower samples on configurations where the
follower beats the leader have low ESS under the leader).
"""

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR

from utils.scout_relay import ISOverlapTracker, LEADER


SAPG_AGENT_DEFAULTS = {
    "off_policy_coef": 1.0,   # SAPG lambda; 0 disables the IS channel (ablation)
    "entropy_coef": 0.0,      # sigma; follower j adds sigma * j * entropy, leader none
    "mu_max": 0.0,            # optional truncation of mu (0 = faithful SAPG, untruncated)
    "chunk_size": 16384,      # forward chunk for old-leader quantities
    "gap_threshold": 0.3,     # H1 diagnostics: "follower clearly better"
    "gap_min_episodes": 20,
}


def finite_gae(rewards, dones, values, last_values, gamma, lam):
    """Same recursion as the vendored skrl PPO, without normalization."""
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_dones = dones.logical_not()
    steps = rewards.shape[0]
    for index in reversed(range(steps)):
        next_values = values[index + 1] if index < steps - 1 else last_values
        advantage = rewards[index] - values[index] + gamma * not_dones[index] * (next_values + lam * advantage)
        advantages[index] = advantage
    return advantages + values, advantages


class SAPGPPO(PPO):
    def __init__(self, models, memory=None, observation_space=None, action_space=None,
                 device=None, cfg=None, sapg_env=None):
        if sapg_env is None or getattr(sapg_env, "sapg_num_branches", 1) < 2:
            raise ValueError("SAPGPPO needs an environment with at least two SAPG branches")
        super().__init__(models=models, memory=memory, observation_space=observation_space,
                         action_space=action_space, device=device, cfg=cfg)
        settings = dict(SAPG_AGENT_DEFAULTS)
        settings.update({key: value for key, value in self.cfg.get("sapg", {}).items()
                         if key in SAPG_AGENT_DEFAULTS})
        self.sapg_env = sapg_env
        self.num_branches = int(sapg_env.sapg_num_branches)
        for name, model in (("policy", self.policy), ("value", self.value)):
            if getattr(model, "num_branches", 1) != self.num_branches:
                raise ValueError(f"The {name} model must have {self.num_branches} SAPG branches")
        self.branch_ids = sapg_env.sapg_branch_id.to(self.device).long()
        self.off_policy_coef = float(settings["off_policy_coef"])
        self.entropy_coef = float(settings["entropy_coef"])
        self.mu_max = float(settings["mu_max"])
        self.chunk_size = int(settings["chunk_size"])
        self.gap_threshold = float(settings["gap_threshold"])
        self.gap_min_episodes = int(settings["gap_min_episodes"])
        if self.off_policy_coef < 0 or self.entropy_coef < 0 or self.mu_max < 0 or self.chunk_size < 1:
            raise ValueError("Invalid SAPG agent settings")
        self.overlap = ISOverlapTracker(sapg_env.sgs_bank.num_configs, device=self.device)
        # The environment exports these per-configuration statistics at the end.
        sapg_env.sapg_overlap_tracker = self.overlap

    def init(self, trainer_cfg=None):
        super().init(trainer_cfg=trainer_cfg)
        if self.memory is not None:
            self.memory.create_tensor(name="branch_ids", size=1, dtype=torch.long)
            self.memory.create_tensor(name="config_ids", size=1, dtype=torch.long)
            self._tensors_names = ["states", "actions", "log_prob", "values", "returns",
                                   "advantages", "branch_ids"]

    # ----------------------------------------------------------------- rollout
    def act(self, states, timestep, timesteps):
        if timestep < self._random_timesteps:
            return self.policy.random_act({"states": self._state_preprocessor(states)}, role="policy")
        actions, log_prob, outputs = self.policy.act(
            {"states": self._state_preprocessor(states), "branch_ids": self.branch_ids}, role="policy")
        self._current_log_prob = log_prob
        return actions, log_prob, outputs

    def record_transition(self, states, actions, rewards, next_states, terminated, truncated,
                          infos, timestep, timesteps):
        Agent.record_transition(self, states, actions, rewards, next_states, terminated,
                                truncated, infos, timestep, timesteps)
        if self.memory is None:
            return
        self._current_next_states = next_states
        if self._rewards_shaper is not None:
            rewards = self._rewards_shaper(rewards, timestep, timesteps)
        values, _, _ = self.value.act(
            {"states": self._state_preprocessor(states), "branch_ids": self.branch_ids}, role="value")
        values = self._value_preprocessor(values, inverse=True)
        if self._time_limit_bootstrap:
            rewards += self._discount_factor * values * truncated
        # Configuration of the episode that produced this transition (before reset).
        config_ids = self.sapg_env.active_config_id.to(self.device).view(-1, 1)
        samples = dict(states=states, actions=actions, rewards=rewards, next_states=next_states,
                       terminated=terminated, truncated=truncated, log_prob=self._current_log_prob,
                       values=values, branch_ids=self.branch_ids.view(-1, 1), config_ids=config_ids)
        self.memory.add_samples(**samples)
        for memory in self.secondary_memories:
            memory.add_samples(**samples)

    # ------------------------------------------------------------------ update
    def _leader_ids(self, count):
        return torch.full((count,), LEADER, dtype=torch.long, device=self.device)

    def _normalize_per_branch(self, advantages):
        normalized = torch.empty_like(advantages)
        for branch in range(self.num_branches):
            columns = self.branch_ids == branch
            values = advantages[:, columns]
            normalized[:, columns] = (values - values.mean()) / (values.std() + 1e-8)
        return normalized

    @torch.no_grad()
    def _prepare_off_policy(self, rewards, dones):
        """Fix the old leader's view of an equal amount of follower data (Algorithm 1)."""
        steps, num_envs = rewards.shape[0], rewards.shape[1]
        follower_columns = (self.branch_ids != LEADER).nonzero(as_tuple=False).flatten()
        num_leader = int((self.branch_ids == LEADER).sum().item())
        available = steps * follower_columns.numel()
        count = min(steps * num_leader, available)
        if count == 0:
            return None
        choice = torch.randperm(available, device=self.device)[:count]
        time = choice // follower_columns.numel()
        columns = follower_columns[choice % follower_columns.numel()]
        flat = time * num_envs + columns

        states = self.memory.get_tensor_by_name("states").reshape(steps * num_envs, -1)
        actions = self.memory.get_tensor_by_name("actions").reshape(steps * num_envs, -1)
        behavior_log_prob = self.memory.get_tensor_by_name("log_prob").reshape(-1, 1)[flat]
        config_ids = self.memory.get_tensor_by_name("config_ids").reshape(-1)[flat]
        next_flat = ((time + 1) * num_envs + columns).clamp(max=steps * num_envs - 1)
        final = time == steps - 1
        next_states_last = self._current_next_states.float()

        old_log_prob = torch.empty(count, 1, device=self.device)
        old_values = torch.empty(count, 1, device=self.device)
        next_values = torch.empty(count, 1, device=self.device)
        for start in range(0, count, self.chunk_size):
            part = slice(start, min(start + self.chunk_size, count))
            ids = self._leader_ids(part.stop - part.start)
            sampled = self._state_preprocessor(states[flat[part]])
            _, log_prob, _ = self.policy.act(
                {"states": sampled, "taken_actions": actions[flat[part]], "branch_ids": ids}, role="policy")
            old_log_prob[part] = log_prob
            value, _, _ = self.value.act({"states": sampled, "branch_ids": ids}, role="value")
            old_values[part] = value
            successor = torch.where(final[part, None], next_states_last[columns[part]], states[next_flat[part]])
            value, _, _ = self.value.act(
                {"states": self._state_preprocessor(successor), "branch_ids": ids}, role="value")
            next_values[part] = value
        old_values_raw = self._value_preprocessor(old_values, inverse=True)
        next_values_raw = self._value_preprocessor(next_values, inverse=True)
        reward = rewards.reshape(-1, 1)[flat]
        alive = dones.reshape(-1, 1)[flat].logical_not()
        target_raw = reward + self._discount_factor * alive * next_values_raw
        advantage = target_raw - old_values_raw

        log_mu = (old_log_prob - behavior_log_prob).clamp(-20.0, 20.0)
        mu_untruncated = log_mu.exp()
        mu = mu_untruncated.clamp(max=self.mu_max) if self.mu_max > 0 else mu_untruncated
        self.overlap.update(config_ids, log_mu)
        first, second = mu_untruncated.sum(), mu_untruncated.square().sum()
        diagnostics = {
            "SAPG / off-policy samples": float(count),
            "SAPG / batch ESS fraction": (first.square() / (count * second).clamp_min(1e-30)).item(),
            "SAPG / mu mean": mu_untruncated.mean().item(),
            "SAPG / mu (max)": mu_untruncated.max().item(),
            "SAPG / mu below 0.1 fraction": (mu_untruncated < 0.1).float().mean().item(),
            "SAPG / mu above 10 fraction": (mu_untruncated > 10).float().mean().item(),
            "SAPG / off-policy TD advantage mean": advantage.mean().item(),
        }
        normalized = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        return {"flat": flat, "states": states, "actions": actions, "old_log_prob": old_log_prob,
                "mu": mu, "advantages": normalized, "target_raw": target_raw,
                "old_values_raw": old_values_raw, "diagnostics": diagnostics}

    def _track_overlap_diagnostics(self):
        ess = self.overlap.ess()
        valid = torch.isfinite(ess)
        if not bool(valid.any()):
            return
        self.track_data("SAPG / config ESS median", torch.nanmedian(ess).item())
        gap, gap_valid = (value.to(self.device) for value in
                          self.sapg_env.scout_relay_gap(self.gap_min_episodes))
        both = valid & gap_valid
        high = both & (gap >= self.gap_threshold)
        low = both & (gap < self.gap_threshold)
        self.track_data("SAPG / configs with ESS and gap", both.sum().item())
        if bool(high.any()):
            self.track_data(f"SAPG / config ESS median (gap>={self.gap_threshold:g})", ess[high].median().item())
        if bool(low.any()):
            self.track_data(f"SAPG / config ESS median (gap<{self.gap_threshold:g})", ess[low].median().item())
        if int(both.sum().item()) >= 3:
            x, y = gap[both], ess[both]
            x, y = x - x.mean(), y - y.mean()
            correlation = (x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-12)
            self.track_data("SAPG / corr(gap, config ESS)", correlation.item())

    def _update(self, timestep, timesteps):
        with torch.no_grad():
            self.value.train(False)
            last_values, _, _ = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float()),
                 "branch_ids": self.branch_ids}, role="value")
            self.value.train(True)
        last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        rewards = self.memory.get_tensor_by_name("rewards")
        dones = self.memory.get_tensor_by_name("terminated")
        returns, advantages = finite_gae(rewards, dones, values, last_values,
                                         self._discount_factor, self._lambda)
        advantages = self._normalize_per_branch(advantages)
        # Old-network quantities are fixed before any optimizer or scaler update.
        off = self._prepare_off_policy(rewards, dones) if self.off_policy_coef > 0 else None

        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)
        if off is not None:
            off["targets"] = self._value_preprocessor(off["target_raw"], train=False)
            off["old_values"] = self._value_preprocessor(off["old_values_raw"], train=False)

        sampled_batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        num_batches = len(sampled_batches)
        branches = self.num_branches
        sums = {name: torch.zeros(branches, device=self.device) for name in ("kl", "policy", "count")}
        cumulative = {"policy": 0.0, "value": 0.0, "entropy": 0.0, "off_policy": 0.0,
                      "off_value": 0.0, "off_clip": 0.0}
        updates = 0

        for epoch in range(self._learning_epochs):
            kl_divergences = []
            off_batches = (torch.randperm(off["flat"].numel(), device=self.device).chunk(num_batches)
                           if off is not None else None)
            for index, (sampled_states, sampled_actions, sampled_log_prob, sampled_values,
                        sampled_returns, sampled_advantages, sampled_branches) in enumerate(sampled_batches):
                sampled_branches = sampled_branches.view(-1)
                sampled_states = self._state_preprocessor(sampled_states, train=not epoch)
                _, next_log_prob, _ = self.policy.act(
                    {"states": sampled_states, "taken_actions": sampled_actions,
                     "branch_ids": sampled_branches}, role="policy")
                with torch.no_grad():
                    log_ratio = next_log_prob - sampled_log_prob
                    kl_samples = ((torch.exp(log_ratio) - 1) - log_ratio).view(-1)
                    kl_divergence = kl_samples.mean()
                    kl_divergences.append(kl_divergence)
                    sums["kl"].scatter_add_(0, sampled_branches, kl_samples)
                    sums["count"].scatter_add_(0, sampled_branches, torch.ones_like(kl_samples))
                if self._kl_threshold and kl_divergence > self._kl_threshold:
                    break

                if self.entropy_coef:
                    entropy = self.policy.get_entropy(role="policy").sum(dim=-1)
                    coefficients = self.entropy_coef * sampled_branches.to(entropy.dtype)
                    entropy_loss = -(coefficients * entropy).mean()
                else:
                    entropy_loss = 0

                ratio = torch.exp(next_log_prob - sampled_log_prob)
                surrogate = sampled_advantages * ratio
                surrogate_clipped = sampled_advantages * torch.clip(ratio, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip)
                per_sample = -torch.min(surrogate, surrogate_clipped)
                policy_loss = per_sample.mean()
                with torch.no_grad():
                    sums["policy"].scatter_add_(0, sampled_branches, per_sample.view(-1))

                predicted_values, _, _ = self.value.act(
                    {"states": sampled_states, "branch_ids": sampled_branches}, role="value")
                if self._clip_predicted_values:
                    predicted_values = sampled_values + torch.clip(predicted_values - sampled_values,
                                                                   min=-self._value_clip, max=self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(sampled_returns, predicted_values)
                loss = policy_loss + entropy_loss + value_loss

                if off is not None and index < len(off_batches):
                    selected = off_batches[index]
                    flat = off["flat"][selected]
                    leader = self._leader_ids(selected.numel())
                    off_states = self._state_preprocessor(off["states"][flat])
                    _, leader_log_prob, _ = self.policy.act(
                        {"states": off_states, "taken_actions": off["actions"][flat],
                         "branch_ids": leader}, role="policy")
                    rho = torch.exp(leader_log_prob - off["old_log_prob"][selected])
                    off_advantages = off["advantages"][selected]
                    off_surrogate = off["mu"][selected] * torch.min(
                        rho * off_advantages,
                        torch.clip(rho, 1.0 - self._ratio_clip, 1.0 + self._ratio_clip) * off_advantages)
                    # Same per-sample weight as an on-policy sample of this minibatch.
                    on_count = sampled_states.shape[0]
                    off_policy_loss = -off_surrogate.sum() / on_count
                    off_predicted, _, _ = self.value.act({"states": off_states, "branch_ids": leader}, role="value")
                    old = off["old_values"][selected]
                    if self._clip_predicted_values:
                        off_predicted = old + torch.clip(off_predicted - old, min=-self._value_clip,
                                                         max=self._value_clip)
                    off_value_loss = (self._value_loss_scale
                                      * (off["targets"][selected] - off_predicted).square().sum() / on_count)
                    loss = loss + self.off_policy_coef * (off_policy_loss + off_value_loss)
                    cumulative["off_policy"] += off_policy_loss.item()
                    cumulative["off_value"] += off_value_loss.item()
                    cumulative["off_clip"] += ((rho - 1.0).abs() > self._ratio_clip).float().mean().item()

                self.optimizer.zero_grad()
                loss.backward()
                if self._grad_norm_clip > 0:
                    if self.policy is self.value:
                        nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)
                    else:
                        nn.utils.clip_grad_norm_(itertools.chain(self.policy.parameters(), self.value.parameters()),
                                                 self._grad_norm_clip)
                self.optimizer.step()

                updates += 1
                cumulative["policy"] += policy_loss.item()
                cumulative["value"] += value_loss.item()
                if self.entropy_coef:
                    cumulative["entropy"] += entropy_loss.item()

            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    self.scheduler.step(torch.tensor(kl_divergences).mean())
                else:
                    self.scheduler.step()

        denominator = max(updates, 1)
        self.track_data("Loss / Policy loss", cumulative["policy"] / denominator)
        self.track_data("Loss / Value loss", cumulative["value"] / denominator)
        if self.entropy_coef:
            self.track_data("Loss / Entropy loss", cumulative["entropy"] / denominator)
        if off is not None:
            self.track_data("Loss / Off-policy loss", cumulative["off_policy"] / denominator)
            self.track_data("Loss / Off-policy value loss", cumulative["off_value"] / denominator)
            self.track_data("SAPG / off-policy clip fraction", cumulative["off_clip"] / denominator)
            for name, value in off["diagnostics"].items():
                self.track_data(name, value)
        counts = sums["count"].clamp_min(1)
        kl = (sums["kl"] / counts).tolist()
        losses = (sums["policy"] / counts).tolist()
        # The Gaussian head clamps log-std when acting; report the effective value.
        log_std = self.policy.log_std_parameter.detach()
        bounds = getattr(self.policy, "_g_log_std_min", {}), getattr(self.policy, "_g_log_std_max", {})
        if "" in bounds[0] and "" in bounds[1]:
            log_std = log_std.clamp(bounds[0][""], bounds[1][""])
        stds = log_std.exp().mean(dim=-1).tolist()
        for branch in range(branches):
            role = "leader" if branch == LEADER else f"scout{branch}"
            self.track_data(f"SAPG / KL {role}", kl[branch])
            self.track_data(f"SAPG / policy loss {role}", losses[branch])
            self.track_data(f"SAPG / action std mean {role}", stds[branch])
        if self.write_interval > 0:
            self._track_overlap_diagnostics()
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
