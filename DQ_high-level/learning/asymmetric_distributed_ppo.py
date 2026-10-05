"""Synchronous data-parallel version of the vendored SKRL Gaussian PPO.

Each rank collects an equal local rollout. Gradients are averaged before norm
clipping and Adam; global advantages and KL keep all optimizers/schedulers in
step. No model wrappers are used, so checkpoint and deployment keys stay plain.
"""
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from skrl.agents.torch.ppo import PPO
from skrl.resources.schedulers.torch import KLAdaptiveLR
from utils import asymmetric_distributed as distributed


class DistributedAsymmetricPPO(PPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not distributed.active():
            raise ValueError("DistributedAsymmetricPPO requires an initialized process group")
        self._parameters = list(dict.fromkeys(
            parameter for model in (self.policy, self.value) for parameter in model.parameters()
            if parameter.requires_grad))

    def track_data(self, tag, value):
        # Non-writer ranks must not accumulate unbounded tracking lists.
        if distributed.is_main():
            super().track_data(tag, value)

    @torch.no_grad()
    def _average_gradients(self):
        # Include absent gradients in the same collective on every rank.
        present = torch.tensor([p.grad is not None for p in self._parameters],
                               dtype=torch.int32, device=self.device)
        dist.all_reduce(present)
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                          for p in self._parameters])
        dist.all_reduce(flat)
        flat /= distributed.world_size()
        if not torch.isfinite(flat).all():
            raise FloatingPointError("Non-finite distributed PPO gradients")
        offset = 0
        for parameter, used in zip(self._parameters, present.tolist()):
            length = parameter.numel()
            parameter.grad = flat[offset:offset + length].view_as(parameter) if used else None
            offset += length

    def _update(self, timestep, timesteps):
        with torch.no_grad():
            self.value.eval()
            last_values = self.value.act(
                {"states": self._state_preprocessor(self._current_next_states.float())}, role="value")[0]
            self.value.train()
            last_values = self._value_preprocessor(last_values, inverse=True)
            values = self.memory.get_tensor_by_name("values")
            rewards = self.memory.get_tensor_by_name("rewards")
            alive = ~self.memory.get_tensor_by_name("terminated")
            advantages = torch.zeros_like(rewards)
            advantage = torch.zeros_like(last_values)
            for index in reversed(range(rewards.shape[0])):
                next_value = values[index + 1] if index + 1 < rewards.shape[0] else last_values
                advantage = (rewards[index] - values[index] + self._discount_factor * alive[index]
                             * (next_value + self._lambda * advantage))
                advantages[index] = advantage
            returns = advantages + values
            average, variance, count = distributed.moments(advantages.reshape(-1, 1))
            # Match the original PPO's sample standard deviation (Bessel correction).
            std = (variance * count / max(1, count - 1)).sqrt().to(advantages)
            advantages = (advantages - average.to(advantages)) / (std + 1e-8)
            self.memory.set_tensor_by_name("advantages", advantages)
            self.memory.set_tensor_by_name("values", self._value_preprocessor(values))
            self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns))

        batches = self.memory.sample_all(names=self._tensors_names, mini_batches=self._mini_batches)
        losses = torch.zeros(3, device=self.device)
        updates = 0
        for _ in range(self._learning_epochs):
            divergences = []
            for states, actions, old_log_prob, old_values, targets, advantages in batches:
                states = self._state_preprocessor(states)
                _, log_prob, _ = self.policy.act({"states": states, "taken_actions": actions}, role="policy")
                with torch.no_grad():
                    delta = log_prob - old_log_prob
                    kl = distributed.mean((delta.exp() - 1 - delta).mean())
                    divergences.append(kl)
                # Every rank makes the same stop/scheduler decision.
                if self._kl_threshold and kl > self._kl_threshold:
                    break
                entropy_loss = (-self._entropy_loss_scale * self.policy.get_entropy(role="policy").mean()
                                if self._entropy_loss_scale else torch.zeros((), device=self.device))
                ratio = (log_prob - old_log_prob).exp()
                policy_loss = -torch.minimum(advantages * ratio, advantages * ratio.clamp(
                    1 - self._ratio_clip, 1 + self._ratio_clip)).mean()
                predictions = self.value.act({"states": states}, role="value")[0]
                if self._clip_predicted_values:
                    predictions = old_values + (predictions - old_values).clamp(-self._value_clip, self._value_clip)
                value_loss = self._value_loss_scale * F.mse_loss(targets, predictions)
                self.optimizer.zero_grad()
                (policy_loss + value_loss + entropy_loss).backward()
                self._average_gradients()
                if self._grad_norm_clip > 0:
                    nn.utils.clip_grad_norm_(self._parameters, self._grad_norm_clip)
                self.optimizer.step()
                losses += torch.stack((policy_loss.detach(), value_loss.detach(), entropy_loss.detach()))
                updates += 1
            if self._learning_rate_scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    self.scheduler.step(torch.stack(divergences).mean())
                else:
                    self.scheduler.step()
        losses = distributed.mean(losses / max(1, updates))
        for label, loss in zip(("Policy loss", "Value loss", "Entropy loss"), losses.tolist()):
            self.track_data("Loss / " + label, loss)
        if self._learning_rate_scheduler:
            self.track_data("Learning / Learning rate", self.scheduler.get_last_lr()[0])
