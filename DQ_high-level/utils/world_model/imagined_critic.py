"""Privileged value learning on detached imagined trajectories only.

The MLP and fixed-target TD(lambda) update follow SuperMimic-Reproduced
``policy/nets.py`` and ``imagination/{objective,policy_trainer}.py`` (MIT;
see THIRD_PARTY_LICENSES.md). The finite DQ task retains its hard success,
failure and timeout boundaries. This value function never shares parameters,
optimizer state or normalization statistics with the real PPO Critic.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .dynamics import MLP


class ImaginedCritic(nn.Module):
    """Value in raw task-reward units; feature construction belongs to the adapter."""

    def __init__(self, feature_dim, hidden=(256, 128)):
        super().__init__()
        self.feature_dim = int(feature_dim)
        if self.feature_dim < 1 or any(int(width) < 1 for width in hidden):
            raise ValueError("Imagined Critic dimensions must be positive")
        self.mlp = MLP(self.feature_dim, hidden, 1)

    def forward(self, features):
        if features.shape[-1] != self.feature_dim:
            raise ValueError("Privileged Critic feature width does not match its adapter")
        return self.mlp(features).squeeze(-1)


@torch.no_grad()
def lambda_returns(rewards, terminated, timeouts, values, gamma=.99, lam=.95,
                   bootstrap_timeouts=False):
    """Fixed TD(lambda) targets, keeping the terminating transition's reward.

    ``values`` is [H+1, B]; all other tensors are [H, B]. A finite-task
    timeout has zero bootstrap by default. If requested explicitly, its
    endpoint value is used once, without continuing into subsequent states.
    Natural success/failure always suppresses any timeout bootstrap.
    """
    if (rewards.ndim != 2 or terminated.shape != rewards.shape
            or timeouts.shape != rewards.shape
            or values.shape != (rewards.shape[0] + 1, rewards.shape[1])):
        raise ValueError("TD(lambda) expects matching [H,B] transitions and [H+1,B] values")
    if not 0 < gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("Invalid TD(lambda) discount or lambda")
    target = values[-1].clone()
    targets = torch.empty_like(rewards)
    for step in range(rewards.shape[0] - 1, -1, -1):
        not_done = 1. - terminated[step].to(values.dtype)
        timeout = timeouts[step].to(values.dtype)
        continuation = not_done * (1. - timeout)
        target = rewards[step] + gamma * continuation * (
            (1. - lam) * values[step + 1] + lam * target)
        if bootstrap_timeouts:
            target = target + gamma * not_done * timeout * values[step + 1]
        targets[step] = target
    return targets


@dataclass
class CriticBatch:
    features: torch.Tensor
    targets: torch.Tensor
    old_values: torch.Tensor


class ImaginedCriticTrainer:
    """Compute targets once, then fit five epochs over four shuffled minibatches."""

    def __init__(self, critic, adapter, *, lr=2e-4, gamma=.99, lam=.95,
                 epochs=5, minibatches=4, max_grad_norm=1., bootstrap_timeouts=False):
        if (not math.isfinite(float(lr)) or not math.isfinite(float(max_grad_norm))
                or lr <= 0 or epochs < 1 or minibatches < 1 or max_grad_norm <= 0):
            raise ValueError("Invalid imagined Critic optimization settings")
        if not 0 < gamma <= 1 or not 0 <= lam <= 1:
            raise ValueError("Invalid imagined Critic return settings")
        self.critic, self.adapter = critic, adapter
        self.gamma, self.lam = gamma, lam
        self.epochs, self.minibatches = int(epochs), int(minibatches)
        self.max_grad_norm = float(max_grad_norm)
        self.bootstrap_timeouts = bool(bootstrap_timeouts)
        self.optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
        self.updates = 0

    @torch.no_grad()
    def prepare(self, rollout, context):
        """Detach the graph and freeze targets before any Critic optimizer step."""
        states = rollout.states.detach()
        horizon_plus_one, batch_size = states.shape[:2]
        conditions = context.detach().unsqueeze(0).expand(horizon_plus_one, *context.shape)
        features = self.adapter.critic_features(
            states.reshape(-1, states.shape[-1]), conditions.reshape(-1, context.shape[-1]))
        features = features.reshape(horizon_plus_one, batch_size, -1).detach()
        values = self.critic(features)
        targets = lambda_returns(rollout.rewards.detach(), rollout.termination.detach(),
                                 rollout.timeouts.detach(), values, self.gamma, self.lam,
                                 self.bootstrap_timeouts)
        # Hard absorbing rows are memory padding, not subsequent transitions
        # of an episode. Keep the endpoint transition but exclude its padding.
        valid = rollout.survival[:-1].detach() > 0
        return CriticBatch(features[:-1][valid], targets[valid], values[:-1][valid])

    @torch.enable_grad()
    def fit(self, batches):
        """Fit the detached, fixed dataset after the Actor update has finished."""
        if isinstance(batches, CriticBatch):
            batches = [batches]
        if not batches:
            return {"critic_updates": 0., "critic_samples": 0.}
        features = torch.cat([batch.features.detach() for batch in batches], 0)
        targets = torch.cat([batch.targets.detach() for batch in batches], 0)
        old_values = torch.cat([batch.old_values.detach() for batch in batches], 0)
        count = len(targets)
        if not count:
            return {"critic_updates": 0., "critic_samples": 0.}
        if not bool(torch.isfinite(features).all() & torch.isfinite(targets).all()):
            return {"critic_updates": 0., "critic_samples": float(count),
                    "critic_skipped_nonfinite": 1.}
        was_training = self.critic.training
        self.critic.train()
        losses, norms = [], []
        skipped = 0
        try:
            for _ in range(self.epochs):
                permutation = torch.randperm(count, device=features.device)
                # tensor_split includes the final remainder instead of silently
                # discarding samples when a smoke-test batch is not divisible.
                for indices in torch.tensor_split(permutation, min(self.minibatches, count)):
                    prediction = self.critic(features[indices])
                    loss = torch.nn.functional.mse_loss(prediction, targets[indices])
                    self.optimizer.zero_grad(set_to_none=True)
                    if not bool(torch.isfinite(loss)):
                        skipped += 1
                        continue
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    if not bool(torch.isfinite(norm)):
                        self.optimizer.zero_grad(set_to_none=True)
                        skipped += 1
                        continue
                    self.optimizer.step()
                    self.updates += 1
                    losses.append(loss.detach())
                    norms.append(norm.detach())
        finally:
            self.critic.train(was_training)
        return {"critic_loss": float(torch.stack(losses).mean()) if losses else 0.,
                "critic_grad_norm": float(torch.stack(norms).mean()) if norms else 0.,
                "critic_value_mean": float(old_values.mean()),
                "critic_target_mean": float(targets.mean()),
                "critic_updates": float(len(losses)), "critic_samples": float(count),
                "critic_skipped_nonfinite": float(skipped),
                "critic_effective_lr": float(self.optimizer.param_groups[0]["lr"])}

    def set_lr(self, learning_rate):
        if not math.isfinite(float(learning_rate)) or learning_rate <= 0:
            raise ValueError("Imagined Critic learning rate must be finite and positive")
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def state_dict(self):
        return {"critic": self.critic.state_dict(), "optimizer": self.optimizer.state_dict(),
                "updates": self.updates}

    def load_state_dict(self, state):
        self.critic.load_state_dict(state["critic"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))
