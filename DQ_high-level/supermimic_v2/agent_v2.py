"""
SuperMimic v2 agent - REFERENCE-BASED VERSION

Uses networks adapted from reference implementation to avoid gradient issues.
Key change: forward() returns normalized output, denormalization done separately.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math
from contextlib import contextmanager

from .networks_v2 import WorldModelEnsemble, Policy, Critic
from .mdp import DQRobotMDP


@contextmanager
def _fixed_parameters(*modules):
    parameters = [p for module in modules for p in module.parameters()]
    flags = [p.requires_grad for p in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


class SuperMimicAgentV2:
    """SuperMimic agent with reference-based networks (gradient-safe)."""

    def __init__(
        self,
        mdp: DQRobotMDP,
        device: str = "cuda:0",
        # Network architecture
        wm_hidden: Tuple[int, ...] = (2048, 2048, 2048, 2048, 2048),
        policy_hidden: Tuple[int, ...] = (2048, 1024, 512, 256),
        critic_hidden: Tuple[int, ...] = (256, 128),
        # Training hyperparameters
        wm_ensemble_size: int = 5,
        wm_window: int = 8,
        horizon: int = 16,
        wm_batch_size: int = 2048,
        policy_batch_size: int = 4096,
        learning_rate: float = 2e-4,
        gamma: float = 0.99,
        lambda_return: float = 0.95,
        uncertainty_penalty: float = 0.5,
        critic_epochs: int = 5,
        critic_minibatches: int = 4,
        max_grad_norm: float = 1.0,
        # Data collection
        replay_capacity: int = 1000000,
        min_buffer_size: int = 10000,
        warmup_iterations: int = 1000,
    ):
        self.mdp = mdp
        self.device = device

        # Hyperparameters
        self.wm_window = wm_window
        self.horizon = horizon
        self.wm_batch_size = wm_batch_size
        self.policy_batch_size = policy_batch_size
        self.gamma = gamma
        self.lambda_return = lambda_return
        self.uncertainty_penalty = uncertainty_penalty
        self.critic_epochs = critic_epochs
        self.critic_minibatches = critic_minibatches
        self.max_grad_norm = max_grad_norm
        self.warmup_iterations = warmup_iterations

        # Networks with reference-based implementation
        self.wm = WorldModelEnsemble(
            input_dim=mdp.wm_input_dim,
            output_dim=mdp.delta_dim,
            hidden=wm_hidden,
            ensemble_size=wm_ensemble_size,
        ).to(device)

        self.policy = Policy(
            obs_dim=mdp.obs_dim,
            action_dim=mdp.action_dim,
            hidden=policy_hidden,
        ).to(device)

        self.critic = Critic(
            feature_dim=mdp.critic_dim,
            hidden=critic_hidden,
        ).to(device)

        # Optimizers
        self.wm_optimizer = torch.optim.Adam(self.wm.parameters(), lr=learning_rate)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Replay buffer
        self.replay_capacity = replay_capacity
        self.min_buffer_size = min_buffer_size
        self._init_replay_buffer()

        # Training counters
        self.world_model_updates = 0
        self.policy_updates = 0
        self.total_transitions = 0
        self.iteration = 0
        self.phase = "warmup"
        self._last_policy_grad_norm = 0.0

    def _init_replay_buffer(self):
        """Initialize replay buffer."""
        self.replay_buffer = {
            'state': torch.zeros(self.replay_capacity, self.mdp.state_dim, device=self.device),
            'action': torch.zeros(self.replay_capacity, self.mdp.action_dim, device=self.device),
            'reward': torch.zeros(self.replay_capacity, device=self.device),
            'next_state': torch.zeros(self.replay_capacity, self.mdp.state_dim, device=self.device),
            'done': torch.zeros(self.replay_capacity, device=self.device),
            'context': torch.zeros(self.replay_capacity, self.mdp.context_dim, device=self.device),
        }
        self.observation_slice = getattr(self.mdp, 'observation_slice', None)
        if self.observation_slice is not None:
            self.replay_buffer['obs'] = self.replay_buffer['state'][:, self.observation_slice]
            self.replay_buffer['next_obs'] = self.replay_buffer['next_state'][:, self.observation_slice]
        else:
            self.replay_buffer['obs'] = torch.zeros(self.replay_capacity, self.mdp.obs_dim, device=self.device)
            self.replay_buffer['next_obs'] = torch.zeros(self.replay_capacity, self.mdp.obs_dim, device=self.device)

        self.buffer_ptr = 0
        self.buffer_size = 0
        self.replay_num_envs = None
        self.replay_total_added = 0

    @torch.no_grad()
    def observe_transition(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
        context: torch.Tensor,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
    ):
        """Store transition in replay buffer."""
        batch_size = state.shape[0]
        if not batch_size:
            return
        if self.replay_num_envs is not None and batch_size != self.replay_num_envs:
            raise ValueError("Replay insertions must be consistent")

        data = dict(state=state, action=action, reward=reward, next_state=next_state,
                    done=done, context=context, obs=obs, next_obs=next_obs)

        for key, value in data.items():
            if key in ("reward", "done") and value.shape == (batch_size, 1):
                value = value.reshape(batch_size)
            data[key] = value.detach().to(device=self.device, dtype=self.replay_buffer[key].dtype)

        dropped = max(0, batch_size - self.replay_capacity)
        indices = (torch.arange(dropped, batch_size, device=self.device) + self.buffer_ptr) % self.replay_capacity

        for key, value in data.items():
            if self.observation_slice is not None and key in ('obs', 'next_obs'):
                continue
            self.replay_buffer[key][indices] = value[dropped:]

        self.buffer_ptr = (self.buffer_ptr + batch_size) % self.replay_capacity
        self.buffer_size = min(self.buffer_size + batch_size, self.replay_capacity)
        self.total_transitions += batch_size
        self.replay_total_added += batch_size
        self.replay_num_envs = batch_size
        self.policy.obs_norm.update(data['next_obs'])

    def sample_windows(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample temporal windows for world model training."""
        if not self.buffer_size or self.replay_num_envs is None:
            raise ValueError("Not enough data")

        oldest = self.replay_total_added - self.buffer_size
        available_steps = min(self.wm_window, 1 + (self.buffer_size - 1) // self.replay_num_envs)
        span = (available_steps - 1) * self.replay_num_envs
        stop = self.replay_total_added - span

        starts = torch.randint(oldest, stop, (batch_size,), device=self.device)
        absolute = starts[:, None] + torch.arange(self.wm_window, device=self.device) * self.replay_num_envs
        available = absolute < self.replay_total_added
        indices = absolute.clamp(max=self.replay_total_added - 1) % self.replay_capacity

        done = self.replay_buffer['done'][indices] > 0.5
        before = torch.cat((torch.ones(batch_size, 1, device=self.device),
                          (~done[:, :-1]).float().cumprod(dim=1)), dim=1)

        return {
            's': self.replay_buffer['state'][indices],
            'a': self.replay_buffer['action'][indices],
            'context': self.replay_buffer['context'][indices],
            's1': self.replay_buffer['next_state'][indices],
            'mask': before * available,
        }

    def update_world_model(self) -> Dict[str, float]:
        """
        ✅ FIXED: Open-loop WM training with reference-based networks.

        Key: forward() returns normalized output, we denormalize separately.
        """
        if self.buffer_size < self.min_buffer_size:
            return {"world_model_loss": 0.0, "skipped": True}

        windows = self.sample_windows(self.wm_batch_size)
        total_loss = 0.0
        total_valid_steps = 0
        grad_norm = 0.0

        for member_idx in range(self.wm.ensemble_size):
            state = windows['s'][:, 0].clone()
            batch_size = state.shape[0]

            padding_state = windows['s'][0, 0].detach()
            padding_action = windows['a'][0, 0].detach()
            padding_context = windows['context'][0, 0].detach()

            step_losses = []
            valid_count = 0

            for step in range(self.wm_window):
                active = windows['mask'][:, step] > 0.5

                if active.all():
                    state_input, action_input, context_input = state, windows['a'][:, step], windows['context'][:, step]
                else:
                    rows = active.unsqueeze(-1)
                    state_input = torch.where(rows, state, padding_state)
                    action_input = torch.where(rows, windows['a'][:, step], padding_action)
                    context_input = torch.where(rows, windows['context'][:, step], padding_context)

                target = windows['s1'][:, step]
                features = self.mdp.inputs(state_input, action_input, context_input)

                # Update normalization
                with torch.no_grad():
                    if active.any():
                        self.wm.in_norm.update(features[active])

                # Forward pass - returns normalized output
                predicted_normalized = self.wm.forward(features, member_idx)

                # Denormalize safely
                with torch.no_grad():
                    std = self.wm.out_norm.std()
                    mean = self.wm.out_norm.mean

                predicted_delta = predicted_normalized * std + mean
                predicted_state = state_input + predicted_delta

                # Compute target and update out_norm
                target_delta = self.mdp.targets(state_input, target)
                with torch.no_grad():
                    if active.any():
                        self.wm.out_norm.update(target_delta[active])

                error = predicted_delta - target_delta

                if not active.all():
                    rows = active.unsqueeze(-1)
                    error = torch.where(rows, error, torch.zeros_like(error))

                per_sample_loss = error.square().mean(dim=-1)
                step_losses.append(per_sample_loss.sum())
                valid_count += active.sum().item()

                if step < self.wm_window - 1:
                    state = predicted_state.detach()

            if step_losses:
                member_loss = sum(step_losses) / max(batch_size, 1)
                self.wm_optimizer.zero_grad(set_to_none=True)
                member_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.wm.nets[member_idx].parameters(), self.max_grad_norm, error_if_nonfinite=True
                )
                self.wm_optimizer.step()
                total_loss += member_loss.item()

            total_valid_steps += valid_count

        self.world_model_updates += 1

        return {
            "world_model_loss": total_loss / self.wm.ensemble_size,
            "wm_valid_steps": total_valid_steps / self.wm.ensemble_size,
            "wm_grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm),
            "skipped": False,
        }

    def sample_starts(self, batch_size: int, reset_fraction: float = 0.1) -> Dict[str, torch.Tensor]:
        """Sample episode starts for policy training."""
        if not self.buffer_size:
            raise ValueError("Cannot sample from empty buffer")

        num_resets = int(batch_size * reset_fraction)
        num_recent = batch_size - num_resets

        indices = []

        if num_recent > 0:
            recent_window = min(self.buffer_size, self.replay_num_envs * 100)
            start_idx = max(0, self.buffer_size - recent_window)
            recent_indices = torch.randint(start_idx, self.buffer_size, (num_recent,), device=self.device)
            indices.append(recent_indices)

        if num_resets > 0:
            done_mask = self.replay_buffer['done'][:self.buffer_size] > 0.5
            start_mask = torch.cat([torch.zeros(1, device=self.device, dtype=torch.bool), done_mask[:-1]])
            start_candidates = torch.where(start_mask)[0]

            if len(start_candidates) >= num_resets:
                perm = torch.randperm(len(start_candidates), device=self.device)[:num_resets]
                reset_indices = start_candidates[perm]
            else:
                reset_indices = torch.randint(0, self.buffer_size, (num_resets,), device=self.device)
            indices.append(reset_indices)

        all_indices = torch.cat(indices)

        return {
            'state': self.replay_buffer['state'][all_indices],
            'obs': self.replay_buffer['obs'][all_indices],
            'context': self.replay_buffer['context'][all_indices],
        }

    def update_policy(self) -> Dict[str, float]:
        """Update policy using imagination rollout."""
        if self.buffer_size < self.min_buffer_size:
            return {"policy_loss": 0.0, "critic_loss": 0.0, "skipped": True}

        batch = self.sample_starts(self.policy_batch_size, reset_fraction=0.1)
        state, obs, context = batch['state'], batch['obs'], batch['context']

        policy_loss, critic_loss, imagined_reward, imagined_survival, disagreement = \
            self._imagine_and_optimize(state, obs, context)

        self.policy_updates += 1

        return {
            "policy_loss": policy_loss,
            "critic_loss": critic_loss,
            "imagination_reward_mean": imagined_reward,
            "imagined_survival": imagined_survival,
            "ensemble_disagreement": disagreement,
            "policy_grad_norm": self._last_policy_grad_norm,
            "skipped": False,
        }

    def _imagine_and_optimize(self, initial_state, initial_obs, context):
        """Optimize policy using imagination."""
        states, observations, rewards, dones, disagreements = [initial_state], [initial_obs], [], [], []
        state, obs = initial_state, initial_obs

        with _fixed_parameters(self.wm, self.critic):
            for _ in range(self.horizon):
                action = self.policy(obs)
                members = torch.randint(self.wm.ensemble_size, (state.shape[0],), device=state.device)

                # Use reference-based imagine method
                next_state, disagreement = self.wm.imagine(state, action, context, self.mdp, members)

                reward, done = self.mdp.reward(state, action, next_state, context)
                reward = reward - self.uncertainty_penalty * disagreement
                next_obs = self.mdp.advance_obs(obs, next_state, action, context)

                states.append(next_state)
                observations.append(next_obs)
                rewards.append(reward)
                dones.append(done.to(dtype=reward.dtype))
                disagreements.append(disagreement)
                state, obs = next_state, next_obs

            rewards_tensor = torch.stack(rewards)
            dones_tensor = torch.stack(dones)
            alive = (1 - dones_tensor).cumprod(dim=0)
            before = torch.cat((torch.ones_like(alive[:1]), alive[:-1]), dim=0)
            discount = self.gamma ** torch.arange(self.horizon, device=state.device, dtype=rewards_tensor.dtype)[:, None]
            objective = (discount * before * rewards_tensor).sum(dim=0)

            final_features = self.mdp.critic_features(states[-1], context, observations[-1])
            objective = objective + self.gamma ** self.horizon * alive[-1] * self.critic(final_features).squeeze(-1)
            policy_loss = -objective.mean()

            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            gradient = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.policy_optimizer.step()
            self._last_policy_grad_norm = gradient.item()

        critic_loss = self._update_critic(
            torch.stack([v.detach() for v in states]), context.detach(),
            [v.detach() for v in observations], rewards_tensor.detach(),
            dones_tensor.detach(), before.detach()
        )

        with torch.no_grad():
            imagined_reward = (before * rewards_tensor).sum(dim=0).mean().item()
            imagined_survival = alive[-1].mean().item()
            disagreement_mean = (torch.stack(disagreements) * before).sum() / before.sum()

        return policy_loss.item(), critic_loss, imagined_reward, imagined_survival, disagreement_mean.item()

    def _update_critic(self, states, context, observations, rewards, dones, weights):
        """Fit TD(lambda) targets."""
        horizon, batch_size = rewards.shape

        with torch.no_grad():
            features = torch.stack([
                self.mdp.critic_features(state, context, obs)
                for state, obs in zip(states, observations)
            ])
            flat_weights = weights.reshape(-1)
            reachable = flat_weights > 0
            training_features = features[:-1].reshape(-1, features.shape[-1])[reachable]
            self.critic.norm.update(training_features)
            values = self.critic(features.reshape(-1, features.shape[-1])).reshape(horizon + 1, batch_size)
            targets = self._compute_lambda_returns(rewards, dones, values)
            targets = targets.reshape(-1)[reachable]
            flat_weights = flat_weights[reachable]

        total_loss, num_updates = 0.0, 0
        sample_count = training_features.shape[0]

        for _ in range(self.critic_epochs):
            permutation = torch.randperm(sample_count, device=training_features.device)
            for indices in permutation.tensor_split(min(self.critic_minibatches, sample_count)):
                predictions = self.critic(training_features[indices]).squeeze(-1)
                error = predictions - targets[indices]
                batch_weights = flat_weights[indices]
                loss = (error.square() * batch_weights).sum() / batch_weights.sum()

                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True)
                self.critic_optimizer.step()

                total_loss += loss.item()
                num_updates += 1

        return total_loss / max(num_updates, 1)

    def _compute_lambda_returns(self, rewards, dones, values):
        """TD(lambda) returns."""
        next_return = values[-1]
        returns = []
        for step in reversed(range(rewards.shape[0])):
            next_return = rewards[step] + self.gamma * (1 - dones[step].to(rewards.dtype)) * (
                (1 - self.lambda_return) * values[step + 1] + self.lambda_return * next_return
            )
            returns.append(next_return)
        return torch.stack(list(reversed(returns)))

    @torch.no_grad()
    def select_action(self, obs, deterministic=False):
        """Select action from policy."""
        return self.policy(obs, deterministic=deterministic)

    def get_stats(self):
        """Get training statistics."""
        return {
            'phase': self.phase,
            'world_model_updates': self.world_model_updates,
            'policy_updates': self.policy_updates,
            'buffer_stats': {
                'buffer_size': self.buffer_size,
                'buffer_utilization': self.buffer_size / self.replay_capacity,
            }
        }

    def save(self, path: str):
        """Save checkpoint."""
        torch.save({
            'wm': self.wm.state_dict(),
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'wm_optimizer': self.wm_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'world_model_updates': self.world_model_updates,
            'policy_updates': self.policy_updates,
            'phase': self.phase,
        }, path + '_agent.pt')

    def load(self, path: str):
        """Load checkpoint."""
        checkpoint = torch.load(path + '_agent.pt', map_location=self.device)
        self.wm.load_state_dict(checkpoint['wm'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.wm_optimizer.load_state_dict(checkpoint['wm_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.world_model_updates = checkpoint['world_model_updates']
        self.policy_updates = checkpoint['policy_updates']
        self.phase = checkpoint.get('phase', 'joint')
