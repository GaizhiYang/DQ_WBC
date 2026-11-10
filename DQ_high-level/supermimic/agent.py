"""
SuperMimic Agent for DQ_WBC
Integrates world model, policy, critic, and replay buffer
Implements the complete SuperMimic training loop
"""
import torch
import torch.nn as nn
from contextlib import contextmanager
import warnings
from typing import Dict, Tuple, Optional
import numpy as np

from .world_model import EnsembleWorldModel, WorldModelTrainer
from .policy import SuperMimicPolicy, SuperMimicCritic, SuperMimicPolicyTrainer
from .replay_buffer import ReplayBuffer
from .state_adapter import DQStateAdapter


@contextmanager
def _fixed_parameters(*modules):
    """Keep input gradients for BPTT without accumulating model/value gradients."""
    parameters = [parameter for module in modules for parameter in module.parameters()]
    requires_grad = [parameter.requires_grad for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, enabled in zip(parameters, requires_grad):
            parameter.requires_grad_(enabled)


class SuperMimicAgent:
    """
    SuperMimic agent for DQ_WBC dynamic object grasping.

    Training Loop:
    1. Collect real transitions using policy with exploration noise
    2. Store in replay buffer
    3. Train world model on sampled sequences
    4. Imagination rollout through world model
    5. Update policy via BPTT with imagination rewards
    6. Update critic with TD(λ) targets
    """

    def __init__(self,
                 state_dim: int = 51,
                 action_dim: int = 12,
                 device: str = 'cuda:0',
                 # World model hyperparameters
                 num_ensemble: int = 5,
                 world_model_hidden: list = [512, 512, 512],
                 world_model_lr: float = 3e-4,
                 # Policy hyperparameters
                 policy_hidden: list = [256, 256, 256],
                 policy_lr: float = 3e-4,
                 critic_lr: float = 1e-3,
                 action_std_init: float = 0.3,
                 # Training hyperparameters
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 imagination_horizon: int = 16,
                 world_model_batch_size: int = 256,
                 world_model_train_steps: int = 10,
                 imagination_batch_size: int = 512,
                 policy_update_steps: int = 5,
                 # Replay buffer
                 replay_capacity: int = 1000000,
                 min_buffer_size: int = 10000):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.imagination_horizon = imagination_horizon
        self.world_model_batch_size = world_model_batch_size
        self.world_model_train_steps = world_model_train_steps
        self.imagination_batch_size = imagination_batch_size
        self.policy_update_steps = policy_update_steps
        self.min_buffer_size = min_buffer_size

        # Initialize components
        self.world_model = EnsembleWorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            num_ensemble=num_ensemble,
            hidden_dims=world_model_hidden,
            device=device
        )

        self.policy = SuperMimicPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=policy_hidden,
            action_std_init=action_std_init,
            device=device
        )

        self.critic = SuperMimicCritic(
            state_dim=state_dim,
            hidden_dims=policy_hidden,
            device=device
        )

        self.replay_buffer = ReplayBuffer(
            capacity=replay_capacity,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )

        # Trainers
        self.world_model_trainer = WorldModelTrainer(
            self.world_model,
            learning_rate=world_model_lr
        )

        self.policy_trainer = SuperMimicPolicyTrainer(
            self.policy,
            self.critic,
            policy_lr=policy_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            gae_lambda=gae_lambda
        )

        # State adapter (will be set externally)
        self.state_adapter = None

        # Training statistics
        self.iteration = 0
        self.warmup_iterations = 1000  # World model warmup phase

    def attach_state_adapter(self, adapter: DQStateAdapter):
        """Attach state adapter for reward computation."""
        self.state_adapter = adapter

    def select_action(self, state: torch.Tensor, exploration_noise: float = 0.0,
                     deterministic: bool = False) -> torch.Tensor:
        """
        Select action from policy with optional exploration noise.

        Args:
            state: [batch_size, state_dim]
            exploration_noise: Std of Gaussian noise to add
            deterministic: If True, use mean action

        Returns:
            action: [batch_size, action_dim]
        """
        self.policy.eval()
        with torch.no_grad():
            action = self.policy.get_action(state, deterministic=deterministic)

            if exploration_noise > 0 and not deterministic:
                noise = torch.randn_like(action) * exploration_noise
                action = torch.clamp(action + noise, -1.0, 1.0)

        return action

    def observe_transition(self, state: torch.Tensor, action: torch.Tensor,
                          reward: torch.Tensor, next_state: torch.Tensor,
                          done: torch.Tensor, native_reward: Optional[torch.Tensor] = None):
        """
        Store transition in replay buffer.

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            reward: [batch_size]
            next_state: [batch_size, state_dim]
            done: [batch_size]
            native_reward: [batch_size] (optional)
        """
        self.replay_buffer.add(state, action, reward, next_state, done, native_reward)

    def update_world_model(self) -> Dict[str, float]:
        """
        Train world model on sequences from replay buffer.

        Returns:
            info: Training statistics
        """
        if not self.replay_buffer.is_ready(self.min_buffer_size):
            return {'world_model_loss': 0.0, 'buffer_size': len(self.replay_buffer)}

        total_info = {}

        for _ in range(self.world_model_train_steps):
            if self.replay_buffer.can_sample_sequences(seq_length=8):
                batch = self.replay_buffer.sample_sequences(
                    batch_size=self.world_model_batch_size,
                    seq_length=8
                )
                info = self.world_model_trainer.train_sequence_step(
                    batch['states'], batch['actions'], batch['next_states']
                )
            else:
                # Parallel envs can fill min_buffer_size before eight time steps
                # exist. The current sequence loss is also single-step MSE, so
                # individual transitions remain valid training data.
                batch = self.replay_buffer.sample(
                    min(self.world_model_batch_size, len(self.replay_buffer))
                )
                info = self.world_model_trainer.train_step(
                    batch['states'], batch['actions'], batch['next_states']
                )

            # Accumulate statistics
            for key, value in info.items():
                total_info[key] = total_info.get(key, 0.0) + value / self.world_model_train_steps

        total_info['buffer_size'] = len(self.replay_buffer)
        return total_info

    def imagination_rollout_with_rewards(self, initial_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform imagination rollout through world model and compute rewards.

        Args:
            initial_states: [batch_size, state_dim]

        Returns:
            states: [batch_size, horizon+1, state_dim]
            actions: [batch_size, horizon, action_dim]
            rewards: [batch_size, horizon]
        """
        batch_size = initial_states.shape[0]

        # Generate actions for imagination horizon
        actions_list = []
        states_list = [initial_states]
        rewards_list = []

        current_state = initial_states

        for t in range(self.imagination_horizon):
            # Sample action from policy (keep gradients for BPTT)
            action, _ = self.policy(current_state, deterministic=False)
            actions_list.append(action)

            # Predict next state with world model (keep gradients)
            next_state = self.world_model(current_state, action)
            states_list.append(next_state)

            # Compute geometric proxy reward (differentiable)
            reward = self.state_adapter.compute_geometric_proxy_reward(current_state, next_state)
            rewards_list.append(reward)

            current_state = next_state

        states = torch.stack(states_list, dim=1)  # [batch_size, horizon+1, state_dim]
        actions = torch.stack(actions_list, dim=1)  # [batch_size, horizon, action_dim]
        rewards = torch.stack(rewards_list, dim=1)  # [batch_size, horizon]

        return states, actions, rewards

    def update_policy_via_imagination(self) -> Dict[str, float]:
        """
        Update policy via BPTT through imagined rollouts.

        Returns:
            info: Training statistics
        """
        if not self.replay_buffer.is_ready(max(self.min_buffer_size, self.imagination_batch_size)):
            return {'policy_loss': 0.0}

        total_info = {}
        for _ in range(self.policy_update_steps):
            batch = self.replay_buffer.sample(self.imagination_batch_size)
            initial_states = batch['states']

            # Each optimizer step needs a new rollout graph. Freezing parameters
            # still lets reward and terminal-value gradients reach the actor
            # through the world model and every imagined state.
            with _fixed_parameters(self.world_model, self.critic):
                states, _, rewards = self.imagination_rollout_with_rewards(initial_states)
                values = self.critic.get_value(states.reshape(-1, self.state_dim))
                values = values.reshape(initial_states.shape[0], self.imagination_horizon + 1)
                discounts = self.gamma ** torch.arange(
                    self.imagination_horizon, device=rewards.device, dtype=rewards.dtype
                )
                imagined_returns = (rewards * discounts).sum(dim=1)
                imagined_returns = imagined_returns + self.gamma ** self.imagination_horizon * values[:, -1]

                # Critic targets and inputs must not reuse the actor's graph:
                # the actor backward consumes it and its optimizer mutates it.
                with torch.no_grad():
                    _, returns = self.policy_trainer.compute_gae(
                        rewards.detach(), values[:, :-1].detach(), values[:, 1:].detach(),
                        torch.zeros_like(rewards, dtype=torch.bool)
                    )
                states_for_training = states[:, :-1].detach().reshape(-1, self.state_dim)
                policy_info = self.policy_trainer.update_policy_bptt(imagined_returns)

            critic_info = self.policy_trainer.update_critic(
                states_for_training, returns.reshape(-1)
            )
            info = {**policy_info, **critic_info}
            info['imagination_reward_mean'] = rewards.detach().mean().item()
            info['imagination_reward_std'] = rewards.detach().std(unbiased=False).item()
            for key, value in info.items():
                total_info[key] = total_info.get(key, 0.0) + value / self.policy_update_steps

        return total_info

    def update(self) -> Dict[str, float]:
        """
        Perform one complete SuperMimic update iteration.

        Returns:
            info: Combined training statistics
        """
        self.iteration += 1
        info = {}

        # Phase 1: World model warmup (first 1000 iterations)
        if self.iteration <= self.warmup_iterations:
            wm_info = self.update_world_model()
            info.update(wm_info)
            info['phase'] = 'warmup'
            return info

        # Phase 2: Joint training
        # Step 1: Update world model
        wm_info = self.update_world_model()
        info.update(wm_info)

        # Step 2: Update policy via imagination
        policy_info = self.update_policy_via_imagination()
        info.update(policy_info)

        info['phase'] = 'joint_training'
        info['iteration'] = self.iteration

        return info

    def save(self, path_prefix: str):
        """Save all components."""
        self.world_model_trainer.save(f"{path_prefix}_world_model.pt")
        self.policy_trainer.save(f"{path_prefix}_policy_critic.pt")

        # Save replay buffer state
        torch.save({
            'buffer_states': self.replay_buffer.states[:self.replay_buffer.size],
            'buffer_actions': self.replay_buffer.actions[:self.replay_buffer.size],
            'buffer_rewards': self.replay_buffer.rewards[:self.replay_buffer.size],
            'buffer_next_states': self.replay_buffer.next_states[:self.replay_buffer.size],
            'buffer_dones': self.replay_buffer.dones[:self.replay_buffer.size],
            'buffer_native_rewards': self.replay_buffer.native_rewards[:self.replay_buffer.size],
            'buffer_ptr': self.replay_buffer.ptr,
            'buffer_size': self.replay_buffer.size,
            'buffer_capacity': self.replay_buffer.capacity,
            'buffer_num_envs': self.replay_buffer.num_envs,
            'buffer_total_added': self.replay_buffer.total_added,
            'iteration': self.iteration
        }, f"{path_prefix}_buffer.pt")

    def load(self, path_prefix: str):
        """Load all components."""
        self.world_model_trainer.load(f"{path_prefix}_world_model.pt")
        self.policy_trainer.load(f"{path_prefix}_policy_critic.pt")

        # Load replay buffer
        buffer_state = torch.load(f"{path_prefix}_buffer.pt", map_location=self.device)
        self.iteration = buffer_state['iteration']
        self.replay_buffer.clear()
        if 'buffer_num_envs' not in buffer_state or 'buffer_total_added' not in buffer_state:
            warnings.warn(
                "Loaded model/optimizer weights, but discarded legacy replay data without "
                "environment/time metadata; collect fresh transitions before updating."
            )
            return
        size = buffer_state['buffer_size']
        if size > self.replay_buffer.capacity or (size and
                buffer_state.get('buffer_capacity', self.replay_buffer.capacity) != self.replay_buffer.capacity):
            raise ValueError("Replay capacity must match the checkpoint to preserve circular-buffer ordering")
        self.replay_buffer.states[:size] = buffer_state['buffer_states']
        self.replay_buffer.actions[:size] = buffer_state['buffer_actions']
        self.replay_buffer.rewards[:size] = buffer_state['buffer_rewards']
        self.replay_buffer.next_states[:size] = buffer_state['buffer_next_states']
        self.replay_buffer.dones[:size] = buffer_state['buffer_dones']
        self.replay_buffer.native_rewards[:size] = buffer_state['buffer_native_rewards']
        self.replay_buffer.ptr = buffer_state['buffer_ptr']
        self.replay_buffer.size = size
        self.replay_buffer.num_envs = buffer_state['buffer_num_envs']
        self.replay_buffer.total_added = buffer_state['buffer_total_added']

    def get_stats(self) -> Dict[str, any]:
        """Get comprehensive statistics."""
        stats = {
            'iteration': self.iteration,
            'phase': 'warmup' if self.iteration <= self.warmup_iterations else 'joint_training',
            'buffer_stats': self.replay_buffer.get_stats(),
            'world_model_train_steps': self.world_model_trainer.train_steps,
            'policy_train_steps': self.policy_trainer.train_steps
        }
        return stats
