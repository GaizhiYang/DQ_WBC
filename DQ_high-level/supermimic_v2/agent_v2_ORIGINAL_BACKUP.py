"""
SuperMimic v2 agent for the DQ environment adapter.

Training components:
- World model training with state_error()
- Imagination with uncertainty penalty
- Policy optimization with survival weighting
- Critic training with TD(λ)
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
    """
    Ensemble dynamics, differentiable policy rollouts, and value learning.
    """

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
        """
        Initialize SuperMimic agent.

        Args:
            mdp: MDP interface
            device: Torch device
            ... (see parameters above)
        """
        self.mdp = mdp
        self.device = device
        for name, value in (
            ("wm_ensemble_size", wm_ensemble_size), ("wm_window", wm_window),
            ("horizon", horizon), ("wm_batch_size", wm_batch_size),
            ("policy_batch_size", policy_batch_size), ("critic_epochs", critic_epochs),
            ("critic_minibatches", critic_minibatches), ("replay_capacity", replay_capacity),
            ("min_buffer_size", min_buffer_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if min_buffer_size > replay_capacity:
            raise ValueError("min_buffer_size cannot exceed replay_capacity")
        if warmup_iterations < 0:
            raise ValueError("warmup_iterations must be nonnegative")

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

        # Networks
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

        # Phase tracking
        self.phase = "warmup"  # "warmup" or "joint"

    def _init_replay_buffer(self):
        """Initialize replay buffer"""
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
            # Full model states already contain the actual policy observation.
            # Views avoid allocating a second copy of both million-row arrays.
            self.replay_buffer['obs'] = self.replay_buffer['state'][:, self.observation_slice]
            self.replay_buffer['next_obs'] = self.replay_buffer['next_state'][:, self.observation_slice]
            if self.replay_buffer['obs'].shape != (self.replay_capacity, self.mdp.obs_dim):
                raise ValueError('The model-state observation slice must match obs_dim')
        else:
            for key in ('obs', 'next_obs'):
                self.replay_buffer[key] = torch.zeros(
                    self.replay_capacity, self.mdp.obs_dim, device=self.device
                )
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
        """
        Store transition in replay buffer.

        Args:
            state: [N, state_dim]
            action: [N, action_dim]
            reward: [N] (native reward, for logging only)
            next_state: [N, state_dim]
            done: [N]
            context: [N, context_dim]
            obs: [N, obs_dim]
            next_obs: [N, obs_dim]
        """
        batch_size = state.shape[0]
        if not batch_size:
            return
        if self.replay_num_envs is not None and batch_size != self.replay_num_envs:
            raise ValueError("Replay insertions must contain one timestep for the same environments")
        data = dict(state=state, action=action, reward=reward, next_state=next_state,
                    done=done, context=context, obs=obs, next_obs=next_obs)
        for key, value in data.items():
            expected = (batch_size,) + self.replay_buffer[key].shape[1:]
            if key in ("reward", "done") and value.shape == (batch_size, 1):
                value = value.reshape(batch_size)
            if value.shape != expected:
                raise ValueError(f"Replay {key} must have shape {expected}, got {tuple(value.shape)}")
            data[key] = value.detach().to(device=self.device, dtype=self.replay_buffer[key].dtype)

        if self.observation_slice is not None:
            for state_key, obs_key in (('state', 'obs'), ('next_state', 'next_obs')):
                if not torch.equal(data[state_key][:, self.observation_slice], data[obs_key]):
                    raise ValueError(f'{obs_key} must match its captured model-state observation')

        # Keep only the newest capacity entries if one vector step is larger
        # than the ring. Absolute transition indices retain the env/time stride.
        dropped = max(0, batch_size - self.replay_capacity)
        indices = (torch.arange(dropped, batch_size, device=self.device) + self.buffer_ptr) % self.replay_capacity
        for key, value in data.items():
            if self.observation_slice is not None and key in ('obs', 'next_obs'):
                continue  # These buffers are views into the states just stored.
            self.replay_buffer[key][indices] = value[dropped:]

        self.buffer_ptr = (self.buffer_ptr + batch_size) % self.replay_capacity
        self.buffer_size = min(self.buffer_size + batch_size, self.replay_capacity)
        self.total_transitions += batch_size
        self.replay_total_added += batch_size
        self.replay_num_envs = batch_size
        # Track real observations only, before any imagination graph is built.
        self.policy.obs_norm.update(data['next_obs'])

    def sample_windows(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample temporal windows for world model training.

        Args:
            batch_size: Number of windows to sample

        Returns:
            windows: Dict with keys 's', 'a', 'context', 's1', 'mask'
        """
        if not self.buffer_size or self.replay_num_envs is None:
            raise ValueError("Not enough data for windows")
        oldest = self.replay_total_added - self.buffer_size
        available_steps = min(self.wm_window, 1 + (self.buffer_size - 1) // self.replay_num_envs)
        span = (available_steps - 1) * self.replay_num_envs
        # Prefer full windows; allow shorter, masked windows during warmup.
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
            # The terminal transition itself is a valid world-model target.
            'mask': before * available,
        }

    def update_world_model(self) -> Dict[str, float]:
        """
        Train world model on sampled windows.

        Returns:
            metrics: Training metrics
        """
        if self.buffer_size < self.min_buffer_size:
            return {"world_model_loss": 0.0, "skipped": True}

        # Sample windows
        windows = self.sample_windows(self.wm_batch_size)

        active = windows['mask'].bool()
        states, actions = windows['s'][active], windows['a'][active]
        contexts, target_states = windows['context'][active], windows['s1'][active]
        features = self.mdp.inputs(states, actions, contexts)
        target_delta = self.mdp.targets(states, target_states)
        total_loss = 0.0
        total_error = 0.0

        # Train each ensemble member
        for member in range(self.wm.ensemble_size):
            model = self.wm.models[member]
            model.in_norm.update(features)
            model.out_norm.update(target_delta)
            predicted_delta = model(features)
            error = self.mdp.state_error(predicted_delta, target_delta)
            scale = model.out_norm._std.clamp_min(model.error_floor)
            loss = (error / scale).square().mean()
            self.wm_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm, error_if_nonfinite=True)
            self.wm_optimizer.step()
            total_loss += loss.item()
            total_error += error.detach().abs().mean().item()

        self.world_model_updates += 1

        return {
            "world_model_loss": total_loss / self.wm.ensemble_size,
            "prediction_error": total_error / self.wm.ensemble_size,
            "skipped": False,
        }

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample random batch from replay buffer"""
        if not self.buffer_size:
            raise ValueError("Cannot sample an empty replay buffer")
        if self.buffer_size < batch_size:
            batch_size = self.buffer_size

        indices = torch.randint(0, self.buffer_size, (batch_size,), device=self.device)

        return {
            'state': self.replay_buffer['state'][indices],
            'action': self.replay_buffer['action'][indices],
            'reward': self.replay_buffer['reward'][indices],
            'next_state': self.replay_buffer['next_state'][indices],
            'done': self.replay_buffer['done'][indices],
            'context': self.replay_buffer['context'][indices],
            'obs': self.replay_buffer['obs'][indices],
            'next_obs': self.replay_buffer['next_obs'][indices],
        }

    def update_policy(self) -> Dict[str, float]:
        """
        Update policy and critic using imagination.

        Returns:
            metrics: Training metrics
        """
        if self.buffer_size < self.min_buffer_size:
            return {"policy_loss": 0.0, "critic_loss": 0.0, "skipped": True}

        # Sample real states
        batch = self.sample_batch(self.policy_batch_size)
        state = batch['state']
        obs = batch['obs']
        context = batch['context']

        # Imagination rollout
        (
            policy_loss,
            critic_loss,
            imagined_reward,
            imagined_survival,
            disagreement
        ) = self._imagine_and_optimize(state, obs, context)

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

    def _imagine_and_optimize(
        self,
        initial_state: torch.Tensor,
        initial_obs: torch.Tensor,
        context: torch.Tensor
    ) -> Tuple[float, float, float, float, float]:
        """Optimize one differentiable rollout, then fit independent TD(lambda) targets."""
        states = [initial_state]
        observations = [initial_obs]
        rewards, dones, disagreements = [], [], []
        state, obs = initial_state, initial_obs
        # Freeze model/value parameters, but keep their input gradients for BPTT.
        # A separate no-grad rollout would disconnect actions from future policy
        # observations and apply uncertainty/survival to a different trajectory.
        with _fixed_parameters(self.wm, self.critic):
            for _ in range(self.horizon):
                action = self.policy(obs)
                members = torch.randint(self.wm.ensemble_size, (state.shape[0],), device=state.device)
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
            discount = self.gamma ** torch.arange(
                self.horizon, device=state.device, dtype=rewards_tensor.dtype
            )[:, None]
            objective = (discount * before * rewards_tensor).sum(dim=0)
            # Bootstrap from the final imagined state, using the same survival
            # weights and differentiable value term as the original algorithm.
            final_features = self.mdp.critic_features(states[-1], context, observations[-1])
            objective = objective + self.gamma ** self.horizon * alive[-1] * self.critic(final_features).squeeze(-1)
            policy_loss = -objective.mean()
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            gradient = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.max_grad_norm, error_if_nonfinite=True
            )
            self.policy_optimizer.step()
            self._last_policy_grad_norm = gradient.item()

        # Neither inputs, survival weights nor targets may retain the consumed
        # rollout graph across critic epochs or minibatches.
        critic_loss = self._update_critic(
            torch.stack([value.detach() for value in states]), context.detach(),
            [value.detach() for value in observations], rewards_tensor.detach(),
            dones_tensor.detach(), before.detach()
        )
        with torch.no_grad():
            imagined_reward = (before * rewards_tensor).sum(dim=0).mean().item()
            imagined_survival = alive[-1].mean().item()
            disagreement = (torch.stack(disagreements) * before).sum() / before.sum()
        return policy_loss.item(), critic_loss, imagined_reward, imagined_survival, disagreement.item()

    def _update_critic(
        self,
        states: torch.Tensor,
        context: torch.Tensor,
        observations: list,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor
    ) -> float:
        """Fit TD(lambda) targets over reachable imagined states, including a final value."""
        horizon, batch_size = rewards.shape
        if states.shape[0] != horizon + 1 or len(observations) != horizon + 1:
            raise ValueError("Critic rollout must include the final imagined state and observation")
        with torch.no_grad():
            features = torch.stack([
                self.mdp.critic_features(state, context, obs)
                for state, obs in zip(states, observations)
            ])
            flat_weights = weights.detach().reshape(-1)
            reachable = flat_weights > 0
            training_features = features[:-1].reshape(-1, features.shape[-1])[reachable]
            self.critic.norm.update(training_features)
            values = self.critic(features.reshape(-1, features.shape[-1])).reshape(horizon + 1, batch_size)
            targets = self._compute_lambda_returns(rewards.detach(), dones.detach(), values)
            targets = targets.reshape(-1)[reachable]
            flat_weights = flat_weights[reachable]

        total_loss, num_updates = 0.0, 0
        sample_count = training_features.shape[0]
        for _ in range(self.critic_epochs):
            permutation = torch.randperm(sample_count, device=training_features.device)
            # tensor_split includes remainders and never creates empty batches
            # when N is smaller than the configured minibatch count.
            for indices in permutation.tensor_split(min(self.critic_minibatches, sample_count)):
                predictions = self.critic(training_features[indices]).squeeze(-1)
                error = predictions - targets[indices]
                batch_weights = flat_weights[indices]
                loss = (error.square() * batch_weights).sum() / batch_weights.sum()
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.max_grad_norm, error_if_nonfinite=True
                )
                self.critic_optimizer.step()
                total_loss += loss.item()
                num_updates += 1
        return total_loss / max(num_updates, 1)

    def _compute_lambda_returns(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """TD(lambda): r_t + gamma (1-d_t) [(1-lambda)V_(t+1) + lambda G_(t+1)]."""
        if rewards.shape != dones.shape or values.shape != (rewards.shape[0] + 1, rewards.shape[1]):
            raise ValueError("TD(lambda) requires H rewards/dones and H+1 values")
        next_return = values[-1]
        returns = []
        for step in reversed(range(rewards.shape[0])):
            next_return = rewards[step] + self.gamma * (1 - dones[step].to(rewards.dtype)) * (
                (1 - self.lambda_return) * values[step + 1] + self.lambda_return * next_return
            )
            returns.append(next_return)
        return torch.stack(list(reversed(returns)))

    def update(self) -> Dict[str, float]:
        """
        Single update step.

        Returns:
            metrics: Combined metrics
        """
        self.iteration += 1
        # Policy updates cannot advance during warmup. Count actual world-model
        # fits so skipped updates before the replay is ready do not consume it.
        if self.world_model_updates < self.warmup_iterations:
            self.phase = "warmup"
            # Warmup: only train world model
            wm_metrics = self.update_world_model()
            return {
                "phase": self.phase,
                **wm_metrics,
            }
        else:
            self.phase = "joint"
            # Joint training
            wm_metrics = self.update_world_model()
            policy_metrics = self.update_policy()
            return {
                "phase": self.phase,
                **wm_metrics,
                **policy_metrics,
            }

    def select_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Select action from policy.

        Args:
            obs: [N, obs_dim]
            deterministic: Whether to add exploration noise

        Returns:
            action: [N, action_dim]
        """
        with torch.no_grad():
            action = self.policy(obs)

            if not deterministic:
                # Add exploration noise during training
                noise = torch.randn_like(action) * 0.1
                action = action + noise

        return action

    def get_stats(self) -> Dict[str, any]:
        """Get training statistics"""
        return {
            "phase": self.phase,
            "iteration": self.iteration,
            "world_model_updates": self.world_model_updates,
            "policy_updates": self.policy_updates,
            "total_transitions": self.total_transitions,
            "buffer_stats": {
                "buffer_size": self.buffer_size,
                "buffer_utilization": self.buffer_size / self.replay_capacity,
            }
        }

    def save(self, path: str):
        """Save agent state"""
        torch.save({
            'mdp_contract': self.mdp.contract(),
            'wm': self.wm.state_dict(),
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'wm_optimizer': self.wm_optimizer.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'world_model_updates': self.world_model_updates,
            'policy_updates': self.policy_updates,
            'total_transitions': self.total_transitions,
            'iteration': self.iteration,
            'phase': self.phase,
        }, path + "_agent.pt")

    def load(self, path: str):
        """Load agent state"""
        checkpoint = torch.load(path + "_agent.pt", map_location=self.device)
        self.wm.load_state_dict(checkpoint['wm'])
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.wm_optimizer.load_state_dict(checkpoint['wm_optimizer'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.world_model_updates = checkpoint['world_model_updates']
        self.policy_updates = checkpoint['policy_updates']
        self.total_transitions = checkpoint['total_transitions']
        self.iteration = checkpoint.get('iteration', self.world_model_updates)
        self.phase = checkpoint.get('phase', 'joint' if self.world_model_updates >= self.warmup_iterations else 'warmup')
