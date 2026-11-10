"""
Policy and Critic networks for SuperMimic DQ_WBC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class SuperMimicPolicy(nn.Module):
    """
    Policy network for SuperMimic.
    Outputs actions given simplified states.
    Trained via BPTT through world model.
    """

    def __init__(self, state_dim: int, action_dim: int,
                 hidden_dims: list = [256, 256, 256],
                 action_std_init: float = 0.3,
                 device='cuda:0'):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Build MLP
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Action mean and log_std heads
        self.action_mean = nn.Linear(input_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.ones(action_dim) * torch.log(torch.tensor(action_std_init)))

        self.to(device)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Small initialization for action head
        nn.init.orthogonal_(self.action_mean.weight, gain=0.01)

    def forward(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            state: [batch_size, state_dim]
            deterministic: If True, return mean action without noise

        Returns:
            action: [batch_size, action_dim]
            log_prob: [batch_size] (None if deterministic)
        """
        features = self.backbone(state)
        action_mean = self.action_mean(features)

        if deterministic:
            return torch.tanh(action_mean), None

        # Sample action with noise
        action_std = torch.exp(self.action_log_std)
        noise = torch.randn_like(action_mean) * action_std
        action_raw = action_mean + noise

        # Apply tanh squashing
        action = torch.tanh(action_raw)

        # Compute log probability (with tanh correction)
        log_prob = -0.5 * ((noise / action_std) ** 2 + 2 * torch.log(action_std) + torch.log(torch.tensor(2 * 3.14159)))
        log_prob = log_prob.sum(dim=-1)

        # Tanh correction
        log_prob -= torch.log(1 - action ** 2 + 1e-6).sum(dim=-1)

        return action, log_prob

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action without log_prob (for inference)."""
        action, _ = self.forward(state, deterministic)
        return action

    def evaluate_actions(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given state-action pairs.

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]

        Returns:
            log_prob: [batch_size]
            entropy: [batch_size]
        """
        features = self.backbone(state)
        action_mean = self.action_mean(features)
        action_std = torch.exp(self.action_log_std)

        # Inverse tanh to get raw action
        action_raw = torch.atanh(torch.clamp(action, -0.999, 0.999))

        # Compute log probability
        log_prob = -0.5 * (((action_raw - action_mean) / action_std) ** 2 + 2 * torch.log(action_std) + torch.log(torch.tensor(2 * 3.14159)))
        log_prob = log_prob.sum(dim=-1)

        # Tanh correction
        log_prob -= torch.log(1 - action ** 2 + 1e-6).sum(dim=-1)

        # Entropy (approximate)
        entropy = 0.5 * (torch.log(2 * 3.14159 * torch.e * action_std ** 2).sum())

        return log_prob, entropy


class SuperMimicCritic(nn.Module):
    """
    Critic network (value function) for SuperMimic.
    Estimates V(s) for advantage computation and bootstrapping.
    """

    def __init__(self, state_dim: int, hidden_dims: list = [256, 256, 256], device='cuda:0'):
        super().__init__()

        self.state_dim = state_dim
        self.device = device

        # Build MLP
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ELU())
            input_dim = hidden_dim

        # Value head
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)

        self.to(device)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict value function V(s).

        Args:
            state: [batch_size, state_dim]

        Returns:
            value: [batch_size, 1]
        """
        return self.network(state)

    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value as 1D tensor."""
        return self.forward(state).squeeze(-1)


class SuperMimicPolicyTrainer:
    """
    Trainer for policy via BPTT through world model.
    """

    def __init__(self, policy: SuperMimicPolicy, critic: SuperMimicCritic,
                 policy_lr: float = 3e-4, critic_lr: float = 1e-3,
                 gamma: float = 0.99, gae_lambda: float = 0.95):
        self.policy = policy
        self.critic = critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.policy_optimizer = torch.optim.Adam(policy.parameters(), lr=policy_lr)
        self.critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)

        self.train_steps = 0

    def compute_gae(self, rewards: torch.Tensor, values: torch.Tensor,
                    next_values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: [batch_size, horizon]
            values: [batch_size, horizon]
            next_values: [batch_size, horizon]
            dones: [batch_size, horizon]

        Returns:
            advantages: [batch_size, horizon]
            returns: [batch_size, horizon]
        """
        batch_size, horizon = rewards.shape

        advantages = torch.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(horizon)):
            if t == horizon - 1:
                next_value = next_values[:, t]
            else:
                next_value = values[:, t + 1]

            delta = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t].float()) - values[:, t]
            advantages[:, t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[:, t].float()) * last_gae

        returns = advantages + values

        return advantages, returns

    def update_policy_bptt(self, imagined_returns: torch.Tensor) -> Dict[str, float]:
        """
        Maximize differentiable imagined returns via the world model.

        Args:
            imagined_returns: [batch_size], retaining the actor rollout graph

        Returns:
            info: Training statistics
        """
        self.policy.train()

        policy_loss = -imagined_returns.mean()

        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=1.0, error_if_nonfinite=True
        )
        self.policy_optimizer.step()

        self.train_steps += 1

        return {
            'policy_loss': policy_loss.item(),
            'policy_grad_norm': grad_norm.item()
        }

    def update_critic(self, states: torch.Tensor, returns: torch.Tensor) -> Dict[str, float]:
        """
        Update critic using TD(λ) targets.

        Args:
            states: [batch_size, state_dim]
            returns: [batch_size]

        Returns:
            info: Training statistics
        """
        self.critic.train()

        # Predict values
        values = self.critic.get_value(states.detach())

        # MSE loss
        critic_loss = F.mse_loss(values, returns.detach())

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0, error_if_nonfinite=True)
        self.critic_optimizer.step()

        return {
            'critic_loss': critic_loss.item(),
            'value_mean': values.mean().item()
        }

    def save(self, path: str):
        """Save policy and critic."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'train_steps': self.train_steps
        }, path)

    def load(self, path: str):
        """Load policy and critic."""
        checkpoint = torch.load(path, map_location=self.policy.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.train_steps = checkpoint['train_steps']
