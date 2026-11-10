"""
World Model for SuperMimic DQ_WBC
Learns differentiable state transition dynamics: s_{t+1} = f(s_t, a_t)
Uses ensemble of MLPs for uncertainty estimation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class WorldModelMLP(nn.Module):
    """
    Single MLP world model predicting next state from current state and action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [512, 512, 512]):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Build MLP layers
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        # Output layer predicts state delta (residual connection)
        layers.append(nn.Linear(input_dim, state_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights with small values for stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state given current state and action.

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]

        Returns:
            next_state: [batch_size, state_dim]
        """
        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Predict state delta
        state_delta = self.network(x)

        # Apply residual connection
        next_state = state + state_delta

        return next_state


class EnsembleWorldModel(nn.Module):
    """
    Ensemble of world models for uncertainty estimation.
    Uses 5 independent MLPs and averages their predictions.
    """

    def __init__(self, state_dim: int, action_dim: int, num_ensemble: int = 5,
                 hidden_dims: List[int] = [512, 512, 512], device='cuda:0'):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_ensemble = num_ensemble
        self.device = device

        # Create ensemble of models
        self.models = nn.ModuleList([
            WorldModelMLP(state_dim, action_dim, hidden_dims)
            for _ in range(num_ensemble)
        ])

        self.to(device)

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                return_all: bool = False) -> torch.Tensor:
        """
        Predict next state using ensemble average.

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            return_all: If True, return all ensemble predictions

        Returns:
            next_state: [batch_size, state_dim] (ensemble mean)
            or [num_ensemble, batch_size, state_dim] if return_all=True
        """
        predictions = torch.stack([
            model(state, action) for model in self.models
        ])  # [num_ensemble, batch_size, state_dim]

        if return_all:
            return predictions

        # Return ensemble mean
        return predictions.mean(dim=0)

    def predict_with_uncertainty(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state with uncertainty estimate.

        Returns:
            mean: [batch_size, state_dim] ensemble mean prediction
            std: [batch_size, state_dim] ensemble standard deviation
        """
        predictions = self.forward(state, action, return_all=True)
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0, unbiased=False)
        return mean, std

    def rollout(self, initial_state: torch.Tensor, actions: torch.Tensor,
                keep_gradients: bool = True) -> torch.Tensor:
        """
        Perform multi-step imagination rollout.

        Args:
            initial_state: [batch_size, state_dim]
            actions: [batch_size, horizon, action_dim]
            keep_gradients: If True, maintain gradient flow for BPTT

        Returns:
            states: [batch_size, horizon+1, state_dim] (includes initial state)
        """
        batch_size, horizon, _ = actions.shape

        states = [initial_state]
        current_state = initial_state

        for t in range(horizon):
            # Predict next state
            if keep_gradients:
                next_state = self.forward(current_state, actions[:, t])
            else:
                with torch.no_grad():
                    next_state = self.forward(current_state, actions[:, t])

            states.append(next_state)
            current_state = next_state

        return torch.stack(states, dim=1)  # [batch_size, horizon+1, state_dim]

    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor,
                     next_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute world model training loss (MSE between predicted and actual next states).

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            next_states: [batch_size, state_dim] (ground truth)

        Returns:
            loss: scalar tensor
            info: dictionary with loss statistics
        """
        # Get predictions from all ensemble members
        predictions = self.forward(states, actions, return_all=True)  # [num_ensemble, batch_size, state_dim]

        # Compute MSE loss for each ensemble member
        target = next_states.unsqueeze(0).expand(self.num_ensemble, -1, -1)
        losses = F.mse_loss(predictions, target, reduction='none').mean(dim=[1, 2])

        # Average across ensemble
        total_loss = losses.mean()

        # Statistics
        with torch.no_grad():
            mean_prediction = predictions.mean(dim=0)
            prediction_error = F.mse_loss(mean_prediction, next_states)
            ensemble_disagreement = predictions.std(dim=0, unbiased=False).mean()

        info = {
            'world_model_loss': total_loss.item(),
            'prediction_error': prediction_error.item(),
            'ensemble_disagreement': ensemble_disagreement.item()
        }

        return total_loss, info

    def compute_sequence_loss(self, states: torch.Tensor, actions: torch.Tensor,
                              next_states: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss over sequences of transitions (for temporal consistency).

        Args:
            states: [batch_size, seq_length, state_dim]
            actions: [batch_size, seq_length, action_dim]
            next_states: [batch_size, seq_length, state_dim]

        Returns:
            loss: scalar tensor
            info: dictionary with statistics
        """
        batch_size, seq_length, _ = states.shape

        # Reshape to process all timesteps at once
        states_flat = states.reshape(-1, self.state_dim)
        actions_flat = actions.reshape(-1, self.action_dim)
        next_states_flat = next_states.reshape(-1, self.state_dim)

        # Compute loss
        return self.compute_loss(states_flat, actions_flat, next_states_flat)


class WorldModelTrainer:
    """
    Trainer for world model with replay buffer sampling.
    """

    def __init__(self, world_model: EnsembleWorldModel, learning_rate: float = 3e-4,
                 weight_decay: float = 1e-5):
        self.world_model = world_model
        self.optimizer = torch.optim.AdamW(
            world_model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        self.train_steps = 0

    def train_step(self, states: torch.Tensor, actions: torch.Tensor,
                   next_states: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step.

        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            next_states: [batch_size, state_dim]

        Returns:
            info: Dictionary with training statistics
        """
        self.world_model.train()

        # Compute loss
        loss, info = self.world_model.compute_loss(states, actions, next_states)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0, error_if_nonfinite=True)

        self.optimizer.step()
        self.train_steps += 1

        return info

    def train_sequence_step(self, states: torch.Tensor, actions: torch.Tensor,
                           next_states: torch.Tensor) -> Dict[str, float]:
        """
        Train on sequences of transitions.

        Args:
            states: [batch_size, seq_length, state_dim]
            actions: [batch_size, seq_length, action_dim]
            next_states: [batch_size, seq_length, state_dim]

        Returns:
            info: Dictionary with training statistics
        """
        self.world_model.train()

        # Compute loss
        loss, info = self.world_model.compute_sequence_loss(states, actions, next_states)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0, error_if_nonfinite=True)

        self.optimizer.step()
        self.train_steps += 1

        return info

    def save(self, path: str):
        """Save world model checkpoint."""
        torch.save({
            'world_model_state_dict': self.world_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps
        }, path)

    def load(self, path: str):
        """Load world model checkpoint."""
        checkpoint = torch.load(path, map_location=self.world_model.device)
        self.world_model.load_state_dict(checkpoint['world_model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_steps = checkpoint['train_steps']
