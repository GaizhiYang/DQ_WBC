"""
Neural Networks with Normalization for SuperMimic v2

Aligned with original SuperMimic's network architecture including:
- Input/output normalization
- MLP with ELU activation
- BF16 autocast support
"""
import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import math


class EmpiricalNormalization(nn.Module):
    """
    Running mean/std normalization (Welford's algorithm).

    Adapted from rsl_rl.modules.EmpiricalNormalization
    """

    def __init__(self, shape: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

        # Running statistics
        self.register_buffer('_mean', torch.zeros(shape))
        self.register_buffer('_var', torch.ones(shape))
        self.register_buffer('_std', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using current statistics"""
        return (x - self._mean) / (self._std + self.eps)

    def update(self, x: torch.Tensor):
        """Update running statistics with new batch"""
        with torch.no_grad():
            if x.ndim != 2 or x.shape[1:] != self._mean.shape:
                raise ValueError("Normalization input must have shape (batch, features)")
            if x.shape[0] == 0:
                return
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]

            delta = batch_mean - self._mean
            total_count = self.count + batch_count

            # Update mean - ✅ FIX: use .copy_() instead of += to avoid breaking gradients
            new_mean = self._mean + delta * batch_count / total_count
            self._mean.copy_(new_mean)

            # Update variance (Welford's algorithm)
            m_a = self._var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
            self._var.copy_((M2 / total_count).clamp_min(0))
            self._std.copy_(torch.sqrt(self._var + self.eps))

            # Update count - ✅ FIX: use .copy_() instead of +=
            self.count.copy_(self.count + batch_count)


def mlp(
    inp: int,
    hidden: Tuple[int, ...],
    out: int,
    activation: nn.Module = nn.ELU,
    layer_norm: bool = False
) -> nn.Sequential:
    """
    Create MLP with specified architecture.

    Args:
        inp: Input dimension
        hidden: Tuple of hidden layer sizes
        out: Output dimension
        activation: Activation function class
        layer_norm: Whether to use layer normalization

    Returns:
        Sequential MLP network
    """
    layers = []
    for width in hidden:
        layers.append(nn.Linear(inp, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        inp = width
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


def run_mlp(net: nn.Module, x: torch.Tensor, amp: str = "none") -> torch.Tensor:
    """
    Run MLP with optional autocast.

    Args:
        net: MLP network
        x: Input tensor (must be FP32)
        amp: Autocast mode ("bf16" or "none")

    Returns:
        Output tensor (FP32)
    """
    if amp == "bf16":
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            return net(x).float()
    else:
        return net(x)


class WorldModel(nn.Module):
    """
    Single world model with input/output normalization.

    Predicts state delta from (state, action, context).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: Tuple[int, ...] = (2048, 2048, 2048, 2048, 2048),
        amp: str = "none",
        error_floor: float = 1e-4
    ):
        super().__init__()
        self.amp = amp
        self.error_floor = error_floor

        # Input/output normalization
        self.in_norm = EmpiricalNormalization(input_dim)
        self.out_norm = EmpiricalNormalization(output_dim)

        # MLP network
        self.net = mlp(input_dim, hidden, output_dim)

        # Small initialization for last layer (as in original)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict state delta from normalized features.

        Args:
            features: [N, input_dim] already prepared by mdp.inputs()

        Returns:
            delta: [N, output_dim] predicted NORMALIZED delta (caller must denormalize)
        """
        normalized = self.in_norm(features)
        output = run_mlp(self.net, normalized, self.amp)
        # ✅ FIX: Return normalized output, let caller denormalize to avoid gradient issues
        return output

    def step(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor, mdp) -> torch.Tensor:
        """
        Predict next state from current (state, action, context).

        Args:
            state: [N, state_dim]
            action: [N, action_dim]
            context: [N, context_dim]
            mdp: MDP instance for feature preparation

        Returns:
            next_state: [N, state_dim]
        """
        features = mdp.inputs(state, action, context)
        delta = self.forward(features)
        return state + delta


class WorldModelEnsemble(nn.Module):
    """
    Ensemble of world models for uncertainty estimation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: Tuple[int, ...] = (2048, 2048, 2048, 2048, 2048),
        ensemble_size: int = 5,
        amp: str = "none"
    ):
        super().__init__()
        self.ensemble_size = ensemble_size

        # Create ensemble
        self.models = nn.ModuleList([
            WorldModel(input_dim, output_dim, hidden, amp)
            for _ in range(ensemble_size)
        ])

    def forward(self, features: torch.Tensor, member: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through ensemble.

        Args:
            features: [N, input_dim]
            member: Optional specific member index (None = mean)

        Returns:
            delta: [N, output_dim] prediction
        """
        if member is not None:
            return self.models[member](features)
        else:
            # Mean prediction
            predictions = torch.stack([m(features) for m in self.models], dim=0)
            return predictions.mean(dim=0)

    def step(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        mdp,
        member: Optional[int] = None
    ) -> torch.Tensor:
        """
        Predict next state.

        Args:
            state: [N, state_dim]
            action: [N, action_dim]
            context: [N, context_dim]
            mdp: MDP instance
            member: Optional member index

        Returns:
            next_state: [N, state_dim]
        """
        features = mdp.inputs(state, action, context)
        delta = self.forward(features, member)
        return state + delta

    def imagine(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: torch.Tensor,
        mdp,
        member_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Imagination step with uncertainty estimation.

        Args:
            state: [N, state_dim]
            action: [N, action_dim]
            context: [N, context_dim]
            mdp: MDP instance
            member_index: [N] member indices per sample (optional)

        Returns:
            next_state: [N, state_dim] predicted next state
            disagreement: [N] ensemble disagreement (uncertainty)
        """
        features = mdp.inputs(state, action, context)

        if self.ensemble_size == 1:
            # No ensemble, no disagreement
            next_state = self.step(state, action, context, mdp, member=0)
            disagreement = torch.zeros(state.shape[0], device=state.device)
        else:
            # Get predictions from all members
            predictions = torch.stack([
                state + self.models[i](features)
                for i in range(self.ensemble_size)
            ], dim=0)  # [ensemble_size, N, state_dim]

            # Disagreement as std across ensemble
            disagreement = predictions.std(dim=0, unbiased=False).mean(dim=-1)  # [N]

            # Select prediction based on member_index
            if member_index is not None:
                # Select specific member per sample
                next_state = predictions[member_index, torch.arange(state.shape[0], device=state.device)]
            else:
                # Use ensemble mean
                next_state = predictions.mean(dim=0)

        return next_state, disagreement


class Policy(nn.Module):
    """
    Policy network with observation normalization.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Tuple[int, ...] = (2048, 1024, 512, 256),
        amp: str = "none"
    ):
        super().__init__()
        self.amp = amp

        # Observation normalization
        self.obs_norm = EmpiricalNormalization(obs_dim)

        # MLP network
        self.net = mlp(obs_dim, hidden, action_dim)

        # Small initialization
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Compute action from observation.

        Args:
            obs: [N, obs_dim] observation
            deterministic: Ignored (policy is deterministic), for compatibility

        Returns:
            action: [N, action_dim] continuous action
        """
        normalized = self.obs_norm(obs)
        action = run_mlp(self.net, normalized, self.amp)
        return action


class Critic(nn.Module):
    """
    Critic network with input normalization.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden: Tuple[int, ...] = (256, 128),
        amp: str = "none"
    ):
        super().__init__()
        self.amp = amp

        # Feature normalization
        self.norm = EmpiricalNormalization(feature_dim)

        # MLP network
        self.net = mlp(feature_dim, hidden, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Estimate value from critic features.

        Args:
            features: [N, feature_dim] from mdp.critic_features()

        Returns:
            value: [N, 1] value estimate
        """
        normalized = self.norm(features)
        value = run_mlp(self.net, normalized, self.amp)
        return value
