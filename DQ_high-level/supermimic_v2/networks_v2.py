"""
SuperMimic networks adapted from reference implementation.

Key differences from our version:
1. Uses RunningNorm instead of EmpiricalNormalization (no in-place +=)
2. Denormalization happens in step(), not forward()
3. Ensemble has imagine() method for uncertainty
"""
import torch
import torch.nn as nn
import math
from typing import Tuple


def mlp(inp, hidden, out, *, activation=nn.ELU, layer_norm=False):
    """Create MLP network."""
    layers = []
    for width in hidden:
        layers.append(nn.Linear(inp, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        inp = width
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


def run_mlp(net, x: torch.Tensor, amp: str = "none") -> torch.Tensor:
    """Run MLP with optional autocast."""
    if amp == "none":
        return net(x)
    if amp != "bf16":
        raise ValueError(f"Unsupported MLP autocast mode: {amp!r}")
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
        return net(x).float()


class RunningNorm(nn.Module):
    """
    ✅ FIXED: Running normalization WITHOUT in-place operations that break gradients.

    Key fix: Uses .copy_() for ALL updates, never uses += or *=
    """

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        if dim < 1 or not math.isfinite(max_count) or max_count <= 0:
            raise ValueError("Normalizer dimension and max_count must be positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Normalizer eps must be positive and finite")
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running statistics - all operations use .copy_() to avoid gradient issues."""
        if x.ndim < 2 or x.shape[-1] != self.mean.numel():
            raise ValueError("Normalizer input must end in its declared feature dimension")
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if x.shape[0] == 0:
            return
        if not torch.isfinite(x).all():
            return  # Skip non-finite samples
        if self.count >= self.max_count:
            return

        count = x.shape[0]
        mean, var = x.mean(0), x.var(0, unbiased=False)
        total = self.count + count
        delta = mean - self.mean

        # ✅ Compute new values, then copy (no in-place ops on self.mean/var)
        combined_var = (
            self.var * self.count + var * count + delta.square() * self.count * count / total
        ) / total
        combined_mean = self.mean + delta * count / total

        if not torch.isfinite(combined_var).all() or not torch.isfinite(combined_mean).all():
            return  # Skip if would overflow

        # ✅ Use .copy_() instead of += to avoid changing tensor version
        self.mean.copy_(combined_mean)
        self.var.copy_(combined_var.clamp_min(0))
        self.count.copy_(total)

    def std(self) -> torch.Tensor:
        """Return standard deviation."""
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input."""
        return (x - self.mean) / self.std()


class EmpiricalNormalization(nn.Module):
    """
    ✅ Policy normalization with update() method (compatible with agent code).

    This is a simplified version that's compatible with the agent's expectations.
    """

    def __init__(self, shape: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.register_buffer('_mean', torch.zeros(shape))
        self.register_buffer('_var', torch.ones(shape))
        self.register_buffer('_std', torch.ones(shape))
        self.register_buffer('count', torch.zeros(1, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using current statistics"""
        return (x - self._mean) / (self._std + self.eps)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Update running statistics with new batch"""
        if x.ndim != 2 or x.shape[1] != self._mean.shape[0]:
            raise ValueError("Normalization input must have shape (batch, features)")
        if x.shape[0] == 0:
            return

        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self._mean
        total_count = self.count + batch_count

        # Update mean - use .copy_() to avoid gradient issues
        new_mean = self._mean + delta * batch_count / total_count
        self._mean.copy_(new_mean)

        # Update variance (Welford's algorithm)
        m_a = self._var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta.square() * self.count * batch_count / total_count
        self._var.copy_((M2 / total_count).clamp_min(0))
        self._std.copy_(torch.sqrt(self._var + self.eps))

        # Update count
        self.count.copy_(self.count + batch_count)


class WorldModel(nn.Module):
    """
    ✅ Single world model - FIXED to avoid gradient issues.

    Key: forward() returns NORMALIZED output, step() does denormalization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: Tuple[int, ...] = (2048, 2048, 2048, 2048, 2048),
        normalizer_max_count: float = 2e8,
        amp: str = "none",
        error_floor: float = 1e-4
    ):
        super().__init__()
        self.amp = amp
        self.error_floor = error_floor

        # Networks
        self.net = mlp(input_dim, hidden, output_dim, activation=nn.ELU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Normalization
        self.in_norm = RunningNorm(input_dim, normalizer_max_count)
        self.out_norm = RunningNorm(output_dim, normalizer_max_count, eps=error_floor)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        ✅ FIXED: Returns NORMALIZED delta, caller does denormalization.

        This avoids using self.out_norm.mean/std during forward pass,
        which would be affected by updates.
        """
        normalized_input = self.in_norm(features)
        normalized_output = run_mlp(self.net, normalized_input, self.amp)
        return normalized_output  # Return normalized, don't denormalize here


class WorldModelEnsemble(nn.Module):
    """
    ✅ FIXED: Ensemble with proper separation of forward/denormalization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: Tuple[int, ...] = (2048, 2048, 2048, 2048, 2048),
        ensemble_size: int = 5,
        normalizer_max_count: float = 2e8,
        amp: str = "none",
        error_floor: float = 1e-4
    ):
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("Ensemble needs at least 2 members")

        self.ensemble_size = ensemble_size
        self.amp = amp
        self.error_floor = error_floor

        # Create ensemble members
        self.nets = nn.ModuleList()
        for _ in range(ensemble_size):
            net = mlp(input_dim, hidden, output_dim, activation=nn.ELU, layer_norm=True)
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.nets.append(net)

        # Shared normalization across ensemble
        self.in_norm = RunningNorm(input_dim, normalizer_max_count)
        self.out_norm = RunningNorm(output_dim, normalizer_max_count, eps=error_floor)

    def forward(self, features: torch.Tensor, member_idx: int) -> torch.Tensor:
        """Forward for specific member - returns NORMALIZED output."""
        normalized_input = self.in_norm(features)
        normalized_output = run_mlp(self.nets[member_idx], normalized_input, self.amp)
        return normalized_output

    def imagine(self, state, action, context, mdp, member_indices):
        """
        Predict next state and disagreement for imagination rollout.

        Args:
            state: [batch, state_dim]
            action: [batch, action_dim]
            context: [batch, context_dim]
            mdp: MDP instance
            member_indices: [batch] which member to use per sample

        Returns:
            next_state: [batch, state_dim]
            disagreement: [batch] ensemble disagreement
        """
        features = mdp.inputs(state, action, context)
        normalized_input = self.in_norm(features)

        # Get predictions from all members
        predictions = torch.stack([
            run_mlp(net, normalized_input, self.amp)
            for net in self.nets
        ])  # [ensemble_size, batch, delta_dim]

        # Disagreement = std over members
        disagreement = predictions.std(dim=0).mean(dim=-1)  # [batch]

        # Select prediction for each sample based on member_indices
        batch_size = state.shape[0]
        selected = predictions[member_indices, torch.arange(batch_size)]  # [batch, delta_dim]

        # Denormalize ONCE here, safely
        with torch.no_grad():
            std = self.out_norm.std()
            mean = self.out_norm.mean

        delta = selected * std + mean
        next_state = state + delta

        return next_state, disagreement


class Policy(nn.Module):
    """Policy network with EmpiricalNormalization (has update() method)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Tuple[int, ...] = (2048, 1024, 512, 256),
        amp: str = "none"
    ):
        super().__init__()
        self.amp = amp
        self.obs_norm = EmpiricalNormalization(obs_dim)
        self.net = mlp(obs_dim, hidden, action_dim)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, obs, deterministic: bool = False):
        """Forward pass - deterministic parameter for compatibility."""
        return run_mlp(self.net, self.obs_norm(obs), self.amp)


class Critic(nn.Module):
    """Critic network."""

    def __init__(
        self,
        feature_dim: int,
        hidden: Tuple[int, ...] = (256, 128),
        normalizer_max_count: float = 2e8,
        amp: str = "none"
    ):
        super().__init__()
        self.amp = amp
        self.net = mlp(feature_dim, hidden, 1)
        self.norm = RunningNorm(feature_dim, normalizer_max_count)

    def forward(self, features):
        return run_mlp(self.net, self.norm(features), self.amp).squeeze(-1)



def mlp(inp, hidden, out, *, activation=nn.ELU, layer_norm=False):
    """Create MLP network."""
    layers = []
    for width in hidden:
        layers.append(nn.Linear(inp, width))
        if layer_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation())
        inp = width
    layers.append(nn.Linear(inp, out))
    return nn.Sequential(*layers)


def run_mlp(net, x: torch.Tensor, amp: str = "none") -> torch.Tensor:
    """Run MLP with optional autocast."""
    if amp == "none":
        return net(x)
    if amp != "bf16":
        raise ValueError(f"Unsupported MLP autocast mode: {amp!r}")
    with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
        return net(x).float()


class RunningNorm(nn.Module):
    """
    ✅ FIXED: Running normalization WITHOUT in-place operations that break gradients.

    Key fix: Uses .copy_() for ALL updates, never uses += or *=
    """

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        if dim < 1 or not math.isfinite(max_count) or max_count <= 0:
            raise ValueError("Normalizer dimension and max_count must be positive")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("Normalizer eps must be positive and finite")
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count = max_count
        self.eps = eps

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """Update running statistics - all operations use .copy_() to avoid gradient issues."""
        if x.ndim < 2 or x.shape[-1] != self.mean.numel():
            raise ValueError("Normalizer input must end in its declared feature dimension")
        x = x.detach().reshape(-1, x.shape[-1]).float()
        if x.shape[0] == 0:
            return
        if not torch.isfinite(x).all():
            return  # Skip non-finite samples
        if self.count >= self.max_count:
            return

        count = x.shape[0]
        mean, var = x.mean(0), x.var(0, unbiased=False)
        total = self.count + count
        delta = mean - self.mean

        # ✅ Compute new values, then copy (no in-place ops on self.mean/var)
        combined_var = (
            self.var * self.count + var * count + delta.square() * self.count * count / total
        ) / total
        combined_mean = self.mean + delta * count / total

        if not torch.isfinite(combined_var).all() or not torch.isfinite(combined_mean).all():
            return  # Skip if would overflow

        # ✅ Use .copy_() instead of += to avoid changing tensor version
        self.mean.copy_(combined_mean)
        self.var.copy_(combined_var.clamp_min(0))
        self.count.copy_(total)

    def std(self) -> torch.Tensor:
        """Return standard deviation."""
        return torch.sqrt(self.var + self.eps**2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input."""
        return (x - self.mean) / self.std()


class WorldModel(nn.Module):
    """
    ✅ Single world model - FIXED to avoid gradient issues.

    Key: forward() returns NORMALIZED output, step() does denormalization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: Tuple[int, ...] = (2048, 2048, 2048, 2048, 2048),
        normalizer_max_count: float = 2e8,
        amp: str = "none",
        error_floor: float = 1e-4
    ):
        super().__init__()
        self.amp = amp
        self.error_floor = error_floor

        # Networks
        self.net = mlp(input_dim, hidden, output_dim, activation=nn.ELU, layer_norm=True)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # Normalization
        self.in_norm = RunningNorm(input_dim, normalizer_max_count)
        self.out_norm = RunningNorm(output_dim, normalizer_max_count, eps=error_floor)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        ✅ FIXED: Returns NORMALIZED delta, caller does denormalization.

        This avoids using self.out_norm.mean/std during forward pass,
        which would be affected by updates.
        """
        normalized_input = self.in_norm(features)
        normalized_output = run_mlp(self.net, normalized_input, self.amp)
        return normalized_output  # Return normalized, don't denormalize here


class WorldModelEnsemble(nn.Module):
    """
    ✅ FIXED: Ensemble with proper separation of forward/denormalization.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: Tuple[int, ...] = (2048, 2048, 2048, 2048, 2048),
        ensemble_size: int = 5,
        normalizer_max_count: float = 2e8,
        amp: str = "none",
        error_floor: float = 1e-4
    ):
        super().__init__()
        if ensemble_size < 2:
            raise ValueError("Ensemble needs at least 2 members")

        self.ensemble_size = ensemble_size
        self.amp = amp
        self.error_floor = error_floor

        # Create ensemble members
        self.nets = nn.ModuleList()
        for _ in range(ensemble_size):
            net = mlp(input_dim, hidden, output_dim, activation=nn.ELU, layer_norm=True)
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            self.nets.append(net)

        # Shared normalization across ensemble
        self.in_norm = RunningNorm(input_dim, normalizer_max_count)
        self.out_norm = RunningNorm(output_dim, normalizer_max_count, eps=error_floor)

    def forward(self, features: torch.Tensor, member_idx: int) -> torch.Tensor:
        """Forward for specific member - returns NORMALIZED output."""
        normalized_input = self.in_norm(features)
        normalized_output = run_mlp(self.nets[member_idx], normalized_input, self.amp)
        return normalized_output

    def imagine(self, state, action, context, mdp, member_indices):
        """
        Predict next state and disagreement for imagination rollout.

        Args:
            state: [batch, state_dim]
            action: [batch, action_dim]
            context: [batch, context_dim]
            mdp: MDP instance
            member_indices: [batch] which member to use per sample

        Returns:
            next_state: [batch, state_dim]
            disagreement: [batch] ensemble disagreement
        """
        features = mdp.inputs(state, action, context)
        normalized_input = self.in_norm(features)

        # Get predictions from all members
        predictions = torch.stack([
            run_mlp(net, normalized_input, self.amp)
            for net in self.nets
        ])  # [ensemble_size, batch, delta_dim]

        # Disagreement = std over members
        disagreement = predictions.std(dim=0).mean(dim=-1)  # [batch]

        # Select prediction for each sample based on member_indices
        batch_size = state.shape[0]
        selected = predictions[member_indices, torch.arange(batch_size)]  # [batch, delta_dim]

        # Denormalize ONCE here, safely
        with torch.no_grad():
            std = self.out_norm.std()
            mean = self.out_norm.mean

        delta = selected * std + mean
        next_state = state + delta

        return next_state, disagreement


class Policy(nn.Module):
    """Policy network - uses existing EmpiricalNormalization (no changes needed)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: Tuple[int, ...] = (2048, 1024, 512, 256),
        amp: str = "none"
    ):
        super().__init__()
        self.amp = amp
        self.obs_norm = EmpiricalNormalization(obs_dim)
        self.net = mlp(obs_dim, hidden, action_dim)
        with torch.no_grad():
            self.net[-1].weight.mul_(0.01)
            self.net[-1].bias.zero_()

    def forward(self, obs, deterministic: bool = False):
        """Forward pass - deterministic parameter for compatibility."""
        return run_mlp(self.net, self.obs_norm(obs), self.amp)


class Critic(nn.Module):
    """Critic network."""

    def __init__(
        self,
        feature_dim: int,
        hidden: Tuple[int, ...] = (256, 128),
        normalizer_max_count: float = 2e8,
        amp: str = "none"
    ):
        super().__init__()
        self.amp = amp
        self.net = mlp(feature_dim, hidden, 1)
        self.norm = RunningNorm(feature_dim, normalizer_max_count)

    def forward(self, features):
        return run_mlp(self.net, self.norm(features), self.amp).squeeze(-1)
