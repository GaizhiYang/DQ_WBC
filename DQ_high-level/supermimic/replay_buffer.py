"""
Replay Buffer for SuperMimic DQ_WBC
Stores real transitions for off-policy world model training
"""
import torch
from typing import Dict, Tuple, Optional


class ReplayBuffer:
    """
    Circular replay buffer for storing real environment transitions.

    Stores tuples: (s_t, a_t, r_t, s_{t+1}, done_t, native_reward_t)
    - s_t: simplified state [51D]
    - a_t: action [action_dim]
    - r_t: reward (scalar)
    - s_{t+1}: next simplified state [51D]
    - done_t: episode termination flag
    - native_reward_t: physical reward for comparison
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, device='cuda:0'):
        """
        Args:
            capacity: Maximum number of transitions to store (e.g., 1,000,000)
            state_dim: Dimension of simplified state (51)
            action_dim: Dimension of action space
            device: Torch device
        """
        if capacity <= 0:
            raise ValueError("Replay buffer capacity must be positive")
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device

        # Pre-allocate tensors
        self.states = torch.zeros(capacity, state_dim, device=device)
        self.actions = torch.zeros(capacity, action_dim, device=device)
        self.rewards = torch.zeros(capacity, device=device)
        self.next_states = torch.zeros(capacity, state_dim, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        self.native_rewards = torch.zeros(capacity, device=device)

        # Buffer management
        self.ptr = 0  # Current write pointer
        self.size = 0  # Current buffer size
        self.num_envs = None
        self.total_added = 0
        self._sequence_starts_cache = {}

    @torch.no_grad()
    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
            next_state: torch.Tensor, done: torch.Tensor, native_reward: Optional[torch.Tensor] = None):
        """
        Add one timestep from a fixed, consistently ordered batch of environments.

        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            reward: [batch_size]
            next_state: [batch_size, state_dim]
            done: [batch_size]
            native_reward: [batch_size] (optional)
        """
        batch_size = state.shape[0]
        if batch_size == 0:
            return
        if self.num_envs is not None and batch_size != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} environments per timestep, got {batch_size}; "
                "clear the replay buffer before changing the environment batch"
            )

        fields = {
            'states': (state, (batch_size, self.state_dim)),
            'actions': (action, (batch_size, self.action_dim)),
            'next_states': (next_state, (batch_size, self.state_dim)),
        }
        for name, (value, shape) in fields.items():
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        for name, value in (('rewards', reward), ('dones', done),
                            ('native_rewards', native_reward)):
            if value is not None:
                if value.numel() != batch_size:
                    raise ValueError(f"{name} must contain one value per environment")
                fields[name] = (value.reshape(batch_size), (batch_size,))

        # Only write the newest capacity entries if a batch exceeds the ring;
        # duplicate advanced-index writes otherwise have undefined ordering.
        skipped = max(0, batch_size - self.capacity)
        indices = torch.arange(self.ptr + skipped, self.ptr + batch_size,
                               device=self.device) % self.capacity
        for name, (value, _) in fields.items():
            target = getattr(self, name)
            target[indices] = value[skipped:].to(device=target.device, dtype=target.dtype)
        if native_reward is None:
            self.native_rewards[indices] = 0

        # Update pointer and size
        self.num_envs = batch_size
        self.total_added += batch_size
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
        self._sequence_starts_cache.clear()

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing sampled transitions
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples in buffer: {self.size} < {batch_size}")

        # Random sampling with replacement
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        return {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'native_rewards': self.native_rewards[indices]
        }

    def _valid_sequence_starts(self, seq_length: int) -> torch.Tensor:
        """Return absolute insertion indices whose full trajectory is retained."""
        if seq_length <= 0:
            raise ValueError("Sequence length must be positive")
        if seq_length in self._sequence_starts_cache:
            return self._sequence_starts_cache[seq_length]

        # Adjacent slots belong to different environments. A timestep in the
        # same environment is num_envs slots away, including across ring wraps.
        span = (seq_length - 1) * (self.num_envs or 1)
        count = max(0, self.size - span) if self.num_envs is not None else 0
        starts = torch.arange(self.total_added - self.size,
                              self.total_added - self.size + count, device=self.device)
        valid = torch.ones(count, dtype=torch.bool, device=self.device)
        for offset in range(seq_length - 1):
            indices = (starts + offset * (self.num_envs or 1)) % self.capacity
            valid &= ~self.dones[indices]
        starts = starts[valid]
        self._sequence_starts_cache[seq_length] = starts
        return starts

    def can_sample_sequences(self, seq_length: int) -> bool:
        """Whether at least one same-environment, same-episode sequence exists."""
        return self._valid_sequence_starts(seq_length).numel() > 0

    def sample_sequences(self, batch_size: int, seq_length: int) -> Dict[str, torch.Tensor]:
        """
        Sample sequences of consecutive transitions for world model training.
        Ensures sequences don't span episode boundaries.

        Args:
            batch_size: Number of sequences to sample
            seq_length: Length of each sequence (e.g., 8 for world model training)

        Returns:
            Dictionary containing sampled sequences:
                - states: [batch_size, seq_length, state_dim]
                - actions: [batch_size, seq_length, action_dim]
                - rewards: [batch_size, seq_length]
                - next_states: [batch_size, seq_length, state_dim]
                - dones: [batch_size, seq_length]
        """
        if batch_size <= 0:
            raise ValueError("Sequence batch size must be positive")
        starts = self._valid_sequence_starts(seq_length)
        if starts.numel() == 0:
            raise ValueError(
                f"No valid {seq_length}-step sequences in {self.size} transitions "
                f"from {self.num_envs or 0} environments"
            )
        sampled = starts[torch.randint(starts.numel(), (batch_size,), device=self.device)]
        offsets = torch.arange(seq_length, device=self.device) * self.num_envs
        indices = (sampled[:, None] + offsets[None, :]) % self.capacity
        return {name: getattr(self, name)[indices]
                for name in ('states', 'actions', 'rewards', 'next_states', 'dones')}

    def get_stats(self) -> Dict[str, float]:
        """Get buffer statistics for logging."""
        if self.size == 0:
            return {
                'buffer_size': 0,
                'buffer_utilization': 0.0,
                'mean_reward': 0.0,
                'mean_native_reward': 0.0
            }

        valid_data = slice(0, self.size)
        return {
            'buffer_size': self.size,
            'buffer_utilization': self.size / self.capacity,
            'mean_reward': self.rewards[valid_data].mean().item(),
            'mean_native_reward': self.native_rewards[valid_data].mean().item(),
            'std_reward': self.rewards[valid_data].std(unbiased=False).item(),
            'done_rate': self.dones[valid_data].float().mean().item()
        }

    def clear(self):
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0
        self.num_envs = None
        self.total_added = 0
        self._sequence_starts_cache.clear()

    def __len__(self):
        return self.size

    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return self.size >= min_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer (optional enhancement).
    Samples transitions based on TD-error or prediction error.
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int,
                 device='cuda:0', alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (anneals from beta to 1)
        """
        super().__init__(capacity, state_dim, action_dim, device)

        self.alpha = alpha
        self.beta = beta
        self.priorities = torch.ones(capacity, device=device)
        self.max_priority = 1.0

    def add(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
            next_state: torch.Tensor, done: torch.Tensor, native_reward: Optional[torch.Tensor] = None):
        """Add transitions with maximum priority."""
        batch_size = state.shape[0]
        start = self.ptr + max(0, batch_size - self.capacity)
        indices = torch.arange(start, self.ptr + batch_size, device=self.device) % self.capacity

        # Call parent add
        super().add(state, action, reward, next_state, done, native_reward)
        self.priorities[indices] = self.max_priority

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Sample based on priorities.

        Returns:
            batch: Dictionary of sampled transitions
            indices: Sampled indices (for priority updates)
            weights: Importance sampling weights
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {batch_size}")

        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()

        # Sample indices
        indices = torch.multinomial(probs, batch_size, replacement=True)

        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize

        batch = {
            'states': self.states[indices],
            'actions': self.actions[indices],
            'rewards': self.rewards[indices],
            'next_states': self.next_states[indices],
            'dones': self.dones[indices],
            'native_rewards': self.native_rewards[indices]
        }

        return batch, indices, weights

    def update_priorities(self, indices: torch.Tensor, priorities: torch.Tensor):
        """Update priorities for sampled transitions."""
        self.priorities[indices] = priorities.clamp(min=1e-6)
        self.max_priority = max(self.max_priority, priorities.max().item())

    def anneal_beta(self, step: int, total_steps: int):
        """Anneal beta from initial value to 1.0."""
        self.beta = min(1.0, 0.4 + 0.6 * step / total_steps)
