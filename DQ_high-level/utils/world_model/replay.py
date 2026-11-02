"""Device-resident, episode-safe replay for high-level physical transitions.

Adapted from SuperMimic-Reproduced ``data/replay_buffer.py`` (MIT). See
THIRD_PARTY_LICENSES.md. Unlike that implementation, each row explicitly stores
the pre-reset next state, so a terminal transition never needs a synthetic row.
The replay, sampling probabilities and returned batches stay on the RL device.
"""

from dataclasses import dataclass
import torch


@dataclass
class WindowBatch:
    states: torch.Tensor       # [K+1, B, S], predictions must use states[0] only
    actions: torch.Tensor      # [K, B, A], actually executed high-level actions
    context: torch.Tensor      # [K, B, C]
    dt: torch.Tensor           # [K, B, 1], actual duration of each transition
    rewards: torch.Tensor      # [K, B]
    terminated: torch.Tensor   # [K, B], natural terminal, including success
    truncated: torch.Tensor    # [K, B], timeout, distinct from natural terminal
    success: torch.Tensor      # [K, B], existing task success proxy
    family: torch.Tensor       # [B], reset family, NOT task execution stage
    episode_ids: torch.Tensor  # [B]
    env_ids: torch.Tensor      # [B]


@dataclass
class StartBatch:
    state: torch.Tensor
    action: torch.Tensor
    next_state: torch.Tensor
    context: torch.Tensor
    dt: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    success: torch.Tensor
    family: torch.Tensor
    episode_ids: torch.Tensor
    env_ids: torch.Tensor


class TransitionReplay:
    """Synchronous GPU ring, laid out [time, environment, feature].

    ``capacity`` is measured in vectorized environment steps. A monotonically
    increasing time stamp forbids false windows across a ring overwrite, even
    inside a very long episode. Train/validation ownership is a deterministic
    hash of (environment, episode), so overlapping windows cannot leak between
    the splits. Recorded terminal next states must be captured BEFORE reset.
    """

    def __init__(self, num_envs, capacity, state_dim, action_dim, context_dim,
                 device, holdout_fraction=0.1, num_families=3):
        self.num_envs, self.capacity = int(num_envs), int(capacity)
        self.state_dim, self.action_dim = int(state_dim), int(action_dim)
        self.context_dim, self.num_families = int(context_dim), int(num_families)
        self.device = torch.empty(0, device=device).device
        self.holdout_fraction = float(holdout_fraction)
        if min(self.num_envs, self.capacity, self.state_dim, self.action_dim,
               self.num_families) < 1 or self.context_dim < 0:
            raise ValueError("Invalid world-model replay dimensions")
        if not 0 <= self.holdout_fraction < 1:
            raise ValueError("holdout_fraction must be in [0, 1)")
        shape = (self.capacity, self.num_envs)
        self.states = torch.empty(*shape, self.state_dim, device=self.device)
        self.next_states = torch.empty_like(self.states)
        self.actions = torch.empty(*shape, self.action_dim, device=self.device)
        self.context = torch.empty(*shape, self.context_dim, device=self.device)
        self.dt = torch.empty(*shape, 1, device=self.device)
        self.rewards = torch.empty(*shape, device=self.device)
        self.terminated = torch.zeros(*shape, device=self.device, dtype=torch.bool)
        self.truncated = torch.zeros_like(self.terminated)
        self.success = torch.zeros_like(self.terminated)
        self.valid = torch.zeros_like(self.terminated)
        self.holdout = torch.zeros_like(self.terminated)
        self.family = torch.zeros(*shape, device=self.device, dtype=torch.long)
        self.episode_ids = torch.full_like(self.family, -1)
        self.stamp = torch.full_like(self.family, -1)
        self._episodes = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._env_ids = torch.arange(self.num_envs, device=self.device)
        self.steps = 0

    @property
    def size(self):
        """Number of valid transitions (a scalar diagnostic synchronizes CUDA)."""
        return int(self.valid.sum().item())

    @property
    def memory_bytes(self):
        return sum(v.numel() * v.element_size() for v in vars(self).values()
                   if isinstance(v, torch.Tensor))

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, device=self.device, dtype=dtype).detach()

    @torch.no_grad()
    def add(self, states, actions, next_states, context, dt, rewards,
            terminated, truncated, family, episode_ids=None, valid=None,
            success=None):
        """Copy one complete vector step without retaining simulator buffers.

        ``valid=False`` excludes a transition with no trustworthy terminal state
        (or an external reset); any multi-step window containing it is excluded.
        Passing explicit episode IDs also handles resets outside normal ``done``.
        """
        row = self.steps % self.capacity
        states = self._tensor(states).reshape(self.num_envs, self.state_dim)
        actions = self._tensor(actions).reshape(self.num_envs, self.action_dim)
        next_states = self._tensor(next_states).reshape(self.num_envs, self.state_dim)
        context = self._tensor(context).reshape(self.num_envs, self.context_dim)
        durations = self._tensor(dt).reshape(-1, 1).expand(self.num_envs, 1)
        rewards = self._tensor(rewards).reshape(self.num_envs)
        terminated = self._tensor(terminated, torch.bool).reshape(self.num_envs)
        truncated = self._tensor(truncated, torch.bool).reshape(self.num_envs)
        family = self._tensor(family, torch.long).reshape(self.num_envs)
        episodes = (self._episodes if episode_ids is None else
                    self._tensor(episode_ids, torch.long).reshape(self.num_envs))
        finite = (torch.isfinite(states).all(-1) & torch.isfinite(next_states).all(-1)
                  & torch.isfinite(actions).all(-1) & torch.isfinite(context).all(-1)
                  & torch.isfinite(durations[:, 0]) & (durations[:, 0] > 0)
                  & torch.isfinite(rewards) & (family >= 0)
                  & (family < self.num_families) & (episodes >= 0))
        if valid is not None:
            finite &= self._tensor(valid, torch.bool).reshape(self.num_envs)
        self.states[row].copy_(states)
        self.next_states[row].copy_(next_states)
        self.actions[row].copy_(actions)
        self.context[row].copy_(context)
        self.dt[row].copy_(durations)
        self.rewards[row].copy_(rewards)
        self.terminated[row].copy_(terminated)
        self.truncated[row].copy_(truncated)
        self.success[row].copy_(torch.zeros_like(terminated) if success is None else
                                self._tensor(success, torch.bool).reshape(self.num_envs))
        self.family[row].copy_(family)
        self.episode_ids[row].copy_(episodes)
        self.stamp[row].fill_(self.steps)
        self.valid[row].copy_(finite)
        # Integer hashing; all transitions of one episode get the same split.
        hashed = ((episodes * 1103515245 + (self._env_ids + 1) * 12345)
                  % 2147483647) % 10000
        self.holdout[row].copy_(hashed < int(self.holdout_fraction * 10000))
        self._episodes.copy_(episodes + (terminated | truncated).long())
        self.steps += 1

    def _split_mask(self, split):
        if split == "train":
            return self.valid & ~self.holdout
        if split in ("validation", "holdout"):
            return self.valid & self.holdout
        if split == "all":
            return self.valid.clone()
        raise ValueError("split must be train, validation, holdout, or all")

    @torch.no_grad()
    def window_mask(self, k, split="train"):
        k = int(k)
        if not 1 <= k <= self.capacity:
            raise ValueError("window must be between 1 and replay capacity")
        mask = self._split_mask(split)
        done = self.terminated | self.truncated
        for offset in range(1, k):
            mask &= (torch.roll(self.valid, -offset, 0)
                     & (torch.roll(self.episode_ids, -offset, 0) == self.episode_ids)
                     & (torch.roll(self.family, -offset, 0) == self.family)
                     & (torch.roll(self.stamp, -offset, 0) == self.stamp + offset)
                     & ~torch.roll(done, -(offset - 1), 0))
        return mask

    def can_sample(self, k=1, split="train"):
        return bool(self.window_mask(k, split).any())

    @torch.no_grad()
    def _sample_indices(self, mask, n, balance_families=True,
                        recent_fraction=0.0, recent_steps=32):
        if int(n) < 1:
            raise ValueError("sample batch size must be positive")
        if not 0 <= float(recent_fraction) <= 1:
            raise ValueError("recent_fraction must be in [0, 1]")
        # nonzero is the only dynamic cardinality operation. Payloads never move
        # to the CPU; the candidate indices and multinomial sampler stay on GPU.
        indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        if indices.numel() == 0:
            raise RuntimeError("World-model replay has no sampleable windows")
        families = self.family.reshape(-1)[indices]
        weights = torch.ones(indices.numel(), device=self.device)
        if balance_families:
            counts = torch.bincount(families, minlength=self.num_families)
            weights /= counts[families].clamp_min(1)
        weights /= weights.sum()
        if recent_fraction > 0:
            recent = (self.stamp.reshape(-1)[indices] >=
                      self.steps - max(1, int(recent_steps))).float()
            if balance_families:
                counts = torch.zeros(self.num_families, device=self.device)
                counts.scatter_add_(0, families, recent)
                recent /= counts[families].clamp_min(1)
            recent = torch.where(recent.sum() > 0,
                                 recent / recent.sum().clamp_min(1e-12), weights)
            weights = (1 - recent_fraction) * weights + recent_fraction * recent
        sampled = indices[torch.multinomial(weights, int(n), replacement=True)]
        return sampled // self.num_envs, sampled % self.num_envs

    @torch.no_grad()
    def sample_windows(self, n, k, split="train", balance_families=True,
                       recent_fraction=0.0, recent_steps=32):
        row, env = self._sample_indices(self.window_mask(k, split), n,
                                       balance_families, recent_fraction, recent_steps)
        offsets = torch.arange(k, device=self.device)[:, None]
        rr = (row[None, :] + offsets) % self.capacity
        ee = env[None, :].expand(k, -1)
        # Explicit next states retain the final physical state when k ends at a
        # termination. No reset observation can become a supervised target.
        states = torch.cat((self.states[row, env][None], self.next_states[rr, ee]), 0)
        return WindowBatch(states, self.actions[rr, ee], self.context[rr, ee],
                           self.dt[rr, ee], self.rewards[rr, ee],
                           self.terminated[rr, ee], self.truncated[rr, ee],
                           self.success[rr, ee], self.family[row, env],
                           self.episode_ids[row, env], env)

    @torch.no_grad()
    def sample_starts(self, n, split="train", balance_families=True,
                      recent_fraction=0.0, recent_steps=32):
        row, env = self._sample_indices(self._split_mask(split), n,
                                       balance_families, recent_fraction, recent_steps)
        return StartBatch(self.states[row, env], self.actions[row, env],
                          self.next_states[row, env], self.context[row, env],
                          self.dt[row, env], self.rewards[row, env],
                          self.terminated[row, env], self.truncated[row, env],
                          self.success[row, env], self.family[row, env],
                          self.episode_ids[row, env], env)
