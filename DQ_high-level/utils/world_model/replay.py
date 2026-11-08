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
    valid_mask: torch.Tensor = None  # optional [K,B], terminal included, reset tail excluded


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


class SliceStateCodec:
    """Lossless storage for selected fields of a larger rollout-state layout.

    ``encode`` retains the specified slices, in ``names`` order. ``decode``
    restores their original positions and fills every omitted field with zero.
    Both operations preserve dtype, device and arbitrary leading dimensions;
    they use only Torch operations and also preserve gradients when called
    outside replay's no-grad collection/sampling methods.

    Use this for physical-model fitting only when omitted controller/task
    memory is not consumed. Decoded zeros are not a valid imagined-policy
    starting history; high-level replay must continue storing full states.
    """

    def __init__(self, state_dim, slices, names):
        self.state_dim = int(state_dim)
        self.names = tuple(names)
        if self.state_dim < 1 or not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("State codec requires a positive state_dim and distinct field names")
        fields, occupied, cursor = [], set(), 0
        for name in self.names:
            if name not in slices:
                raise ValueError("Unknown state codec field: " + str(name))
            field = slices[name]
            if (not isinstance(field, slice) or field.start is None or field.stop is None
                    or field.step not in (None, 1)
                    or not 0 <= field.start < field.stop <= self.state_dim):
                raise ValueError("State codec fields require bounded, contiguous slices: " + str(name))
            coordinates = set(range(field.start, field.stop))
            if occupied.intersection(coordinates):
                raise ValueError("State codec fields cannot overlap: " + str(name))
            occupied.update(coordinates)
            width = field.stop - field.start
            fields.append((field, slice(cursor, cursor + width)))
            cursor += width
        self.fields = tuple(fields)
        self.storage_dim = cursor

    def encode(self, states):
        if states.ndim < 1 or states.shape[-1] != self.state_dim:
            raise ValueError("State codec encode expected final dimension {}".format(self.state_dim))
        return torch.cat([states[..., source] for source, _ in self.fields], dim=-1)

    def decode(self, stored):
        if stored.ndim < 1 or stored.shape[-1] != self.storage_dim:
            raise ValueError("State codec decode expected final dimension {}".format(self.storage_dim))
        states = stored.new_zeros(*stored.shape[:-1], self.state_dim)
        for target, source in self.fields:
            states[..., target] = stored[..., source]
        return states


class TransitionReplay:
    """Synchronous GPU ring, laid out [time, environment, feature].

    ``capacity`` is measured in vectorized environment steps. A monotonically
    increasing time stamp forbids false windows across a ring overwrite, even
    inside a very long episode. Train/validation ownership is a deterministic
    hash of (environment, episode), or fixed held-out environment slots. Neither
    split allows overlapping training/validation windows. Recorded terminal
    next states must be captured BEFORE reset.
    """

    def __init__(self, num_envs, capacity, state_dim, action_dim, context_dim,
                 device, holdout_fraction=0.1, num_families=3, state_codec=None,
                 holdout_mode="episode", allow_partial_windows=False):
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
        if holdout_mode not in ("episode", "environment"):
            raise ValueError("holdout_mode must be episode or environment")
        self.holdout_mode = holdout_mode
        self.allow_partial_windows = bool(allow_partial_windows)
        self.holdout_envs = (max(1, round(self.num_envs * self.holdout_fraction))
                             if holdout_mode == "environment" and self.holdout_fraction else 0)
        if self.holdout_envs >= self.num_envs:
            raise ValueError("Environment holdout must leave at least one training world")
        self.state_codec = state_codec
        self.storage_state_dim = self.state_dim
        if state_codec is not None:
            if (not callable(getattr(state_codec, "encode", None))
                    or not callable(getattr(state_codec, "decode", None))):
                raise ValueError("state_codec must provide encode/decode methods")
            if getattr(state_codec, "state_dim", self.state_dim) != self.state_dim:
                raise ValueError("state_codec state_dim does not match replay state_dim")
            self.storage_state_dim = int(getattr(state_codec, "storage_dim", 0))
            if self.storage_state_dim < 1:
                raise ValueError("state_codec storage_dim must be positive")
        shape = (self.capacity, self.num_envs)
        # state_dim is the external contract. Only the resident state payload
        # changes width; sampled batches always contain decoded full states.
        self.states = torch.empty(*shape, self.storage_state_dim, device=self.device)
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
        # Monotone sample clock, independent of ring size/overwrites. Updated
        # once per newly collected real transition; no replay sampling counts.
        self.valid_train_transitions = torch.zeros((), device=self.device, dtype=torch.long)
        # Training performs many draws between vectorized environment steps.
        # Reuse eligibility and probabilities until their source tensors change.
        # These derived values are not part of any replay/checkpoint payload.
        self._sampling_cache = {}
        self._sampling_cache_version = None

    @property
    def size(self):
        """Number of valid transitions (a scalar diagnostic synchronizes CUDA)."""
        return int(self.valid.sum().item())

    @property
    def memory_bytes(self):
        resident = sum(v.numel() * v.element_size() for v in vars(self).values()
                       if isinstance(v, torch.Tensor))
        cached = sum(v.numel() * v.element_size()
                     for entry in self._sampling_cache.values() for v in entry.values()
                     if isinstance(v, torch.Tensor))
        return resident + cached

    def _invalidate_sampling_cache(self):
        self._sampling_cache.clear()
        self._sampling_cache_version = None

    def _sampling_version(self):
        # Tensor versions are host-side metadata, with no CUDA synchronization.
        # Besides add(), support explicit in-place invalidation/split edits by
        # diagnostic callers and replacing one of the public index tensors.
        sources = (self.valid, self.holdout, self.episode_ids, self.family,
                   self.stamp, self.terminated, self.truncated)
        try:
            return (self.steps, tuple((id(value), value._version) for value in sources))
        except RuntimeError:
            # Tensors created inside inference_mode do not have version
            # counters. Conservatively rebuild rather than risk stale indices.
            return None

    def _tensor(self, value, dtype=torch.float32):
        return torch.as_tensor(value, device=self.device, dtype=dtype).detach()

    def _encode_states(self, states):
        return states if self.state_codec is None else self.state_codec.encode(states)

    def _decode_states(self, states):
        return states if self.state_codec is None else self.state_codec.decode(states)

    @torch.no_grad()
    def add(self, states, actions, next_states, context, dt, rewards,
            terminated, truncated, family, episode_ids=None, valid=None,
            success=None):
        """Copy one complete vector step without retaining simulator buffers.

        ``valid=False`` excludes a transition with no trustworthy terminal state
        (or an external reset). Complete windows containing it are excluded;
        partial windows mask that transition and the remaining suffix.
        Passing explicit episode IDs also handles resets outside normal ``done``.
        Returns the valid training-row mask for one-time normalization updates.
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
        self._invalidate_sampling_cache()
        self.states[row].copy_(self._encode_states(states))
        self.next_states[row].copy_(self._encode_states(next_states))
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
        if self.holdout_mode == "environment":
            held_out = self._env_ids >= self.num_envs - self.holdout_envs
        else:
            # Integer hashing; all transitions of one episode get the same split.
            hashed = ((episodes * 1103515245 + (self._env_ids + 1) * 12345)
                      % 2147483647) % 10000
            held_out = hashed < int(self.holdout_fraction * 10000)
        self.holdout[row].copy_(held_out)
        train_mask = finite & ~held_out
        self.valid_train_transitions.add_(train_mask.sum())
        self._episodes.copy_(episodes + (terminated | truncated).long())
        self.steps += 1
        return train_mask

    def _split_mask(self, split):
        if split == "train":
            return self.valid & ~self.holdout
        if split in ("validation", "holdout"):
            return self.valid & self.holdout
        if split == "all":
            return self.valid.clone()
        raise ValueError("split must be train, validation, holdout, or all")

    @torch.no_grad()
    def _window_candidates(self, k, split="train"):
        k = int(k)
        if not 1 <= k <= self.capacity:
            raise ValueError("window must be between 1 and replay capacity")
        if split == "holdout":
            split = "validation"
        if split not in ("train", "validation", "all"):
            raise ValueError("split must be train, validation, holdout, or all")
        version = self._sampling_version()
        if version is None or version != self._sampling_cache_version:
            self._invalidate_sampling_cache()
            self._sampling_cache_version = version
        key = (k, split)
        if key in self._sampling_cache:
            return self._sampling_cache[key]
        mask = self._split_mask(split)
        done = self.terminated | self.truncated
        for offset in range(1, k):
            chronological = torch.roll(self.stamp, -offset, 0) == self.stamp + offset
            if self.allow_partial_windows:
                # All K rows must exist, even when only their terminal prefix
                # is supervised. Never manufacture an uncollected ring tail.
                mask &= chronological
                continue
            mask &= (torch.roll(self.valid, -offset, 0)
                     & (torch.roll(self.episode_ids, -offset, 0) == self.episode_ids)
                     & (torch.roll(self.family, -offset, 0) == self.family)
                     & (torch.roll(self.holdout, -offset, 0) == self.holdout)
                     & chronological
                     & ~torch.roll(done, -(offset - 1), 0))
        indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
        entry = {"mask": mask, "indices": indices}
        self._sampling_cache[key] = entry
        return entry

    @torch.no_grad()
    def window_mask(self, k, split="train"):
        # Preserve the public method's ownership: callers may edit their mask.
        return self._window_candidates(k, split)["mask"].clone()

    def can_sample(self, k=1, split="train"):
        return self._window_candidates(k, split)["indices"].numel() > 0

    @torch.no_grad()
    def _sample_indices(self, candidates, n, balance_families=True,
                        recent_fraction=0.0, recent_steps=32):
        if int(n) < 1:
            raise ValueError("sample batch size must be positive")
        if not 0 <= float(recent_fraction) <= 1:
            raise ValueError("recent_fraction must be in [0, 1]")
        # Candidate discovery (including nonzero's dynamic cardinality) is
        # shared by can_sample and every draw in the same update block.
        indices = candidates["indices"]
        if indices.numel() == 0:
            raise RuntimeError("World-model replay has no sampleable windows")
        options = (bool(balance_families), float(recent_fraction),
                   max(1, int(recent_steps)) if recent_fraction > 0 else 0)
        if candidates.get("weight_options") == options:
            weights = candidates["weights"]
        else:
            weights = self._sampling_weights(indices, *options)
            # Keep only the latest distribution for this window/split; callers
            # sweeping sampling parameters cannot grow this cache unboundedly.
            candidates.update(weight_options=options, weights=weights)
        sampled = indices[torch.multinomial(weights, int(n), replacement=True)]
        return sampled // self.num_envs, sampled % self.num_envs

    def _sampling_weights(self, indices, balance_families, recent_fraction, recent_steps):
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
        return weights

    @torch.no_grad()
    def sample_windows(self, n, k, split="train", balance_families=True,
                       recent_fraction=0.0, recent_steps=32):
        row, env = self._sample_indices(self._window_candidates(k, split), n,
                                       balance_families, recent_fraction, recent_steps)
        offsets = torch.arange(k, device=self.device)[:, None]
        rr = (row[None, :] + offsets) % self.capacity
        ee = env[None, :].expand(k, -1)
        # Explicit next states retain the final physical state when k ends at a
        # termination. No reset observation can become a supervised target.
        states = torch.cat((self.states[row, env][None], self.next_states[rr, ee]), 0)
        states = self._decode_states(states)
        valid_mask = None
        if self.allow_partial_windows:
            valid_mask = (self.valid[rr, ee]
                          & (self.episode_ids[rr, ee] == self.episode_ids[row, env][None])
                          & (self.family[rr, ee] == self.family[row, env][None])
                          & (self.holdout[rr, ee] == self.holdout[row, env][None])
                          & (self.stamp[rr, ee] == self.stamp[row, env][None] + offsets))
            done = self.terminated[rr, ee] | self.truncated[rr, ee]
            before = torch.cat((torch.ones_like(done[:1]), ~done[:-1]), 0)
            valid_mask = torch.cumprod((valid_mask & before).long(), 0).bool()
        return WindowBatch(states, self.actions[rr, ee], self.context[rr, ee],
                           self.dt[rr, ee], self.rewards[rr, ee],
                           self.terminated[rr, ee], self.truncated[rr, ee],
                           self.success[rr, ee], self.family[row, env],
                           self.episode_ids[row, env], env, valid_mask)

    @torch.no_grad()
    def sample_starts(self, n, split="train", balance_families=True,
                      recent_fraction=0.0, recent_steps=32):
        row, env = self._sample_indices(self._window_candidates(1, split), n,
                                       balance_families, recent_fraction, recent_steps)
        return StartBatch(self._decode_states(self.states[row, env]), self.actions[row, env],
                          self._decode_states(self.next_states[row, env]), self.context[row, env],
                          self.dt[row, env], self.rewards[row, env],
                          self.terminated[row, env], self.truncated[row, env],
                          self.success[row, env], self.family[row, env],
                          self.episode_ids[row, env], env)
