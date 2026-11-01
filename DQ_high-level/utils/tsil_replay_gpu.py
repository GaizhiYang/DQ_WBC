"""Device-resident TSIL episode archive and paper sampler.

The scientific rules are the same as :class:`utils.tsil.TemporalReplay`.
Payloads stay on the training device: complete episode recording, shared
Top-K archives, deadline/reward relabeling, zero-tail returns and weighted
positive-gap sampling. Only small aggregate diagnostics cross to the host.

The archive has 2K stable physical slots per configuration. Its fastest and
highest-return lists contain slot indices, so an episode present in both
lists owns one copy. Updating a list copies only newly retained episodes;
survivors never move. The exact static observation prefix is stored once.

``insert``, ``selected``, ``iter_selected``, ``fastest`` and ``high_return``
are deliberately synchronizing inspection/fixture interfaces. They are not
used by the training path. The original CPU implementation remains the
reference implementation for the replay and sampling rules.
"""

import math

import torch

from utils.tsil import ReplaySelection, TemporalReplay, Trajectory


class GPUTemporalReplay:
    """Batched tensor backend; CPU operation is supported for oracle tests."""

    def __init__(self, state, num_envs, observation_size, action_size, gamma=0.99,
                 topk=5, time_indices=(-191, -190), max_episode_steps=100,
                 static_prefix_size=1024, success_reward_scale=50.0, device=None):
        self.state, self.num_envs = state, int(num_envs)
        # Simulation owns TemporalState; the learner owns replay. These may
        # intentionally use different devices (SIM_DEVICE versus RL_DEVICE).
        # Resolve index-free "cuda" now so subsequent ownership is explicit.
        self.device = torch.empty(0, device=state.device if device is None else device).device
        self.observation_size, self.action_size = int(observation_size), int(action_size)
        self.gamma, self.topk = float(gamma), int(topk)
        self.time_indices = tuple(int(i) % self.observation_size for i in time_indices)
        self.max_episode_steps = int(max_episode_steps)
        self.static_prefix_size = min(int(static_prefix_size), self.observation_size)
        self.success_reward_scale = float(success_reward_scale)
        if not 0 <= self.gamma <= 1 or self.topk < 1 or self.max_episode_steps < 1:
            raise ValueError("Invalid TSIL replay configuration")
        if not math.isfinite(self.success_reward_scale) or self.success_reward_scale <= 0:
            raise ValueError("TSIL success reward scale must be finite and positive")
        self._capacity = 2 * self.topk
        self._num_slots = state.num_configs * self._capacity
        options = dict(device=self.device)
        self._episode_ids = torch.full((self.num_envs,), -1, dtype=torch.long, **options)
        self._config_ids = self._episode_ids.clone()
        self._lengths = torch.zeros(self.num_envs, dtype=torch.long, **options)
        self._valid = torch.zeros(self.num_envs, dtype=torch.bool, **options)
        self._scores = torch.zeros(self.num_envs, dtype=torch.float64, **options)
        self._completed_count = torch.zeros((), dtype=torch.long, **options)
        self._aborted_count = torch.zeros((), dtype=torch.long, **options)
        self._slot_valid = torch.zeros(self._num_slots, dtype=torch.bool, **options)
        self._slot_uid = torch.full((self._num_slots, 2), -1, dtype=torch.long, **options)
        self._slot_length = torch.zeros(self._num_slots, dtype=torch.long, **options)
        self._slot_success = torch.zeros(self._num_slots, dtype=torch.bool, **options)
        # Double metadata matches the CPU recorder's discounted-score sum and
        # direct Trajectory fixtures' Python float duration/tie ordering.
        self._slot_elapsed = torch.zeros(self._num_slots, dtype=torch.float64, **options)
        self._slot_score = torch.zeros(self._num_slots, dtype=torch.float64, **options)
        self._fast = torch.full((state.num_configs, self.topk), -1, dtype=torch.long, **options)
        self._high = self._fast.clone()
        self._local_slots = torch.arange(self._capacity, device=self.device)
        self._steps = torch.arange(self.max_episode_steps, device=self.device)
        self._prefix, self._allocated = None, False
        self._generation, self._debug_generation = 0, -1
        self._debug_fast, self._debug_high = {}, {}

    def _allocate(self, states):
        p = self.static_prefix_size
        if p and torch.equal(states[:, :p], states[:1, :p].expand(len(states), -1)):
            # Time channels must remain relabelable. A one-step debug fixture
            # must not accidentally compress the deadline into its prefix.
            p = min(p, min(self.time_indices))
            self.static_prefix_size = p
            self._prefix = states[0, :p].detach().to(self.device, torch.float32).clone() if p else None
        else:
            self.static_prefix_size = p = 0
        width = self.observation_size - p
        self._payload_width = width + self.action_size + 3
        self._ring = torch.empty((self.num_envs, self.max_episode_steps, self._payload_width),
                                 dtype=torch.float32, device=self.device)
        self._archive = torch.empty((self._num_slots, self.max_episode_steps, self._payload_width),
                                    dtype=torch.float32, device=self.device)
        self._obs, self._archive_obs = self._ring[..., :width], self._archive[..., :width]
        stop = width + self.action_size
        self._actions, self._archive_actions = self._ring[..., width:stop], self._archive[..., width:stop]
        self._fixed, self._archive_fixed = self._ring[..., stop], self._archive[..., stop]
        self._command, self._archive_command = self._ring[..., stop + 1], self._archive[..., stop + 1]
        self._raw, self._archive_raw = self._ring[..., stop + 2], self._archive[..., stop + 2]
        self._allocated = True

    def _vector(self, value, dtype):
        return torch.as_tensor(value, device=self.device, dtype=dtype).detach().reshape(-1)

    @torch.no_grad()
    def record(self, states, actions, raw_rewards, terminated, tsil_info):
        states = states.detach().to(device=self.device, dtype=torch.float32)
        actions = actions.detach().to(device=self.device, dtype=torch.float32)
        if not self._allocated:
            self._allocate(states)
        if self._prefix is not None and not torch.equal(
                states[:, :self.static_prefix_size], self._prefix.expand(self.num_envs, -1)):
            raise ValueError("TSIL static feature changed: disable prefix compression for noisy/dynamic features")
        ids = self._vector(tsil_info["config_id"], torch.long)
        episodes = self._vector(tsil_info["episode_id"], torch.long)
        done = self._vector(terminated, torch.bool)
        natural = self._vector(tsil_info.get("natural_done", terminated), torch.bool)
        success = self._vector(tsil_info["success"], torch.bool)
        elapsed = self._vector(tsil_info["elapsed"], torch.float32)
        fixed = self._vector(tsil_info.get("reward_fixed", tsil_info.get("r_fixed")), torch.float32)
        command = self._vector(tsil_info.get("reward_command_raw", tsil_info.get("command_raw")), torch.float32)
        raw = self._vector(raw_rewards, torch.float32)
        starts = episodes != self._episode_ids
        self._aborted_count.add_((starts & self._valid & (self._lengths > 0)).sum())
        self._lengths[starts] = 0
        self._scores[starts] = 0
        self._episode_ids[starts] = episodes[starts]
        self._config_ids[starts] = ids[starts]
        start_time = states[:, self.time_indices[0]]
        self._valid[starts] = ((start_time[starts].abs() <= 1e-6)
                               & (ids[starts] >= 0) & (ids[starts] < self.state.num_configs))
        if bool((self._valid & (ids != self._config_ids)).any()):
            raise ValueError("TSIL configuration changed without an episode reset")
        active = self._valid.nonzero(as_tuple=False).flatten()
        lengths = self._lengths[active]
        if bool((lengths >= self.max_episode_steps).any()):
            raise ValueError("TSIL episode exceeded the configured finite task budget")
        self._obs[active, lengths] = states[active, self.static_prefix_size:]
        self._actions[active, lengths] = actions[active]
        self._fixed[active, lengths], self._command[active, lengths] = fixed[active], command[active]
        self._raw[active, lengths] = raw[active]
        self._scores[active] += (self.gamma ** lengths.double()) * raw[active].double()
        self._lengths[active] += 1
        ended = done & self._valid
        successful = ended & natural & success
        self.state.observe_success(ids[successful], elapsed[successful])
        finished = (ended & natural).nonzero(as_tuple=False).flatten()
        self._completed_count.add_(finished.numel())
        self._aborted_count.add_((ended & ~natural).sum())
        if finished.numel():
            self._admit_batch(ids[finished], torch.stack((finished, episodes[finished]), -1),
                              self._lengths[finished], success[finished], elapsed[finished].double(),
                              self._scores[finished], finished, self._ring, strict_admission=True)
        self._valid[ended] = False
        self._lengths[ended] = 0

    @torch.no_grad()
    def abort(self, env_ids=None):
        ids = (torch.arange(self.num_envs, device=self.device) if env_ids is None else
               self._vector(env_ids, torch.long))
        self._aborted_count.add_((self._valid[ids] & (self._lengths[ids] > 0)).sum())
        self._valid[ids] = False
        self._lengths[ids] = 0

    def _strict_admission(self, configs, success, elapsed, scores):
        """Match the CPU recorder's sequential strict admission thresholds.

        Ended environments arrive in ascending env-ID order. A candidate can
        enter a full pool only by improving its numeric threshold; a smaller
        UID alone does not admit an exact tie. The kth prior numeric threshold
        is unchanged by skipped candidates, so counting prior values at least
        as good exactly reproduces the sequential rule. Metadata comparisons
        are chunked and vectorized, including the all-envs-one-config case.
        """
        slots = configs[:, None] * self._capacity + self._local_slots
        valid = self._slot_valid[slots]
        high_counts = (valid & (self._slot_score[slots] >= scores[:, None])).sum(-1)
        fast_counts = (valid & self._slot_success[slots]
                       & (self._slot_elapsed[slots] <= elapsed[:, None])).sum(-1)
        order = torch.arange(len(configs), device=self.device)
        for begin in range(0, len(configs), 256):
            stop = min(begin + 256, len(configs))
            prior = ((configs[begin:stop, None] == configs[None, :])
                     & (order[None, :] < order[begin:stop, None]))
            high_counts[begin:stop] += (prior & (scores[None, :] >= scores[begin:stop, None])).sum(-1)
            fast_counts[begin:stop] += (prior & success[None, :]
                                       & (elapsed[None, :] <= elapsed[begin:stop, None])).sum(-1)
        return (high_counts < self.topk) | (success & (fast_counts < self.topk))

    @staticmethod
    def _group_ranks(configs):
        position = torch.arange(len(configs), device=configs.device)
        boundary = torch.ones_like(configs, dtype=torch.bool)
        boundary[1:] = configs[1:] != configs[:-1]
        starts = torch.where(boundary, position, torch.zeros_like(position)).cummax(0).values
        return position - starts

    def _rank_pool(self, configs, uid, value, eligible, descending):
        order = eligible.nonzero(as_tuple=False).flatten()
        # Stable sorts implement the CPU's (primary score, env ID, serial)
        # lexicographic order within each configuration, without packing keys.
        for key, reverse in ((uid[:, 1], False), (uid[:, 0], False),
                             (value, descending), (configs, False)):
            order = order[torch.argsort(key[order], stable=True, descending=reverse)]
        ranks = self._group_ranks(configs[order])
        keep = ranks < self.topk
        return order[keep], ranks[keep]

    @torch.no_grad()
    def _admit_batch(self, configs, uid, lengths, success, elapsed, scores,
                     source_rows, source_payload, strict_admission=False):
        count = len(configs)
        admitted = torch.zeros(count, dtype=torch.bool, device=self.device)
        if not count:
            return admitted
        if not bool((torch.isfinite(elapsed) & (elapsed > 0) & torch.isfinite(scores)).all()):
            raise ValueError("Invalid archived TSIL episode score or duration")
        old_by_config = configs[:, None] * self._capacity + self._local_slots
        duplicate = (self._slot_valid[old_by_config]
                     & (self._slot_uid[old_by_config] == uid[:, None, :]).all(-1)).any(-1)
        qualifies = ~duplicate
        if strict_admission:
            qualifies &= self._strict_admission(configs, success, elapsed, scores)
        incoming = qualifies.nonzero(as_tuple=False).flatten()
        if not incoming.numel():
            return admitted
        configs, uid, lengths = configs[incoming], uid[incoming], lengths[incoming]
        success, elapsed, scores = success[incoming], elapsed[incoming], scores[incoming]
        source_rows = source_rows[incoming]
        touched = torch.unique(configs, sorted=True)
        touched_slots = (touched[:, None] * self._capacity + self._local_slots).reshape(-1)
        old_slots = touched_slots[self._slot_valid[touched_slots]]
        old_count = len(old_slots)
        combined_config = torch.cat((old_slots // self._capacity, configs))
        combined_uid = torch.cat((self._slot_uid[old_slots], uid))
        combined_success = torch.cat((self._slot_success[old_slots], success))
        combined_score = torch.cat((self._slot_score[old_slots], scores))
        combined_elapsed = torch.cat((self._slot_elapsed[old_slots], elapsed))
        high, high_ranks = self._rank_pool(combined_config, combined_uid, combined_score,
                                          torch.ones_like(combined_success), True)
        fast, fast_ranks = self._rank_pool(combined_config, combined_uid, combined_elapsed,
                                          combined_success, False)
        retained = torch.zeros(len(combined_config), dtype=torch.bool, device=self.device)
        retained[high], retained[fast] = True, True
        new_kept = retained[old_count:].nonzero(as_tuple=False).flatten()
        # Survivors keep their original physical slot; evictions immediately
        # release slots. No existing trajectory payload is gathered or moved.
        self._slot_valid[old_slots] = retained[:old_count]
        local_free = torch.where(~self._slot_valid[touched_slots].reshape(-1, self._capacity),
                                 self._local_slots.expand(len(touched), -1), self._capacity)
        local_free = local_free.sort(-1).values
        new_kept = new_kept[torch.argsort(configs[new_kept], stable=True)]
        new_config = configs[new_kept]
        free_rank = self._group_ranks(new_config)
        slots = (new_config * self._capacity
                 + local_free[torch.searchsorted(touched, new_config), free_rank])
        mapping = torch.full((len(combined_config),), -1, dtype=torch.long, device=self.device)
        mapping[:old_count] = old_slots
        mapping[old_count + new_kept] = slots
        self._slot_valid[slots] = True
        self._slot_uid[slots] = uid[new_kept]
        self._slot_length[slots] = lengths[new_kept]
        self._slot_success[slots] = success[new_kept]
        self._slot_elapsed[slots] = elapsed[new_kept]
        self._slot_score[slots] = scores[new_kept]
        # Bound the temporary gather even if all 8192 environments finish at
        # once. Padded rows are never read by relabeling or debug export.
        for begin in range(0, len(slots), 128):
            end = begin + 128
            self._archive[slots[begin:end]] = source_payload[source_rows[new_kept[begin:end]]]
        self._high[touched], self._fast[touched] = -1, -1
        self._high[combined_config[high], high_ranks] = mapping[high]
        self._fast[combined_config[fast], fast_ranks] = mapping[fast]
        admitted[incoming[new_kept]] = True
        self._generation += 1
        return admitted

    def _selected_slots(self):
        candidates = torch.cat((self._fast, self._high), -1)
        positions = self._local_slots
        duplicates = ((candidates[:, :, None] == candidates[:, None, :])
                      & (positions[None, None, :] < positions[None, :, None])).any(-1)
        valid = (candidates >= 0) & ~duplicates
        return candidates[valid & (valid.long().cumsum(-1) <= self.topk)]

    def _stat_tensors(self):
        return torch.stack(((self._high >= 0).any(-1).sum(), self._slot_valid.sum(),
                            (self._fast >= 0).any(-1).sum(), self._completed_count, self._aborted_count))

    @property
    def completed_episodes(self):
        return int(self._completed_count.item())

    @property
    def aborted_episodes(self):
        return int(self._aborted_count.item())

    @property
    def stats(self):
        names = ("archive_configurations", "archive_trajectories", "archive_success_configurations",
                 "completed_episodes", "aborted_episodes")
        return dict(zip(names, self._stat_tensors().cpu().tolist()))

    def _empty_selection(self, stats):
        def empty(*shape, dtype=torch.float32):
            return torch.empty(shape, dtype=dtype, device=self.device)
        # No compression has been verified before the first recorded state.
        width = self.observation_size - (self.static_prefix_size if self._allocated else 0)
        return ReplaySelection(empty(0, width),
                               empty(0, self.action_size), empty(0, 1), empty(0, 1),
                               empty(0, dtype=torch.long), empty(0, dtype=torch.bool), empty(0),
                               empty(0, 3, dtype=torch.long), self._prefix, stats)

    def _returns(self, rewards):
        if self.gamma == 0:
            return rewards.clone()
        # Accumulate in double, matching the CPU reference's Python recurrence.
        # At the finite 100-step horizon float64 cumsum is well conditioned for
        # the training gamma. The recurrence handles extreme underflow cases.
        if self.gamma ** (self.max_episode_steps - 1) < 1e-280:
            result = torch.empty_like(rewards)
            running = torch.zeros(len(rewards), dtype=torch.float64, device=self.device)
            for step in range(self.max_episode_steps - 1, -1, -1):
                running = rewards[:, step].double() + self.gamma * running
                result[:, step] = running
            return result
        power = self.gamma ** self._steps.double()
        weighted = rewards.double() * power
        return (weighted.flip(-1).cumsum(-1).flip(-1) / power).float()

    @torch.no_grad()
    def prepare(self, screen_gap_fn, reward_scale=1.0, command_gate=1.0,
                transitions_per_trajectory=32, screen_batch_size=4096):
        if transitions_per_trajectory < 1 or screen_batch_size < 1:
            raise ValueError("TSIL sampling and screening batch sizes must be positive")
        if not math.isfinite(float(reward_scale)) or reward_scale <= 0:
            raise ValueError("TSIL reward scale must be finite and positive")
        slots = self._selected_slots()
        if not len(slots):
            stats = dict(self.stats, replay_trajectories=0, replay_dataset_steps=0,
                         replay_positive_available=0, replay_steps=0, replay_success_fraction=0.)
            return self._empty_selection(stats)
        configs, lengths = slots // self._capacity, self._slot_length[slots]
        success, elapsed = self._slot_success[slots], self._slot_elapsed[slots]
        targets = self.state.targets.to(self.device)[configs].double()
        deadlines = (targets / self.state.t_ref).float()
        efficiency = 1. + torch.where(success, (targets / elapsed).clamp_max(1.), torch.zeros_like(targets))
        rewards = self._archive_fixed[slots] + float(command_gate) * self._archive_command[slots]
        row = torch.arange(len(slots), device=self.device)
        rewards[row, lengths - 1] += (self.success_reward_scale * efficiency * success).float()
        valid = self._steps[None, :] < lengths[:, None]
        # Do not propagate uninitialized padded episode data into the return.
        rewards = torch.where(valid, rewards / float(reward_scale), torch.zeros_like(rewards))
        returns = self._returns(rewards)
        flat_valid = valid.reshape(-1).nonzero(as_tuple=False).flatten()
        gaps = torch.zeros_like(rewards)
        invalid_gap = torch.zeros((), dtype=torch.bool, device=self.device)
        for begin in range(0, len(flat_valid), int(screen_batch_size)):
            flat = flat_valid[begin:begin + int(screen_batch_size)]
            trajectory, step = flat // self.max_episode_steps, flat % self.max_episode_steps
            suffix = self._archive_obs[slots[trajectory], step]
            suffix[:, self.time_indices[1] - self.static_prefix_size] = deadlines[trajectory]
            states = (suffix if self._prefix is None else
                      torch.cat((self._prefix.expand(len(flat), -1), suffix), -1))
            gap = screen_gap_fn(states, returns[trajectory, step, None]).detach().to(
                device=self.device, dtype=torch.float32).reshape(-1)
            if gap.numel() != len(flat):
                raise ValueError("TSIL screening returned invalid value gaps")
            invalid_gap |= ~torch.isfinite(gap).all()
            gaps[trajectory, step] = gap.clamp_min(0)
        if bool(invalid_gap):
            raise ValueError("TSIL screening returned invalid value gaps")
        # Independent exponential races implement gap-proportional sampling
        # without replacement (the same Plackett-Luce rule as multinomial).
        # One race per complete trajectory keeps the <=32 budget independent
        # of screening microbatch boundaries and never samples zero gaps.
        noise = torch.empty_like(gaps).exponential_().clamp_min_(torch.finfo(gaps.dtype).tiny)
        keys = gaps.double().log() - noise.double().log()
        count = min(int(transitions_per_trajectory), self.max_episode_steps)
        choices = keys.topk(count, dim=-1).indices
        chosen_positive = gaps.gather(1, choices) > 0
        trajectory = row[:, None].expand(-1, count)[chosen_positive]
        step = choices[chosen_positive]
        chosen_slots = slots[trajectory]
        suffix = self._archive_obs[chosen_slots, step]
        suffix[:, self.time_indices[1] - self.static_prefix_size] = deadlines[trajectory]
        actions = self._archive_actions[chosen_slots, step]
        sampled_success = success[trajectory]
        values = torch.cat((self._stat_tensors(), (gaps > 0).sum().reshape(1))).cpu().tolist()
        names = ("archive_configurations", "archive_trajectories", "archive_success_configurations",
                 "completed_episodes", "aborted_episodes", "replay_positive_available")
        stats = dict(zip(names, values))
        stats.update(replay_trajectories=len(slots), replay_dataset_steps=len(flat_valid),
                     replay_steps=len(step), replay_success_fraction=(
                         float(sampled_success.float().mean()) if len(step) else 0.))
        return ReplaySelection(suffix, actions, returns[trajectory, step, None],
                               efficiency[trajectory, None].float(), configs[trajectory],
                               sampled_success, elapsed[trajectory].float(),
                               torch.cat((self._slot_uid[chosen_slots], step[:, None]), -1),
                               self._prefix, stats)

    @torch.no_grad()
    def insert(self, trajectory):
        """Synchronizing fixture/debug insertion, with the CPU insertion rule."""
        c, length = trajectory.config_id, len(trajectory)
        if c < 0 or c >= self.state.num_configs or not length or length > self.max_episode_steps:
            raise ValueError("Invalid archived TSIL episode")
        if not math.isfinite(trajectory.return_score) or not math.isfinite(trajectory.elapsed) or trajectory.elapsed <= 0:
            raise ValueError("Invalid archived TSIL episode score or duration")
        states = trajectory.full_observations().detach().to(self.device, torch.float32)
        if not self._allocated:
            self._allocate(states)
        if self._prefix is not None and not torch.equal(
                states[:, :self.static_prefix_size], self._prefix.expand(length, -1)):
            raise ValueError("TSIL static feature changed: disable prefix compression for noisy/dynamic features")
        payload = torch.zeros((1, self.max_episode_steps, self._payload_width), device=self.device)
        width = self.observation_size - self.static_prefix_size
        payload[0, :length, :width] = states[:, self.static_prefix_size:]
        payload[0, :length, width:width + self.action_size] = trajectory.actions.to(self.device)
        for column, source in enumerate((trajectory.reward_fixed, trajectory.reward_command_raw,
                                         trajectory.collected_rewards), width + self.action_size):
            payload[0, :length, column] = source.to(self.device).reshape(-1)
        result = self._admit_batch(
            torch.tensor([c], dtype=torch.long, device=self.device),
            torch.tensor([trajectory.uid], dtype=torch.long, device=self.device),
            torch.tensor([length], dtype=torch.long, device=self.device),
            torch.tensor([trajectory.success], dtype=torch.bool, device=self.device),
            torch.tensor([trajectory.elapsed], dtype=torch.float64, device=self.device),
            torch.tensor([trajectory.return_score], dtype=torch.float64, device=self.device),
            torch.zeros(1, dtype=torch.long, device=self.device), payload)
        return bool(result[0])

    def _refresh_debug_export(self):
        if self._debug_generation == self._generation:
            return
        self._debug_fast, self._debug_high = {}, {}
        fast, high = self._fast.cpu().tolist(), self._high.cpu().tolist()
        occupied = self._slot_valid.nonzero(as_tuple=False).flatten()
        if not len(occupied):
            self._debug_generation = self._generation
            return
        # Export only requested occupied slots once; both logical lists share
        # the same owned Trajectory objects, exactly like the CPU reference.
        uid, length = self._slot_uid[occupied].cpu().tolist(), self._slot_length[occupied].cpu().tolist()
        success = self._slot_success[occupied].cpu().tolist()
        elapsed, score = self._slot_elapsed[occupied].cpu().tolist(), self._slot_score[occupied].cpu().tolist()
        prefix = self._prefix.cpu().clone() if self._prefix is not None else None
        exported = {}
        width = self.observation_size - self.static_prefix_size
        for index, slot in enumerate(occupied.cpu().tolist()):
            payload = self._archive[slot, :length[index]].cpu().clone()
            stop = width + self.action_size
            exported[slot] = Trajectory(tuple(uid[index]), slot // self._capacity,
                                       payload[:, :width], payload[:, width:stop], payload[:, stop],
                                       payload[:, stop + 1], payload[:, stop + 2], success[index],
                                       elapsed[index], score[index], prefix)
        for target, table in ((self._debug_fast, fast), (self._debug_high, high)):
            for config, slots in enumerate(table):
                entries = [exported[slot] for slot in slots if slot >= 0]
                if entries:
                    target[config] = entries
        self._debug_generation = self._generation

    @property
    def fastest(self):
        self._refresh_debug_export()
        return self._debug_fast

    @property
    def high_return(self):
        self._refresh_debug_export()
        return self._debug_high

    def selected(self, config_id):
        self._refresh_debug_export()
        result, seen = [], set()
        for trajectory in self._debug_fast.get(config_id, []) + self._debug_high.get(config_id, []):
            if trajectory.uid not in seen:
                result.append(trajectory)
                seen.add(trajectory.uid)
            if len(result) == self.topk:
                break
        return result

    def iter_selected(self):
        self._refresh_debug_export()
        for config in sorted(set(self._debug_fast) | set(self._debug_high)):
            yield from self.selected(config)

    def relabel(self, trajectory, target, reward_scale=1.0, command_gate=1.0):
        """Debug/reference relabel; training uses batched device relabeling."""
        return TemporalReplay.relabel(self, trajectory, target, reward_scale, command_gate)
