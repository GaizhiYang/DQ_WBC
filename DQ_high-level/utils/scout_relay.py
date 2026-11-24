"""Scout & Relay additions to SGS, independent of Isaac Gym.

Design: doc/SGS_SAPG_scout_relay_design.md, section 4.

* ``branch_layout``: contiguous SAPG environment blocks; branch 0 is the leader.
* ``scout_probabilities``: scout curriculum q_S(k) proportional to
  [(w(p_S,k) + eps) * (1 - p_L,k + eps)^beta]^(1/T) over the fixed bank.
* ``SnapshotBuffer``: per-episode physical snapshots of the relay-source envs.
* ``RelayBank``: the online-growing reset set R. The leader samples B and R with
  one SGS kernel; R's total probability mass is capped by ``alpha_max``.
* ``ISOverlapTracker``: per-configuration Kish ESS of follower samples under
  the leader (hypothesis H1 diagnostics).

Everything here is a training-only simulator privilege. None of it changes the
observations of the deployed leader policy.
"""

import math

import torch

from utils.sgs import SGSSampler, sgs_logits


# Same physical fields as ConfigurationBank.REQUIRED_TENSORS (minus family_id),
# plus the table's mode interval, so a relay entry restores through the very
# same path as a bank configuration.
RELAY_STATE_FIELDS = (
    "robot_root", "object_root", "table_root", "rigid_body_state", "ee_jacobian",
    "dof_pos", "dof_vel", "ee_goal_cart", "ee_goal_orn_rpy", "gripper_target",
    "table_vy", "rotation_enabled", "table_yaw_rate", "table_height",
    "table_change_interval",
)

# Optional (ablation "snapshot completeness"): controller/history buffers that
# ordinary bank resets clear. Only names that exist on the environment are used.
CONTROLLER_STATE_FIELDS = (
    "gait_indices", "clock_inputs", "gait_wait_timer", "is_walking",
    "low_obs_history_buf", "last_low_actions", "commands", "actions",
    "last_actions", "action_history_buf", "command_history_buf",
)

LEADER = 0


def branch_layout(num_envs, num_branches, leader_fraction, device):
    """Return one branch ID per environment, as contiguous blocks.

    Branch 0 (the leader) gets ``round(num_envs * leader_fraction)`` environments;
    the scouts split the rest equally (the last one takes the remainder).
    Domain randomization is drawn i.i.d. per environment, so blocks are unbiased.
    """
    if type(num_branches) is not int or num_branches < 1:
        raise ValueError("SAPG branches must be a positive integer")
    branch = torch.zeros(num_envs, dtype=torch.long, device=device)
    if num_branches == 1:
        return branch
    if not 0.0 < float(leader_fraction) < 1.0:
        raise ValueError("SAPG leader_fraction must be in (0, 1)")
    num_leader = int(round(num_envs * float(leader_fraction)))
    num_scouts = num_branches - 1
    rest = num_envs - num_leader
    if num_leader < 1 or rest < num_scouts:
        raise ValueError("Every SAPG branch needs at least one environment")
    sizes = [rest // num_scouts] * num_scouts
    sizes[-1] += rest - sum(sizes)
    start = num_leader
    for scout, size in enumerate(sizes, start=1):
        branch[start:start + size] = scout
        start += size
    return branch


@torch.no_grad()
def scout_probabilities(scout_sampler, leader_rates, need_beta):
    """Scout curriculum over the fixed bank (design section 4.3).

    beta = 0 recovers an independent per-branch SGS; leader_rates uses the
    leader's p_init convention, so unvisited configurations count as "needed".
    """
    if scout_sampler.mode == "uniform":
        return scout_sampler.probabilities()
    logits = scout_sampler.logits()
    if need_beta:
        logits = logits + float(need_beta) * torch.log(1.0 - leader_rates + scout_sampler.epsilon)
    return torch.softmax(logits / max(scout_sampler.temperature, 1.0), dim=0)


def entropy_of(probabilities):
    tiny = torch.finfo(probabilities.dtype).tiny
    return -(probabilities * probabilities.clamp_min(tiny).log()).sum()


class SnapshotBuffer:
    """Snapshots of the current episode of each relay-source environment.

    A snapshot of progress s (s = interval, 2*interval, ...) is the physical state
    after s high-level steps, i.e. the state from which action s is chosen.
    Episodes are at most ``max_episode_length`` steps, so slot = s // interval - 1
    needs no ring: a new episode simply invalidates its row.
    """

    def __init__(self, source_env_ids, num_envs, max_episode_length, interval,
                 field_specs, device):
        if type(interval) is not int or interval < 1:
            raise ValueError("relay snapshot_interval must be a positive integer")
        self.device = torch.device(device)
        self.interval = interval
        self.num_slots = max(1, (int(max_episode_length) - 1) // interval)
        self.source_env_ids = torch.as_tensor(source_env_ids, device=self.device).long()
        if self.source_env_ids.numel() == 0:
            raise ValueError("relay snapshots need at least one source environment")
        self.row_of_env = torch.full((num_envs,), -1, dtype=torch.long, device=self.device)
        self.row_of_env[self.source_env_ids] = torch.arange(
            self.source_env_ids.numel(), device=self.device)
        rows = self.source_env_ids.numel()
        self.storage = {
            name: torch.zeros((rows, self.num_slots, *shape), dtype=dtype, device=self.device)
            for name, (shape, dtype) in field_specs.items()}
        self.valid = torch.zeros(rows, self.num_slots, dtype=torch.bool, device=self.device)
        self.step = torch.zeros(rows, self.num_slots, dtype=torch.long, device=self.device)

    @torch.no_grad()
    def clear(self, env_ids):
        rows = self.row_of_env[env_ids]
        rows = rows[rows >= 0]
        if rows.numel():
            self.valid[rows] = False

    @torch.no_grad()
    def capture(self, env_ids, progress, values):
        rows = self.row_of_env[env_ids]
        slots = progress.long() // self.interval - 1
        ok = (rows >= 0) & (slots >= 0) & (slots < self.num_slots)
        rows, slots = rows[ok], slots[ok]
        for name, value in values.items():
            self.storage[name][rows, slots] = value[ok].to(self.storage[name].dtype)
        self.valid[rows, slots] = True
        self.step[rows, slots] = progress[ok].long()

    @torch.no_grad()
    def choose(self, env_ids, horizons, fractions):
        """Nearest valid snapshot to f * horizon for each fraction f.

        Only snapshots with 0 < step < horizon qualify (strictly before the hold
        starts, so a restart never cuts a hold count). Duplicate picks within one
        episode keep the smallest fraction. Returns rows [n], slots [n, F], ok [n, F].
        """
        rows = self.row_of_env[env_ids]
        steps = self.step[rows]
        valid = self.valid[rows] & (steps > 0) & (steps < horizons[:, None])
        targets = horizons[:, None].float() * fractions[None, :]
        distance = (steps[:, None, :].float() - targets[:, :, None]).abs()
        distance = torch.where(valid[:, None, :], distance, torch.full_like(distance, math.inf))
        best = distance.argmin(dim=-1)
        ok = torch.isfinite(distance.gather(-1, best[..., None]).squeeze(-1))
        count = fractions.numel()
        same = best[:, :, None] == best[:, None, :]
        earlier = torch.tril(torch.ones(count, count, dtype=torch.bool, device=self.device),
                             diagonal=-1)
        duplicate = (same & earlier[None] & ok[:, None, :]).any(dim=-1)
        return rows, best, ok & ~duplicate

    @torch.no_grad()
    def gather(self, rows, slots):
        return {name: value[rows, slots] for name, value in self.storage.items()}


class RelayBank:
    """Online relay reset set R with SGS statistics and chain bookkeeping.

    An entry is one snapshot of a successful source episode; a chain is the set of
    entries taken from the same episode. Slots are reused, so each allocation gets
    a fresh ``uid``; an episode's outcome is credited only if its slot still holds
    the entry it started from.
    """

    REASONS = ("mastered", "useless", "evicted", "quarantined")

    def __init__(self, capacity, field_specs, num_bank_configs, fractions, history_size,
                 target_success_rate, concentration, epsilon, temperature, alpha_max,
                 sampling, max_chains_per_config, master_rate, master_count,
                 useless_count, device, seed=43):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("relay capacity must be a positive integer")
        if not 0.0 < float(alpha_max) <= 1.0:
            raise ValueError("relay alpha_max must be in (0, 1]")
        if sampling not in ("sgs", "uniform"):
            raise ValueError("relay sampling must be sgs or uniform")
        self.device = torch.device(device)
        self.capacity = capacity
        self.num_bank_configs = int(num_bank_configs)
        self.fraction_values = [float(value) for value in fractions]
        self.fractions = torch.as_tensor(self.fraction_values, dtype=torch.float32, device=self.device)
        self.alpha_max = float(alpha_max)
        self.sampling = sampling
        self.max_chains_per_config = int(max_chains_per_config)
        self.master_rate = float(master_rate)
        self.master_count = int(master_count)
        self.useless_count = int(useless_count)
        self.target_success_rate = float(target_success_rate)
        self.concentration = float(concentration)
        self.epsilon = float(epsilon)
        self.temperature = float(temperature)
        self.storage = {
            name: torch.zeros((capacity, *shape), dtype=dtype, device=self.device)
            for name, (shape, dtype) in field_specs.items()}
        long = dict(dtype=torch.long, device=self.device)
        self.active = torch.zeros(capacity, dtype=torch.bool, device=self.device)
        self.uid = torch.full((capacity,), -1, **long)
        self.source_config = torch.full((capacity,), -1, **long)
        self.chain_slot = torch.full((capacity,), -1, **long)
        self.fraction_index = torch.full((capacity,), -1, **long)
        self.created_step = torch.full((capacity,), -1, **long)
        # Outcome history per slot; Laplace estimates are derived from its sums.
        self.stats = SGSSampler(capacity, history_size=history_size,
                                target_success_rate=target_success_rate,
                                concentration=concentration, temperature=temperature,
                                epsilon=epsilon, p_init=0.5, device=self.device,
                                seed=seed, mode="sgs")
        # A chain always has at least one active entry, so C chain slots suffice.
        self.chain_source = torch.full((capacity,), -1, **long)
        self.chain_alive = torch.zeros(capacity, **long)
        self.config_chains = torch.zeros(self.num_bank_configs, **long)
        self._next_uid = 0
        self.counters = {name: torch.zeros((), **long) for name in (
            "created_chains", "created_entries", "episodes", "successes",
            "front_mastered") + self.REASONS}
        # Exponentially decayed "recent" statistics for logging (H5 curves).
        floats = dict(dtype=torch.float32, device=self.device)
        self.decay = 0.995
        self.fraction_episodes = torch.zeros(self.fractions.numel(), **floats)
        self.fraction_successes = torch.zeros(self.fractions.numel(), **floats)
        self.sampled_fraction_sum = torch.zeros((), **floats)
        self.sampled_count = torch.zeros((), **floats)
        self.leader_resets_relay = torch.zeros((), **floats)
        self.leader_resets_total = torch.zeros((), **floats)
        self.last_mass = torch.zeros((), **floats)
        self.last_union_entropy = torch.zeros((), **floats)

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def laplace_rates(self):
        return ((self.stats.success_sum + 1).to(torch.float32)
                / (self.stats.count + 2).to(torch.float32))

    @torch.no_grad()
    def logits(self):
        logits = sgs_logits(self.laplace_rates(), self.target_success_rate,
                            self.concentration, self.epsilon)
        return torch.where(self.active, logits, torch.full_like(logits, -math.inf))

    @torch.no_grad()
    def leader_distribution(self, bank_logits, uniform=False):
        """Leader q_L over B followed by R (union index space).

        One SGS softmax over B and R; if R's mass exceeds alpha_max, rescale B and
        R so R has exactly alpha_max. ``sampling=uniform`` keeps that mass but
        spreads it evenly over active entries (ablation: relay without SGS).
        """
        if uniform:
            bank_logits = torch.zeros_like(bank_logits)
            relay_logits = torch.where(self.active, torch.zeros_like(self.laplace_rates()),
                                       torch.full_like(self.laplace_rates(), -math.inf))
        else:
            relay_logits = self.logits()
        temperature = max(self.temperature, 1.0)
        probabilities = torch.softmax(torch.cat((bank_logits, relay_logits)) / temperature, dim=0)
        nb = bank_logits.numel()
        bank, relay = probabilities[:nb], probabilities[nb:]
        mass = relay.sum()
        over = mass > self.alpha_max
        relay = torch.where(over, relay * (self.alpha_max / mass.clamp_min(1e-12)), relay)
        bank = torch.where(over, bank * ((1.0 - self.alpha_max) / (1.0 - mass).clamp_min(1e-12)), bank)
        if self.sampling == "uniform":
            relay_mass = relay.sum()
            relay = torch.where(self.active,
                                relay_mass / self.active.sum().clamp_min(1).to(relay.dtype),
                                torch.zeros_like(relay))
        result = torch.cat((bank, relay))
        self.last_mass = relay.sum()
        self.last_union_entropy = entropy_of(result)
        return result

    @torch.no_grad()
    def batch(self, slots):
        return {name: value[slots] for name, value in self.storage.items()}

    @torch.no_grad()
    def record_sampled(self, slots, num_leader_resets):
        """Decayed share of leader resets from R and mean fraction f (H5 curve)."""
        self.leader_resets_relay.mul_(self.decay).add_(float(slots.numel()))
        self.leader_resets_total.mul_(self.decay).add_(float(num_leader_resets))
        if slots.numel():
            fractions = self.fractions[self.fraction_index[slots].clamp_min(0)]
            self.sampled_fraction_sum.mul_(self.decay).add_(fractions.sum())
            self.sampled_count.mul_(self.decay).add_(float(slots.numel()))

    # --------------------------------------------------------------- bookkeeping
    @torch.no_grad()
    def _retire(self, slots, reason):
        if slots.numel() == 0:
            return
        slots = torch.unique(slots)
        slots = slots[self.active[slots]]
        if slots.numel() == 0:
            return
        self.active[slots] = False
        self.uid[slots] = -1
        chains = self.chain_slot[slots]
        self.chain_alive.scatter_add_(0, chains, -torch.ones_like(chains))
        chains = torch.unique(chains)
        dead = chains[self.chain_alive[chains] == 0]
        if dead.numel():
            sources = self.chain_source[dead]
            self.config_chains.scatter_add_(0, sources, -torch.ones_like(sources))
            self.chain_source[dead] = -1
        self.counters[reason] += slots.numel()
        if reason == "mastered":
            self.counters["front_mastered"] += (self.fraction_index[slots] == 0).sum()
        self.stats.reset_entries(slots)

    @torch.no_grad()
    def chain_allowance(self, configs):
        return self.max_chains_per_config - self.config_chains[configs]

    @torch.no_grad()
    def insert(self, states, source_configs, chain_local, fraction_index, step):
        """Add entries; ``chain_local`` numbers the new chains 0..K-1 (contiguous)."""
        count = source_configs.numel()
        if count == 0:
            return None
        if count > self.capacity:
            raise ValueError("relay insertion exceeds the bank capacity")
        num_chains = int(chain_local.max().item()) + 1
        free = (~self.active).nonzero(as_tuple=False).flatten()
        if free.numel() >= count:
            slots = free[:count]
        else:
            # Evict the active entries the leader's SGS currently values least.
            score = torch.where(self.active, self.logits(),
                                torch.full_like(self.laplace_rates(), math.inf))
            evict = torch.topk(score, count - free.numel(), largest=False).indices
            self._retire(evict, "evicted")
            slots = torch.cat((free, evict))
        free_chains = (self.chain_alive == 0).nonzero(as_tuple=False).flatten()[:num_chains]
        if free_chains.numel() < num_chains:
            raise RuntimeError("relay chain table exhausted; this indicates a bookkeeping bug")
        chains = free_chains[chain_local]
        chain_sources = torch.zeros(num_chains, dtype=torch.long, device=self.device)
        chain_sources.scatter_(0, chain_local, source_configs)
        self.chain_source[free_chains] = chain_sources
        self.chain_alive.scatter_add_(0, chains, torch.ones_like(chains))
        self.config_chains.scatter_add_(0, chain_sources, torch.ones_like(chain_sources))
        for name, value in states.items():
            self.storage[name][slots] = value.to(self.storage[name].dtype)
        self.active[slots] = True
        self.uid[slots] = torch.arange(self._next_uid, self._next_uid + count, device=self.device)
        self._next_uid += count
        self.source_config[slots] = source_configs
        self.chain_slot[slots] = chains
        self.fraction_index[slots] = fraction_index
        self.created_step[slots] = int(step)
        self.stats.reset_entries(slots)
        self.counters["created_chains"] += num_chains
        self.counters["created_entries"] += count
        return slots

    @torch.no_grad()
    def observe(self, slots, uids, successes, quick_failure, step):
        """Credit leader outcomes of relay starts; quarantine and retire entries."""
        if slots.numel() == 0:
            return
        valid = self.active[slots] & (self.uid[slots] == uids)
        quarantine = valid & quick_failure
        if bool(quarantine.any()):
            self._retire(slots[quarantine], "quarantined")
        keep = valid & ~quick_failure
        observed, outcomes = slots[keep], successes[keep]
        if observed.numel() == 0:
            return
        self.stats.observe(observed, outcomes, step=int(step))
        self.counters["episodes"] += observed.numel()
        self.counters["successes"] += outcomes.long().sum()
        fractions = self.fraction_index[observed]
        self.fraction_episodes.mul_(self.decay).scatter_add_(
            0, fractions, torch.ones_like(fractions, dtype=torch.float32))
        self.fraction_successes.mul_(self.decay).scatter_add_(0, fractions, outcomes.float())
        unique = torch.unique(observed)
        count = self.stats.count[unique]
        wins = self.stats.success_sum[unique]
        mastered = (count >= self.master_count) & (wins.float() >= self.master_rate * count.float())
        useless = (count >= self.useless_count) & (wins == 0) & ~mastered
        self._retire(unique[mastered], "mastered")
        self._retire(unique[useless], "useless")

    @torch.no_grad()
    def metrics(self):
        active = self.active
        num_active = active.sum()
        rates = self.laplace_rates()
        result = {
            "Relay / active_entries": num_active,
            "Relay / active_chains": (self.chain_alive > 0).sum(),
            "Relay / configs_with_chains": (self.config_chains > 0).sum(),
            "Relay / laplace_rate_mean": (rates * active).sum() / num_active.clamp_min(1),
            "Relay / mass": self.last_mass,
            "Relay / union_entropy": self.last_union_entropy,
            "Relay / leader_reset_share": self.leader_resets_relay / self.leader_resets_total.clamp_min(1e-6),
            "Relay / sampled_fraction_mean": self.sampled_fraction_sum / self.sampled_count.clamp_min(1e-6),
            "Relay / success_rate_total": (self.counters["successes"].float()
                                           / self.counters["episodes"].clamp_min(1).float()),
        }
        for name, value in self.counters.items():
            result["Relay / " + name] = value
        for index, fraction in enumerate(self.fraction_values):
            result["Relay / success_f{:.2f}".format(fraction)] = (
                self.fraction_successes[index] / self.fraction_episodes[index].clamp_min(1e-6))
        return result

    def diagnostics(self):
        return {
            "active": self.active.cpu(), "uid": self.uid.cpu(),
            "source_config": self.source_config.cpu(), "chain_slot": self.chain_slot.cpu(),
            "fraction_index": self.fraction_index.cpu(), "fractions": self.fractions.cpu(),
            "created_step": self.created_step.cpu(),
            "history_count": self.stats.count.cpu(), "success_sum": self.stats.success_sum.cpu(),
            "visits_total": self.stats.visits_total.cpu(),
            "laplace_rate": self.laplace_rates().cpu(),
            "config_chains": self.config_chains.cpu(),
            "counters": {name: int(value) for name, value in self.counters.items()},
        }


class ISOverlapTracker:
    """Per-configuration Kish ESS of follower samples under the leader.

    For a configuration k, ESS_k = (E[mu])^2 / E[mu^2] with mu = pi_L / pi_follower
    on follower transitions of episodes started at k. Moments are exponential
    moving averages over PPO updates with at least ``min_samples`` samples.
    """

    def __init__(self, num_configs, device, ema=0.2, min_samples=32):
        self.device = torch.device(device)
        self.num_configs = int(num_configs)
        self.ema = float(ema)
        self.min_samples = int(min_samples)
        floats = dict(dtype=torch.float32, device=self.device)
        self.m1 = torch.zeros(self.num_configs, **floats)
        self.m2 = torch.zeros(self.num_configs, **floats)
        self.samples = torch.zeros(self.num_configs, **floats)
        self.valid = torch.zeros(self.num_configs, dtype=torch.bool, device=self.device)

    @torch.no_grad()
    def update(self, config_ids, log_mu):
        config_ids = config_ids.reshape(-1).long()
        log_mu = log_mu.reshape(-1).float().clamp(-20.0, 20.0)
        keep = (config_ids >= 0) & (config_ids < self.num_configs)
        config_ids, mu = config_ids[keep], log_mu[keep].exp()
        count = torch.zeros(self.num_configs, dtype=torch.float32, device=self.device)
        first = torch.zeros_like(count)
        second = torch.zeros_like(count)
        count.scatter_add_(0, config_ids, torch.ones_like(mu))
        first.scatter_add_(0, config_ids, mu)
        second.scatter_add_(0, config_ids, mu.square())
        ok = count >= self.min_samples
        mean1 = first / count.clamp_min(1)
        mean2 = second / count.clamp_min(1)
        blend = torch.where(self.valid, torch.full_like(count, self.ema), torch.ones_like(count))
        self.m1 = torch.where(ok, (1 - blend) * self.m1 + blend * mean1, self.m1)
        self.m2 = torch.where(ok, (1 - blend) * self.m2 + blend * mean2, self.m2)
        self.samples += count
        self.valid |= ok

    @torch.no_grad()
    def ess(self):
        value = self.m1.square() / self.m2.clamp_min(1e-30)
        return torch.where(self.valid, value.clamp(max=1.0), torch.full_like(value, math.nan))

    def diagnostics(self):
        return {"ess": self.ess().cpu(), "mu_mean": self.m1.cpu(), "mu_second_moment": self.m2.cpu(),
                "samples": self.samples.cpu(), "valid": self.valid.cpu(),
                "min_samples": self.min_samples, "ema": self.ema}
