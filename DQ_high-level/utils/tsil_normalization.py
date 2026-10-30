"""Normalization snapshots shared by online PPO and temporal self-imitation.

Statistics are learned exclusively from real interactions. Observation statistics
and the discounted-reward scale become active only after the corresponding PPO
update, so the old policy likelihoods retain their original preprocessing.
"""

import math

import torch

from skrl.resources.preprocessors.torch import RunningStandardScaler


class FrozenObservationScaler(RunningStandardScaler):
    """Running physical-observation scaler with two untouched temporal inputs."""

    def __init__(self, size, time_indices=(-191, -190), **kwargs):
        super().__init__(size=size, **kwargs)
        width = self.running_mean.numel()
        indices = [int(index) % width for index in time_indices]
        if len(set(indices)) != len(indices):
            raise ValueError("TSIL time channels must be distinct")
        if any(not -width <= int(index) < width for index in time_indices):
            raise ValueError("TSIL time channel lies outside the observation")
        self.register_buffer("time_indices", torch.tensor(indices, dtype=torch.long, device=self.device))
        self.register_buffer("pending_mean", torch.zeros_like(self.running_mean))
        self.register_buffer("pending_m2", torch.zeros_like(self.running_mean))
        self.register_buffer("pending_count", torch.zeros_like(self.current_count))

    @torch.no_grad()
    def observe(self, states):
        """Accumulate online samples without changing the active transform."""
        flat = states.detach().reshape(-1, self.running_mean.numel()).double()
        if not flat.shape[0]:
            return
        mean = flat.mean(0)
        variance = flat.var(0, unbiased=False)
        count = flat.shape[0]
        total = self.pending_count + count
        delta = mean - self.pending_mean
        self.pending_m2.add_(variance * count + delta.square() * self.pending_count * count / total)
        self.pending_mean.add_(delta * count / total)
        self.pending_count.copy_(total)

    @torch.no_grad()
    def commit(self):
        if self.pending_count.item() == 0:
            return
        self._parallel_variance(self.pending_mean, self.pending_m2 / self.pending_count,
                                self.pending_count)
        # Keep the checkpoint interpretation of these two columns explicit, too.
        self.running_mean[self.time_indices] = 0
        self.running_variance[self.time_indices] = 1
        self.pending_count.zero_()
        self.pending_mean.zero_()
        self.pending_m2.zero_()

    def _compute(self, x, train=False, inverse=False):
        if train:
            raise ValueError("TSIL observations use observe()/commit(); optimization cannot update RMS")
        output = super()._compute(x, train=False, inverse=inverse)
        output[..., self.time_indices] = x[..., self.time_indices]
        return output


class ValueScaler(RunningStandardScaler):
    """Affine value normalization with a statistics-invariant Critic output.

The network's raw output is mapped to the active normalized coordinates by
``normalize_prediction``. Updating return statistics compensates that mapping,
preserving the *denormalized* prediction without modifying network parameters
or Adam moments. The adapter is part of this module's checkpoint state.

Unlike the observation scaler, value normalization must be invertible: clipping
either direction would invalidate both the compensation and PPO's old values.
    """

    ADAPTER_STATE_KEYS = frozenset({"prediction_scale", "prediction_bias", "standard_deviation_floor"})

    def __init__(self, size, **kwargs):
        super().__init__(size=size, **kwargs)
        self.register_buffer("prediction_scale", torch.ones_like(self.running_mean))
        self.register_buffer("prediction_bias", torch.zeros_like(self.running_mean))
        self.register_buffer("standard_deviation_floor", torch.full_like(self.running_mean, self.epsilon))

    def _standard_deviation(self):
        # The identical sigma is used in both affine directions and compensation.
        return self.running_variance.sqrt().clamp_min(self.standard_deviation_floor)

    def _compute(self, x, train=False, inverse=False):
        if train:
            raise ValueError("TSIL values use update_returns(); optimization cannot update RMS")
        mean = self.running_mean.to(x)
        sigma = self._standard_deviation().to(x)
        return x * sigma + mean if inverse else (x - mean) / sigma

    def normalize_prediction(self, network_output):
        """Differentiable network-output adapter; never use forward's no_grad."""
        return (network_output * self.prediction_scale.to(network_output)
                + self.prediction_bias.to(network_output))

    @torch.no_grad()
    def update_returns(self, returns):
        flat = returns.detach().reshape(-1, self.running_mean.numel()).double()
        if flat.shape[0]:
            old_mean = self.running_mean.clone()
            old_sigma = self._standard_deviation().clone()
            self._parallel_variance(flat.mean(0), flat.var(0, unbiased=False), flat.shape[0])
            sigma = self._standard_deviation()
            self.prediction_scale.mul_(old_sigma / sigma)
            self.prediction_bias.copy_((old_sigma * self.prediction_bias
                                        + old_mean - self.running_mean) / sigma)

    @torch.no_grad()
    def rescale_units(self, ratio):
        """Change reward units, retaining the same normalized network output.

This is a physical unit conversion rather than a statistical update: both
targets and predictions acquire the same factor, so the adapter stays fixed.
        """
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("Value reward-unit conversion requires a finite positive ratio")
        self.running_mean.mul_(ratio)
        self.running_variance.mul_(ratio * ratio)
        # The numerical floor has value units too. A fixed epsilon would break
        # sigma_new == ratio * sigma_old near zero variance.
        self.standard_deviation_floor.mul_(ratio)


class RewardTraceNormalizer(torch.nn.Module):
    """Independent raw trace for each env, one shared task reward divisor.

Unlike the upstream TSIL per-task trace implementation, repeated task IDs never
overwrite one another. A terminal reward enters the statistics before its trace
is cleared. Replay has no write path into this normalizer.
    """

    def __init__(self, num_envs, gamma=.99, epsilon=1e-8, device="cpu"):
        super().__init__()
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.register_buffer("trace", torch.zeros(int(num_envs), dtype=torch.float64, device=device))
        self.register_buffer("running_mean", torch.zeros((), dtype=torch.float64, device=device))
        self.register_buffer("running_variance", torch.ones((), dtype=torch.float64, device=device))
        self.register_buffer("current_count", torch.zeros((), dtype=torch.float64, device=device))
        self.register_buffer("active_scale", torch.ones((), dtype=torch.float64, device=device))

    @torch.no_grad()
    def observe(self, raw_rewards, terminated, episode_start=None):
        rewards = raw_rewards.detach().reshape(-1).to(self.trace)
        if rewards.shape != self.trace.shape:
            raise ValueError("Reward trace requires one raw reward per environment")
        if episode_start is not None:
            self.trace[episode_start.reshape(-1).bool()] = 0
        self.trace.mul_(self.gamma).add_(rewards)
        count = self.trace.numel()
        mean = self.trace.mean()
        variance = self.trace.var(unbiased=False)
        total = self.current_count + count
        delta = mean - self.running_mean
        m2 = self.running_variance * self.current_count + variance * count
        m2 += delta.square() * self.current_count * count / total
        self.running_mean.add_(delta * count / total)
        self.running_variance.copy_(m2 / total)
        self.current_count.copy_(total)
        self.trace[terminated.reshape(-1).bool()] = 0

    def forward(self, rewards):
        return rewards / self.active_scale.to(dtype=rewards.dtype, device=rewards.device)

    @torch.no_grad()
    def commit(self):
        previous = self.active_scale.clone()
        if self.current_count.item() > 0:
            self.active_scale.copy_(self.running_variance.sqrt().clamp_min(self.epsilon))
        return (previous / self.active_scale).item()

    def load_state_dict(self, state_dict, strict=True):
        # Checkpoints are model-only; evaluation may have a different env count.
        # Live episode traces do not belong to freshly reset evaluation episodes.
        state_dict = dict(state_dict)
        state_dict["trace"] = torch.zeros_like(self.trace)
        return super().load_state_dict(state_dict, strict=strict)
