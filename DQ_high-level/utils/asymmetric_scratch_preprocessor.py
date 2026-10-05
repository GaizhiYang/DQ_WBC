"""Scratch-training normalization without changing PPO's sampled likelihoods."""

import torch
from . import asymmetric_distributed as distributed

from .teacher_vision_preprocessor import (
    FrozenRunningStandardScaler, PrefixRunningStandardScaler,
)


class RolloutPrefixRunningStandardScaler(PrefixRunningStandardScaler):
    """Update raw privileged statistics explicitly between complete rollouts.

    All ordinary forwards remain frozen, including SKRL's ``train=True``
    minibatch calls. The trainer calls ``update_rollout`` only after every PPO
    epoch has finished and before collecting the next rollout. Images, task
    state and belief retain their original units. Checkpoints retain the
    original three scaler buffers, including their one-sample initial prior.
    """

    @torch.no_grad()
    def update_rollout(self, observations):
        if observations.ndim not in (2, 3) or observations.shape[-1] != self.input_size:
            raise ValueError("Expected complete raw packed rollout observations")
        prefix = observations[..., :self.privileged_dim].reshape(-1, self.privileged_dim)
        if not len(prefix):
            raise ValueError("Cannot update normalization from an empty rollout")
        # Convert only the small vector prefix, never the camera history.
        prefix = prefix.to(device=self.running_mean.device, dtype=torch.float64)
        valid = torch.isfinite(prefix).all().to(dtype=torch.int32)
        if distributed.active():
            torch.distributed.all_reduce(valid, op=torch.distributed.ReduceOp.MIN)
        if not valid:
            raise ValueError("Cannot update normalization from non-finite observations")
        if distributed.active():
            mean, variance, count = distributed.moments(prefix)
        else:
            variance, mean = torch.var_mean(prefix, dim=0, unbiased=False)
            count = len(prefix)
        self._parallel_variance(mean, variance, count)

    def _compute(self, x, train=False, inverse=False):
        # Explicitly forbid SKRL's minibatch updates even if .freeze is changed.
        return super()._compute(x, train=False, inverse=inverse)


class IdentityValuePreprocessor(FrozenRunningStandardScaler):
    """Keep scratch Critic values and returns in reward units without clipping.

    The standard scaler buffers remain checkpoint-compatible, but normalization
    statistics from a scaled Critic cannot be silently loaded into this mode.
    """

    def _compute(self, x, train=False, inverse=False):
        return x

    def load_state_dict(self, state_dict, strict=True):
        mean = state_dict.get("running_mean")
        variance = state_dict.get("running_variance")
        if ((mean is not None and not torch.equal(mean, torch.zeros_like(mean)))
                or (variance is not None and not torch.equal(variance, torch.ones_like(variance)))):
            raise ValueError("Rollout normalization requires identity value statistics; retain frozen normalization when migrating a scaled Critic")
        return super().load_state_dict(state_dict, strict=strict)
