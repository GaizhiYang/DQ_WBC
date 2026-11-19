"""Teacher state scalers for the grasp-token block (direction 8).

skrl's RunningStandardScaler keeps separate statistics for every dimension. For
the grasp tokens that would give each token its own offset and scale, so the
shared Key/Value projection would see the same geometry as different inputs,
and the binary ``future_active`` flag would be rescaled. Two alternatives:

- ``raw`` (PassthroughStandardScaler): the token block reaches the model
  unscaled. The first G1/G2 runs used it; their critics diverged.
- ``shared`` (TokenStandardScaler): every token feature (x, y, z, rotation
  entries, distance, bearing) is standardized with one mean/std pooled over all
  tokens, so tokens stay comparable; the flag stays raw.

Statistics are still tracked for every dimension, so both keep the parent's buffers.
"""

import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler


class PassthroughStandardScaler(RunningStandardScaler):
    def __init__(self, size, passthrough, epsilon=1e-8, clip_threshold=5.0, device=None):
        super().__init__(size, epsilon=epsilon, clip_threshold=clip_threshold, device=device)
        start, stop = passthrough
        if not 0 <= start < stop <= int(self.running_mean.numel()):
            raise ValueError("passthrough must be a non-empty (start, stop) slice inside the state")
        self.passthrough = slice(int(start), int(stop))

    def _compute(self, x, train=False, inverse=False):
        out = super()._compute(x, train, inverse)
        out[..., self.passthrough] = x[..., self.passthrough]
        return out


class TokenStandardScaler(RunningStandardScaler):
    """Standardize the token features with statistics shared by all tokens.

    ``token_block`` is the whole block: ``num_tokens * feature_dim`` features followed
    by the raw ``future_active`` flag. Other dimensions are scaled as usual.
    """

    def __init__(self, size, token_block, feature_dim, epsilon=1e-8, clip_threshold=5.0, device=None):
        super().__init__(size, epsilon=epsilon, clip_threshold=clip_threshold, device=device)
        start, stop = (int(index) for index in token_block)
        if not 0 <= start < stop <= int(self.running_mean.numel()):
            raise ValueError("token_block must be a non-empty (start, stop) slice inside the state")
        if feature_dim <= 0 or (stop - start - 1) <= 0 or (stop - start - 1) % feature_dim:
            raise ValueError("token_block must hold whole tokens followed by one flag")
        self.token_features = slice(start, stop - 1)
        self.flag_index = stop - 1
        self.feature_dim = int(feature_dim)
        dtype, device = self.running_mean.dtype, self.running_mean.device
        self.register_buffer("token_mean", torch.zeros(self.feature_dim, dtype=dtype, device=device))
        self.register_buffer("token_variance", torch.ones(self.feature_dim, dtype=dtype, device=device))
        self.register_buffer("token_count", torch.ones((), dtype=dtype, device=device))

    def _update_token_statistics(self, features):
        """Same parallel update as the parent, over [samples * tokens, feature_dim]."""
        count = features.shape[0]
        mean, variance = torch.mean(features, dim=0), torch.var(features, dim=0)
        delta = mean - self.token_mean
        total = self.token_count + count
        m2 = (self.token_variance * self.token_count + variance * count
              + delta ** 2 * self.token_count * count / total)
        self.token_mean = self.token_mean + delta * count / total
        self.token_variance = m2 / total
        self.token_count = total

    def _compute(self, x, train=False, inverse=False):
        out = super()._compute(x, train, inverse)
        block = x[..., self.token_features]
        tokens = block.reshape(*block.shape[:-1], -1, self.feature_dim)
        if train:
            self._update_token_statistics(tokens.reshape(-1, self.feature_dim))
        mean = self.token_mean.float()
        std = torch.sqrt(self.token_variance.float())
        if inverse:
            scaled = std * torch.clamp(tokens, -self.clip_threshold, self.clip_threshold) + mean
        else:
            scaled = torch.clamp((tokens - mean) / (std + self.epsilon), -self.clip_threshold, self.clip_threshold)
        out[..., self.token_features] = scaled.reshape(block.shape)
        out[..., self.flag_index] = x[..., self.flag_index]
        return out


def teacher_state_preprocessor(observation_space, action_space, device, token_layout=None):
    """(class, kwargs) of the teacher state scaler. The grasp-token block sits just
    before the last action; its scaling follows ``token_layout.normalization``."""
    kwargs = {"size": observation_space, "device": device}
    if token_layout is None:
        return RunningStandardScaler, kwargs
    stop = observation_space.shape[0] - action_space.shape[0]
    block = (stop - token_layout.width, stop)
    if token_layout.normalization == "shared":
        kwargs.update(token_block=block, feature_dim=token_layout.feature_dim)
        return TokenStandardScaler, kwargs
    kwargs["passthrough"] = block
    return PassthroughStandardScaler, kwargs
