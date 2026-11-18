"""State scaler that leaves one observation slice unscaled (direction 8 grasp tokens).

skrl's RunningStandardScaler keeps separate statistics for every dimension. For
the grasp tokens that would give each of the 90 tokens its own offset and
scale, so the shared Key/Value projection would see the same geometry as
different inputs, and the binary ``future_active`` flag would be rescaled.
Token features are already O(1) (metres, rotation-matrix entries, radians),
so they reach the model raw. Statistics are still tracked for every dimension,
which keeps the checkpoint buffers identical to the parent class.
"""

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


def teacher_state_preprocessor(observation_space, action_space, device, token_layout=None):
    """(class, kwargs) of the teacher state scaler; the grasp-token block, just before
    the last action, stays unscaled when tokens are enabled."""
    kwargs = {"size": observation_space, "device": device}
    if token_layout is None:
        return RunningStandardScaler, kwargs
    stop = observation_space.shape[0] - action_space.shape[0]
    kwargs["passthrough"] = (stop - token_layout.width, stop)
    return PassthroughStandardScaler, kwargs
