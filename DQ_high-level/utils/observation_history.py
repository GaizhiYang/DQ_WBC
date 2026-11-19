"""Raw teacher frames, stored oldest first and owned by the environment wrapper."""

import torch


class ObservationHistory:
    def __init__(self, frames):
        if type(frames) is not int or frames < 1:
            raise ValueError("actor_history_frames must be a positive integer")
        self.frames = frames
        self.buffer = None

    def update(self, observation, *, advance=False, reset_mask=None):
        if self.buffer is None:
            self.buffer = observation.unsqueeze(1).repeat(1, self.frames, 1)
        else:
            # record_transition gets states and next_states AFTER env.step.
            # Never mutate a tensor already returned to the learner.
            if advance:
                self.buffer = torch.cat((self.buffer[:, 1:], observation.unsqueeze(1)), dim=1)
            else:
                self.buffer = self.buffer.clone()
                self.buffer[:, -1] = observation
            if reset_mask is not None:
                reset_mask = reset_mask.to(device=observation.device, dtype=torch.bool)
                self.buffer[reset_mask] = observation[reset_mask].unsqueeze(1)
        return self.buffer.flatten(1)
