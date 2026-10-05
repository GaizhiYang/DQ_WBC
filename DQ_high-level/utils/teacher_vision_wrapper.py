"""Expose privileged teacher observations together with the existing camera history."""

from collections.abc import Mapping

import gym
import numpy as np
import torch

from .wrapper import IsaacGymPreview3Wrapper


class TeacherVisionWrapper(IsaacGymPreview3Wrapper):
    """Pack ``obs`` and the image prefix of ``states`` into one fresh tensor.

    The environment's student ``states`` contains image history followed by 61
    proprioceptive values. Those values already occur in the privileged teacher
    observation and are deliberately not copied a second time. Episode reset,
    action handling, rewards, and termination retain the original wrapper's
    semantics, including its all-zero ``truncated`` output.
    """

    def __init__(self, env, privileged_dim=1276, image_dim=62208):
        super().__init__(env)
        self.privileged_dim = int(privileged_dim)
        self.image_dim = int(image_dim)
        self.student_state_dim = self.image_dim + 61
        if self.privileged_dim <= 0 or self.image_dim <= 0:
            raise ValueError("privileged_dim and image_dim must be positive")
        if tuple(env.observation_space.shape) != (self.privileged_dim,):
            raise ValueError(
                "TeacherVisionWrapper requires the original privileged observation "
                f"space ({self.privileged_dim},), got {env.observation_space.shape}"
            )
        if getattr(env, "num_states", 0) != self.student_state_dim:
            raise ValueError(
                "TeacherVisionWrapper requires full camera observations with "
                f"num_states={self.student_state_dim} (images plus 61 state values); "
                f"got {getattr(env, 'num_states', 0)}"
            )
        self._teacher_vision_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.privileged_dim + self.image_dim,),
            dtype=np.float32,
        )

    @property
    def observation_space(self):
        return self._teacher_vision_observation_space

    def _pack_observation(self, observations):
        if not isinstance(observations, Mapping):
            raise TypeError("Expected an environment observation mapping with obs and states")
        if "obs" not in observations or "states" not in observations:
            raise ValueError("Full camera observations must contain both obs and states")
        privileged = observations["obs"]
        student = observations["states"]
        if not isinstance(privileged, torch.Tensor) or not isinstance(student, torch.Tensor):
            raise TypeError("obs and states must be torch tensors")
        expected_privileged = (self.num_envs, self.privileged_dim)
        expected_student = (self.num_envs, self.student_state_dim)
        if tuple(privileged.shape) != expected_privileged:
            raise ValueError(f"Expected obs shape {expected_privileged}, got {tuple(privileged.shape)}")
        if tuple(student.shape) != expected_student:
            raise ValueError(f"Expected states shape {expected_student}, got {tuple(student.shape)}")
        if privileged.device != student.device or privileged.dtype != student.dtype:
            raise ValueError("obs and states must have the same device and dtype")
        # torch.cat allocates fresh storage. The simulator reuses observation
        # buffers in place, so returning a view would corrupt previous states.
        return torch.cat((privileged, student[:, :self.image_dim]), dim=-1)

    def step(self, actions):
        observations, reward, terminated, truncated, info = super().step(actions)
        return self._pack_observation(observations), reward, terminated, truncated, info

    def reset(self):
        observations, info = super().reset()
        return self._pack_observation(observations), info
