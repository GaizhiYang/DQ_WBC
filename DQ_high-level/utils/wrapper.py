# modified wrapper from skrl
from typing import Union, Tuple, Any, Optional

import gym
import gymnasium
import collections
import numpy as np
from packaging import version

import torch

from skrl import logger
from utils.observation_history import ObservationHistory

__all__ = ["wrap_env"]


class Wrapper(object):
    def __init__(self, env: Any) -> None:
        """Base wrapper class for RL environments

        :param env: The environment to wrap
        :type env: Any supported RL environment
        """
        self._env = env

        # device (faster than @property)
        if hasattr(self._env, "device"):
            self.device = torch.device(self._env.device)
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __getattr__(self, key: str) -> Any:
        """Get an attribute from the wrapped environment

        :param key: The attribute name
        :type key: str

        :raises AttributeError: If the attribute does not exist

        :return: The attribute value
        :rtype: Any
        """
        if hasattr(self._env, key):
            return getattr(self._env, key)
        raise AttributeError("Wrapped environment ({}) does not have attribute '{}'" \
            .format(self._env.__class__.__name__, key))

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        raise NotImplementedError

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raises NotImplementedError: Not implemented

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        raise NotImplementedError

    def render(self, *args, **kwargs) -> None:
        """Render the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    def close(self) -> None:
        """Close the environment

        :raises NotImplementedError: Not implemented
        """
        raise NotImplementedError

    @property
    def num_envs(self) -> int:
        """Number of environments

        If the wrapped environment does not have the ``num_envs`` property, it will be set to 1
        """
        return self._env.num_envs if hasattr(self._env, "num_envs") else 1

    @property
    def state_space(self) -> gym.Space:
        """State space

        If the wrapped environment does not have the ``state_space`` property,
        the value of the ``observation_space`` property will be used
        """
        return self._env.state_space if hasattr(self._env, "state_space") else self._env.observation_space

    @property
    def observation_space(self) -> gym.Space:
        """Observation space
        """
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Action space
        """
        return self._env.action_space


class IsaacGymPreview3Wrapper(Wrapper):
    def __init__(self, env: Any, actor_history_frames: int = 1) -> None:
        """Isaac Gym environment (preview 3) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 3) environment
        """
        super().__init__(env)

        self.actor_history_frames = actor_history_frames
        self._actor_history = ObservationHistory(actor_history_frames)
        space = env.observation_space
        self._history_observation_space = gym.spaces.Box(
            low=np.tile(space.low, actor_history_frames),
            high=np.tile(space.high, actor_history_frames), dtype=space.dtype)

        self._reset_once = True
        self._obs_dict = None

    @property
    def observation_space(self):
        return self._history_observation_space

    def _stack_actor_observations(self, advance=False, reset_mask=None):
        if self.actor_history_frames > 1:
            self._obs_dict = dict(self._obs_dict)
            self._obs_dict["obs"] = self._actor_history.update(
                self._obs_dict["obs"], advance=advance, reset_mask=reset_mask)

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._obs_dict, reward, terminated, info = self._env.step(actions)
        self._stack_actor_observations(advance=True)
        truncated = torch.zeros_like(terminated)
        if self.num_states:
            return self._obs_dict, reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info
        return self._obs_dict["obs"], reward.view(-1, 1), terminated.view(-1, 1), truncated.view(-1, 1), info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        # if self._reset_once:
        #     self._obs_dict = self._env.reset()
        #     self._reset_once = False
        # return self._obs_dict["obs"], {}
        reset_mask = self._env.reset_buf.bool().clone() if self.actor_history_frames > 1 else None
        self._obs_dict = self._env.reset()
        if reset_mask is not None:
            # Also handles a raw reset_idx() followed by wrapper.reset().
            reset_mask |= self._env.progress_buf == 0
        self._stack_actor_observations(reset_mask=reset_mask)
        if self.num_states:
            return self._obs_dict, {}
        return self._obs_dict["obs"], {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment
            change here
        """
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment
        """
        pass
    
class RSLWrapper(Wrapper):
    def __init__(self, env: Any) -> None:
        """Isaac Gym environment (preview 3) wrapper

        :param env: The environment to wrap
        :type env: Any supported Isaac Gym environment (preview 3) environment
        """
        super().__init__(env)

        self._reset_once = True
        self._obs_dict = None

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        self._obs_dict, reward, terminated, info = self._env.step(actions)
        truncated = torch.zeros_like(terminated)
        if self.num_states:
            return self._obs_dict, reward, terminated, truncated, info
        return self._obs_dict["obs"], reward, terminated, truncated, info

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        self._obs_dict = self._env.reset()
        if self.num_states:
            return self._obs_dict, {}
        return self._obs_dict["obs"], {}

    def render(self, *args, **kwargs) -> None:
        """Render the environment
            change here
        """
        return self._env.render(*args, **kwargs)

    def close(self) -> None:
        """Close the environment
        """
        pass
