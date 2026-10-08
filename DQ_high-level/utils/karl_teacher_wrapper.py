"""Select on physical observations BEFORE RunningStandardScaler and PPO storage."""
import gym
import numpy as np
import torch

from modules.karl_grasp_selector import KarlGraspSelector
from .wrapper import IsaacGymPreview3Wrapper


class KarlTeacherWrapper(IsaacGymPreview3Wrapper):
    NUM_FEATURES = 1024
    NUM_CANDIDATES = 30
    NUM_ACTIONS = 9

    def __init__(self, env, switch_margin_deg=30.0, orientation_preference="none",
                 visualize_grasp=False, grasp_vis_envs=8, selector=None, selector_name="karl"):
        super().__init__(env)
        self.raw_dim = env.observation_space.shape[0]
        if env.action_space.shape != (9,) or self.raw_dim != 1276:
            raise ValueError("Single-grasp teacher requires original 30-grasp/1024-feature observations, 9 actions and --roboinfo")
        if getattr(env, "num_states", 0):
            raise ValueError("Single-grasp teacher comparison requires sensor.enableCamera: false")
        self._packed_space = gym.spaces.Box(-np.inf, np.inf, (self.raw_dim - 180 + 6,), np.float32)
        self.selector = selector if selector is not None else KarlGraspSelector(
            env.num_envs, env.rl_device, switch_margin_deg, orientation_preference)
        self._selection_started = False
        self._grasp_visualizer = None
        if visualize_grasp and getattr(env, "viewer", None) is not None:
            from .karl_grasp_visualization import KarlGraspVisualizer
            self._grasp_visualizer = KarlGraspVisualizer(env, grasp_vis_envs, selector_name)
            env._karl_grasp_visualizer = self._grasp_visualizer

    @property
    def observation_space(self):
        return self._packed_space

    @property
    def state_space(self):
        return self._packed_space

    @torch.no_grad()
    def _select(self, poses, ee_pose):
        return self.selector.select(poses, ee_pose)

    @torch.no_grad()
    def _pack(self, raw):
        if raw.shape != (self.num_envs, self.raw_dim):
            raise ValueError("Unexpected raw teacher observation shape")
        poses = torch.cat((raw[:, -189:-99].reshape(-1, 30, 3),
                           raw[:, -99:-9].reshape(-1, 30, 3)), -1)
        selected, metrics = self._select(poses, raw[:, 1030:1036])
        if self._grasp_visualizer is not None:
            self._grasp_visualizer.update(selected, metrics)
        # Fresh storage: Isaac Gym reuses raw observation buffers in-place.
        packed = torch.cat((raw[:, :-189], selected, raw[:, -9:]), -1)
        return packed, metrics

    def step(self, actions):
        raw, reward, terminated, truncated, info = super().step(actions)
        packed, metrics = self._pack(raw)
        # B1Z1Base returns TERMINAL observations here. Do not reset selection
        # until reset() actually resets these environments below.
        return packed, reward, terminated, truncated, dict(info, **metrics)

    def reset(self):
        if self._selection_started:
            reset_ids = self._env.reset_buf.nonzero(as_tuple=False).flatten().to(self.selector.indices.device)
            self.selector.reset(reset_ids)
            if self._grasp_visualizer is not None:
                self._grasp_visualizer.reset(reset_ids)
        else:
            self.selector.reset()
            if self._grasp_visualizer is not None:
                self._grasp_visualizer.reset()
            self._selection_started = True
        raw, info = super().reset()
        packed, metrics = self._pack(raw)
        return packed, dict(info, **metrics)
