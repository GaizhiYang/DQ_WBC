"""DQ model state, learned observation transition and geometric imagination reward."""
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F

from .state import PhysicalStateLayout


class DQRobotMDP:
    """Model the measured physical state, end effector and complete policy observation.

    The world model predicts a delta for every field, including the environment's
    observation vector. Imagination therefore uses the same observation layout as
    real collection; it does not replace features/history with padded proxy data.
    Reward and termination here are geometric proxies, not the environment's
    contact dynamics, consecutive-hold success condition or reported success rate.
    """

    def __init__(self, env, device: str = "cuda:0"):
        self.env = env
        self.device = torch.device(device)
        self.num_envs = env.num_envs
        self.physical_dim = PhysicalStateLayout.total_dim
        self.task_state_dim = self.physical_dim + 7
        self.action_dim = env.num_actions
        self.obs_dim = env.num_obs
        self.context_dim = 16
        self.state_dim = self.task_state_dim + self.obs_dim
        self.ee_position_slice = slice(self.physical_dim, self.physical_dim + 3)
        self.ee_quaternion_slice = slice(self.physical_dim + 3, self.task_state_dim)
        self.observation_slice = slice(self.task_state_dim, self.state_dim)
        self.wm_input_dim = self.state_dim + self.action_dim + self.context_dim
        self.delta_dim = self.state_dim
        # The critic receives measured task fields and derived geometry; avoid
        # duplicating the potentially large feature/history observation vector.
        self.critic_dim = self.task_state_dim + self.context_dim + 6

        self.w_distance = 1.0
        self.w_approach = 0.5
        self.w_alignment = 0.3
        self.w_grasp = 2.0
        self.w_height = 1.5
        self.grasp_force_threshold = 0.5  # Normalized measured closure, not a force measurement.
        self._capture_gripper_limits()
        self.state_mean = torch.zeros(self.state_dim, device=self.device)
        self.state_std = torch.ones(self.state_dim, device=self.device)
        self.action_mean = torch.zeros(self.action_dim, device=self.device)
        self.action_std = torch.ones(self.action_dim, device=self.device)

    def _capture_gripper_limits(self):
        # B1Z1Base.set_gripper sends positive commands to the lower joint limits
        # (open) and negative commands to the upper limits (closed). StateCapture
        # stores the measured mean of these same gripper joints at physical[22].
        joints = getattr(self.env, 'num_gripper_joints', getattr(self.env, 'num_gripper_dof', None))
        if joints is None and hasattr(self.env, 'dof_pos_gripper'):
            joints = self.env.dof_pos_gripper.shape[-1]
        if joints is None or joints <= 0:
            raise AttributeError("DQRobotMDP requires the measured gripper joint count")
        if not all(hasattr(self.env, name) for name in ('dof_limits_lower', 'dof_limits_upper')):
            raise AttributeError("DQRobotMDP requires actual gripper joint limits, not assumed position ranges")
        lower = torch.as_tensor(self.env.dof_limits_lower, device=self.device, dtype=torch.float32)
        upper = torch.as_tensor(self.env.dof_limits_upper, device=self.device, dtype=torch.float32)
        if lower.ndim != 1 or upper.shape != lower.shape or lower.numel() < joints:
            raise ValueError("Gripper limits must be one-dimensional DOF arrays containing the gripper joints")
        self.gripper_open = lower[-joints:].mean().detach().clone()
        self.gripper_closed = upper[-joints:].mean().detach().clone()
        if not bool(torch.isfinite(self.gripper_open) & torch.isfinite(self.gripper_closed)
                    & (self.gripper_closed > self.gripper_open)):
            raise ValueError("The measured gripper mean must have distinct finite open/closed limits")

    @staticmethod
    def _require_shape(tensor, expected, name):
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != tuple(expected):
            actual = tuple(tensor.shape) if isinstance(tensor, torch.Tensor) else type(tensor).__name__
            raise ValueError(f"{name} must have shape {tuple(expected)}, got {actual}")

    def _check_state(self, state, name="state"):
        if not isinstance(state, torch.Tensor) or state.ndim != 2:
            raise ValueError(f"{name} must be a two-dimensional model state tensor")
        self._require_shape(state, (state.shape[0], self.state_dim), name)
        return state.shape[0]

    @torch.no_grad()
    def capture_state(self, physical_state: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Snapshot [physical34, actual EE position/XYZW quaternion, actual policy obs]."""
        self._require_shape(physical_state, (self.num_envs, self.physical_dim), "physical_state")
        self._require_shape(obs, (self.num_envs, self.obs_dim), "policy observation")
        if not hasattr(self.env, 'ee_pos') or not hasattr(self.env, 'ee_orn'):
            raise AttributeError("Model state capture requires measured env.ee_pos and env.ee_orn")
        self._require_shape(self.env.ee_pos, (self.num_envs, 3), "env.ee_pos")
        self._require_shape(self.env.ee_orn, (self.num_envs, 4), "env.ee_orn (XYZW)")
        fields = [physical_state, self.env.ee_pos, self.env.ee_orn, obs]
        return torch.cat([field.detach().to(device=self.device, dtype=torch.float32)
                          for field in fields], dim=-1)

    def inputs(self, state: torch.Tensor, action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch = self._check_state(state)
        self._require_shape(action, (batch, self.action_dim), "action")
        self._require_shape(context, (batch, self.context_dim), "context")
        return torch.cat(((state - self.state_mean) / (self.state_std + 1e-8),
                          (action - self.action_mean) / (self.action_std + 1e-8), context), dim=-1)

    def targets(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        batch = self._check_state(state)
        self._require_shape(next_state, (batch, self.state_dim), "next_state")
        return next_state - state

    def state_error(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._require_shape(predicted, (target.shape[0], self.delta_dim), "predicted delta")
        self._require_shape(target, predicted.shape, "target delta")
        return predicted - target

    def critic_features(self, state: torch.Tensor, context: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        batch = self._check_state(state)
        self._require_shape(context, (batch, self.context_dim), "context")
        distance = torch.linalg.vector_norm(state[:, 7:10] - state[:, self.ee_position_slice], dim=-1, keepdim=True)
        task_features = torch.cat((
            distance,
            state[:, 9:10] - context[:, 11:12],
            torch.linalg.vector_norm(state[:, 16:19], dim=-1, keepdim=True),
            torch.linalg.vector_norm(state[:, 19:22], dim=-1, keepdim=True),
            self._gripper_closure(state).unsqueeze(-1),
            torch.ones_like(distance),
        ), dim=-1)
        return torch.cat((state[:, :self.task_state_dim], context, task_features), dim=-1)

    def observe(self, state: torch.Tensor, context: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        """Return the complete learned observation slice with its gradient intact."""
        self._check_state(state)
        return state[:, self.observation_slice]

    def advance_obs(self, obs: torch.Tensor, next_state: torch.Tensor,
                    action: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """The world model predicts the actual next policy observation as part of state."""
        batch = self._check_state(next_state, "next_state")
        self._require_shape(obs, (batch, self.obs_dim), "previous policy observation")
        return self.observe(next_state, context, action)

    def _gripper_closure(self, state: torch.Tensor) -> torch.Tensor:
        return ((state[:, 22] - self.gripper_open) / (self.gripper_closed - self.gripper_open)).clamp(0, 1)

    def _get_ee_forward_axis(self, state: torch.Tensor) -> torch.Tensor:
        # Isaac Gym quaternions are XYZW. Native DQ _reward_ee_orn points the
        # gripper's local +X axis toward the object (not a fixed world -Z axis).
        q = F.normalize(state[:, self.ee_quaternion_slice], dim=-1, eps=1e-8)
        x, y, z, w = q.unbind(dim=-1)
        return torch.stack((1 - 2 * (y.square() + z.square()),
                            2 * (x * y + w * z), 2 * (x * z - w * y)), dim=-1)

    def reward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor,
               context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable approach/alignment/grasp/lift shaping, with proxy termination.

        Proximity is a smooth gate; closure uses measured joint limits. Episode
        target height/proximity come from context, including SGS reset geometry.
        Proxy completion does not assert contact or the native hold requirement.
        """
        batch = self._check_state(state)
        self._require_shape(next_state, (batch, self.state_dim), "next_state")
        self._require_shape(action, (batch, self.action_dim), "action")
        self._require_shape(context, (batch, self.context_dim), "context")
        direction = next_state[:, 7:10] - next_state[:, self.ee_position_slice]
        distance = torch.linalg.vector_norm(direction, dim=-1)
        previous_distance = torch.linalg.vector_norm(state[:, 7:10] - state[:, self.ee_position_slice], dim=-1)
        threshold = context[:, 10].clamp_min(1e-4)
        proximity = torch.exp(-(distance / threshold).square())
        closure = self._gripper_closure(next_state)
        direction_unit = direction / distance.unsqueeze(-1).clamp_min(1e-6)
        alignment = (self._get_ee_forward_axis(next_state) * direction_unit).sum(dim=-1).clamp(0, 1)
        height_error = next_state[:, 9] - context[:, 11]
        reward = (-self.w_distance * distance
                  + self.w_approach * (previous_distance - distance)
                  + self.w_alignment * alignment
                  + self.w_grasp * proximity * closure
                  + self.w_height * proximity * torch.tanh(10.0 * height_error))
        completed = ((height_error >= 0) & (distance < threshold)
                     & (closure >= self.grasp_force_threshold))
        # Clock is normalized by this episode's configured length. Distance and
        # relative height avoid tying the proxy to an arbitrary world origin.
        ended = completed | (distance > 2.0) | (next_state[:, 33] >= 1.0)
        return reward, ended.to(dtype=reward.dtype)

    def contract(self) -> Dict[str, Any]:
        return {
            "schema": "dq_physical34_ee_xyzw_policy_observation_v1",
            "state_dim": self.state_dim,
            "physical_dim": self.physical_dim,
            "ee_position_slice": [self.ee_position_slice.start, self.ee_position_slice.stop],
            "ee_quaternion_slice": [self.ee_quaternion_slice.start, self.ee_quaternion_slice.stop],
            "quaternion_order": "xyzw",
            "observation_slice": [self.observation_slice.start, self.observation_slice.stop],
            "observation_transition": "learned full policy observation delta",
            "action_dim": self.action_dim,
            "obs_dim": self.obs_dim,
            "context_dim": self.context_dim,
            "wm_input_dim": self.wm_input_dim,
            "delta_dim": self.delta_dim,
            "critic_dim": self.critic_dim,
            "target_height_source": "context[11]",
            "proximity_threshold_source": "context[10]",
            "gripper_open": self.gripper_open.item(),
            "gripper_closed": self.gripper_closed.item(),
            "reward": "geometric proxy; native success/hold statistics remain environment-owned",
        }
