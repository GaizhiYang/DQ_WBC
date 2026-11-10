"""
Physical State Representation and Capture for DQ_WBC

This module defines the physical state for SuperMimic world model
and provides methods to capture it from the DQ_WBC environment.
"""
import torch
from typing import Tuple
from dataclasses import dataclass


def environment_root_state(env, kind):
    """Read Isaac Gym's per-environment actor views, or explicit indexed roots."""
    attribute = '_robot_root_states' if kind == 'robot' else '_cube_root_states'
    value = getattr(env, attribute, None)
    if value is None:
        roots = getattr(env, 'root_states', getattr(env, '_root_states', None))
        names = ('robot_indices', '_robot_actor_ids') if kind == 'robot' else (
            'object_indices', 'cube_indices', '_cube_actor_ids')
        indices = next((getattr(env, name) for name in names if hasattr(env, name)), None)
        if roots is None or indices is None:
            raise AttributeError(f"StateCapture requires {attribute} or actor root states and {kind} indices")
        value = roots[indices.long()]
    if value.shape != (env.num_envs, 13):
        raise ValueError(f"{attribute} must have shape ({env.num_envs}, 13), got {tuple(value.shape)}")
    return value


def episode_length(env):
    value = getattr(env, 'max_episode_length', None)
    if value is None:
        value = getattr(env, 'cfg', {}).get('env', {}).get('maxEpisodeLength')
    if value is None or float(value) <= 0:
        raise ValueError("The environment must provide a positive max_episode_length")
    return float(value)


@dataclass
class PhysicalStateLayout:
    """
    Physical state layout for DQ_WBC manipulation task.

    Total: 34 dimensions

    Breakdown:
    - Robot base: pose (3 pos + 4 quat = 7D)
    - Object: pose + velocity (3 pos + 6 rot6d + 3 vel + 3 angvel = 15D)
    - End-effector: derived from robot (handled separately)
    - Gripper: position + velocity (1 + 1 = 2D)
    - Previous action: (9D)
    - Clock: normalized timestep (1D)

    Note: This is a PHYSICAL state, not observation!
    """

    # Robot (7D)
    robot_base_pos_start = 0
    robot_base_pos_end = 3
    robot_base_quat_start = 3
    robot_base_quat_end = 7

    # Object (15D)
    object_pos_start = 7
    object_pos_end = 10
    object_rot6d_start = 10  # 6D rotation representation
    object_rot6d_end = 16
    object_lin_vel_start = 16
    object_lin_vel_end = 19
    object_ang_vel_start = 19
    object_ang_vel_end = 22

    # Gripper (2D)
    gripper_pos_start = 22
    gripper_pos_end = 23
    gripper_vel_start = 23
    gripper_vel_end = 24

    # Previous action (9D)
    prev_action_start = 24
    prev_action_end = 33

    # Clock (1D)
    clock_start = 33
    clock_end = 34

    total_dim = 34


class StateCapture:
    """
    Capture physical state from DQ_WBC environment.

    This bridges the gap between the environment's internal representation
    and the physical state needed by SuperMimic.
    """

    def __init__(self, env, device: str = "cuda:0"):
        """
        Args:
            env: DQ_WBC environment (B1Z1PickMulti)
            device: Torch device
        """
        self.env = env
        self.device = torch.device(device)
        self.num_envs = env.num_envs
        self.layout = PhysicalStateLayout()

        if env.num_actions != 9:
            raise ValueError(f"The 34D physical state requires 9 actions, got {env.num_actions}")
        # Track previous action for state
        self.prev_action = torch.zeros(
            self.num_envs,
            env.num_actions,
            device=device
        )

        # Episode timestep counter
        self.timesteps = torch.zeros(self.num_envs, device=device)
        self.max_episode_length = episode_length(env)

    @torch.no_grad()
    def capture(self) -> torch.Tensor:
        """
        Capture current physical state from environment.

        Returns:
            state: [num_envs, 34] physical state tensor
        """
        state = torch.zeros(
            self.num_envs,
            self.layout.total_dim,
            device=self.device
        )

        # 1. Robot base (from environment's root_states)
        # Base position
        robot_root = environment_root_state(self.env, 'robot').to(self.device)
        object_root = environment_root_state(self.env, 'object').to(self.device)
        robot_base_pos = robot_root[:, :3]
        state[:, self.layout.robot_base_pos_start:self.layout.robot_base_pos_end] = robot_base_pos

        # Base quaternion
        robot_base_quat = torch.nn.functional.normalize(robot_root[:, 3:7], dim=-1)
        state[:, self.layout.robot_base_quat_start:self.layout.robot_base_quat_end] = robot_base_quat

        # 2. Object state
        # Object position
        obj_pos = object_root[:, :3]
        state[:, self.layout.object_pos_start:self.layout.object_pos_end] = obj_pos

        # Object rotation (convert quaternion to 6D representation)
        obj_quat = object_root[:, 3:7]
        obj_rot6d = self._quat_to_rot6d(obj_quat)
        state[:, self.layout.object_rot6d_start:self.layout.object_rot6d_end] = obj_rot6d

        # Object linear velocity
        obj_lin_vel = object_root[:, 7:10]
        state[:, self.layout.object_lin_vel_start:self.layout.object_lin_vel_end] = obj_lin_vel

        # Object angular velocity
        obj_ang_vel = object_root[:, 10:13]
        state[:, self.layout.object_ang_vel_start:self.layout.object_ang_vel_end] = obj_ang_vel

        # 3. Gripper state
        for name, column in (('pos', 22), ('vel', 23)):
            value = getattr(self.env, 'dof_' + name + '_gripper', None)
            if value is None:
                joints = getattr(self.env, 'num_gripper_joints', getattr(self.env, 'num_gripper_dof', 0))
                all_dofs = getattr(self.env, '_dof_' + name, getattr(self.env, 'dof_' + name, None))
                if all_dofs is None or joints <= 0:
                    raise AttributeError(f"StateCapture requires measured dof_{name}_gripper")
                value = all_dofs[:, -joints:]
            if value.ndim != 2 or value.shape[0] != self.num_envs or value.shape[1] == 0:
                raise ValueError(f"dof_{name}_gripper must contain joints for every environment")
            state[:, column] = value.to(self.device).mean(dim=-1)

        # 4. Previous action
        state[:, self.layout.prev_action_start:self.layout.prev_action_end] = self.prev_action

        # 5. Clock (normalized timestep)
        progress = getattr(self.env, 'progress_buf', None)
        if progress is not None:
            self.timesteps.copy_(progress.reshape(self.num_envs).to(self.device))
        normalized_time = self.timesteps / self.max_episode_length
        state[:, self.layout.clock_start:self.layout.clock_end] = normalized_time.unsqueeze(-1)

        return state

    @torch.no_grad()
    def update_action(self, action: torch.Tensor):
        """
        Update previous action buffer.

        Args:
            action: [num_envs, action_dim] action taken
        """
        if action.shape != self.prev_action.shape:
            raise ValueError(f"Expected action shape {tuple(self.prev_action.shape)}, got {tuple(action.shape)}")
        self.prev_action.copy_(action.detach().to(self.device))

    @torch.no_grad()
    def increment_timestep(self, done: torch.Tensor = None):
        """
        Advance after physics, preserving terminal time until the actual reset.

        Args:
            done: Optional legacy argument; reset_episodes handles actual resets.
        """
        progress = getattr(self.env, 'progress_buf', None)
        if progress is None:
            self.timesteps += 1
        else:
            self.timesteps.copy_(progress.reshape(self.num_envs).to(self.device))

    @torch.no_grad()
    def reset_episodes(self, env_ids: torch.Tensor):
        """
        Reset state tracking for specific environments.

        Args:
            env_ids: Indices of environments to reset
        """
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)
        if env_ids.numel() > 0:
            self.timesteps[env_ids] = 0
            self.prev_action[env_ids] = 0.0

    # ========== Utility Methods ==========

    def _quat_to_rot6d(self, quat: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to 6D rotation representation.

        6D representation uses first two columns of rotation matrix,
        which is continuous and suitable for learning.

        Args:
            quat: [N, 4] Isaac Gym quaternions (x, y, z, w)

        Returns:
            rot6d: [N, 6] 6D rotation representation
        """
        # Normalize quaternion
        quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

        x, y, z, w = quat.unbind(dim=-1)

        # Convert to rotation matrix (first two columns)
        # Column 1
        c1_x = 1 - 2 * (y**2 + z**2)
        c1_y = 2 * (x*y + w*z)
        c1_z = 2 * (x*z - w*y)

        # Column 2
        c2_x = 2 * (x*y - w*z)
        c2_y = 1 - 2 * (x**2 + z**2)
        c2_z = 2 * (y*z + w*x)

        # Stack into 6D representation
        rot6d = torch.stack([c1_x, c1_y, c1_z, c2_x, c2_y, c2_z], dim=-1)

        return rot6d

    def _rot6d_to_quat(self, rot6d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation back to quaternion.

        Args:
            rot6d: [N, 6] 6D rotation

        Returns:
            quat: [N, 4] quaternions (x, y, z, w)
        """
        # Extract first two columns
        c1 = rot6d[:, :3]
        c2 = rot6d[:, 3:6]

        # Normalize first column
        c1 = c1 / (torch.norm(c1, dim=-1, keepdim=True) + 1e-8)

        # Gram-Schmidt orthogonalization for second column
        c2 = c2 - (c1 * c2).sum(dim=-1, keepdim=True) * c1
        c2 = c2 / (torch.norm(c2, dim=-1, keepdim=True) + 1e-8)

        # Third column from cross product
        c3 = torch.cross(c1, c2, dim=-1)

        # Build rotation matrix
        R = torch.stack([c1, c2, c3], dim=-1)  # [N, 3, 3]

        # Select the best-conditioned component. Dividing only by w fails at
        # the 180-degree rotations present in the object's reset orientations.
        m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
        magnitudes = torch.sqrt(torch.stack((1 + m00 + m11 + m22,
                                            1 + m00 - m11 - m22,
                                            1 - m00 + m11 - m22,
                                            1 - m00 - m11 + m22), -1).clamp_min(0))
        candidates = torch.stack((
            torch.stack((magnitudes[:, 0] ** 2, R[:, 2, 1] - R[:, 1, 2],
                         R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]), -1),
            torch.stack((R[:, 2, 1] - R[:, 1, 2], magnitudes[:, 1] ** 2,
                         R[:, 1, 0] + R[:, 0, 1], R[:, 0, 2] + R[:, 2, 0]), -1),
            torch.stack((R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] + R[:, 0, 1],
                         magnitudes[:, 2] ** 2, R[:, 2, 1] + R[:, 1, 2]), -1),
            torch.stack((R[:, 1, 0] - R[:, 0, 1], R[:, 0, 2] + R[:, 2, 0],
                         R[:, 2, 1] + R[:, 1, 2], magnitudes[:, 3] ** 2), -1),
        ), -2) / (2 * magnitudes.clamp_min(0.1).unsqueeze(-1))
        selected = candidates[torch.arange(R.shape[0], device=R.device), magnitudes.argmax(-1)]
        return torch.nn.functional.normalize(selected[:, [1, 2, 3, 0]], dim=-1)

    def get_object_pose(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract object position and rotation from state.

        Args:
            state: [N, 34] physical state

        Returns:
            pos: [N, 3] object position
            quat: [N, 4] object quaternion
        """
        pos = state[:, self.layout.object_pos_start:self.layout.object_pos_end]
        rot6d = state[:, self.layout.object_rot6d_start:self.layout.object_rot6d_end]
        quat = self._rot6d_to_quat(rot6d)

        return pos, quat

    def get_gripper_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Extract gripper position from state.

        Args:
            state: [N, 34] physical state

        Returns:
            gripper: [N, 1] gripper position
        """
        return state[:, self.layout.gripper_pos_start:self.layout.gripper_pos_end]
