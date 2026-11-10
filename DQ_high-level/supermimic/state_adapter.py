"""
DQ_WBC State Adapter for SuperMimic
Captures full environment state and converts to simplified 51D representation
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class DQStateAdapter:
    """
    Adapts DQ_WBC environment state to SuperMimic's simplified state representation.

    Simplified State Space (51D):
    - Robot DOF positions (12D): leg joints
    - Robot DOF velocities (12D): leg joint velocities
    - End-effector pose (7D): position (3D) + quaternion (4D)
    - Object 6D pose (7D): position (3D) + quaternion (4D)
    - Object velocities (6D): linear (3D) + angular (3D)
    - Gripper state (2D): position + velocity
    - Task context (5D): distance to object, approach alignment, grasp readiness, time remaining, success flag
    """

    def __init__(self, env, device='cuda:0'):
        self.env = env
        self.device = device
        self.num_envs = env.num_envs

        # State dimensions
        self.robot_dof_dim = 12  # B1 leg joints (3 per leg × 4 legs)
        self.ee_pose_dim = 7  # end-effector position + quaternion
        self.object_pose_dim = 7  # object position + quaternion
        self.object_vel_dim = 6  # linear + angular velocity
        self.gripper_dim = 2  # gripper position + velocity
        self.context_dim = 5  # task-specific context

        self.state_dim = (self.robot_dof_dim + self.robot_dof_dim +
                         self.ee_pose_dim + self.object_pose_dim +
                         self.object_vel_dim + self.gripper_dim + self.context_dim)

        assert self.state_dim == 51, f"State dimension mismatch: {self.state_dim} != 51"

        # Cache for state components
        self._last_simplified_state = None

    def capture_state(self) -> torch.Tensor:
        """
        Capture current full environment state and convert to simplified 51D representation.

        Returns:
            simplified_state: [num_envs, 51] tensor
        """
        env = self.env

        # 1. Robot DOF positions (12D) - only leg joints, exclude arm
        # B1 has 12 leg DOFs: FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...
        robot_dof_pos = env._dof_pos[:, :12].clone()  # [N, 12]

        # 2. Robot DOF velocities (12D)
        robot_dof_vel = env._dof_vel[:, :12].clone()  # [N, 12]

        # 3. End-effector pose (7D)
        # Get from rigid body states (Z1 end-effector index - ee_gripper_link)
        ee_pos = env.ee_pos.clone()  # [N, 3]
        ee_quat = env.ee_orn.clone()  # [N, 4]
        ee_pose = torch.cat([ee_pos, ee_quat], dim=-1)  # [N, 7]

        # 4. Object 6D pose (7D)
        obj_pos = env._cube_root_states[:, :3].clone()  # [N, 3]
        obj_quat = env._cube_root_states[:, 3:7].clone()  # [N, 4]
        obj_pose = torch.cat([obj_pos, obj_quat], dim=-1)  # [N, 7]

        # 5. Object velocities (6D)
        obj_lin_vel = env._cube_root_states[:, 7:10].clone()  # [N, 3]
        obj_ang_vel = env._cube_root_states[:, 10:13].clone()  # [N, 3]
        obj_vel = torch.cat([obj_lin_vel, obj_ang_vel], dim=-1)  # [N, 6]

        # 6. Gripper state (2D)
        # Z1 gripper DOFs
        gripper_pos = env.dof_pos_gripper.mean(dim=-1, keepdim=True)  # [N, 1] averaged
        gripper_vel = env.dof_vel_gripper.mean(dim=-1, keepdim=True)  # [N, 1] averaged
        gripper_state = torch.cat([gripper_pos, gripper_vel], dim=-1)  # [N, 2]

        # 7. Task context (5D)
        # Distance to object
        ee_to_obj_dist = torch.norm(ee_pos - obj_pos, dim=-1, keepdim=True)  # [N, 1]

        # Approach alignment: dot product of ee z-axis and direction to object
        ee_z_axis = self._quat_rotate_vec(ee_quat, ee_quat.new_tensor([0.0, 0.0, 1.0]))
        ee_to_obj_dir = (obj_pos - ee_pos) / (ee_to_obj_dist + 1e-6)
        approach_alignment = (ee_z_axis * ee_to_obj_dir).sum(dim=-1, keepdim=True)  # [N, 1]

        # Grasp readiness: combination of distance and gripper state
        grasp_readiness = torch.sigmoid((0.1 - ee_to_obj_dist) * 10.0)  # [N, 1]

        # Time remaining (normalized)
        time_remaining = (env.max_episode_length - env.progress_buf) / env.max_episode_length  # [N]
        time_remaining = time_remaining.unsqueeze(-1)  # [N, 1]

        # Success flag (from env if available)
        if hasattr(env, 'success_buf'):
            success_flag = env.success_buf.float().unsqueeze(-1)  # [N, 1]
        else:
            success_flag = torch.zeros(self.num_envs, 1, device=self.device)

        context = torch.cat([
            ee_to_obj_dist,
            approach_alignment,
            grasp_readiness,
            time_remaining,
            success_flag
        ], dim=-1)  # [N, 5]

        # Concatenate all components
        simplified_state = torch.cat([
            robot_dof_pos,      # 12D
            robot_dof_vel,      # 12D
            ee_pose,            # 7D
            obj_pose,           # 7D
            obj_vel,            # 6D
            gripper_state,      # 2D
            context             # 5D
        ], dim=-1)  # [N, 51]

        self._last_simplified_state = simplified_state
        return simplified_state

    def _quat_rotate_vec(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """
        Rotate vector by quaternion.
        quat: [N, 4] (x, y, z, w), matching Isaac Gym rigid-body states
        vec: [3] or [N, 3]
        """
        if vec.dim() == 1:
            vec = vec.unsqueeze(0).expand(quat.shape[0], -1)

        # Isaac Gym stores the scalar component last.
        xyz = quat[:, :3]
        w = quat[:, 3:4]

        # Quaternion rotation formula
        t = 2 * torch.cross(xyz, vec, dim=-1)
        rotated = vec + w * t + torch.cross(xyz, t, dim=-1)

        return rotated

    def get_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get structured state dictionary for analysis and debugging.
        """
        if self._last_simplified_state is None:
            self.capture_state()

        state = self._last_simplified_state
        return {
            'robot_dof_pos': state[:, :12],
            'robot_dof_vel': state[:, 12:24],
            'ee_pose': state[:, 24:31],
            'object_pose': state[:, 31:38],
            'object_vel': state[:, 38:44],
            'gripper_state': state[:, 44:46],
            'context': state[:, 46:51]
        }

    def compute_geometric_proxy_reward(self, state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable geometric proxy rewards for imagination rollouts.
        Approximates contact-based rewards with smooth distance functions.

        Args:
            state: [N, 51] current simplified state
            next_state: [N, 51] next simplified state

        Returns:
            reward: [N] differentiable reward signal
        """
        # Extract components from states
        ee_pos = state[:, 24:27]
        obj_pos = state[:, 31:34]
        next_ee_pos = next_state[:, 24:27]
        next_obj_pos = next_state[:, 31:34]

        # Reward 1: Approach object (distance reduction)
        curr_dist = torch.norm(ee_pos - obj_pos, dim=-1)
        next_dist = torch.norm(next_ee_pos - next_obj_pos, dim=-1)
        approach_reward = (curr_dist - next_dist) * 5.0  # reward for getting closer

        # Reward 2: Proximity reward (exponential)
        proximity_reward = torch.exp(-next_dist * 10.0)

        # Reward 3: Gripper alignment
        next_ee_quat = next_state[:, 27:31]
        ee_z = self._quat_rotate_vec(next_ee_quat, next_ee_quat.new_tensor([0.0, 0.0, 1.0]))
        ee_to_obj = (next_obj_pos - next_ee_pos) / (next_dist.unsqueeze(-1) + 1e-6)
        alignment = (ee_z * ee_to_obj).sum(dim=-1)
        alignment_reward = torch.clamp(alignment, 0, 1) * 0.5

        # Reward 4: Grasp success surrogate (object height + gripper closure)
        obj_height = next_obj_pos[:, 2]  # z-coordinate
        grasp_readiness = next_state[:, 48]  # from context
        lift_reward = torch.clamp(obj_height - 0.5, 0, 0.3) * 10.0 * grasp_readiness

        # Reward 5: Stability penalty (velocity smoothness)
        robot_vel = next_state[:, 12:24]
        velocity_penalty = -torch.norm(robot_vel, dim=-1) * 0.01

        # Total reward
        total_reward = (
            approach_reward +
            proximity_reward +
            alignment_reward +
            lift_reward +
            velocity_penalty
        )

        return total_reward

    def native_candidate(self) -> torch.Tensor:
        """
        Capture native (physical) reward from environment for comparison.
        This is called after environment step but before reward computation.
        """
        # Return the environment's actual reward buffer
        if hasattr(self.env, 'rew_buf'):
            return self.env.rew_buf.clone()
        else:
            return torch.zeros(self.num_envs, device=self.device)


class StateCapture(nn.Module):
    """
    Hook for capturing states during environment stepping.
    Used by SuperMimic training loop to collect transitions.
    """

    def __init__(self):
        super().__init__()
        self.adapter = None
        self.state = None
        self.native_candidate = None
        self.step = 0

    def attach_adapter(self, adapter: DQStateAdapter):
        """Attach state adapter before environment stepping."""
        self.adapter = adapter

    def __call__(self, env) -> torch.Tensor:
        """
        Called during environment stepping to capture state.
        Returns dummy observation (actual capture happens in-place).
        """
        if self.adapter is None:
            raise RuntimeError("Must attach adapter before stepping capture-enabled env")

        self.state = self.adapter.capture_state()
        self.native_candidate = self.adapter.native_candidate().clone()
        self.step = int(env.common_step_counter) if hasattr(env, 'common_step_counter') else 0

        return torch.zeros(env.num_envs, device=env.device)
