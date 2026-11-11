"""Object poses in the full-orientation mechanical-arm base frame."""

from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate_inverse, euler_from_quat
import torch
from torch import Tensor


@torch.jit.script
def object_pose_in_arm_frame(position_world: Tensor, quaternion_world: Tensor,
                             arm_base_world: Tensor, robot_quaternion: Tensor) -> Tensor:
    """Return [N,7] arm-relative position and xyzw orientation.

    Frame A has its origin at ``arm_base_world`` and axes following the robot's
    complete roll, pitch and yaw. Each batch item uses the robot pose at the
    same time as its object pose, including when constructing EKF history.
    All inputs are flat batches and quaternions use xyzw order.
    """
    position = quat_rotate_inverse(robot_quaternion, position_world - arm_base_world)
    orientation = quat_mul(quat_conjugate(robot_quaternion), quaternion_world)
    return torch.cat((position, orientation), dim=-1)


@torch.jit.script
def pose_to_rpy_observation(poses: Tensor) -> Tensor:
    """Encode [N,7] xyz/xyzw poses as [N,6] xyz/roll/pitch/yaw in radians."""
    rpy = torch.stack(euler_from_quat(poses[:, 3:7]), dim=-1)
    return torch.cat((poses[:, :3], rpy), dim=-1)


@torch.jit.script
def object_pose_in_teacher_frame(position_world: Tensor, quaternion_world: Tensor,
                                 arm_base_world: Tensor, robot_quaternion: Tensor) -> Tensor:
    """Return the teacher's [N,6] object observation in full arm frame A."""
    poses = object_pose_in_arm_frame(
        position_world, quaternion_world, arm_base_world, robot_quaternion
    )
    return pose_to_rpy_observation(poses)
