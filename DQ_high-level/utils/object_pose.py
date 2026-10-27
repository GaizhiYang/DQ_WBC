"""The original teacher's object-pose convention, shared by current/future poses."""

from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate_inverse, euler_from_quat
import torch
from torch import Tensor


@torch.jit.script
def object_pose_in_teacher_frame(position_world: Tensor, quaternion_world: Tensor,
                                 arm_base_world: Tensor, robot_yaw_quaternion: Tensor) -> Tensor:
    """[N,6]: arm-relative heading-aligned xy, WORLD z, yaw-relative RPY.

    All inputs are flat batches. Use the SAME current arm base and robot yaw
    for both current and future object poses. Quaternions are xyzw, angles rad.
    """
    position = quat_rotate_inverse(robot_yaw_quaternion, position_world - arm_base_world)
    position[:, 2] = position_world[:, 2]
    orientation = quat_mul(quat_conjugate(robot_yaw_quaternion), quaternion_world)
    rpy = torch.stack(euler_from_quat(orientation), dim=-1)
    return torch.cat((position, rpy), dim=-1)
