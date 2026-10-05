"""Camera calibration and timestamped robot kinematics for M2.

Camera coordinates use optical x-right/y-down/z-forward. Target state is kept
in a separate training-reward record and never passed to the perception API.
"""
import numpy as np
import torch


def pose_matrix(position, quaternion_xyzw):
    q = quaternion_xyzw / quaternion_xyzw.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    rotation = torch.stack((
        1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y),
    ), -1).reshape(*position.shape[:-1], 3, 3)
    result = torch.eye(4, device=position.device, dtype=position.dtype).expand(*position.shape[:-1], 4, 4).clone()
    result[..., :3, :3] = rotation
    result[..., :3, 3] = position
    return result


@torch.no_grad()
def record_camera_geometry(env):
    """Called immediately after rendering, including the existing camera delay."""
    views = []
    projections = []
    for handle, cameras in zip(env.envs, env.camera_handles):
        views.append([env.gym.get_camera_view_matrix(env.sim, handle, camera) for camera in cameras])
        if not hasattr(env, "_m2_camera_intrinsics"):
            projections.append([env.gym.get_camera_proj_matrix(env.sim, handle, camera) for camera in cameras])
    view = torch.as_tensor(np.asarray(views), dtype=torch.float32, device=env.device)
    optical_conversion = torch.diag(torch.tensor([1., -1., -1., 1.], device=env.device))
    transforms = torch.linalg.inv(view).transpose(-1, -2) @ optical_conversion
    # Isaac camera views use the simulation world, while root-state tensors
    # use each environment's local world. Align both before filtering/rewards.
    if not hasattr(env, "_m2_environment_origins"):
        origins = [env.gym.get_env_origin(handle) for handle in env.envs]
        env._m2_environment_origins = torch.tensor(
            [[point.x, point.y, point.z] for point in origins], device=env.device)
    transforms[..., :3, 3] -= env._m2_environment_origins[:, None]
    if not hasattr(env, "_m2_camera_intrinsics"):
        projection = torch.as_tensor(np.asarray(projections), dtype=torch.float32, device=env.device)
        width, height = env.cfg["sensor"]["resized_resolution"]
        intrinsics = torch.zeros(env.num_envs, 2, 3, 3, device=env.device)
        intrinsics[..., 0, 0] = width * projection[..., 0, 0] / 2
        intrinsics[..., 1, 1] = height * projection[..., 1, 1] / 2
        intrinsics[..., 0, 2], intrinsics[..., 1, 2] = width / 2, height / 2
        intrinsics[..., 2, 2] = 1
        env._m2_camera_intrinsics = intrinsics
    env._m2_camera_transforms = transforms
    env._m2_camera_timestamps = torch.full((env.num_envs, 2), float(env.gym.get_sim_time(env.sim)),
                                         device=env.device, dtype=torch.float64)
    # Ground truth here has a single consumer: training reward reference points.
    env._m2_reward_capture_object_states = env._cube_root_states.detach().clone()


@torch.no_grad()
def sensor_packet(env):
    """Runtime sensor contract also implemented by CPU test environments."""
    if hasattr(env, "get_asymmetric_sensor_packet"):
        return env.get_asymmetric_sensor_packet()
    if not hasattr(env, "_m2_camera_transforms"):
        raise RuntimeError("M2 requires capture-time geometry: enable sensor.recordCameraGeometry before creating the environment")
    frame = env._camera_frame_observation().reshape(env.num_envs, 4, 54, 96)
    images = torch.stack((frame[:, [0, 2]], frame[:, [1, 3]]), dim=1).clone()
    images[:, :, 1] *= 3.0  # Existing DQ image depth is clipped at 3 m and divided by 3.
    base = env._robot_root_states
    return {
        "images": images,
        "T_world_camera": env._m2_camera_transforms,
        "intrinsics": env._m2_camera_intrinsics,
        "timestamps": env._m2_camera_timestamps,
        "now": torch.full((env.num_envs,), float(env.gym.get_sim_time(env.sim)), device=env.device, dtype=torch.float64),
        "T_world_base": pose_matrix(base[:, :3], base[:, 3:7]),
    }


@torch.no_grad()
def visible_surface_points(images, transforms, intrinsics, config=None):
    """Clean rendered surface centroid for TRAINING REWARD references only.

    This is not an object centre or a deployable perfect measurement. The
    actor's estimate is produced separately from corrupted sensor frames.
    """
    from .asymmetric_perception import PerceptionConfig
    config = config or PerceptionConfig()
    depth = images[:, :, 1]
    mask = ((images[:, :, 0] > config.mask_threshold) & torch.isfinite(depth)
            & (depth >= config.min_depth_m) & (depth < config.max_depth_m))
    height, width = depth.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=depth.device), torch.arange(width, device=depth.device), indexing="ij")
    depth = torch.where(mask, depth, torch.zeros_like(depth))
    k = intrinsics
    points = torch.stack(((xx-k[..., 0, 2, None, None])*depth/k[..., 0, 0, None, None],
                          (yy-k[..., 1, 2, None, None])*depth/k[..., 1, 1, None, None], depth), -1)
    count = mask.sum(dim=(-1, -2))
    centre = points.sum(dim=(-3, -2)) / count.clamp_min(1)[..., None]
    world = (transforms[..., :3, :3] @ centre[..., None]).squeeze(-1) + transforms[..., :3, 3]
    return world, count >= config.min_valid_pixels, count
