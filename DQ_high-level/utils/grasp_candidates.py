"""Grasp-candidate inputs of PredictPoint, without Isaac Gym (moved from the teacher entry)."""

from termcolor import cprint


def get_predict_point(camera_6p_info=None, cube_predict_info_path=None, cube_root_states_info_path=None, num_env=None, intervel=None, delta_height=None, object_indices=None):
    import json
    import os
    import torch
    import numpy as np
    from scipy.spatial.transform import Rotation as R

    curr_dir = os.getcwd()
    data_dir = os.path.join(curr_dir,"PredictPiontTransform" , "transform_info")
    cube_root_states_info_path = os.path.join(data_dir, cube_root_states_info_path)

    # load cubes's states（position + quaternion + velocity...）
    cube_states = torch.load(cube_root_states_info_path, map_location="cpu")[0]
    if object_indices is None:
        object_indices = list(range(cube_states.shape[0]))
    if not object_indices or any(i < 0 or i >= cube_states.shape[0] for i in object_indices):
        raise ValueError(f"Object dict_idx values must be in [0, {cube_states.shape[0] - 1}]")
    # Keep initial poses and grasp predictions in the same order as asset_multi.
    cube_states = cube_states[object_indices]
    cprint(f"cube_states.shape is {cube_states.shape}", "yellow")

    cube_pos = cube_states[:, :3]
    cube_quat = cube_states[:, 3:7]
    cube_rpy = torch.tensor(R.from_quat(cube_quat.numpy()).as_euler('xyz'))  # turn to RPY
    cube_init_tensor = torch.cat([cube_pos, cube_rpy], dim=1)
    cube_init_tensor = cube_init_tensor.to(torch.float32)

    base_n = cube_init_tensor.shape[0]  
    T_grasp_cv_all = []

    # read base_num sample's all transform
    for i in object_indices:
        sample_dir = os.path.join(data_dir, cube_predict_info_path)
        json_path = os.path.join(sample_dir, f"predictions_image_{(i):02d}.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        grasp_matrices = np.array(data['transform'])  
        T_grasp_cv_all.append(grasp_matrices)

    # Shape: [number of selected objects, candidate grasps per object, 4, 4].
    T_grasp_cv_all = np.stack(T_grasp_cv_all, axis=0)
    n_grasps = T_grasp_cv_all.shape[1]  # get n
    T_grasp_cv_all = torch.tensor(T_grasp_cv_all, dtype=torch.float32)

    # extent to num_env
    T_grasp_cv_list = [T_grasp_cv_all[i % base_n] for i in range(num_env)]
    grasp_cv_tensor = torch.stack(T_grasp_cv_list, dim=0)  # 形状 [num_env, n, 4, 4]

    camera_tensor = []
    cube_init_list = []
    cube_init_ori_list = []

    for i in range(num_env):
        base_idx = i % base_n
        height_offset = (delta_height / intervel) * ((i % intervel) + 1)

        # the initial pose of the camera, x,y is related to the initial pose of the float base 
        camera_x = -0.3
        camera_y = 2.0
        camera_z = 0.80 + height_offset
        camera_info = [camera_x, camera_y, camera_z, 0.0, 0.70, 0.0]
        camera_tensor.append([camera_info] * n_grasps)

        cube_init = cube_init_tensor[base_idx].clone()
        cube_init[2] += height_offset 
        cube_init_list.append([cube_init] * n_grasps)
        cube_init_ori_list.append(cube_init)

    camera_tensor = torch.tensor(camera_tensor, dtype=torch.float32)  # [num_env, n, 6]
    cube_init_tensor = torch.stack([torch.stack(cube_init, dim=0) for cube_init in cube_init_list], dim=0)  # [num_env, n, 6]
    return camera_tensor, grasp_cv_tensor, cube_init_tensor
