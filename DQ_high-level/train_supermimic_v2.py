"""
SuperMimic v2 Training Script for DQ_WBC

Uses the same validated training loop as the teacher entry point.
"""

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import os
from copy import deepcopy
from pathlib import Path
from termcolor import cprint

# Import DQ_WBC components
from train_multistate_DQ_teacher import (
    create_env,
    teacher_config_path,
    get_predict_point
)
from utils.config import (
    load_cfg, get_params, configure_teacher_experiment,
    apply_pose_prediction_config, apply_tsil_config
)
from skrl.utils import set_seed

# Import SuperMimic v2 components
from simple_supermimic_trainer import run_supermimic_training


def train_supermimic_v2():
    """
    Main training function for SuperMimic v2.
    """
    # Parse arguments (reuse existing argument parser)
    args = get_params()
    args.eval = False

    set_seed(args.seed)
    args.wandb = args.wandb and (not args.eval) and (not args.debug)

    if args.sgs_mode in ("uniform", "sgs") and (args.resume or args.checkpoint):
        raise ValueError("SGS/uniform-bank training starts from scratch")

    file_path = teacher_config_path(args, is_eval=False, strict_eval=False)
    cfg_file = Path(file_path).name

    cprint(f"[SuperMimic v2] Training script: {os.path.realpath(__file__)}", "yellow")
    cprint(f"[SuperMimic v2] Using config file: {os.path.realpath(file_path)}", "red")

    # Load and configure environment
    cfg = load_cfg(file_path)
    configure_teacher_experiment(cfg, args, is_eval=False, strict_eval=False)
    prediction_cfg = apply_pose_prediction_config(cfg, args)

    cprint(f"[SuperMimic v2] Future pose observations: {prediction_cfg}", "yellow")

    if args.no_velocity_obs:
        cfg["env"]["noVelocityObs"] = True

    # Handle object_name (required for SGS)
    multi_obj = cfg["env"]["asset"]["asset_multi"]
    if args.object_name is not None:
        if args.object_name not in multi_obj:
            raise ValueError(f"Unknown --object_name {args.object_name!r}")
        multi_obj = {args.object_name: multi_obj[args.object_name]}
        cfg["env"]["asset"]["asset_multi"] = multi_obj
    object_indices = [obj["dict_idx"] for obj in multi_obj.values()]
    cprint(f"[SuperMimic v2] Using {len(multi_obj)} object(s): {', '.join(multi_obj)}", "yellow")

    # Apply other configs
    cfg['env']['wandb'] = args.wandb
    cfg['env']["useTanh"] = getattr(args, 'use_tanh', False)
    cfg['env']["near_goal_stop"] = getattr(args, 'near_goal_stop', False)
    cfg['env']["obj_move_prob"] = getattr(args, 'obj_move_prob', 0.0)

    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError("--num_envs must be positive")
        cfg['env']['numEnvs'] = args.num_envs
        cprint(f"[SuperMimic v2] Environment count: {args.num_envs}", "yellow")

    # Apply TSIL config
    tsil_cfg = apply_tsil_config(cfg, args, is_eval=False)

    # Get prediction point data
    intervel = 23
    args.intervel = intervel
    cube_predict_info_path = "contact_grasp_info_mul"
    cube_root_states_info_path = "30all_nomove_cube_root_states5.pt"

    cprint(f"[SuperMimic v2] Generating prediction point data...", "cyan")
    camera_6p_tensor, grasp_cv_tensor, cube_init_tensor = get_predict_point(
        cube_predict_info_path=cube_predict_info_path,
        cube_root_states_info_path=cube_root_states_info_path,
        num_env=cfg['env']['numEnvs'],
        intervel=intervel,
        delta_height=0.1,
        object_indices=object_indices
    )

    # Preserve configuration before the simulator fills runtime objects.
    experiment_cfg = deepcopy(cfg)
    env = create_env(cfg=cfg, args=args, camera_6p_tensor=camera_6p_tensor,
                     grasp_cv_tensor=grasp_cv_tensor, cube_init_tensor=cube_init_tensor)

    return run_supermimic_training(env, args, experiment_cfg)


if __name__ == "__main__":
    train_supermimic_v2()
