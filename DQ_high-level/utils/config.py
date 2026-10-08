import os
import yaml
import argparse
from pathlib import Path

def load_cfg(file_path):
    with open(os.path.join(os.getcwd(), file_path), 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    
    return cfg

def copy_cfg(file_path, target_path, cfg=None):
    import subprocess
    Path(target_path).mkdir(parents=True, exist_ok=True)
    if cfg is not None:
        with open(Path(target_path) / Path(file_path).name, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        return
    subprocess.run(["cp", file_path, target_path])

def get_params():
    parser = argparse.ArgumentParser(
        prog="Z1 training",
    )
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--timesteps", type=int, default=24000)
    parser.add_argument("--control_freq", type=int, default=None) # how frequent low-level request high-level
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=-1)
    parser.add_argument("--graphics_device_ids", type=str, default=None,
                        help="Asymmetric torchrun: comma-separated Vulkan graphics indices, one per local rank.")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="isaacgym")
    parser.add_argument("--wandb_name", type=str, default="isaacgym")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--experiment_dir", type=str, default="experiments")
    parser.add_argument("--debugvis", action="store_true")
    parser.add_argument("--save_image", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wrist_seg", action="store_true")
    parser.add_argument("--front_only", action="store_true")
    parser.add_argument("--seperate", action="store_true")
    parser.add_argument("--teacher_ckpt_path", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--roboinfo", action="store_true")
    parser.add_argument("--observe_gait_commands", action="store_true")
    parser.add_argument("--small_value_set_zero", action="store_true")
    parser.add_argument("--fixed_base", action="store_true")
    parser.add_argument("--use_tanh", action="store_true")
    parser.add_argument("--reach_only", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--last_commands", action="store_true")
    parser.add_argument("--no_feature", action="store_true")
    parser.add_argument("--mask_arm", action="store_true")
    parser.add_argument("--mlp_stu", action="store_true")
    parser.add_argument("--depth_random", action="store_true")
    parser.add_argument("--pitch_control", action="store_true")
    parser.add_argument("--pred_success", action="store_true")
    parser.add_argument("--near_goal_stop", action="store_true")
    parser.add_argument("--obj_move_prob", type=float, default=0.0)
    parser.add_argument("--rand_control", action="store_true")
    parser.add_argument("--arm_delay", action="store_true")
    parser.add_argument("--rand_cmd_scale", action="store_true")
    parser.add_argument("--rand_depth_clip", action="store_true")
    parser.add_argument("--stop_pick", action="store_true")
    parser.add_argument("--arm_kp", type=int, default=40) # only useful when log data
    parser.add_argument("--arm_kd", type=float, default=2) # only useful when log data
    parser.add_argument("--table_height", type=float, default=None) # only useful when log data
    parser.add_argument("--seed", type=int, default=43) # only useful when log data
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Number of environments to create. Overrides the config value.")
    parser.add_argument("--object_name", type=str, default=None,
                        help="Train/evaluate DQ_teacher on one object from env.asset.asset_multi (e.g. sugar_box).")
    parser.add_argument("--grasp_selector", choices=("gfm", "karl", "geometric"), default=None,
                        help="Privileged teacher: learned GFM (default), KARL orientation selection, or center/top-down geometric selection. Restored from checkpoint when omitted.")
    parser.add_argument("--karl_switch_margin_deg", type=float, default=None,
                        help="KARL selection: minimum total-cost improvement to switch, expressed in degrees (default 30).")
    parser.add_argument("--karl_orientation_preference", choices=("none", "karl"), default=None,
                        help="KARL selection: optional original UR5 0/1 orientation preference, evaluated in the DQ robot-base frame (default none).")
    parser.add_argument("--geometric_switch_margin", type=float, default=None,
                        help="Geometric selection: minimum normalized score improvement to switch targets (not an angle).")
    parser.add_argument("--geometric_center_weight", type=float, default=None,
                        help="Geometric selection: weight of horizontal distance from the object center.")
    parser.add_argument("--geometric_topdown_weight", type=float, default=None,
                        help="Geometric selection: weight of deviation from a downward approach.")
    parser.add_argument("--geometric_height_weight", type=float, default=None,
                        help="Geometric selection: weight of deviation from the preferred upper object region.")
    parser.add_argument("--geometric_table_clearance", type=float, default=None,
                        help="Geometric selection: padding around the table collision proxy in meters (default 0.002).")
    parser.add_argument("--geometric_lock_distance", type=float, default=None,
                        help="Geometric selection: distance in meters to latch a valid target while commanded closed (default 0.08).")
    parser.add_argument("--vis_selected_grasp", action="store_true",
                        help="KARL/geometric teacher: draw the selected grasp position and RGB axes in the viewer (requires non-headless mode).")
    parser.add_argument("--grasp_vis_envs", type=int, default=8,
                        help="Maximum number of environments with selected-grasp markers (default 8).")
    # Parsed here as well as by the new entrypoint because B1Z1PickMulti
    # reparses the process command line during environment construction.
    parser.add_argument("--vision_config", type=str, default=None,
                        help="Configuration for train/play_multistate_DQ_teacher_vision.py.")
    parser.add_argument("--vision_mode", choices=("images", "zero", "none"), default=None,
                        help="Visual teacher: real images, zero-image control, or original teacher control.")
    parser.add_argument("--teacher_init_checkpoint", type=str, default="",
                        help="Warm-start a new visual/asymmetric experiment. Asymmetric training starts from scratch when neither this nor --checkpoint is supplied.")
    parser.add_argument("--teacher_initial_step", type=int, default=None,
                        help="Legacy teacher environment step (required for checkpoints without a numeric suffix).")
    parser.add_argument("--vision_minibatch_size", type=int, default=None,
                        help="Number of high-level observations per PPO mini-batch.")
    parser.add_argument("--vision_learning_epochs", type=int, default=None,
                        help="Override PPO epochs for the visual-teacher experiment.")
    parser.add_argument("--vision_task_level", choices=tuple("Level%02d" % i for i in range(6)), default=None,
                        help="Override the benchmark level for visual-teacher training/evaluation.")
    parser.add_argument("--asymmetric_config", type=str, default=None,
                        help="Asymmetric-teacher YAML for a new experiment.")
    parser.add_argument("--asymmetric_mode", choices=("m0", "m1", "m2"), default=None,
                        help="m0: visual actor; m1: task-aware Critic; m2: additionally persistent perception.")
    parser.add_argument("--asymmetric_allow_legacy", action="store_true",
                        help="Explicitly allow an old teacher without visual weights as M0 initialization.")
    parser.add_argument("--asymmetric_eval_interval", type=int, default=None,
                        help="Periodic evaluation interval in training vector steps; 0 disables it.")
    parser.add_argument("--asymmetric_eval_steps", type=int, default=None,
                        help="Vector steps per periodic evaluation (outside the training budget).")
    
    args = parser.parse_args()
    
    return args
