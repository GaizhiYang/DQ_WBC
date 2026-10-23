import os
import yaml
import argparse
from pathlib import Path


SGS_DEFAULTS = {
    "mode": "disabled",
    "bank_path": "",
    "bank_size": 1536,
    "history_size": 100,
    "target_success": 0.5,
    "kappa": 1.0,
    "temperature": 2.0,
    "epsilon": 1e-4,
    "p_init": 0.0,
}


def configure_teacher_experiment(cfg, args, is_eval=False, strict_eval=False):
    """Apply entrypoint overrides without changing the planned reward horizon."""
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError("--stop_after must be positive")
    env_cfg = cfg["env"]
    sgs = dict(SGS_DEFAULTS)
    sgs.update(env_cfg.get("sgs", {}))
    if args.sgs_mode is not None:
        sgs["mode"] = args.sgs_mode
    if args.sgs_bank is not None:
        sgs["bank_path"] = args.sgs_bank
    if sgs["mode"] not in ("disabled", "uniform", "sgs"):
        raise ValueError("env.sgs.mode must be disabled, uniform, or sgs")
    sgs["seed"] = args.seed
    env_cfg["sgs"] = sgs
    cfg["seed"] = args.seed
    env_cfg["successMode"] = args.success_mode or env_cfg.get("successMode", "legacy")
    if env_cfg["successMode"] not in ("legacy", "hold"):
        raise ValueError("successMode must be legacy or hold")

    if strict_eval:
        if not is_eval or not args.checkpoint or args.resume:
            raise ValueError("Strict evaluation requires --checkpoint and does not accept --resume")
        if args.eval_episodes <= 0:
            raise ValueError("--eval_episodes must be positive")
        if args.task not in ("", "B1Z1PickMulti"):
            raise ValueError("Strict SGS evaluation supports --task B1Z1PickMulti")
        if args.object_name not in (None, "sugar_box"):
            raise ValueError("Strict SGS evaluation supports --object_name sugar_box")
        args.task = "B1Z1PickMulti"
        args.object_name = "sugar_box"
        sgs["mode"] = "disabled"
        env_cfg.update({"successMode": "hold", "holdSteps": 15,
                        "terrainType": "plane", "D1_bench_task_level": "Level02",
                        "robotStartDistanceRange": [2.0, 2.0], "maxEpisodeLength": 100})
    elif is_eval and sgs["mode"] != "disabled":
        raise ValueError("Use eval_sgs_teacher.py for original-task evaluation; training-bank sampling is disabled there")

    if not is_eval and sgs["mode"] != "disabled":
        if args.resume or args.checkpoint:
            raise ValueError("SGS/uniform-bank training currently starts from scratch; --resume/--checkpoint cannot restore sampler state")
        if args.task != "B1Z1PickMulti" or args.object_name != "sugar_box":
            raise ValueError("SGS training requires --task B1Z1PickMulti --object_name sugar_box")
        if not sgs["bank_path"] or not Path(sgs["bank_path"]).is_file():
            raise ValueError("SGS training requires an existing validated bank via --sgs_bank or env.sgs.bank_path")
        # The bank loader validates the schema, source task, and physical validation.
        env_cfg["successMode"] = "hold"
    return cfg


def teacher_run_end(planned_timesteps, initial_timestep=0, stop_after=None):
    """Return the loop end; callers keep args.timesteps as the total plan."""
    if planned_timesteps <= initial_timestep:
        raise ValueError("--timesteps must exceed the checkpoint step (it is the total plan, not remaining steps)")
    if stop_after is None:
        return planned_timesteps
    if stop_after <= 0:
        raise ValueError("--stop_after must be positive")
    return min(planned_timesteps, initial_timestep + stop_after)

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

def get_params(argv=None):
    parser = argparse.ArgumentParser(
        prog="Z1 training",
    )
    parser.add_argument("--task", type=str, default="")
    parser.add_argument("--timesteps", type=int, default=24000)
    parser.add_argument("--stop_after", type=int, default=None,
                        help="Stop training after this many steps in this invocation without shortening the --timesteps reward schedule.")
    parser.add_argument("--cfg_file", type=str, default=None,
                        help="Teacher YAML path; defaults to data/cfg/DQ_teacher.yaml.")
    parser.add_argument("--sgs_mode", choices=("disabled", "uniform", "sgs"), default=None,
                        help="Override env.sgs.mode. Enabled modes require a validated reset bank.")
    parser.add_argument("--sgs_bank", type=str, default=None,
                        help="Path to a validated SGS reset bank; does not build a bank.")
    parser.add_argument("--success_mode", choices=("legacy", "hold"), default=None,
                        help="Override task success semantics. Strict evaluation always uses hold.")
    parser.add_argument("--eval_episodes", type=int, default=2000,
                        help="Number of completed episodes in eval_sgs_teacher.py.")
    parser.add_argument("--eval_output", type=str, default=None,
                        help="Optional JSON result path for strict teacher evaluation.")
    parser.add_argument("--control_freq", type=int, default=None) # how frequent low-level request high-level
    parser.add_argument("--rl_device", type=str, default="cuda:0")
    parser.add_argument("--sim_device", type=str, default="cuda:0")
    parser.add_argument("--graphics_device_id", type=int, default=-1)
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
    parser.add_argument("--no_velocity_obs", action="store_true",
                        help="Remove robot local linear velocity (3D) and object-relative planar velocity (2D) from B1Z1PickMulti teacher observations.")
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
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num_envs", type=int, default=None,
                        help="Number of environments to create. Overrides the config value.")
    parser.add_argument("--object_name", type=str, default=None,
                        help="Train/evaluate DQ_teacher or DQ_stu on one object from env.asset.asset_multi (e.g. sugar_box).")
    
    args = parser.parse_args(argv)
    
    return args
