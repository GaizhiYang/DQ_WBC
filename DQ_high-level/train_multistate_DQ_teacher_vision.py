"""Warm-start a privileged teacher with optional dual-camera visual features.

images = visual teacher; none = continued-training teacher control;
zero = identical visual network with zero image input and real state guidance.
"""
from copy import deepcopy
from pathlib import Path
import os
import re
import sys

from utils.config import get_params, load_cfg, copy_cfg


HIGH_LEVEL_DIR = Path(__file__).resolve().parent
ENVIRONMENT_OPTIONS = (
    "roboinfo", "observe_gait_commands", "small_value_set_zero", "mask_arm",
    "rand_control", "arm_delay", "rand_cmd_scale", "rand_depth_clip", "stop_pick",
    "control_freq", "table_height", "near_goal_stop", "obj_move_prob",
)


def _explicit_option(name):
    flag = "--" + name
    return any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:])


def _latest_checkpoint(directory):
    candidates = []
    for path in directory.glob("agent_*.pt"):
        match = re.fullmatch(r"agent_(\d+)\.pt", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise ValueError("No numbered visual-teacher checkpoint found in " + str(directory))
    return str(max(candidates)[1])


def get_trainer(is_eval=False):
    args = get_params()
    args.eval = is_eval
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.task not in ("", "B1Z1PickMulti"):
        raise ValueError("The visual teacher currently supports --task B1Z1PickMulti only")
    args.task = "B1Z1PickMulti"
    if any((args.use_tanh, args.no_feature, args.pitch_control, args.fixed_base,
            args.depth_random, args.front_only, args.wrist_seg, args.seperate)):
        raise ValueError("Use the standard 9-action teacher, full cameras, object features, and no --use_tanh/--depth_random")
    if args.teacher_ckpt_path:
        raise ValueError("Use --teacher_init_checkpoint here; --teacher_ckpt_path belongs to student distillation")
    if args.record_video:
        raise ValueError("Video recording is not wired into this entrypoint; use evaluation with the viewer")
    args.experiment_dir = str(Path(args.experiment_dir).expanduser().resolve())
    if args.resume and not args.checkpoint:
        args.checkpoint = _latest_checkpoint(Path(args.experiment_dir) / args.wandb_name / "checkpoints")
    if args.teacher_init_checkpoint and (args.checkpoint or is_eval):
        raise ValueError("Use --teacher_init_checkpoint for new training, or --checkpoint for restore/evaluation")
    source_path = args.checkpoint or args.teacher_init_checkpoint
    if not source_path:
        raise ValueError("A teacher checkpoint is required: --teacher_init_checkpoint for a new run, --checkpoint to restore")
    source_path = str(Path(source_path).expanduser().resolve())
    if not Path(source_path).is_file():
        raise FileNotFoundError(source_path)
    config_path = (Path(args.vision_config).expanduser().resolve() if args.vision_config
                   else HIGH_LEVEL_DIR / "data/cfg/DQ_teacher_vision.yaml")

    # Isaac Gym must be imported before torch. Keep this below argument parsing
    # so --help does not require the simulator or a graphics device.
    from isaacgym import gymapi  # noqa: F401
    import torch
    from learning.teacher_vision_trainer import TeacherVisionTrainer
    from skrl.utils import set_seed
    from train_multistate_DQ_teacher import create_env, get_predict_point
    from utils.teacher_vision_wrapper import TeacherVisionWrapper
    from utils.teacher_vision_training import (
        legacy_step, make_agent, initialize_from_teacher,
        TeacherVisionTrainingState, restore_experiment,
    )

    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if args.checkpoint:
        saved = checkpoint.get("teacher_vision_state")
        if saved is None:
            raise ValueError("Legacy teacher checkpoint: start a new run with --teacher_init_checkpoint")
        if args.vision_config:
            raise ValueError("Restoring uses the configuration stored in the checkpoint, not --vision_config")
        cfg = deepcopy(saved["experiment_config"])
        if not _explicit_option("seed"):
            args.seed = int(cfg["teacher_vision"].get("seed", args.seed))
        for key, value in saved["environment_options"].items():
            if _explicit_option(key) and getattr(args, key) != value:
                raise ValueError("Environment option --%s conflicts with the saved experiment" % key)
            setattr(args, key, value)
        completed_steps = int(saved["completed_steps"])
        origin_step = int(saved["origin_step"])
        global_step = int(saved["global_step"])
        saved_mode = cfg["teacher_vision"]["vision_mode"]
        mode = args.vision_mode or saved_mode
        if mode != saved_mode and not (is_eval and {mode, saved_mode} <= {"images", "zero"}):
            raise ValueError("Restore the saved vision_mode; images/zero may only be switched for evaluation ablations")
    else:
        cfg = load_cfg(str(config_path))
        completed_steps = 0
        origin_step = global_step = legacy_step(source_path, args.teacher_initial_step)
        mode = args.vision_mode or cfg["teacher_vision"]["vision_mode"]
    if not is_eval and args.timesteps <= completed_steps:
        raise ValueError("--timesteps is the total additional-step target of this experiment and must exceed completed_steps")
    if not args.roboinfo or not args.observe_gait_commands:
        raise ValueError("The 1276/61-dimensional setup requires --roboinfo --observe_gait_commands")
    if cfg["env"].get("floatingBase", False):
        raise ValueError("Floating-base configurations are not supported by this experiment")
    if args.debug and args.num_envs is None:
        args.num_envs = 4
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError("--num_envs must be positive")
        cfg["env"]["numEnvs"] = args.num_envs
    if args.object_name is not None:
        assets = cfg["env"]["asset"]["asset_multi"]
        if args.object_name not in assets:
            raise ValueError("Unknown --object_name: " + args.object_name)
        cfg["env"]["asset"]["asset_multi"] = {args.object_name: assets[args.object_name]}
    if args.vision_task_level:
        cfg["env"]["D1_bench_task_level"] = args.vision_task_level
    cfg["enableCameraSensors"] = True
    cfg["sensor"]["enableCamera"] = True
    cfg["env"]["refreshCameraOnReset"] = True
    cfg["env"]["globalStepCounter"] = global_step
    cfg["env"]["useTanh"] = False
    cfg["env"]["lastCommands"] = False
    cfg["env"]["near_goal_stop"] = args.near_goal_stop
    cfg["env"]["obj_move_prob"] = args.obj_move_prob
    args.wandb = args.wandb and not (is_eval or args.debug)
    cfg["env"]["wandb"] = args.wandb
    cfg["teacher_vision"]["vision_mode"] = mode
    cfg["teacher_vision"]["seed"] = args.seed
    settings = cfg["teacher_vision"]
    if args.vision_minibatch_size is not None:
        settings["minibatch_size"] = args.vision_minibatch_size
    if args.vision_learning_epochs is not None:
        settings["learning_epochs"] = args.vision_learning_epochs
    if args.debug and args.vision_minibatch_size is None:
        settings["minibatch_size"] = min(settings["minibatch_size"], cfg["env"]["numEnvs"] * settings["rollouts"])
    if not is_eval:
        sample_count = int(cfg["env"]["numEnvs"]) * int(settings["rollouts"])
        minibatch_size = int(settings["minibatch_size"])
        if minibatch_size <= 0 or sample_count % minibatch_size:
            raise ValueError("num_envs * rollouts must be divisible by --vision_minibatch_size")
    if args.graphics_device_id < 0:
        args.graphics_device_id = int(args.sim_device.split(":")[-1]) if ":" in args.sim_device else 0
    for camera in ("onboard_camera", "wrist_camera"):
        if cfg["sensor"][camera]["resolution"] != [96, 54]:
            raise ValueError("The reused visual encoder requires 96x54 cameras")
    if cfg["sensor"]["resized_resolution"] != [96, 54]:
        raise ValueError("The reused visual encoder requires resized_resolution [96, 54]")
    horizon = int(settings["reward_schedule_steps"])
    if horizon <= 0:
        raise ValueError("teacher_vision.reward_schedule_steps must be positive")
    # Legacy environment assets/transforms use paths relative to DQ_high-level.
    os.chdir(HIGH_LEVEL_DIR)
    # The imported legacy training module seeds at import; override it here.
    set_seed(args.seed)
    object_indices = [obj["dict_idx"] for obj in cfg["env"]["asset"]["asset_multi"].values()]
    args.intervel = 23
    camera, grasps, cubes = get_predict_point(
        cube_predict_info_path="contact_grasp_info_mul", cube_root_states_info_path="30all_nomove_cube_root_states5.pt",
        num_env=cfg["env"]["numEnvs"], intervel=args.intervel, delta_height=0.1, object_indices=object_indices,
    )
    # Snapshot before Isaac Gym adds tensors/runtime settings to cfg.
    experiment_cfg = deepcopy(cfg)
    raw_env = create_env(cfg, args, camera, grasps, cubes)._env
    # B1Z1PickMulti otherwise derives these from the new run's CLI timesteps,
    # which would restart/change the teacher reward curriculum during warm start.
    raw_env.total_timesteps = horizon
    raw_env.train_reward_strict = horizon / 2
    env = TeacherVisionWrapper(raw_env)
    agent = make_agent(env, settings, mode, args.experiment_dir, args.wandb_name,
                       deterministic=is_eval, wandb=args.wandb, wandb_project=args.wandb_project)
    options = {key: getattr(args, key) for key in ENVIRONMENT_OPTIONS}
    agent.checkpoint_modules["teacher_vision_state"] = TeacherVisionTrainingState(
        env, experiment_cfg, options, origin_step,
    )
    if args.checkpoint:
        restore_experiment(agent, checkpoint, evaluation=is_eval)
    else:
        initialize_from_teacher(agent, checkpoint)
    trainer = TeacherVisionTrainer(env=env, agents=agent, cfg={
        "timesteps": args.timesteps,
        "initial_timestep": 0 if is_eval else completed_steps,
        "headless": args.headless,
    })
    if not is_eval:
        copy_cfg(str(config_path), agent.experiment_dir, cfg=experiment_cfg)
    # Visual/non-visual model initialization consumes different random draws.
    # Reset the sampling stream so the first environment reset is comparable.
    set_seed(args.seed)
    print("Visual teacher: mode=%s, envs=%d, global_step=%d, completed_extra_steps=%d, seed=%d" %
          (mode, env.num_envs, global_step, completed_steps, args.seed))
    return trainer


if __name__ == "__main__":
    trainer = get_trainer()
    trainer.train()
    # Also save short runs that end before the periodic checkpoint interval.
    output = Path(trainer.agents.experiment_dir) / "checkpoints" / ("agent_%d.pt" % trainer.timesteps)
    output.parent.mkdir(parents=True, exist_ok=True)
    trainer.agents.save(str(output))
    print("Saved visual teacher:", output)
