"""Train a directly deployable asymmetric actor with M1/M2 optional features."""
from copy import deepcopy
from pathlib import Path
import os

from utils.config import get_params, load_cfg, copy_cfg
from train_multistate_DQ_teacher_vision import ENVIRONMENT_OPTIONS, _explicit_option, _latest_checkpoint
from utils import asymmetric_distributed as distributed


HIGH_LEVEL_DIR = Path(__file__).resolve().parent
M0_ENVIRONMENT_OPTIONS = ENVIRONMENT_OPTIONS + ("pred_success",)


def configure_experiment(args, checkpoint, config_path, is_eval=False):
    """Resolve provenance before simulator construction; never guess visual steps."""
    from utils.asymmetric_teacher_training import validate_settings
    from utils.teacher_vision_training import legacy_step

    checkpoint = checkpoint or {}
    if is_eval and not args.checkpoint:
        raise ValueError("Evaluation requires --checkpoint from a trained asymmetric experiment")
    if not args.teacher_init_checkpoint and args.teacher_initial_step is not None:
        raise ValueError("--teacher_initial_step requires --teacher_init_checkpoint; scratch training starts at step 0")
    if not args.teacher_init_checkpoint and args.asymmetric_allow_legacy:
        raise ValueError("--asymmetric_allow_legacy requires --teacher_init_checkpoint")
    saved = checkpoint.get("asymmetric_teacher_state") if args.checkpoint else None
    source = checkpoint.get("teacher_vision_state") or (checkpoint.get("asymmetric_teacher_state") if not args.checkpoint else None)
    if args.checkpoint:
        if saved is None:
            raise ValueError("Use --teacher_init_checkpoint to migrate a visual teacher")
        if saved.get("version") != 2:
            raise ValueError("Older checkpoints require --teacher_init_checkpoint for weights initialization, not --checkpoint")
        if args.asymmetric_config or args.vision_config:
            raise ValueError("Resume/evaluation uses the configuration stored in the asymmetric checkpoint")
        cfg = deepcopy(saved["experiment_config"])
        origin, completed, global_step = (int(saved[key]) for key in ("origin_step", "completed_steps", "global_step"))
        options = saved["environment_options"]
        for key, value in options.items():
            if _explicit_option(key) and getattr(args, key) != value:
                raise ValueError("Environment option --%s conflicts with the saved experiment" % key)
            setattr(args, key, value)
        if not _explicit_option("seed"):
            args.seed = int(cfg["asymmetric_teacher"]["seed"])
    else:
        cfg = load_cfg(str(config_path))
        completed = 0
        if source:
            if not args.asymmetric_config:
                settings = deepcopy(cfg["asymmetric_teacher"])
                cfg = deepcopy(source["experiment_config"])
                old_settings = cfg.get("teacher_vision", cfg.get("asymmetric_teacher", {}))
                settings["reward_schedule_steps"] = old_settings["reward_schedule_steps"]
                cfg.pop("teacher_vision", None)
                cfg["asymmetric_teacher"] = settings
            origin = global_step = int(source["global_step"])
            if args.teacher_initial_step is not None and args.teacher_initial_step != origin:
                raise ValueError("--teacher_initial_step conflicts with visual-teacher metadata")
            for key, value in source["environment_options"].items():
                if not _explicit_option(key):
                    setattr(args, key, value)
        elif args.teacher_init_checkpoint:
            origin = global_step = legacy_step(args.teacher_init_checkpoint, args.teacher_initial_step)
        else:
            origin = global_step = 0

    settings = cfg["asymmetric_teacher"]
    process_count = distributed.launch_world_size()
    if saved:
        if not is_eval and int(settings.get("world_size", 1)) != process_count:
            raise ValueError("Resume must retain world_size; use --teacher_init_checkpoint to start a new experiment with a different GPU count")
    else:
        settings["world_size"] = process_count
    overrides = {
        "vision_minibatch_size": "minibatch_size", "vision_learning_epochs": "learning_epochs",
        "asymmetric_eval_interval": "eval_interval", "asymmetric_eval_steps": "eval_steps",
    }
    for flag, key in overrides.items():
        value = getattr(args, flag)
        if value is not None:
            if saved and value != settings[key]:
                raise ValueError("Cannot change --%s when restoring an experiment" % flag)
            settings[key] = value
    if args.asymmetric_mode:
        if saved and args.asymmetric_mode != settings["mode"]:
            raise ValueError("Cannot change saved network mode on resume; initialize a new experiment instead")
        settings["mode"] = args.asymmetric_mode
    if not saved:
        settings["seed"] = args.seed
        settings["initialization"] = "teacher" if args.teacher_init_checkpoint else "scratch"
        # Source-free training learns observation statistics between rollouts.
        # A scratch-trained source also carries identity value units; retain
        # that normalization mode when its weights seed another experiment.
        source_settings = (source or {}).get("experiment_config", {}).get("asymmetric_teacher", {})
        settings["normalization"] = (source_settings.get("normalization", "frozen")
                                     if args.teacher_init_checkpoint else "rollout")
    elif not is_eval and args.seed != settings["seed"]:
        raise ValueError("Resume must retain the saved training seed")
    validate_settings(settings)
    if not _explicit_option("timesteps"):
        args.timesteps = 5000 if is_eval else int(settings["total_steps"])
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if not is_eval:
        if not completed < args.timesteps <= int(settings["total_steps"]):
            raise ValueError("--timesteps must exceed completed_steps and not exceed total_steps")
        if args.timesteps % int(settings["rollouts"]):
            raise ValueError("--timesteps must finish a complete PPO rollout")
    if not args.roboinfo or not args.observe_gait_commands:
        raise ValueError("The 1276/61 observation layout requires --roboinfo --observe_gait_commands")
    if cfg["env"].get("floatingBase", False):
        raise ValueError("Asymmetric training requires the standard non-floating robot")
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError("--num_envs must be positive")
        if saved and not is_eval and args.num_envs != cfg["env"]["numEnvs"]:
            raise ValueError("Resume must retain num_envs and the PPO sample budget")
        cfg["env"]["numEnvs"] = args.num_envs
    if args.object_name is not None:
        assets = cfg["env"]["asset"]["asset_multi"]
        if args.object_name not in assets:
            raise ValueError("Unknown object: " + args.object_name)
        if saved and not is_eval and set(assets) != {args.object_name}:
            raise ValueError("Resume must retain the saved object set")
        cfg["env"]["asset"]["asset_multi"] = {args.object_name: assets[args.object_name]}
    if args.vision_task_level:
        if saved and not is_eval and args.vision_task_level != cfg["env"]["D1_bench_task_level"]:
            raise ValueError("Resume must retain the saved task level")
        cfg["env"]["D1_bench_task_level"] = args.vision_task_level
    if not is_eval and int(cfg["env"]["numEnvs"]) * int(settings["rollouts"]) % int(settings["minibatch_size"]):
        raise ValueError("num_envs * rollouts must be divisible by minibatch_size")
    cfg["enableCameraSensors"] = cfg["sensor"]["enableCamera"] = True
    cfg["sensor"]["recordCameraGeometry"] = settings.get("mode", "m2") == "m2"
    cfg["env"].update({"refreshCameraOnReset": True, "terminateOnCameraConstraint": False,
                       "globalStepCounter": global_step, "useTanh": False, "lastCommands": False,
                       "near_goal_stop": args.near_goal_stop, "obj_move_prob": args.obj_move_prob})
    if any(cfg["sensor"][camera]["resolution"] != [96, 54] for camera in ("onboard_camera", "wrist_camera")):
        raise ValueError("Asymmetric training requires 96x54 cameras")
    if cfg["sensor"]["resized_resolution"] != [96, 54]:
        raise ValueError("Asymmetric training requires resized_resolution [96,54]")
    return cfg, origin, completed, global_step


def get_trainer(is_eval=False):
    args = get_params()
    args.eval = is_eval
    if args.task not in ("", "B1Z1PickMulti"):
        raise ValueError("Asymmetric training supports B1Z1PickMulti only")
    args.task = "B1Z1PickMulti"
    if any((args.use_tanh, args.no_feature, args.pitch_control, args.fixed_base,
            args.depth_random, args.front_only, args.wrist_seg, args.seperate, args.record_video)):
        raise ValueError("Asymmetric training requires standard 9 actions, full camera input, and no video recording")
    if args.vision_mode not in (None, "images"):
        raise ValueError("Asymmetric training uses real images; vision_mode zero/none are visual-teacher ablations")
    if args.vision_config or args.teacher_ckpt_path:
        raise ValueError("Use --asymmetric_config and --teacher_init_checkpoint for asymmetric training")
    args.experiment_dir = str(Path(args.experiment_dir).expanduser().resolve())
    distributed.configure_devices(args)
    if args.resume and not args.checkpoint:
        args.checkpoint = _latest_checkpoint(Path(args.experiment_dir) / args.wandb_name / "checkpoints")
    if args.teacher_init_checkpoint and (args.checkpoint or is_eval):
        raise ValueError("Choose --teacher_init_checkpoint for migration or --checkpoint for asymmetric restore")
    source_path = args.checkpoint or args.teacher_init_checkpoint
    if is_eval and not source_path:
        raise ValueError("Evaluation requires --checkpoint from a trained asymmetric experiment")
    if not args.teacher_init_checkpoint and args.teacher_initial_step is not None:
        raise ValueError("--teacher_initial_step requires --teacher_init_checkpoint; scratch training starts at step 0")
    if not args.teacher_init_checkpoint and args.asymmetric_allow_legacy:
        raise ValueError("--asymmetric_allow_legacy requires --teacher_init_checkpoint")
    if source_path:
        source_path = Path(source_path).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        args.teacher_init_checkpoint = str(source_path) if args.teacher_init_checkpoint else ""
    config_path = (Path(args.asymmetric_config).expanduser().resolve() if args.asymmetric_config
                   else HIGH_LEVEL_DIR / "data/cfg/DQ_asymmetric_teacher.yaml")

    # Isaac Gym must load before torch, but --help stays simulator-independent.
    from isaacgym import gymapi  # noqa: F401
    import torch
    distributed.initialize(args)
    from skrl.utils import set_seed
    from learning.asymmetric_teacher_trainer import AsymmetricTeacherTrainer
    from train_multistate_DQ_teacher import create_env, get_predict_point
    from utils.asymmetric_teacher_wrapper import AsymmetricTeacherWrapper
    from utils.asymmetric_teacher_training import (
        make_agent, AsymmetricTeacherTrainingState, initialize_from_teacher, restore_experiment,
    )

    checkpoint = torch.load(str(source_path), map_location="cpu", weights_only=False) if source_path else {}
    cfg, origin, completed, global_step = configure_experiment(args, checkpoint, config_path, is_eval)
    settings = cfg["asymmetric_teacher"]
    args.wandb = args.wandb and not (is_eval or args.debug) and distributed.is_main()
    cfg["env"]["wandb"] = args.wandb
    if args.graphics_device_id < 0:
        args.graphics_device_id = int(args.sim_device.split(":")[-1]) if ":" in args.sim_device else 0
    os.chdir(HIGH_LEVEL_DIR)
    runtime_seed = args.seed + distributed.rank()
    set_seed(runtime_seed)
    object_indices = [obj["dict_idx"] for obj in cfg["env"]["asset"]["asset_multi"].values()]
    args.intervel = 23
    camera, grasps, cubes = get_predict_point(
        cube_predict_info_path="contact_grasp_info_mul", cube_root_states_info_path="30all_nomove_cube_root_states5.pt",
        num_env=cfg["env"]["numEnvs"], intervel=args.intervel, delta_height=0.1, object_indices=object_indices,
    )
    experiment_cfg = deepcopy(cfg)
    raw = create_env(cfg, args, camera, grasps, cubes)._env
    # PickMulti consumes eval without forwarding it to Base, which otherwise
    # overwrites the flag with False. Keep the intended success protocol here.
    raw.eval = is_eval
    horizon = int(settings["reward_schedule_steps"])
    if horizon <= 0:
        raise ValueError("reward_schedule_steps must be positive")
    raw.total_timesteps, raw.train_reward_strict = horizon, horizon / 2
    env = AsymmetricTeacherWrapper(raw, settings)
    env.seed_runtime(runtime_seed)
    agent = make_agent(env, settings, args.experiment_dir, args.wandb_name,
                       deterministic=is_eval, wandb=args.wandb, wandb_project=args.wandb_project)
    options = {key: getattr(args, key) for key in M0_ENVIRONMENT_OPTIONS}
    agent.checkpoint_modules["asymmetric_teacher_state"] = AsymmetricTeacherTrainingState(
        env, experiment_cfg, options, origin,
    )
    if args.checkpoint:
        restore_experiment(agent, checkpoint, evaluation=is_eval)
    elif args.teacher_init_checkpoint:
        initialize_from_teacher(agent, checkpoint, allow_legacy=args.asymmetric_allow_legacy)
    distributed.synchronize_agent(agent)
    trainer = AsymmetricTeacherTrainer(env=env, agents=agent, cfg={
        "timesteps": args.timesteps, "initial_timestep": 0 if is_eval else completed,
        "headless": args.headless, "eval_interval": settings["eval_interval"],
        "eval_steps": settings["eval_steps"], "checkpoint_interval": settings["checkpoint_interval"],
        "eval_seed": 1729 + distributed.rank(),
        "disable_progressbar": not distributed.is_main(),
    })
    if not is_eval and distributed.is_main():
        copy_cfg(str(config_path), agent.experiment_dir, cfg=experiment_cfg)
    set_seed(runtime_seed)
    initialization = "resume" if args.checkpoint else settings["initialization"]
    print("Asymmetric %s: initialization=%s, normalization=%s, envs_per_rank=%d, origin=%d, completed=%d, target=%d" %
          (settings["mode"], initialization, settings.get("normalization", "frozen"),
           env.num_envs, origin, completed, args.timesteps))
    print("Rank %d/%d: sim=%s, policy=%s, graphics=%d, seed=%d, global_envs=%d" %
          (distributed.rank(), distributed.world_size(), args.sim_device, args.rl_device,
           args.graphics_device_id, runtime_seed, env.num_envs * distributed.world_size()))
    return trainer


def main():
    trainer = get_trainer()
    trainer.train()
    if not distributed.is_main():
        distributed.barrier()
        return
    output = Path(trainer.agents.experiment_dir) / "checkpoints" / ("agent_%d.pt" % trainer.timesteps)
    output.parent.mkdir(parents=True, exist_ok=True)
    trainer.agents.save(str(output))
    print("Saved asymmetric checkpoint:", output)
    from utils.asymmetric_teacher_preprocessor import export_deployment
    import torch
    destination = output.parent / ("actor_%d.pt" % trainer.timesteps)
    payload = export_deployment(trainer.agents.policy, trainer.agents.checkpoint_modules["state_preprocessor"])
    payload["metadata"]["perception_config"] = trainer.training_state.settings.get("perception", {})
    torch.save(payload, destination)
    print("Exported deployment Actor:", destination)
    distributed.barrier()


if __name__ == "__main__":
    try:
        main()
    finally:
        distributed.shutdown()
