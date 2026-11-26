import os
import json
import yaml
import argparse
import math
from copy import deepcopy
from pathlib import Path


SGS_DEFAULTS = {
    "mode": "disabled",
    "bank_path": "",
    "bank_size": None,
    "history_size": 100,
    "target_success": 0.5,
    "kappa": 1.0,
    "temperature": 2.0,
    "epsilon": 1e-4,
    "p_init": 0.0,
}

TSIL_DEFAULTS = {
    "enabled": False,
    "top_k": 5,
    "samples_per_trajectory": 32,
    "coefficient": 0.05,
    "value_coefficient": 0.05,
    "micro_batch_size": 2048,
    "screen_batch_size": 4096,
}


def apply_tsil_config(cfg, args, is_eval=False):
    """One complete temporal method; chunk sizes change memory use, not sampling."""
    settings = dict(TSIL_DEFAULTS)
    settings.update(cfg["env"].get("tsil", {}))
    for key, argument in (("enabled", "tsil"), ("micro_batch_size", "tsil_micro_batch_size"),
                          ("screen_batch_size", "tsil_screen_batch_size")):
        value = getattr(args, argument, None)
        if value is not None:
            settings[key] = value
    if type(settings["enabled"]) is not bool:
        raise ValueError("env.tsil.enabled must be Boolean")
    for name in ("top_k", "samples_per_trajectory", "micro_batch_size", "screen_batch_size"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"env.tsil.{name} must be a positive integer")
    for name in ("coefficient", "value_coefficient"):
        if not math.isfinite(float(settings[name])) or float(settings[name]) <= 0:
            raise ValueError(f"env.tsil.{name} must be finite and positive")
    if settings["enabled"]:
        if args.resume or (args.checkpoint and not is_eval):
            raise ValueError("TSIL training starts from scratch; checkpoints support evaluation, not archive/sampler resume")
        if not is_eval and cfg["env"]["sgs"]["mode"] == "disabled":
            raise ValueError("TSIL teacher training requires SGS configuration identities (--sgs_mode sgs or uniform)")
        if args.task != "B1Z1PickMulti":
            raise ValueError("TSIL is implemented for the B1Z1PickMulti teacher")
        if cfg.get("reward", {}).get("only_positive_rewards", False):
            raise ValueError("TSIL requires signed task rewards (only_positive_rewards=False)")
    cfg["env"]["tsil"] = settings
    return settings


def configure_teacher_experiment(cfg, args, is_eval=False, strict_eval=False):
    """Apply entrypoint overrides without changing the planned reward horizon."""
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive")
    if args.stop_after is not None and args.stop_after <= 0:
        raise ValueError("--stop_after must be positive")
    env_cfg = cfg["env"]
    apply_actor_history_config(cfg, args)
    sgs = dict(SGS_DEFAULTS)
    sgs.update(env_cfg.get("sgs", {}))
    if args.sgs_mode is not None:
        sgs["mode"] = args.sgs_mode
    if args.sgs_bank is not None:
        sgs["bank_path"] = args.sgs_bank
    if sgs["mode"] not in ("disabled", "uniform", "sgs"):
        raise ValueError("env.sgs.mode must be disabled, uniform, or sgs")
    if sgs["bank_size"] is not None and (type(sgs["bank_size"]) is not int or sgs["bank_size"] <= 0):
        raise ValueError("env.sgs.bank_size must be null (automatic) or a positive integer")
    sgs["seed"] = args.seed
    env_cfg["sgs"] = sgs
    cfg["seed"] = args.seed
    success_mode = args.success_mode or env_cfg.get("successMode", "hold")
    if success_mode not in ("legacy", "hold"):
        raise ValueError("successMode must be legacy or hold")
    # Keep old commands/configurations readable while using one success rule
    # for ordinary resets, bank resets and evaluation.
    env_cfg["successMode"] = "hold"
    env_cfg.setdefault("holdSteps", 10)
    if type(env_cfg["holdSteps"]) is not int or env_cfg["holdSteps"] <= 0:
        raise ValueError("holdSteps must be a positive integer")

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
        env_cfg.update({"terrainType": "plane", "D1_bench_task_level": "Level02",
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
    apply_tsil_config(cfg, args, is_eval=is_eval)
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


def apply_pose_prediction_config(cfg, args):
    """Apply only explicit CLI overrides, preserving saved experiment settings."""
    settings = cfg["env"].setdefault("posePrediction", {})
    defaults = dict(enabled=False, horizons=[0.30, 0.60], history=5, qScale=1.0, rScale=0.01)
    for key, value in defaults.items():
        settings.setdefault(key, value)
    overrides = dict(enabled="future_pose_obs", horizons="future_pose_horizons",
                     history="ekf_history", qScale="ekf_q_scale", rScale="ekf_r_scale")
    for key, argument in overrides.items():
        value = getattr(args, argument, None)
        if value is not None:
            settings[key] = value
    return settings


def restore_student_teacher_observation_config(cfg, teacher_checkpoint):
    """Restore teacher-only input settings before applying explicit CLI overrides.

    Saved runs can use a custom YAML recorded in teacher_run.json. With no saved
    YAML, leave the student's existing defaults intact for standalone legacy
    checkpoints; never substitute the repository's current teacher defaults.
    """
    if not teacher_checkpoint:
        return None
    run_dir = Path(teacher_checkpoint).expanduser().resolve().parent.parent
    candidates = []
    metadata_path = run_dir / "teacher_run.json"
    if metadata_path.is_file():
        with metadata_path.open() as stream:
            metadata = json.load(stream)
        config_file = metadata.get("config_file")
        if config_file:
            candidates.append(run_dir / Path(config_file).name)
    candidates.append(run_dir / "DQ_teacher.yaml")
    config_path = next((path for path in candidates if path.is_file()), None)
    if config_path is None:
        return None
    with config_path.open() as stream:
        saved = yaml.safe_load(stream)
    saved_env = saved["env"]
    cfg["env"]["posePrediction"] = deepcopy(saved_env.get("posePrediction", {}))
    cfg["env"]["noVelocityObs"] = saved_env.get("noVelocityObs", False)
    if "actionScaling" in saved_env:
        # The student env must map teacher actions the way the teacher was trained.
        cfg["env"]["actionScaling"] = saved_env["actionScaling"]
    # ... and with its EE increment limits; a YAML without them used 0.02 m / 0.06 rad.
    for name, default in LEGACY_EE_DELTA_LIMITS.items():
        cfg["env"][name] = saved_env.get(name, default)
    return config_path


def apply_actor_history_config(cfg, args):
    frames = getattr(args, "actor_history_frames", None)
    if frames is None:
        frames = cfg["env"].get("actorHistoryFrames", 1)
    if type(frames) is not int or frames < 1:
        raise ValueError("actor_history_frames must be a positive integer (including the current frame)")
    cfg["env"]["actorHistoryFrames"] = frames
    return frames


ACTION_SCALING_MODES = ("legacy", "normalized")


def validate_action_scaling(env_cfg):
    """Return env.actionScaling; configurations without the key mean legacy.

    legacy:     the environment clips raw policy outputs at the physical limits.
    normalized: the environment clips outputs to [-1, 1], then scales them to
                the physical limits. It replaces the Tanh path, not combines with it.
    """
    mode = env_cfg.get("actionScaling", "legacy")
    if mode not in ACTION_SCALING_MODES:
        raise ValueError(f"env.actionScaling must be one of {ACTION_SCALING_MODES}, got {mode!r}")
    if mode == "normalized" and env_cfg.get("useTanh", False):
        raise ValueError("env.actionScaling=normalized cannot be combined with --use_tanh")
    return mode


def apply_action_scaling_config(cfg, args):
    """An explicit --action_scaling overrides the YAML; otherwise keep the YAML value."""
    mode = getattr(args, "action_scaling", None)
    if mode is not None:
        cfg["env"]["actionScaling"] = mode
    cfg["env"]["actionScaling"] = validate_action_scaling(cfg["env"])
    return cfg["env"]["actionScaling"]


# Per-step increments of the EE goal: meters per axis (actions 0-2) and radians
# per RPY channel (actions 3-5). A configuration or run without these keys
# predates them and used 0.02 m / 0.06 rad.
LEGACY_EE_DELTA_LIMITS = {"eeDeltaPositionLimit": 0.02, "eeDeltaRotationLimit": 0.06}


def ee_delta_limits(env_cfg):
    """(position m, rotation rad) per-step EE-goal increment limits of env_cfg."""
    return tuple(float(env_cfg.get(name, default)) for name, default in LEGACY_EE_DELTA_LIMITS.items())


def validate_ee_delta_limits(env_cfg):
    for name, default in LEGACY_EE_DELTA_LIMITS.items():
        value = env_cfg.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"env.{name} must be positive and finite")
    return ee_delta_limits(env_cfg)


def saved_ee_delta_limits(checkpoint):
    """The limits a checkpoint's run trained with, or None without teacher_run.json.

    A run whose metadata lacks them predates the setting: 0.02 m / 0.06 rad.
    """
    metadata_path = Path(checkpoint).resolve().parent.parent / "teacher_run.json"
    if not metadata_path.is_file():
        return None
    with metadata_path.open() as stream:
        saved = json.load(stream).get("ee_delta_limits", {})
    return {name: saved.get(name, default) for name, default in LEGACY_EE_DELTA_LIMITS.items()}


def apply_ee_delta_limits(cfg, checkpoint=None):
    """Evaluating a saved run replays its own limits, whatever YAML is passed;
    otherwise keep the YAML values. The result is stored back in cfg."""
    saved = saved_ee_delta_limits(checkpoint) if checkpoint else None
    if saved is not None:
        cfg["env"].update(saved)
    position, rotation = validate_ee_delta_limits(cfg["env"])
    cfg["env"]["eeDeltaPositionLimit"], cfg["env"]["eeDeltaRotationLimit"] = position, rotation
    return {"eeDeltaPositionLimit": position, "eeDeltaRotationLimit": rotation}


COMMAND_REWARD_SCHEDULES = ("step", "constant", "ramp")
APPROACH_SHAPING_MODES = ("record", "potential")
REWARD_OPTION_DEFAULTS = {
    "commandRewardSchedule": "step",     # step: off until total_timesteps / 2 (original)
    "commandRewardRampFraction": 0.2,    # ramp: reaches full weight at this training fraction
    "approachShaping": "record",         # record: reward only new closest distance / height
    "shapingGamma": 0.99,                # potential: must equal the PPO discount factor
}


def validate_reward_options(reward_cfg):
    """Return the T2 reward options with defaults; configurations without them mean the original rewards."""
    options = {name: reward_cfg.get(name, default) for name, default in REWARD_OPTION_DEFAULTS.items()}
    if options["commandRewardSchedule"] not in COMMAND_REWARD_SCHEDULES:
        raise ValueError(f"reward.commandRewardSchedule must be one of {COMMAND_REWARD_SCHEDULES}")
    ramp = options["commandRewardRampFraction"]
    if isinstance(ramp, bool) or not isinstance(ramp, (int, float)) or not 0 < ramp <= 1:
        raise ValueError("reward.commandRewardRampFraction must be in (0, 1]")
    if options["approachShaping"] not in APPROACH_SHAPING_MODES:
        raise ValueError(f"reward.approachShaping must be one of {APPROACH_SHAPING_MODES}")
    gamma = options["shapingGamma"]
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float)) or not 0 < gamma <= 1:
        raise ValueError("reward.shapingGamma must be in (0, 1]")
    if options["approachShaping"] == "potential" and reward_cfg.get("only_positive_rewards", False):
        # Clipping negative rewards would break the telescoping potential difference.
        raise ValueError("reward.approachShaping=potential requires only_positive_rewards=False")
    return options


def apply_reward_options(cfg, args):
    """Explicit CLI reward options override the YAML; the result is stored back in cfg."""
    overrides = {"commandRewardSchedule": getattr(args, "command_reward_schedule", None),
                 "approachShaping": getattr(args, "approach_shaping", None)}
    for name, value in overrides.items():
        if value is not None:
            cfg["reward"][name] = value
    cfg["reward"].update(validate_reward_options(cfg["reward"]))
    return {name: cfg["reward"][name] for name in REWARD_OPTION_DEFAULTS}


MARGIN_CURRICULUM_MODES = ("disabled", "prior", "frontier")
# intercept: Phi = 1 - tau*/max_time (earliest interception time). slack: the first
# design, Phi from the clipped margin M; it failed (ours_s44) and stays as an ablation.
MARGIN_POTENTIALS = ("intercept", "slack")
MARGIN_DEFAULTS = {
    "reachModel": "",            # trained T file, "analytic" (B3), "new:mlp"/"new:quasimetric" (online), "" = off
    "horizonStep": 0.3,          # seconds between margin horizons; the first horizon is 0
    "horizonCount": 16,          # 0.0, 0.3, ..., 4.5 s
    "clip": 2.0,                 # seconds; clips the token margins and the slack potential
    "chunkEnvs": 1024,           # environments per T evaluation chunk (GPU memory)
    "candidates": 30,            # grasp candidates per object (the first N of PredictPoint's)
    "reward": False,             # potential-based shaping, weight reward.scales.margin_shaping
    "potential": "intercept",    # intercept | slack (see MARGIN_POTENTIALS)
    "keepTracking": True,        # with reward: keep approaching/base_approaching (False: replace them)
    "tokens": False,             # actor: per-horizon best-grasp tokens from world-frame EKF forecasts
    "curriculum": "disabled",    # disabled | prior | frontier (dynamic SGS; requires SGS)
    "priorStrength": 4.0,        # pseudo-episodes of the M0 prior in the SGS success estimate
    "priorTemperature": 0.5,     # seconds
    "deltaStart": 0.5,           # seconds; delta decreases linearly over training ...
    "deltaEnd": 0.0,             # ... to this value
    "frontierProbability": 0.25, # frontier: fraction of full_task resets offered a faster table
    "frontierMaxSpeedScale": 1.6,
    "frontierCandidates": 8,
    "online": False,             # refine T during training from the policy's own rollouts
    "onlineInterval": 24,        # environment steps between updates (one PPO rollout)
    "onlineUpdates": 8,
    "onlineBatch": 8192,
    "onlineLearningRate": 3.0e-4,
    "onlineEmaDecay": 0.99,
    "onlineFreezeFraction": 0.5, # stop updating T after this training fraction (fixed potential after)
    "window": 48,                # hindsight window, environment steps
    "maxPairSteps": 32,          # longest hindsight pair, environment steps
    "expectile": 0.1,
}
# Phi changes by about 0.15 s / 8 s = 0.02 per step of optimal progress, the size
# of the approaching reward. Weight 10 (ours_s44) dominated the return and the critic collapsed.
MARGIN_REWARD_SCALE_DEFAULT = 1.0


def validate_margin_config(env_cfg, reward_cfg=None):
    """Return env.margin with defaults; a missing section means main line A is off."""
    settings = dict(MARGIN_DEFAULTS)
    settings.update(env_cfg.get("margin", {}) or {})
    unknown = set(settings) - set(MARGIN_DEFAULTS)
    if unknown:
        raise ValueError("unknown env.margin keys: " + ", ".join(sorted(unknown)))
    model = settings["reachModel"]
    if not isinstance(model, str):
        raise ValueError("env.margin.reachModel must be a string")
    for name in ("reward", "keepTracking", "tokens", "online"):
        if type(settings[name]) is not bool:
            raise ValueError(f"env.margin.{name} must be a boolean")
    if settings["curriculum"] not in MARGIN_CURRICULUM_MODES:
        raise ValueError(f"env.margin.curriculum must be one of {MARGIN_CURRICULUM_MODES}")
    if settings["potential"] not in MARGIN_POTENTIALS:
        raise ValueError(f"env.margin.potential must be one of {MARGIN_POTENTIALS}")
    for name in ("horizonStep", "clip", "priorTemperature", "frontierMaxSpeedScale",
                 "onlineLearningRate"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"env.margin.{name} must be positive and finite")
    for name in ("priorStrength", "deltaStart", "deltaEnd"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"env.margin.{name} must be finite")
    if settings["priorStrength"] < 0:
        raise ValueError("env.margin.priorStrength must be nonnegative")
    for name in ("horizonCount", "chunkEnvs", "candidates", "frontierCandidates", "onlineInterval",
                 "onlineUpdates", "onlineBatch", "window", "maxPairSteps"):
        if type(settings[name]) is not int or settings[name] < 1:
            raise ValueError(f"env.margin.{name} must be a positive integer")
    if settings["candidates"] > 30:
        raise ValueError("env.margin.candidates must not exceed the 30 PredictPoint candidates")
    if settings["maxPairSteps"] >= settings["window"]:
        raise ValueError("env.margin.maxPairSteps must be smaller than env.margin.window")
    for name in ("frontierProbability", "onlineFreezeFraction"):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"env.margin.{name} must be in [0, 1]")
    if not 0 < settings["expectile"] < 1 or not 0 <= settings["onlineEmaDecay"] < 1:
        raise ValueError("env.margin.expectile must be in (0, 1) and onlineEmaDecay in [0, 1)")
    if settings["frontierMaxSpeedScale"] < 1:
        raise ValueError("env.margin.frontierMaxSpeedScale must be at least 1")
    enabled = bool(model)
    uses = [name for name in ("reward", "tokens", "online") if settings[name]]
    if settings["curriculum"] != "disabled":
        uses.append("curriculum")
    if uses and not enabled:
        raise ValueError("env.margin." + "/".join(uses) + " requires env.margin.reachModel")
    if model.startswith("new:"):
        if model not in ("new:mlp", "new:quasimetric"):
            raise ValueError("env.margin.reachModel new:<kind> supports new:mlp and new:quasimetric")
        if not settings["online"]:
            raise ValueError("an untrained reach model (new:...) requires env.margin.online")
    if model == "analytic" and settings["online"]:
        raise ValueError("the analytic reach model cannot be trained online")
    scale = (reward_cfg or {}).get("scales", {}).get("margin_shaping", 0.0)
    if settings["reward"] != bool(scale):
        raise ValueError("env.margin.reward and a nonzero reward.scales.margin_shaping go together")
    if settings["reward"] and (reward_cfg or {}).get("only_positive_rewards", False):
        raise ValueError("margin shaping requires only_positive_rewards=False")
    return settings


def apply_margin_config(cfg, args):
    """Explicit CLI flags override env.margin; the result is stored back in cfg.

    --margin_reward also sets reward.scales.margin_shaping (default 1.0); with
    --no_margin_keep_tracking it zeroes approaching/base_approaching.
    """
    settings = dict(MARGIN_DEFAULTS)
    settings.update(cfg["env"].get("margin", {}) or {})
    scales = cfg["reward"].setdefault("scales", {})
    model = getattr(args, "reach_model", None)
    if model is not None:
        if model not in ("", "analytic") and not model.startswith("new:"):
            path = Path(model).expanduser()
            if not path.is_file():
                raise ValueError(f"--reach_model file not found: {model}")
            model = str(path.resolve())
        settings["reachModel"] = model
    for key, argument in (("tokens", "margin_tokens"), ("online", "reach_online"),
                          ("curriculum", "margin_curriculum"), ("keepTracking", "margin_keep_tracking"),
                          ("potential", "margin_potential")):
        value = getattr(args, argument, None)
        if value is not None:
            settings[key] = value
    if getattr(args, "margin_reward", None) is not None:
        settings["reward"] = args.margin_reward
        if args.margin_reward:
            scale = getattr(args, "margin_reward_scale", None)
            scales["margin_shaping"] = MARGIN_REWARD_SCALE_DEFAULT if scale is None else scale
        else:
            scales["margin_shaping"] = 0.0
    elif getattr(args, "margin_reward_scale", None) is not None:
        scales["margin_shaping"] = args.margin_reward_scale
    scales.setdefault("margin_shaping", 0.0)
    if settings["reward"] and not settings["keepTracking"]:
        # Ablation: margin shaping as the only approach signal (doc section 2).
        scales["approaching"] = 0.0
        scales["base_approaching"] = 0.0
    cfg["env"]["margin"] = settings
    validate_margin_config(cfg["env"], cfg["reward"])
    return settings


def get_parser():
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
    temporal = parser.add_mutually_exclusive_group()
    temporal.add_argument("--tsil", dest="tsil", action="store_true",
                          help="Enable the complete temporal self-imitation teacher method.")
    temporal.add_argument("--no_tsil", dest="tsil", action="store_false")
    parser.set_defaults(tsil=None)
    parser.add_argument("--tsil_micro_batch_size", type=int, default=None,
                        help="Replay GPU chunk size; the full logical batch and optimizer count are unchanged.")
    parser.add_argument("--tsil_screen_batch_size", type=int, default=None,
                        help="Critic screening chunk size; every archived configuration is still visited.")
    parser.add_argument("--success_mode", choices=("legacy", "hold"), default=None,
                        help="Compatibility option: legacy and hold use the same 633_test_pro success rule.")
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
    parser.add_argument("--actor_history_frames", type=int, default=None,
                        help="Actor frames including current: 3 means [t-2, t-1, t]. Defaults to saved YAML or 1.")
    parser.add_argument("--no_velocity_obs", action="store_true",
                        help="Remove robot local linear velocity (3D) and object-relative planar velocity (2D) from B1Z1PickMulti teacher observations.")
    # Not a mutually exclusive group: the last flag wins, so launcher defaults
    # can be overridden by appending the opposite flag.
    parser.add_argument("--critic_time_obs", dest="critic_time_obs", action="store_true",
                        help="Add episode progress (step / maxEpisodeLength) to the teacher critic input only; the actor ignores it.")
    parser.add_argument("--no_critic_time_obs", dest="critic_time_obs", action="store_false",
                        help="Disable the critic episode-progress input, e.g. for an A/B baseline.")
    parser.set_defaults(critic_time_obs=None)
    parser.add_argument("--critic_reward_state", dest="critic_reward_state", action="store_true",
                        help="Give the critic only the hidden variables the rewards depend on: closest "
                             "distance, highest lift, hold counters, lifted flag, command-reward weight.")
    parser.add_argument("--no_critic_reward_state", dest="critic_reward_state", action="store_false")
    parser.set_defaults(critic_reward_state=None)
    parser.add_argument("--approach_shaping", choices=APPROACH_SHAPING_MODES, default=None,
                        help="record: original record-breaking approach/lift rewards. potential: "
                             "potential-based shaping gamma*Phi(s')-Phi(s) on EE distance and lift height.")
    parser.add_argument("--command_reward_schedule", choices=COMMAND_REWARD_SCHEDULES, default=None,
                        help="step: command_reward starts at total_timesteps/2 (original). constant: "
                             "from the first step. ramp: linear over the first 20%% of training.")
    parser.add_argument("--value_clip", dest="value_clip", action="store_true",
                        help="PPO-style clipped value loss (the current default).")
    parser.add_argument("--no_value_clip", dest="value_clip", action="store_false",
                        help="Unclipped value loss, e.g. for the Andrychowicz et al. 2021 ablation.")
    parser.set_defaults(value_clip=None)
    parser.add_argument("--gae_lambda", type=float, default=None,
                        help="GAE lambda (default 0.95).")
    # Main line A: learned time-to-reach T and interception margin M.
    parser.add_argument("--reach_model", type=str, default=None,
                        help="Time-to-reach model: a file from train_reachability.py, 'analytic' "
                             "(baseline B3), or new:mlp / new:quasimetric with --reach_online.")
    parser.add_argument("--margin_reward", dest="margin_reward", action="store_true",
                        help="Potential-based shaping from the earliest interception time, added to "
                             "approaching and base_approaching.")
    parser.add_argument("--no_margin_reward", dest="margin_reward", action="store_false")
    parser.set_defaults(margin_reward=None)
    parser.add_argument("--margin_reward_scale", type=float, default=None,
                        help="reward.scales.margin_shaping with --margin_reward (default 1.0).")
    parser.add_argument("--margin_potential", choices=MARGIN_POTENTIALS, default=None,
                        help="intercept (default): Phi = 1 - tau*/8 s. slack: Phi from the clipped "
                             "margin M (ablation; it saturates near the object).")
    parser.add_argument("--margin_keep_tracking", dest="margin_keep_tracking", action="store_true",
                        help="Keep approaching/base_approaching with margin shaping (the default).")
    parser.add_argument("--no_margin_keep_tracking", dest="margin_keep_tracking", action="store_false",
                        help="Ablation: zero approaching/base_approaching with margin shaping.")
    parser.set_defaults(margin_keep_tracking=None)
    parser.add_argument("--margin_tokens", dest="margin_tokens", action="store_true",
                        help="Actor and critic GFM attend to per-horizon best-grasp tokens with margins "
                             "(world-frame EKF forecasts, real-obtainable with odometry).")
    parser.add_argument("--no_margin_tokens", dest="margin_tokens", action="store_false")
    parser.set_defaults(margin_tokens=None)
    parser.add_argument("--margin_curriculum", choices=MARGIN_CURRICULUM_MODES, default=None,
                        help="prior: M0 feasibility prior in SGS sampling. frontier: prior plus "
                             "faster-table variants of full_task resets at the M0 frontier.")
    parser.add_argument("--reach_online", dest="reach_online", action="store_true",
                        help="Refine T during training from hindsight pairs (EMA target, frozen later).")
    parser.add_argument("--no_reach_online", dest="reach_online", action="store_false")
    parser.set_defaults(reach_online=None)
    parser.add_argument("--action_scaling", choices=ACTION_SCALING_MODES, default=None,
                        help="legacy: clip raw policy outputs at the physical limits. "
                             "normalized: clip outputs to [-1, 1], then scale to the limits "
                             "(policy log-std starts at log 0.5). Defaults to the saved YAML or legacy.")
    prediction = parser.add_mutually_exclusive_group()
    prediction.add_argument("--future_pose_obs", dest="future_pose_obs", action="store_true",
                            help="Add EKF future object poses to teacher observations (default horizons: 0.30/0.60 seconds).")
    prediction.add_argument("--no_future_pose_obs", dest="future_pose_obs", action="store_false",
                            help="Disable future pose observations, e.g. for an original teacher checkpoint.")
    parser.set_defaults(future_pose_obs=None)
    parser.add_argument("--future_pose_horizons", type=float, nargs="+", default=None, metavar="SECONDS")
    parser.add_argument("--ekf_history", type=int, default=None, help="Rolling history length (default: 5 poses).")
    parser.add_argument("--ekf_q_scale", type=float, default=None)
    parser.add_argument("--ekf_r_scale", type=float, default=None)
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
    
    return parser


def get_params(argv=None):
    return get_parser().parse_args(argv)
