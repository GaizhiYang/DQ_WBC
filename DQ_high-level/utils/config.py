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

# Scout & Relay (doc/SGS_SAPG_scout_relay_design.md). Defaults reproduce the
# single-policy SGS teacher exactly: one branch and no relay bank.
SAPG_DEFAULTS = {
    "branches": 1,               # M; branch 0 is the deployed leader, 1..M-1 are scouts
    "leader_fraction": 0.5,      # share of environments for the leader; scouts split the rest
    "latent_dim": 16,            # learned per-branch condition vector (actor and critic)
    "off_policy_coef": 1.0,      # SAPG lambda; 0 turns the IS channel off (ablation)
    "entropy_coef": 0.0,         # scout j adds entropy_coef * j * entropy; the leader none
    "mu_max": 0.0,               # optional IS-weight truncation; 0 = faithful (untruncated)
    "chunk_size": 16384,
    "scout_need_beta": 1.0,      # q_S ~ [(w(p_S)+eps)(1-p_L+eps)^beta]^(1/T); 0 = per-branch SGS
    "scout_target_success": None,  # None: leader's target; list: one value per scout (last repeats)
}

RELAY_DEFAULTS = {
    "enabled": False,
    "source": "scouts",          # scouts (method) / leader (self-relay baselines D and G)
    "trigger": "success",        # success (method) / failure (ablation)
    "capacity": 4096,
    "alpha_max": 0.3,            # cap on R's share of the leader's reset distribution
    "tau_need": 0.3,             # create chains only where leader success <= tau_need
    "max_chains_per_config": 2,
    "fractions": [0.2, 0.4, 0.6, 0.8],
    "snapshot_interval": 5,      # high-level steps between snapshots
    "sampling": "sgs",           # sgs (method) / uniform within R (ablation)
    "history_size": 50,
    "master_rate": 0.9,
    "master_count": 20,
    "useless_count": 40,
    "quarantine_steps": 3,
    "max_new_chains_per_reset": 32,
    "controller_state": False,   # ablation: also restore gait/history controller buffers
}


def _positive_int(settings, name, prefix):
    if type(settings[name]) is not int or settings[name] < 1:
        raise ValueError(f"env.{prefix}.{name} must be a positive integer")


def _unit_interval(settings, name, prefix, low_open=False, high_open=False):
    value = settings[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"env.{prefix}.{name} must be a finite number")
    if (value < 0 or value > 1 or (low_open and value == 0) or (high_open and value == 1)):
        raise ValueError(f"env.{prefix}.{name} is outside its allowed range in [0, 1]")


def apply_scout_relay_config(cfg, args, is_eval=False):
    """Merge YAML and CLI settings for SAPG branches and the relay bank."""
    env_cfg = cfg["env"]
    sapg = dict(SAPG_DEFAULTS)
    sapg.update(env_cfg.get("sapg", {}) or {})
    relay = dict(RELAY_DEFAULTS)
    relay.update(env_cfg.get("relay", {}) or {})
    for key, argument in (("branches", "sapg_branches"), ("leader_fraction", "sapg_leader_fraction"),
                          ("latent_dim", "sapg_latent_dim"), ("off_policy_coef", "sapg_off_policy_coef"),
                          ("entropy_coef", "sapg_entropy_coef"), ("mu_max", "sapg_mu_max"),
                          ("scout_need_beta", "scout_need_beta"),
                          ("scout_target_success", "scout_target_success")):
        value = getattr(args, argument, None)
        if value is not None:
            sapg[key] = value
    for key, argument in (("enabled", "relay"), ("source", "relay_source"),
                          ("trigger", "relay_trigger"), ("capacity", "relay_capacity"),
                          ("alpha_max", "relay_alpha_max"), ("tau_need", "relay_tau_need"),
                          ("max_chains_per_config", "relay_max_chains_per_config"),
                          ("fractions", "relay_fractions"),
                          ("snapshot_interval", "relay_snapshot_interval"),
                          ("sampling", "relay_sampling"), ("controller_state", "relay_controller_state")):
        value = getattr(args, argument, None)
        if value is not None:
            relay[key] = value

    for name in ("branches", "latent_dim", "chunk_size"):
        _positive_int(sapg, name, "sapg")
    _unit_interval(sapg, "leader_fraction", "sapg", low_open=True, high_open=True)
    for name in ("off_policy_coef", "entropy_coef", "mu_max", "scout_need_beta"):
        value = sapg[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"env.sapg.{name} must be a finite nonnegative number")
    if sapg["scout_target_success"] is not None:
        targets = sapg["scout_target_success"]
        if not isinstance(targets, (list, tuple)) or not targets:
            raise ValueError("env.sapg.scout_target_success must be null or a nonempty list")
        sapg["scout_target_success"] = [float(value) for value in targets]
        if any(not 0 <= value <= 1 for value in sapg["scout_target_success"]):
            raise ValueError("env.sapg.scout_target_success values must be in [0, 1]")

    if type(relay["enabled"]) is not bool or type(relay["controller_state"]) is not bool:
        raise ValueError("env.relay.enabled and controller_state must be Boolean")
    if relay["source"] not in ("scouts", "leader"):
        raise ValueError("env.relay.source must be scouts or leader")
    if relay["trigger"] not in ("success", "failure"):
        raise ValueError("env.relay.trigger must be success or failure")
    if relay["sampling"] not in ("sgs", "uniform"):
        raise ValueError("env.relay.sampling must be sgs or uniform")
    for name in ("capacity", "max_chains_per_config", "snapshot_interval", "history_size",
                 "master_count", "useless_count", "max_new_chains_per_reset"):
        _positive_int(relay, name, "relay")
    if type(relay["quarantine_steps"]) is not int or relay["quarantine_steps"] < 0:
        raise ValueError("env.relay.quarantine_steps must be a nonnegative integer")
    _unit_interval(relay, "alpha_max", "relay", low_open=True)
    _unit_interval(relay, "tau_need", "relay")
    _unit_interval(relay, "master_rate", "relay", low_open=True)
    fractions = relay["fractions"]
    if (not isinstance(fractions, (list, tuple)) or not fractions
            or any(isinstance(f, bool) or not isinstance(f, (int, float)) or not 0 < f < 1 for f in fractions)
            or len(set(float(f) for f in fractions)) != len(fractions)):
        raise ValueError("env.relay.fractions must be distinct values in (0, 1)")
    relay["fractions"] = sorted(float(f) for f in fractions)
    if relay["max_new_chains_per_reset"] * len(relay["fractions"]) > relay["capacity"]:
        raise ValueError("env.relay.max_new_chains_per_reset * len(fractions) exceeds the relay capacity")

    active = sapg["branches"] > 1 or relay["enabled"]
    if active and not is_eval:
        if env_cfg["sgs"]["mode"] == "disabled":
            raise ValueError("SAPG branches and relay resets require --sgs_mode sgs or uniform")
        if env_cfg.get("tsil", {}).get("enabled", False):
            raise ValueError("TSIL cannot be combined with SAPG branches or relay resets")
        if relay["enabled"] and relay["source"] == "scouts" and sapg["branches"] < 2:
            raise ValueError("--relay_source scouts requires --sapg_branches >= 2")
    env_cfg["sapg"] = sapg
    env_cfg["relay"] = relay
    return sapg, relay


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
    apply_scout_relay_config(cfg, args, is_eval=is_eval)
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
    parser.add_argument("--action_scaling", choices=ACTION_SCALING_MODES, default=None,
                        help="legacy: clip raw policy outputs at the physical limits. "
                             "normalized: clip outputs to [-1, 1], then scale to the limits "
                             "(policy log-std starts at log 0.5). Defaults to the saved YAML or legacy.")
    # Scout & Relay (doc/SGS_SAPG_scout_relay_design.md). None keeps the YAML/default value.
    parser.add_argument("--sapg_branches", type=int, default=None,
                        help="SAPG branches M (1 = ordinary PPO). Branch 0 is the deployed leader.")
    parser.add_argument("--sapg_leader_fraction", type=float, default=None,
                        help="Share of environments for the leader (default 0.5); scouts split the rest.")
    parser.add_argument("--sapg_latent_dim", type=int, default=None,
                        help="Learned per-branch condition vector size (default 16).")
    parser.add_argument("--sapg_off_policy_coef", type=float, default=None,
                        help="SAPG lambda for the leader's IS term (default 1; 0 disables the IS channel).")
    parser.add_argument("--sapg_entropy_coef", type=float, default=None,
                        help="Scout j adds coef * j * entropy (default 0; the leader never does).")
    parser.add_argument("--sapg_mu_max", type=float, default=None,
                        help="Optional truncation of the IS weight mu (default 0 = untruncated SAPG).")
    parser.add_argument("--scout_need_beta", type=float, default=None,
                        help="Scout curriculum exponent on (1 - leader success); 0 = per-branch SGS (default 1).")
    parser.add_argument("--scout_target_success", type=float, nargs="+", default=None,
                        help="SGS target success per scout (last value repeats); default: the leader's target.")
    relay = parser.add_mutually_exclusive_group()
    relay.add_argument("--relay", dest="relay", action="store_true",
                       help="Enable the relay bank: snapshots of successful source episodes become leader resets.")
    relay.add_argument("--no_relay", dest="relay", action="store_false")
    parser.set_defaults(relay=None)
    parser.add_argument("--relay_source", choices=("scouts", "leader"), default=None,
                        help="scouts (method) or leader (self-relay baselines).")
    parser.add_argument("--relay_trigger", choices=("success", "failure"), default=None,
                        help="success (method) or failure (ablation).")
    parser.add_argument("--relay_capacity", type=int, default=None)
    parser.add_argument("--relay_alpha_max", type=float, default=None,
                        help="Cap on the relay share of leader resets (default 0.3; 1 = no cap).")
    parser.add_argument("--relay_tau_need", type=float, default=None,
                        help="Create relay chains only where leader success <= tau (default 0.3).")
    parser.add_argument("--relay_max_chains_per_config", type=int, default=None)
    parser.add_argument("--relay_fractions", type=float, nargs="+", default=None,
                        help="Fractions of the pre-hold horizon to snapshot (default 0.2 0.4 0.6 0.8).")
    parser.add_argument("--relay_snapshot_interval", type=int, default=None,
                        help="High-level steps between snapshots (default 5).")
    parser.add_argument("--relay_sampling", choices=("sgs", "uniform"), default=None,
                        help="Leader SGS within the relay bank (method) or uniform (ablation).")
    parser.add_argument("--relay_controller_state", dest="relay_controller_state", action="store_true",
                        help="Ablation: also restore gait/history controller buffers at relay starts.")
    parser.set_defaults(relay_controller_state=None)
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
