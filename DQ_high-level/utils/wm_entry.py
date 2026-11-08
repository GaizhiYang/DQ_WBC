"""Configuration and isolated real-physics validation for the teacher."""

import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import yaml

INTEGER_OPTIONS = ("replay_envs", "replay_capacity", "low_replay_capacity", "window", "batch_size",
                   "batches_per_update", "warmup_steps", "horizon", "imagination_batch",
                   "probe_interval", "probe_envs", "probe_prefix_steps", "model_microbatch_size",
                   "imagination_chunk_size", "critic_epochs", "critic_minibatches", "lr_decay_low_steps")
REVISION_INTEGER_OPTIONS = ("wm_max_batches", "wm_prepare_boundaries", "value_prepare_boundaries",
                            "value_prepare_batches", "actor_stage1_boundaries", "actor_stage2_boundaries",
                            "actor_stage1_updates", "actor_stage2_updates", "actor_max_updates",
                            "diagnostics_interval", "diagnostics_batch_size")
REVISION_FLOAT_OPTIONS = ("wm_epochs", "task_aux_alpha", "raw_action_bound_weight", "raw_gripper_latent_bound")
BOOLEAN_OPTIONS = ("critic_memory", "layer_norm", "running_normalization", "allow_partial_windows")
REFERENCE_INTEGER_OPTIONS = ("collection_warmup_high_steps", "world_model_only_iterations",
                             "policy_update_interval", "lr_decay_iterations")
REFERENCE_FLOAT_OPTIONS = ("collection_updates_per_high_step", "action_noise")
REFERENCE_BOOLEAN_OPTIONS = ("actor_running_normalization", "critic_running_normalization")
FLOAT_OPTIONS = ("learning_rate", "actor_learning_rate", "auxiliary_strength",
                 "max_kl", "max_validation_error", "max_start_error", "updates_per_low_step",
                 "critic_learning_rate", "observation_noise_scale", "auxiliary_weight")
PATH_OPTIONS = ("candidate_checkpoint", "validated_checkpoint", "probe_report")
OPTIONAL_FLOAT_OPTIONS = ("initial_action_std", "actor_last_gain", "action_rate_weight")


def reject_candidate_checkpoint(path=None, *, payload=None):
    """Reject a model-probe bundle passed as the ordinary teacher checkpoint.

    skrl's generic loader silently skips unknown top-level modules. A candidate
    bundle nests its policy under ``teacher`` and would therefore leave a fresh
    Actor unloaded. Callers that already loaded the checkpoint on CPU may pass
    ``payload`` to avoid a second read; this guard checks its schema only.
    """
    if payload is None:
        if path is None:
            raise ValueError("Provide a checkpoint path or its already-loaded payload")
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
    schema = payload.get("schema") if isinstance(payload, dict) else None
    if isinstance(schema, str) and schema.startswith("dq-world-model-candidate-"):
        raise ValueError(
            "A world-model candidate bundle cannot be used as --checkpoint: "
            "it does not expose teacher modules to the ordinary policy loader. "
            "Use a full teacher agent_<step>.pt checkpoint for evaluation; "
            "candidate bundles are for probe_teacher_world_model.py or "
            "--wm_candidate_checkpoint initialization.")


def add_world_model_arguments(parser):
    parser.add_argument("--world_model", choices=("off", "fit", "hybrid"), default=None,
                        help="fit trains dynamics only; hybrid also trains the Actor through imagination.")
    parser.add_argument("--wm_recipe", choices=("reference", "manipulation", "supermimic", "gated"), default=None,
                        help="reference uses pure BPTT; older recipes retain PPO plus BPTT.")
    parser.add_argument("--wm_task_contract", choices=("legacy", "manipulation_v2"), default=None)
    parser.add_argument("--wm_holdout_mode", choices=("episode", "environment"), default=None)
    parser.add_argument("--wm_amp", choices=("none", "bf16"), default=None)
    for name in BOOLEAN_OPTIONS + REFERENCE_BOOLEAN_OPTIONS:
        parser.add_argument("--wm_" + name, type=lambda value: _parse_bool(value), default=None,
                            metavar="true|false")
    parser.add_argument("--wm_state_loss_reduction", choices=("sum", "mean", "reference"), default=None,
                        help="reference: feature mean, time sum, logical-batch mean.")
    parser.add_argument("--wm_action_coordinates", choices=("normalized_commands", "physical"), default=None)
    parser.add_argument("--wm_action_rate_mode", choices=("squared_all", "base_l2"), default=None)
    parser.add_argument("--wm_config", default=None, help="Optional world-model YAML settings.")
    for name in INTEGER_OPTIONS + REVISION_INTEGER_OPTIONS + REFERENCE_INTEGER_OPTIONS:
        flags = ["--wm_" + name] + (["--" + name] if name.startswith("wm_") else [])
        parser.add_argument(*flags, dest="wm_" + name, type=int, default=None)
    for name in FLOAT_OPTIONS + OPTIONAL_FLOAT_OPTIONS + REVISION_FLOAT_OPTIONS + REFERENCE_FLOAT_OPTIONS:
        flags = ["--wm_" + name] + (["--" + name] if name.startswith("wm_") else [])
        parser.add_argument(*flags, dest="wm_" + name, type=float, default=None)
    for name in PATH_OPTIONS:
        parser.add_argument("--wm_" + name, default=None)
    parser.add_argument("--wm_probe_device", default=None,
                        help="GPU for isolated real action-effect probes; default is RL device.")


def _parse_bool(value):
    if str(value).lower() not in ("true", "false"):
        raise ValueError("Expected true or false")
    return str(value).lower() == "true"


def configure_world_model(cfg, args, is_eval=False):
    original = cfg["env"].get("world_model", {})
    if not isinstance(original, dict):
        raise ValueError("env.world_model must be a mapping")
    settings = dict(original)
    filename = getattr(args, "wm_config", None)
    if filename:
        with open(filename) as stream:
            extra = yaml.safe_load(stream)
        if not isinstance(extra, dict):
            raise ValueError("--wm_config must contain a mapping")
        extra = extra.get("world_model", extra)
        if not isinstance(extra, dict):
            raise ValueError("--wm_config world_model must contain a mapping")
        settings.update(extra)
    mode = getattr(args, "world_model", None)
    if mode is not None:
        settings["mode"] = mode
    settings.setdefault("mode", "off")
    for name in (INTEGER_OPTIONS + REVISION_INTEGER_OPTIONS + FLOAT_OPTIONS + OPTIONAL_FLOAT_OPTIONS
                 + REVISION_FLOAT_OPTIONS + BOOLEAN_OPTIONS + REFERENCE_INTEGER_OPTIONS
                 + REFERENCE_FLOAT_OPTIONS + REFERENCE_BOOLEAN_OPTIONS + PATH_OPTIONS + (
            "probe_device", "recipe", "state_loss_reduction", "action_coordinates", "action_rate_mode",
            "task_contract", "holdout_mode", "amp")):
        value = getattr(args, "wm_" + name, None)
        if value is not None:
            settings[name] = value
    if is_eval:
        settings["mode"] = "off"
    if settings["mode"] not in ("off", "fit", "hybrid"):
        raise ValueError("world_model.mode must be off, fit or hybrid")
    active = settings["mode"] != "off"
    # Explicit mode=off with the revised recipe is the same real-task baseline:
    # preserve initialization, preprocessing and task semantics without WM buffers.
    if active or settings.get("recipe") == "reference" or (settings.get("recipe") == "manipulation" and not is_eval):
        # All new training uses the explicit frozen low-controller loop. Keep
        # the flag in exported task configs so isolated mode=off probes and
        # evaluation execute the same eight low steps per high action.
        settings["fixed8"] = True
        if args.task != "B1Z1PickMulti":
            raise ValueError("World model supports the B1Z1PickMulti teacher")
        if cfg["env"]["tsil"]["enabled"]:
            raise ValueError("The world-model mainline does not combine TSIL")
        if active and args.resume:
            raise ValueError("World-model/SGS training resume is not supported; use model checkpoints for evaluation or candidate initialization")
        def canonical(device):
            if not isinstance(device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", device):
                raise ValueError("World-model teacher entry requires GPU simulation and learning")
            return "cuda:{}".format(int(device.split(":")[1]) if ":" in device else 0)
        if active and canonical(args.rl_device) != canonical(args.sim_device):
            raise ValueError("World-model simulator and learner must use the same CUDA device")
        from utils.wm_ppo import DEFAULT_WORLD_MODEL_CONFIG
        from utils.world_model.recipe import SUPERMIMIC_CONFIG, MANIPULATION_CONFIG
        from utils.world_model.reference_recipe import REFERENCE_CONFIG
        defaults = dict(DEFAULT_WORLD_MODEL_CONFIG)
        for name in (REVISION_INTEGER_OPTIONS + REVISION_FLOAT_OPTIONS + BOOLEAN_OPTIONS
                     + ("task_aux_config", "holdout_mode", "amp", "task_contract")):
            defaults.setdefault(name, MANIPULATION_CONFIG[name])
        defaults.update(task_contract="legacy", task_aux_alpha=0., holdout_mode="episode",
                        allow_partial_windows=False)
        recipe = settings.get("recipe", "supermimic")
        if recipe not in ("reference", "manipulation", "supermimic", "gated"):
            raise ValueError("world_model.recipe must be reference, manipulation, supermimic or gated")
        if recipe == "supermimic":
            defaults.update(SUPERMIMIC_CONFIG)
        elif recipe == "manipulation":
            defaults.update(MANIPULATION_CONFIG)
        elif recipe == "reference":
            defaults.update(REFERENCE_CONFIG)
        defaults.update(probe_envs=128, probe_prefix_steps=8, probe_device=args.rl_device,
                        probe_timeout_seconds=900)
        unknown = set(settings).difference(defaults)
        if unknown:
            raise ValueError("Unknown world-model settings: " + ", ".join(sorted(unknown)))
        defaults.update(settings)
        settings = defaults
        reference_ints = REFERENCE_INTEGER_OPTIONS if recipe == "reference" else ()
        reference_floats = REFERENCE_FLOAT_OPTIONS if recipe == "reference" else ()
        reference_bools = REFERENCE_BOOLEAN_OPTIONS if recipe == "reference" else ()
        for name in INTEGER_OPTIONS + REVISION_INTEGER_OPTIONS + reference_ints + ("validation_batch_size", "recent_steps"):
            value = settings[name]
            minimum = 0 if name in ("probe_interval", "probe_prefix_steps", "warmup_steps",
                                    "model_microbatch_size", "imagination_chunk_size", "diagnostics_interval") else 1
            if recipe == "reference" and name in ("wm_max_batches", "world_model_only_iterations"):
                minimum = 0
            if type(value) is not int or value < minimum:
                raise ValueError("world_model.{} has an invalid integer value".format(name))
        for name in FLOAT_OPTIONS + REVISION_FLOAT_OPTIONS + reference_floats + ("model_grad_clip", "termination_weight", "contact_weight", "max_grad_norm",
                                    "holdout_fraction", "recent_fraction", "probe_timeout_seconds",
                                    "grad_clip_value", "critic_gae_lambda", "critic_max_grad_norm", "lr_final_ratio"):
            try:
                value = float(settings[name])
            except (TypeError, ValueError) as error:
                raise ValueError("world_model.{} must be numeric".format(name)) from error
            zero_allowed = name in ("auxiliary_strength", "auxiliary_weight", "termination_weight", "contact_weight", "recent_fraction",
                                    "updates_per_low_step", "observation_noise_scale", "grad_clip_value", "critic_gae_lambda",
                                    "task_aux_alpha", "raw_action_bound_weight", "action_noise")
            if (isinstance(settings[name], bool) or not math.isfinite(value)
                    or value < 0 or (not zero_allowed and value == 0)):
                raise ValueError("world_model.{} has an invalid value".format(name))
            # PyYAML may parse scientific notation such as 2e-4 as a string.
            # Pass a number, not the validated-but-original string, to Adam.
            settings[name] = value
        if not 0 < settings["holdout_fraction"] < 1 or not 0 <= settings["recent_fraction"] <= 1:
            raise ValueError("World-model holdout/recent fractions must be valid probabilities")
        if (settings["window"] > settings["low_replay_capacity"]
                or settings["horizon"] > settings["replay_capacity"]):
            raise ValueError("World-model replay capacity must cover both rollout windows")
        for name in ("hidden", "term_hidden", "contact_hidden", "critic_hidden") + (("actor_hidden",) if recipe == "reference" else ()):
            values = settings[name]
            if (not isinstance(values, (list, tuple)) or not values
                    or any(type(value) is not int or value < 1 for value in values)):
                raise ValueError("world_model.{} must contain positive integer widths".format(name))
            settings[name] = list(values)
        for name in BOOLEAN_OPTIONS + reference_bools + ("use_tail_value", "fixed8", "compact_low_replay", "balance_families", "fixed_preprocessing", "kl_rollback"):
            if type(settings[name]) is not bool:
                raise ValueError("world_model.{} must be Boolean".format(name))
        for name, choices in (("holdout_mode", ("episode", "environment")),
                              ("amp", ("none", "bf16")),
                              ("task_contract", ("legacy", "manipulation_v2"))):
            if settings[name] not in choices:
                raise ValueError("Invalid world_model.{}".format(name))
        if not isinstance(settings["task_aux_config"], dict):
            raise ValueError("world_model.task_aux_config must be a mapping")
        from utils.world_model.task_objective import resolve_task_aux_config
        settings["task_aux_config"] = resolve_task_aux_config(settings["task_aux_config"])
        if recipe == "reference":
            from utils.world_model.reference_task import resolve_reference_task_config
            settings["reference_task_config"] = resolve_reference_task_config(settings["reference_task_config"])
            required = dict(value_source="imagined", holdout_mode="environment", fixed8=True,
                            dynamics_target="physical_increment", state_loss_reduction="reference",
                            actor_objective_reduction="sum", critic_activation="elu",
                            imagined_reward_mode="reference_task", critic_features_mode="reference_full",
                            running_normalization=True, fixed_preprocessing=False,
                            use_tail_value=True, kl_rollback=False, auxiliary_strength=1.,
                            auxiliary_weight=0., task_aux_alpha=0., balance_families=False,
                            recent_fraction=0., compact_low_replay=True, allow_partial_windows=True)
            for name, expected in required.items():
                if settings[name] != expected:
                    raise ValueError("reference recipe requires {}={!r}".format(name, expected))
            if settings["policy_amp"] not in ("none", "bf16"):
                raise ValueError("world_model.policy_amp must be none or bf16")
            if settings["updates_per_low_step"] != 0 or settings["observation_noise_scale"] != 0:
                raise ValueError("reference uses a high-step collection clock and fixed action exploration only")
            if not 0 < settings["collection_updates_per_high_step"] <= 1:
                raise ValueError("reference collection_updates_per_high_step must be in (0, 1]")
            gamma = settings["gamma"]
            if isinstance(gamma, bool) or not isinstance(gamma, (float, int)) or not 0 < gamma < 1:
                raise ValueError("reference gamma must be in (0, 1)")
        if recipe == "manipulation":
            if settings["value_source"] != "imagined":
                raise ValueError("Manipulation preparation requires the independent imagined Critic")
            if settings["holdout_mode"] != "environment":
                raise ValueError("Manipulation uses a fixed environment holdout")
            if settings["updates_per_low_step"] != 0:
                raise ValueError("Manipulation budgets use wm_epochs, not updates_per_low_step")
            if not settings["actor_stage1_updates"] <= settings["actor_stage2_updates"] <= settings["actor_max_updates"] <= 4:
                raise ValueError("Manipulation Actor budgets must be nondecreasing and at most four")
        if settings["value_source"] not in ("ppo", "imagined"):
            raise ValueError("world_model.value_source must be ppo or imagined")
        if settings["lr_schedule"] not in ("constant", "cosine"):
            raise ValueError("world_model.lr_schedule must be constant or cosine")
        if settings["state_loss_reduction"] not in ("sum", "mean", "reference"):
            raise ValueError("world_model.state_loss_reduction must be sum, mean or reference")
        if settings["action_coordinates"] not in ("normalized_commands", "physical"):
            raise ValueError("world_model.action_coordinates must be normalized_commands or physical")
        if settings["action_rate_mode"] not in ("squared_all", "base_l2"):
            raise ValueError("world_model.action_rate_mode must be squared_all or base_l2")
        for name in OPTIONAL_FLOAT_OPTIONS:
            raw = settings[name]
            if raw is None:
                continue  # Legacy configurations retain their original initialization/reward.
            try:
                value = float(raw)
            except (TypeError, ValueError) as error:
                raise ValueError("world_model.{} must be numeric or null".format(name)) from error
            if (isinstance(raw, bool) or not math.isfinite(value)
                    or (value > 0 if name == "action_rate_weight" else value <= 0)):
                raise ValueError("world_model.{} has an invalid value".format(name))
            settings[name] = value
        if not 0 <= settings["critic_gae_lambda"] <= 1 or not 0 < settings["lr_final_ratio"] <= 1:
            raise ValueError("World-model lambda and final LR ratio must be valid probabilities")
        betas = settings["actor_betas"]
        if (not isinstance(betas, (list, tuple)) or len(betas) != 2
                or any(isinstance(b, bool) or not isinstance(b, (int, float)) or not 0 <= b < 1 for b in betas)):
            raise ValueError("world_model.actor_betas must contain two values in [0, 1)")
        if settings["recipe"] == "supermimic" and settings["updates_per_low_step"] <= 0:
            raise ValueError("SuperMimic updates_per_low_step must be positive")
        max_age = settings["report_max_age_seconds"]
        if max_age is not None:
            max_age = float(max_age)
            if not math.isfinite(max_age) or max_age <= 0:
                raise ValueError("world_model.report_max_age_seconds must be positive or null")
            settings["report_max_age_seconds"] = max_age
        settings["probe_device"] = canonical(settings["probe_device"])
        for name in PATH_OPTIONS:
            if settings[name] is None:
                settings[name] = ""
            if not isinstance(settings[name], (str, Path)):
                raise ValueError("world_model.{} must be a file path".format(name))
            settings[name] = str(settings[name])
            if active and settings[name] and not Path(settings[name]).is_file():
                raise ValueError("world_model.{} does not exist: {}".format(name, settings[name]))
        if bool(settings["validated_checkpoint"]) != bool(settings["probe_report"]):
            raise ValueError("A validated checkpoint requires its matching probe report")
        if settings["recipe"] in ("reference", "supermimic", "manipulation") and settings["validated_checkpoint"]:
            raise ValueError("SuperMimic uses one current model; initialize via candidate_checkpoint")
        if settings["action_rate_mode"] == "squared_all" and settings["action_rate_weight"] is not None:
            cfg.setdefault("reward", {}).setdefault("scales", {})["action_rate"] = settings["action_rate_weight"]
    cfg["env"]["world_model"] = settings
    if settings.get("task_contract", "legacy") == "manipulation_v2":
        cfg["env"]["smallValueSetZero"] = False
    return settings


def _terminate_probe_group(process):
    """Reap the probe parent and kill its dedicated worker process group."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # A parent can exit while a CUDA worker survives or ignores SIGTERM.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5)


def run_probe_subprocess(command, cwd, stream, timeout, environment=None):
    """Run an isolated process tree; timeout/KeyboardInterrupt leave no workers."""
    process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                               env=os.environ.copy() if environment is None else environment,
                               start_new_session=True)
    try:
        return process.wait(timeout=float(timeout))
    except BaseException:
        _terminate_probe_group(process)
        raise


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def install_probe_callback(agent, run_dir, seed):
    """Run probes in child processes; never step or rewind the training world."""
    settings = agent.wm_config
    if settings["mode"] != "hybrid" or int(settings["probe_interval"]) <= 0:
        agent.maybe_validate = None
        return
    directory = Path(run_dir).resolve() / "world_model"
    directory.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "probe_teacher_world_model.py"

    def validate(current):
        step = current.wm_replay.steps
        candidate = directory / ("candidate_{:08d}.pt".format(step))
        report = directory / ("probe_{:08d}.json".format(step))
        log = directory / ("probe_{:08d}.log".format(step))
        current.save_candidate(candidate)
        command = [sys.executable, str(script), "--checkpoint", str(candidate),
                   "--output", str(report), "--num_envs", str(settings.get("probe_envs", 128)),
                   "--prefix_steps", str(settings.get("probe_prefix_steps", 8)),
                   "--seed", str(seed), "--device", settings.get("probe_device", str(current.device))]
        started = time.monotonic()
        accepted = False
        failure = ""
        try:
            print("[world model] isolated GPU action-effect probe at step {}".format(step), flush=True)
            with log.open("w") as stream:
                returncode = run_probe_subprocess(command, cwd=str(script.parent), stream=stream,
                                                   timeout=settings.get("probe_timeout_seconds", 900))
            if returncode:
                failure = "probe process exit {}".format(returncode)
            elif report.is_file():
                accepted = current.activate_validated(candidate, report)
                if not accepted:
                    failure = "; ".join(current.wm_probe_rejection_reasons)
            else:
                failure = "probe produced no report"
        except (subprocess.SubprocessError, OSError, ValueError, RuntimeError) as error:
            failure = str(error)
        duration = time.monotonic() - started
        current.track_data("World model / probe/accepted", float(accepted))
        current.track_data("World model / probe/seconds", duration)
        (directory / "last_probe_status.json").write_text(json.dumps(_json_safe(
            dict(step=step, accepted=accepted, reason=failure, seconds=duration,
                 candidate=str(candidate), report=str(report), log=str(log))), indent=2, allow_nan=False))
        print("[world model] probe {}: {}".format(
            "accepted" if accepted else "not accepted", failure or str(report)), flush=True)
    agent.maybe_validate = validate


def save_world_model_artifacts(trainer):
    agent = trainer.agents
    if not hasattr(agent, "wm_candidate"):
        return
    directory = Path(trainer.teacher_args.experiment_dir) / trainer.teacher_args.wandb_name / "world_model"
    directory.mkdir(parents=True, exist_ok=True)
    agent.save_candidate(directory / "candidate_latest.pt")
    invalid = [key for key, value in agent.last_world_model_metrics.items()
               if isinstance(value, float) and not math.isfinite(value)]
    (directory / "status.json").write_text(json.dumps(_json_safe(
        dict(status=agent.wm_status, device=str(agent.device),
             recipe=getattr(agent, "wm_config", {}).get("recipe", "gated"),
             value_source=getattr(agent, "wm_config", {}).get("value_source", "ppo"),
             update_rounds=getattr(agent, "wm_update_rounds", 0),
             reference_schedule=(agent.reference_schedule.state_dict()
                                 if getattr(agent, "reference_schedule", None) is not None else None),
             manipulation_schedule=(agent.wm_schedule.state_dict()
                                    if getattr(agent, "wm_schedule", None) is not None else None),
             resolved_config=getattr(agent, "wm_config", {}),
             resolved_config_sha256=hashlib.sha256(json.dumps(
                 _json_safe(getattr(agent, "wm_config", {})), sort_keys=True,
                 separators=(",", ":")).encode()).hexdigest(),
             state_dim=agent.wm_adapter.state_dim,
             physical_dim=getattr(agent.wm_adapter, "physical_dim", None),
             residual_dim=getattr(agent.wm_adapter, "delta_dim", None),
             low_replay_steps=(agent.wm_low_replay.steps if getattr(agent, "wm_low_replay", None) is not None else None),
             fixed8=bool(getattr(agent, "wm_config", {}).get("fixed8", False)),
             replay_steps=agent.wm_replay.steps,
             model_updates=agent.wm_trainer.updates,
             validated_snapshot=agent.wm_has_validated_snapshot,
             nonfinite_metrics=invalid,
             metrics=agent.last_world_model_metrics)), indent=2, allow_nan=False))
