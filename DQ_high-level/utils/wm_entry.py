"""Configuration and isolated real-physics validation for the teacher."""

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
FLOAT_OPTIONS = ("learning_rate", "actor_learning_rate", "auxiliary_strength",
                 "max_kl", "max_validation_error", "max_start_error", "updates_per_low_step",
                 "critic_learning_rate", "observation_noise_scale")
PATH_OPTIONS = ("candidate_checkpoint", "validated_checkpoint", "probe_report")


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
                        help="Real PPO plus model fitting; hybrid also enables short BPTT.")
    parser.add_argument("--wm_recipe", choices=("supermimic", "gated"), default=None,
                        help="SuperMimic update recipe (default); gated preserves the older auxiliary path.")
    parser.add_argument("--wm_config", default=None, help="Optional world-model YAML settings.")
    for name in INTEGER_OPTIONS:
        parser.add_argument("--wm_" + name, type=int, default=None)
    for name in FLOAT_OPTIONS:
        parser.add_argument("--wm_" + name, type=float, default=None)
    for name in PATH_OPTIONS:
        parser.add_argument("--wm_" + name, default=None)
    parser.add_argument("--wm_probe_device", default=None,
                        help="GPU for isolated real action-effect probes; default is RL device.")


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
    for name in INTEGER_OPTIONS + FLOAT_OPTIONS + PATH_OPTIONS + ("probe_device", "recipe"):
        value = getattr(args, "wm_" + name, None)
        if value is not None:
            settings[name] = value
    if is_eval:
        settings["mode"] = "off"
    if settings["mode"] not in ("off", "fit", "hybrid"):
        raise ValueError("world_model.mode must be off, fit or hybrid")
    if settings["mode"] != "off":
        # All new training uses the explicit frozen low-controller loop. Keep
        # the flag in exported task configs so isolated mode=off probes and
        # evaluation execute the same eight low steps per high action.
        settings["fixed8"] = True
        if args.task != "B1Z1PickMulti":
            raise ValueError("World model supports the B1Z1PickMulti teacher")
        if cfg["env"]["tsil"]["enabled"]:
            raise ValueError("The world-model mainline does not combine TSIL")
        if args.resume:
            raise ValueError("World-model/SGS training resume is not supported; use model checkpoints for evaluation or candidate initialization")
        def canonical(device):
            if not isinstance(device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", device):
                raise ValueError("World-model teacher entry requires GPU simulation and learning")
            return "cuda:{}".format(int(device.split(":")[1]) if ":" in device else 0)
        if canonical(args.rl_device) != canonical(args.sim_device):
            raise ValueError("World-model simulator and learner must use the same CUDA device")
        from utils.wm_ppo import DEFAULT_WORLD_MODEL_CONFIG
        from utils.world_model.recipe import SUPERMIMIC_CONFIG
        defaults = dict(DEFAULT_WORLD_MODEL_CONFIG)
        recipe = settings.get("recipe", "supermimic")
        if recipe not in ("supermimic", "gated"):
            raise ValueError("world_model.recipe must be supermimic or gated")
        if recipe == "supermimic":
            defaults.update(SUPERMIMIC_CONFIG)
        defaults.update(probe_envs=128, probe_prefix_steps=8, probe_device=args.rl_device,
                        probe_timeout_seconds=900)
        unknown = set(settings).difference(defaults)
        if unknown:
            raise ValueError("Unknown world-model settings: " + ", ".join(sorted(unknown)))
        defaults.update(settings)
        settings = defaults
        for name in INTEGER_OPTIONS + ("validation_batch_size", "recent_steps"):
            value = settings[name]
            minimum = 0 if name in ("probe_interval", "probe_prefix_steps", "warmup_steps",
                                    "model_microbatch_size", "imagination_chunk_size") else 1
            if type(value) is not int or value < minimum:
                raise ValueError("world_model.{} has an invalid integer value".format(name))
        for name in FLOAT_OPTIONS + ("model_grad_clip", "termination_weight", "contact_weight", "max_grad_norm",
                                    "holdout_fraction", "recent_fraction", "probe_timeout_seconds",
                                    "grad_clip_value", "critic_gae_lambda", "critic_max_grad_norm", "lr_final_ratio"):
            try:
                value = float(settings[name])
            except (TypeError, ValueError) as error:
                raise ValueError("world_model.{} must be numeric".format(name)) from error
            zero_allowed = name in ("auxiliary_strength", "termination_weight", "contact_weight", "recent_fraction",
                                    "updates_per_low_step", "observation_noise_scale", "grad_clip_value", "critic_gae_lambda")
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
        for name in ("hidden", "term_hidden", "contact_hidden", "critic_hidden"):
            values = settings[name]
            if (not isinstance(values, (list, tuple)) or not values
                    or any(type(value) is not int or value < 1 for value in values)):
                raise ValueError("world_model.{} must contain positive integer widths".format(name))
            settings[name] = list(values)
        for name in ("use_tail_value", "fixed8", "compact_low_replay", "balance_families", "fixed_preprocessing", "kl_rollback"):
            if type(settings[name]) is not bool:
                raise ValueError("world_model.{} must be Boolean".format(name))
        if settings["value_source"] not in ("ppo", "imagined"):
            raise ValueError("world_model.value_source must be ppo or imagined")
        if settings["lr_schedule"] not in ("constant", "cosine"):
            raise ValueError("world_model.lr_schedule must be constant or cosine")
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
            if settings[name] and not Path(settings[name]).is_file():
                raise ValueError("world_model.{} does not exist: {}".format(name, settings[name]))
        if bool(settings["validated_checkpoint"]) != bool(settings["probe_report"]):
            raise ValueError("A validated checkpoint requires its matching probe report")
        if settings["recipe"] == "supermimic" and settings["validated_checkpoint"]:
            raise ValueError("SuperMimic uses one current model; initialize via candidate_checkpoint")
    cfg["env"]["world_model"] = settings
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
