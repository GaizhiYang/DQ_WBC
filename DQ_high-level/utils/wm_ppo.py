"""Real PPO with GPU dynamics and mean-action BPTT.

The SuperMimic recipe fits one current model, freezes it during BPTT and trains
an independent imagined Critic. The older gated snapshot recipe remains an
explicit compatibility option. Imagined samples never enter PPO or SGS memory.
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from utils.finite_ppo import FinitePPO as PPO

from utils.wm_env import TeacherWorldModelAdapter
from utils.world_model import (HierarchicalDynamics, ResidualDynamics, TransitionReplay,
                               WMTrainerConfig, WorldModelTrainer)
from utils.world_model.imagination import ImaginationConfig, ImaginationTrainer
from utils.world_model.probes import (configuration_fingerprint, load_verified_report,
                                     model_fingerprint)
from utils.world_model.recipe import MANIPULATION_CONFIG, SUPERMIMIC_CONFIG, cosine_ratio
from utils.world_model.replay import SliceStateCodec
from utils.world_model.schedule import ManipulationSchedule, ScheduleConfig
from utils.world_model.stage_counts import RealStageTransitionCounts, captured_stage_flags
from utils.world_model.live_logging import flush_before_model_update, log_model_snapshot


DEFAULT_WORLD_MODEL_CONFIG = {
    "recipe": "gated",             # direct legacy clients; entry selects supermimic
    "mode": "fit",                 # fit | hybrid; only gated recipe requires a probe
    "fixed8": False,               # production entry enables the hierarchical path
    "replay_envs": 128,
    "replay_capacity": 256,
    "low_replay_capacity": 256,    # counted in 20 ms transitions, independently of high replay
    "window": 8,                   # low steps for fixed8; high steps for legacy checkpoints
    "batch_size": 128,
    "batches_per_update": 2,
    "hidden": [512, 512, 512],
    "term_hidden": [64, 64],
    "contact_hidden": [128, 128],
    "learning_rate": 2e-4,
    "model_grad_clip": 1.0,
    "termination_weight": 0.1,
    "contact_weight": 0.1,
    "state_loss_reduction": "mean",
    "auxiliary_weight": 1.0,
    "action_coordinates": "physical",
    "initial_action_std": None,
    "actor_last_gain": None,
    "action_rate_mode": "base_l2",
    "action_rate_weight": None,
    "warmup_steps": 96,
    "horizon": 4,                  # high decisions: 4 * 8 low steps = 0.64 s
    "imagination_batch": 128,
    "actor_learning_rate": 1e-5,
    "auxiliary_strength": 0.05,
    "max_kl": 0.005,
    "max_grad_norm": 0.5,
    "max_validation_error": 0.1,
    "max_start_error": 0.1,
    "holdout_fraction": 0.1,
    "recent_fraction": 0.5,
    "recent_steps": 32,
    "validation_batch_size": 128,
    "use_tail_value": True,
    "probe_interval": 0,           # aligned: diagnostic; gated: snapshot activation
    "report_max_age_seconds": None,
    "candidate_checkpoint": "",
    "validated_checkpoint": "",
    "probe_report": "",
    "compact_low_replay": False,
    "balance_families": True,
    "fixed_preprocessing": False,
    "observation_noise_scale": 0.0,
    "model_microbatch_size": 0,
    "imagination_chunk_size": 0,
    "updates_per_low_step": 0.0,
    "actor_betas": [0.9, 0.999],
    "grad_clip_value": 0.0,
    "kl_rollback": True,
    "value_source": "ppo",
    "critic_hidden": [256, 128],
    "critic_learning_rate": 2e-4,
    "critic_epochs": 5,
    "critic_minibatches": 4,
    "critic_gae_lambda": 0.95,
    "critic_max_grad_norm": 1.0,
    "lr_schedule": "constant",
    "lr_final_ratio": 0.03,
    "lr_decay_low_steps": 250000,
}

CANDIDATE_SCHEMA = "dq-world-model-candidate-v1"
TRAINING_SCHEMA = "dq-world-model-training-v1"
# The locally reviewed implementation reference, not a claim of official code.
REFERENCE_SOURCE_COMMIT = "eda7da480c6d92a68cc41c9628b7efdae08cae23"


def _cloned_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def _audit_rng(device):
    """Save audit evidence without advancing RNGs or initializing other GPUs."""
    device = torch.device(device)
    cuda_device = None
    cuda_state = None
    if device.type == "cuda" and torch.cuda.is_initialized():
        cuda_device = device.index if device.index is not None else torch.cuda.current_device()
        cuda_state = torch.cuda.get_rng_state(cuda_device).clone()
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(), "torch_cuda": cuda_state,
            "torch_cuda_device": cuda_device}


class _SnapshotWeights:
    """Allow fingerprinting checkpoint weights without mutating the active model."""

    def __init__(self, weights):
        self.weights = weights

    def state_dict(self):
        return self.weights


class _TrainingCheckpoint:
    def __init__(self, agent):
        self.agent = agent

    def state_dict(self):
        agent = self.agent
        result = {
            "schema": TRAINING_SCHEMA,
            "metadata": deepcopy(agent.wm_metadata),
            "config": deepcopy(agent.wm_config),
            "candidate": agent.wm_trainer.state_dict(),
            "validated_model": (_cloned_state(agent.wm_validated)
                                if agent.wm_has_validated_snapshot else None),
            "validated_report": deepcopy(agent.wm_validated_report),
            "imagination": agent.wm_imagination.state_dict(),
            "ppo_updates": agent.wm_ppo_updates,
            "update_rounds": agent.wm_update_rounds,
            "update_credit": agent._wm_update_credit,
            "audit": agent._checkpoint_audit(),
        }
        if agent.wm_manipulation:
            result["manipulation_schedule"] = agent.wm_schedule.state_dict()
            result["total_low_steps"] = agent._wm_total_low_steps
        return result

    def load_state_dict(self, state):
        agent = self.agent
        if state.get("schema") != TRAINING_SCHEMA:
            raise ValueError("Unsupported world-model training checkpoint schema")
        agent._check_metadata(state.get("metadata"))
        agent._check_model_configuration(state.get("config"))
        agent.wm_trainer.load_state_dict(state["candidate"])
        agent.wm_imagination.load_state_dict(state["imagination"])
        agent.wm_ppo_updates = int(state.get("ppo_updates", 0))
        agent.wm_update_rounds = int(state.get("update_rounds", 0))
        agent._wm_update_credit = 0.0
        agent._wm_scheduled_low_steps = 0
        agent._wm_total_high_steps = int(state.get("audit", {}).get(
            "sampling_counts", {}).get("high_vector_steps", 0))
        if agent.wm_manipulation:
            agent.wm_schedule.reset_replay()
            agent.wm_stage_counts.reset()
            agent._wm_total_low_steps = int(state.get("total_low_steps", 0))
        agent.wm_has_validated_snapshot = False
        agent.wm_validated_report = None
        agent.wm_validated_fingerprint = None
        if state.get("validated_model") is not None:
            payload = {"schema": CANDIDATE_SCHEMA, "metadata": state["metadata"],
                       "model": state["validated_model"]}
            agent.activate_validated(payload, state.get("validated_report"))
        # Replay is deliberately not checkpointed. New on-policy interaction
        # must rebuild a held-out set and warmup before any resumed BPTT.
        agent.wm_replay.valid.zero_()
        agent.wm_replay.stamp.fill_(-1)
        agent.wm_replay._episodes.zero_()
        agent.wm_replay.steps = 0
        if hasattr(agent.wm_replay, "valid_train_transitions"):
            agent.wm_replay.valid_train_transitions.zero_()
        if agent.wm_low_replay is not None:
            agent.wm_low_replay.valid.zero_()
            agent.wm_low_replay.stamp.fill_(-1)
            agent.wm_low_replay._episodes.zero_()
            agent.wm_low_replay.steps = 0
            if hasattr(agent.wm_low_replay, "valid_train_transitions"):
                agent.wm_low_replay.valid_train_transitions.zero_()
        agent._last_probe_step = 0
        agent.wm_status = "warming_up_after_resume"


class WorldModelPPO(PPO):
    """Preserve real PPO, then fit the candidate and optionally improve Actor.

    ``infos['world_model']`` contains selected-environment tensors captured by
    the simulator wrapper before automatic reset. Its required fields are
    ``state, context, next_state, actions, dt, reward, terminated, truncated,
    success, family, episode_id``. The first ``min(replay_envs, num_envs)`` slots
    are selected once; context keeps their original simulator lookup IDs.

    ``maybe_validate`` is an optional callback installed by the launcher. It
    receives this agent after fitting, at ``probe_interval`` vector steps, and
    may call ``save_candidate`` and ``activate_validated``. It must use isolated
    simulators. Only the legacy gated recipe waits for a passing report.
    """

    def __init__(self, *args, wm_env, wm_config=None, maybe_validate=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wm_env = wm_env
        self.wm_config = deepcopy(DEFAULT_WORLD_MODEL_CONFIG)
        if (wm_config or {}).get("recipe") == "supermimic":
            self.wm_config.update(deepcopy(SUPERMIMIC_CONFIG))
        elif (wm_config or {}).get("recipe") == "manipulation":
            self.wm_config.update(deepcopy(MANIPULATION_CONFIG))
        self.wm_config.update(wm_config or {})
        self.maybe_validate = maybe_validate
        config = self.wm_config
        self.wm_manipulation = config["recipe"] == "manipulation"
        self.wm_aligned = config["recipe"] in ("supermimic", "manipulation")
        if config["recipe"] not in ("gated", "supermimic", "manipulation"):
            raise ValueError("Unknown world-model recipe")
        if self.wm_aligned and not config["fixed8"]:
            raise ValueError("SuperMimic-aligned training requires fixed8")
        if config["mode"] not in ("fit", "hybrid"):
            raise ValueError("world-model mode must be fit or hybrid")
        self.wm_schedule = (ManipulationSchedule(ScheduleConfig.from_mapping(config))
                            if self.wm_manipulation else None)
        if self.wm_manipulation:
            if config["value_source"] != "imagined" or not config["use_tail_value"]:
                raise ValueError("Manipulation preparation requires independent imagined value and tail value")
            if config["holdout_mode"] != "environment":
                raise ValueError("Manipulation recipe requires a fixed environment holdout")
            for name in ("diagnostics_interval", "diagnostics_batch_size"):
                minimum = 0 if name == "diagnostics_interval" else 1
                if isinstance(config[name], bool) or int(config[name]) != config[name] or config[name] < minimum:
                    raise ValueError(name + " must be an integer >= " + str(minimum))
        positive = ("replay_envs", "replay_capacity", "window", "batch_size",
                    "batches_per_update", "horizon", "imagination_batch",
                    "validation_batch_size", "max_validation_error", "max_start_error")
        if any(not math.isfinite(float(config[key])) or float(config[key]) <= 0 for key in positive):
            raise ValueError("World-model dimensions and error thresholds must be positive")
        if config["warmup_steps"] < 0 or config["probe_interval"] < 0:
            raise ValueError("World-model warmup/probe intervals must be nonnegative")
        if not 0 < config["holdout_fraction"] < 1:
            raise ValueError("World-model validation requires holdout_fraction in (0, 1)")
        fit_capacity = (config["low_replay_capacity"] if config["fixed8"]
                        else config["replay_capacity"])
        if config["window"] > fit_capacity or config["horizon"] > config["replay_capacity"]:
            raise ValueError("Replay capacity must cover model and imagination windows")
        if self.policy is self.value:
            raise ValueError("World-model BPTT requires independent Actor and Critic")
        if self._rewards_shaper is not None:
            raise ValueError("World-model rewards must use the same raw task units as real PPO")
        environment_device = torch.empty(0, device=wm_env.device).device
        learner_device = torch.empty(0, device=self.device).device
        if environment_device != learner_device:
            raise ValueError("World-model training requires simulator and learner on the same device")
        if config["fixed8"]:
            from utils.wm_fixed8 import Fixed8TeacherWorldModelAdapter
            self.wm_adapter = Fixed8TeacherWorldModelAdapter(wm_env)
        else:
            self.wm_adapter = TeacherWorldModelAdapter(wm_env)
        self.wm_metadata = self._adapter_metadata()
        selected = min(int(config["replay_envs"]), int(wm_env.num_envs))
        self.wm_env_ids = torch.arange(selected, device=learner_device)
        # The external collector uses the exact adapter and selected slots.
        wm_env.world_model_adapter = self.wm_adapter
        wm_env.world_model_env_ids = self.wm_env_ids
        model_class = HierarchicalDynamics if config["fixed8"] else ResidualDynamics
        model_options = dict(hidden=config["hidden"], device=learner_device)
        if config["fixed8"]:
            model_options.update(contact_hidden=config["contact_hidden"], nominal_dt=0.16)
            if self.wm_manipulation:
                model_options.update(layer_norm=config["layer_norm"],
                                     running_normalization=config["running_normalization"], amp=config["amp"])
        else:
            model_options.update(term_hidden=config["term_hidden"])
        self.wm_candidate = model_class(self.wm_adapter, **model_options)
        if self.wm_aligned:
            # One current model: freeze parameters only during Actor BPTT.
            self.wm_validated = self.wm_candidate
        else:
            self.wm_validated = model_class(self.wm_adapter, **model_options)
            self.wm_validated.requires_grad_(False)
            self.wm_validated.eval()
        replay_options = ({"holdout_mode": config["holdout_mode"]} if self.wm_manipulation else {})
        self.wm_replay = TransitionReplay(selected, config["replay_capacity"],
                                          self.wm_adapter.state_dim, self.wm_adapter.action_dim,
                                          self.wm_adapter.context_dim, learner_device,
                                          holdout_fraction=config["holdout_fraction"], **replay_options)
        codec = (SliceStateCodec(self.wm_adapter.state_dim, self.wm_adapter.slices,
                 ("robot", "object", "dof_pos", "dof_vel", "table", "table_height", "foot_contacts"))
                 if config["fixed8"] and config["compact_low_replay"] else None)
        low_replay_options = dict(replay_options)
        if self.wm_manipulation:
            low_replay_options["allow_partial_windows"] = config["allow_partial_windows"]
        self.wm_low_replay = (TransitionReplay(
            selected, config["low_replay_capacity"], self.wm_adapter.state_dim,
            self.wm_adapter.low_action_dim, self.wm_adapter.context_dim,
            learner_device, holdout_fraction=config["holdout_fraction"], state_codec=codec,
            **low_replay_options)
            if config["fixed8"] else None)
        self.wm_trainer = WorldModelTrainer(self.wm_candidate, WMTrainerConfig(
            window=config["window"], batches_per_update=config["batches_per_update"],
            batch_size=config["batch_size"], learning_rate=config["learning_rate"],
            grad_clip=config["model_grad_clip"], termination_weight=config["termination_weight"],
            contact_weight=config["contact_weight"],
            state_loss_reduction=config["state_loss_reduction"],
            auxiliary_weight=config["auxiliary_weight"],
            recent_fraction=config["recent_fraction"], recent_steps=config["recent_steps"],
            validation_batch_size=config["validation_batch_size"],
            balance_families=config["balance_families"],
            microbatch_size=config["model_microbatch_size"]))
        self.wm_imagination = ImaginationTrainer(
            self.wm_validated, self.wm_adapter, self.policy, self.value,
            ImaginationConfig(horizon=config["horizon"], gamma=self._discount_factor,
                              lr=config["actor_learning_rate"], strength=config["auxiliary_strength"],
                              max_grad_norm=config["max_grad_norm"], max_kl=config["max_kl"],
                              use_tail_value=config["use_tail_value"],
                              value_source=config["value_source"],
                              critic_hidden=tuple(config["critic_hidden"]),
                              critic_lr=config["critic_learning_rate"],
                              critic_epochs=config["critic_epochs"],
                              critic_minibatches=config["critic_minibatches"],
                              critic_gae_lambda=config["critic_gae_lambda"],
                              critic_max_grad_norm=config["critic_max_grad_norm"],
                              actor_betas=tuple(config["actor_betas"]),
                              grad_clip_value=config["grad_clip_value"] or None,
                              kl_rollback=config["kl_rollback"],
                              chunk_size=config["imagination_chunk_size"],
                              dt=0.16 if config["fixed8"] else 0.15),
            state_preprocessor=self._state_preprocessor,
            value_preprocessor=self._value_preprocessor)
        self.wm_has_validated_snapshot = False
        self.wm_validated_report = None
        self.wm_validated_fingerprint = None
        self.wm_probe_rejection_reasons = []
        self.wm_ppo_updates = 0
        self.wm_update_rounds = 0
        self._wm_update_credit = 0.0
        self._wm_scheduled_low_steps = 0
        self._wm_total_high_steps = 0
        self._wm_total_low_steps = 0
        bank = getattr(wm_env, "sgs_bank", None)
        self.wm_stage_counts = (RealStageTransitionCounts(
            bank.family_names if bank is not None else ("full_task",), learner_device)
            if self.wm_manipulation else None)
        self._last_probe_step = 0
        self.wm_status = "collecting"
        self.last_world_model_metrics = {}
        # Installed by the entry, independently of W&B/TensorBoard logging.
        self.wm_progress_callback = None
        self.checkpoint_modules["world_model_training"] = _TrainingCheckpoint(self)
        if config["candidate_checkpoint"]:
            self.load_candidate(config["candidate_checkpoint"])
        if bool(config["validated_checkpoint"]) != bool(config["probe_report"]):
            raise ValueError("validated_checkpoint and probe_report must be provided together")
        if config["validated_checkpoint"]:
            self.activate_validated(self._read_candidate(config["validated_checkpoint"]),
                                    config["probe_report"])

    def _adapter_metadata(self):
        metadata = getattr(self.wm_adapter, "metadata", None)
        if callable(metadata):
            result = metadata()
        else:
            # Compatible with tensor-only adapter fixtures. Production supplies
            # the complete physical/task contract through its adapter.
            result = {"schema": "dq-teacher-state-v1", "state_dim": self.wm_adapter.state_dim,
                      "delta_dim": self.wm_adapter.delta_dim, "context_dim": self.wm_adapter.context_dim,
                      "context_feature_dim": self.wm_adapter.context_feature_dim,
                      "slices": {key: [value.start, value.stop]
                                 for key, value in self.wm_adapter.slices.items()}}
        if self.wm_manipulation:
            result["model_execution"] = {name: self.wm_config[name] for name in
                                          ("recipe", "layer_norm", "running_normalization", "amp")}
        return result

    def _check_model_configuration(self, config):
        if not self.wm_manipulation:
            return
        names = ("recipe", "layer_norm", "running_normalization", "amp")
        if config is None or any(config.get(name) != self.wm_config[name] for name in names):
            raise ValueError("World-model checkpoint model execution configuration mismatch")

    def _checkpoint_audit(self):
        high = self._wm_total_high_steps
        low = self._wm_total_low_steps if self.wm_manipulation else high * (8 if self.wm_low_replay is not None else 0)
        selected = len(self.wm_env_ids)
        environments = int(self.wm_env.num_envs)
        return {
            "rng_state": _audit_rng(self.device),
            "rng_restore_supported": False,
            "reference_source": "supermimic_manipulation (local implementation reference)",
            "reference_source_commit": REFERENCE_SOURCE_COMMIT,
            "resolved_config_hash": configuration_fingerprint(self.wm_config),
            "resolved_config_scope": "world_model",
            "real_stage_transitions": (self.wm_stage_counts.state_dict()
                                       if self.wm_stage_counts is not None else None),
            "sampling_counts": {
                "high_vector_steps": high, "low_vector_steps": low,
                "simulated_high_rows": high * environments,
                "simulated_low_rows": low * environments,
                "selected_high_rows": high * selected,
                "selected_low_rows": low * selected,
                "high_valid_train_rows_since_replay_reset": int(self.wm_replay.valid_train_transitions),
                "low_valid_train_rows_since_replay_reset": (
                    int(self.wm_low_replay.valid_train_transitions) if self.wm_low_replay is not None else 0),
            },
        }

    def _check_metadata(self, metadata):
        def canonical(value):
            if value is None:
                return None
            result = deepcopy(value)
            # These absent fields mean the pre-manipulation task, never v2.
            # All existing state widths, physical/action contracts and hashes
            # remain checked; no network/normalizer weights are migrated.
            defaults = {"task_contract": "legacy", "task_aux_alpha": 0.,
                        "task_objective": None, "raw_action_bound_weight": 0.,
                        "raw_gripper_latent_bound": 3., "reward_contract": "legacy",
                        "critic_memory": False, "critic_features_mode": "compact",
                        "imagined_reward_mode": "native_aux", "reference_task": None}
            for name, default in defaults.items():
                result.setdefault(name, default)
            if "small_commands_zero" not in result and "small_commands" in result.get("control", {}):
                result["small_commands_zero"] = result["control"]["small_commands"]
            return result
        if metadata is None or configuration_fingerprint(canonical(metadata)) != configuration_fingerprint(canonical(self.wm_metadata)):
            raise ValueError("World-model checkpoint does not match this state schema/task contract")

    def candidate_payload(self):
        teacher = {"policy": self.policy, "value": self.value,
                   "state_preprocessor": self._state_preprocessor,
                   "value_preprocessor": self._value_preprocessor}
        return {"schema": CANDIDATE_SCHEMA, "metadata": deepcopy(self.wm_metadata),
                "config": deepcopy(self.wm_config), "model": _cloned_state(self.wm_candidate),
                "imagined_critic": (_cloned_state(self.wm_imagination.imagined_critic)
                                    if self.wm_imagination.imagined_critic is not None else None),
                "imagined_critic_updates": (int(self.wm_imagination.imagined_critic_trainer.updates)
                                            if self.wm_imagination.imagined_critic_trainer is not None else 0),
                "imagined_gamma": float(self.wm_imagination.cfg.gamma),
                "imagined_bootstrap_timeouts": bool(self.wm_imagination.cfg.bootstrap_timeouts),
                "imagined_use_tail_value": bool(self.wm_imagination.cfg.use_tail_value),
                "updates": self.wm_trainer.updates, "collected_steps": self.wm_replay.steps,
                "teacher": {name: _cloned_state(module) if hasattr(module, "state_dict") else None
                            for name, module in teacher.items()},
                "teacher_config": deepcopy(getattr(self, "wm_teacher_config", self.wm_env.cfg)),
                "teacher_arguments": deepcopy(dict(getattr(self, "wm_teacher_arguments", {}))),
                "global_step": int(getattr(self.wm_env, "global_step_counter", self.wm_replay.steps)),
                "sgs_bank_fingerprint": getattr(getattr(self.wm_env, "sgs_bank", None), "fingerprint", None)}

    def save_candidate(self, path):
        """Write a probe-ready immutable snapshot; return its in-memory payload."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.candidate_payload()
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=str(destination.parent),
                                             prefix=destination.name + ".", suffix=".tmp",
                                             delete=False) as handle:
                temporary = handle.name
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)
        return payload

    def _read_candidate(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if "world_model_training" in payload:
            training = payload["world_model_training"]
            payload = {"schema": CANDIDATE_SCHEMA, "metadata": training["metadata"],
                       "config": training.get("config"),
                       "model": training["candidate"]["model"],
                       "updates": training["candidate"].get("updates", 0)}
        if payload.get("schema") != CANDIDATE_SCHEMA:
            raise ValueError("Expected a DQ world-model candidate or full teacher checkpoint")
        self._check_metadata(payload.get("metadata"))
        self._check_model_configuration(payload.get("config"))
        return payload

    def load_candidate(self, path):
        payload = self._read_candidate(path)
        self.wm_candidate.load_state_dict(payload["model"])
        self.wm_trainer.updates = int(payload.get("updates", 0))

    def _verify_report(self, report, fingerprint):
        options = dict(model_sha256=fingerprint, configuration=self.wm_metadata,
                       max_age_seconds=self.wm_config["report_max_age_seconds"])
        if isinstance(report, dict):
            # Full teacher checkpoints carry the report itself rather than a
            # path whose file might disappear or be replaced after resumption.
            # This low-frequency compatibility path reuses the exact verifier.
            with tempfile.TemporaryDirectory(prefix="dq-wm-report-") as directory:
                path = Path(directory) / "report.json"
                path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False), encoding="utf-8")
                return load_verified_report(str(path), **options)
        if not report:
            return False, ["missing_action_effect_report"], None
        return load_verified_report(str(report), **options)

    def activate_validated(self, payload, report):
        """Promote exactly the snapshot covered by a passing real action probe.

        Returns False with ``wm_probe_rejection_reasons`` on failed evidence.
        An existing approved snapshot stays active if a new candidate fails.
        Payloads with an incompatible state/task schema raise immediately.
        """
        if isinstance(payload, (str, Path)):
            payload = self._read_candidate(payload)
        self._check_metadata(payload.get("metadata"))
        expected = self.wm_validated.state_dict()
        incoming = payload["model"]
        if set(incoming) != set(expected) or any(
                incoming[key].shape != value.shape or incoming[key].dtype != value.dtype
                for key, value in expected.items()):
            raise ValueError("Validated snapshot architecture/dtype differs from the active world model")
        fingerprint = model_fingerprint(_SnapshotWeights(payload["model"]))
        accepted, reasons, verified = self._verify_report(report, fingerprint)
        self.wm_probe_rejection_reasons = list(reasons)
        if not accepted:
            return False
        if self.wm_aligned:
            # Aligned training uses the current model. Reports are diagnostics;
            # neither a pass nor a rejection can replace/freeze its weights.
            self.wm_last_diagnostic_report = deepcopy(verified)
            return True
        # Validation happens before replacing the live imagination snapshot.
        self.wm_validated.load_state_dict(payload["model"])
        self.wm_validated.requires_grad_(False)
        self.wm_validated.eval()
        self.wm_validated_report = deepcopy(verified)
        self.wm_validated_fingerprint = fingerprint
        self.wm_has_validated_snapshot = True
        return True

    def record_transition(self, states, actions, rewards, next_states, terminated,
                          truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated,
                                  truncated, infos, timestep, timesteps)
        if not isinstance(infos, dict) or "world_model" not in infos:
            raise RuntimeError("WorldModelPPO requires pre-reset infos['world_model'] from the environment collector")
        transition = infos["world_model"]
        required = ("state", "context", "next_state", "actions", "dt", "reward",
                    "terminated", "truncated", "success", "family", "episode_id")
        missing = set(required).difference(transition)
        if missing:
            raise ValueError("World-model transition is missing {}".format(sorted(missing)))
        for key in required:
            value = transition[key]
            if not torch.is_tensor(value) or value.device != self.wm_replay.device:
                raise ValueError("World-model {} must be a tensor on the simulator/learner device".format(key))
            if value.shape[0] != len(self.wm_env_ids):
                raise ValueError("World-model collector must use the fixed selected environment IDs")
        self.wm_replay.add(transition["state"], transition["actions"], transition["next_state"],
                           transition["context"], transition["dt"], transition["reward"],
                           transition["terminated"], transition["truncated"], transition["family"],
                           episode_ids=transition["episode_id"], valid=transition.get("valid"),
                           success=transition["success"])
        self._wm_total_high_steps += 1
        if self.wm_stage_counts is not None:
            self._record_real_stage_transitions(transition)
        if self.wm_low_replay is not None:
            low_steps = transition.get("low_steps")
            if not isinstance(low_steps, (list, tuple)) or len(low_steps) != 8:
                raise ValueError("Fixed8 collector must provide exactly eight low_steps transitions")
            for low in low_steps:
                missing = set(required).difference(low)
                if missing:
                    raise ValueError("Low-level transition is missing {}".format(sorted(missing)))
                for key in required:
                    value = low[key]
                    if (not torch.is_tensor(value) or value.device != self.wm_low_replay.device
                            or value.shape[0] != len(self.wm_env_ids)):
                        raise ValueError("Low-level transition tensors must stay on the selected CUDA slots")
                train_mask = self.wm_low_replay.add(
                    low["state"], low["actions"], low["next_state"], low["context"],
                    low["dt"], low["reward"], low["terminated"], low["truncated"],
                    low["family"], episode_ids=low["episode_id"], valid=low.get("valid"),
                    success=low["success"])
                if self.wm_manipulation:
                    self.wm_candidate.observe_transition(
                        low["state"], low["actions"], low["next_state"], low["context"],
                        low["dt"], train_mask=train_mask)
                    self._wm_total_low_steps += 1

    @torch.no_grad()
    def _record_real_stage_transitions(self, transition):
        previous, state = transition["state"], transition["next_state"]
        previous_action = previous[:, self.wm_adapter.slices["prev_action"]]
        before = captured_stage_flags(self.wm_adapter, previous, previous_action, transition["context"])
        after = captured_stage_flags(self.wm_adapter, state, transition["actions"], transition["context"])
        success = transition["success"].reshape(-1).bool()
        after["completed"] = after["completed"] | success
        failure = (transition["terminated"].reshape(-1).bool()
                   & ~transition["truncated"].reshape(-1).bool() & ~after["completed"])
        row = (self.wm_replay.steps - 1) % self.wm_replay.capacity
        self.wm_stage_counts.update(transition["family"], self.wm_replay.valid[row], before, after, failure)

    @torch.no_grad()
    def _prediction_errors(self, predicted, target):
        physical = self.wm_validated.normalized_sq_error(predicted, target)
        result = {"nse": physical, "confidence_error": physical}
        relative_loss = getattr(self.wm_adapter, "auxiliary_loss", None)
        if relative_loss is not None:
            relative = relative_loss(predicted, target)
            result["relative_error"] = relative
            # The confidence threshold applies to the MAXIMUM of normalized
            # physical and task-relative error for each sample, not their sum.
            # Take the maximum before any batch averaging: physical error in
            # one sample must not hide relative error in another sample.
            result["confidence_error"] = torch.maximum(physical, relative)
        return result

    @torch.no_grad()
    def _validated_errors(self, batch_size=None):
        # Microstep fitting windows and high-decision imagination horizons have
        # different clocks. Validate the latter on genuine high-level records.
        window = (self.wm_config["horizon"] if self.wm_config["fixed8"] else
                  max(self.wm_config["window"], self.wm_config["horizon"]))
        if not self.wm_replay.can_sample(window, "validation"):
            return None
        batch = self.wm_replay.sample_windows(batch_size or self.wm_config["validation_batch_size"],
                                              window, split="validation", balance_families=True)
        predicted = batch.states[0]
        errors = {}
        for step in range(window):
            predicted = self.wm_validated(predicted, batch.actions[step],
                                           batch.context[0] if self.wm_config["fixed8"] else batch.context[step],
                                           batch.dt[step])
            for name, value in self._prediction_errors(predicted, batch.states[step + 1]).items():
                errors.setdefault(name, []).append(value)
            if self.wm_manipulation:
                predicted_contacts = self.wm_adapter.get(predicted, "foot_contacts")
                actual_contacts = self.wm_adapter.get(batch.states[step + 1], "foot_contacts")
                accuracy = (predicted_contacts == actual_contacts).float().mean(-1)
                errors.setdefault("foot_contact_accuracy", []).append(accuracy)
        # Gate every rollout step on its mean per-sample confidence score.
        return {name: torch.stack(values).mean(-1) for name, values in errors.items()}

    @torch.no_grad()
    def _anchor_observations(self, starts):
        observations = self.wm_adapter.observation(starts.state, starts.context)
        if self.memory is not None:
            real = self.memory.get_tensor_by_name("states").reshape(-1, observations.shape[-1])
            if len(real):
                index = torch.randint(len(real), (len(starts.state),), device=real.device)
                observations = torch.cat((observations, real[index]), 0)
        return observations.detach()

    def _update_world_model(self):
        """Run after a completed real PPO update, using separate optimizers."""
        if self.wm_manipulation:
            return self._update_manipulation_world_model()
        if self.wm_aligned:
            return self._update_aligned_world_model()
        config = self.wm_config
        self.wm_ppo_updates += 1
        metrics = {"candidate/" + key: value
                   for key, value in self.wm_trainer.update(
                       self.wm_low_replay if self.wm_low_replay is not None else self.wm_replay).items()}
        metrics.update({"replay_steps": float(self.wm_replay.steps),
                        "replay_memory_mib": self.wm_replay.memory_bytes / 2 ** 20,
                        "on_gpu": float(self.wm_replay.device.type == "cuda")})
        if self.wm_low_replay is not None:
            metrics["low_replay_steps"] = float(self.wm_low_replay.steps)
            metrics["replay_memory_mib"] += self.wm_low_replay.memory_bytes / 2 ** 20
        interval = int(config["probe_interval"])
        if (config["mode"] == "hybrid" and self.maybe_validate is not None and interval > 0
                and self.wm_replay.steps >= config["warmup_steps"]
                and self.wm_replay.steps - self._last_probe_step >= interval):
            self.maybe_validate(self)
            self._last_probe_step = self.wm_replay.steps
        max_age = config["report_max_age_seconds"]
        if self.wm_has_validated_snapshot and max_age is not None:
            age = time.time() - float(self.wm_validated_report.get("created_unix", 0))
            if age > float(max_age) or age < -60:
                self.wm_has_validated_snapshot = False
                self.wm_probe_rejection_reasons = ["report_expired_or_future_dated"]
        if config["mode"] == "fit":
            self.wm_status = "fit_only"
        elif self.wm_replay.steps < config["warmup_steps"]:
            self.wm_status = "warming_up"
        elif not self.wm_has_validated_snapshot:
            self.wm_status = "waiting_for_probe"
        else:
            errors = self._validated_errors()
            if errors is not None:
                for name, values in errors.items():
                    metrics["validated/" + name + "_1step"] = float(values[0])
                    metrics["validated/" + name + "_kstep"] = float(values[-1])
                    metrics["validated/" + name + "_max"] = float(values.max())
            if errors is None:
                self.wm_status = "waiting_for_heldout_windows"
            elif (not bool(torch.isfinite(errors["confidence_error"]).all())
                  or float(errors["confidence_error"].max()) > config["max_validation_error"]):
                self.wm_status = "heldout_error_rejected"
            elif not self.wm_replay.can_sample(1, "train"):
                self.wm_status = "waiting_for_train_starts"
            else:
                starts = self.wm_replay.sample_starts(config["imagination_batch"], split="train",
                                                      balance_families=True,
                                                      recent_fraction=config["recent_fraction"],
                                                      recent_steps=config["recent_steps"])
                with torch.no_grad():
                    next_state = self.wm_validated(starts.state, starts.action, starts.context, starts.dt)
                    start_errors = self._prediction_errors(next_state, starts.next_state)
                    error = start_errors["confidence_error"]
                    weights = (1.0 - error / config["max_start_error"]).clamp(0, 1)
                    weights = torch.where(torch.isfinite(error), weights, torch.zeros_like(weights)).detach()
                for name, values in start_errors.items():
                    metrics["validated/start_" + name + "_mean"] = float(values.mean())
                metrics["validated/trusted_start_fraction"] = float((weights > 0).float().mean())
                if not bool((weights > 0).any()):
                    self.wm_status = "start_error_rejected"
                else:
                    durations = torch.stack([self.wm_adapter.sample_dt(len(starts.state))
                                             for _ in range(config["horizon"])])
                    result = self.wm_imagination.update(starts.state, starts.context,
                                                        weights=weights,
                                                        anchor_observations=self._anchor_observations(starts),
                                                        dt=durations)
                    metrics.update({"bptt/" + key: value for key, value in result.items()})
                    self.wm_status = "hybrid_updated" if result.get("accepted", 0) else "bptt_rejected"
        metrics["waiting_for_probe"] = float(self.wm_status == "waiting_for_probe")
        metrics["validated_snapshot_available"] = float(self.wm_has_validated_snapshot)
        self.last_world_model_metrics = metrics
        for name, value in metrics.items():
            self.track_data("World model / " + name, value)
        return metrics

    def _report_progress(self, message):
        if self.wm_progress_callback is not None:
            self.wm_progress_callback(message)

    @torch.no_grad()
    def _manipulation_diagnostics(self):
        """Held-out diagnostics; never an Actor activation or acceptance gate.

        The multi-step metric replays recorded HIGH commands while recomputing
        the frozen LOW controller from predictions. It is not a new mean-Actor
        real rollout or an off-policy target for the imagined Critic.
        """
        batch_size = self.wm_config["diagnostics_batch_size"]
        result = {}
        errors = self._validated_errors(batch_size=batch_size)
        result["diagnostic/closed_loop_available"] = float(errors is not None)
        if errors is not None:
            for name, value in errors.items():
                result["diagnostic/low_controller_closed_loop_" + name] = float(value.mean())
                result["diagnostic/low_controller_closed_loop_last_" + name] = float(value[-1])
        if not self.wm_replay.can_sample(1, "validation"):
            result["diagnostic/reward_available"] = 0.
            return result
        starts = self.wm_replay.sample_starts(batch_size, split="validation", balance_families=False)
        components = getattr(self.wm_adapter, "reward_components", None)
        if components is not None:
            rewards = components(starts.state, starts.next_state, starts.action, starts.context)
            task_reward, imagined_reward = rewards["task"], rewards["total"]
            for name, value in rewards.items():
                if name not in ("task", "total") and torch.is_tensor(value):
                    result["diagnostic/real_state_" + name] = float(value.float().mean())
        else:
            task = getattr(self.wm_adapter, "task_reward", self.wm_adapter.reward)
            task_reward = task(starts.state, starts.next_state, starts.action, starts.context)
            imagined_reward = self.wm_adapter.reward(
                starts.state, starts.next_state, starts.action, starts.context)
        result.update({"diagnostic/reward_available": 1.,
                       "diagnostic/native_reward": float(starts.rewards.mean()),
                       "diagnostic/reconstructed_task_reward": float(task_reward.mean()),
                       "diagnostic/task_reward_abs_error": float((task_reward-starts.rewards).abs().mean()),
                       "diagnostic/real_state_imagined_reward": float(imagined_reward.mean()),
                       "diagnostic/imagined_native_reward_gap": float((imagined_reward-starts.rewards).mean())})
        outcome = self.wm_adapter.task_outcome(starts.state, starts.next_state, starts.action, starts.context)
        reconstructed = outcome["failed"] | outcome["completed"] | outcome["timeout"]
        result["diagnostic/done_disagreement"] = float(
            (reconstructed != (starts.terminated | starts.truncated)).float().mean())
        for name in ("lifted", "success", "completed", "failed", "timeout"):
            result["diagnostic/stage_" + name + "_fraction"] = float(outcome[name].float().mean())
        result["diagnostic/stage_hold_progress"] = float(outcome["hold_progress"].float().mean())
        result["diagnostic/stage_hold_fraction"] = float(
            outcome["hold"].float().mean() / max(self.wm_adapter.hold_steps, 1))
        result["diagnostic/stage_held_not_complete_fraction"] = float(
            ((outcome["hold"] > 0) & ~outcome["completed"]).float().mean())
        axes = starts.action[:, [0, 1, 2, 3, 4, 5, 7, 8]]
        limits = (torch.ones(8, device=axes.device) if self.wm_adapter.action_coordinates == "normalized_commands"
                  else axes.new_tensor([.02, .02, .02, .06, .06, .06, self.wm_adapter.forward_limit, .6]))
        fractions = (axes.abs() >= limits).float().mean(0)
        for index, value in zip((0, 1, 2, 3, 4, 5, 7, 8), fractions):
            result["diagnostic/action_{}_saturation".format(index)] = float(value)
        mean, _ = self.wm_imagination._policy(self.wm_adapter.observation(starts.state, starts.context))
        opening = mean[:, 6].sigmoid()
        result["diagnostic/gripper_mean_sigmoid_derivative"] = float((opening * (1. - opening)).mean())
        result["diagnostic/gripper_executed_endpoint_fraction"] = float(
            ((starts.action[:, 6] <= 0) | (starts.action[:, 6] >= 1)).float().mean())
        result["diagnostic/real_state_raw_latent_cost_available"] = 0.
        return result

    def _update_manipulation_world_model(self):
        """One data budget, then W/V/bounded-B work, always after real PPO."""
        config = self.wm_config
        self.wm_ppo_updates += 1
        plan = self.wm_schedule.plan(int(self.wm_low_replay.valid_train_transitions),
                                     can_sample=self.wm_low_replay.can_sample(config["window"], "train"))
        ratio = (cosine_ratio(self._wm_total_low_steps, config["lr_decay_low_steps"], config["lr_final_ratio"])
                 if config["lr_schedule"] == "cosine" else 1.)
        for group in self.wm_trainer.optimizer.param_groups:
            group["lr"] = config["learning_rate"] * ratio
        self.wm_imagination.set_lr(config["actor_learning_rate"] * config["auxiliary_strength"] * ratio,
                                   config["critic_learning_rate"] * ratio)
        self._report_progress("manipulation phase={}: {} new train transitions, {} WM batches, "
                              "{} value-only batches, at most {} Actor steps".format(
                                  plan.phase, plan.new_transitions, plan.model_batches,
                                  plan.value_batches, plan.actor_updates))
        started = time.monotonic()
        metrics = {}
        before = self.wm_trainer.updates
        if plan.model_batches:
            def progress(done, total):
                elapsed = time.monotonic() - started
                self._report_progress("manipulation WM {}/{} batches, {:.1f}s elapsed, {:.1f}s remaining".format(
                    done, total, elapsed, elapsed / max(done, 1) * (total - done)))
                log_model_snapshot(self, {"fit/batches_completed": done,
                    "fit/batches_in_block": total, "fit/elapsed_seconds": elapsed})
            fitted = self.wm_trainer.update(self.wm_low_replay, batches=plan.model_batches,
                                            progress_callback=progress)
            metrics.update({"candidate/" + name: value for name, value in fitted.items()})
        model_updates = self.wm_trainer.updates - before
        fit_seconds = time.monotonic() - started
        model_complete = model_updates == plan.model_batches and model_updates > 0
        actor_accepted, actors_completed, values_completed, critic_updates = 0, 0, 0, 0
        branch = {}
        calls = 0
        if (config["mode"] == "hybrid" and model_complete
                and self.wm_replay.can_sample(1, "train")):
            for index in range(plan.value_batches or plan.actor_updates):
                self._report_progress("manipulation {} {}/{}".format(
                    "value preparation" if plan.phase == "V" else "Actor + value",
                    index + 1, plan.value_batches or plan.actor_updates))
                starts = self.wm_replay.sample_starts(
                    config["imagination_batch"], split="train", balance_families=config["balance_families"],
                    recent_fraction=config["recent_fraction"], recent_steps=config["recent_steps"])
                if plan.phase == "V":
                    result = self.wm_imagination.update_value(starts.state, starts.context, dt=.16)
                    values_completed += int(result.get("value_prepared", 0))
                else:
                    result = self.wm_imagination.update(starts.state, starts.context, dt=.16)
                    accepted = bool(result.get("accepted", 0))
                    actor_accepted += int(accepted)
                    # A phase advances only when both policy and its value fit
                    # completed. Rejected work is never carried to a later block.
                    actors_completed += int(accepted and result.get("critic_updates", 0) > 0
                                             and not result.get("critic_skipped_nonfinite", 0))
                critic_updates += int(result.get("critic_updates", 0))
                calls += 1
                for name, value in result.items():
                    branch[name] = branch.get(name, 0.) + value
        completed = self.wm_schedule.complete(plan, model_updates=model_updates,
                                              value_batches=values_completed, actor_updates=actors_completed)
        metrics.update({"imagination/" + name: value / max(calls, 1) for name, value in branch.items()})
        phases = {name: index for index, name in enumerate(("W", "V", "B1", "B2", "B3"))}
        metrics.update({"schedule/phase": float(phases[plan.phase]),
                        "schedule/next_phase": float(phases[self.wm_schedule.phase]),
                        "schedule/boundary_completed": float(completed),
                        "schedule/new_valid_train_transitions": float(plan.new_transitions),
                        "schedule/wm_budget": float(plan.model_batches),
                        "schedule/wm_updates": float(model_updates),
                        "schedule/discarded_wm_batches": float(plan.discarded_batches),
                        "schedule/wm_fraction": self.wm_schedule.credit,
                        "schedule/effective_epochs": model_updates * config["batch_size"] / max(plan.new_transitions, 1),
                        "schedule/actor_budget": float(plan.actor_updates),
                        "schedule/actor_accepted": float(actor_accepted),
                        "schedule/value_batches_completed": float(values_completed),
                        "schedule/critic_updates": float(critic_updates),
                        "candidate/updates": float(self.wm_trainer.updates),
                        "replay_steps": float(self.wm_replay.steps),
                        "low_replay_steps": float(self.wm_low_replay.steps),
                        "total_low_steps": float(self._wm_total_low_steps),
                        "lr_ratio": ratio, "fit_seconds": fit_seconds,
                        "imagination_seconds": time.monotonic() - started - fit_seconds,
                        "on_gpu": float(self.wm_replay.device.type == "cuda")})
        if config["diagnostics_interval"] > 0 and self.wm_ppo_updates % config["diagnostics_interval"] == 0:
            metrics.update(self._manipulation_diagnostics())
        metrics.update(self.wm_stage_counts.metrics())
        interval = int(config["probe_interval"])
        if (self.maybe_validate is not None and interval > 0
                and self.wm_replay.steps - self._last_probe_step >= interval):
            self.maybe_validate(self)
            self._last_probe_step = self.wm_replay.steps
        metrics["update_seconds"] = time.monotonic() - started
        self.wm_status = "manipulation_fit_only" if config["mode"] == "fit" else "manipulation_" + plan.phase
        self.last_world_model_metrics = metrics
        for name, value in metrics.items():
            self.track_data("World model / " + name, value)
        return metrics

    def _update_aligned_world_model(self):
        """Fit current dynamics, BPTT Actor, then fit independent imagined V.

        The UTD clock counts 20 ms vector transitions. Execute the accrued
        updates only after PPO completes, never while collecting its rollout.
        State errors and optional real probes remain diagnostic, not gates.
        """
        config = self.wm_config
        self.wm_ppo_updates += 1
        low_steps = self.wm_low_replay.steps
        eligible = max(0, low_steps - int(config["warmup_steps"]) * 8)
        newly_eligible = max(0, eligible - self._wm_scheduled_low_steps)
        self._wm_scheduled_low_steps = eligible
        self._wm_update_credit += newly_eligible * config["updates_per_low_step"]
        rounds = int(self._wm_update_credit + 1e-9)
        if not self.wm_low_replay.can_sample(config["window"], "train"):
            rounds = 0
        self._wm_update_credit -= rounds
        ratio = (cosine_ratio(low_steps, config["lr_decay_low_steps"], config["lr_final_ratio"])
                 if config["lr_schedule"] == "cosine" else 1.)
        for group in self.wm_trainer.optimizer.param_groups:
            group["lr"] = config["learning_rate"] * ratio
        self.wm_imagination.set_lr(
            config["actor_learning_rate"] * config["auxiliary_strength"] * ratio,
            config["critic_learning_rate"] * ratio)
        started = time.monotonic()
        fit_seconds = imagination_seconds = 0.
        if rounds:
            self._report_progress(
                "high_step={} low_step={}: {} update rounds; model={}x{} windows, "
                "imagination={} starts x {} low steps. Environment progress resumes after this block.".format(
                    self.wm_replay.steps, low_steps, rounds,
                    config["batches_per_update"], config["batch_size"],
                    config["imagination_batch"] if config["mode"] == "hybrid" else 0,
                    config["horizon"] * 8))
        aggregate, bptt_calls, accepted = {}, 0, 0
        for index in range(rounds):
            self._report_progress("round {}/{}: fitting dynamics".format(index + 1, rounds))
            phase_started = time.monotonic()
            fitted = self.wm_trainer.update(self.wm_low_replay)
            fitted_seconds = time.monotonic() - phase_started
            round_metrics = {"candidate/" + key: value for key, value in fitted.items()}
            fit_seconds += fitted_seconds
            for key, value in fitted.items():
                aggregate["candidate/" + key] = aggregate.get("candidate/" + key, 0.) + value
            self.wm_update_rounds += 1
            if config["mode"] == "hybrid" and self.wm_replay.can_sample(1, "train"):
                starts = self.wm_replay.sample_starts(
                    config["imagination_batch"], split="train",
                    balance_families=config["balance_families"],
                    recent_fraction=config["recent_fraction"], recent_steps=config["recent_steps"])
                self._report_progress("round {}/{}: model fit {:.1f}s; starting Actor BPTT + imagined Critic".format(
                    index + 1, rounds, fitted_seconds))
                phase_started = time.monotonic()
                result = self.wm_imagination.update(starts.state, starts.context, dt=0.16)
                imagination_seconds += time.monotonic() - phase_started
                if not result.get("accepted", 0):
                    reasons = [name for name in ("nonfinite_loss", "nonfinite_gradient",
                               "nonfinite_parameter", "nonfinite_optimizer", "kl_rejected", "no_gradient")
                               if result.get(name, 0)]
                    self._report_progress("round {}/{}: Actor update rejected ({})".format(
                        index + 1, rounds, ", ".join(reasons) or "disabled or empty trust weights"))
                for key, value in result.items():
                    aggregate["bptt/" + key] = aggregate.get("bptt/" + key, 0.) + value
                    round_metrics["bptt/" + key] = value
                bptt_calls += 1
                accepted += int(result.get("accepted", 0))
            elapsed = time.monotonic() - started
            remaining = elapsed / (index + 1) * (rounds - index - 1)
            round_metrics.update(round_in_block=index + 1, rounds_in_block=rounds,
                                 elapsed_seconds=elapsed, eta_seconds=remaining,
                                 update_rounds=self.wm_update_rounds)
            log_model_snapshot(self, round_metrics, force=index == 0 or index + 1 == rounds)
            self._report_progress(
                "round {}/{} complete: elapsed {:.1f}s, block ETA {:.1f}s, accepted Actor updates {}/{}".format(
                    index + 1, rounds, elapsed, remaining, accepted, bptt_calls))
        metrics = {key: value / max(1, bptt_calls if key.startswith("bptt/") else rounds)
                   for key, value in aggregate.items()}
        metrics.update({
            "candidate/updates": float(self.wm_trainer.updates),
            "update_rounds": float(self.wm_update_rounds),
            "rounds_this_ppo_update": float(rounds),
            "bptt/accepted_this_ppo_update": float(accepted),
            "bptt/accepted_total": float(self.wm_imagination.accepted_updates),
            "replay_steps": float(self.wm_replay.steps),
            "low_replay_steps": float(low_steps),
            "replay_memory_mib": (self.wm_replay.memory_bytes + self.wm_low_replay.memory_bytes) / 2**20,
            "on_gpu": float(self.wm_replay.device.type == "cuda"),
            "lr_ratio": ratio,
            "model_learning_rate": config["learning_rate"] * ratio,
            "actor_learning_rate": config["actor_learning_rate"] * config["auxiliary_strength"] * ratio,
            "critic_learning_rate": config["critic_learning_rate"] * ratio,
            "update_seconds": time.monotonic() - started,
            "fit_seconds": fit_seconds,
            "imagination_seconds": imagination_seconds,
            "waiting_for_probe": 0., "validated_snapshot_available": 0.,
        })
        self.wm_status = ("warming_up" if low_steps < config["warmup_steps"] * 8 else
                          "collecting_update_credit" if not rounds else
                          "fit_only" if config["mode"] == "fit" else
                          "hybrid_updated" if accepted else "bptt_nonfinite_or_empty")
        interval = int(config["probe_interval"])
        if (self.maybe_validate is not None and interval > 0
                and self.wm_replay.steps - self._last_probe_step >= interval):
            self.maybe_validate(self)
            self._last_probe_step = self.wm_replay.steps
        self.last_world_model_metrics = metrics
        for name, value in metrics.items():
            self.track_data("World model / " + name, value)
        return metrics

    def _update(self, timestep, timesteps):
        started = time.monotonic()
        self._report_progress("high_step={}: updating real PPO".format(self.wm_replay.steps))
        super()._update(timestep, timesteps)
        self._report_progress("real PPO complete in {:.1f}s".format(time.monotonic() - started))
        flush_before_model_update(self, self.wm_replay.steps)
        self._update_world_model()
        if not self.wm_aligned or self.wm_manipulation:
            log_model_snapshot(self, self.last_world_model_metrics, force=True)
        self._report_progress("update block complete in {:.1f}s; resuming environment collection".format(
            time.monotonic() - started))
