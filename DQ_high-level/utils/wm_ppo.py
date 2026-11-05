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
import tempfile
import time

import torch
from skrl.agents.torch.ppo import PPO

from utils.wm_env import TeacherWorldModelAdapter
from utils.world_model import (HierarchicalDynamics, ResidualDynamics, TransitionReplay,
                               WMTrainerConfig, WorldModelTrainer)
from utils.world_model.imagination import ImaginationConfig, ImaginationTrainer
from utils.world_model.probes import (configuration_fingerprint, load_verified_report,
                                     model_fingerprint)
from utils.world_model.recipe import SUPERMIMIC_CONFIG, cosine_ratio
from utils.world_model.replay import SliceStateCodec


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


def _cloned_state(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


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
        return {
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
        }

    def load_state_dict(self, state):
        agent = self.agent
        if state.get("schema") != TRAINING_SCHEMA:
            raise ValueError("Unsupported world-model training checkpoint schema")
        agent._check_metadata(state.get("metadata"))
        agent.wm_trainer.load_state_dict(state["candidate"])
        agent.wm_imagination.load_state_dict(state["imagination"])
        agent.wm_ppo_updates = int(state.get("ppo_updates", 0))
        agent.wm_update_rounds = int(state.get("update_rounds", 0))
        agent._wm_update_credit = 0.0
        agent._wm_scheduled_low_steps = 0
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
        if agent.wm_low_replay is not None:
            agent.wm_low_replay.valid.zero_()
            agent.wm_low_replay.stamp.fill_(-1)
            agent.wm_low_replay._episodes.zero_()
            agent.wm_low_replay.steps = 0
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
        self.wm_config.update(wm_config or {})
        self.maybe_validate = maybe_validate
        config = self.wm_config
        self.wm_aligned = config["recipe"] == "supermimic"
        if config["recipe"] not in ("gated", "supermimic"):
            raise ValueError("Unknown world-model recipe")
        if self.wm_aligned and not config["fixed8"]:
            raise ValueError("SuperMimic-aligned training requires fixed8")
        if config["mode"] not in ("fit", "hybrid"):
            raise ValueError("world-model mode must be fit or hybrid")
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
        self.wm_replay = TransitionReplay(selected, config["replay_capacity"],
                                          self.wm_adapter.state_dim, self.wm_adapter.action_dim,
                                          self.wm_adapter.context_dim, learner_device,
                                          holdout_fraction=config["holdout_fraction"])
        codec = (SliceStateCodec(self.wm_adapter.state_dim, self.wm_adapter.slices,
                 ("robot", "object", "dof_pos", "dof_vel", "table", "table_height", "foot_contacts"))
                 if config["fixed8"] and config["compact_low_replay"] else None)
        self.wm_low_replay = (TransitionReplay(
            selected, config["low_replay_capacity"], self.wm_adapter.state_dim,
            self.wm_adapter.low_action_dim, self.wm_adapter.context_dim,
            learner_device, holdout_fraction=config["holdout_fraction"], state_codec=codec)
            if config["fixed8"] else None)
        self.wm_trainer = WorldModelTrainer(self.wm_candidate, WMTrainerConfig(
            window=config["window"], batches_per_update=config["batches_per_update"],
            batch_size=config["batch_size"], learning_rate=config["learning_rate"],
            grad_clip=config["model_grad_clip"], termination_weight=config["termination_weight"],
            contact_weight=config["contact_weight"],
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
        self._last_probe_step = 0
        self.wm_status = "collecting"
        self.last_world_model_metrics = {}
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
            return metadata()
        # Compatible with tensor-only adapter fixtures. Production supplies a
        # detailed task/geometry contract through TeacherWorldModelAdapter.
        return {"schema": "dq-teacher-state-v1", "state_dim": self.wm_adapter.state_dim,
                "delta_dim": self.wm_adapter.delta_dim, "context_dim": self.wm_adapter.context_dim,
                "context_feature_dim": self.wm_adapter.context_feature_dim,
                "slices": {key: [value.start, value.stop]
                           for key, value in self.wm_adapter.slices.items()}}

    def _check_metadata(self, metadata):
        if metadata is None or configuration_fingerprint(metadata) != configuration_fingerprint(self.wm_metadata):
            raise ValueError("World-model checkpoint does not match this state schema/task contract")

    def candidate_payload(self):
        teacher = {"policy": self.policy, "value": self.value,
                   "state_preprocessor": self._state_preprocessor,
                   "value_preprocessor": self._value_preprocessor}
        return {"schema": CANDIDATE_SCHEMA, "metadata": deepcopy(self.wm_metadata),
                "config": deepcopy(self.wm_config), "model": _cloned_state(self.wm_candidate),
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
                       "model": training["candidate"]["model"],
                       "updates": training["candidate"].get("updates", 0)}
        if payload.get("schema") != CANDIDATE_SCHEMA:
            raise ValueError("Expected a DQ world-model candidate or full teacher checkpoint")
        self._check_metadata(payload.get("metadata"))
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
                self.wm_low_replay.add(
                    low["state"], low["actions"], low["next_state"], low["context"],
                    low["dt"], low["reward"], low["terminated"], low["truncated"],
                    low["family"], episode_ids=low["episode_id"], valid=low.get("valid"),
                    success=low["success"])

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
    def _validated_errors(self):
        # Microstep fitting windows and high-decision imagination horizons have
        # different clocks. Validate the latter on genuine high-level records.
        window = (self.wm_config["horizon"] if self.wm_config["fixed8"] else
                  max(self.wm_config["window"], self.wm_config["horizon"]))
        if not self.wm_replay.can_sample(window, "validation"):
            return None
        batch = self.wm_replay.sample_windows(self.wm_config["validation_batch_size"],
                                              window, split="validation", balance_families=True)
        predicted = batch.states[0]
        errors = {}
        for step in range(window):
            predicted = self.wm_validated(predicted, batch.actions[step],
                                           batch.context[0] if self.wm_config["fixed8"] else batch.context[step],
                                           batch.dt[step])
            for name, value in self._prediction_errors(predicted, batch.states[step + 1]).items():
                errors.setdefault(name, []).append(value)
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
        aggregate, bptt_calls, accepted = {}, 0, 0
        for _ in range(rounds):
            fitted = self.wm_trainer.update(self.wm_low_replay)
            for key, value in fitted.items():
                aggregate["candidate/" + key] = aggregate.get("candidate/" + key, 0.) + value
            self.wm_update_rounds += 1
            if config["mode"] == "hybrid" and self.wm_replay.can_sample(1, "train"):
                starts = self.wm_replay.sample_starts(
                    config["imagination_batch"], split="train",
                    balance_families=config["balance_families"],
                    recent_fraction=config["recent_fraction"], recent_steps=config["recent_steps"])
                result = self.wm_imagination.update(starts.state, starts.context, dt=0.16)
                for key, value in result.items():
                    aggregate["bptt/" + key] = aggregate.get("bptt/" + key, 0.) + value
                bptt_calls += 1
                accepted += int(result.get("accepted", 0))
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
        super()._update(timestep, timesteps)
        self._update_world_model()
