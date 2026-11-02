"""Real PPO with separately fitted GPU dynamics and validated-snapshot BPTT.

The candidate world model keeps learning from real transitions. Actor BPTT
uses a different, fixed model instance, activated only by a matching isolated
Isaac Gym action-effect report. Improving a prediction loss does not certify
the derivative of a contact model. Real PPO and SGS keep their original data
and update rules; imagined transitions never enter PPO memory or SGS results.
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
from utils.world_model import (ResidualDynamics, TransitionReplay,
                               WMTrainerConfig, WorldModelTrainer)
from utils.world_model.imagination import ImaginationConfig, ImaginationTrainer
from utils.world_model.probes import (configuration_fingerprint, load_verified_report,
                                     model_fingerprint)


DEFAULT_WORLD_MODEL_CONFIG = {
    "mode": "fit",                 # fit | hybrid (hybrid still requires a passed probe)
    "replay_envs": 128,
    "replay_capacity": 256,
    "window": 3,
    "batch_size": 128,
    "batches_per_update": 2,
    "hidden": [512, 512, 512],
    "term_hidden": [64, 64],
    "learning_rate": 2e-4,
    "model_grad_clip": 1.0,
    "termination_weight": 0.1,
    "warmup_steps": 96,
    "horizon": 3,
    "imagination_batch": 128,
    "actor_learning_rate": 1e-5,
    "auxiliary_strength": 0.05,
    "max_kl": 0.005,
    "max_grad_norm": 0.5,
    "max_validation_error": 0.25,
    "max_start_error": 0.25,
    "holdout_fraction": 0.1,
    "recent_fraction": 0.5,
    "recent_steps": 32,
    "validation_batch_size": 128,
    "use_tail_value": True,
    "probe_interval": 0,           # 0 means explicit external/manual validation
    "report_max_age_seconds": None,
    "candidate_checkpoint": "",
    "validated_checkpoint": "",
    "probe_report": "",
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
        }

    def load_state_dict(self, state):
        agent = self.agent
        if state.get("schema") != TRAINING_SCHEMA:
            raise ValueError("Unsupported world-model training checkpoint schema")
        agent._check_metadata(state.get("metadata"))
        agent.wm_trainer.load_state_dict(state["candidate"])
        agent.wm_imagination.load_state_dict(state["imagination"])
        agent.wm_ppo_updates = int(state.get("ppo_updates", 0))
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
    simulators. No callback or passing report means ordinary PPO plus fitting.
    """

    def __init__(self, *args, wm_env, wm_config=None, maybe_validate=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.wm_env = wm_env
        self.wm_config = deepcopy(DEFAULT_WORLD_MODEL_CONFIG)
        self.wm_config.update(wm_config or {})
        self.maybe_validate = maybe_validate
        config = self.wm_config
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
        if max(config["window"], config["horizon"]) > config["replay_capacity"]:
            raise ValueError("Replay capacity must cover model and imagination windows")
        if self.policy is self.value:
            raise ValueError("World-model BPTT requires independent Actor and Critic")
        if self._rewards_shaper is not None:
            raise ValueError("World-model rewards must use the same raw task units as real PPO")
        environment_device = torch.empty(0, device=wm_env.device).device
        learner_device = torch.empty(0, device=self.device).device
        if environment_device != learner_device:
            raise ValueError("World-model training requires simulator and learner on the same device")
        self.wm_adapter = TeacherWorldModelAdapter(wm_env)
        self.wm_metadata = self._adapter_metadata()
        selected = min(int(config["replay_envs"]), int(wm_env.num_envs))
        self.wm_env_ids = torch.arange(selected, device=learner_device)
        # The external collector uses the exact adapter and selected slots.
        wm_env.world_model_adapter = self.wm_adapter
        wm_env.world_model_env_ids = self.wm_env_ids
        self.wm_candidate = ResidualDynamics(self.wm_adapter, hidden=config["hidden"],
                                             term_hidden=config["term_hidden"], device=learner_device)
        self.wm_validated = ResidualDynamics(self.wm_adapter, hidden=config["hidden"],
                                             term_hidden=config["term_hidden"], device=learner_device)
        self.wm_validated.requires_grad_(False)
        self.wm_validated.eval()
        self.wm_replay = TransitionReplay(selected, config["replay_capacity"],
                                          self.wm_adapter.state_dim, self.wm_adapter.action_dim,
                                          self.wm_adapter.context_dim, learner_device,
                                          holdout_fraction=config["holdout_fraction"])
        self.wm_trainer = WorldModelTrainer(self.wm_candidate, WMTrainerConfig(
            window=config["window"], batches_per_update=config["batches_per_update"],
            batch_size=config["batch_size"], learning_rate=config["learning_rate"],
            grad_clip=config["model_grad_clip"], termination_weight=config["termination_weight"],
            recent_fraction=config["recent_fraction"], recent_steps=config["recent_steps"],
            validation_batch_size=config["validation_batch_size"]))
        self.wm_imagination = ImaginationTrainer(
            self.wm_validated, self.wm_adapter, self.policy, self.value,
            ImaginationConfig(horizon=config["horizon"], gamma=self._discount_factor,
                              lr=config["actor_learning_rate"], strength=config["auxiliary_strength"],
                              max_grad_norm=config["max_grad_norm"], max_kl=config["max_kl"],
                              use_tail_value=config["use_tail_value"]),
            state_preprocessor=self._state_preprocessor,
            value_preprocessor=self._value_preprocessor)
        self.wm_has_validated_snapshot = False
        self.wm_validated_report = None
        self.wm_validated_fingerprint = None
        self.wm_probe_rejection_reasons = []
        self.wm_ppo_updates = 0
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

    @torch.no_grad()
    def _prediction_errors(self, predicted, target):
        physical = self.wm_validated.normalized_sq_error(predicted, target)
        result = {"nse": physical, "confidence_error": physical}
        relative_loss = getattr(self.wm_adapter, "auxiliary_loss", None)
        if relative_loss is not None:
            relative = relative_loss(predicted, target)
            result["relative_error"] = relative
            # The default .25 threshold applies to the MAXIMUM of normalized
            # physical and task-relative error for each sample, not their sum.
            # Take the maximum before any batch averaging: physical error in
            # one sample must not hide relative error in another sample.
            result["confidence_error"] = torch.maximum(physical, relative)
        return result

    @torch.no_grad()
    def _validated_errors(self):
        window = max(self.wm_config["window"], self.wm_config["horizon"])
        if not self.wm_replay.can_sample(window, "validation"):
            return None
        batch = self.wm_replay.sample_windows(self.wm_config["validation_batch_size"],
                                              window, split="validation", balance_families=True)
        predicted = batch.states[0]
        errors = {}
        for step in range(window):
            predicted = self.wm_validated(predicted, batch.actions[step],
                                           batch.context[step], batch.dt[step])
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
        config = self.wm_config
        self.wm_ppo_updates += 1
        metrics = {"candidate/" + key: value
                   for key, value in self.wm_trainer.update(self.wm_replay).items()}
        metrics.update({"replay_steps": float(self.wm_replay.steps),
                        "replay_memory_mib": self.wm_replay.memory_bytes / 2 ** 20,
                        "on_gpu": float(self.wm_replay.device.type == "cuda")})
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

    def _update(self, timestep, timesteps):
        super()._update(timestep, timesteps)
        self._update_world_model()
