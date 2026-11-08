"""Independent real collection / WM / BPTT Agent for the fixed8 DQ teacher.

The schedule and Actor/Critic workflow follow supermimic_manipulation's runner.
This Agent has no PPO optimizer, GAE, likelihood replay or real-data Critic.
Isaac Gym, the frozen low policy, SGS resets and EKF remain task-owned.
"""

from copy import deepcopy
import math
import os
from pathlib import Path
import tempfile
import time

import torch
from skrl.agents.torch import Agent

from utils.wm_fixed8 import Fixed8TeacherWorldModelAdapter
from utils.world_model import HierarchicalDynamics, TransitionReplay, WMTrainerConfig, WorldModelTrainer
from utils.world_model.imagination import ImaginationConfig, ImaginationTrainer
from utils.world_model.probes import configuration_fingerprint, load_verified_report, model_fingerprint
from utils.world_model.reference_recipe import REFERENCE_CONFIG
from utils.world_model.replay import SliceStateCodec
from utils.world_model.stage_counts import RealStageTransitionCounts, captured_stage_flags
from utils.world_model.live_logging import flush_before_model_update, log_model_snapshot


def reference_metadata(adapter, config):
    """One contract constructor shared by training and isolated probe loading."""
    result = adapter.metadata()
    result["learning_algorithm"] = "reference_pure_bptt_v1"
    result["policy_architecture"] = "reference_flat_deterministic_v1"
    result["policy_network"] = {"hidden": list(config["actor_hidden"]),
        "amp": config["policy_amp"], "normalization": ("empirical_real_next" if
            config["actor_running_normalization"] else "none"),
        "action_noise": config["action_noise"], "last_gain": config["actor_last_gain"]}
    result["model_execution"] = {name: config[name] for name in
        ("recipe", "layer_norm", "running_normalization", "amp", "dynamics_target")}
    if config["dynamics_target"] == "physical_increment":
        from utils.world_model.physical_increments import physical_increment_metadata
        result["model_coordinates"] = physical_increment_metadata()
    else:
        result["model_coordinates"] = {"dynamics_target": "residual", "version": 1,
            "frame": "PRE robot heading", "units": "fixed-scale semi-implicit residual"}
    return result


class ReferenceSchedule:
    """Source fractional 2/3-HIGH collection, independent LOW-window budget."""

    def __init__(self, config):
        self.config = config
        self.warmup = int(config["collection_warmup_high_steps"])
        self.rate = float(config["collection_updates_per_high_step"])
        if self.warmup < 1 or not 0 < self.rate <= 1:
            raise ValueError("Reference collection needs positive warmup and rate in (0, 1]")
        self.iteration = 0
        self.collection_fraction = 0.
        self.model_fraction = 0.
        self.seen_valid = 0
        self.discarded_batches = 0
        self.next_boundary = self.warmup + self._collection_steps()

    def _collection_steps(self):
        self.collection_fraction += 1/self.rate
        steps = math.floor(self.collection_fraction+1e-12)
        self.collection_fraction -= steps
        return steps

    def ready(self, high_steps, valid_low):
        # Source warmup supplies replay/statistics, not an accumulated optimizer
        # debt. The first iteration collects new rows before fitting.
        if high_steps <= self.warmup:
            self.seen_valid = int(valid_low)
        return high_steps >= self.next_boundary

    def plan(self, valid_low):
        new = int(valid_low)-self.seen_valid
        if new < 0:
            raise ValueError("Reference valid-transition counter moved backwards")
        self.seen_valid = int(valid_low)
        self.model_fraction += self.config["wm_epochs"]*new/self.config["batch_size"]
        budget = math.floor(self.model_fraction+1e-12)
        self.model_fraction -= budget
        cap = int(self.config["wm_max_batches"])
        discarded = max(0, budget-cap) if cap else 0
        self.discarded_batches += discarded
        phase = self.iteration-int(self.config["world_model_only_iterations"])
        actor = phase >= 0 and phase % int(self.config["policy_update_interval"]) == 0
        return {"new_valid_low": new, "model_batches": budget-discarded,
                "discarded_batches": discarded, "actor": actor, "iteration": self.iteration}

    def advance(self):
        self.iteration += 1
        self.next_boundary += self._collection_steps()

    def state_dict(self):
        return {key: deepcopy(value) for key, value in self.__dict__.items() if key != "config"}


class _ReferenceCheckpoint:
    def __init__(self, agent):
        self.agent = agent

    def state_dict(self):
        agent = self.agent
        return {"schema": "dq-reference-teacher-training-v1", "metadata": deepcopy(agent.wm_metadata),
                "config": deepcopy(agent.wm_config), "candidate": agent.wm_trainer.state_dict(),
                "imagination": agent.wm_imagination.state_dict(),
                "schedule": agent.reference_schedule.state_dict(),
                "audit": agent._checkpoint_audit(), "exact_resume_supported": False}

    def load_state_dict(self, payload):
        if payload.get("schema") != "dq-reference-teacher-training-v1":
            raise ValueError("Reference checkpoints cannot load hybrid/PPO training state")
        self.agent._check_metadata(payload.get("metadata"))
        self.agent.wm_trainer.load_state_dict(payload["candidate"])
        self.agent.wm_imagination.load_state_dict(payload["imagination"])
        # Full simulator/replay restoration is unsupported. Entry rejects train
        # resume; this loader is useful only for explicit inspection/evaluation.
        self.agent.wm_status = "loaded_for_inspection_replay_not_restored"


class ReferenceTeacherAgent(Agent):
    """One Actor optimizer; genuine PRE/POST replay; fixed-noise real actions."""

    def __init__(self, *args, wm_env, wm_config=None, maybe_validate=None, **kwargs):
        if kwargs.get("memory") is not None:
            raise ValueError("Reference training does not use PPO/likelihood memory")
        super().__init__(*args, **kwargs)
        self.policy = self.models["policy"]
        if set(self.models) != {"policy"} or not getattr(self.policy, "reference_policy", False):
            raise ValueError("Reference training requires only a ReferenceTeacherPolicy")
        self.value = None
        self._state_preprocessor = self._empty_preprocessor
        self._value_preprocessor = self._empty_preprocessor
        self.wm_env, self.maybe_validate = wm_env, maybe_validate
        self.wm_config = {**deepcopy(REFERENCE_CONFIG), **deepcopy(wm_config or {})}
        config = self.wm_config
        self.checkpoint_modules["policy"] = self.policy
        if config["recipe"] != "reference" or config["mode"] not in ("off", "fit", "hybrid"):
            raise ValueError("ReferenceTeacherAgent requires reference configuration")
        supported = {"fixed8": True, "value_source": "imagined", "use_tail_value": True,
                     "kl_rollback": False, "auxiliary_strength": 1., "auxiliary_weight": 0.,
                     "fixed_preprocessing": False, "observation_noise_scale": 0.,
                     "holdout_mode": "environment"}
        for name, value in supported.items():
            if config.get(name, value) != value:
                raise ValueError("Reference training requires {}={!r}".format(name, value))
        if config["lr_schedule"] not in ("cosine", "constant"):
            raise ValueError("Reference learning-rate schedule must be cosine or constant")
        if not 0 < config["gamma"] < 1 or not 0 < config["lr_final_ratio"] <= 1:
            raise ValueError("Reference gamma/final learning-rate ratio is out of range")
        if config["mode"] == "off":
            self.training = False
            return
        if torch.empty(0, device=wm_env.device).device != torch.empty(0, device=self.device).device:
            raise ValueError("Reference simulator, replay and learner must share the device")
        self.wm_adapter = Fixed8TeacherWorldModelAdapter(wm_env)
        selected = min(config["replay_envs"], wm_env.num_envs)
        self.wm_env_ids = torch.arange(selected, device=self.device)
        wm_env.world_model_adapter, wm_env.world_model_env_ids = self.wm_adapter, self.wm_env_ids
        self.wm_candidate = HierarchicalDynamics(self.wm_adapter, hidden=config["hidden"],
            contact_hidden=config["contact_hidden"], nominal_dt=.16, device=self.device,
            layer_norm=config["layer_norm"], running_normalization=config["running_normalization"],
            amp=config["amp"], dynamics_target=config["dynamics_target"])
        self.wm_validated = self.wm_candidate  # compatibility: single current model
        self.wm_has_validated_snapshot = False
        self.wm_validated_report = None
        self.wm_metadata = reference_metadata(self.wm_adapter, config)
        options = dict(holdout_fraction=config["holdout_fraction"], holdout_mode="environment")
        self.wm_replay = TransitionReplay(selected, config["replay_capacity"],
            self.wm_adapter.state_dim, self.wm_adapter.action_dim, self.wm_adapter.context_dim,
            self.device, **options)
        codec = (SliceStateCodec(self.wm_adapter.state_dim, self.wm_adapter.slices,
            ("robot", "object", "dof_pos", "dof_vel", "table", "table_height", "foot_contacts"))
            if config["compact_low_replay"] else None)
        self.wm_low_replay = TransitionReplay(selected, config["low_replay_capacity"],
            self.wm_adapter.state_dim, self.wm_adapter.low_action_dim, self.wm_adapter.context_dim,
            self.device, state_codec=codec, allow_partial_windows=config["allow_partial_windows"], **options)
        self.wm_trainer = WorldModelTrainer(self.wm_candidate, WMTrainerConfig(
            window=config["window"], batch_size=config["batch_size"], batches_per_update=1,
            learning_rate=config["learning_rate"], grad_clip=config["model_grad_clip"],
            state_loss_reduction=config["state_loss_reduction"],
            contact_weight=config["contact_weight"], auxiliary_weight=0., recent_fraction=config["recent_fraction"],
            recent_steps=config["recent_steps"], balance_families=config["balance_families"],
            validation_batch_size=config["validation_batch_size"],
            microbatch_size=config["model_microbatch_size"]))
        self.wm_imagination = ImaginationTrainer(self.wm_candidate, self.wm_adapter,
            self.policy, None, ImaginationConfig(horizon=config["horizon"], gamma=config["gamma"],
                lr=config["actor_learning_rate"], strength=1., max_grad_norm=config["max_grad_norm"],
                grad_clip_value=config["grad_clip_value"], actor_betas=tuple(config["actor_betas"]),
                kl_rollback=False, value_source="imagined", reference_mode=True,
                objective_reduction=config["actor_objective_reduction"],
                critic_running_norm=config["critic_running_normalization"],
                critic_activation=config["critic_activation"], critic_amp=config["policy_amp"],
                critic_hidden=tuple(config["critic_hidden"]), critic_lr=config["critic_learning_rate"],
                critic_epochs=config["critic_epochs"], critic_minibatches=config["critic_minibatches"],
                critic_gae_lambda=config["critic_gae_lambda"],
                critic_max_grad_norm=config["critic_max_grad_norm"],
                chunk_size=config["imagination_chunk_size"], dt=.16))
        self.reference_schedule = ReferenceSchedule(config)
        self.wm_schedule = None
        self.wm_update_rounds = self.wm_ppo_updates = 0
        self._wm_total_high_steps = self._wm_total_low_steps = 0
        self.wm_status = "reference_real_warmup"
        self.last_world_model_metrics = {}
        self.wm_progress_callback = None
        self._last_probe_step = 0
        self.wm_probe_rejection_reasons = []
        bank = getattr(wm_env, "sgs_bank", None)
        self.wm_stage_counts = RealStageTransitionCounts(
            bank.family_names if bank is not None else ("full_task",), self.device)
        self.checkpoint_modules.update(policy=self.policy, world_model_training=_ReferenceCheckpoint(self))
        if config.get("candidate_checkpoint"):
            self.load_candidate(config["candidate_checkpoint"])

    def _report_progress(self, message):
        if callable(self.wm_progress_callback):
            self.wm_progress_callback(message)

    @torch.no_grad()
    def act(self, states, timestep=0, timesteps=0):
        result = self.policy.act({"states": states, "explore": self.training and self.wm_config["mode"] != "off"}, role="policy")
        if not bool(torch.isfinite(result[0]).all()):
            raise FloatingPointError("Reference Actor produced nonfinite real commands; refusing simulator step")
        return result

    @torch.no_grad()
    def record_transition(self, states, actions, rewards, next_states, terminated,
                          truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated,
                                  truncated, infos, timestep, timesteps)
        if not self.training or self.wm_config["mode"] == "off":
            return
        self.policy.observe_real_next(next_states)
        transition = infos.get("world_model")
        if not isinstance(transition, dict) or len(transition.get("low_steps", ())) != 8:
            raise ValueError("Reference collector requires one PRE/POST high row and eight low rows")

        def append(replay, row):
            required = ("state", "actions", "next_state", "context", "dt", "reward", "terminated", "truncated", "family")
            for name in required:
                if name not in row or row[name].device != replay.device or len(row[name]) != len(self.wm_env_ids):
                    raise ValueError("Reference replay rows must stay on the selected learner device: " + name)
            return replay.add(*(row[name] for name in required), episode_ids=row["episode_id"],
                              valid=row.get("valid"), success=row["success"])

        append(self.wm_replay, transition)
        for row in transition["low_steps"]:
            train_mask = append(self.wm_low_replay, row)
            self.wm_candidate.observe_transition(row["state"], row["actions"], row["next_state"],
                row["context"], row["dt"], train_mask=train_mask)
        self._wm_total_high_steps += 1
        self._wm_total_low_steps += 8
        before = captured_stage_flags(self.wm_adapter, transition["state"],
            self.wm_adapter.get(transition["state"], "prev_action"), transition["context"])
        after = captured_stage_flags(self.wm_adapter, transition["next_state"],
            transition["actions"], transition["context"])
        after["completed"] |= transition["success"].reshape(-1).bool()
        failed = transition["terminated"].reshape(-1).bool() & ~after["completed"]
        row = (self.wm_replay.steps-1) % self.wm_replay.capacity
        self.wm_stage_counts.update(transition["family"], self.wm_replay.valid[row], before, after, failed)

    def post_interaction(self, timestep, timesteps):
        if self.training and self.wm_config["mode"] != "off" and self.reference_schedule.ready(
                self._wm_total_high_steps, self.wm_low_replay.valid_train_transitions):
            self._update_reference()
        super().post_interaction(timestep, timesteps)

    def _update_reference(self):
        config, schedule = self.wm_config, self.reference_schedule
        flush_before_model_update(self, self.wm_replay.steps)
        plan = schedule.plan(self.wm_low_replay.valid_train_transitions)
        iteration = plan["iteration"]
        fraction = min(iteration/config["lr_decay_iterations"], 1.)
        end = config["lr_final_ratio"]
        ratio = (end+(1-end)*.5*(1+math.cos(math.pi*fraction))
                 if config["lr_schedule"] == "cosine" else 1.)
        for group in self.wm_trainer.optimizer.param_groups:
            group["lr"] = config["learning_rate"]*ratio
        self.wm_imagination.set_lr(config["actor_learning_rate"]*ratio, config["critic_learning_rate"]*ratio)
        self._report_progress("reference iteration={} high_step={}: {} WM batches, Actor={} (no PPO)".format(
            iteration, self._wm_total_high_steps, plan["model_batches"], int(plan["actor"] and config["mode"] == "hybrid")))
        started = time.monotonic()
        metrics = {}
        if plan["model_batches"]:
            def fit_progress(done, total):
                elapsed = time.monotonic() - started
                self._report_progress("reference WM {}/{} batches, {:.1f}s".format(done, total, elapsed))
                log_model_snapshot(self, {"fit/batches_completed": done,
                    "fit/batches_in_block": total, "fit/elapsed_seconds": elapsed})
            fit = self.wm_trainer.update(self.wm_low_replay, batches=plan["model_batches"],
                progress_callback=fit_progress)
            metrics.update({"candidate/"+name: value for name, value in fit.items()})
            if fit.get("skipped_nonfinite", 0):
                raise FloatingPointError("Reference world-model fitting produced nonfinite loss/gradient")
        fit_seconds = time.monotonic()-started
        if (plan["actor"] and config["mode"] == "hybrid" and self.wm_trainer.updates
                and self.wm_replay.can_sample(1, "train")):
            starts = self.wm_replay.sample_starts(config["imagination_batch"], split="train",
                balance_families=config["balance_families"], recent_fraction=config["recent_fraction"],
                recent_steps=config["recent_steps"])
            imagined = self.wm_imagination.update(starts.state, starts.context, dt=.16)
            metrics.update({"imagination/"+name: value for name, value in imagined.items()})
            if not imagined.get("accepted", 0) or imagined.get("critic_skipped_nonfinite", 0):
                raise FloatingPointError("Reference Actor/Critic update failed; inspect numerical diagnostics: " + str(imagined))
        schedule.advance()
        self.wm_update_rounds += 1
        self.wm_status = ("reference_fit_only" if config["mode"] == "fit" else
                          "reference_actor_value" if plan["actor"] else "reference_model_preparation")
        metrics.update({"reference/iteration": float(schedule.iteration),
            "reference/new_valid_low_transitions": float(plan["new_valid_low"]),
            "reference/model_batches": float(plan["model_batches"]),
            "reference/discarded_model_batches": float(plan["discarded_batches"]),
            "reference/actor_scheduled": float(plan["actor"]), "reference/ppo_updates": 0.,
            "reference/next_high_boundary": float(schedule.next_boundary),
            "fit_seconds": fit_seconds, "update_seconds": time.monotonic()-started,
            "total_low_steps": float(self._wm_total_low_steps), "replay_steps": float(self.wm_replay.steps),
            "lr_ratio": ratio, "on_gpu": float(self.device.type == "cuda")})
        interval = config["diagnostics_interval"]
        if interval and schedule.iteration % interval == 0:
            metrics.update(self._diagnostics())
        metrics.update(self.wm_stage_counts.metrics())
        self.last_world_model_metrics = metrics
        log_model_snapshot(self, metrics, force=True)
        for name, value in metrics.items():
            self.track_data("World model / "+name, value)
        if (self.maybe_validate and config["probe_interval"]
                and self._wm_total_high_steps-self._last_probe_step >= config["probe_interval"]):
            self.maybe_validate(self)
            self._last_probe_step = self._wm_total_high_steps
        self._report_progress("reference iteration {} complete in {:.1f}s".format(schedule.iteration, time.monotonic()-started))
        return metrics

    @torch.no_grad()
    def _diagnostics(self):
        count, horizon = self.wm_config["diagnostics_batch_size"], self.wm_config["horizon"]
        metrics = {}
        if self.wm_replay.can_sample(horizon, "validation"):
            batch = self.wm_replay.sample_windows(count, horizon, split="validation", balance_families=False)
            predicted, errors = batch.states[0], []
            for step in range(horizon):
                predicted = self.wm_candidate.step(predicted, batch.actions[step], batch.context[0], .16)
                errors.append(self.wm_adapter.auxiliary_loss(predicted, batch.states[step+1]).mean())
            metrics["diagnostic/closed_loop_relative_error"] = float(torch.stack(errors).mean())
            metrics["diagnostic/closed_loop_last_relative_error"] = float(errors[-1])
        if self.wm_replay.can_sample(1, "validation"):
            batch = self.wm_replay.sample_starts(count, split="validation", balance_families=False)
            parts = self.wm_adapter.reward_components(batch.state, batch.next_state, batch.action, batch.context)
            for name, value in parts.items():
                if torch.is_tensor(value):
                    metrics["diagnostic/reward_"+name] = float(value.mean())
            metrics["diagnostic/native_reward"] = float(batch.rewards.mean())
            mean = self.policy.forward(self.wm_adapter.observation(batch.state, batch.context))
            for index in (0, 1, 2, 3, 4, 5, 7, 8):
                metrics["diagnostic/action_{}_saturation".format(index)] = float((mean[:, index].abs() >= 1).float().mean())
            opening = mean[:, 6].sigmoid()
            metrics["diagnostic/gripper_mean_sigmoid_derivative"] = float((opening*(1-opening)).mean())
        return metrics

    def _checkpoint_audit(self):
        return {"algorithm": "pure_bptt_no_ppo", "schedule": self.reference_schedule.state_dict(),
                "high_vector_steps": self._wm_total_high_steps, "low_vector_steps": self._wm_total_low_steps,
                "actor_normalization_real_next_count": int(self.policy.obs_norm.count),
                "source": "supermimic_manipulation", "exact_resume_supported": False,
                "resolved_config_sha256": configuration_fingerprint(self.wm_config)}

    def _check_metadata(self, metadata):
        if configuration_fingerprint(metadata) != configuration_fingerprint(self.wm_metadata):
            raise ValueError("Reference checkpoint state/task/model contract mismatch")

    def candidate_payload(self):
        critic = self.wm_imagination.imagined_critic
        return {"schema": "dq-world-model-candidate-v1", "metadata": deepcopy(self.wm_metadata),
                "config": deepcopy(self.wm_config), "model": deepcopy(self.wm_candidate.state_dict()),
                "updates": self.wm_trainer.updates, "collected_steps": self.wm_replay.steps,
                "teacher": {"policy": deepcopy(self.policy.state_dict()), "value": None,
                            "state_preprocessor": None, "value_preprocessor": None},
                "teacher_config": deepcopy(getattr(self, "wm_teacher_config", self.wm_env.cfg)),
                "teacher_arguments": deepcopy(getattr(self, "wm_teacher_arguments", {})),
                "imagined_critic": deepcopy(critic.state_dict()),
                "imagined_critic_updates": self.wm_imagination.imagined_critic_trainer.updates,
                "imagined_gamma": self.wm_imagination.cfg.gamma,
                "imagined_use_tail_value": True, "imagined_bootstrap_timeouts": False,
                "global_step": int(getattr(self.wm_env, "global_step_counter", self.wm_replay.steps)),
                "sgs_bank_fingerprint": getattr(getattr(self.wm_env, "sgs_bank", None), "fingerprint", None),
                "audit": self._checkpoint_audit()}

    def save_candidate(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = self.candidate_payload()
        with tempfile.NamedTemporaryFile(dir=str(destination.parent), suffix=".tmp", delete=False) as handle:
            temporary = handle.name
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return payload

    def load_candidate(self, path):
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self._check_metadata(payload.get("metadata"))
        self.wm_candidate.load_state_dict(payload["model"])
        self.wm_trainer.updates = int(payload.get("updates", 0))

    def activate_validated(self, payload, report):
        # Reference probes remain diagnostics and never swap the single model.
        if isinstance(payload, (str, Path)):
            payload = torch.load(payload, map_location=self.device, weights_only=False)
        self._check_metadata(payload.get("metadata"))

        class Weights:
            def state_dict(self):
                return payload["model"]

        accepted, reasons, verified = load_verified_report(str(report),
            model_sha256=model_fingerprint(Weights()), configuration=self.wm_metadata,
            max_age_seconds=self.wm_config.get("report_max_age_seconds"))
        self.wm_probe_rejection_reasons = list(reasons)
        self.wm_last_diagnostic_report = deepcopy(verified)
        return accepted
