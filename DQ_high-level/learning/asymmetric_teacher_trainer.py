"""Strict asymmetric PPO with isolated deterministic policy evaluation.

The checkpoint state counts *training* vector steps. Evaluation reuses the
simulator but discards interrupted episodes and starts training again from a
full reset, including perception state and history. It never records PPO
transitions or reproduces the interrupted physics trajectory.
"""

from copy import deepcopy
from collections.abc import Mapping
from pathlib import Path
import json
import math
import random

import numpy as np
import torch
import tqdm

from skrl.trainers.torch import SequentialTrainer
from utils import asymmetric_distributed as distributed


_ENV_STATISTICS = (
    "global_step_counter", "local_step_counter", "episode_counter",
    "success_counter", "success_onetime_counter", "predlift_success_counter",
    "episode_step_sum", "episode_sums", "episode_metric_sums", "extras",
)
_PERCEPTION_METRICS = (
    "prediction_error_m", "initialized_fraction", "visible_fraction", "reward_mean",
    "camera_age_s", "prediction_age_s", "rejected_stale_measurements",
)


def _perception_metrics(infos):
    """Extract finite wrapper diagnostics; absent estimates are not zeros."""
    values = infos.get("perception", {}) if isinstance(infos, Mapping) else {}
    if not isinstance(values, Mapping):
        return {}
    result = {}
    for key in _PERCEPTION_METRICS:
        value = values.get(key)
        if value is not None:
            value = float(value)
            if math.isfinite(value):
                result[key] = value
    return result


def _raw_environment(env):
    while "_env" in vars(env):
        env = vars(env)["_env"]
    return env


def _copy_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return deepcopy(value)


def _restore_attribute(owner, name, value):
    current = getattr(owner, name, None)
    if isinstance(current, torch.Tensor) and isinstance(value, torch.Tensor):
        current.copy_(value)
    else:
        setattr(owner, name, _copy_value(value))


class AsymmetricTeacherTrainer(SequentialTrainer):
    """Single-agent trainer using the existing Gaussian PPO implementation.

    ``asymmetric_teacher_state`` must expose ``completed_steps``,
    ``mark_completed(step)``, ``best_eval_score`` and
    ``last_eval_step``. ``cfg.timesteps`` is the total training-step target,
    including steps completed before a resume, and must end on a rollout
    boundary. Evaluation and checkpoint intervals must also align to rollouts.

    An optional ``evaluation_callback(trainer, metrics, completed_steps)`` runs
    after evaluation and restoration. ``best_agent.pt`` is selected using
    deterministic policy GSR. The Actor has the same sensor inputs throughout
    training and evaluation; the trainer does not change its architecture.
    """

    def __init__(self, env, agents, agents_scope=None, cfg=None, *,
                 evaluation_callback=None):
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=cfg)
        if self.num_simultaneous_agents != 1 or self.env.num_agents != 1:
            raise ValueError("Asymmetric teacher training requires one agent")
        self.training_state = self.agents.checkpoint_modules.get("asymmetric_teacher_state")
        if self.training_state is None:
            raise ValueError("Missing asymmetric_teacher_state checkpoint module")
        self.rollouts = int(self.agents._rollouts)
        self.eval_interval = int(self.cfg.get("eval_interval", 2400))
        self.eval_steps = int(self.cfg.get("eval_steps", 1000))
        self.checkpoint_interval = int(self.cfg.get("checkpoint_interval", 2400))
        self.eval_seed = int(self.cfg.get("eval_seed", 1729))
        self.evaluation_callback = evaluation_callback
        self.evaluation_history = []
        self.last_evaluation = None
        self._is_evaluating = False
        self._resume_states = None
        if self.rollouts <= 0:
            raise ValueError("rollouts must be positive")
        for name, value in (("eval_interval", self.eval_interval),
                            ("checkpoint_interval", self.checkpoint_interval)):
            if value < 0 or value % self.rollouts:
                raise ValueError(name + " must be zero or a multiple of rollouts")
        if self.eval_steps <= 0:
            raise ValueError("eval_steps must be positive")
        # Select the best policy from evaluation success, rather than stock
        # skrl's potentially shaped training reward.
        self.agents.checkpoint_interval = 0

    def _validate_training_budget(self):
        completed = int(self.training_state.completed_steps)
        if completed < 0 or completed % self.rollouts:
            raise ValueError("Resume completed_steps must be a rollout boundary")
        if self.timesteps < completed or self.timesteps % self.rollouts:
            raise ValueError("timesteps must be >= completed_steps and a rollout boundary")
        if int(getattr(self.agents, "_rollout", 0)) % self.rollouts:
            raise ValueError("Cannot resume with a partially collected PPO rollout")
        if int(getattr(self.agents, "_learning_starts", 0)) > completed:
            raise ValueError("PPO updates must start from the first resumed rollout")
        return completed

    @torch.no_grad()
    def _full_reset(self):
        raw = _raw_environment(self.env)
        if hasattr(raw, "reset_idx"):
            # DQ's ordinary reset() resets only reset_buf entries. reset_idx()
            # with no ids resets every robot and refreshes camera history.
            raw.reset_idx()
        reset_runtime = getattr(self.env, "reset_runtime", None)
        if callable(reset_runtime):
            # M2 filters, perception samples and history belong to the newly
            # reset episodes. Clear them before packing the first observation.
            reset_runtime()
        return self.env.reset()[0]

    @torch.no_grad()
    def _policy_mean(self, states):
        processed = self.agents._state_preprocessor(states, train=False)
        mean, _, _ = self.agents.policy.compute({"states": processed}, role="policy")
        return mean.detach().clone()

    def _save(self, name):
        if not distributed.is_main():
            return
        directory = Path(self.agents.experiment_dir) / "checkpoints"
        directory.mkdir(parents=True, exist_ok=True)
        self.agents.save(str(directory / name))

    def _finish_evaluation(self, metrics, completed):
        self.training_state.last_eval_step = int(completed)
        score = metrics["gsr"]
        improved = score is not None and math.isfinite(score) and score > self.training_state.best_eval_score
        if improved:
            self.training_state.best_eval_score = score
            self._save("best_agent.pt")
        metrics = dict(metrics, completed_steps=int(completed), best=bool(improved))
        self.last_evaluation = metrics
        self.evaluation_history.append(metrics)
        if not distributed.is_main():
            return
        path = Path(self.agents.experiment_dir) / "evaluations.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        for key in ("gsr", "ossr", "mean_reward", "mean_episode_return", "tsc"):
            if metrics[key] is not None:
                self.agents.track_data("Evaluation / " + key, metrics[key])
        for key, value in metrics.get("perception", {}).items():
            self.agents.track_data("Evaluation perception / " + key, value)
        if self.evaluation_callback is not None:
            self.evaluation_callback(self, metrics, completed)

    def single_agent_train(self):
        completed = self._validate_training_budget()
        self.initial_timestep = completed
        states = self._full_reset()
        for timestep in tqdm.tqdm(range(completed, self.timesteps), disable=self.disable_progressbar):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)
            with torch.no_grad():
                actions = self.agents.act(states, timestep=timestep, timesteps=self.timesteps)[0]
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                if not self.headless:
                    self.env.render()
                self.agents.record_transition(
                    states=states, actions=actions, rewards=rewards,
                    next_states=next_states, terminated=terminated, truncated=truncated,
                    infos=infos, timestep=timestep, timesteps=self.timesteps)
                for key, value in _perception_metrics(infos).items():
                    self.agents.track_data("Perception / " + key, value)
            # PPO synchronously completes all epochs before evaluation or save.
            self.agents.post_interaction(timestep=timestep, timesteps=self.timesteps)
            with torch.no_grad():
                states = self.env.reset()[0] if (terminated.any() or truncated.any()) else next_states
            completed = timestep + 1
            if completed % self.rollouts:
                continue
            # Scratch statistics advance only after the rollout's final PPO
            # epoch. Both its sampled and recomputed likelihoods therefore use
            # the same normalization. Evaluation never enters this path.
            update_normalization = getattr(self.agents._state_preprocessor, "update_rollout", None)
            if callable(update_normalization):
                update_normalization(self.agents.memory.get_tensor_by_name("states"))
            self.training_state.mark_completed(completed)
            due = self.eval_interval and completed % self.eval_interval == 0
            final = completed == self.timesteps
            if (due or (final and self.eval_interval)) and completed != self.training_state.last_eval_step:
                metrics = self.evaluate_policy(self.eval_steps)
                states = self._resume_states
                self._finish_evaluation(metrics, completed)
            if final or (self.checkpoint_interval and completed % self.checkpoint_interval == 0):
                self._save("agent_{}.pt".format(completed))

    @torch.no_grad()
    def evaluate_policy(self, steps):
        """Evaluate means with legacy DQ eval success, then reset training fully.

        The same simulator's reset distribution is retained. The actual initial
        robot position is reported: legacy environment code can override the
        train/play entrypoint's requested position.
        Statistics use *completed* evaluation episodes; unfinished final
        episodes are excluded. No completions => no GSR and no best update.
        Perception diagnostics average the available per-step wrapper values
        across the evaluation window, independently of episode completion.
        """
        steps = int(steps)
        if steps <= 0 or self._is_evaluating:
            raise ValueError("Evaluation requires positive steps and cannot be nested")
        if int(getattr(self.agents, "_rollout", 0)) % self.rollouts:
            raise ValueError("Evaluation is only allowed after a complete PPO rollout")
        self._is_evaluating = True
        raw = _raw_environment(self.env)
        statistics = {name: _copy_value(getattr(raw, name)) for name in _ENV_STATISTICS if hasattr(raw, name)}
        old_eval = getattr(raw, "eval", None)
        old_running = getattr(self.agents, "training", True)
        model_modes = [(model, model.training) for model in self.agents.models.values() if model is not None]
        preprocessors = [(module, deepcopy(module.state_dict())) for name, module in self.agents.checkpoint_modules.items()
                         if "preprocessor" in name and hasattr(module, "state_dict")]
        rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(),
               torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
        get_runtime_rng = getattr(self.env, "get_runtime_rng_state", None)
        set_runtime_rng = getattr(self.env, "set_runtime_rng_state", None)
        runtime_rng = (_copy_value(get_runtime_rng())
                       if callable(get_runtime_rng) and callable(set_runtime_rng) else None)
        try:
            raw.eval = True
            self.agents.training = False
            self.agents.set_mode("eval")
            random.seed(self.eval_seed)
            np.random.seed(self.eval_seed)
            torch.manual_seed(self.eval_seed)
            seed_runtime = getattr(self.env, "seed_runtime", None)
            if callable(seed_runtime):
                seed_runtime(self.eval_seed)
            states = self._full_reset()
            for name in _ENV_STATISTICS:
                if name in ("global_step_counter", "extras") or not hasattr(raw, name):
                    continue
                value = getattr(raw, name)
                if isinstance(value, torch.Tensor):
                    value.zero_()
                elif isinstance(value, dict):
                    for item in value.values():
                        if isinstance(item, torch.Tensor):
                            item.zero_()
            if hasattr(raw, "one_epoch_step"):
                raw.one_epoch_step.zero_()
            raw.extras = {}
            device = states.device
            returns = torch.zeros(self.env.num_envs, device=device)
            lengths = torch.zeros(self.env.num_envs, device=device)
            successes = torch.zeros(self.env.num_envs, dtype=torch.bool, device=device)
            one_shots = torch.zeros_like(successes)
            completed_returns, success_lengths = [], []
            completed_count = success_count = one_shot_count = 0
            total_reward = 0.
            perception_sums, perception_counts = {}, {}
            has_success = hasattr(raw, "success_counter")
            has_one_shot = hasattr(raw, "success_onetime_counter")
            previous_success = raw.success_counter.clone() if has_success else None
            previous_one_shot = raw.success_onetime_counter.clone() if has_one_shot else None
            for _ in range(steps):
                if "global_step_counter" in statistics:
                    raw.global_step_counter = statistics["global_step_counter"]
                mean = self._policy_mean(states)
                next_states, reward, terminated, truncated, infos = self.env.step(mean)
                for key, value in _perception_metrics(infos).items():
                    perception_sums[key] = perception_sums.get(key, 0.) + value
                    perception_counts[key] = perception_counts.get(key, 0) + 1
                if not self.headless:
                    self.env.render()
                returns += reward.reshape(self.env.num_envs)
                lengths += 1
                total_reward += float(reward.sum())
                if has_success:
                    successes |= raw.success_counter > previous_success
                    previous_success.copy_(raw.success_counter)
                if has_one_shot:
                    one_shots |= raw.success_onetime_counter > previous_one_shot
                    previous_one_shot.copy_(raw.success_onetime_counter)
                done = (terminated | truncated).reshape(self.env.num_envs).bool()
                if done.any():
                    completed_count += int(done.sum())
                    success_count += int((done & successes).sum())
                    one_shot_count += int((done & one_shots).sum())
                    completed_returns.extend(returns[done].tolist())
                    success_lengths.extend(lengths[done & successes].tolist())
                    returns[done] = 0
                    lengths[done] = 0
                    successes[done] = False
                    one_shots[done] = False
                    states = self.env.reset()[0]
                else:
                    states = next_states
            metrics = {
                "vector_steps": steps,
                "transitions": steps * self.env.num_envs,
                "completed_episodes": completed_count,
                "successes": success_count if has_success else None,
                "gsr": success_count / completed_count if has_success and completed_count else None,
                "ossr": one_shot_count / completed_count if has_one_shot and completed_count else None,
                "mean_reward": total_reward / (steps * self.env.num_envs),
                "mean_episode_return": float(np.mean(completed_returns)) if completed_returns else None,
                "tsc": float(np.mean(success_lengths)) if success_lengths else None,
                "protocol": "legacy_eval_success_current_initial_pose_completed_episodes",
                "perception": {key: total / perception_counts[key]
                               for key, total in perception_sums.items()},
            }
            initial_roots = getattr(raw, "_initial_robot_root_states", None)
            metrics["initial_base_x_range"] = (
                [float(initial_roots[:, 0].min()), float(initial_roots[:, 0].max())]
                if isinstance(initial_roots, torch.Tensor) else None)
            return distributed.aggregate_evaluation(metrics, perception_counts)
        finally:
            # Restore RNG before the new training reset: evaluation random draws
            # do not advance the training stream; only the explicit restart does.
            random.setstate(rng[0])
            np.random.set_state(rng[1])
            torch.random.set_rng_state(rng[2])
            if rng[3] is not None:
                torch.cuda.set_rng_state_all(rng[3])
            if runtime_rng is not None:
                set_runtime_rng(runtime_rng)
            if old_eval is None:
                delattr(raw, "eval")
            else:
                raw.eval = old_eval
            for name in ("global_step_counter", "local_step_counter"):
                if name in statistics:
                    _restore_attribute(raw, name, statistics[name])
            self.agents.training = old_running
            for model, training in model_modes:
                model.train(training)
            for module, saved in preprocessors:
                module.load_state_dict(saved)
            try:
                self._resume_states = self._full_reset()
            finally:
                for name, value in statistics.items():
                    if name in ("episode_sums", "episode_metric_sums"):
                        # These are per-episode reward/metric accumulators.
                        # Restoring a discarded partial episode would splice
                        # its values onto the freshly reset training episode.
                        for item in getattr(raw, name, {}).values():
                            if isinstance(item, torch.Tensor):
                                item.zero_()
                    else:
                        _restore_attribute(raw, name, value)
                # The old incomplete training episodes were intentionally
                # discarded. Never attach their length/return to new episodes.
                for owner, names in ((raw, ("one_epoch_step",)),
                                     (self.agents, ("_cumulative_rewards", "_cumulative_timesteps"))):
                    for name in names:
                        value = getattr(owner, name, None)
                        if isinstance(value, torch.Tensor):
                            value.zero_()
                self._is_evaluating = False

    def single_agent_eval(self):
        """Standalone bounded policy evaluation without training callbacks."""
        self.last_evaluation = self.evaluate_policy(self.timesteps)
        if distributed.is_main():
            print(json.dumps(self.last_evaluation, sort_keys=True, allow_nan=False))
