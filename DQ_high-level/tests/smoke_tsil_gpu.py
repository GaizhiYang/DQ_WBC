"""Run the complete SGS+EKF+TSIL training path in a small real Isaac Gym scene.

Run from DQ_high-level in the dqwbc environment. Defaults are the production
teacher settings, with 64 environments and a 144-step stop for this check:

    python tests/smoke_tsil_gpu.py --experiment_dir /tmp/dq_tsil_smoke

Teacher arguments may override the defaults. --stop_after must be 120 or 144;
the environment retains the real 100-step task budget and ten-step hold. This
script never fabricates outcomes, shortens episodes or modifies the sampler.
It checks training mechanics, not learned performance or expected success rate.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke_json", default=None)
    smoke_args, overrides = parser.parse_known_args()
    defaults = [
        "--cfg_file", "data/cfg/DQ_teacher.yaml", "--task", "B1Z1PickMulti",
        "--object_name", "sugar_box", "--num_envs", "64", "--timesteps", "120000",
        "--stop_after", "144", "--roboinfo", "--observe_gait_commands",
        "--small_value_set_zero", "--rand_control", "--no_velocity_obs",
        "--sgs_mode", "sgs", "--sgs_bank", "data/sgs/sugar_box_v10.pt",
        "--success_mode", "hold", "--future_pose_obs", "--future_pose_horizons", ".60", ".90",
        "--ekf_history", "5", "--ekf_q_scale", "1.0", "--ekf_r_scale", ".01",
        "--tsil", "--headless", "--seed", "43",
        "--experiment_dir", "/tmp/dq_tsil_smoke",
        "--wandb_name", "tsil_smoke_" + time.strftime("%Y%m%d_%H%M%S"),
    ]
    sys.argv = [str(ROOT / "train_multistate_DQ_teacher.py")] + defaults + overrides
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from isaacgym import gymapi  # noqa: F401; Isaac Gym must be imported before torch.
    import torch
    from utils.config import get_params
    from utils.tsil_checkpoint import read_temporal_checkpoint
    from train_multistate_DQ_teacher import get_trainer, finalize_teacher_logging

    args = get_params()
    if args.stop_after not in (120, 144) or args.timesteps < args.stop_after:
        raise ValueError("Use --stop_after 120 or 144, with planned timesteps >= stop_after")
    if args.resume or args.checkpoint or getattr(args, "eval", False) or not args.headless:
        raise ValueError("This smoke check requires a fresh, headless training run")
    if args.wandb:
        raise ValueError("This local smoke check does not publish a W&B run")
    run_dir = Path(args.experiment_dir).resolve() / args.wandb_name
    output = Path(smoke_args.smoke_json).resolve() if smoke_args.smoke_json else run_dir / "smoke_tsil_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "kind": "tsil_gpu_integration_not_learning_result", "checks": {}}
    trainer, started = None, time.monotonic()

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        report["checks"][name] = True

    def equal(name, actual, expected, exact=True):
        torch.testing.assert_close(actual, expected, atol=0 if exact else 1e-5,
                                   rtol=0 if exact else 1e-5, msg=name)
        report["checks"][name] = True

    def active_scaler_state(scaler):
        return {name: getattr(scaler, name).detach().clone()
                for name in ("running_mean", "running_variance", "current_count",
                             "prediction_scale", "prediction_bias", "standard_deviation_floor")
                if hasattr(scaler, name)}

    def same_scaler(name, scaler, expected):
        for field, value in expected.items():
            equal(name + "/" + field, getattr(scaler, field), value)

    try:
        trainer = get_trainer()
        env, agent = trainer.env._env, trainer.agents
        check("full_method_enabled", env.sgs_enabled and env.tsil_enabled
              and env.object_pose_predictor is not None and agent.replay is not None)
        check("training_task_unchanged", env.max_episode_length == 100 and env.hold_steps == 10
              and env.total_timesteps == args.timesteps and env.num_envs >= 2)
        check("production_ppo_update_shape", agent._rollouts == 24
              and agent._learning_epochs == 5 and agent._mini_batches == 6)
        check("ekf_horizons_unchanged", env.object_pose_predictor.horizons == (.6, .9))
        check("observation_space_1286", trainer.env.observation_space.shape == (1286,))
        check("time_reference_derived", abs(env.tsil_time_reference -
              env.max_episode_length * env.control_freq_low_init * env.dt) < 1e-8)
        check("all_configuration_targets_initially_reference",
              bool((env.tsil_state.targets == env.tsil_time_reference).all()))

        first, _ = trainer.env.reset()
        equal("initial_time_used_zero", first[:, -191], torch.zeros(env.num_envs, device=first.device))
        equal("initial_target_one", first[:, -190], torch.ones(env.num_envs, device=first.device))
        equal("initial_ekf_cold_start", first[:, -204:-191], env.future_object_obs)
        check("initial_ekf_invalid", not bool(env.object_pose_predictor.valid.any()))
        check("initial_sgs_has_no_outcomes", env.sgs_sampler.visits_total.sum().item() == 0)
        models = (agent.policy, agent.value)
        before = [parameter.detach().clone() for model in models for parameter in model.parameters()]
        critic_before = [parameter.detach().clone() for parameter in agent.value.parameters()]
        counts = dict(steps=0, updates=0, prepare=0, optimizer=0, valid_ekf=0, natural_done=0)
        online_critic_first_grad_norms, value_coordinate_errors = [], []
        natural_episodes, received_history = {}, {}
        selection_reports = []
        rollout_snapshot, update_snapshot = {}, {}
        originals = dict(step=env.step, record=agent.record_transition, update=agent._update,
                         prepare=agent.replay.prepare, backward=agent._backward_replay,
                         optimizer=agent.optimizer.step)

        def checked_step(actions):
            # Online behavior likelihoods must use one transform per rollout.
            if counts["steps"] % agent._rollouts == 0:
                rollout_snapshot["state"] = active_scaler_state(agent._state_preprocessor)
                rollout_snapshot["value"] = active_scaler_state(agent._value_preprocessor)
                rollout_snapshot["reward"] = agent.reward_normalizer.active_scale.clone()
            same_scaler("rollout_frozen_observation", agent._state_preprocessor, rollout_snapshot["state"])
            same_scaler("rollout_frozen_value", agent._value_preprocessor, rollout_snapshot["value"])
            equal("rollout_frozen_reward_scale", agent.reward_normalizer.active_scale, rollout_snapshot["reward"])
            elapsed, targets = env.tsil_elapsed.clone(), env.tsil_episode_D.clone()
            result = originals["step"](actions)
            obs, rewards, done, info = result[0]["obs"], result[1], result[2], result[3]["tsil"]
            equal("actual_clock_increment", env.tsil_elapsed, elapsed + env.control_freq_low * env.dt, False)
            equal("episode_target_unchanged_during_step", env.tsil_episode_D, targets)
            equal("info_clock_matches_observation", obs[:, -191], info["elapsed"] / env.tsil_time_reference)
            equal("target_matches_observation", obs[:, -190], info["episode_D"] / env.tsil_time_reference)
            equal("causal_ekf_channels", obs[:, -204:-191], env.future_object_obs)
            normalized = agent._state_preprocessor(obs)
            equal("time_channels_bypass_scaler", normalized[:, -191:-189], obs[:, -191:-189])
            predictor_count = env.object_pose_predictor.count.clone()
            env._compute_observations()
            equal("observation_read_does_not_update_ekf", env.object_pose_predictor.count, predictor_count)
            reconstruction = info["reward_fixed"] + float(info["command_gate"]) * info["reward_command_raw"]
            multiplier = 1 + (info["episode_D"] / info["elapsed"].clamp_min(1e-8)).clamp(max=1)
            reconstruction += env.reward_scales["pick_up"] * info["success"].float() * multiplier
            equal("raw_reward_decomposition", rewards, reconstruction, False)
            check("step_observation_finite", bool(torch.isfinite(obs).all()))
            check("reset_upload_consumed", not env._sgs_has_pending_reset)
            ended = done.bool().nonzero(as_tuple=False).flatten()
            for slot in ended.tolist():
                uid = (slot, int(info["episode_id"][slot]))
                natural_episodes[uid] = dict(config_id=int(info["config_id"][slot]),
                    length=int(env.progress_buf[slot]), success=bool(info["success"][slot]),
                    elapsed=float(info["elapsed"][slot]))
            counts["natural_done"] += len(ended)
            counts["valid_ekf"] += int(env.object_pose_predictor.valid.sum())
            counts["steps"] += 1
            return result

        def checked_record(states, actions, rewards, next_states, terminated, truncated,
                           infos, timestep, timesteps):
            ids = infos["tsil"]["episode_id"].cpu().tolist()
            received = states[:, -204:-189].detach().cpu().clone()
            for slot, serial in enumerate(ids):
                received_history.setdefault((slot, int(serial)), []).append(received[slot])
            return originals["record"](states, actions, rewards, next_states, terminated, truncated,
                                       infos, timestep, timesteps)

        def checked_prepare(*prepare_args, **prepare_kwargs):
            # This hook is after return RMS changes and before any optimizer
            # step. Cached old values and the unchanged critic must agree in
            # the new coordinates, including all previous adapter updates.
            with torch.no_grad():
                observations = agent.memory.get_tensor_by_name("states").flatten(0, 1)
                cached = agent.memory.get_tensor_by_name("values").flatten(0, 1)
                predicted = torch.cat([agent._predict_values(agent._state_preprocessor(chunk))
                                       for chunk in observations.split(256)])
                error = (predicted - cached).abs().max().item()
                value_coordinate_errors.append(error)
                torch.testing.assert_close(predicted, cached, atol=3e-5, rtol=3e-5)
                check("critic_coordinate_invariant_before_optimization", True)
            expected = list(agent.replay.iter_selected())
            selection = originals["prepare"](*prepare_args, **prepare_kwargs)
            counts["prepare"] += 1
            check("screen_all_selected_trajectories", selection.stats["replay_trajectories"] == len(expected))
            check("screen_all_selected_trajectory_steps", selection.stats["replay_dataset_steps"] ==
                  sum(len(trajectory) for trajectory in expected))
            update_snapshot["selection"] = selection
            update_snapshot["uids"] = selection.uids.clone()
            update_snapshot["usage"] = torch.zeros(len(selection), dtype=torch.long)
            selection_reports.append(dict(selection.stats))
            return selection

        def checked_backward(selection, indices):
            # Online backward has run, SI backward has not: this isolates the
            # PPO critic from gradients contributed by replay.
            if not update_snapshot.get("checked_first_online_gradient", False):
                gradients = [p.grad.detach().square().sum() for p in agent.value.parameters()
                             if p.grad is not None]
                norm = torch.stack(gradients).sum().sqrt().item() if gradients else 0.
                online_critic_first_grad_norms.append(norm)
                check("initial_online_critic_gradient_nonzero", norm > 0)
                update_snapshot["checked_first_online_gradient"] = True
            check("one_replay_selection_reused", selection is update_snapshot["selection"])
            if indices.numel():
                update_snapshot["usage"].index_add_(0, indices.cpu(), torch.ones_like(indices, device="cpu"))
            return originals["backward"](selection, indices)

        def checked_optimizer(*optimizer_args, **optimizer_kwargs):
            same_scaler("optimization_frozen_observation", agent._state_preprocessor, update_snapshot["state"])
            equal("optimization_frozen_reward_scale", agent.reward_normalizer.active_scale, update_snapshot["reward"])
            equal("optimization_frozen_targets", env.tsil_state.targets, update_snapshot["targets"])
            if "value" not in update_snapshot:
                update_snapshot["value"] = active_scaler_state(agent._value_preprocessor)
            same_scaler("optimization_frozen_value", agent._value_preprocessor, update_snapshot["value"])
            counts["optimizer"] += 1
            return originals["optimizer"](*optimizer_args, **optimizer_kwargs)

        def checked_update(timestep, timesteps):
            counts["updates"] += 1
            update_snapshot.clear()
            update_snapshot.update(state=active_scaler_state(agent._state_preprocessor),
                                   reward=agent.reward_normalizer.active_scale.clone(),
                                   targets=env.tsil_state.targets.clone())
            original_value_count = agent._value_preprocessor.current_count.item()
            originals["update"](timestep, timesteps)
            equal("replay_selection_indices_immutable", update_snapshot["selection"].uids,
                  update_snapshot["uids"])
            check("each_replay_transition_used_once_per_epoch",
                  bool((update_snapshot["usage"] == agent._learning_epochs).all()))
            check("value_statistics_only_online_returns", agent._value_preprocessor.current_count.item() ==
                  original_value_count + env.num_envs * agent._rollouts)
            check("targets_only_tighten", bool((env.tsil_state.targets <= update_snapshot["targets"]).all()))
            check("pending_targets_committed", bool(torch.isinf(env.tsil_state.pending_min).all()))

        env.step, agent.record_transition, agent._update = checked_step, checked_record, checked_update
        agent.replay.prepare, agent._backward_replay, agent.optimizer.step = checked_prepare, checked_backward, checked_optimizer
        try:
            trainer.train()
        finally:
            env.step, agent.record_transition, agent._update = originals["step"], originals["record"], originals["update"]
            agent.replay.prepare, agent._backward_replay, agent.optimizer.step = originals["prepare"], originals["backward"], originals["optimizer"]
        finalize_teacher_logging(trainer)
        torch.cuda.synchronize(agent.device)
        expected_updates = args.stop_after // agent._rollouts
        check("all_rollouts_updated", counts["updates"] == expected_updates and counts["steps"] == args.stop_after)
        check("one_full_screen_per_update", counts["prepare"] == expected_updates)
        check("joint_optimizer_frequency", counts["optimizer"] == expected_updates * 5 * 6)
        optimizer_steps = [int(value["step"].item() if torch.is_tensor(value["step"]) else value["step"])
                           for value in agent.optimizer.state.values() if "step" in value]
        check("optimizer_state_step_matches", bool(optimizer_steps) and max(optimizer_steps) == counts["optimizer"])
        check("ekf_became_valid", counts["valid_ekf"] > 0)
        check("natural_episodes_recorded", counts["natural_done"] > 0
              and agent.replay.completed_episodes == counts["natural_done"])
        check("sgs_outcomes_from_real_episodes", int(env.sgs_sampler.visits_total.sum()) == counts["natural_done"])
        archive = {trajectory.uid: trajectory for rows in
                   list(agent.replay.fastest.values()) + list(agent.replay.high_return.values()) for trajectory in rows}
        check("archive_nonempty", bool(archive))
        for uid, trajectory in archive.items():
            completed = natural_episodes[uid]
            check("archive_contains_only_complete_natural_episodes",
                  len(trajectory) == completed["length"] and trajectory.config_id == completed["config_id"]
                  and trajectory.success == completed["success"] and abs(trajectory.elapsed - completed["elapsed"]) < 1e-5)
            received = torch.stack(received_history[uid])
            equal("archive_preserves_received_ekf_and_time", trajectory.full_observations()[:, -204:-189], received)
        check("online_reward_statistics_count", agent.reward_normalizer.current_count.item() == env.num_envs * args.stop_after)
        check("observation_statistics_count", agent._state_preprocessor.current_count.item() == 1 + env.num_envs * args.stop_after)
        check("no_pending_observation_samples", agent._state_preprocessor.pending_count.item() == 0)
        after = [parameter for model in models for parameter in model.parameters()]
        check("parameters_finite", all(bool(torch.isfinite(parameter).all()) for parameter in after))
        changed = max(float((new.detach() - old).abs().max()) for old, new in zip(before, after))
        check("parameters_updated", changed > 0)
        critic_changed = max(float((new.detach() - old).abs().max())
                             for old, new in zip(critic_before, agent.value.parameters()))
        check("critic_parameters_updated", critic_changed > 0)

        checkpoint = run_dir / "checkpoints" / ("agent_" + str(args.stop_after) + ".pt")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(checkpoint))
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = read_temporal_checkpoint(str(checkpoint))
        check("checkpoint_complete_for_evaluation", metadata is not None and metadata["observation_dim"] == 1286)
        check("checkpoint_bank_matches", payload["tsil_state"]["bank_fingerprint"] == env.sgs_bank.fingerprint)
        for name in ("tsil_state", "state_preprocessor", "value_preprocessor", "reward_normalizer"):
            for field, value in agent.checkpoint_modules[name].state_dict().items():
                if torch.is_tensor(value):
                    equal("checkpoint/" + name + "/" + field, payload[name][field].cpu(), value.cpu())
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(agent.experiment_dir).Reload()
        logged_tags = events.Tags()["scalars"]
        for tag in ("TSIL / reward_scale", "TSIL / replay_transitions", "TSIL Archive / completed_episodes",
                    "TSIL Selection / replay_trajectories", "Loss / Policy loss", "Loss / Value loss"):
            check("logged/" + tag, tag in logged_tags and bool(events.Scalars(tag)))
        report.update(passed=True, num_envs=env.num_envs, executed_steps=counts["steps"],
                      counts=counts, archive=agent.replay.stats, selections=selection_reports,
                      archive_successes=sum(t.success for t in archive.values()),
                      archive_failures=sum(not t.success for t in archive.values()),
                      longest_archived_episode=max(len(t) for t in archive.values()),
                      bank_fingerprint=env.sgs_bank.fingerprint, checkpoint=str(checkpoint),
                      tensorboard_dir=str(agent.experiment_dir), largest_parameter_change=changed,
                      largest_critic_parameter_change=critic_changed,
                      online_critic_first_grad_norms=online_critic_first_grad_norms,
                      value_coordinate_errors=value_coordinate_errors,
                      reward_scale=agent.reward_normalizer.active_scale.item(),
                      peak_cuda_gib=torch.cuda.max_memory_allocated(agent.device) / 2**30)
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("TSIL_SMOKE_RESULT " + json.dumps(report, ensure_ascii=False))
        if trainer is not None:
            env = trainer.env._env
            if getattr(env, "viewer", None) is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
