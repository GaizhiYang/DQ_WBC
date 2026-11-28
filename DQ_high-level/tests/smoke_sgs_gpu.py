"""Manual Isaac Gym + PPO smoke check; unittest discovery must not import this.

Run from DQ_high-level with the same options as the teacher entrypoint, e.g.:

    python tests/smoke_sgs_gpu.py --task B1Z1PickMulti --object_name sugar_box \
      --sgs_mode sgs --sgs_bank data/reset_banks/sugar_box.pt --num_envs 96 \
      --rl_device cuda:0 --sim_device cuda:0 --timesteps 120000 --stop_after 96 \
      --roboinfo --observe_gait_commands --small_value_set_zero --rand_control \
      --no_velocity_obs --headless --experiment_dir /tmp/dq_sgs_smoke \
      --wandb_name smoke

An optional --smoke_json PATH selects the diagnostic JSON destination. This
script uses the original get_trainer()/trainer.train() update path. It does not
build a bank, fabricate episode outcomes, or manually update PPO.
"""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke_json", default=None)
    smoke_args, teacher_arguments = parser.parse_known_args()
    if not any(arg == "--stop_after" or arg.startswith("--stop_after=")
               for arg in teacher_arguments):
        teacher_arguments += ["--stop_after", "96"]
    sys.argv = [str(ROOT / "train_multistate_DQ_teacher.py")] + teacher_arguments
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    # Isaac Gym must precede torch, including any module that imports torch.
    from isaacgym import gymapi  # noqa: F401
    from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul
    import torch
    from utils.config import get_params
    from utils.sgs_level03 import snapshot_motion
    from train_multistate_DQ_teacher import get_trainer, finalize_teacher_logging

    args = get_params()
    if args.stop_after != 96 or args.timesteps < 96:
        raise ValueError("This smoke check requires --stop_after 96 and a plan of at least 96 steps")
    if not args.headless:
        raise ValueError("Pass --headless for this GPU smoke check")
    if args.resume or args.checkpoint:
        raise ValueError("This check starts fresh; do not pass --resume or --checkpoint")
    output = (Path(smoke_args.smoke_json) if smoke_args.smoke_json else
              Path(args.experiment_dir) / args.wandb_name / "smoke_sgs_result.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"passed": False, "checks": {}, "kind": "sgs_gpu_smoke_not_learning_result"}
    trainer = None
    start = time.monotonic()

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        report["checks"][name] = True

    def compare(name, actual, expected, exact=True):
        torch.testing.assert_close(
            actual, expected, atol=0 if exact else 1e-5, rtol=0 if exact else 1e-5,
            msg=name)
        report["checks"][name] = True

    def check_observation_stack(env, wrapper, prefix, returned=None, reset_ids=None):
        """Validate the wrapper's full stack and its current raw frame."""
        current = torch.clamp(env.obs_buf, -env.clip_obs, env.clip_obs).to(env.rl_device)
        frames = wrapper.history_frames
        cached = wrapper._obs_dict["obs"]
        check(prefix + "/shape", tuple(cached.shape) == (
            env.num_envs, current.shape[1] * frames))
        check(prefix + "/finite", bool(torch.isfinite(cached).all()))
        history = cached.reshape(env.num_envs, frames, current.shape[1])
        compare(prefix + "/latest_frame_matches_raw_buffer", history[:, -1], current)
        if frames > 1:
            compare(prefix + "/matches_history_buffer", history, wrapper._actor_history.buffer)
        if returned is not None:
            compare(prefix + "/returned_matches_cache", returned, cached)
        if reset_ids is not None:
            expected = current[reset_ids].unsqueeze(1).expand(-1, frames, -1)
            compare(prefix + "/reset_frames_repeat_current", history[reset_ids], expected)

    def check_bank_state(env, ids, prefix):
        state = env.sgs_bank.batch(env.active_config_id[ids])
        pairs = (
            ("robot_root", env._robot_root_states[ids], state["robot_root"]),
            ("object_root", env._cube_root_states[ids], state["object_root"]),
            ("table_root", env._table_root_states[ids], state["table_root"]),
            ("dof_pos", env._dof_pos[ids], state["dof_pos"]),
            ("dof_vel", env._dof_vel[ids], state["dof_vel"]),
            ("last_dof_vel", env._last_dof_vel[ids], state["dof_vel"]),
            ("local_ee_goal", env.curr_ee_goal_cart[ids], state["ee_goal_cart"]),
            ("local_ee_orientation", env.curr_ee_goal_orn_rpy[ids], state["ee_goal_orn_rpy"]),
            ("gripper_target", env.gripper_dof_pos[ids], state["gripper_target"]),
            ("table_height", env.table_heights[ids], state["table_height"]),
            ("rigid_body_cache",
             env._rigid_body_state.view(env.num_envs, -1, 13)[ids], state["rigid_body_state"]),
            ("jacobian_cache", env.ee_j_eef[ids], state["ee_jacobian"]),
        )
        if env.table_level == "Level02":
            pairs += (
                ("table_speed", env.vel_line_forward_y_level02[ids], state["table_vy"]),
                ("table_rotation", env.lin_circle_select[ids], state["rotation_enabled"]),
                ("table_yaw_rate", env.vel_angular_yaw[ids], state["table_yaw_rate"]),
            )
        else:
            check(prefix + "/supported_motion_level", env.table_level == "Level03")
            pairs += tuple((name, value[ids], state[name])
                           for name, value in snapshot_motion(env.level03_motion).items())
            check(prefix + "/rest_start_commands",
                  bool((env.table_commands[ids][:, [0, 1, 5]] == 0).all()))
        for name, actual, expected in pairs:
            compare(prefix + "/" + name, actual, expected)
        expected_global = quat_apply(
            env._robot_root_states[ids, 3:7], state["ee_goal_cart"]) + env.arm_base[ids]
        expected_orientation = quat_mul(
            env._robot_root_states[ids, 3:7],
            quat_from_euler_xyz(*state["ee_goal_orn_rpy"].unbind(-1)))
        compare(prefix + "/global_ee_goal", env.ee_goal_global_cart[ids], expected_global, False)
        compare(prefix + "/global_ee_orientation", env.ee_goal_global_orn_quat[ids],
                expected_orientation, False)
        expected_height = state["object_root"][:, 2] - state["table_height"] - env.init_height[ids]
        expected_distance = (env.ee_pos[ids] - state["object_root"][:, :3]).norm(dim=-1)
        compare(prefix + "/curr_height", env.curr_height[ids], expected_height, False)
        compare(prefix + "/highest_object", env.highest_object[ids], expected_height, False)
        compare(prefix + "/curr_dist", env.curr_dist[ids], expected_distance, False)
        compare(prefix + "/closest_dist", env.closest_dist[ids], expected_distance, False)
        check(prefix + "/episode_clean", bool(
            (env.pick_counter[ids] == 0).all() & ~env.episode_success[ids].any()))
        check(prefix + "/initial_foot_contacts_cleared",
              not bool(env.foot_contacts_from_sensor[ids].any()))

    try:
        trainer = get_trainer()
        env = trainer.env._env
        wrapped_env = trainer.env
        agent = trainer.agents
        report["collision_geometry"] = {
            "chassis_body_index": int(env.chassis_idx),
            "chassis_half_extents": env._chassis_half_extents.cpu().tolist(),
            "chassis_box_offset": env._chassis_box_offset[0].cpu().tolist(),
            "chassis_box_rotation": env._chassis_box_rotation[0].cpu().tolist(),
            "table_half_extents": env._table_half_extents.cpu().tolist(),
            "table_chassis_rest_offset": float(env._table_chassis_rest_offset),
        }
        print("SGS_COLLISION_GEOMETRY " + json.dumps(report["collision_geometry"]), flush=True)
        check("sgs_enabled", getattr(env, "sgs_enabled", False))
        check("multiple_environments", env.num_envs >= 3)
        check("sampler_cuda", env.sgs_sampler.history.is_cuda)
        recent = env.recent_episode_stats
        check("recent_statistics_cuda", all(getattr(recent, name).is_cuda for name in (
            "family_count", "family_successes")))
        check("recent_statistics_window", recent.family_window == 100)
        check("hold_success_mode", env.success_mode == "hold")
        check("four_rollouts_planned",
              trainer.timesteps - trainer.initial_timestep == 96 and agent._rollouts == 24)
        check("reward_schedule_preserved", env.total_timesteps == args.timesteps)
        all_ids = torch.arange(env.num_envs, device=env.device)
        first_observation, _ = trainer.env.reset()
        first_observation = first_observation.clone()
        check_bank_state(env, all_ids, "initial_reset")
        check("initial_observation_finite", bool(torch.isfinite(first_observation).all()))
        check_observation_stack(env, wrapped_env, "initial_observation", first_observation, all_ids)
        report["observation_layout"] = dict(
            raw_frame_width=int(env.obs_buf.shape[1]),
            stacked_width=int(first_observation.shape[1]),
            actor_history_frames=wrapped_env.actor_history_frames,
            critic_history_frames=wrapped_env.critic_history_frames,
            wrapper_history_frames=wrapped_env.history_frames)
        snapshot_obs = env.obs_buf.clone()
        env._compute_observations(all_ids)
        compare("initial_observation_recomputes_from_current_cache", env.obs_buf, snapshot_obs, False)
        check("initial_reset_no_outcomes", env.sgs_sampler.visits_total.sum().item() == 0)

        # Cancel one actual active episode, without executing a physics step.
        # Other live environments must retain their state and observation.
        snapshots = {
            "root": env._root_states.view(env.num_envs, -1, 13)[1:].clone(),
            "body": env._rigid_body_state.view(env.num_envs, -1, 13)[1:].clone(),
            "dof": env._dof_state.view(env.num_envs, -1, 2)[1:].clone(),
            "obs": env.obs_buf[1:].clone(),
            "progress": env.progress_buf[1:].clone(),
            "ids": env.active_config_id[1:].clone(),
            "visits": env.sgs_sampler.visits_total.clone(),
            "episodes": env.episode_counter.clone(),
            "recent_family_count": recent.family_count.clone(),
            "recent_family_successes": recent.family_successes.clone(),
            "wrapped_obs": wrapped_env._obs_dict["obs"].clone(),
        }
        if env.table_level == "Level03":
            snapshots["motion"] = snapshot_motion(env.level03_motion)
        aborted = env.sgs_aborted.item()
        single_id = all_ids[:1]
        env.reset_idx(single_id)
        compare("cancel_other_roots_unchanged",
                env._root_states.view(env.num_envs, -1, 13)[1:], snapshots["root"])
        compare("cancel_other_bodies_unchanged",
                env._rigid_body_state.view(env.num_envs, -1, 13)[1:], snapshots["body"])
        compare("cancel_other_dofs_unchanged",
                env._dof_state.view(env.num_envs, -1, 2)[1:], snapshots["dof"])
        compare("cancel_other_observations_unchanged", env.obs_buf[1:], snapshots["obs"])
        compare("cancel_other_progress_unchanged", env.progress_buf[1:], snapshots["progress"])
        compare("cancel_other_config_ids_unchanged", env.active_config_id[1:], snapshots["ids"])
        if env.table_level == "Level03":
            for name, value in snapshot_motion(env.level03_motion).items():
                compare("cancel_other_motion_unchanged/" + name,
                        value[1:], snapshots["motion"][name][1:])
        compare("cancel_no_sampler_outcomes", env.sgs_sampler.visits_total, snapshots["visits"])
        compare("cancel_no_completed_episodes", env.episode_counter, snapshots["episodes"])
        for name in ("family_count", "family_successes"):
            compare("cancel_no_recent_" + name, getattr(recent, name), snapshots["recent_" + name])
        check("cancel_counted_separately", env.sgs_aborted.item() == aborted + 1)
        check_bank_state(env, single_id, "cancel_reset")
        # Raw reset_idx does not own observation history. The public wrapper
        # reset call must seed this selected slot and preserve all other stacks.
        cancelled_observation, _ = wrapped_env.reset()
        check_observation_stack(env, wrapped_env, "cancel_observation",
                                cancelled_observation, single_id)
        compare("cancel_other_stacked_observations_unchanged",
                cancelled_observation[1:], snapshots["wrapped_obs"][1:])
        compare("initial_returned_observation_not_mutated", first_observation,
                snapshots["wrapped_obs"])

        models = (agent.policy, agent.value)
        before = [parameter.detach().clone()
                  for model in models for parameter in model.parameters()]
        # Observe the real step boundary without changing trainer order. The
        # following reset may legitimately queue another upload, including on
        # the final training iteration.
        original_step = env.step
        original_table_control = env.floating_table_control
        motion_steps = [0]
        motion_maxima = torch.zeros(5, device=env.device)
        if env.table_level == "Level03":
            limits = env.level03_motion_cfg

            def checked_table_control():
                previous = env.table_commands[:, [0, 1, 5]].clone()
                result = original_table_control()
                actual = env.table_commands[:, [0, 1, 5]]
                delta = (actual - previous).abs().amax(dim=0)
                peak = actual.abs().amax(dim=0)
                dt = float(env.sim_params.dt)
                check("level03_motion_finite", bool(torch.isfinite(actual).all()))
                if env.level03_motion.translation_axes == "y":
                    check("level03_local_vx_zero", bool((actual[:, 0] == 0).all()))
                check("level03_vx_limit", bool(peak[0] <= limits["maxSpeed"] + 1e-6))
                check("level03_vy_limit", bool(peak[1] <= limits["maxSpeed"] + 1e-6))
                check("level03_yaw_limit", bool(peak[2] <= limits["maxYawRate"] + 1e-6))
                check("level03_linear_acceleration_limit",
                      bool(delta[:2].max() <= limits["linearAcceleration"] * dt + 1e-7))
                check("level03_yaw_acceleration_limit",
                      bool(delta[2] <= limits["yawAcceleration"] * dt + 1e-7))
                motion_maxima.copy_(torch.maximum(
                    motion_maxima, torch.stack((peak[0], peak[1], peak[2], delta[:2].max() / dt, delta[2] / dt))))
                motion_steps[0] += 1
                return result

            env.floating_table_control = checked_table_control
        checked_steps = [0]
        full_task_id = env.sgs_family_names.index("full_task")
        full_task_outcomes = deque(maxlen=1000)

        def checked_step(actions):
            result = original_step(actions)
            check("pending_resets_consumed_by_physics_step",
                  not env._sgs_has_pending_reset and not bool(env._sgs_pending_reset.any()))
            done_ids = env.reset_buf.nonzero(as_tuple=False).flatten()
            families = env.sgs_bank.tensors["family_id"][env.active_config_id[done_ids]]
            full_ids = done_ids[families == full_task_id]
            full_task_outcomes.extend(env.pick_success_recorded[full_ids].cpu().tolist())
            checked_steps[0] += 1
            return result

        env.step = checked_step
        try:
            trainer.train()
        finally:
            env.step = original_step
            env.floating_table_control = original_table_control
        if env.table_level == "Level03":
            check("level03_physics_commands_checked", motion_steps[0] > 0)
            # After real rollout motion, a selected reset must leave the other
            # environments' filter histories and driver clocks untouched.
            live_motion = snapshot_motion(env.level03_motion)
            live_observation = wrapped_env._obs_dict["obs"].clone()
            env.reset_idx(single_id)
            for name, value in snapshot_motion(env.level03_motion).items():
                compare("live_cancel_other_motion_unchanged/" + name,
                        value[1:], live_motion[name][1:])
            check_bank_state(env, single_id, "live_cancel_reset")
            live_cancelled_observation, _ = wrapped_env.reset()
            check_observation_stack(env, wrapped_env, "live_cancel_observation",
                                    live_cancelled_observation, single_id)
            compare("live_cancel_other_stacked_observations_unchanged",
                    live_cancelled_observation[1:], live_observation[1:])
            report["level03_motion"] = dict(
                checked_physics_steps=motion_steps[0],
                max_abs_local_vx=float(motion_maxima[0]),
                max_abs_local_vy=float(motion_maxima[1]),
                max_abs_yaw_rate=float(motion_maxima[2]),
                max_linear_acceleration=float(motion_maxima[3]),
                max_yaw_acceleration=float(motion_maxima[4]),
                configured_limits=limits,
            )
        finalize_teacher_logging(trainer)
        torch.cuda.synchronize(env.sgs_sampler.device)
        after = [parameter for model in models for parameter in model.parameters()]
        check("parameters_finite", all(bool(torch.isfinite(parameter).all()) for parameter in after))
        changed = max((new.detach() - old).abs().max().item()
                      for old, new in zip(before, after))
        check("parameters_updated", changed > 0)
        optimizer_steps = [
            float(value["step"].item() if isinstance(value["step"], torch.Tensor) else value["step"])
            for value in agent.optimizer.state.values() if "step" in value
        ]
        check("optimizer_performed_updates", bool(optimizer_steps) and max(optimizer_steps) > 0)
        check("four_ppo_updates", agent._rollout == 96 and agent._rollout // agent._rollouts == 4)
        check("all_physics_step_boundaries_checked", checked_steps[0] == 96)
        check("tail_pending_flag_consistent",
              env._sgs_has_pending_reset == bool(env._sgs_pending_reset.any()))
        check("completed_episode_outcomes_observed", env.sgs_sampler.visits_total.sum().item() > 0)
        check("all_reset_families_sampled", bool((env.sgs_family_samples > 0).all()))
        check("family_statistics_match_loaded_bank", len(env.sgs_family_samples) == len(env.sgs_bank.family_names))
        check("final_observation_finite", bool(torch.isfinite(env.obs_buf).all()))
        check_observation_stack(env, wrapped_env, "final_observation")
        # Read the real TensorBoard event file used by wandb sync, including the
        # final reset outcomes written after the trainer's last ordinary write.
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        events = EventAccumulator(agent.experiment_dir).Reload()
        family_rates = {}
        for index, family in enumerate(env.sgs_family_names):
            episodes = int(recent.family_count[index].item())
            successes = int(recent.family_successes[index].item())
            check("family_window/" + family, 0 <= successes <= episodes <= 100
                  and episodes == min(100, int(env.sgs_family_episodes[index].item())))
            for tag, expected in (("SGS / episodes_" + family, episodes),
                                  ("SGS / successes_" + family, successes)):
                rows = events.Scalars(tag)
                check("family_log/" + tag, rows[-1].step == env.global_step_counter
                      and abs(rows[-1].value - expected) < 1e-6)
            rate = successes / episodes if episodes else None
            family_rates[family] = rate
            tag = "SuccessRate / " + family
            if episodes:
                rows = events.Scalars(tag)
                check("family_log/" + tag, rows[-1].step == env.global_step_counter
                      and abs(rows[-1].value - rate) < 1e-6)
            else:
                check("family_log/unknown_" + family, tag not in events.Tags()["scalars"])
        # The baseline comparison curve uses the last 1000 full_task episodes,
        # independently of the shorter family diagnostic windows above.
        main_metrics = {
            "SuccessWindow / completed": len(full_task_outcomes),
            "SuccessWindow / successes": sum(full_task_outcomes),
        }
        main_tag = "SuccessRate / " + env.obj_list[0]
        if full_task_outcomes:
            main_metrics[main_tag] = sum(full_task_outcomes) / len(full_task_outcomes)
        else:
            check("full_task_main_rate_undefined", main_tag not in events.Tags()["scalars"])
        check("full_task_main_window", env.success_window.window_size == 1000
              and len(full_task_outcomes) == min(1000, int(env.sgs_family_episodes[full_task_id])))
        check("full_task_main_metrics", env.success_window.metrics(env.obj_list) == main_metrics)
        for tag, expected in main_metrics.items():
            rows = events.Scalars(tag)
            check("full_task_main_log/" + tag, rows[-1].step == env.global_step_counter
                  and abs(rows[-1].value - expected) < 1e-6)
            check("full_task_main_no_stale_snapshot/" + tag,
                  tag not in env.extras["scalar_snapshots"])
        for tag in ("SuccessRate / overall", "SuccessRate / family_mean", "SGS / episodes_overall",
                    "SGS / successes_overall", "SGS / success_overall"):
            check("family_only_log/no_" + tag, tag not in events.Tags()["scalars"]
                  and tag not in env.extras["scalar_snapshots"])
        diagnostics = output.with_suffix(".sampler.pt")
        env.save_sgs_diagnostics(str(diagnostics))
        report.update({
            "passed": True,
            "num_envs": env.num_envs,
            "motion_level": env.table_level,
            "bank_fingerprint": env.sgs_bank.fingerprint,
            "bank_size": env.sgs_bank.num_configs,
            "planned_timesteps": args.timesteps,
            "executed_timesteps": agent._rollout,
            "ppo_updates": agent._rollout // agent._rollouts,
            "optimizer_max_step": max(optimizer_steps),
            "largest_parameter_change": changed,
            "completed_episodes": int(env.sgs_sampler.visits_total.sum().item()),
            "family_samples": env.sgs_family_samples.cpu().tolist(),
            "family_names": list(env.sgs_bank.family_names),
            "family_completed": env.sgs_family_episodes.cpu().tolist(),
            "family_successes": env.sgs_family_successes.cpu().tolist(),
            "family_success_rates": family_rates,
            "family_recent_completed": recent.family_count.cpu().tolist(),
            "family_recent_successes": recent.family_successes.cpu().tolist(),
            "family_window_size": recent.family_window,
            "main_success_scope": "full_task",
            "main_success_window_size": 1000,
            "main_success_metrics": main_metrics,
            "recent_statistics_device": str(recent.family_count.device),
            "tensorboard_dir": str(Path(agent.experiment_dir).resolve()),
            "cancelled_episodes": int(env.sgs_aborted.item()),
            "sampler_device": str(env.sgs_sampler.device),
            "torch_peak_allocated_gib": torch.cuda.max_memory_allocated(agent.device) / 2**30,
            "torch_peak_reserved_gib": torch.cuda.max_memory_reserved(agent.device) / 2**30,
            "tail_pending_resets": int(env._sgs_pending_reset.sum().item()),
            "sampler_diagnostics": str(diagnostics.resolve()),
        })
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - start
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("SGS_SMOKE_RESULT " + json.dumps(report, ensure_ascii=False))
        if trainer is not None:
            if args.wandb:
                import wandb
                wandb.finish()
            env = trainer.env._env
            if getattr(env, "viewer", None) is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
