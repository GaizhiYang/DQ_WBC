#!/usr/bin/env python3
"""Record three environments during actual early PPO training on Level03.

Uses the same get_trainer().train() path and flags as train_teacher_ekf.sh.
Only the number of parallel environments, logging destination, run duration,
and offscreen recording cameras differ. Ordinary episode resets, policy
actions, low-level control and PPO updates remain active.
Level03 uses signed local vx and vy; legacy local-Y configurations remain supported.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=40.)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--record-envs", type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < .04:
        parser.error("seconds must be finite and >= 0.04")
    if args.num_envs < 3 or len(set(args.record_envs)) != 3 or any(
            index < 0 or index >= args.num_envs for index in args.record_envs):
        parser.error("record-envs must select three distinct valid environments")
    output = args.output.expanduser().resolve()
    try:
        output.relative_to((ROOT / "outputs").resolve())
    except ValueError:
        parser.error("output must be inside DQ_high-level/outputs")
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)

    from isaacgym import gymapi
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import yaml
    from train_multistate_DQ_teacher import get_trainer, finalize_teacher_logging
    from utils.training_motion_video import compose_training_frame

    # rand_control advances >= 0.14 s per high-level step. Finish a whole PPO
    # rollout after the requested film duration; keep the 120000-step schedule.
    stop_after = int(math.ceil((args.seconds / .14 + 2) / 24)) * 24
    argv = ["--cfg_file", "data/cfg/DQ_teacher.yaml", "--motion_level", "Level03",
        "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
        "--rl_device", args.device, "--sim_device", args.device,
        "--graphics_device_id", args.device.split(":")[-1],
        "--num_envs", str(args.num_envs), "--timesteps", "120000", "--stop_after", str(stop_after),
        "--roboinfo", "--observe_gait_commands", "--small_value_set_zero",
        "--rand_control", "--no_velocity_obs", "--success_mode", "hold",
        "--future_pose_horizons", "0.60", "0.90", "--ekf_history", "5",
        "--ekf_q_scale", "1.0", "--ekf_r_scale", "0.01",
        "--experiment_dir", str(output / "training_run"), "--wandb_name", "preview",
        "--seed", str(args.seed), "--headless", "--sgs_mode", "disabled", "--no_tsil",
        "--future_pose_obs"]
    trainer = get_trainer(is_eval=False, argv=argv, enable_camera_sensors=True)
    env, agent = trainer.env, trainer.agents
    raw = env._env
    assert raw.table_level == "Level03" and not raw.eval
    assert raw.max_episode_length == 100
    assert list(raw.robot_start_distance_range) == [2., 2.]
    assert not raw.sgs_enabled and not raw.tsil_enabled
    assert raw.object_pose_predictor.horizons == (.6, .9)
    trainer.disable_progressbar = True
    fps, frame_limit = 25, round(args.seconds * 25)
    cameras, writers, samples = {}, {}, {index: [] for index in args.record_envs}
    episode = {index: 1 for index in args.record_envs}
    episode_start = {index: 0. for index in args.record_envs}
    reset_events = {index: [] for index in args.record_envs}
    start_time, frame_count = None, 0
    stats = dict(high_level_steps=0, ppo_updates=0, max_action_abs=0.,
                 nonzero_action_steps=0, physics_steps=0,
                 max_speed=0., max_vx=0., max_vy=0., max_yaw_rate=0.,
                 max_speed_step=0., max_yaw_step=0.)
    original_reset_idx, original_reset, original_step = raw.reset_idx, env.reset, env.step
    original_update, original_control = agent._update, raw.floating_table_control
    error = None

    def array(tensor):
        return tensor.detach().cpu().numpy().copy()

    def capture():
        nonlocal frame_count
        if start_time is None or frame_count >= frame_limit:
            return
        now = float(raw.gym.get_sim_time(raw.sim))
        elapsed = now - start_time
        if elapsed + 1e-6 < frame_count / fps:
            return
        table_states = array(raw._table_root_states)
        object_states = array(raw._cube_root_states)
        robot_states = array(raw._robot_root_states)
        commands = array(raw.table_commands)
        motion = raw.level03_motion
        targets_v, targets_k = array(motion.target_speed), array(motion.target_curvature)
        targets_x = (np.zeros_like(targets_v) if motion.x_motion is None else
                     array(motion.x_motion.target_speed))
        curvature, phase, progress = array(motion.curvature), array(motion.driver.phase), array(raw.progress_buf)
        raw.gym.fetch_results(raw.sim, True)
        for index in args.record_envs:
            table, obj, robot, command = table_states[index], object_states[index], robot_states[index], commands[index]
            sample = dict(time=elapsed, episode=episode[index], episode_step=int(progress[index]),
                episode_start_time=episode_start[index], reset_elapsed_s=elapsed-episode_start[index],
                vx=float(command[0]), vy=float(command[1]), yaw=float(command[5]),
                target_vx=float(targets_x[index]), target_vy=float(targets_v[index]),
                target_yaw=float(targets_v[index]*targets_k[index]),
                curvature=float(curvature[index]), driver_phase=int(phase[index]),
                table_speed=float(np.linalg.norm(table[7:9])), object_speed=float(np.linalg.norm(obj[7:9])),
                table_yaw_rate=float(table[12]), object_yaw_rate=float(obj[12]),
                table_x=float(table[0]), table_y=float(table[1]), table_z=float(table[2]),
                object_x=float(obj[0]), object_y=float(obj[1]), object_z=float(obj[2]),
                robot_x=float(robot[0]), robot_y=float(robot[1]), robot_z=float(robot[2]))
            samples[index].append(sample)
            # Root tensors and camera coordinates are local to each Isaac Gym
            # environment. Keep both robot and carrier visible in the wide view.
            target = (robot[:3] + obj[:3]) / 2
            target[2] = .43
            scale = max(1., float(np.linalg.norm(robot[:2]-obj[:2])) / 2.5)
            eye = target + np.array([.3, -3.5, 2.1]) * scale
            close_target = (table[:3] + obj[:3]) / 2
            close_eye = close_target + np.array([.48, -.68, .52])
            for handle, origin, focus in ((cameras[index][0], eye, target),
                                           (cameras[index][1], close_eye, close_target)):
                raw.gym.set_camera_location(handle, raw.envs[index], gymapi.Vec3(*origin), gymapi.Vec3(*focus))
        raw.gym.step_graphics(raw.sim)
        raw.gym.render_all_camera_sensors(raw.sim)
        for index in args.record_envs:
            frames = []
            for handle, width, height in ((cameras[index][0], 824, 500), (cameras[index][1], 440, 332)):
                rgba = raw.gym.get_camera_image(raw.sim, raw.envs[index], handle, gymapi.IMAGE_COLOR)
                frames.append(np.asarray(rgba).reshape(height, width, 4)[..., :3].copy())
            frame = compose_training_frame(*frames, samples[index], args.seconds,
                raw.level03_linear_acceleration, raw.level03_yaw_acceleration,
                env_id=index, max_episode_steps=raw.max_episode_length)
            writers[index].append_data(frame)
            if frame_count == 0 or (frame_count+1) % (fps*5) == 0:
                imageio.imwrite(output / ("env{:02d}".format(index)) / "preview_latest.png", frame)
        frame_count += 1
        if frame_count % (fps*5) == 0:
            print("Recorded three training views: {:.2f}/{:.2f} s; episodes {}; PPO updates {}".format(
                frame_count/fps, args.seconds, episode, stats["ppo_updates"]), flush=True)

    def reset_idx(env_ids=None):
        ids = list(range(raw.num_envs)) if env_ids is None else array(env_ids).tolist()
        tracked = [index for index in args.record_envs if index in ids]
        previous = {index: dict(episode=episode[index], steps=int(raw.progress_buf[index].item()),
            terminated=bool(raw._terminate_buf[index].item())) for index in tracked}
        result = original_reset_idx(env_ids)
        if start_time is not None:
            elapsed = float(raw.gym.get_sim_time(raw.sim)) - start_time
            for index in tracked:
                reset_events[index].append(dict(time=elapsed, **previous[index]))
                episode[index] += 1
                episode_start[index] = elapsed
            capture()
        return result

    def reset(*reset_args, **reset_kwargs):
        nonlocal start_time
        result = original_reset(*reset_args, **reset_kwargs)
        if start_time is None:
            start_time = float(raw.gym.get_sim_time(raw.sim))
            capture()
        return result

    def step(actions):
        magnitude = float(actions.abs().max().item())
        stats["max_action_abs"] = max(stats["max_action_abs"], magnitude)
        stats["nonzero_action_steps"] += int(magnitude > 1e-7)
        result = original_step(actions)
        stats["high_level_steps"] += 1
        return result

    def update(*update_args, **update_kwargs):
        result = original_update(*update_args, **update_kwargs)
        stats["ppo_updates"] += 1
        return result

    def control():
        previous = raw.table_commands[:, [0, 1, 5]].clone()
        original_control()
        actual = raw.table_commands[:, [0, 1, 5]]
        delta = (actual-previous).abs().max(dim=0).values
        vx, vy, yaw = actual.abs().max(dim=0).values
        speed = torch.maximum(vx, vy)
        assert speed <= raw.level03_motion_cfg["maxSpeed"] + 1e-6
        assert yaw <= raw.level03_motion_cfg["maxYawRate"] + 1e-6
        if raw.level03_motion.translation_axes == "y":
            assert bool((raw.table_commands[:, 0] == 0).all())
        assert delta[:2].max() <= raw.level03_linear_acceleration*raw.sim_params.dt + 1e-7
        assert delta[2] <= raw.level03_yaw_acceleration*raw.sim_params.dt + 1e-7
        stats["physics_steps"] += 1
        for key, value in (("max_speed", speed), ("max_vx", vx), ("max_vy", vy),
                            ("max_yaw_rate", yaw), ("max_speed_step", delta[:2].max()),
                            ("max_yaw_step", delta[2])):
            stats[key] = max(stats[key], float(value))

    try:
        for index in args.record_envs:
            directory = output / ("env{:02d}".format(index))
            directory.mkdir()
            cameras[index] = []
            for width, height, fov in ((824, 500, 62.), (440, 332, 48.)):
                props = gymapi.CameraProperties()
                props.width, props.height, props.horizontal_fov = width, height, fov
                handle = raw.gym.create_camera_sensor(raw.envs[index], props)
                if handle < 0:
                    raise RuntimeError("Could not create training preview camera")
                cameras[index].append(handle)
            writers[index] = imageio.get_writer(str(directory / "level03_training.mp4"), fps=fps,
                codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p")
        raw.reset_idx, env.reset, env.step = reset_idx, reset, step
        agent._update, raw.floating_table_control = update, control
        raw._video_frame_callback = capture
        trainer.train()
        finalize_teacher_logging(trainer)
        if frame_count != frame_limit:
            raise RuntimeError("Training ended before all requested frames were captured")
        assert stats["ppo_updates"] > 0 and stats["nonzero_action_steps"] > 0
    except BaseException as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        raw._video_frame_callback = None
        for writer in writers.values():
            writer.close()
        for index in args.record_envs:
            directory = output / ("env{:02d}".format(index))
            if samples[index]:
                with (directory / "motion.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(samples[index][0]))
                    writer.writeheader()
                    writer.writerows(samples[index])
                (directory / "resets.json").write_text(json.dumps(reset_events[index], indent=2)+"\n")
        summary = dict(error=error, seed=args.seed, frames=frame_count, fps=fps,
            duration_seconds=frame_count/fps, recorded_envs=args.record_envs,
            parallel_envs=args.num_envs, planned_training_steps=120000, stop_after=stop_after,
            training_statistics=stats, episode_limit=raw.max_episode_length,
            robot_start_distance_range=raw.robot_start_distance_range,
            motion_config=raw.level03_motion_cfg, teacher_argv=argv,
            motion_axes=dict(frame="table_local", translation_axes=raw.level03_motion.translation_axes,
                             zero_velocity_axis="x" if raw.level03_motion.x_motion is None else None,
                             speed_and_acceleration_limits="per local axis", yaw_rate="vy * curvature"),
            mode="Actual early PPO training via get_trainer(is_eval=False).train(); no checkpoint loaded",
            differences_from_launcher=["16 instead of 8192 parallel environments", "short preview run",
                "offscreen spectator cameras", "logging under outputs with W&B disabled"],
            reset_events=reset_events,
            videos=[str(output / ("env{:02d}".format(index)) / "level03_training.mp4") for index in args.record_envs])
        (output / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
        (output / "environment.yaml").write_text(yaml.safe_dump(trainer.teacher_config, sort_keys=False))
        writer = getattr(agent, "writer", None)
        if writer is not None:
            writer.flush()
            writer.close()
        for index, handles in cameras.items():
            for handle in handles:
                raw.gym.destroy_camera_sensor(raw.sim, raw.envs[index], handle)
        if raw.viewer is not None:
            raw.gym.destroy_viewer(raw.viewer)
        raw.gym.destroy_sim(raw.sim)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
