#!/usr/bin/env python3
"""Record Level03 car-like table/object motion using the production simulator.

Run from an activated dqwbc environment. No checkpoint or training is needed.
The robot starts outside the preview area and receives zero high-level actions.
The ordinary low-level policy, table controller and object contacts still run.
Level03 uses signed local vx and vy; legacy local-Y configurations remain supported.

Examples (outputs are placed inside DQ_high-level/outputs):
  python record_table_motion.py --output_dir outputs/level03_full_turn_preview --seconds 20
  python record_table_motion.py --output_dir outputs/level03_yaw_limit_preview --scenario yaw-limit
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def yaw_limit_segments(max_speed, curvature_limit):
    """Five-second preset targets; production filters still control the response."""
    targets = (
        ("forward_positive_yaw", max_speed, curvature_limit),
        ("forward_negative_yaw", max_speed, -curvature_limit),
        ("reverse_negative_yaw", -max_speed, curvature_limit),
        ("reverse_positive_yaw", -max_speed, -curvature_limit),
        ("stop", 0., 0.),
    )
    return [dict(name=name, start_seconds=5. * index, end_seconds=5. * (index + 1),
                 target_speed=speed, target_curvature=curvature,
                 target_yaw_rate=speed * curvature)
            for index, (name, speed, curvature) in enumerate(targets)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="New directory within DQ_high-level/outputs; relative names go there")
    parser.add_argument("--scenario", choices=("random", "yaw-limit"), default="random",
                        help="Production random driving or a labeled sequence of boundary targets")
    parser.add_argument("--seconds", type=float,
                        help="Video duration; default 20 s for random, 25 s for yaw-limit")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.seconds is None:
        args.seconds = 25. if args.scenario == "yaw-limit" else 20.
    if not math.isfinite(args.seconds) or args.seconds < 1. / 25:
        parser.error("--seconds must be at least one video frame (0.04 s)")
    output_root = (ROOT / "outputs").resolve()
    requested_output = args.output_dir.expanduser()
    if not requested_output.is_absolute():
        requested_output = ((ROOT if requested_output.parts and requested_output.parts[0] == "outputs" else output_root)
                            / requested_output)
    output = requested_output.resolve()
    try:
        output.relative_to(output_root)
    except ValueError:
        parser.error("--output_dir must be inside {}".format(output_root))
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)

    # Isaac Gym must precede torch, including imports through other modules.
    from isaacgym import gymapi
    import imageio.v2 as imageio
    import numpy as np
    import torch
    import yaml
    from skrl.utils import set_seed
    from train_multistate_DQ_teacher import create_env, get_predict_point
    from utils.config import get_params, load_cfg, configure_teacher_experiment
    from utils.table_motion_video import compose_frame

    set_seed(args.seed)
    teacher_args = get_params([
        "--task", "B1Z1PickMulti", "--object_name", "sugar_box", "--num_envs", "1",
        "--motion_level", "Level03", "--sgs_mode", "disabled", "--no_tsil",
        "--rl_device", args.device, "--sim_device", args.device,
        "--graphics_device_id", args.device.split(":")[-1], "--headless",
        "--roboinfo", "--observe_gait_commands", "--small_value_set_zero",
        "--no_velocity_obs", "--rand_control", "--seed", str(args.seed),
    ])
    teacher_args.eval = True
    cfg = configure_teacher_experiment(load_cfg(ROOT / "data/cfg/DQ_teacher.yaml"), teacher_args)
    assets = cfg["env"]["asset"]["asset_multi"]
    cfg["env"]["asset"]["asset_multi"] = {"sugar_box": assets["sugar_box"]}
    cfg["env"].update(numEnvs=1, wandb=False, useTanh=False, obj_move_prob=0.,
                       noVelocityObs=True, totalTimesteps=teacher_args.timesteps,
                       maxEpisodeLength=100000,
                       robotStartDistanceRange=[10., 10.])
    cfg["sensor"]["enableCamera"] = False
    cfg["enableCameraSensors"] = True
    inputs = get_predict_point(
        cube_predict_info_path="contact_grasp_info_mul",
        cube_root_states_info_path="30all_nomove_cube_root_states5.pt",
        num_env=1, intervel=23, delta_height=.1,
        object_indices=[assets["sugar_box"]["dict_idx"]])
    env = create_env(cfg, teacher_args, *inputs)
    raw = env._env
    cameras, writer = [], None
    samples, switches = [], []
    stats = {"max_local_velocity_step": 0., "max_yaw_rate_step": 0., "physics_steps": 0,
             "max_forward_acceleration": 0., "max_yaw_acceleration": 0.,
             "max_forward_speed": 0., "max_x_speed": 0., "max_x_acceleration": 0.,
             "max_yaw_rate": 0.}
    fps = 25
    frame_limit = round(args.seconds * fps)
    video_path = output / "level03_car_motion.mp4"
    preset_segments = []
    try:
        raw.reset_idx()
        env.reset()
        start_time = float(raw.gym.get_sim_time(raw.sim))
        initial = raw._table_root_states[0, :3].detach().cpu().numpy().copy()
        motion = raw.level03_motion
        active_segment = -1
        if args.scenario == "yaw-limit":
            preset_segments = yaw_limit_segments(motion.max_speed, motion.curvature_limit)

            def preset_segment_index():
                elapsed = float(raw.gym.get_sim_time(raw.sim)) - start_time
                # Resolve switches on the physics clock, independent of float32
                # countdown accumulation and randomized high-level durations.
                return min(int((elapsed + 1e-7) / 5.), len(preset_segments) - 1)

            def sample_preset():
                nonlocal active_segment
                active_segment = preset_segment_index()
                segment = preset_segments[active_segment]
                motion.set_manual_targets(segment["target_speed"], segment["target_curvature"])

            # Only replace target generation; never assign either filter stage
            # or its output. The initial video frame already shows its target.
            sample_preset()
        for width, height, fov in ((824, 500, 58.), (440, 332, 48.)):
            props = gymapi.CameraProperties()
            props.width, props.height, props.horizontal_fov = width, height, fov
            handle = raw.gym.create_camera_sensor(raw.envs[0], props)
            if handle < 0:
                raise RuntimeError("Unable to create the offscreen camera")
            cameras.append(handle)
        overview_target = initial + np.array([0., 0., 0.05])
        # Natural driving can keep travelling in one direction. Follow it in
        # the wide shot, with a fixed orientation and visible ground reference.
        overview_follow = not bool(preset_segments)
        overview_scale = 2.0 if preset_segments else 1.0
        overview_offset = np.array([2.0, -2.7, 2.6]) * overview_scale
        overview_eye = overview_target + overview_offset
        last_capture_time = 0.
        raw.gym.set_camera_location(cameras[0], raw.envs[0],
                                    gymapi.Vec3(*overview_eye), gymapi.Vec3(*overview_target))
        writer = imageio.get_writer(str(video_path), fps=fps, codec="libx264", quality=8,
                                   macro_block_size=1, pixelformat="yuv420p")
        original_control = raw.floating_table_control

        def checked_control():
            motion = raw.level03_motion
            if preset_segments and preset_segment_index() != active_segment:
                sample_preset()
            previous = raw.table_commands.clone()
            original_control()
            delta = raw.table_commands - previous
            xy_delta = float(delta[0, :2].abs().max().item())
            yaw_delta = float(delta[0, 5].abs().item())
            if xy_delta > raw.level03_linear_acceleration * raw.sim_params.dt + 1e-7:
                raise AssertionError("Local velocity component changed faster than configured")
            if yaw_delta > raw.level03_yaw_acceleration * raw.sim_params.dt + 1e-7:
                raise AssertionError("Yaw rate changed faster than configured")
            forward_speed = float(motion.speed[0].abs().item())
            x_speed = float(raw.table_commands[0, 0].abs().item())
            yaw_rate = float(raw.table_commands[0, 5].abs().item())
            forward_accel = (motion.speed_stage - motion.speed) / motion.speed_tau
            steering_rate = (motion.curvature_stage - motion.curvature) / motion.steering_tau
            yaw_accel = forward_accel * motion.curvature + motion.speed * steering_rate
            if max(x_speed, forward_speed) > raw.level03_motion_cfg["maxSpeed"] + 1e-6:
                raise AssertionError("Local velocity component exceeded the speed limit")
            if yaw_rate > raw.level03_motion_cfg["maxYawRate"] + 1e-6:
                raise AssertionError("Yaw rate exceeded the car yaw limit")
            if motion.translation_axes == "y" and x_speed != 0:
                raise AssertionError("Legacy local-Y motion must keep vx at zero")
            x_accel = (0. if motion.x_motion is None else
                       float(((motion.x_motion.speed_stage - motion.x_motion.speed)
                              / motion.x_motion.speed_tau).abs().max()))
            stats["max_forward_acceleration"] = max(stats["max_forward_acceleration"], float(forward_accel.abs().max()))
            stats["max_yaw_acceleration"] = max(stats["max_yaw_acceleration"], float(yaw_accel.abs().max()))
            stats["max_forward_speed"] = max(stats["max_forward_speed"], forward_speed)
            stats["max_x_speed"] = max(stats["max_x_speed"], x_speed)
            stats["max_x_acceleration"] = max(stats["max_x_acceleration"], x_accel)
            stats["max_yaw_rate"] = max(stats["max_yaw_rate"], yaw_rate)
            stats["max_local_velocity_step"] = max(stats["max_local_velocity_step"], xy_delta)
            stats["max_yaw_rate_step"] = max(stats["max_yaw_rate_step"], yaw_delta)
            stats["physics_steps"] += 1
            if raw.mask_step_change[0].item():
                switches.append(float(raw.gym.get_sim_time(raw.sim)) - start_time)

        def capture():
            nonlocal overview_target, last_capture_time
            elapsed = float(raw.gym.get_sim_time(raw.sim)) - start_time
            if len(samples) >= frame_limit or elapsed + 1e-6 < len(samples) / fps:
                return
            command = raw.table_commands[0].detach().cpu().numpy()
            table = raw._table_root_states[0].detach().cpu().numpy()
            obj = raw._cube_root_states[0].detach().cpu().numpy()
            motion = raw.level03_motion
            sample = {
                "time": elapsed, "vx": float(command[0]), "vy": float(command[1]),
                "yaw": float(command[5]),
                "target_vx": (0. if motion.x_motion is None else
                              float(motion.x_motion.target_speed[0].item())),
                "target_vy": float(motion.target_speed[0].item()),
                "target_yaw": float((motion.target_speed * motion.target_curvature)[0].item()),
                "curvature": float(motion.curvature[0].item()),
                "target_curvature": float(motion.target_curvature[0].item()),
                "driver_phase": int(motion.driver.phase[0].item()) if motion.driver_enabled else -1,
                "object_speed": float(np.linalg.norm(obj[7:9])),
                "table_speed": float(np.linalg.norm(table[7:9])),
                "table_yaw_rate": float(table[12]), "object_yaw_rate": float(obj[12]),
                "table_x": float(table[0]), "table_y": float(table[1]), "table_z": float(table[2]),
                "object_x": float(obj[0]), "object_y": float(obj[1]), "object_z": float(obj[2]),
            }
            samples.append(sample)
            close_target = (table[:3] + obj[:3]) / 2
            close_eye = close_target + np.array([.48, -.68, .52])
            raw.gym.fetch_results(raw.sim, True)
            if overview_follow:
                weight = -math.expm1(-max(0., elapsed - last_capture_time) / .8)
                overview_target += weight * (table[:3] + np.array([0., 0., .05]) - overview_target)
                raw.gym.set_camera_location(cameras[0], raw.envs[0],
                                            gymapi.Vec3(*(overview_target + overview_offset)),
                                            gymapi.Vec3(*overview_target))
            last_capture_time = elapsed
            raw.gym.set_camera_location(cameras[1], raw.envs[0],
                                        gymapi.Vec3(*close_eye), gymapi.Vec3(*close_target))
            raw.gym.step_graphics(raw.sim)
            raw.gym.render_all_camera_sensors(raw.sim)
            frames = []
            for camera, width, height in ((cameras[0], 824, 500), (cameras[1], 440, 332)):
                rgba = raw.gym.get_camera_image(raw.sim, raw.envs[0], camera, gymapi.IMAGE_COLOR)
                frames.append(np.asarray(rgba).reshape(height, width, 4)[..., :3].copy())
            frame = compose_frame(*frames, samples, args.seconds,
                                  raw.level03_linear_acceleration, raw.level03_yaw_acceleration,
                                  scenario=args.scenario, overview_follow=overview_follow)
            writer.append_data(frame)
            if len(samples) == 1:
                imageio.imwrite(output / "preview.png", frame)
            if len(samples) % (fps * 5) == 0:
                imageio.imwrite(output / "preview_latest.png", frame)
                print("Recorded {:.2f}/{:.2f} seconds ({} frames)".format(
                    len(samples) / fps, args.seconds, len(samples)), flush=True)

        raw.floating_table_control = checked_control
        raw._video_frame_callback = capture
        capture()
        actions = torch.zeros((1, env.action_space.shape[0]), device=env.rl_device)
        with torch.no_grad():
            while len(samples) < frame_limit:
                env.step(actions)
        writer.close()
        writer = None
        with (output / "motion.csv").open("w") as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            csv_writer.writeheader()
            csv_writer.writerows(samples)
        summary = {
            "video": str(video_path), "frames": len(samples), "fps": fps,
            "duration_seconds": len(samples) / fps, "seed": args.seed,
            "scenario": args.scenario, "preset_segments": preset_segments,
            "target_source": ("preset boundary targets; production motion filters"
                              if preset_segments else "production correlated remote-control driver"),
            "overview_camera": "following wide view" if overview_follow else "fixed view",
            "motion_level": raw.table_level, "physics_dt": raw.sim_params.dt,
            "motion_axes": {"frame": "table_local", "translation_axes": motion.translation_axes,
                            "zero_velocity_axis": "x" if motion.x_motion is None else None,
                            "speed_and_acceleration_limits": "per local axis",
                            "yaw_rate": "vy * curvature"},
            "linear_acceleration_local": raw.level03_linear_acceleration,
            "yaw_acceleration": raw.level03_yaw_acceleration,
            "motion_config": raw.level03_motion_cfg,
            "speed_time_constant": raw.level03_motion.speed_tau,
            "steering_time_constant": raw.level03_motion.steering_tau,
            "target_switch_times": switches, "validation": stats,
            "measured_table_yaw_range": [min(s["table_yaw_rate"] for s in samples),
                                         max(s["table_yaw_rate"] for s in samples)],
            "preview_setup": "Production environment; zero high-level actions; robot 10 m away; no episode resets.",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "environment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        raw._video_frame_callback = None
        if writer is not None:
            writer.close()
        for handle in cameras:
            raw.gym.destroy_camera_sensor(raw.sim, raw.envs[0], handle)
        raw.gym.destroy_sim(raw.sim)


if __name__ == "__main__":
    main()
