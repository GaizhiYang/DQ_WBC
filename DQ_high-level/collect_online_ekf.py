#!/usr/bin/env python3
"""Capture the teacher's actual online EKF and dense future truth, without training.

Run from the dqwbc environment, e.g.:
  python collect_online_ekf.py --level Level03 --output outputs/ekf_check/level03_seed43
The zero high-level action policy leaves the original low-level standing policy
active. This measures forecasting, not learned grasp-policy success.
"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def append_truth(dense, sample):
    """Keep one snapshot per time, retaining the new episode's reset seed.

    Some reset paths do not advance simulation time. Replacing the terminal
    snapshot at that timestamp avoids duplicate times and leaves old episodes
    ending at the previous dense sample, so interpolation cannot cross a
    teleport. Ordinary task resets do advance physics and append a new sample.
    """
    if dense["times"] and sample["times"] <= dense["times"][-1] + 1e-9:
        if sample["times"] < dense["times"][-1] - 1e-9:
            raise ValueError("Simulation time moved backwards")
        for key, value in sample.items():
            dense[key][-1] = value
        return
    for key, value in sample.items():
        dense[key].append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=("Level02", "Level03"), default="Level03")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ekf-history", type=int, default=5)
    parser.add_argument("--ekf-q-scale", type=float, default=1.)
    parser.add_argument("--ekf-r-scale", type=float, default=.01)
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 12:
        parser.error("num_envs must be positive and steps must be at least 12")
    if args.ekf_history < 2 or any(not math.isfinite(value) or value <= 0
            for value in (args.ekf_q_scale, args.ekf_r_scale)):
        parser.error("EKF history must be >= 2 and Q/R scales must be finite and positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)

    # Isaac Gym must be imported before torch.
    from isaacgym import gymapi  # noqa: F401
    import numpy as np
    import torch
    import yaml
    from skrl.utils import set_seed
    from train_multistate_DQ_teacher import create_env, get_predict_point
    from utils.config import get_params, load_cfg, configure_teacher_experiment

    set_seed(args.seed)
    teacher_args = get_params([
        "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
        "--num_envs", str(args.num_envs), "--motion_level", args.level,
        "--sgs_mode", "disabled", "--no_tsil", "--future_pose_obs",
        "--future_pose_horizons", "0.30", "0.60", "0.90",
        "--ekf_history", str(args.ekf_history), "--ekf_q_scale", str(args.ekf_q_scale),
        "--ekf_r_scale", str(args.ekf_r_scale),
        "--rl_device", args.device, "--sim_device", args.device,
        "--graphics_device_id", args.device.split(":")[-1], "--headless",
        "--roboinfo", "--observe_gait_commands", "--small_value_set_zero",
        "--no_velocity_obs", "--rand_control", "--success_mode", "hold",
        "--seed", str(args.seed),
    ])
    teacher_args.eval = False  # Match training reset distance, not eval's close start.
    cfg = configure_teacher_experiment(load_cfg(ROOT / "data/cfg/DQ_teacher.yaml"), teacher_args)
    assets = cfg["env"]["asset"]["asset_multi"]
    cfg["env"]["asset"]["asset_multi"] = {"sugar_box": assets["sugar_box"]}
    cfg["env"].update(numEnvs=args.num_envs, wandb=False, useTanh=False,
                      obj_move_prob=0., noVelocityObs=True, totalTimesteps=args.steps)
    cfg["sensor"]["enableCamera"] = False
    cfg["enableCameraSensors"] = False
    inputs = get_predict_point(
        cube_predict_info_path="contact_grasp_info_mul",
        cube_root_states_info_path="30all_nomove_cube_root_states5.pt",
        num_env=args.num_envs, intervel=23, delta_height=.1,
        object_indices=[assets["sugar_box"]["dict_idx"]])
    env = create_env(cfg, teacher_args, *inputs)
    raw = env._env
    predictor = raw.object_pose_predictor
    assert predictor is not None and predictor.history == args.ekf_history
    fields = ("times", "episode_ids", "world_poses", "relative_poses", "predictions",
              "valid", "history_poses", "history_times", "driver_phase",
              "table_commands", "lifted", "robot_roots", "done")
    records = {key: [] for key in fields}
    dense = {key: [] for key in ("times", "episode_ids", "relative_poses", "world_poses",
                                 "driver_phase", "table_commands")}
    episode_ids = np.zeros(args.num_envs, dtype=np.int64)
    error = None

    def array(tensor):
        return tensor.detach().cpu().numpy().copy()

    def capture_truth():
        now = float(raw.gym.get_sim_time(raw.sim))
        phase = (array(raw.level03_motion.driver.phase) if args.level == "Level03"
                 else np.full(args.num_envs, -1, dtype=np.int64))
        append_truth(dense, dict(times=now, episode_ids=episode_ids.copy(),
            relative_poses=array(raw._current_object_pose_in_arm_frame()),
            world_poses=array(raw._cube_root_states[:, :7]), driver_phase=phase,
            table_commands=array(raw.table_commands)))

    try:
        raw.reset_idx()
        env.reset()
        capture_truth()
        raw._video_frame_callback = capture_truth
        actions = torch.zeros((args.num_envs, env.action_space.shape[0]), device=env.rl_device)
        with torch.no_grad():
            for step in range(args.steps):
                _, _, terminated, truncated, _ = env.step(actions)
                now = float(raw.gym.get_sim_time(raw.sim))
                done = array((terminated | truncated).view(-1)).astype(bool)
                phase = (array(raw.level03_motion.driver.phase) if args.level == "Level03"
                         else np.full(args.num_envs, -1, dtype=np.int64))
                sample = dict(times=now, episode_ids=episode_ids.copy(),
                    world_poses=array(raw._cube_root_states[:, :7]),
                    relative_poses=array(raw._current_object_pose_in_arm_frame()),
                    predictions=array(predictor.future_poses), valid=array(predictor.valid),
                    history_poses=array(predictor.poses), history_times=array(predictor.times),
                    driver_phase=phase, table_commands=array(raw.table_commands),
                    lifted=array(raw.lifted_now), robot_roots=array(raw._robot_root_states), done=done)
                for key, value in sample.items():
                    records[key].append(value)
                capture_truth()
                if done.any() and step + 1 < args.steps:
                    # The terminal frame belongs to the old episode. Record the
                    # post-reset seed with new IDs; never interpolate a teleport.
                    env.reset()
                    episode_ids[done] += 1
                    capture_truth()
                if (step + 1) % 50 == 0 or step + 1 == args.steps:
                    print("{} seed {}: {}/{} steps, {:.2f}s sim time".format(
                        args.level, args.seed, step + 1, args.steps, now), flush=True)
    except BaseException as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        raw._video_frame_callback = None
        if records["times"]:
            packed = {key: np.asarray(value) for key, value in records.items()}
            packed.update({"dense_truth_" + key: np.asarray(value) for key, value in dense.items()})
            packed["horizons"] = np.asarray(predictor.horizons, dtype=np.float64)
            np.savez_compressed(output / "capture.npz", **packed)
        metadata = dict(motion_level=args.level, seed=args.seed, num_envs=args.num_envs,
            motion_translation_axis=("local_" + raw.level03_motion.translation_axes
                                     if args.level == "Level03" else "local_y"),
            completed_steps=len(records["times"]), requested_steps=args.steps, error=error,
            policy="zero_high_level_with_original_low_level_standing_policy",
            frame="object relative to moving arm base at each measurement/future timestamp",
            quaternion_order="xyzw", timestamps="actual gym.get_sim_time",
            truth_sampling="each low-level control cycle and reset; same-time reset seeds replace terminal dense sample",
            measurement_noise="none; same privileged root poses as teacher EKF",
            ekf=dict(history=predictor.history, q_scale=args.ekf_q_scale, r_scale=args.ekf_r_scale,
                     horizons=list(predictor.horizons), extra_diagnostic_horizon=.3),
            sim_dt=float(raw.sim_params.dt), control_frequency_inv=raw.control_freq_inv,
            dense_truth_interval_s=float(raw.sim_params.dt) * raw.control_freq_inv,
            episode_length=cfg["env"]["maxEpisodeLength"],
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            note="Forecast quality only; no trained high-level policy or grasp-success comparison")
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "environment.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        if raw.viewer is not None:
            raw.gym.destroy_viewer(raw.viewer)
        raw.gym.destroy_sim(raw.sim)


if __name__ == "__main__":
    main()
