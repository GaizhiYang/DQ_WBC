#!/usr/bin/env python3
"""Collect real Isaac Gym object poses and evaluate Paparazzo forecasts.

Run in the dqwbc environment:
  python collect_object_pose_prediction.py --headless --rand_control \
      --object_name sugar_box --num_envs 4 --timesteps 200 \
      --output outputs/sugar_box_pose

Without --checkpoint: zero high-level actions, pretrained low-level locomotion,
and the existing moving-table controller. With --checkpoint: deterministic
teacher actions including the saved observation standardizer. No training.
"""

import json
import os
import re
from copy import deepcopy
from pathlib import Path

from utils.config import get_parser, load_cfg


def build_parser():
    # Keep --help and argument validation usable without importing Isaac Gym.
    from evaluate_object_pose_prediction import add_evaluation_arguments
    parser = get_parser()
    parser.description = __doc__
    parser.set_defaults(task="B1Z1PickMulti", object_name="sugar_box", num_envs=4,
                        timesteps=200, roboinfo=True, observe_gait_commands=True,
                        small_value_set_zero=True, no_velocity_obs=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for trajectories, config, and evaluation")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "data/cfg/DQ_teacher.yaml")
    parser.add_argument("--motion_level", choices=["Level0" + str(i) for i in range(6)], help="Override motion level in memory only")
    parser.add_argument("--episode_length", type=int, help="Override maximum high-level steps per episode")
    parser.add_argument("--velocity_obs", action="store_false", dest="no_velocity_obs", help="For checkpoints trained with velocity observations")
    parser.add_argument("--collect_only", action="store_true", help="Save CSV trajectories without running prediction evaluation")
    add_evaluation_arguments(parser)
    return parser


def collect(args):
    # Isaac Gym MUST be imported before PyTorch, including indirect imports.
    from isaacgym import gymapi
    import torch
    import yaml
    from skrl.utils import set_seed
    from skrl.resources.preprocessors.torch import RunningStandardScaler
    from train_multistate_DQ_teacher import create_env, get_predict_point, Policy
    from utils.pose_trajectory import PoseTrajectoryRecorder

    set_seed(args.seed)  # The imported training module sets its own seed first.
    cfg = load_cfg(str(args.config))
    assets = cfg["env"]["asset"]["asset_multi"]
    if args.object_name not in assets:
        raise ValueError("Unknown object: {}".format(args.object_name))
    cfg["env"]["asset"]["asset_multi"] = {args.object_name: assets[args.object_name]}
    cfg["env"].update(numEnvs=args.num_envs, wandb=False, useTanh=args.use_tanh,
                       near_goal_stop=args.near_goal_stop, obj_move_prob=0.0,
                       noVelocityObs=args.no_velocity_obs, totalTimesteps=args.timesteps)
    if args.motion_level:
        cfg["env"]["D1_bench_task_level"] = args.motion_level
    if args.episode_length:
        cfg["env"]["maxEpisodeLength"] = args.episode_length
    # This benchmark records privileged poses; no camera rendering is needed.
    cfg["sensor"]["enableCamera"] = False
    cfg["env"]["enableDebugVis"] = False
    args.eval = True
    args.wandb = False
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=args.rl_device)
        if "policy" not in checkpoint or "state_preprocessor" not in checkpoint:
            raise ValueError("Expected a teacher checkpoint containing policy and state_preprocessor")
        match = re.search(r"agent_(\d+)\.pt$", args.checkpoint)
        if match:
            cfg["env"]["globalStepCounter"] = int(match.group(1))
    camera, grasps, initial = get_predict_point(cube_predict_info_path="contact_grasp_info_mul",
        cube_root_states_info_path="30all_nomove_cube_root_states5.pt", num_env=args.num_envs,
        intervel=23, delta_height=0.1, object_indices=[assets[args.object_name]["dict_idx"]])
    env = create_env(cfg, args, camera, grasps, initial)
    saved_cfg = deepcopy(cfg)  # Include settings applied by create_env().
    raw = env._env
    recorder = None
    completed_steps = 0
    error = None
    try:
        policy = scaler = None
        if checkpoint is not None:
            policy = Policy(env.observation_space, env.action_space, env.rl_device,
                0 if args.no_feature else 1024, 0 if args.no_feature else 128,
                camera, grasps, initial, use_tanh=args.use_tanh,
                clip_actions=args.use_tanh, deterministic=True)
            policy.load_state_dict(checkpoint["policy"])
            policy.to(env.rl_device)
            policy.eval()
            scaler = RunningStandardScaler(size=env.observation_space, device=env.rl_device)
            scaler.load_state_dict(checkpoint["state_preprocessor"])
            scaler.eval()
        # Explicit initial reset, just as the trainer does for terminated envs.
        raw.reset_idx()
        observations, _ = env.reset()
        recorder = PoseTrajectoryRecorder(args.output / "trajectories", env.num_envs)
        previous_time = float(raw.gym.get_sim_time(raw.sim))
        print("Collecting {} / {} with {} environment(s), policy={}".format(
            args.object_name, raw.table_level, env.num_envs, "teacher" if policy else "zero_high_level"), flush=True)
        for step in range(args.timesteps):
            with torch.no_grad():
                if policy is None:
                    actions = torch.zeros((env.num_envs, env.action_space.shape[0]), device=env.rl_device)
                else:
                    actions = policy.act({"states": scaler(observations, train=False)}, role="policy")[0]
                observations, _, terminated, truncated, _ = env.step(actions)
            now = float(raw.gym.get_sim_time(raw.sim))
            done = (terminated | truncated).view(-1)
            # Capture terminal poses BEFORE env.reset(), so no teleport enters a trajectory.
            recorder.append(now, now-previous_time,
                raw._cube_root_states[:, :7].detach().cpu().numpy(),
                done.detach().cpu().numpy(), raw.lifted_now.detach().cpu().numpy(), step)
            previous_time = now
            completed_steps = step + 1
            if done.any() and completed_steps < args.timesteps:
                observations, _ = env.reset()  # Resets only envs whose reset_buf is set.
            if completed_steps % 25 == 0:
                print("Collected {}/{} high-level steps".format(completed_steps, args.timesteps), flush=True)
    except BaseException as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
        raise
    finally:
        if recorder is not None:
            recorder.close("capture_error" if error else "capture_end")
        metadata = dict(arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
            completed_steps=completed_steps, error=error, object_name=args.object_name,
            motion_level=raw.table_level, policy="teacher" if args.checkpoint else "zero_high_level",
            pose_frame="fixed Isaac Gym world frame", quaternion_order="xyzw",
            timestamp_source="gym.get_sim_time", sampling="post high-level step, before reset",
            sim_dt=float(raw.sim_params.dt), control_frequency_inv=raw.control_freq_inv,
            sensor_noise="none; optional measurement noise is added only by offline evaluator")
        (args.output / "collection.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (args.output / "environment.yaml").write_text(yaml.safe_dump(saved_cfg, sort_keys=False))
        if raw.viewer is not None:
            raw.gym.destroy_viewer(raw.viewer)
        raw.gym.destroy_sim(raw.sim)
    return args.output


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.task != "B1Z1PickMulti":
        parser.error("This recorder currently supports --task B1Z1PickMulti")
    if args.num_envs < 1 or args.timesteps < 1 or (args.episode_length is not None and args.episode_length < 1):
        parser.error("num_envs, timesteps, and episode_length must be positive")
    if args.obj_move_prob != 0:
        parser.error("Use --obj_move_prob 0: object teleports would break trajectory continuity")
    if args.resume or args.record_video or args.debugvis:
        parser.error("Use --checkpoint directly; resume/video/debugvis are not supported by this collector")
    if args.control_freq is not None and args.control_freq < (2 if args.rand_control else 1):
        parser.error("control_freq must leave at least one simulation cycle")
    from evaluate_object_pose_prediction import validate_evaluation_arguments
    validate_evaluation_arguments(args)
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    args.paparazzo_dir = args.paparazzo_dir.resolve()
    if args.checkpoint:
        args.checkpoint = str(Path(args.checkpoint).resolve())
        if not Path(args.checkpoint).is_file():
            parser.error("Checkpoint not found: " + args.checkpoint)
    # Check location before allocating a simulator, but import it after Isaac Gym.
    benchmark_path = args.paparazzo_dir / "scripts/evaluate_pose_prediction.py"
    if not args.collect_only and not benchmark_path.is_file():
        parser.error("Paparazzo benchmark not found: " + str(benchmark_path))
    args.output.mkdir(parents=True, exist_ok=False)
    os.chdir(str(Path(__file__).resolve().parent))
    collect(args)
    if not args.collect_only:
        from evaluate_object_pose_prediction import evaluate_recordings
        evaluate_recordings(args.output, args.output / "evaluation", args)


if __name__ == "__main__":
    main()
