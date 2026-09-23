#!/usr/bin/env python3
"""Test the D1 + Piper-L task-space IK controller.

This test uses the same ``ManipLoco.step`` and ``ManipLoco._control_ik`` path
as the training task.  The policy is not loaded: in this task the six arm
policy outputs are masked in ``step`` and the arm target is generated from
the task-space end-effector goal.  Zero actions are sent to the environment
so that the legs hold their configured default pose and wheel speed targets
remain zero.

By default the robot base is fixed and the end effector follows a small,
circle around its pose after reset.  Use ``--target_mode training`` to test
the actual sampled training poses, or ``init_start``/``init_end`` to hold
either initialization pose.  Both position and orientation are checked in
every environment.  Use ``--free_base`` to include base motion.

Example with a viewer::

    cd DQ_low-level/legged_gym/scripts
    python test_d1_piper_l_ik.py --sim_device cuda:0 --rl_device cuda:0

Example for a headless run and CSV output::

    python test_d1_piper_l_ik.py --headless --sim_device cuda:0 \
        --rl_device cuda:0 --target_mode training --num_envs 16 \
        --duration 30 --csv /tmp/d1_piper_l_ik.csv
"""

from __future__ import print_function

import csv
import copy
import math
import os
import sys
from pathlib import Path

import numpy as np


# Make ``legged_gym`` importable when this file is run directly from scripts/.
LOW_LEVEL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if LOW_LEVEL_ROOT not in sys.path:
    sys.path.insert(0, LOW_LEVEL_ROOT)

import isaacgym  # noqa: F401  (must be imported before legged_gym)
from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import orientation_error, quat_mul, quat_conjugate

import torch

from legged_gym.envs import *  # noqa: F401,F403  (registers d1_piper_l)
from legged_gym.utils import task_registry


def parse_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "d1_piper_l",
            "help": "Registered task to test (must be d1_piper_l)",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Accepted for compatibility with train.py; no policy is loaded",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 1,
            "help": "Number of environments (all are checked)",
        },
        {
            "name": "--target_mode",
            "type": str,
            "default": "circle",
            "choices": ["circle", "init_start", "init_end", "training"],
            "help": "Follow a circle, hold a configured pose, or follow training commands",
        },
        {
            "name": "--duration",
            "type": float,
            "default": 20.0,
            "help": "Test duration in seconds",
        },
        {
            "name": "--warmup",
            "type": float,
            "default": 1.0,
            "help": "Initial hold time excluded from the summary metrics",
        },
        {
            "name": "--target_radius",
            "type": float,
            "default": 0.08,
            "help": "Circle radius in meters",
        },
        {
            "name": "--target_height",
            "type": float,
            "default": 0.04,
            "help": "Vertical circle amplitude in meters",
        },
        {
            "name": "--target_period",
            "type": float,
            "default": 5.0,
            "help": "Circle period in seconds",
        },
        {
            "name": "--log_interval",
            "type": float,
            "default": 1.0,
            "help": "Console logging interval in seconds",
        },
        {
            "name": "--position_tolerance",
            "type": float,
            "default": 0.03,
            "help": "Position RMS threshold used for the final PASS/FAIL line",
        },
        {
            "name": "--orientation_tolerance_deg",
            "type": float,
            "default": 8.0,
            "help": "Orientation RMS threshold in degrees (each environment)",
        },
        {
            "name": "--csv",
            "type": str,
            "default": "",
            "help": "Optional CSV path for per-step IK metrics",
        },
        {
            "name": "--free_base",
            "action": "store_true",
            "help": "Keep the robot base free instead of fixing it for the test",
        },
        {
            "name": "--seed",
            "type": int,
            "default": 1,
            "help": "Random seed used by the task reset",
        },
    ]
    # Isaac Gym adds --headless, --sim_device, --pipeline, --physx/--flex,
    # and the remaining simulator options when headless=True is requested.
    args = gymutil.parse_arguments(
        description="D1 + Piper-L task-space IK test",
        headless=True,
        custom_parameters=custom_parameters,
    )

    # task_registry.make_env expects the aliases produced by legged_gym's
    # get_args(), while gymutil.parse_arguments exposes compute_device_id.
    args.sim_device_id = args.compute_device_id
    args.stop_update_goal = False
    args.rows = 2
    args.cols = 1
    args.observe_gait_commands = False
    args.record_video = False
    args.stand_by = False
    args.vel_obs = False
    args.pitch_control = False
    args.resume = False
    args.experiment_name = None
    args.run_name = None
    args.load_run = None
    args.checkpoint = -1
    args.max_iterations = None

    if args.task != "d1_piper_l":
        raise ValueError(
            "This script validates the d1_piper_l task, got {!r}".format(args.task)
        )
    if args.duration <= 0.0:
        raise ValueError("--duration must be positive")
    if args.warmup < 0.0:
        raise ValueError("--warmup must be non-negative")
    if args.target_radius < 0.0 or args.target_height < 0.0:
        raise ValueError("--target_radius and --target_height must be non-negative")
    if args.target_period <= 0.0:
        raise ValueError("--target_period must be positive")
    if args.log_interval <= 0.0:
        raise ValueError("--log_interval must be positive")
    if args.position_tolerance <= 0.0:
        raise ValueError("--position_tolerance must be positive")
    if args.orientation_tolerance_deg <= 0.0:
        raise ValueError("--orientation_tolerance_deg must be positive")
    if args.warmup >= args.duration:
        raise ValueError("--warmup must be shorter than --duration")
    if args.num_envs < 1:
        raise ValueError("--num_envs must be at least one")
    return args


def configure_test_environment(args):
    """Create a deterministic, single-purpose copy of the task config."""
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.env.num_envs = args.num_envs

    # Fixing the base isolates the arm controller.  The option remains
    # available because a free-base run is useful as a secondary check.
    env_cfg.asset.fix_base_link = not args.free_base
    env_cfg.env.teleop_mode = args.target_mode != "training"
    env_cfg.env.episode_length_s = max(env_cfg.env.episode_length_s, args.duration + 2.0)
    env_cfg.init_state.rand_yaw_range = 0.0
    env_cfg.init_state.origin_perturb_range = 0.0
    env_cfg.init_state.init_vel_perturb_range = 0.0
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]

    # Disable task randomization that is unrelated to the IK measurement.
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_motor = False
    env_cfg.domain_rand.randomize_gripper_mass = False
    env_cfg.domain_rand.push_robots = False

    # Two terrain rows avoid the one-row curriculum division by zero while
    # keeping terrain construction small for a one-env diagnostic.
    env_cfg.terrain.num_rows = args.rows
    env_cfg.terrain.num_cols = args.cols
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.height = [0.0, 0.0]
    return task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)[0]


def set_target(env, target_pos, target_quat):
    """Set the target consumed by ManipLoco.step on the next control step."""
    env.curr_ee_goal_cart_world[:] = target_pos
    env.ee_goal_orn_quat[:] = target_quat


def compute_ik_diagnostics(env, target_pos, target_quat):
    """Evaluate the exact damped-least-squares command used by the task."""
    current_quat = env.ee_orn / torch.linalg.vector_norm(
        env.ee_orn, dim=-1, keepdim=True
    ).clamp_min(1e-8)
    dpos = target_pos - env.ee_pos
    drot = orientation_error(target_quat, current_quat)
    dpose = torch.cat((dpos, drot), dim=-1).unsqueeze(-1)
    delta_q = env._control_ik(dpose)
    singular_values = torch.linalg.svdvals(env.ee_j_eef)
    return (
        delta_q,
        torch.linalg.vector_norm(dpos, dim=-1),
        torch.linalg.vector_norm(drot, dim=-1),
        singular_values[:, -1],
    )


def make_target(center, target_quat, time_s, args):
    """Generate a smooth reachable Cartesian target around the start pose."""
    if time_s < args.warmup:
        return center.clone(), target_quat.clone()

    phase = 2.0 * math.pi * (time_s - args.warmup) / args.target_period
    offset = torch.zeros_like(center)
    # Start at the hold pose so the first moving sample has no position jump.
    offset[:, 0] = args.target_radius * math.sin(phase)
    offset[:, 1] = 0.5 * args.target_radius * (1.0 - math.cos(phase))
    offset[:, 2] = args.target_height * (1.0 - math.cos(phase))
    return center + offset, target_quat.clone()


def draw_debug(env, target_pos, target_quat, target_geom, actual_geom, axes_geom):
    if env.viewer is None:
        return

    target_np = target_pos[0].detach().cpu().numpy()
    actual_np = env.ee_pos[0].detach().cpu().numpy()
    target_q = target_quat[0].detach().cpu().numpy()
    actual_q = (
        env.ee_orn[0]
        / torch.linalg.vector_norm(env.ee_orn[0]).clamp_min(1e-8)
    ).detach().cpu().numpy()

    target_pose = gymapi.Transform(
        p=gymapi.Vec3(*[float(v) for v in target_np]),
        r=gymapi.Quat(*[float(v) for v in target_q]),
    )
    actual_pose = gymapi.Transform(
        p=gymapi.Vec3(*[float(v) for v in actual_np]),
        r=gymapi.Quat(*[float(v) for v in actual_q]),
    )

    env.gym.clear_lines(env.viewer)
    gymutil.draw_lines(target_geom, env.gym, env.viewer, env.envs[0], target_pose)
    gymutil.draw_lines(actual_geom, env.gym, env.viewer, env.envs[0], actual_pose)
    gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[0], target_pose)
    gymutil.draw_lines(axes_geom, env.gym, env.viewer, env.envs[0], actual_pose)
    env.gym.step_graphics(env.sim)
    env.gym.draw_viewer(env.viewer, env.sim, True)


def run_test(args):
    env = None
    csv_file = None
    writer = None
    try:
        env = configure_test_environment(args)
        print(
            "Testing {} on sim_device={} (rl_device={} is unused; no policy is loaded)".format(
                args.task, args.sim_device, args.rl_device
            )
        )
        print(
            "base_mode={}, num_envs={}, dt={:.4f}s, arm_dofs={}".format(
                "free" if args.free_base else "fixed",
                env.num_envs,
                env.dt,
                env.cfg.arm.dof_names,
            )
        )

        # BaseTask.reset() performs one real control step and initializes all
        # task tensors, Jacobians and end-effector state.
        env.reset()
        env.debug_viz = False
        if args.target_mode in ("init_start", "init_end"):
            sphere = env.init_start_ee_sphere if args.target_mode == "init_start" else env.init_end_ee_sphere
            env.curr_ee_goal_sphere[:] = sphere
            env._update_ee_goal_pose()
        center = env.ee_pos.clone().detach()
        target_quat = (
            env.ee_orn
            / torch.linalg.vector_norm(env.ee_orn, dim=-1, keepdim=True).clamp_min(1e-8)
        ).clone().detach()

        if env.viewer is not None:
            root = env.root_states[0, :3].detach().cpu().numpy()
            env.set_camera(
                np.asarray(root) + np.asarray([1.4, -1.4, 1.0]),
                np.asarray(root) + np.asarray([0.25, 0.0, 0.55]),
            )
        target_geom = gymutil.WireframeSphereGeometry(
            0.045, 12, 12, None, color=(1.0, 0.1, 0.1)
        )
        actual_geom = gymutil.WireframeSphereGeometry(
            0.045, 12, 12, None, color=(0.1, 1.0, 0.1)
        )
        axes_geom = gymutil.AxesGeometry(scale=0.18)

        if args.csv:
            csv_path = Path(args.csv).expanduser().resolve()
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            csv_file = csv_path.open("w", newline="")
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "time_s",
                    "env_id",
                    "target_x",
                    "target_y",
                    "target_z",
                    "ee_x",
                    "ee_y",
                    "ee_z",
                    "position_error_m",
                    "orientation_error_rad",
                    "ik_delta_q_norm",
                    "jacobian_sigma_min",
                ],
            )
            writer.writeheader()
            print("Writing per-step metrics to {}".format(csv_path))

        zero_actions = torch.zeros(
            env.num_envs, env.num_actions, device=env.device, dtype=torch.float
        )
        num_steps = int(math.ceil(args.duration / env.dt))
        warmup_steps = int(math.ceil(args.warmup / env.dt))
        if warmup_steps >= num_steps:
            raise ValueError("No control steps remain after warmup; increase --duration")
        log_every = max(1, int(round(args.log_interval / env.dt)))
        metric_rows = []
        reset_count = 0

        for step in range(num_steps):
            time_s = step * env.dt
            if args.target_mode == "circle":
                target_pos, target_q = make_target(center, target_quat, time_s, args)
            else:
                # Snapshot the command actually consumed by this step;
                # post_physics_step advances the training trajectory.
                target_pos = env.curr_ee_goal_cart_world.clone()
                target_q = env.ee_goal_orn_quat.clone()
            delta_q, _, _, sigma_min = compute_ik_diagnostics(env, target_pos, target_q)
            set_target(env, target_pos, target_q)
            _, _, _, _, dones, _ = env.step(zero_actions)
            reset_count += int(dones.sum().item())

            position_error = torch.linalg.vector_norm(target_pos - env.ee_pos, dim=-1)
            current_quat = env.ee_orn / torch.linalg.vector_norm(
                env.ee_orn, dim=-1, keepdim=True
            ).clamp_min(1e-8)
            # Geodesic quaternion angle; atan2 remains well behaved at zero
            # error and at 180 degrees, unlike sign(w)-based IK diagnostics.
            relative = quat_mul(target_q, quat_conjugate(current_quat))
            orientation_err = 2 * torch.atan2(
                torch.linalg.vector_norm(relative[:, :3], dim=-1), relative[:, 3].abs()
            )
            for env_id in range(env.num_envs):
                row = {
                    "time_s": time_s,
                    "env_id": env_id,
                    "target_x": float(target_pos[env_id, 0]),
                    "target_y": float(target_pos[env_id, 1]),
                    "target_z": float(target_pos[env_id, 2]),
                    "ee_x": float(env.ee_pos[env_id, 0]),
                    "ee_y": float(env.ee_pos[env_id, 1]),
                    "ee_z": float(env.ee_pos[env_id, 2]),
                    "position_error_m": float(position_error[env_id]),
                    "orientation_error_rad": float(orientation_err[env_id]),
                    "ik_delta_q_norm": float(torch.linalg.vector_norm(delta_q[env_id])),
                    "jacobian_sigma_min": float(sigma_min[env_id]),
                }
                if writer is not None:
                    writer.writerow(row)
                if step >= warmup_steps:
                    metric_rows.append(row)
            if step % log_every == 0 or step == num_steps - 1:
                print(
                    "t={:.2f}s  max_pos_err={:.4f}m  max_ori_err={:.2f}deg  "
                    "min_sigma={:.5f}".format(
                        time_s, float(position_error.max()),
                        math.degrees(float(orientation_err.max())), float(sigma_min.min()),
                    )
                )
            draw_debug(env, target_pos, target_q, target_geom, actual_geom, axes_geom)

        position_errors = np.asarray([r["position_error_m"] for r in metric_rows])
        orientation_errors = np.asarray([r["orientation_error_rad"] for r in metric_rows])
        sigma_mins = np.asarray([r["jacobian_sigma_min"] for r in metric_rows])
        pos_rms = float(np.sqrt(np.mean(position_errors ** 2)))
        pos_max = float(np.max(position_errors))
        ori_rms = float(np.sqrt(np.mean(orientation_errors ** 2)))
        sigma_min = float(np.min(sigma_mins))
        per_env_pos_rms = np.sqrt(np.mean(position_errors.reshape(-1, env.num_envs) ** 2, axis=0))
        per_env_ori_rms = np.sqrt(np.mean(orientation_errors.reshape(-1, env.num_envs) ** 2, axis=0))
        passed = (
            np.all(per_env_pos_rms <= args.position_tolerance)
            and np.all(per_env_ori_rms <= math.radians(args.orientation_tolerance_deg))
            and reset_count == 0
        )
        print(
            "IK summary: position_rms={:.4f}m, position_max={:.4f}m, "
            "orientation_rms={:.2f}deg, min_jacobian_sigma={:.5f}, resets={}".format(
                pos_rms, pos_max, math.degrees(ori_rms), sigma_min, reset_count
            )
        )
        print(
            "IK result: {} (worst-env RMS: {:.4f}m / {:.2f}deg; limits: {:.4f}m / {:.2f}deg)".format(
                "PASS" if passed else "FAIL", np.max(per_env_pos_rms),
                math.degrees(np.max(per_env_ori_rms)), args.position_tolerance,
                args.orientation_tolerance_deg,
            )
        )
        return 0 if passed else 2
    finally:
        if csv_file is not None:
            csv_file.close()
        if env is not None:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    try:
        raise SystemExit(run_test(parse_args()))
    except KeyboardInterrupt:
        raise SystemExit(130)
