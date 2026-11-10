"""
SuperMimic Training Script for DQ_WBC Dynamic Object Grasping
Adapted from train_multistate_DQ_teacher.py
"""
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
from copy import deepcopy
import torch
import os
import json
import re
from pathlib import Path
from termcolor import cprint
import torch.nn as nn
from datetime import datetime
import wandb
from tqdm import tqdm

from envs import *
from PredictPiontTransform.predictpoint_mul import PredictPoint
from utils.config import (load_cfg, get_params, copy_cfg, configure_teacher_experiment,
                          teacher_run_end, apply_pose_prediction_config)
import utils.wrapper as wrapper
from legged_gym.envs.manip_loco.b1z1_config import B1Z1RoughCfg

# SuperMimic components
from supermimic import (
    DQStateAdapter,
    StateCapture,
    SuperMimicAgent,
    ReplayBuffer
)


def create_env(cfg, args, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor):
    """Create DQ_WBC environment (same as original)."""
    apply_pose_prediction_config(cfg, args)
    cfg_terrain = B1Z1RoughCfg()

    cfg["env"]["enableDebugVis"] = args.debugvis
    cfg["env"]["cameraMode"] = "full"
    cfg["env"]["smallValueSetZero"] = args.small_value_set_zero
    if args.last_commands:
        cfg["env"]["lastCommands"] = True
    if args.record_video:
        cfg["record_video"] = True
    if args.control_freq is not None:
        cfg["env"]["controlFrequencyLow"] = int(args.control_freq)

    robot_start_pose = (-2.00, 0, 0.55)
    if args.eval:
        robot_start_pose = (-0.85, 0, 0.55)

    _env = eval(args.task)(
        cfg=cfg, rl_device=args.rl_device, sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id, headless=args.headless,
        use_roboinfo=args.roboinfo, observe_gait_commands=args.observe_gait_commands,
        no_feature=args.no_feature, mask_arm=args.mask_arm, pitch_control=args.pitch_control,
        rand_control=args.rand_control, arm_delay=args.arm_delay, robot_start_pose=robot_start_pose,
        rand_cmd_scale=args.rand_cmd_scale, rand_depth_clip=args.rand_depth_clip,
        stop_pick=args.stop_pick, table_height=args.table_height, eval=args.eval,
        camera_6p_tensor=camera_6p_tensor, grasp_cv_tensor=grasp_cv_tensor,
        cube_init_tensor=cube_init_tensor, cfg_terrain=cfg_terrain
    )

    wrapped_env = wrapper.IsaacGymPreview3Wrapper(_env)
    return wrapped_env


def teacher_config_path(args, is_eval=False, strict_eval=False):
    """Get config path (same as original)."""
    if args.cfg_file:
        return args.cfg_file
    if is_eval and args.checkpoint:
        run_dir = Path(args.checkpoint).resolve().parent.parent
        metadata_path = run_dir / "teacher_run.json"
        if metadata_path.is_file():
            with metadata_path.open() as stream:
                config_name = json.load(stream).get("config_file", "DQ_teacher.yaml")
            candidate = run_dir / Path(config_name).name
            if candidate.is_file():
                return str(candidate)
        candidate = run_dir / "DQ_teacher.yaml"
        if candidate.is_file():
            return str(candidate)
    if strict_eval:
        raise ValueError("No saved teacher YAML found; specify --cfg_file explicitly")
    return "data/cfg/DQ_teacher.yaml"


def train_supermimic(args):
    """Main SuperMimic training loop."""

    # Setup
    device = args.rl_device

    # Load config
    cfg_path = teacher_config_path(args)
    cfg = load_cfg(cfg_path)

    cprint(f"[SuperMimic] Loaded config from {cfg_path}", "cyan")
    cprint(f"[SuperMimic] Training with {cfg['env']['numEnvs']} parallel environments", "cyan")

    # Initialize data for PredictPoint (dynamically generated)
    from train_multistate_DQ_teacher import get_predict_point

    # Get prediction point data using same method as original training
    cube_predict_info_path = cfg["env"].get("cube_predict_info", "SUGAR_BOX")
    cube_root_states_info_path = cfg["env"].get("cube_root_states_info", "all_cube_root_states.pt")
    num_env = cfg['env']['numEnvs']
    intervel = cfg["env"].get("table_level_intervel", 512)
    delta_height = cfg["env"].get("table_level_delta_height", 0.11)
    object_indices = cfg["env"].get("object_indices")

    cprint(f"[SuperMimic] Generating prediction point data for {num_env} environments...", "cyan")
    camera_6p_tensor, grasp_cv_tensor, cube_init_tensor = get_predict_point(
        cube_predict_info_path=cube_predict_info_path,
        cube_root_states_info_path=cube_root_states_info_path,
        num_env=num_env,
        intervel=intervel,
        delta_height=delta_height,
        object_indices=object_indices
    )
    cprint(f"[SuperMimic] Data generated: camera {camera_6p_tensor.shape}, grasp {grasp_cv_tensor.shape}, cube {cube_init_tensor.shape}", "green")

    # Create environment
    env = create_env(cfg, args, camera_6p_tensor, grasp_cv_tensor, cube_init_tensor)

    num_envs = env.num_envs
    action_dim = env.num_actions

    cprint(f"[SuperMimic] Environment created: {num_envs} envs, action_dim={action_dim}", "green")

    # Initialize SuperMimic components
    state_adapter = DQStateAdapter(env.unwrapped, device=device)

    agent = SuperMimicAgent(
        state_dim=51,
        action_dim=action_dim,
        device=device,
        num_ensemble=5,
        world_model_hidden=[512, 512, 512],
        world_model_lr=3e-4,
        policy_hidden=[256, 256, 256],
        policy_lr=3e-4,
        critic_lr=1e-3,
        action_std_init=0.3,
        gamma=0.99,
        gae_lambda=0.95,
        imagination_horizon=16,
        world_model_batch_size=256,
        world_model_train_steps=10,
        imagination_batch_size=512,
        policy_update_steps=5,
        replay_capacity=1000000,
        min_buffer_size=10000
    )

    agent.attach_state_adapter(state_adapter)

    cprint("[SuperMimic] Agent initialized", "green")

    # Setup logging
    if args.wandb:
        wandb.init(
            project="dq_wbc_supermimic",
            name=f"supermimic_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config={
                "num_envs": num_envs,
                "state_dim": 51,
                "action_dim": action_dim,
                "imagination_horizon": 16,
                "world_model_ensemble": 5
            }
        )

    # Setup output directory
    output_dir = Path("runs") / f"supermimic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    cprint(f"[SuperMimic] Output directory: {output_dir}", "cyan")

    # Training loop parameters
    # Calculate iterations from timesteps (same as original PPO training)
    # timesteps = iterations * num_envs * collect_steps_per_iteration
    collect_steps_per_iteration = 100  # 100 steps per iteration
    max_iterations = args.timesteps // (num_envs * collect_steps_per_iteration)
    exploration_noise = 0.1

    cprint(f"[SuperMimic] Total timesteps: {args.timesteps}", "cyan")
    cprint(f"[SuperMimic] Calculated iterations: {max_iterations}", "cyan")

    # Reset environment
    obs_dict = env.reset()
    current_state = state_adapter.capture_state()

    # Training statistics
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    total_episodes = 0

    cprint(f"[SuperMimic] Starting training for {max_iterations} iterations", "green")
    cprint(f"[SuperMimic] Warmup phase: first 1000 iterations (world model only)", "yellow")

    # Main training loop
    for iteration in tqdm(range(1, max_iterations + 1)):

        # ==== Step 1: Collect real transitions ====
        iteration_rewards = []
        iteration_native_rewards = []

        for step in range(collect_steps_per_iteration):
            # Select action with exploration noise
            with torch.no_grad():
                action = agent.select_action(
                    current_state,
                    exploration_noise=exploration_noise,
                    deterministic=False
                )

            # Step environment
            obs_dict, rewards, dones, infos = env.step(action)

            # Capture next state
            next_state = state_adapter.capture_state()

            # Compute geometric proxy reward
            geometric_reward = state_adapter.compute_geometric_proxy_reward(current_state, next_state)

            # Store transition in replay buffer
            agent.observe_transition(
                state=current_state,
                action=action,
                reward=geometric_reward,
                next_state=next_state,
                done=dones,
                native_reward=rewards
            )

            # Update statistics
            episode_rewards += rewards
            episode_lengths += 1

            # Handle episode termination
            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
                for idx in done_indices:
                    total_episodes += 1
                    iteration_rewards.append(episode_rewards[idx].item())
                    iteration_native_rewards.append(episode_rewards[idx].item())

                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

            # Update current state
            current_state = next_state

        # ==== Step 2: Update world model and policy ====
        update_info = agent.update()

        # ==== Step 3: Logging ====
        if iteration % 10 == 0:
            stats = agent.get_stats()
            buffer_stats = stats['buffer_stats']

            log_info = {
                'iteration': iteration,
                'phase': update_info.get('phase', 'unknown'),
                'buffer_size': buffer_stats['buffer_size'],
                'buffer_utilization': buffer_stats['buffer_utilization'],
                'mean_reward': buffer_stats.get('mean_reward', 0),
                'world_model_loss': update_info.get('world_model_loss', 0),
                'prediction_error': update_info.get('prediction_error', 0),
                'ensemble_disagreement': update_info.get('ensemble_disagreement', 0),
                'total_episodes': total_episodes
            }

            if iteration > 1000:  # After warmup
                log_info.update({
                    'policy_loss': update_info.get('policy_loss', 0),
                    'critic_loss': update_info.get('critic_loss', 0),
                    'imagination_reward_mean': update_info.get('imagination_reward_mean', 0),
                    'approx_kl': update_info.get('approx_kl', 0)
                })

            if iteration_rewards:
                log_info['episode_reward_mean'] = np.mean(iteration_rewards)
                log_info['episode_reward_std'] = np.std(iteration_rewards)

            # Print to console
            if iteration % 50 == 0:
                phase_str = "WARMUP" if iteration <= 1000 else "JOINT"
                cprint(f"\n[Iter {iteration}] Phase: {phase_str}", "cyan")
                cprint(f"  Buffer: {buffer_stats['buffer_size']}/{agent.replay_buffer.capacity} ({buffer_stats['buffer_utilization']*100:.1f}%)", "white")
                cprint(f"  World Model Loss: {update_info.get('world_model_loss', 0):.4f}", "white")
                if iteration > 1000:
                    cprint(f"  Policy Loss: {update_info.get('policy_loss', 0):.4f}", "white")
                    cprint(f"  Imagination Reward: {update_info.get('imagination_reward_mean', 0):.3f}", "white")
                if iteration_rewards:
                    cprint(f"  Episode Reward: {np.mean(iteration_rewards):.3f} ± {np.std(iteration_rewards):.3f}", "green")

            # Log to wandb
            if args.wandb:
                wandb.log(log_info, step=iteration)

        # ==== Step 4: Checkpointing ====
        if iteration % 500 == 0:
            checkpoint_path = checkpoint_dir / f"agent_{iteration}"
            agent.save(str(checkpoint_path))
            cprint(f"[Checkpoint] Saved at iteration {iteration}", "magenta")

        # ==== Step 5: Decay exploration noise ====
        if iteration > 1000:
            exploration_noise = max(0.05, exploration_noise * 0.9995)

    # Final save
    final_path = checkpoint_dir / "agent_final"
    agent.save(str(final_path))
    cprint(f"\n[SuperMimic] Training completed! Final checkpoint saved to {final_path}", "green")

    if args.wandb:
        wandb.finish()

    env.close()


if __name__ == "__main__":
    args = get_params()

    # Set default task if not specified
    if not hasattr(args, 'task') or args.task is None:
        args.task = "B1Z1PickMulti"

    # Enable wandb by default
    if not hasattr(args, 'wandb'):
        args.wandb = True

    # Set default iterations
    if not hasattr(args, 'iterations'):
        args.iterations = 10000

    train_supermimic(args)
