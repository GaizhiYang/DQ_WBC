"""
SuperMimic Training Entry Point for DQ_WBC
Direct replacement for train_multistate_DQ_teacher.py with SuperMimic algorithm
Reuses all existing environment setup, SGS, EKF configurations
"""
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import *

import numpy as np
from copy import deepcopy
import torch
import os
import json
from pathlib import Path
from termcolor import cprint
from datetime import datetime
from tqdm import tqdm

# Import all necessary utilities from original training script
from train_multistate_DQ_teacher import (
    create_env,
    teacher_config_path,
    get_predict_point
)
from utils.config import (
    load_cfg, get_params, configure_teacher_experiment,
    teacher_run_end, apply_pose_prediction_config
)
from skrl.utils import set_seed

# Import SuperMimic components
from supermimic import (
    DQStateAdapter,
    SuperMimicAgent
)

# WandB
import wandb


def get_supermimic_trainer():
    """
    Get SuperMimic trainer using same setup as original PPO trainer.
    This function mirrors get_trainer() from train_multistate_DQ_teacher.py
    """
    from supermimic.logging import SuperMimicMetrics

    args = get_params()
    args.eval = False  # SuperMimic is for training only

    set_seed(args.seed)
    args.wandb = args.wandb and (not args.eval) and (not args.debug)

    if args.sgs_mode in ("uniform", "sgs") and (args.resume or args.checkpoint):
        raise ValueError("SGS/uniform-bank training starts from scratch")

    file_path = teacher_config_path(args, is_eval=False, strict_eval=False)
    cfg_file = Path(file_path).name

    cprint(f"[SuperMimic] Training script: {os.path.realpath(__file__)}", "yellow")
    cprint(f"[SuperMimic] Using config file: {os.path.realpath(file_path)}", "red")

    cfg = load_cfg(file_path)
    configure_teacher_experiment(cfg, args, is_eval=False, strict_eval=False)
    prediction_cfg = apply_pose_prediction_config(cfg, args)

    cprint(f"[SuperMimic] Future pose observations: {prediction_cfg}", "yellow")

    if args.no_velocity_obs:
        cfg["env"]["noVelocityObs"] = True

    cprint(f"[SuperMimic] Velocity observations: {'disabled' if cfg['env'].get('noVelocityObs', False) else 'enabled'}", "yellow")

    # Apply TSIL config
    from utils.config import apply_tsil_config
    tsil_cfg = apply_tsil_config(cfg, args, is_eval=False)

    # Handle object_name (required for SGS) - same as original training script
    multi_obj = cfg["env"]["asset"]["asset_multi"]
    if args.object_name is not None:
        if args.object_name not in multi_obj:
            raise ValueError(f"Unknown --object_name {args.object_name!r}. Available objects: {', '.join(multi_obj)}")
        multi_obj = {args.object_name: multi_obj[args.object_name]}
        cfg["env"]["asset"]["asset_multi"] = multi_obj
    object_indices = [obj["dict_idx"] for obj in multi_obj.values()]
    cprint(f"[SuperMimic] Using {len(multi_obj)} object asset(s): {', '.join(multi_obj)}", "yellow")

    # Apply other env configs
    cfg['env']['wandb'] = args.wandb
    cfg['env']["useTanh"] = args.use_tanh if hasattr(args, 'use_tanh') else False
    cfg['env']["near_goal_stop"] = args.near_goal_stop if hasattr(args, 'near_goal_stop') else False
    cfg['env']["obj_move_prob"] = args.obj_move_prob if hasattr(args, 'obj_move_prob') else 0.0

    # Apply num_envs override if specified
    if args.num_envs is not None:
        if args.num_envs <= 0:
            raise ValueError("--num_envs must be a positive integer")
        cfg['env']['numEnvs'] = args.num_envs
        cprint(f"[SuperMimic] Environment count overridden to: {args.num_envs}", "yellow")

    # Get prediction point data (same as original)
    intervel = 23
    args.intervel = intervel
    cube_predict_info_path = "contact_grasp_info_mul"
    cube_root_states_info_path = "30all_nomove_cube_root_states5.pt"
    cprint(f"[SuperMimic] Generating prediction point data...", "cyan")
    camera_6p_tensor, grasp_cv_tensor, cube_init_tensor = get_predict_point(
        cube_predict_info_path=cube_predict_info_path,
        cube_root_states_info_path=cube_root_states_info_path,
        num_env=cfg['env']['numEnvs'],
        intervel=intervel,
        delta_height=0.1,
        object_indices=object_indices
    )

    # Create environment (same as original)
    experiment_cfg = deepcopy(cfg)
    env = create_env(cfg=cfg, args=args,
                     camera_6p_tensor=camera_6p_tensor,
                     grasp_cv_tensor=grasp_cv_tensor,
                     cube_init_tensor=cube_init_tensor)

    device = args.rl_device
    num_envs = env.num_envs
    action_dim = env.num_actions

    cprint(f"[SuperMimic] Environment created: {num_envs} envs, action_dim={action_dim}", "green")

    # Initialize SuperMimic components
    # Access the wrapped environment correctly
    if hasattr(env, '_env'):
        raw_env = env._env  # IsaacGymPreview3Wrapper uses _env
    else:
        raw_env = env

    state_adapter = DQStateAdapter(raw_env, device=device)

    agent_config = dict(
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
    agent = SuperMimicAgent(**agent_config)

    agent.attach_state_adapter(state_adapter)

    cprint("[SuperMimic] Agent initialized", "green")

    # Setup output directory
    experiment_dir = Path(args.experiment_dir) / args.wandb_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # Save config
    import yaml
    with open(experiment_dir / cfg_file, 'w') as f:
        yaml.safe_dump(experiment_cfg, f, default_flow_style=False)

    # Setup WandB
    if args.wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            dir=str(experiment_dir),
            config={
                "algorithm": "SuperMimic",
                "num_envs": num_envs,
                "timesteps": args.timesteps,
                "state_dim": 51,
                "action_dim": action_dim,
                "sgs_mode": args.sgs_mode,
                "ekf_enabled": prediction_cfg['enabled'],
                **experiment_cfg,
                "supermimic": {**agent_config, "warmup_iterations": agent.warmup_iterations},
                "arguments": vars(args),
                "logging": {
                    "interval_vector_steps": 10,
                    "x_axis": "iteration",
                    "reward_components": "weighted native contributions, averaged over envs and window steps",
                    "episode_rewards": "all episodes completed in each logging window",
                    "family_success_window": "latest 100 completed episodes per SGS family",
                },
            }
        )
        # Use vector-environment steps on charts; keep individual transition
        # counts in global_step for comparing runs with different env counts.
        wandb.define_metric("iteration")
        wandb.define_metric("*", step_metric="iteration", step_sync=True)
        if not getattr(wandb_run, "offline", False):
            run_url = getattr(wandb_run, "url", None)
            if run_url:
                cprint(f"[SuperMimic] W&B run: {run_url}", "cyan")

    # Training loop
    # Match the PPO entrypoint: timesteps counts vector-environment steps.
    # Count individual transitions separately; do not divide the training plan
    # by num_envs (that could end training before policy warmup finishes).
    max_timesteps = args.timesteps
    steps_per_env_per_iteration = 1  # How many steps each environment takes per iteration
    max_iterations = teacher_run_end(max_timesteps, stop_after=args.stop_after)

    exploration_noise = 0.1

    cprint(f"\n[SuperMimic] Training Configuration:", "cyan")
    cprint(f"  Planned vector steps: {max_timesteps}", "white")
    cprint(f"  Iterations: {max_iterations}", "white")
    cprint(f"  Steps per iteration per env: {steps_per_env_per_iteration}", "white")
    cprint(f"  Total steps per iteration: {num_envs * steps_per_env_per_iteration}", "white")
    cprint(f"  Warmup iterations: {agent.warmup_iterations}", "white")
    cprint(f"  Imagination horizon: 16", "white")
    cprint(f"  Replay buffer: 1M", "white")
    cprint(f"\n", "white")

    # Training statistics
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    total_episodes = 0
    global_step = 0
    metrics = SuperMimicMetrics(device=device)
    last_iteration = 0
    update_info = {}

    def write_metrics(iteration, update_info):
        # Also used at a short final window or on interruption. An empty window
        # must not repeat the previous W&B point.
        if metrics.pending_steps:
            stats = agent.get_stats()
            buffer_stats = stats['buffer_stats']

            log_info = {
                'iteration': iteration,
                'global_step': global_step,
                'phase': update_info.get('phase', 'unknown'),
                'buffer_size': buffer_stats['buffer_size'],
                'buffer_utilization': buffer_stats['buffer_utilization'],
                'world_model_loss': update_info.get('world_model_loss', 0),
                'prediction_error': update_info.get('prediction_error', 0),
                'ensemble_disagreement': update_info.get('ensemble_disagreement', 0),
                'total_episodes': total_episodes,
                'exploration_noise': exploration_noise,
                'world_model_train_steps': stats.get('world_model_train_steps', 0),
                'policy_train_steps': stats.get('policy_train_steps', 0),
            }

            if iteration > agent.warmup_iterations:
                log_info.update({
                    'policy_loss': update_info.get('policy_loss', 0),
                    'critic_loss': update_info.get('critic_loss', 0),
                    'imagination_reward_mean': update_info.get('imagination_reward_mean', 0),
                    'imagination_reward_std': update_info.get('imagination_reward_std', 0),
                    'value_mean': update_info.get('value_mean', 0),
                    'policy_grad_norm': update_info.get('policy_grad_norm', 0)
                })

            log_info.update(metrics.flush())

            # Console output
            if iteration % 50 == 0:
                phase_str = "WARMUP" if iteration <= agent.warmup_iterations else "JOINT"
                cprint(f"\n[Iter {iteration}/{max_iterations}] Phase: {phase_str}", "cyan")
                cprint(f"  Vector steps: {iteration}/{max_timesteps}; transitions: {global_step}", "white")
                cprint(f"  Buffer: {buffer_stats['buffer_size']}/{agent.replay_buffer.capacity} ({buffer_stats['buffer_utilization']*100:.1f}%)", "white")
                cprint(f"  World Model Loss: {update_info.get('world_model_loss', 0):.4f}", "white")
                if iteration > agent.warmup_iterations:
                    cprint(f"  Policy Loss: {update_info.get('policy_loss', 0):.4f}", "white")
                    cprint(f"  Imagination Reward: {update_info.get('imagination_reward_mean', 0):.3f}", "white")
                if log_info['episode_count']:
                    cprint(f"  Episode Reward: {log_info['episode_reward_mean']:.3f} ± {log_info['episode_reward_std']:.3f}", "green")
                cprint(f"  Episodes: {total_episodes}", "white")

            # WandB logging
            if args.wandb:
                wandb.log(log_info, step=global_step)

    try:
        env.reset()
        current_state = state_adapter.capture_state()

        for iteration in tqdm(range(1, max_iterations + 1), desc="SuperMimic Training"):
            action = agent.select_action(
                current_state, exploration_noise=exploration_noise, deterministic=False
            )
            obs_dict, rewards, terminated, truncated, infos = env.step(action)
            # Isaac Gym returns [num_envs, 1] rewards and Long flags.
            rewards = rewards.reshape(num_envs).to(device=device, dtype=torch.float32)
            dones = (terminated.reshape(num_envs).to(device=device, dtype=torch.bool)
                     | truncated.reshape(num_envs).to(device=device, dtype=torch.bool))
            global_step += num_envs

            next_state = state_adapter.capture_state()
            geometric_reward = state_adapter.compute_geometric_proxy_reward(current_state, next_state)
            agent.observe_transition(
                state=current_state, action=action, reward=geometric_reward,
                next_state=next_state, done=dones, native_reward=rewards
            )

            # Read rewards before reset can mutate the environment buffers.
            metrics.record_step(rewards, geometric_reward, infos)
            last_iteration = iteration
            episode_rewards += rewards
            episode_lengths += 1
            if dones.any():
                done_indices = dones.nonzero(as_tuple=False).flatten()
                total_episodes += done_indices.numel()
                metrics.record_episodes(episode_rewards[done_indices], episode_lengths[done_indices])
                episode_rewards[done_indices] = 0
                episode_lengths[done_indices] = 0

                # Store the terminal transition before resetting. SGS settles
                # its family windows during reset, so read snapshots afterwards.
                env.reset()
                current_state = state_adapter.capture_state()
            else:
                current_state = next_state
            metrics.update_snapshots(getattr(raw_env, 'extras', infos))

            if not args.headless:
                env.render()
            update_info = agent.update()
            if iteration % 10 == 0 or iteration == max_iterations:
                write_metrics(iteration, update_info)

            if iteration % 500 == 0 or iteration == max_iterations:
                checkpoint_path = checkpoint_dir / f"agent_{global_step}"
                agent.save(str(checkpoint_path))
                cprint(f"[Checkpoint] Saved at iteration {iteration} (step {global_step})", "magenta")

            if iteration > agent.warmup_iterations:
                exploration_noise = max(0.05, exploration_noise * 0.9995)
    except KeyboardInterrupt:
        write_metrics(last_iteration, update_info)
        raise
    finally:
        try:
            if args.wandb:
                wandb.finish()
        finally:
            env.close()

    # Training complete
    cprint(f"\n[SuperMimic] Training completed!", "green")
    cprint(f"  Total episodes: {total_episodes}", "white")
    cprint(f"  Vector steps: {max_iterations}; transitions: {global_step}", "white")
    cprint(f"  Output directory: {experiment_dir}", "white")

    return agent


if __name__ == "__main__":
    agent = get_supermimic_trainer()
