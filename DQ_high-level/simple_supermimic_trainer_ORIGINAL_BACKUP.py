"""Train SuperMimic v2 with the DQ teacher's reset and logging conventions."""
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from termcolor import cprint

from supermimic.logging import SuperMimicMetrics
from supermimic_v2 import DQRobotMDP, StateCapture, Context, SuperMimicAgentV2
from utils.config import teacher_run_end


def _policy_observation(result):
    """Unwrap reset tuples/asymmetric obs and snapshot mutable simulator buffers."""
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, dict):
        result = result['obs']
    if not isinstance(result, torch.Tensor):
        raise TypeError(f"Expected a tensor policy observation, got {type(result).__name__}")
    return result.detach().clone()


def run_supermimic_training(env, args, cfg):
    """Run v2 on an existing environment; timesteps counts vector steps."""
    device = args.rl_device
    num_envs = env.num_envs
    raw_env = getattr(env, '_env', env)
    max_iterations = teacher_run_end(args.timesteps, stop_after=args.stop_after)
    run_dir = Path(args.experiment_dir) / args.wandb_name
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config_name = Path(args.cfg_file or 'DQ_teacher.yaml').name
    with (run_dir / config_name).open('w') as stream:
        yaml.safe_dump(deepcopy(cfg), stream, default_flow_style=False)

    wandb = None
    metrics = SuperMimicMetrics(device)
    global_step = 0
    last_iteration = 0
    total_episodes = 0
    update_metrics = {}
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)

    try:
        cprint("\n[SuperMimic v2] Initializing components...", "cyan")
        state_capture = StateCapture(raw_env, device=device)
        context = Context(raw_env, device=device)
        mdp = DQRobotMDP(raw_env, device=device)
        agent_config = dict(
            wm_hidden=(2048, 2048, 2048, 2048, 2048),
            policy_hidden=(2048, 1024, 512, 256), critic_hidden=(256, 128),
            wm_ensemble_size=5, wm_window=8, horizon=16,
            wm_batch_size=2048, policy_batch_size=4096,
            learning_rate=2e-4, gamma=0.99, lambda_return=0.95,
            uncertainty_penalty=0.5, critic_epochs=5, critic_minibatches=4,
            max_grad_norm=1.0, warmup_iterations=1000,
            replay_capacity=1000000, min_buffer_size=10000,
        )
        agent = SuperMimicAgentV2(mdp=mdp, device=device, **agent_config)

        if args.wandb:
            import wandb as wandb_sdk
            wandb = wandb_sdk
            run = wandb.init(
                project=args.wandb_project, name=args.wandb_name, dir=str(run_dir),
                config={
                    **deepcopy(cfg), 'arguments': vars(args), 'algorithm': 'supermimic_v2',
                    'num_envs': num_envs, 'state_dim': mdp.state_dim,
                    'obs_dim': mdp.obs_dim, 'action_dim': mdp.action_dim,
                    'context_dim': mdp.context_dim, 'supermimic': agent_config,
                    'model_state': 'physical34 + measured_ee_pose7 + full_policy_observation',
                    'logging': {'interval_vector_steps': 10, 'x_axis': 'iteration',
                                'episode_rewards': 'all completed episodes in the window',
                                'family_success_window': 'latest 100 completed episodes per SGS family'},
                },
            )
            wandb.define_metric('iteration')
            wandb.define_metric('*', step_metric='iteration', step_sync=True)
            if not getattr(run, 'offline', False):
                url = getattr(run, 'url', None)
                if url:
                    cprint(f"[SuperMimic v2] W&B run: {url}", "cyan")

        cprint(f"[SuperMimic v2] Vector steps: {max_iterations}/{args.timesteps}; "
               f"environments: {num_envs}; warmup: {agent.warmup_iterations}", "cyan")
        cprint(f"  State: {mdp.state_dim}; observation: {mdp.obs_dim}; "
               f"action: {mdp.action_dim}; context: {mdp.context_dim}", "white")

        def write_metrics(iteration):
            if not metrics.pending_steps:
                return
            stats = agent.get_stats()
            log_info = {
                'iteration': iteration, 'global_step': global_step,
                'phase': stats['phase'], 'total_episodes': total_episodes,
                'buffer_size': stats['buffer_stats']['buffer_size'],
                'buffer_utilization': stats['buffer_stats']['buffer_utilization'],
                'world_model_train_steps': stats['world_model_updates'],
                'policy_train_steps': stats['policy_updates'],
                **update_metrics, **metrics.flush(),
            }
            if iteration % 50 == 0 or iteration == max_iterations:
                cprint(f"[Iter {iteration}/{max_iterations}] {stats['phase']} | "
                       f"WM loss: {log_info.get('world_model_loss', 0):.4f} | "
                       f"episodes: {total_episodes}", "cyan")
            if wandb is not None:
                wandb.log(log_info, step=global_step)

        current_obs = _policy_observation(env.reset()).to(device)
        env_ids = torch.arange(num_envs, device=device)
        state_capture.reset_episodes(env_ids)
        context.reset_context(env_ids)
        current_state = mdp.capture_state(state_capture.capture(), current_obs)
        current_context = context.get_context().clone()

        try:
            for iteration in range(1, max_iterations + 1):
                with torch.no_grad():
                    action = agent.select_action(current_obs, deterministic=False)
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_obs, rewards, terminated, truncated, infos = step_result
                elif len(step_result) == 4:
                    next_obs, rewards, terminated, infos = step_result
                    truncated = infos.get('time_outs', torch.zeros_like(terminated))
                else:
                    raise ValueError(f"Unexpected step format: {len(step_result)}")
                next_obs = _policy_observation(next_obs).to(device)
                rewards = rewards.reshape(num_envs).to(device=device, dtype=torch.float32)
                dones = (terminated.reshape(num_envs).to(device=device, dtype=torch.bool)
                         | truncated.reshape(num_envs).to(device=device, dtype=torch.bool))
                global_step += num_envs

                # Capture the terminal state with the action and clock from
                # this transition; reset bookkeeping only after storing it.
                state_capture.update_action(action)
                state_capture.increment_timestep()
                next_state = mdp.capture_state(state_capture.capture(), next_obs)
                agent.observe_transition(
                    state=current_state, action=action, reward=rewards,
                    next_state=next_state, done=dones, context=current_context,
                    obs=current_obs, next_obs=next_obs,
                )
                with torch.no_grad():
                    proxy_reward, _ = mdp.reward(current_state, action, next_state, current_context)
                metrics.record_step(rewards, proxy_reward, infos)
                last_iteration = iteration
                episode_rewards += rewards
                episode_lengths += 1

                if dones.any():
                    done_ids = dones.nonzero(as_tuple=False).flatten()
                    total_episodes += done_ids.numel()
                    metrics.record_episodes(episode_rewards[done_ids], episode_lengths[done_ids])
                    episode_rewards[done_ids] = 0
                    episode_lengths[done_ids] = 0
                    current_obs = _policy_observation(env.reset()).to(device)
                    state_capture.reset_episodes(done_ids)
                    context.reset_context(done_ids)
                    current_state = mdp.capture_state(state_capture.capture(), current_obs)
                    current_context = context.get_context().clone()
                else:
                    current_obs = next_obs
                    current_state = next_state
                metrics.update_snapshots(getattr(raw_env, 'extras', infos))
                if not args.headless:
                    env.render()

                update_metrics = agent.update()
                if iteration % 10 == 0 or iteration == max_iterations:
                    write_metrics(iteration)
                if iteration % 500 == 0 or iteration == max_iterations:
                    path = checkpoint_dir / f'agent_{global_step}'
                    agent.save(str(path))
                    cprint(f"[Checkpoint] Saved to {path}_agent.pt", "magenta")
        except KeyboardInterrupt:
            write_metrics(last_iteration)
            raise

        cprint(f"[SuperMimic v2] Completed {max_iterations} vector steps, "
               f"{global_step} transitions, {total_episodes} episodes.", "green")
        return agent
    finally:
        try:
            if wandb is not None:
                wandb.finish()
        finally:
            env.close()
