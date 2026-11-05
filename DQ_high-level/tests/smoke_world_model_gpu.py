"""Manual real-Isaac-Gym CUDA diagnostic, independent of the training gate.

Run with the project's Isaac Gym Python environment, for example::

    python DQ_high-level/tests/smoke_world_model_gpu.py --device cuda:0

The disposable world model is fitted briefly to real diagnostic transitions.
This checks continuous gripper control, deterministic mean-action BPTT, unchanged
PPO exploration standard deviation, gradient plumbing, frozen modules, KL rollback
and actual timeout transitions. It does not assess model accuracy or grant
permission to use imagined updates during training.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _runtime_environment():
    """Make the active Python environment visible to Isaac Gym and Ninja."""
    library = str(Path(sys.prefix) / "lib")
    binary = str(Path(sys.executable).parent)
    if binary not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = binary + os.pathsep + os.environ.get("PATH", "")
    if library not in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        environment = dict(os.environ)
        environment["LD_LIBRARY_PATH"] = library + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_envs", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("/tmp/dq_world_model_gpu_smoke.json"))
    parser.add_argument("--sgs_bank", type=Path, default=Path("data/sgs/sugar_box_v10.pt"))
    args = parser.parse_args()
    if args.num_envs < 3:
        parser.error("--num_envs must be at least 3 to exercise all SGS families")
    output = args.output.expanduser().resolve()
    bank = args.sgs_bank.expanduser()
    bank = (ROOT / bank).resolve() if not bank.is_absolute() else bank.resolve()
    if not bank.is_file():
        parser.error("SGS bank does not exist: " + str(bank))
    _runtime_environment()

    # Isaac Gym must be imported before torch.
    from isaacgym import gymapi  # noqa: F401
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("This manual diagnostic requires CUDA and GPU simulation")
    device = torch.device("cuda", 0 if device.index is None else device.index)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    output.parent.mkdir(parents=True, exist_ok=True)

    original_cwd, original_argv = Path.cwd(), sys.argv
    original_path = list(sys.path)
    trainer = env = agent = session = original_sample = None
    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        import train_multistate_DQ_teacher as teacher
        from utils.wm_env import TeacherWorldModelAdapter
        from utils.world_model import ResidualDynamics
        from utils.world_model.gradients import freeze_parameters
        from utils.world_model.imagination import ImaginationConfig, ImaginationTrainer
        from utils.world_model.probes import IsaacGymProbeSession, configuration_fingerprint

        sys.argv = [
            "world_model_gpu_diagnostic", "--task", "B1Z1PickMulti",
            "--object_name", "sugar_box", "--roboinfo", "--observe_gait_commands",
            "--small_value_set_zero", "--rand_control", "--no_velocity_obs",
            "--sgs_mode", "sgs", "--sgs_bank", str(bank), "--future_pose_obs",
            "--future_pose_horizons", ".60", ".90", "--headless",
            "--num_envs", str(args.num_envs), "--rl_device", str(device),
            "--sim_device", str(device), "--world_model", "off",
            "--experiment_dir", str(output.parent / (output.stem + "_runs")),
            "--wandb_name", "isolated_mechanics", "--seed", "43", "--timesteps", "120000",
        ]
        trainer = teacher.get_trainer(is_eval=False, enable_logging=False)
        sys.argv = original_argv
        env, agent = trainer.env._env, trainer.agents
        adapter = TeacherWorldModelAdapter(env)
        session = IsaacGymProbeSession(env, adapter)
        metadata = adapter.metadata()
        report = {
            "diagnostic_only": True, "model_untrained": True,
            "training_gate_bypassed": False, "gate_file_generated": False,
            "logging_enabled": False, "imagined_action_mode": "policy_mean",
            "device": str(device), "num_envs": env.num_envs,
            "adapter_schema": metadata["version"],
            "configuration_fingerprint": configuration_fingerprint(metadata),
        }
        # Exercise the real environment's position-target controller at the
        # endpoints and an intermediate aperture; no binary threshold remains.
        old_actions, old_target = env.actions.clone(), env.gripper_dof_pos.clone()
        target_errors = []
        for opening in (0., .25, .5, 1.):
            env.actions[:, 6] = opening
            env.set_gripper()
            expected = env.dof_limits_upper[-1] + opening * (
                env.dof_limits_lower[-1] - env.dof_limits_upper[-1])
            target_errors.append(float((env.gripper_dof_pos - expected).abs().max()))
        env.actions.copy_(old_actions)
        env.gripper_dof_pos.copy_(old_target)
        report["continuous_gripper_control"] = {
            "action_mode": env.cfg["env"]["gripperActionMode"],
            "tested_openings": [0., .25, .5, 1.],
            "target_errors_rad": target_errors,
            "all_targets_match": max(target_errors) < 1e-6,
            "action_space_lower_bound": float(env.action_space.low[6]),
        }
        assert report["continuous_gripper_control"]["all_targets_match"], report

        def state_copy(module):
            return {key: value.detach().clone() for key, value in module.state_dict().items()}

        def unchanged(before, module):
            return all(torch.equal(before[key], value) for key, value in module.state_dict().items())

        def tree_equal(left, right):
            if torch.is_tensor(left):
                return torch.equal(left, right)
            if isinstance(left, dict):
                return left.keys() == right.keys() and all(tree_equal(left[key], right[key]) for key in left)
            if isinstance(left, (list, tuple)):
                return len(left) == len(right) and all(tree_equal(x, y) for x, y in zip(left, right))
            return left == right

        def on_device(tensors):
            tensors = list(tensors)
            return bool(tensors) and all(tensor.is_cuda and tensor.device == device for tensor in tensors)

        # Seed the isolated simulator from every SGS family. This never changes
        # training sampler probabilities or writes a sampler/model checkpoint.
        families = env.sgs_bank.tensors["family_id"]
        ids = torch.stack([(families == i).nonzero().flatten()[0] for i in range(3)])
        original_sample = env.sgs_sampler.sample
        env.sgs_sampler.sample = lambda n: ids[torch.arange(n, device=env.device) % 3]
        env.reset_buf.fill_(1)
        env.reset()
        env.sgs_sampler.sample = original_sample
        transitions = []
        for step in range(6):
            previous, conditions = adapter.capture()
            action = torch.zeros(env.num_envs, 9, device=device)
            action[:, 6] = (step % 3) / 2.
            env.step(action)
            following, _ = adapter.capture()
            transitions.append((previous, env.actions.clone(), conditions, following,
                                torch.full((env.num_envs, 1), env.control_freq_low * env.dt,
                                           device=device)))
            env.reset()
        start, context = adapter.capture()
        observed = adapter.observation(start, context)
        # Initialize real skrl statistics with real diagnostic transitions, then
        # verify that neither auxiliary update changes those statistics.
        agent._state_preprocessor(observed, train=True)
        agent._value_preprocessor(env.rew_buf[:, None], train=True)
        model = ResidualDynamics(adapter, hidden=(512, 512, 512), term_hidden=(64, 64), device=device)
        # The production residual head starts at zero, so its initial physical
        # action Jacobian is correctly zero. Fit this disposable model briefly
        # on real transitions before checking physical gradient connectivity.
        fit_state, fit_action, fit_context, fit_target, fit_dt = (
            torch.cat(parts, 0) for parts in zip(*transitions))
        fit_optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        fit_losses = []
        for _ in range(16):
            fit_optimizer.zero_grad(set_to_none=True)
            prediction = model.step(fit_state, fit_action, fit_context, fit_dt)
            loss = model.normalized_sq_error(prediction, fit_target).mean()
            assert bool(torch.isfinite(loss)), report
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            fit_optimizer.step()
            fit_losses.append(float(loss.detach()))
        fit_optimizer.zero_grad(set_to_none=True)
        report["model_untrained"] = False
        report["model_validated"] = False
        report["diagnostic_model_fit"] = {
            "real_transitions": len(fit_state), "optimizer_steps": len(fit_losses),
            "first_loss": fit_losses[0], "last_loss": fit_losses[-1],
            "purpose": "gradient plumbing only; no prediction-quality or task-performance claim",
        }
        config = ImaginationConfig(horizon=3, lr=1e-5, strength=.05, max_kl=.005)
        bptt = ImaginationTrainer(model, adapter, agent.policy, agent.value, config,
                                  agent._state_preprocessor, agent._value_preprocessor)
        dt = torch.stack([adapter.sample_dt(env.num_envs) for _ in range(3)])
        report["cuda_assertions"] = {
            "actor": on_device(agent.policy.state_dict().values()),
            "critic": on_device(agent.value.state_dict().values()),
            "world_model": on_device(model.state_dict().values()),
            "state_scaler": on_device(agent._state_preprocessor.state_dict().values()),
            "value_scaler": on_device(agent._value_preprocessor.state_dict().values()),
            "inputs": on_device((start, context, observed, dt)),
            "adapter_tables": on_device((adapter.feature_table, adapter.physical_context, adapter.feature_mask)),
        }
        assert all(report["cuda_assertions"].values()), report

        frozen = [model, agent.value, agent._state_preprocessor, agent._value_preprocessor]
        snapshots = [state_copy(module) for module in frozen]
        log_std = agent.policy.log_std_parameter
        original_log_std = log_std.detach().clone()
        assert log_std.requires_grad, "Real PPO must retain trainable exploration variance"

        # The real teacher still samples its Gaussian policy. Only imagined
        # actions use its mean; neither actor mode nor PPO variance is disabled.
        with torch.no_grad():
            normalized = agent._state_preprocessor(observed, train=False)
            sample_a, _, _ = agent.policy.act({"states": normalized}, role="policy")
            sample_b, _, _ = agent.policy.act({"states": normalized}, role="policy")
        assert bool(((sample_a[:, 6] > 0) & (sample_a[:, 6] < 1)).all()), report
        # Differentiate physical state predictions with respect to aperture,
        # independently of reward/action penalties or action-memory copying.
        physical_action = sample_a.detach().clone().requires_grad_(True)
        with freeze_parameters(model):
            predicted = model.step(start, physical_action, context, dt[0])
            physical_gradient = torch.autograd.grad(
                predicted[:, :adapter.physical_dim].square().sum(), physical_action)[0][:, 6]
        report["continuous_gripper_control"].update({
            "physical_gradient_cuda": physical_gradient.is_cuda,
            "physical_gradient_finite": bool(torch.isfinite(physical_gradient).all()),
            "physical_gradient_abs_max": float(physical_gradient.abs().max()),
        })
        assert physical_gradient.is_cuda and bool(torch.isfinite(physical_gradient).all()), report
        assert float(physical_gradient.abs().max()) > 0, report
        report["real_policy_sampling"] = {
            "two_samples_differ": not torch.equal(sample_a, sample_b),
            "log_std_trainable": bool(log_std.requires_grad),
            "log_std_unchanged": torch.equal(original_log_std, log_std),
        }
        assert all(report["real_policy_sampling"].values()), report

        # Durations are fixed before capturing RNG state: the task's random
        # control period remains legitimate randomness outside imagined actions.
        rng_cpu = torch.get_rng_state().clone()
        rng_cuda = torch.cuda.get_rng_state(device).clone()
        with freeze_parameters(*frozen):
            rollout_a = bptt.imagine(start, context, dt=dt)
            rollout_b = bptt.imagine(start, context, dt=dt)
        report["deterministic_imagination"] = {
            "same_start_context_dt": True,
            "all_rollout_tensors_equal": all(
                torch.equal(value, getattr(rollout_b, key))
                for key, value in vars(rollout_a).items()),
            "cpu_rng_unchanged": torch.equal(rng_cpu, torch.get_rng_state()),
            "cuda_rng_unchanged": torch.equal(rng_cuda, torch.cuda.get_rng_state(device)),
            "log_std_unchanged": torch.equal(original_log_std, log_std),
            "log_std_trainable": bool(log_std.requires_grad),
        }
        assert all(report["deterministic_imagination"].values()), report
        del rollout_a, rollout_b

        # Emulate an old auxiliary checkpoint with nonzero Adam moments for
        # log_std. A zero gradient would still move it; grad=None must skip it.
        bptt.optimizer.state[log_std] = {
            "step": torch.tensor(7.),
            "exp_avg": torch.full_like(log_std, .2),
            "exp_avg_sq": torch.full_like(log_std, .3),
        }
        old_std_optimizer = deepcopy(bptt.optimizer.state[log_std])
        log_std.grad = torch.ones_like(log_std)
        model_first, value_first = next(model.parameters()), next(agent.value.parameters())
        model_first.grad = torch.full_like(model_first, .125)
        value_first.grad = torch.full_like(value_first, .25)
        old_model_gradient, old_value_gradient = model_first.grad.clone(), value_first.grad.clone()
        old_actor = state_copy(agent.policy)
        metrics = bptt.update(start, context, dt=dt)
        gradients = [parameter.grad for parameter in agent.policy.parameters() if parameter.grad is not None]
        report["accepted_update"] = {
            "metrics": metrics, "actor_changed": not unchanged(old_actor, agent.policy),
            "actor_gradients_finite": bool(gradients) and all(bool(torch.isfinite(g).all()) for g in gradients),
            "actor_gradients_cuda": on_device(gradients), "actor_gradient_tensor_count": len(gradients),
            "world_model_unchanged": unchanged(snapshots[0], model),
            "critic_unchanged": unchanged(snapshots[1], agent.value),
            "state_scaler_unchanged": unchanged(snapshots[2], agent._state_preprocessor),
            "value_scaler_unchanged": unchanged(snapshots[3], agent._value_preprocessor),
            "existing_model_gradient_preserved": torch.equal(old_model_gradient, model_first.grad),
            "existing_critic_gradient_preserved": torch.equal(old_value_gradient, value_first.grad),
            "frozen_parameter_flags_restored": all(p.requires_grad for m in (model, agent.value) for p in m.parameters()),
            "log_std_unchanged": torch.equal(original_log_std, log_std),
            "log_std_trainable": bool(log_std.requires_grad),
            "log_std_gradient_none": log_std.grad is None,
            "legacy_std_adam_state_unchanged": tree_equal(old_std_optimizer, bptt.optimizer.state[log_std]),
            "gripper_mean_head_gradient_nonzero": bool(agent.policy.net[-1].weight.grad[6].abs().sum() > 0),
        }
        assert metrics.get("accepted") == 1. and metrics["actor_grad_norm"] > 0, report
        assert all(value for key, value in report["accepted_update"].items()
                   if key not in ("metrics", "actor_gradient_tensor_count")), report

        # Deliberately oversize a disposable diagnostic update: the actual KL
        # check must reject it and restore both Actor and Adam state exactly.
        bptt.cfg.max_kl = 1e-12
        bptt.optimizer.param_groups[0]["lr"] = .05
        actor_before_rollback = state_copy(agent.policy)
        optimizer_before_rollback = deepcopy(bptt.optimizer.state_dict())
        rollback = bptt.update(start, context, dt=dt)
        report["kl_rollback"] = {
            "metrics": rollback, "actor_restored": unchanged(actor_before_rollback, agent.policy),
            "adam_state_restored": tree_equal(optimizer_before_rollback, bptt.optimizer.state_dict()),
            "all_frozen_modules_unchanged": all(unchanged(s, m) for s, m in zip(snapshots, frozen)),
            "log_std_unchanged": torch.equal(original_log_std, log_std),
            "log_std_trainable": bool(log_std.requires_grad),
            "log_std_gradient_none": log_std.grad is None,
            "legacy_std_adam_state_unchanged": tree_equal(old_std_optimizer, bptt.optimizer.state[log_std]),
        }
        assert rollback.get("rollback") == 1. and all(
            value for key, value in report["kl_rollback"].items() if key != "metrics"), report

        # Reach the real episode limit through physics, without modifying time
        # counters or shortening the configured task time limit.
        full_task = env.sgs_bank.family_names.index("full_task")
        full_id = (families == full_task).nonzero().flatten()[0]
        env.sgs_sampler.sample = lambda n: full_id.expand(n)
        env.reset_buf.fill_(1)
        env.reset()
        timeout_records, failure_count = [], 0
        steps = int(env.max_episode_length) + 10
        for step in range(steps):
            previous, conditions = adapter.capture()
            action = torch.zeros(env.num_envs, 9, device=device)
            action[:, 6] = 1.
            result = env.step(action)
            current, _ = adapter.capture()
            actual_timeout = env.timeout_buf.bool().clone()
            failure_count += int(env._terminate_buf.bool().sum())
            if actual_timeout.any():
                prediction = adapter.task_outcome(previous, current, env.actions, conditions)
                reward = adapter.reward(previous, current, env.actions, conditions)
                timeout_records.append({
                    "step": step + 1, "count": int(actual_timeout.sum()),
                    "true_progress": env.progress_buf[actual_timeout].tolist(),
                    "timeout_mask_equal": torch.equal(prediction["timeout"], actual_timeout),
                    "reward_max_error": float((reward[actual_timeout] - result[1].reshape(-1)[actual_timeout]).abs().max()),
                    "obs_max_error": float((adapter.observation(current, conditions)[actual_timeout] - env.obs_buf[actual_timeout]).abs().max()),
                    "success_on_timeout": int(env.pick_success[actual_timeout].sum()),
                    "failure_on_timeout": int(env._terminate_buf[actual_timeout].sum()),
                })
            env.reset()
        env.sgs_sampler.sample = original_sample
        report["real_timeout"] = {
            "steps": steps, "max_episode_length": env.max_episode_length, "counter_modified": False,
            "timeout_count": sum(item["count"] for item in timeout_records),
            "other_failures": failure_count, "records": timeout_records,
        }
        assert timeout_records and all(item["timeout_mask_equal"] and item["reward_max_error"] < 1e-4
                                       and item["obs_max_error"] < 1e-3 for item in timeout_records), report
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        report["resources"] = {
            "max_cuda_memory_allocated_bytes": peak, "max_cuda_memory_allocated_mib": peak / 2**20,
            "memory_scope": "PyTorch allocator only; excludes Isaac Gym/PhysX external allocations",
        }
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("WORLD_MODEL_GPU_SMOKE", json.dumps(report, allow_nan=False))
        print("Diagnostic report:", output)
        return 0
    finally:
        # Clean up on assertions/import failures as well as a successful run.
        try:
            if env is not None and original_sample is not None:
                env.sgs_sampler.sample = original_sample
            if agent is not None and getattr(agent, "writer", None) is not None:
                agent.writer.close()
        finally:
            try:
                if session is not None:
                    session.close()
                elif env is not None:
                    if getattr(env, "viewer", None) is not None:
                        env.gym.destroy_viewer(env.viewer)
                    env.gym.destroy_sim(env.sim)
            finally:
                sys.argv = original_argv
                sys.path[:] = original_path
                os.chdir(original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
