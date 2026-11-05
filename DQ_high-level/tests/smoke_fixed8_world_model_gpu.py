"""Real GPU integration diagnostic for the fixed-eight imagined teacher loop.

Uses a disposable Isaac Gym scene and teacher. Checks live observation/FK
parity, the eight 20 ms records per high decision, supervised model updates,
and a 4-high-decision / 32-model-step BPTT graph on CUDA. This report is not an
action-effect approval report and cannot activate the production BPTT gate.

Run from any directory with the dqwbc Python::

    python DQ_high-level/tests/smoke_fixed8_world_model_gpu.py --device cuda:1
"""

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]


def _runtime_environment():
    library, binary = str(Path(sys.prefix) / "lib"), str(Path(sys.executable).parent)
    environment = dict(os.environ)
    if binary not in environment.get("PATH", "").split(os.pathsep):
        environment["PATH"] = binary + os.pathsep + environment.get("PATH", "")
    if library not in environment.get("LD_LIBRARY_PATH", "").split(os.pathsep):
        environment["LD_LIBRARY_PATH"] = library + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    if environment != dict(os.environ):
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--num_envs", type=int, default=12)
    parser.add_argument("--high_steps", type=int, default=12)
    parser.add_argument("--fit_updates", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "doc" / "SuperMimic_fixed8_gpu_validation.json")
    parser.add_argument("--sgs_bank", type=Path, default=ROOT / "data/sgs/sugar_box_v10.pt")
    args = parser.parse_args()
    if min(args.num_envs, args.high_steps, args.fit_updates) < 1:
        parser.error("Diagnostic dimensions must be positive")
    _runtime_environment()
    # Isaac's extension must be loaded before torch.
    from isaacgym import gymapi  # noqa: F401
    import torch

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("This diagnostic requires real CUDA hardware")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"diagnostic_only": True, "production_gate_activated": False,
              "model_prediction_quality_certified": False, "training_success_claimed": False,
              "device": str(device), "num_envs": args.num_envs, "passed": False}
    original_cwd, original_args, original_path = Path.cwd(), sys.argv, list(sys.path)
    trainer = env = agent = original_low_observation = None

    def snapshot(module):
        return {key: value.detach().clone() for key, value in module.state_dict().items()}

    def unchanged(before, module):
        return all(torch.equal(before[key], value) for key, value in module.state_dict().items())

    def on_device(values):
        values = list(values)
        return bool(values) and all(value.device == device for value in values)

    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        import train_multistate_DQ_teacher as teacher
        from utils.world_model.gradients import freeze_parameters
        from utils.world_model.imagination import ImaginationConfig, ImaginationTrainer

        sys.argv = ["fixed8_gpu_diagnostic", "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                    "--roboinfo", "--observe_gait_commands", "--small_value_set_zero", "--no_velocity_obs",
                    "--sgs_mode", "sgs", "--sgs_bank", str(args.sgs_bank.resolve()), "--future_pose_obs",
                    "--future_pose_horizons", ".60", ".90", "--headless", "--num_envs", str(args.num_envs),
                    "--rl_device", str(device), "--sim_device", str(device), "--world_model", "fit",
                    "--wm_probe_interval", "0", "--wm_horizon", "4", "--wm_replay_envs", str(args.num_envs),
                    "--wm_batch_size", "32", "--experiment_dir", "/tmp/dq_fixed8_gpu_diagnostic",
                    "--wandb_name", "isolated_fixed8", "--seed", "43", "--timesteps", "120000"]
        trainer = teacher.get_trainer(is_eval=False, enable_logging=False)
        sys.argv = original_args
        wrapper, agent = trainer.env, trainer.agents
        env, adapter = wrapper._env, agent.wm_adapter
        model = agent.wm_candidate
        assert env.control_freq_low_init == 8 and not env.rand_control
        assert adapter.physical_dim == 64 and adapter.delta_dim == 62
        assert agent.wm_low_replay is not None and adapter.low_shape == (10, 71)
        report["contract"] = {"state_dim_including_memory": adapter.state_dim,
                              "predicted_physical_dim": adapter.physical_dim, "residual_dim": adapter.delta_dim,
                              "dynamics_context_feature_dim": adapter.context_feature_dim,
                              "low_dt_seconds": adapter.low_dt, "high_dt_seconds": adapter.high_dt,
                              "low_steps_per_high": model.low_steps, "schema": adapter.metadata()["schema"],
                              "root_body_com": adapter.kinematics.base_com.tolist()}
        low_weights = snapshot(adapter.low_actor)
        max_errors = {"high_observation": 0., "low_observation": 0., "low_history": 0.,
                      "gait": 0., "clock": 0., "ee_pos": 0., "ee_quat": 0., "ee_vel": 0., "ee_omega": 0.,
                      "collector_low_observation": 0., "executed_low_action": 0., "high_target_cache": 0.}
        low_calls = 0
        table_diagnostics = {"position_l2_max_m": 0., "linear_velocity_l2_max_m_per_s": 0.,
                             "orientation_quat_max_abs_sign_invariant": 0.}
        original_low_observation = env._compute_low_level_observations

        @torch.no_grad()
        def low_observation_spy():
            nonlocal low_calls
            before, conditions = adapter.capture()
            imagined, advanced = adapter.low_observation(before, conditions)
            original_low_observation()
            after, _ = adapter.capture()
            max_errors["low_observation"] = max(max_errors["low_observation"],
                                                float((imagined - env.low_obs_buf).abs().max()))
            for field in ("low_history", "gait", "clock"):
                max_errors[field] = max(max_errors[field],
                                         float((adapter.get(advanced, field) - adapter.get(after, field)).abs().max()))
            low_calls += 1

        env._compute_low_level_observations = low_observation_spy
        states, _ = wrapper.reset()
        collected = []
        for step in range(args.high_steps):
            with torch.no_grad():
                actions, _, _ = agent.act(states, timestep=step, timesteps=120000)
                next_states, rewards, terminated, truncated, infos = wrapper.step(actions)
                agent.record_transition(states, actions, rewards, next_states, terminated, truncated,
                                        infos, timestep=step, timesteps=120000)
            transition = infos["world_model"]
            low = transition["low_steps"]
            assert len(low) == 8
            assert torch.allclose(transition["dt"], torch.full_like(transition["dt"], .16))
            assert all(torch.allclose(item["dt"], torch.full_like(item["dt"], .02)) for item in low)
            assert all(item["actions"].shape[-1] == 20 for item in low)
            assert all(on_device(value for value in item.values() if torch.is_tensor(value)) for item in low)
            with torch.no_grad():
                prepared = adapter.prepare_high(transition["state"], transition["actions"])
                for field in ("goal_pos", "goal_rpy", "commands", "held_ee_target", "held_gripper_target", "prev_action"):
                    max_errors["high_target_cache"] = max(
                        max_errors["high_target_cache"],
                        float((adapter.get(prepared, field) - adapter.get(low[0]["state"], field)).abs().max()))
                for item in low:
                    imagined_observation, _ = adapter.low_observation(item["state"], item["context"])
                    _, imagined_action = adapter.low_control(item["state"], item["context"])
                    max_errors["collector_low_observation"] = max(
                        max_errors["collector_low_observation"],
                        float((imagined_observation - item["low_observation"]).abs().max()))
                    max_errors["executed_low_action"] = max(
                        max_errors["executed_low_action"], float((imagined_action - item["actions"]).abs().max()))
                # This intentionally compares a known nominal support surface
                # against contact-affected physics. Report its approximation
                # error; there is no learned table head or table target loss.
                nominal_state = low[0]["state"]
                for _ in range(8):
                    nominal_state = adapter.replace(
                        nominal_state, table=adapter.nominal_table_step(nominal_state, low[0]["context"]))
                nominal_table = adapter.get(nominal_state, "table")
                measured_table = adapter.get(low[-1]["next_state"], "table")
                table_diagnostics["position_l2_max_m"] = max(
                    table_diagnostics["position_l2_max_m"],
                    float(torch.linalg.vector_norm(nominal_table[:, :3] - measured_table[:, :3], dim=-1).max()))
                table_diagnostics["linear_velocity_l2_max_m_per_s"] = max(
                    table_diagnostics["linear_velocity_l2_max_m_per_s"],
                    float(torch.linalg.vector_norm(nominal_table[:, 7:10] - measured_table[:, 7:10], dim=-1).max()))
                quaternion_error = torch.minimum((nominal_table[:, 3:7] - measured_table[:, 3:7]).abs().amax(-1),
                                                   (nominal_table[:, 3:7] + measured_table[:, 3:7]).abs().amax(-1))
                table_diagnostics["orientation_quat_max_abs_sign_invariant"] = max(
                    table_diagnostics["orientation_quat_max_abs_sign_invariant"], float(quaternion_error.max()))
            current, conditions = adapter.capture()
            reconstructed = adapter.observation(current, conditions)
            assert reconstructed.shape[-1] == 1284
            max_errors["high_observation"] = max(max_errors["high_observation"],
                                                 float((reconstructed - env.obs_buf).abs().max()))
            predicted_ee = adapter.fk(current)
            measured_ee = current[:, adapter.slices["ee"]]
            for field, sl in (("ee_pos", slice(0, 3)), ("ee_quat", slice(3, 7)),
                              ("ee_vel", slice(7, 10)), ("ee_omega", slice(10, 13))):
                errors = (predicted_ee[field] - measured_ee[:, sl]).abs().amax(-1)
                if field == "ee_quat":
                    errors = torch.minimum(errors, (predicted_ee[field] + measured_ee[:, sl]).abs().amax(-1))
                max_errors[field] = max(max_errors[field], float(errors.max()))
            collected.append({"microsteps": len(low), "high_dt": float(transition["dt"][0]),
                              "micro_dt": float(low[0]["dt"][0])})
            states, _ = wrapper.reset()
        env._compute_low_level_observations = original_low_observation
        report["real_collection"] = {"high_vector_steps": agent.wm_replay.steps,
                                     "low_vector_steps": agent.wm_low_replay.steps,
                                     "observed_low_policy_calls": low_calls, "records": collected}
        report["observation_and_fk_max_abs_errors"] = max_errors
        report["nominal_table_approximation"] = {"diagnostic_only": True, "learned_table_outputs": 0,
                                                "future_real_table_used_as_input": False, **table_diagnostics}
        assert agent.wm_low_replay.steps == 8 * agent.wm_replay.steps
        assert max_errors["high_observation"] < 2e-3, max_errors
        assert max_errors["low_observation"] < 2e-4, max_errors
        assert max(max_errors[name] for name in ("collector_low_observation", "executed_low_action", "high_target_cache")) < 2e-4, max_errors
        assert max(max_errors[name] for name in ("low_history", "gait", "clock")) < 2e-4, max_errors
        assert max(max_errors[name] for name in ("ee_pos", "ee_quat")) < 2e-4, max_errors
        assert max(max_errors[name] for name in ("ee_vel", "ee_omega")) < 2e-3, max_errors

        model_before, actor_before_fit = snapshot(model), snapshot(agent.policy)
        fit_metrics = [agent.wm_trainer.update(agent.wm_low_replay) for _ in range(args.fit_updates)]
        report["supervised_fit"] = {"metrics": fit_metrics, "model_changed": not unchanged(model_before, model),
                                    "actor_unchanged": unchanged(actor_before_fit, agent.policy),
                                    "low_policy_unchanged": unchanged(low_weights, adapter.low_actor)}
        assert report["supervised_fit"]["model_changed"]
        assert report["supervised_fit"]["actor_unchanged"] and report["supervised_fit"]["low_policy_unchanged"]
        assert agent.wm_trainer.updates > 0
        assert all("contact_bce" in metrics for metrics in fit_metrics)

        start, context = adapter.capture()
        observed = adapter.observation(start, context)
        # Freezing the controller must preserve its input Jacobian, otherwise
        # model states cannot affect subsequent leg responses during BPTT.
        differentiable_start = start.detach().requires_grad_(True)
        _, differentiated_action = adapter.low_control(differentiable_start, context)
        low_input_gradient = torch.autograd.grad(differentiated_action[:, :12].square().sum(), differentiable_start)[0]
        low_joint_gradient = low_input_gradient[:, adapter.slices["dof_pos"]]
        assert bool(torch.isfinite(low_joint_gradient).all()) and float(low_joint_gradient.abs().max()) > 0
        agent._state_preprocessor(observed, train=True)
        agent._value_preprocessor(env.rew_buf[:, None], train=True)
        imagination = ImaginationTrainer(
            model, adapter, agent.policy, agent.value,
            ImaginationConfig(horizon=4, dt=.16, lr=1e-5, strength=.05, max_kl=.005),
            agent._state_preprocessor, agent._value_preprocessor)
        frozen = [model, agent.value, agent._state_preprocessor, agent._value_preprocessor]
        frozen_before = [snapshot(module) for module in frozen]
        std_before = agent.policy.log_std_parameter.detach().clone()
        actor_before = snapshot(agent.policy)

        with torch.no_grad():
            normalized = agent._state_preprocessor(observed, train=False)
            sample_a, _, _ = agent.policy.act({"states": normalized}, role="policy")
            sample_b, _, _ = agent.policy.act({"states": normalized}, role="policy")
        assert not torch.equal(sample_a, sample_b)
        calls = []
        original_low_step = model.low_step

        def count_model_step(*arguments, **options):
            calls.append(1)
            return original_low_step(*arguments, **options)

        model.low_step = count_model_step
        with freeze_parameters(*frozen):
            rollout = imagination.imagine(start, context, dt=.16)
        model.low_step = original_low_step
        assert len(calls) == 32 and rollout.actions.shape == (4, args.num_envs, 9)
        assert all(bool(torch.isfinite(value).all()) for value in vars(rollout).values())
        assert all(value.device == device for value in vars(rollout).values())
        del rollout
        physical_action = sample_a.detach().requires_grad_(True)
        with freeze_parameters(model):
            predicted = model.step(start, physical_action, context, .16)
            physical = torch.cat([adapter.get(predicted, name) for name in adapter.learned_names], -1)
            physical_gradient = torch.autograd.grad(physical.square().sum(), physical_action)[0]
        assert bool(torch.isfinite(physical_gradient).all()) and float(physical_gradient[:, 6].abs().max()) > 0
        metrics = imagination.update(start, context, dt=.16)
        gradients = [p.grad for p in agent.policy.parameters() if p.grad is not None]
        report["imagined_bptt"] = {
            "high_decisions": 4, "model_steps": len(calls), "imagined_seconds": .64, "metrics": metrics,
            "actor_changed": not unchanged(actor_before, agent.policy),
            "actor_gradients_finite_on_cuda": bool(gradients) and on_device(gradients)
            and all(bool(torch.isfinite(g).all()) for g in gradients),
            "gripper_physical_gradient_abs_max": float(physical_gradient[:, 6].abs().max()),
            "frozen_low_policy_joint_input_gradient_abs_max": float(low_joint_gradient.abs().max()),
            "low_policy_unchanged": unchanged(low_weights, adapter.low_actor),
            "world_model_critic_scalers_unchanged": all(unchanged(before, module)
                                                       for before, module in zip(frozen_before, frozen)),
            "log_std_unchanged": torch.equal(std_before, agent.policy.log_std_parameter),
            "log_std_gradient_none": agent.policy.log_std_parameter.grad is None,
            "real_policy_still_stochastic": not torch.equal(sample_a, sample_b),
        }
        assert metrics.get("accepted") == 1. and metrics["actor_grad_norm"] > 0, report["imagined_bptt"]
        assert all(report["imagined_bptt"][key] for key in (
            "actor_changed", "actor_gradients_finite_on_cuda", "low_policy_unchanged",
            "world_model_critic_scalers_unchanged", "log_std_unchanged", "log_std_gradient_none",
            "real_policy_still_stochastic")), report["imagined_bptt"]
        report["cuda_residency"] = {"actor": on_device(agent.policy.state_dict().values()),
                                    "critic": on_device(agent.value.state_dict().values()),
                                    "model": on_device(model.state_dict().values()),
                                    "low_policy": on_device(adapter.low_actor.state_dict().values()),
                                    "fk": on_device(adapter.kinematics.state_dict().values()),
                                    "high_replay": on_device([agent.wm_replay.states, agent.wm_replay.actions]),
                                    "low_replay": on_device([agent.wm_low_replay.states, agent.wm_low_replay.actions])}
        assert all(report["cuda_residency"].values())
        assert not agent.wm_has_validated_snapshot
        torch.cuda.synchronize(device)
        report["max_torch_cuda_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        report["passed"] = True
        print("FIXED8_GPU_DIAGNOSTIC_PASSED", json.dumps(report, allow_nan=False))
        return 0
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
        if env is not None:
            if original_low_observation is not None:
                env._compute_low_level_observations = original_low_observation
            if getattr(env, "viewer", None) is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)
        if agent is not None and getattr(agent, "writer", None) is not None:
            agent.writer.close()
        os.chdir(original_cwd)
        sys.argv, sys.path = original_args, original_path


if __name__ == "__main__":
    raise SystemExit(main())
