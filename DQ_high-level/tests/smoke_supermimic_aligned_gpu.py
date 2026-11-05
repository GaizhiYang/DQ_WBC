"""Real Isaac Gym entry + full-width SuperMimic recipe on a disposable CUDA scene.

Uses fewer environments/batches/update rounds for a bounded mechanism check.
Optional --full-batch runs one 2048-window fit and one 4096-start Actor/Critic
update with the production chunks. This is not a convergence benchmark.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--full-batch", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "doc/SuperMimic_aligned_gpu_validation.json")
    args = parser.parse_args()
    from smoke_fixed8_world_model_gpu import _runtime_environment
    _runtime_environment()
    from isaacgym import gymapi  # noqa: F401; load before torch
    import torch
    import yaml

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = args.output.resolve()
    report = dict(passed=False, device=str(device), training_success_claimed=False,
                  num_envs=12, high_steps=48, full_batch_requested=args.full_batch)
    env = agent = None
    previous_cwd, previous_argv = Path.cwd(), sys.argv
    started = time.monotonic()

    def snapshot(module):
        return {k: v.detach().clone() for k, v in module.state_dict().items()}

    def same(before, module):
        return all(torch.equal(before[k], v) for k, v in module.state_dict().items())

    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        import train_multistate_DQ_teacher as teacher
        from utils.world_model.policy_inputs import FixedScale
        with tempfile.TemporaryDirectory(prefix="dq_aligned_gpu_") as temporary:
            settings = dict(replay_envs=12, replay_capacity=16, low_replay_capacity=128,
                            batch_size=16, batches_per_update=1, model_microbatch_size=8,
                            imagination_batch=12, imagination_chunk_size=6,
                            updates_per_low_step=.01, validation_batch_size=16)
            config_path = Path(temporary) / "wm.yaml"
            config_path.write_text(yaml.safe_dump(settings))
            sys.argv = ["aligned_gpu_smoke", "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                        "--roboinfo", "--observe_gait_commands", "--small_value_set_zero", "--no_velocity_obs",
                        "--sgs_mode", "sgs", "--sgs_bank", str(ROOT / "data/sgs/sugar_box_v10.pt"),
                        "--future_pose_obs", "--future_pose_horizons", ".60", ".90", "--headless",
                        "--num_envs", "12", "--rl_device", str(device), "--sim_device", str(device),
                        "--world_model", "hybrid", "--wm_recipe", "supermimic", "--wm_config", str(config_path),
                        "--experiment_dir", temporary, "--wandb_name", "aligned", "--seed", "43", "--timesteps", "48"]
            trainer = teacher.get_trainer(is_eval=False, enable_logging=False)
            sys.argv = previous_argv
            wrapper, agent = trainer.env, trainer.agents
            env, adapter = wrapper._env, agent.wm_adapter
            assert isinstance(agent._state_preprocessor, FixedScale)
            assert agent.wm_candidate is agent.wm_validated
            assert agent.wm_config["hidden"] == [2048] * 3
            assert agent.wm_imagination.imagined_critic is not None
            assert agent.wm_low_replay.storage_state_dim == 82
            assert env.control_freq_low_init == 8 and not env.rand_control
            report["resolved_config"] = agent.wm_config
            report["dimensions"] = dict(state=adapter.state_dim, physical=adapter.physical_dim,
                                        residual=adapter.delta_dim, low_features=adapter.low_feature_dim,
                                        low_action_features=adapter.low_action_feature_dim,
                                        context=adapter.context_feature_dim, critic=adapter.critic_feature_dim,
                                        compact_low_replay=agent.wm_low_replay.storage_state_dim)
            low_before = snapshot(adapter.low_actor)
            model_before = snapshot(agent.wm_candidate)
            real_value_before = snapshot(agent.value)
            initial_std = agent.policy.log_std_parameter.detach().clone()
            checks = []
            update = agent.wm_imagination.update

            def checked_update(*positional, **keywords):
                actor = snapshot(agent.policy)
                real_value = snapshot(agent.value)
                model = snapshot(agent.wm_candidate)
                imagined = snapshot(agent.wm_imagination.imagined_critic)
                std = agent.policy.log_std_parameter.detach().clone()
                result = update(*positional, **keywords)
                checks.append(dict(metrics=result, actor_changed=not same(actor, agent.policy),
                                   imagined_critic_changed=not same(imagined, agent.wm_imagination.imagined_critic),
                                   real_critic_unchanged=same(real_value, agent.value),
                                   model_unchanged=same(model, agent.wm_candidate),
                                   std_unchanged=torch.equal(std, agent.policy.log_std_parameter),
                                   std_grad_none=agent.policy.log_std_parameter.grad is None))
                return result

            agent.wm_imagination.update = checked_update
            trainer.train()
            report["entry"] = dict(high_steps=agent.wm_replay.steps, low_steps=agent.wm_low_replay.steps,
                                   update_rounds=agent.wm_update_rounds, model_updates=agent.wm_trainer.updates,
                                   status=agent.wm_status, probe_required=False,
                                   model_changed=not same(model_before, agent.wm_candidate),
                                   real_critic_changed_by_ppo=not same(real_value_before, agent.value),
                                   std_changed_by_ppo=not torch.equal(initial_std, agent.policy.log_std_parameter),
                                   low_policy_unchanged=same(low_before, adapter.low_actor))
            assert report["entry"]["high_steps"] == 48 and report["entry"]["low_steps"] == 384
            assert report["entry"]["update_rounds"] == 3
            assert all(report["entry"][k] for k in ("model_changed", "real_critic_changed_by_ppo",
                                                    "std_changed_by_ppo", "low_policy_unchanged"))
            current, context = adapter.capture()
            clean = adapter.observation(current, context)
            error = float((clean - env.obs_buf).abs().max())
            noisy = adapter.policy_observation(current, context)
            report["observations"] = dict(clean_reconstruction_max_error=error,
                                          noise_applied=not torch.equal(clean, noisy))
            assert error < 2e-3 and report["observations"]["noise_applied"]
            if args.full_batch:
                full_started = time.monotonic()
                agent.wm_trainer.cfg.batch_size = 2048
                agent.wm_trainer.cfg.microbatch_size = 256
                fit = agent.wm_trainer.update(agent.wm_low_replay)
                starts = agent.wm_replay.sample_starts(4096, split="train", balance_families=False, recent_fraction=0.)
                agent.wm_imagination.cfg.chunk_size = 128
                bptt = agent.wm_imagination.update(starts.state, starts.context, dt=.16)
                report["full_batch"] = dict(fit=fit, bptt=bptt, seconds=time.monotonic() - full_started,
                                            note="One model minibatch; production uses ten per round.")
                assert fit["updates"] > 0 and bptt["accepted"] == 1
            report["auxiliary_update_checks"] = checks
            assert len(checks) >= 3 and all(c["metrics"]["accepted"] == 1 for c in checks), checks
            assert all(all(c[k] for k in ("actor_changed", "imagined_critic_changed", "real_critic_unchanged",
                                         "model_unchanged", "std_unchanged", "std_grad_none")) for c in checks), checks
            modules = dict(actor=agent.policy, real_critic=agent.value, imagined_critic=agent.wm_imagination.imagined_critic,
                           dynamics=agent.wm_candidate, low_policy=adapter.low_actor, fixed_scaler=agent._state_preprocessor,
                           high_noise=env.world_model_observation_noise, fk=adapter.kinematics)
            report["cuda_residency"] = {k: all(v.device == device for v in module.state_dict().values())
                                       for k, module in modules.items()}
            report["cuda_residency"].update(high_replay=agent.wm_replay.states.device == device,
                                             low_replay=agent.wm_low_replay.states.device == device)
            assert all(report["cuda_residency"].values())
            torch.cuda.synchronize(device)
            report["peak_torch_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            report["passed"] = True
            print("SUPERMIMIC_ALIGNED_GPU_PASSED", flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf8")
        if env is not None:
            env.gym.destroy_sim(env.sim)
        if agent is not None and getattr(agent, "writer", None) is not None:
            agent.writer.close()
        os.chdir(previous_cwd)
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
