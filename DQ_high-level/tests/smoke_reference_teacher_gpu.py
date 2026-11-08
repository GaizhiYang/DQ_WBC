"""Real Isaac Gym + CUDA pure-BPTT integration, not a convergence benchmark.

Keeps fixed8 and the full 16-high/128-low imagination horizon. Small default
networks/batches and two WM-only iterations make all training phases testable.
--full-networks retains production network widths, with small diagnostic batches.
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--high-steps", type=int, default=74)
    parser.add_argument("--full-networks", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT.parent / "doc/SuperMimic_reference_gpu_validation.json")
    parser.add_argument("--candidate-output", type=Path)
    args = parser.parse_args()
    if args.high_steps < 74:
        parser.error("At least 74 high steps exercise warmup, WM-only and two Actor updates")
    from smoke_fixed8_world_model_gpu import _runtime_environment
    _runtime_environment()
    from isaacgym import gymapi  # noqa: F401; before torch
    import torch
    import yaml

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, training_success_claimed=False, device=str(device),
                  num_envs=args.num_envs, full_networks=args.full_networks,
                  real_high_steps=0, blocks=[], phase="constructing")
    started = time.monotonic()
    env = agent = None
    original_cwd, original_argv = Path.cwd(), sys.argv

    def save():
        report["seconds"] = time.monotonic()-started
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        os.replace(temporary, output)

    def snapshot(module, parameters_only=False):
        items = module.named_parameters() if parameters_only else module.state_dict().items()
        return {name: value.detach().clone() for name, value in items}

    def same(saved, module, parameters_only=False):
        items = module.named_parameters() if parameters_only else module.state_dict().items()
        return all(torch.equal(saved[name], value) for name, value in items)

    def check(module):
        return all(value.device == device and bool(torch.isfinite(value).all())
                   for value in module.state_dict().values())

    save()
    try:
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        import train_multistate_DQ_teacher as teacher
        from utils.wm_entry import save_world_model_artifacts
        with tempfile.TemporaryDirectory(prefix="dq_reference_gpu_") as temporary:
            overrides = dict(world_model_only_iterations=2, wm_max_batches=2,
                replay_envs=args.num_envs, replay_capacity=96, low_replay_capacity=256,
                batch_size=16, model_microbatch_size=8, imagination_batch=8,
                imagination_chunk_size=4, diagnostics_interval=2,
                diagnostics_batch_size=8, validation_batch_size=8)
            if not args.full_networks:
                overrides.update(hidden=[64, 64], actor_hidden=[64, 32], critic_hidden=[32, 16])
            config = Path(temporary) / "smoke.yaml"
            config.write_text(yaml.safe_dump(overrides))
            report["diagnostic_overrides"] = overrides
            sys.argv = ["reference_gpu_check", "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                "--roboinfo", "--observe_gait_commands", "--no_velocity_obs", "--sgs_mode", "sgs",
                "--sgs_bank", str(ROOT / "data/sgs/sugar_box_v10.pt"), "--future_pose_obs",
                "--future_pose_horizons", ".60", ".90", "--headless", "--num_envs", str(args.num_envs),
                "--rl_device", str(device), "--sim_device", str(device), "--world_model", "hybrid",
                "--wm_recipe", "reference", "--wm_config", str(config), "--experiment_dir", temporary,
                "--wandb_name", "reference_smoke", "--seed", "43", "--timesteps", "40000",
                "--stop_after", str(args.high_steps)]
            trainer = teacher.get_trainer(is_eval=False, enable_logging=False)
            agent, wrapper = trainer.agents, trainer.env
            env, adapter = wrapper._env, agent.wm_adapter
            assert agent.memory is None and agent.value is None and not hasattr(agent, "optimizer")
            assert set(agent.models) == {"policy"}
            assert agent.wm_imagination.cfg.horizon == 16
            assert env.control_freq_low_init == 8 and not env.rand_control
            assert not any(parameter.requires_grad for parameter in adapter.low_actor.parameters())
            initial_actor = snapshot(agent.policy, True)
            low_saved = snapshot(adapter.low_actor)
            std_saved = agent.policy.exploration_std.clone()
            report["resolved_config"] = agent.wm_config
            report["metadata"] = agent.wm_metadata
            report["imagined_low_steps"] = 128
            original_update, original_imagine = agent._update_reference, agent.wm_imagination.update
            original_step = wrapper.step

            def checked_step(actions):
                assert actions.device == device and bool(torch.isfinite(actions).all())
                result = original_step(actions)
                rows = result[-1]["world_model"]
                assert len(rows["low_steps"]) == 8
                assert torch.allclose(rows["dt"], torch.full_like(rows["dt"], .16))
                for row in rows["low_steps"]:
                    assert torch.allclose(row["dt"], torch.full_like(row["dt"], .02))
                    for name in ("state", "actions", "next_state", "context"):
                        assert row[name].device == device and bool(torch.isfinite(row[name]).all())
                report["real_high_steps"] += 1
                return result

            def checked_imagine(*positional, **keywords):
                report["phase"] = "128_low_step_bptt"
                save()
                model_saved = snapshot(agent.wm_candidate)
                norm_saved = snapshot(agent.policy.obs_norm)
                actor_saved = snapshot(agent.policy, True)
                result = original_imagine(*positional, **keywords)
                torch.cuda.synchronize(device)
                assert result["accepted"] == 1, result
                assert same(model_saved, agent.wm_candidate), "BPTT updated frozen dynamics/statistics"
                assert same(norm_saved, agent.policy.obs_norm), "Imagined rows updated Actor normalization"
                assert same(low_saved, adapter.low_actor), "BPTT updated low controller"
                assert not same(actor_saved, agent.policy, True), "Accepted Actor update changed no weights"
                assert torch.equal(std_saved, agent.policy.exploration_std)
                assert agent.policy.log_std_parameter.grad is None
                assert result.get("critic_updates", 0) > 0, result
                report["last_bptt_metrics"] = result
                return result

            def checked_update():
                iteration = agent.reference_schedule.iteration
                result = original_update()
                if iteration < 2:
                    assert same(initial_actor, agent.policy, True), "WM-only preparation updated Actor weights"
                    assert not agent.wm_imagination.optimizer.state
                for module in (agent.policy, agent.wm_candidate, agent.wm_imagination.imagined_critic,
                               adapter.low_actor, adapter.kinematics):
                    assert check(module), "Parameters/buffers not finite on requested CUDA device"
                assert same(low_saved, adapter.low_actor)
                report["blocks"].append(dict(iteration=iteration, high_step=agent._wm_total_high_steps,
                                              model_updates=agent.wm_trainer.updates, metrics=result))
                report["phase"] = "collecting"
                save()
                return result

            wrapper.step, agent._update_reference = checked_step, checked_update
            agent.wm_imagination.update = checked_imagine
            report["phase"] = "collecting"
            trainer.train()
            torch.cuda.synchronize(device)
            assert [block["high_step"] for block in report["blocks"][:4]] == [66, 69, 71, 74]
            assert agent.wm_replay.steps == args.high_steps == report["real_high_steps"]
            assert agent.wm_low_replay.steps == 8*args.high_steps
            assert agent.wm_ppo_updates == 0
            assert int(agent.policy.obs_norm.count) == args.high_steps * args.num_envs
            report["normalization_real_next_count"] = int(agent.policy.obs_norm.count)
            report["pure_bptt_no_ppo"] = True
            report["frozen_model_low_controller_and_exploration_checked"] = True
            save_world_model_artifacts(trainer)
            checkpoint = Path(temporary) / "reference_teacher.pt"
            agent.save(str(checkpoint))
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            assert "value" not in payload and "optimizer" not in payload
            from utils.world_model.action_checkpoint import validate_checkpoint_action_contract
            validate_checkpoint_action_contract(env.cfg, payload=payload)
            # Reconstruction of only the inference policy, including empirical moments.
            from utils.world_model.reference_policy import ReferenceTeacherPolicy
            from utils.wm_reference import ReferenceTeacherAgent
            settings = agent.wm_config
            policy = ReferenceTeacherPolicy(wrapper.observation_space, wrapper.action_space, device,
                hidden=settings["actor_hidden"], amp=settings["policy_amp"], action_noise=settings["action_noise"],
                last_gain=settings["actor_last_gain"],
                running_normalization=settings["actor_running_normalization"])
            policy.load_state_dict(payload["policy"])
            state, context = adapter.capture(agent.wm_env_ids)
            observations = adapter.observation(state, context)
            torch.testing.assert_close(policy.forward(observations), agent.policy.forward(observations))
            evaluator = ReferenceTeacherAgent(models={"policy": policy}, memory=None,
                observation_space=wrapper.observation_space, action_space=wrapper.action_space,
                device=device, cfg={"experiment": {"write_interval": 0, "checkpoint_interval": 0}},
                wm_env=env, wm_config={**settings, "mode": "off"})
            assert not hasattr(evaluator, "wm_candidate")
            evaluator.set_running_mode("eval")
            torch.testing.assert_close(evaluator.act(observations)[0], policy.mean_to_action(policy.forward(observations)))
            report["inference_checkpoint_roundtrip"] = True
            if args.candidate_output:
                agent.save_candidate(args.candidate_output.resolve())
                report["candidate_output"] = str(args.candidate_output.resolve())
            report["peak_torch_memory_mib"] = torch.cuda.max_memory_allocated(device)/2**20
            report["phase"] = "teardown"
            save()
    except BaseException:
        report["error"] = traceback.format_exc()
        report["failed_phase"] = report["phase"]
        report["phase"] = "failed"
        save()
        raise
    finally:
        if env is not None:
            env.gym.destroy_sim(env.sim)
        if agent is not None and getattr(agent, "writer", None) is not None:
            agent.writer.close()
        os.chdir(original_cwd)
        sys.argv = original_argv
    report.update(passed=True, phase="completed")
    save()
    print("REFERENCE_TEACHER_GPU_PASSED " + str(output), flush=True)


if __name__ == "__main__":
    main()
