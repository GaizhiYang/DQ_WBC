"""W/V/B cross-stage numerical integration check of the revised real teacher.

Runs small or 4096 Isaac Gym environments with the manipulation recipe, seed 43,
and the production 40000-step curriculum. --high-steps only limits this run;
it does not shorten that curriculum or shrink model/replay/training batches.
The default 192 steps exercises W x4, V x2, B1 x2. --production keeps all recipe budgets; small mode explicitly limits model sizes and batches.

The JSON report is replaced atomically before physics, after each real step,
and during updates, so a native CUDA/PhysX crash retains the last safe boundary.
Numerical checks and reporting add overhead: this is not a speed benchmark or
a convergence certificate. W and V phases must perform no Actor optimizer steps;
B phases must accept their limited Actor budget, with no deferred update debt.

Run with the dqwbc interpreter (loads Isaac Gym before torch)::

    python DQ_high-level/tests/smoke_manipulation_teacher_gpu.py --device cuda:0
"""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def _json_safe(value):
    """Keep diagnostic NaN/Inf visible without producing invalid JSON."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError("Unexpected report value: " + type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--high-steps", type=int, default=192)
    parser.add_argument("--output", type=Path,
                        default=ROOT.parent / "doc/SuperMimic_manipulation_small_gpu.json")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--experimental", action="store_true", help="Exercise LN + RunningNorm + memory Critic + neural BF16")
    parser.add_argument("--candidate-output", type=Path,
                        help="Optional candidate + matching teacher/V snapshot for isolated diagnostic tests")
    args = parser.parse_args()
    if args.production:
        args.num_envs = 4096
    if args.high_steps < 24 or args.high_steps > 40000 or args.high_steps % 24:
        parser.error("--high-steps must be a positive multiple of 24 between 24 and 40000")
    from smoke_fixed8_world_model_gpu import _runtime_environment
    _runtime_environment()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(passed=False, completed=False, training_success_claimed=False,
                  device=args.device, num_envs=args.num_envs, production=args.production, experimental=args.experimental, seed=43, planned_timesteps=40000,
                  requested_high_steps=args.high_steps, completed_high_steps=0,
                  completed_low_steps=0, blocks=[], first_bptt_ownership=None,
                  input_checks=0, input_failures=[], phase="initializing")
    original_cwd, original_args, original_path = Path.cwd(), sys.argv, list(sys.path)
    env = agent = trainer = None

    def write_report():
        report["seconds"] = time.monotonic() - started
        temporary = output.with_name(output.name + ".tmp")
        with temporary.open("w", encoding="utf8") as stream:
            json.dump(_json_safe(report), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)

    write_report()
    try:
        from isaacgym import gymapi  # noqa: F401; must precede torch
        import torch

        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("This check requires real CUDA hardware")
        torch.cuda.set_device(device)
        device = torch.device("cuda", torch.cuda.current_device())
        report["device"] = str(device)
        torch.cuda.reset_peak_memory_stats(device)
        os.chdir(ROOT)
        sys.path.insert(0, str(ROOT))
        import train_multistate_DQ_teacher as teacher
        from utils.world_model.recipe import MANIPULATION_CONFIG

        def tensor_entries(value, prefix=""):
            if torch.is_tensor(value):
                yield prefix, value
            elif isinstance(value, dict):
                for key, child in value.items():
                    yield from tensor_entries(child, prefix + "/" + str(key))
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    yield from tensor_entries(child, prefix + "/" + str(index))

        def tensor_status(value, *, check_finite=True, optimizer=False):
            entries = list(tensor_entries(value))
            wrong = [name for name, tensor in entries if tensor.device != device and not
                     (optimizer and name.endswith("/step") and tensor.device.type == "cpu"
                      and tensor.numel() == 1)]
            bad = []
            if check_finite:
                groups = {}
                for name, tensor in entries:
                    groups.setdefault(tensor.device, []).append((name, torch.isfinite(tensor).all()))
                for items in groups.values():
                    flags = torch.stack([flag for _, flag in items]).detach().cpu().tolist()
                    bad.extend(name for (name, _), good in zip(items, flags) if not good)
            return dict(finite=not bad if check_finite else None, nonfinite_tensors=bad,
                        cuda_resident=bool(entries) and not wrong, wrong_device_tensors=wrong,
                        devices=dict(Counter(str(tensor.device) for _, tensor in entries)),
                        tensor_count=len(entries), cpu_step_counters_allowed=optimizer)

        def check_inputs(label, value):
            status = tensor_status(value)
            report["input_checks"] += 1
            if not status["finite"] or not status["cuda_resident"]:
                report["input_failures"].append(dict(label=label, **status))
                write_report()
                raise FloatingPointError("Invalid real/training input: " + label)

        def snapshot(module):
            return {name: value.detach().clone() for name, value in module.state_dict().items()}

        def unchanged(saved, module):
            current = module.state_dict()
            return (saved.keys() == current.keys() and bool(torch.stack([
                (saved[name] == value).all() for name, value in current.items()]).all()))

        with tempfile.TemporaryDirectory(prefix="dq_stability_gpu_") as temporary:
            sys.argv = ["world_model_stability_gpu", "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                        "--roboinfo", "--observe_gait_commands", "--small_value_set_zero", "--no_velocity_obs",
                        "--sgs_mode", "sgs", "--sgs_bank", str(ROOT / "data/sgs/sugar_box_v10.pt"),
                        "--future_pose_obs", "--future_pose_horizons", ".60", ".90", "--headless",
                        "--num_envs", str(args.num_envs), "--rl_device", str(device), "--sim_device", str(device),
                        "--world_model", "hybrid", "--wm_recipe", "manipulation",
                        "--experiment_dir", temporary, "--wandb_name", "stability", "--seed", "43",
                        "--timesteps", "40000", "--stop_after", str(args.high_steps)]
            import yaml
            overrides = {} if args.production else dict(
                hidden=[128, 128], batch_size=32, model_microbatch_size=16,
                wm_max_batches=4, imagination_batch=64, imagination_chunk_size=32,
                diagnostics_batch_size=16, validation_batch_size=16,
                replay_envs=args.num_envs, critic_hidden=[64, 32])
            if args.experimental:
                overrides.update(layer_norm=True, running_normalization=True,
                                 critic_memory=True, amp="bf16")
            config_path = Path(temporary) / "smoke.yaml"
            config_path.write_text(yaml.safe_dump(overrides))
            sys.argv += ["--wm_config", str(config_path)]
            report["smoke_overrides"] = overrides
            report["teacher_arguments"] = sys.argv[1:]
            write_report()
            trainer = teacher.get_trainer(is_eval=False, enable_logging=False)
            sys.argv = original_args
            wrapper, agent = trainer.env, trainer.agents
            env, adapter = wrapper._env, agent.wm_adapter
            assert env.num_envs == args.num_envs and agent._rollouts == 24
            assert env.total_timesteps == 40000 and trainer.timesteps == args.high_steps
            assert env.control_freq_low_init == 8 and not env.rand_control
            assert env.clip_actions == adapter.clip_actions
            assert math.isinf(env.clip_actions), "Source-aligned default must not add a raw-action hard clamp"
            assert agent.wm_candidate is agent.wm_validated
            assert agent.wm_config["mode"] == "hybrid" and agent.wm_config["fixed8"]
            assert agent.wm_imagination.imagined_critic is not None
            assert not env.cfg["env"]["smallValueSetZero"] and not adapter.small_commands
            assert agent.wm_config["task_contract"] == "manipulation_v2"
            for name, expected in MANIPULATION_CONFIG.items():
                if name not in overrides and name != "task_aux_config":
                    assert agent.wm_config[name] == expected, (name, agent.wm_config[name], expected)
            report["resolved_config"] = dict(agent.wm_config)
            report["contract"] = dict(low_steps_per_high=8, low_dt=.02, high_dt=.16,
                                      action_clip=env.clip_actions,
                                      action_coordinates=agent.wm_config["action_coordinates"],
                                      action_rate_mode=agent.wm_config["action_rate_mode"],
                                      state_loss_reduction=agent.wm_trainer.cfg.state_loss_reduction,
                                      auxiliary_weight=agent.wm_trainer.cfg.auxiliary_weight,
                                      critic_feature_dim=adapter.critic_feature_dim,
                                      critic_features=adapter.metadata()["critic_features"],
                                      imagined_high_steps=agent.wm_imagination.cfg.horizon,
                                      imagined_low_steps=agent.wm_imagination.cfg.horizon * 8,
                                      compact_low_state=agent.wm_low_replay.storage_state_dim)
            assert report["contract"]["imagined_low_steps"] == 32
            low_before = snapshot(adapter.low_actor)
            initial_std = agent.policy.log_std_parameter.detach().clone()
            torch.testing.assert_close(initial_std.exp(), torch.full_like(initial_std, .1))
            modules = dict(actor=agent.policy, real_critic=agent.value, dynamics=agent.wm_candidate,
                           imagined_critic=agent.wm_imagination.imagined_critic, low_policy=adapter.low_actor,
                           fixed_scaler=agent._state_preprocessor, fk=adapter.kinematics,
                           observation_noise=env.world_model_observation_noise)

            def training_status():
                result = {name: tensor_status(module.state_dict()) for name, module in modules.items()}
                optimizers = dict(ppo=agent.optimizer, dynamics=agent.wm_trainer.optimizer,
                                  bptt=agent.wm_imagination.optimizer,
                                  imagined_critic=agent.wm_imagination.imagined_critic_trainer.optimizer)
                for name, optimizer in optimizers.items():
                    # Adam's scalar host step counters are intentional; moments
                    # and parameters must remain on the requested CUDA device.
                    state = optimizer.state_dict()
                    if state["state"]:
                        result[name + "_optimizer"] = tensor_status(state, optimizer=True)
                # Unwritten replay slots are torch.empty, so only inspect their
                # residency. Every actual transition is checked before insertion.
                for name, replay in (("high_replay", agent.wm_replay), ("low_replay", agent.wm_low_replay)):
                    result[name] = tensor_status(vars(replay), check_finite=False)
                result["ppo_memory"] = tensor_status(agent.memory.tensors, check_finite=False)
                return result

            def require_healthy(status):
                bad = [name for name, item in status.items()
                       if item["finite"] is False or not item["cuda_resident"]]
                if bad:
                    raise FloatingPointError("Nonfinite parameters/state or wrong device: " + ", ".join(bad))

            @torch.no_grad()
            def actor_diagnostics():
                # Do not call act(): diagnostics must not consume exploration
                # RNG or add another observation-noise sample to this rollout.
                was_training = agent.policy.training
                agent.policy.eval()
                try:
                    observations = agent._current_next_states.float()
                    normalized = agent._state_preprocessor(observations, train=False)
                    mean, log_std, _ = agent.policy.compute({"states": normalized}, role="policy")
                    opening = agent.policy.mean_to_action(mean)[..., 6]
                    limits = torch.finfo(opening.dtype)
                    std_rows = log_std.reshape(-1, mean.shape[-1])
                    return dict(observation_source="agent._current_next_states; fixed preprocessing, no new noise",
                                mean_min=mean.amin(0).tolist(), mean_max=mean.amax(0).tolist(),
                                command_mean_saturation_fraction=(mean[:, [0, 1, 2, 3, 4, 5, 7, 8]].abs() >= 1.).float().mean(0).tolist(),
                                log_std_min=std_rows.amin(0).tolist(), log_std_max=std_rows.amax(0).tolist(),
                                log_std_parameter=agent.policy.log_std_parameter.detach().tolist(),
                                opening_min=float(opening.min()), opening_max=float(opening.max()),
                                opening_at_lower_endpoint_fraction=float((opening <= limits.tiny).float().mean()),
                                opening_at_upper_endpoint_fraction=float((opening >= 1. - limits.eps).float().mean()),
                                finite=bool(torch.isfinite(mean).all() & torch.isfinite(log_std).all()
                                            & torch.isfinite(opening).all()))
                finally:
                    agent.policy.train(was_training)

            report["initial_training_status"] = training_status()
            require_healthy(report["initial_training_status"])
            originals = dict(act=agent.act, step=wrapper.step, update=agent._update,
                             imagination=agent.wm_imagination.update, value_only=agent.wm_imagination.update_value, progress=agent.wm_progress_callback)
            current_block = None
            bptt_calls = 0

            def checked_act(states, *positional, **keywords):
                check_inputs("policy_observation", states)
                result = originals["act"](states, *positional, **keywords)
                check_inputs("sampled_action_and_log_probability", result)
                return result

            def checked_step(actions):
                check_inputs("physics_action", actions)
                report["phase"] = "physics"
                report["pending_high_step"] = report["completed_high_steps"] + 1
                write_report()
                result = originals["step"](actions)
                next_states, rewards, terminated, truncated, infos = result
                check_inputs("real_step_outputs", (next_states, rewards, terminated, truncated))
                transition = infos["world_model"]
                check_inputs("real_world_model_transitions", transition)
                assert bool((transition["actions"].abs() <= env.clip_actions).all())
                assert bool((adapter.get(transition["next_state"], "prev_action").abs()
                             <= env.clip_actions).all()), "Real previous-action feedback escaped its bounds"
                assert len(transition["low_steps"]) == 8
                assert bool(torch.isclose(transition["dt"], transition["dt"].new_tensor(.16)).all())
                assert all(bool(torch.isclose(item["dt"], item["dt"].new_tensor(.02)).all())
                           for item in transition["low_steps"])
                report["completed_high_steps"] += 1
                report["completed_low_steps"] += 8
                report["phase"] = "recording_transition"
                write_report()
                return result

            def checked_imagination(start_state, context, *positional, **keywords):
                nonlocal bptt_calls
                check_inputs("bptt_start", (start_state, context))
                assert bool((adapter.get(start_state, "prev_action").abs() <= env.clip_actions).all())
                first = bptt_calls == 0
                bptt_calls += 1
                std = agent.policy.log_std_parameter.detach().clone()
                # These large snapshots are taken once, not once per round.
                if first:
                    model_before = snapshot(agent.wm_candidate)
                    value_before = snapshot(agent.value)
                result = originals["imagination"](start_state, context, *positional, **keywords)
                std_unchanged = torch.equal(std, agent.policy.log_std_parameter)
                std_grad_none = agent.policy.log_std_parameter.grad is None
                assert std_unchanged and std_grad_none, "BPTT must not update Gaussian standard deviation"
                if first:
                    report["first_bptt_ownership"] = dict(
                        dynamics_unchanged=unchanged(model_before, agent.wm_candidate),
                        real_critic_unchanged=unchanged(value_before, agent.value),
                        low_policy_unchanged=unchanged(low_before, adapter.low_actor),
                        std_unchanged=std_unchanged, std_grad_none=std_grad_none)
                    assert all(report["first_bptt_ownership"].values())
                current_block["bptt_calls"] += 1
                current_block["bptt_accepted"] += int(result.get("accepted", 0))
                current_block["last_bptt_metrics"] = _json_safe(result)
                if not result.get("accepted", 0) or result.get("critic_skipped_nonfinite", 0):
                    current_block["protected_rejections"].append(dict(round=current_block["bptt_calls"],
                                                                       metrics=_json_safe(result)))
                report["bptt_std_ownership_checks"] = bptt_calls
                write_report()
                return result

            def checked_value_only(start_state, context, *positional, **keywords):
                actor_before = snapshot(agent.policy)
                model_before = snapshot(agent.wm_candidate)
                optimizer_before = len(agent.wm_imagination.optimizer.state)
                result = originals["value_only"](start_state, context, *positional, **keywords)
                assert unchanged(actor_before, agent.policy), "Value preparation changed Actor"
                assert unchanged(model_before, agent.wm_candidate), "Value preparation changed WM"
                assert len(agent.wm_imagination.optimizer.state) == optimizer_before
                current_block["value_only_calls"] += 1
                current_block["last_value_only_metrics"] = _json_safe(result)
                return result

            def progress(message):
                report["last_progress"] = message
                report["last_ppo_numerics"] = _json_safe(getattr(agent, "last_ppo_numerics", None))
                if current_block is not None:
                    current_block["last_ppo_numerics"] = report["last_ppo_numerics"]
                    if message.startswith("real PPO complete"):
                        current_block["actor_after_ppo"] = actor_diagnostics()
                write_report()
                if message.startswith("real PPO complete") and not current_block["actor_after_ppo"]["finite"]:
                    raise FloatingPointError("Actor predictions became nonfinite after real PPO")
                if originals["progress"] is not None:
                    originals["progress"](message)

            def checked_update(timestep, timesteps):
                nonlocal current_block
                torch.cuda.synchronize(device)
                block_started = time.monotonic()
                previous_rounds = agent.wm_update_rounds
                current_block = dict(index=len(report["blocks"]) + 1, high_steps=agent.wm_replay.steps,
                                     low_steps=agent.wm_low_replay.steps, completed=False,
                                     bptt_calls=0, bptt_accepted=0, value_only_calls=0, protected_rejections=[],
                                     stage=agent.wm_schedule.phase)
                report["blocks"].append(current_block)
                report["phase"] = "ppo_and_world_model_update"
                current_block["actor_before_ppo"] = actor_diagnostics()
                write_report()
                if not current_block["actor_before_ppo"]["finite"]:
                    raise FloatingPointError("Actor predictions were already nonfinite before PPO")
                originals["update"](timestep, timesteps)
                torch.cuda.synchronize(device)
                current_block["seconds"] = time.monotonic() - block_started
                current_block["last_ppo_numerics"] = _json_safe(getattr(agent, "last_ppo_numerics", None))
                current_block["last_world_model_metrics"] = _json_safe(agent.last_world_model_metrics)
                current_block["training_status"] = training_status()
                current_block["actor_after_bptt"] = actor_diagnostics()
                current_block["low_policy_unchanged"] = unchanged(low_before, adapter.low_actor)
                current_block["low_policy_frozen"] = all(not parameter.requires_grad
                                                          for parameter in adapter.low_actor.parameters())
                current_block["schedule"] = agent.wm_schedule.state_dict()
                current_block["update_rounds"] = agent.wm_update_rounds
                write_report()
                require_healthy(current_block["training_status"])
                assert all(current_block[name]["finite"] for name in
                           ("actor_before_ppo", "actor_after_ppo", "actor_after_bptt"))
                assert current_block["last_ppo_numerics"] is not None, "FinitePPO instrumentation is required"
                assert current_block["last_ppo_numerics"]["accepted_minibatches"] > 0, \
                    "No real PPO minibatch was accepted in this block"
                assert current_block["last_ppo_numerics"]["rejected_rollouts"] == 0, \
                    "A rejected entire real PPO rollout cannot pass this stability check"
                assert agent.wm_replay.steps == (timestep + 1) == report["completed_high_steps"]
                assert agent.wm_low_replay.steps == agent.wm_replay.steps * 8
                phase = current_block["stage"]
                if phase in ("W", "V"):
                    assert current_block["bptt_calls"] == 0, "Actor ran before BPTT phase"
                else:
                    budget = agent.wm_config[{"B1": "actor_stage1_updates", "B2": "actor_stage2_updates", "B3": "actor_max_updates"}[phase]]
                    assert current_block["bptt_calls"] == budget
                    assert current_block["bptt_accepted"] == budget
                assert current_block["value_only_calls"] == (agent.wm_config["value_prepare_batches"] if phase == "V" else 0)
                assert agent.wm_schedule.completed[phase] > 0
                assert 0 <= agent.wm_schedule.credit < 1
                assert current_block["low_policy_unchanged"] and current_block["low_policy_frozen"]
                current_block["completed"] = True
                report["phase"] = "collecting"
                write_report()

            agent.act, wrapper.step, agent._update = checked_act, checked_step, checked_update
            agent.wm_imagination.update, agent.wm_progress_callback = checked_imagination, progress
            agent.wm_imagination.update_value = checked_value_only
            try:
                trainer.train()
            except FloatingPointError:
                # Python finite guards leave CUDA usable. Save the policy and
                # latest real observation for offline diagnosis; never attempt
                # this after a native CUDA/PhysX illegal-address RuntimeError.
                failure_path = output.with_suffix(".failure.pt")
                torch.save(dict(policy=agent.policy.state_dict(), value=agent.value.state_dict(),
                                dynamics=agent.wm_candidate.state_dict(),
                                imagined_critic=agent.wm_imagination.imagined_critic.state_dict(),
                                next_states=agent._current_next_states,
                                ppo_numerics=getattr(agent, "last_ppo_numerics", None)), failure_path)
                report["failure_tensor_snapshot"] = str(failure_path)
                raise
            finally:
                agent.act, wrapper.step, agent._update = originals["act"], originals["step"], originals["update"]
                agent.wm_imagination.update = originals["imagination"]
                agent.wm_imagination.update_value = originals["value_only"]
                agent.wm_progress_callback = originals["progress"]
            torch.cuda.synchronize(device)
            assert report["completed_high_steps"] == agent.wm_replay.steps == args.high_steps
            assert report["completed_low_steps"] == agent.wm_low_replay.steps == args.high_steps * 8
            assert len(report["blocks"]) == args.high_steps // agent._rollouts
            assert all(block["completed"] for block in report["blocks"]) and not report["input_failures"]
            report["final_training_status"] = training_status()
            require_healthy(report["final_training_status"])
            report["std_changed_by_ppo"] = not torch.equal(initial_std, agent.policy.log_std_parameter)
            assert report["std_changed_by_ppo"], "Real PPO never changed the Gaussian standard deviation"
            report["total_bptt_calls"] = bptt_calls
            report["update_rounds"] = agent.wm_update_rounds
            if args.candidate_output:
                args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
                agent.save_candidate(args.candidate_output.resolve())
                report["candidate_output"] = str(args.candidate_output.resolve())
            report["peak_torch_memory_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            free, total = torch.cuda.mem_get_info(device)
            report["device_memory_mib"] = dict(free=free / 2**20, total=total / 2**20)
            report.update(completed=True, phase="teardown")
            write_report()
    except BaseException:
        report["error"] = traceback.format_exc()
        report["failed_phase"] = report["phase"]
        report["phase"] = "failed"
        report["passed"] = False
        # Do not query CUDA here: a device fault can make further CUDA calls fail.
        write_report()
        raise
    finally:
        # Save before destroying PhysX, including when teardown itself crashes.
        write_report()
        try:
            if env is not None:
                env.gym.destroy_sim(env.sim)
            if agent is not None and getattr(agent, "writer", None) is not None:
                agent.writer.close()
        except BaseException:
            report["cleanup_error"] = traceback.format_exc()
            report["passed"] = False
            write_report()
            raise
        finally:
            os.chdir(original_cwd)
            sys.argv, sys.path = original_args, original_path
    report.update(passed=True, phase="completed")
    write_report()
    print("MANIPULATION_TEACHER_GPU_PASSED " + str(output), flush=True)


if __name__ == "__main__":
    main()
