"""Exercise temporal target inference through real strict-evaluation resets.

Run in the dqwbc environment, for example:

    python tests/smoke_tsil_lookup_gpu.py --checkpoint /path/agent_144.pt \
        --require_no_success_checkpoint --sim_device cuda:1 --rl_device cuda:1 \
        --smoke_json /tmp/dq_tsil_lookup.json

The existing checkpoint is never written. After testing its actual reference
availability, this test marks full-task references successful ONLY IN MEMORY
with D=8 seconds to exercise the previously broken lookup branch. This is a
lookup fixture, not a learned success rate, and no episode outcome is recorded.
Targets and all scaler statistics must remain frozen during each reset/query.
"""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--smoke_json", default="/tmp/dq_tsil_lookup.json")
    parser.add_argument("--lookup_waves", type=int, default=3)
    parser.add_argument("--require_no_success_checkpoint", action="store_true")
    options, overrides = parser.parse_known_args()
    if options.lookup_waves < 1:
        raise ValueError("--lookup_waves must be positive")
    output = Path(options.smoke_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sys.argv = [str(ROOT / "eval_sgs_teacher.py"), "--num_envs", "32",
                "--eval_episodes", "32", "--headless", "--seed", "91"] + overrides
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    # The production entrypoint imports Isaac Gym before torch.
    from train_multistate_DQ_teacher import get_trainer
    import numpy as np
    import torch
    from utils.tsil import (geometry_descriptors, GEOMETRY_DESCRIPTOR_SCHEMA,
                            LOOKUP_NEAREST, LOOKUP_NO_SUCCESS, LOOKUP_OUT_OF_SUPPORT,
                            LOOKUP_INVALID, LOOKUP_STATUS_NAMES)

    report = {"passed": False, "kind": "strict_reset_lookup_regression_not_learning_result",
              "synthetic_references_are_in_memory_only": True, "checks": {}, "waves": []}
    trainer, original, started = None, None, time.monotonic()

    def check(name, value):
        if not value:
            raise AssertionError(name)
        report["checks"][name] = True

    def equal(first, second):
        if torch.is_tensor(first):
            return torch.is_tensor(second) and torch.equal(first, second)
        if isinstance(first, dict):
            return first.keys() == second.keys() and all(equal(first[k], second[k]) for k in first)
        if isinstance(first, (tuple, list)):
            return len(first) == len(second) and all(equal(a, b) for a, b in zip(first, second))
        return first == second

    def status_counts(status):
        return {name: int((status == code).sum().item())
                for code, name in LOOKUP_STATUS_NAMES.items()}

    try:
        trainer = get_trainer(is_eval=True, strict_eval=True)
        env, agent, args = trainer.env._env, trainer.agents, trainer.teacher_args
        state = env.tsil_state
        check("strict_task_and_sampler_unchanged", not env.sgs_enabled and env.hold_steps == 10
              and env.max_episode_length == 100 and agent.memory is None)
        check("stable_descriptor_schema", state.descriptor_schema == GEOMETRY_DESCRIPTOR_SCHEMA
              and state.descriptors.shape[1] == 5)
        agent.set_mode("eval")
        agent.policy.deterministic = True
        scaler_objects = {name: getattr(agent, name) for name in
                          ("_state_preprocessor", "_value_preprocessor", "reward_normalizer")}
        for scaler in scaler_objects.values():
            if hasattr(scaler, "eval"):
                scaler.eval()
        frozen_scalers = {name: deepcopy(scaler.state_dict()) for name, scaler in scaler_objects.items()}
        original = deepcopy(state.state_dict())
        actual_successful = int((state.seen_success & (state.family_ids == 0)).sum())
        report.update(checkpoint=str(Path(args.checkpoint).resolve()), num_envs=env.num_envs,
                      sim_device=args.sim_device, rl_device=args.rl_device,
                      actual_successful_full_task_references=actual_successful,
                      descriptor_schema=state.descriptor_schema,
                      descriptor_names=state.descriptor_names)
        if options.require_no_success_checkpoint:
            check("checkpoint_has_no_real_success_references", actual_successful == 0)

        def reset(seed):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            env.reset_buf.fill_(1)
            observations, _ = trainer.env.reset()
            with torch.no_grad():
                action = agent.act(observations, timestep=0, timesteps=trainer.timesteps)[0]
            check("actor_inference_finite", bool(torch.isfinite(action).all()))
            check("reset_target_matches_observation", torch.allclose(
                observations[:, -190], env.tsil_episode_D / env.tsil_time_reference))
            check("reset_elapsed_zero", bool((env.tsil_elapsed == 0).all()))
            return observations

        reset(args.seed)
        report["actual_checkpoint_reset_status"] = status_counts(env.tsil_lookup_status)
        if actual_successful == 0:
            check("real_no_success_returns_reference", bool(
                (env.tsil_lookup_status == LOOKUP_NO_SUCCESS).all()
                and (env.tsil_episode_D == env.tsil_time_reference).all()))
        check("actual_checkpoint_reset_does_not_modify_state", equal(original, state.state_dict()))

        # This intentional fixture never changes a checkpoint file or simulator
        # success flags; it supplies references independently of learned ability.
        full = state.family_ids == 0
        check("full_task_references_available", bool(full.any()))
        state.seen_success[full] = True
        state.targets[full] = 8.
        synthetic = deepcopy(state.state_dict())
        report["synthetic_successful_full_task_references"] = int(full.sum())
        report["synthetic_target_seconds"] = 8.
        for wave in range(options.lookup_waves):
            reset(args.seed + wave)
            check("all_real_resets_use_successful_neighbors", bool(
                (env.tsil_lookup_status == LOOKUP_NEAREST).all()))
            check("all_real_resets_receive_eight_seconds", bool((env.tsil_episode_D == 8.).all()))
            check("query_does_not_modify_temporal_state", equal(synthetic, state.state_dict()))
            report["waves"].append(dict(wave=wave, seed=args.seed + wave,
                                        queries=env.num_envs,
                                        status_counts=status_counts(env.tsil_lookup_status),
                                        targets_seconds=env.tsil_episode_D.unique().cpu().tolist()))

        descriptor = geometry_descriptors(env._robot_root_states, env._cube_root_states,
                                          env.table_heights, env._dof_pos)
        outliers = descriptor[:1].repeat(4, 1)
        outliers[0, 0] = state.descriptor_max[0] + 2.  # true distance OOD
        outliers[1, 4] = state.descriptor_max[4] + 1.  # true table-height OOD
        outliers[2, 2:4] = torch.tensor([-1., 0.], device=outliers.device)  # object heading OOD
        outliers[3, 0] = float("nan")
        targets, statuses = state.lookup_targets(outliers, return_status=True)
        check("genuine_ood_and_invalid_inputs_fall_back", torch.equal(
            statuses, torch.tensor([LOOKUP_OUT_OF_SUPPORT] * 3 + [LOOKUP_INVALID], device=statuses.device))
            and bool((targets == env.tsil_time_reference).all()))
        report["outlier_status_counts"] = status_counts(statuses)
        check("outlier_queries_do_not_modify_temporal_state", equal(synthetic, state.state_dict()))
        check("all_scalers_remain_frozen", all(equal(frozen_scalers[name], scaler.state_dict())
                                               for name, scaler in scaler_objects.items()))
        check("no_episode_outcomes_fabricated", not bool(env.episode_success.any()))
        report["real_reset_queries_with_synthetic_references"] = options.lookup_waves * env.num_envs
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if trainer is not None:
            env = trainer.env._env
            if original is not None:
                env.tsil_state.load_state_dict(original)
                report["original_temporal_state_restored"] = equal(original, env.tsil_state.state_dict())
            env.gym.destroy_sim(env.sim)
        report["wall_time_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
