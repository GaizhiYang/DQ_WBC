"""Sample a Near-Goal bank for independent cold-restore and release diagnostics.

Run manually in the Isaac Gym / dqwbc environment, for example:

    python tests/audit_sgs_near_goal_gpu.py --bank data/sgs/sugar_box_v9.pt \
      --device cuda:0 --num-envs 32 --seed 1043 \
      --output ../doc/sgs_v9_near_goal_restore_audit.json

Paths resolve from DQ_high-level. Sample IDs are drawn uniformly without
replacement. The closed-gripper test uses saved targets and the builder's hold
criteria in a freshly constructed simulator. The release control restores those
same states again, opens the gripper, and runs 0.8 s of ordinary physics with the
low-level policy and Level02 table controller. It reports contact and height
changes without requiring every object to fall. Neither phase is a task-success
evaluation or a learned-policy rollout. This file never runs under unittest
discovery. A failed closed-gripper check writes its report and exits nonzero.
"""

import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", default="data/sgs/sugar_box_v9.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1043)
    parser.add_argument("--output", default="../doc/sgs_v9_near_goal_restore_audit.json")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    return args


def json_finite(value):
    """Keep a failed numeric run report writable as strict JSON."""
    if isinstance(value, dict):
        return {key: json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main():
    args = parse_args()  # --help does not import Isaac Gym or allocate a GPU.
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "kind": "near_goal_cold_restore_and_release_diagnostic_not_task_evaluation",
        "passed": False, "bank": str(Path(args.bank).resolve()),
        "device": args.device, "seed": args.seed, "num_envs": args.num_envs,
        "sampling": "uniform_without_replacement_within_near_goal",
        "release_note": "Release is a negative-control diagnostic. Loss of two-finger contact "
                        "and object height/relative-pose changes are reported, not required for every state.",
        "nonfinite_numbers": "Encoded as null if an invalid physical state occurs.",
    }
    env = None
    started = time.monotonic()
    try:
        # Isaac Gym must precede torch and every transitive torch import.
        from isaacgym import gymapi  # noqa: F401
        import torch
        from utils.sgs import ConfigurationBank
        from utils.sgs_bank_builder import (
            PhysicsBankBuilder, create_builder_environment, restore_snapshot,
            quaternion_matrix,
        )
        from utils.sgs_grounding import grasp_contact_body_indices

        bank = ConfigurationBank.load(args.bank, device=args.device, expected_metadata={
            "object_name": "sugar_box", "table_level": "Level02",
            "physical_validation": {"validation_profile": "geometry_grasp_v2",
                                    "near_goal_grasp_verified": True},
        })
        report["fingerprint"] = bank.fingerprint
        names = bank.metadata["family_names"]
        if "near_goal" not in names:
            raise ValueError("The selected bank has no near_goal family")
        candidates = (bank.tensors["family_id"] == names.index("near_goal")).nonzero().flatten().cpu().tolist()
        if args.num_envs > len(candidates):
            raise ValueError(f"--num-envs={args.num_envs} exceeds {len(candidates)} near_goal states; "
                             "sampling must be without replacement")
        selected = random.Random(args.seed).sample(candidates, args.num_envs)
        report["sample_config_ids"] = selected
        state = bank.batch(torch.tensor(selected, device=bank.device))
        env = create_builder_environment(num_envs=args.num_envs, device=args.device, seed=args.seed)
        builder = PhysicsBankBuilder(
            env, per_family=1, seed=args.seed,
            reset_config=bank.metadata["construction"]["reset_config"])
        report["effective_near_goal_config"] = builder.reset_config["near_goal"]

        def plain(tensor):
            return tensor.detach().cpu().tolist()

        def measurements():
            contacts, between, clear = builder._grasp_geometry()
            rotation = quaternion_matrix(env._cube_root_states[:, 3:7])
            lowest_z = ((rotation[:, 2, :] @ builder.object_vertices.T).amin(dim=1)
                        + env._cube_root_states[:, 2])
            return {
                "lift_height_m": plain(env._cube_root_states[:, 2] - env.table_heights - env.init_height),
                "gripper_opening": plain(builder._joint_to_opening(env._dof_pos[:, -1])),
                "gripper_target_opening": plain(builder._joint_to_opening(env.gripper_dof_pos[:, -1])),
                "two_finger_contact": plain(contacts), "object_in_grip_region": plain(between),
                "object_clear_of_table": plain(clear),
                "lowest_mesh_clearance_m": plain(lowest_z - env.table_heights),
                "finger_contact_force_n": plain(
                    env._contact_forces[:, grasp_contact_body_indices(env)].norm(dim=-1)),
                "object_speed_m_s": plain(env._cube_root_states[:, 7:10].norm(dim=1)),
            }

        with torch.no_grad():
            restore_snapshot(env, state)
            builder._near_elapsed_ticks = 0
            checks, evidence = builder._near_goal_hold(
                env.ee_goal_global_cart.clone(), env.ee_goal_global_orn_quat.clone(),
                state["gripper_target"].clone())
            accepted = torch.stack(list(checks.values()), dim=1).all(dim=1)
            closed_passed = bool(accepted.all())
            report["closed_hold"] = {
                "duration_s": builder._near_elapsed_ticks * env.sim_params.dt,
                "passed": closed_passed, "passed_count": int(accepted.sum()),
                "checks": {name: plain(value) for name, value in checks.items()},
                "evidence": {name: plain(value) for name, value in evidence.items()},
                "failed_config_ids": [selected[i] for i, passed in enumerate(plain(accepted)) if not passed],
                "failures_by_check": {
                    name: [selected[i] for i, passed in enumerate(plain(value)) if not passed]
                    for name, value in checks.items()},
                "final": measurements(),
            }
            print("SGS_NEAR_GOAL_CLOSED", {"passed": closed_passed, "count": int(accepted.sum()),
                                           "total": args.num_envs}, flush=True)

            # Restore the same buffered initial states; do not reuse the warmed
            # final state of the closed-gripper test or its contact/history cache.
            restore_snapshot(env, state)
            builder._near_elapsed_ticks = 0
            position = env.ee_goal_global_cart.clone()
            quaternion = env.ee_goal_global_orn_quat.clone()
            opened = builder._opening_to_joint(torch.ones(args.num_envs, device=env.device))[:, None]
            initial_height = env._cube_root_states[:, 2] - env.table_heights - env.init_height
            initial_opening = builder._joint_to_opening(env._dof_pos[:, -1])
            initial_relative_position, _ = builder._grasp_relative_pose()
            min_height = initial_height.clone()
            max_relative_displacement = torch.zeros(args.num_envs, device=env.device)
            lost_contact_frames = torch.zeros_like(max_relative_displacement)
            release_steps = math.ceil(.8 / env.sim_params.dt)
            for step in range(release_steps):
                builder._near_goal_step(position, quaternion, opened, step, table_target=None)
                contacts, _, _ = builder._grasp_geometry()
                lost_contact_frames += (~contacts).float()
                height = env._cube_root_states[:, 2] - env.table_heights - env.init_height
                min_height = torch.minimum(min_height, height)
                relative, _ = builder._grasp_relative_pose()
                max_relative_displacement = torch.maximum(
                    max_relative_displacement, (relative - initial_relative_position).norm(dim=1))
            final = measurements()
            final_height = env._cube_root_states[:, 2] - env.table_heights - env.init_height
            report["release"] = {
                "duration_s": release_steps * env.sim_params.dt,
                "initial_lift_height_m": plain(initial_height),
                "initial_gripper_opening": plain(initial_opening),
                "final_height_change_m": plain(final_height - initial_height),
                "minimum_lift_height_m": plain(min_height),
                "maximum_relative_displacement_m": plain(max_relative_displacement),
                "lost_two_finger_contact_fraction": plain(lost_contact_frames / release_steps),
                "lost_two_finger_contact_final_count": sum(not value for value in final["two_finger_contact"]),
                "height_decreased_over_2cm_count": int(((initial_height - final_height) > .02).sum()),
                "final": final,
            }
            report["passed"] = closed_passed
            print("SGS_NEAR_GOAL_RELEASE", {
                key: report["release"][key] for key in (
                    "lost_two_finger_contact_final_count", "height_decreased_over_2cm_count")}, flush=True)
    except Exception as error:
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        try:
            output.write_text(json.dumps(json_finite(report), indent=2, allow_nan=False) + "\n",
                              encoding="utf-8")
            print("SGS_NEAR_GOAL_AUDIT", str(output), flush=True)
        finally:
            if env is not None:
                env.gym.destroy_sim(env.sim)
    if not report["passed"]:
        raise SystemExit("Closed-gripper cold-restore checks failed; inspect the written audit report.")


if __name__ == "__main__":
    main()
