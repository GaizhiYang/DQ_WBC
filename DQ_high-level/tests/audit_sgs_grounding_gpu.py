"""Audit every SGS configuration at its saved frame and first GPU physics tick.

Run with the Isaac Gym / dqwbc environment, for example:

    python tests/audit_sgs_grounding_gpu.py --bank data/sgs/sugar_box_v9.pt \
      --device cuda:0 --num-envs 128 --seed 1043 \
      --output ../doc/sgs_v9_grounding_audit.json

Relative paths resolve from DQ_high-level, even when launched elsewhere. Each
configuration is restored once and advanced exactly one 5 ms physics tick with
the production low-level controller and Level02 table motion. No settling steps
are inserted. The final partial batch repeats its last ID only to fill simulator
slots; repeated padding is excluded from the audit records and counts.

For geometry_v1 and geometry_grasp_v2 banks, both frames pass solely when every
foot collision bottom is within [-5 mm, +20 mm] of the ground. Contact force and
base velocity are reported for inspection but do not determine acceptance.
This audit does not certify near-goal grasp retention. Historical strict banks
retain their original saved-stance and first-tick support criteria.

A failed audit writes the JSON report and exits nonzero. This is a manual GPU
audit, not a unittest, and must not run as part of ordinary test discovery.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("num-envs must be a positive integer")
    return value


def summarize(report, records):
    """Summarize the selected profile without imposing historical force gates."""
    family_names = report.get("family_names") or [
        "family_{}".format(family) for family in range(
            max((row["family"] for row in records), default=-1) + 1)]
    for family, name in enumerate(family_names):
        rows = [row for row in records if row["family"] == family]
        report["families"][str(family)] = {
            "name": name,
            "count": len(rows),
            "saved_frame_valid": sum(row["saved_frame_valid"] for row in rows),
            "first_tick_valid": sum(row["first_tick_valid"] for row in rows),
            "saved_frame_feet_near_ground": sum(row["saved_feet_near_ground"] for row in rows),
            "first_tick_feet_near_ground": sum(row["first_tick_feet_near_ground"] for row in rows),
            "min_support_feet": min((row["support_feet"] for row in rows), default=None),
            "min_foot_bottom_m": min(
                (min(row["feet_bottom_m"]) for row in rows), default=None),
            "max_foot_bottom_m": max(
                (max(row["feet_bottom_m"]) for row in rows), default=None),
            "saved_frame_min_foot_bottom_m": min(
                (min(row["saved_feet_bottom_m"]) for row in rows), default=None),
            "saved_frame_max_foot_bottom_m": max(
                (max(row["saved_feet_bottom_m"]) for row in rows), default=None),
        }
        if report.get("validation_profile") == "legacy_strict":
            report["families"][str(family)].update({
                "first_tick_supported": sum(row["reset_supported"] for row in rows),
                "first_tick_settled": sum(row["settled"] for row in rows),
                "saved_frame_grounded": sum(row["saved_frame_valid"] for row in rows),
                "saved_frame_min_support_feet": min(
                    (row["saved_support_feet"] for row in rows), default=None),
            })
    report["failed_config_ids"] = [
        row["config_id"] for row in records
        if not row["first_tick_valid"] or not row["saved_frame_valid"]]
    report["saved_frame_failed_config_ids"] = [
        row["config_id"] for row in records if not row["saved_frame_valid"]]
    report["completed_configs"] = len(records)
    report["audit_complete"] = len(records) == report["num_configs"] and bool(records)
    report["saved_frame_passed"] = (
        report["audit_complete"] and not report["saved_frame_failed_config_ids"])
    report["passed"] = (report["audit_complete"] and not report["failed_config_ids"]
                        and "error" not in report and "cleanup_error" not in report)
    report["records"] = records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", default="data/sgs/sugar_box_v9.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=positive_int, default=128)
    parser.add_argument("--seed", type=int, default=1043)
    parser.add_argument("--output", default="../doc/sgs_v9_grounding_audit.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "bank": str(Path(args.bank).resolve()),
        "bank_fingerprint": None,
        "num_configs": 0,
        "device": args.device,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "families": {},
        "failed_config_ids": [],
        "passed": False,
        "physics_steps_per_reset": 1,
        "settling_steps_after_reset": 0,
    }
    records = []
    env = None
    started = time.monotonic()
    try:
        # Isaac Gym must precede torch and all project modules importing torch.
        from isaacgym import gymapi  # noqa: F401
        import torch
        from utils.sgs import ConfigurationBank
        from utils.sgs_bank_builder import create_builder_environment, restored_physics_tick
        from utils.sgs_grounding import (
            APPROX_GROUND_LIMITS, FOOT_RADIUS, GROUND_LIMITS, RESET_MIN_SUPPORT_FEET,
            foot_indices, grounding_state, approx_grounded_mask, grounded_mask,
            reset_supported_mask,
        )

        if torch.device(args.device).type != "cuda":
            raise ValueError("This GPU audit requires a CUDA device, e.g. --device cuda:0")
        bank = ConfigurationBank.load(args.bank, device=args.device)
        physical = bank.metadata.get("physical_validation", {})
        profile_name = physical.get("validation_profile")
        geometry_profile = profile_name in ("geometry_v1", "geometry_grasp_v2")
        if not geometry_profile:
            bank.validate({"physical_validation": {"grounded_initial_states": True}})
        limits = APPROX_GROUND_LIMITS if geometry_profile else GROUND_LIMITS
        family_names = bank.metadata.get("family_names") or [
            "family_{}".format(family)
            for family in range(int(bank.tensors["family_id"].max().item()) + 1)]
        report.update(bank_fingerprint=bank.fingerprint, num_configs=bank.num_configs,
                      family_names=family_names,
                      bank_validation_profile=profile_name,
                      validation_profile=profile_name if geometry_profile else "legacy_strict",
                      saved_frame_limits=dict(limits),
                      first_tick_limits=dict(limits),
                      force_and_velocity_affect_acceptance=not geometry_profile)
        if not geometry_profile:
            report["first_tick_min_support_feet"] = RESET_MIN_SUPPORT_FEET
        geometry = physical["grounding"]
        saved_bottom = bank.tensors["rigid_body_state"][:, geometry["foot_body_indices"], 2]
        saved_bottom = saved_bottom - FOOT_RADIUS
        saved_force = bank.tensors.get("initial_foot_contact_force")
        saved_near = approx_grounded_mask(saved_bottom)
        saved_valid = (saved_near if geometry_profile else grounded_mask(
            bank.tensors["robot_root"], saved_bottom, saved_force))

        def support_count(bottom, force):
            near = ((bottom >= limits["min_foot_bottom_m"])
                    & (bottom <= limits["max_foot_bottom_m"]))
            return (near & (force[..., 2] > GROUND_LIMITS["min_contact_force_z_n"])).sum(dim=1)

        saved_support = (support_count(saved_bottom, saved_force)
                         if saved_force is not None else None)
        env = create_builder_environment(args.num_envs, args.device, args.seed)
        report["physics_dt_s"] = float(env.sim_params.dt)
        if not math.isclose(env.sim_params.dt, .005, rel_tol=0., abs_tol=1e-9):
            raise ValueError("The first-tick audit requires the configured 5 ms physics timestep")
        if foot_indices(env) != geometry["foot_body_indices"]:
            raise ValueError("Bank foot indices do not match the simulator's robot asset")

        with torch.no_grad():
            for start in range(0, bank.num_configs, env.num_envs):
                ids = torch.arange(start, min(start + env.num_envs, bank.num_configs),
                                   device=env.device)
                count = ids.numel()
                padded = torch.cat((ids, ids[-1:].expand(env.num_envs - count)))
                state = bank.batch(padded)
                restored_physics_tick(env, state)
                bottom, force = grounding_state(env)
                near = approx_grounded_mask(bottom)
                first_tick_valid = near if geometry_profile else reset_supported_mask(bottom, force)
                # Transfer batch measurements after the real physics tick. The
                # saved contact evidence is never written into live contacts.
                values = {
                    "config_id": ids,
                    "family": state["family_id"][:count],
                    "feet_bottom_m": bottom[:count],
                    "force_z_n": force[:count, :, 2],
                    "contact_force_n": force[:count],
                    "support_feet": support_count(bottom, force)[:count],
                    "first_tick_valid": first_tick_valid[:count],
                    "first_tick_feet_near_ground": near[:count],
                    "base_linear_speed_m_s": env._robot_root_states[:count, 7:10].norm(dim=-1),
                    "base_angular_speed_rad_s": env._robot_root_states[:count, 10:13].norm(dim=-1),
                    "saved_feet_bottom_m": saved_bottom[ids],
                    "saved_feet_near_ground": saved_near[ids],
                    "saved_frame_valid": saved_valid[ids],
                }
                if saved_force is not None:
                    values.update({
                        "saved_force_z_n": saved_force[ids, :, 2],
                        "saved_contact_force_n": saved_force[ids],
                        "saved_support_feet": saved_support[ids],
                    })
                if not geometry_profile:
                    values.update({
                        "reset_supported": first_tick_valid[:count],
                        "settled": grounded_mask(env._robot_root_states, bottom, force)[:count],
                        "saved_grounded": saved_valid[ids],
                    })
                values = {name: value.detach().cpu().tolist() for name, value in values.items()}
                records.extend({name: value[index] for name, value in values.items()}
                               for index in range(count))
                print("Audited {}/{} configurations".format(len(records), bank.num_configs),
                      flush=True)
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                try:
                    if getattr(env, "viewer", None) is not None:
                        env.gym.destroy_viewer(env.viewer)
                finally:
                    env.gym.destroy_sim(env.sim)
            except Exception as error:
                report["cleanup_error"] = "{}: {}".format(type(error).__name__, error)
                traceback.print_exc()
        summarize(report, records)
        report["wall_seconds"] = time.monotonic() - started
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("SGS_GROUNDING_AUDIT " + json.dumps(
            {name: value for name, value in report.items() if name != "records"},
            ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
