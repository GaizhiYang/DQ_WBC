"""Exercise real PhysX table/chassis collision, latching and episode reset.

Run manually from DQ_high-level in the dqwbc environment. This creates four
small GPU scenes; it is not included in unittest discovery.
"""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from isaacgym import gymtorch
    from isaacgym.torch_utils import quat_apply, quat_mul
    import torch
    from utils.sgs_bank_builder import create_builder_environment
    from utils.table_collision import box_separation

    env = create_builder_environment(num_envs=4, device=args.device,
                                     seed=71, motion_level="Level03")
    report = {"passed": False, "device": args.device}
    try:
        env.reset_idx()
        assert not bool(env.table_chassis_collision.any()), "Ordinary reset overlaps chassis"
        initial_table = env._table_root_states.clone()
        initial_object = env._cube_root_states.clone()
        chassis_q = env._rigid_body_rot[:, env.chassis_idx]
        chassis_center = (env._rigid_body_pos[:, env.chassis_idx]
                          + quat_apply(chassis_q, env._chassis_box_offset))
        box_q = quat_mul(chassis_q, env._chassis_box_rotation)
        # Scene 0: actual 2 cm overlap into the side of the trunk box.
        # Scene 1: a 1 cm gap; scene 2: table against the end effector, away
        # from the trunk; scene 3: untouched ordinary reset.
        offsets = torch.zeros(4, 3, device=env.device)
        offsets[:, 1] = env._chassis_half_extents[1] + env._table_half_extents[1]
        offsets[0, 1] -= .02
        offsets[1, 1] += .01
        env._table_root_states[:2, :3] = (chassis_center + quat_apply(box_q, offsets))[:2]
        env._table_root_states[:2, 3:7] = box_q[:2]
        env._table_root_states[2, :3] = env.ee_pos[2]
        env._table_root_states[:, 7:] = 0.
        displacement = env._table_root_states[:, :3] - initial_table[:, :3]
        env._cube_root_states[:, :3] += displacement
        env.table_heights[:] = env._table_root_states[:, 2] + env.table_dimz / 2
        env._contact_forces.zero_()
        env._update_table_chassis_collision()
        report["pre_physics_collision"] = env.table_chassis_collision.tolist()
        assert report["pre_physics_collision"] == [True, False, False, False]
        env.table_chassis_collision.zero_()

        def upload_and_simulate():
            actor_ids = torch.cat((env._table_actor_ids, env._cube_actor_ids)).to(torch.int32)
            env.gym.set_actor_root_state_tensor_indexed(
                env.sim, gymtorch.unwrap_tensor(env._root_states),
                gymtorch.unwrap_tensor(actor_ids), len(actor_ids))
            env.gym.simulate(env.sim)
            env._refresh_sim_tensors()

        upload_and_simulate()
        report["post_physics_collision"] = env.table_chassis_collision.tolist()
        report["trunk_contact_force_n"] = env._contact_forces[:, env.chassis_idx].norm(dim=-1).tolist()
        report["rest_offset_m"] = env._table_chassis_rest_offset
        q = env._rigid_body_rot[:, env.chassis_idx]
        report["post_physics_box_gap_m"] = box_separation(
            env._rigid_body_pos[:, env.chassis_idx] + quat_apply(q, env._chassis_box_offset),
            quat_mul(q, env._chassis_box_rotation), env._chassis_half_extents,
            env._table_root_states[:, :3], env._table_root_states[:, 3:7],
            env._table_half_extents).tolist()
        assert report["post_physics_collision"] == [True, False, False, False]
        assert report["trunk_contact_force_n"][0] > .01, "No physical trunk contact occurred"
        # Separating the actors does not erase an earlier collision this episode.
        env._table_root_states.copy_(initial_table)
        env._cube_root_states.copy_(initial_object)
        env.table_heights[:] = initial_table[:, 2] + env.table_dimz / 2
        upload_and_simulate()
        assert bool(env.table_chassis_collision[0])
        env.check_termination()
        report["collision_terminal"] = bool(env._terminate_buf[0])
        report["collision_reset"] = bool(env.reset_buf[0])
        report["collision_success"] = bool(env.pick_success_recorded[0])
        report["collision_timeout"] = bool(env.timeout_buf[0])
        assert report["collision_terminal"] and report["collision_reset"]
        assert not report["collision_success"] and not report["collision_timeout"]
        env.reset_idx(torch.tensor([0], device=env.device))
        report["reset_cleared_collision"] = not bool(env.table_chassis_collision[0])
        assert report["reset_cleared_collision"]
        report["passed"] = True
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        env.gym.destroy_sim(env.sim)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
