"""Check actual ordinary-reset parity and construct a tiny four-family bank."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from isaacgym import gymapi  # noqa: F401
    import torch
    from utils.sgs import ConfigurationBank
    from utils.sgs_bank_builder import PhysicsBankBuilder, create_builder_environment, snapshot
    from utils.sgs_build_config import load_build_config

    args.output.mkdir(parents=True, exist_ok=False)
    env = create_builder_environment(num_envs=args.num_envs, device=args.device,
                                     seed=73, motion_level="Level03")
    report = {"passed": False, "num_envs": args.num_envs, "device": args.device}
    try:
        builder = PhysicsBankBuilder(env, per_family=2, seed=73,
            reset_config=load_build_config("data/cfg/sgs_bank_level03.yaml"))
        captured = {}
        original_reset = env.reset_idx

        def capture_reset(*a, **kw):
            original_reset(*a, **kw)
            captured["state"] = snapshot(env)
            captured["time"] = float(env.gym.get_sim_time(env.sim))

        env.reset_idx = capture_reset
        before_time = float(env.gym.get_sim_time(env.sim))
        full, checks = builder._full()
        assert all(bool(value.all()) for value in checks.values())
        for name, value in captured["state"].items():
            torch.testing.assert_close(full[name], value, atol=0., rtol=0., msg=name)
        assert captured["time"] == float(env.gym.get_sim_time(env.sim))
        report["ordinary_snapshot_fields_identical"] = list(captured["state"])
        report["full_reset_physics_ticks"] = round((captured["time"] - before_time) / env.sim_params.dt)
        assert report["full_reset_physics_ticks"] == 1
        root = full["robot_root"]
        x,y,z,w = root[:,3:7].unbind(-1)
        yaw = torch.rad2deg(torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
        report["ordinary_base_yaw_range_deg"] = [float(yaw.min()),float(yaw.max())]
        report["ordinary_base_height_range_m"] = [float(root[:,2].min()),float(root[:,2].max())]
        assert bool((yaw.abs() <= 15.1).all())
        preserved = {name: value.clone() for name,value in full.items()}
        course = builder._curriculum_initial(full)
        for name,value in full.items():
            torch.testing.assert_close(value,preserved[name],atol=0.,rtol=0.,msg=name)
        for prefix in ("level03_","level03_x_"):
            assert not bool(full[prefix+"driver_initialized"].any())
            assert bool(course[prefix+"driver_initialized"].all())
        report["curriculum_does_not_mutate_full_snapshot"] = True
        env.reset_idx = original_reset

        bank = builder.build(max_batches=128)
        path = args.output / "tiny_bank.pt"
        bank.save(path)
        restored = ConfigurationBank.load(path,device="cpu")
        assert restored.fingerprint == bank.fingerprint
        report["bank_fingerprint"] = bank.fingerprint
        report["family_names"] = list(bank.family_names)
        report["family_counts"] = {
            name:int((bank.tensors["family_id"]==i).sum()) for i,name in enumerate(bank.family_names)}
        at = bank.tensors["family_id"] == bank.family_names.index("at_grasp")
        report["at_grasp_distance_m"] = bank.tensors["candidate_local_offset"][at].norm(dim=-1).tolist()
        report["at_grasp_opening"] = bank.tensors["gripper_opening"][at].tolist()
        report["validation_profile"] = bank.metadata["physical_validation"]["validation_profile"]
        report["passed"] = True
        (args.output/"bank_metadata.json").write_text(json.dumps(bank.metadata,indent=2)+"\n")
    finally:
        (args.output/"validation.json").write_text(json.dumps(report,indent=2)+"\n")
        env.gym.destroy_sim(env.sim)
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
