"""Build an explicit, offline SGS reset bank with real Isaac Gym validation."""
import argparse
import json
from pathlib import Path

# Isaac Gym requires this import before PyTorch or modules importing PyTorch.
from isaacgym import gymapi  # noqa: F401

from utils.sgs_bank_builder import PhysicsBankBuilder, create_builder_environment
from utils.sgs_build_config import DEFAULT_BUILD_CONFIG_PATH, load_build_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--per-family", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-batches", type=int, default=512)
    parser.add_argument("--config", default="data/cfg/DQ_teacher.yaml")
    parser.add_argument("--reset-config", default=str(DEFAULT_BUILD_CONFIG_PATH),
                        help="YAML ranges for the three reset families (separate from --config).")
    parser.add_argument("--output", default="data/sgs/sugar_box_v5.pt")
    args = parser.parse_args()
    if args.num_envs < 1:
        parser.error("--num-envs must be positive")
    # Reject invalid geometry before allocating a GPU simulator.
    reset_config = load_build_config(args.reset_config)
    print("SGS reset config:", str(Path(args.reset_config).resolve()), flush=True)
    env = create_builder_environment(
        num_envs=args.num_envs, device=args.device, seed=args.seed,
        config_path=args.config)
    try:
        builder = PhysicsBankBuilder(env, args.per_family, args.seed,
                                     reset_config=reset_config)
        bank = builder.build(args.max_batches)
        output = Path(args.output)
        bank.save(output)
        audit_path = output.with_suffix(".json")
        audit_path.write_text(json.dumps(
            {"fingerprint": bank.fingerprint, "num_configs": bank.num_configs,
             "metadata": bank.metadata}, indent=2), encoding="utf-8")
        print(f"Saved {bank.num_configs} validated configurations to {output}", flush=True)
        print(f"Audit: {audit_path}; fingerprint: {bank.fingerprint}", flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
