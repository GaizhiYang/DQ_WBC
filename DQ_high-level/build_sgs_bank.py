"""Build an explicit, offline SGS reset bank with real Isaac Gym validation."""
import argparse
import json
import time
import yaml
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
                        help="YAML ranges for reset families (separate from --config).")
    parser.add_argument("--output", default="data/sgs/sugar_box_v10.pt")
    args = parser.parse_args()
    if min(args.num_envs, args.per_family, args.max_batches) < 1:
        parser.error("--num-envs, --per-family and --max-batches must be positive")
    # Reject invalid geometry before allocating a GPU simulator.
    teacher_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    reset_config = load_build_config(args.reset_config, lifted_threshold=teacher_config["env"]["liftedSuccessThreshold"])
    print("SGS reset config:", str(Path(args.reset_config).resolve()), flush=True)
    started = time.monotonic()
    print("SGS: initializing simulator and loading robot/controller assets...", flush=True)
    env = create_builder_environment(
        num_envs=args.num_envs, device=args.device, seed=args.seed,
        config_path=args.config)
    print(f"SGS: simulator ready in {time.monotonic() - started:.1f}s; starting geometry sampling.", flush=True)
    try:
        builder = PhysicsBankBuilder(env, args.per_family, args.seed,
                                     reset_config=reset_config)
        output = Path(args.output)
        try:
            bank = builder.build(args.max_batches)
        except RuntimeError as error:
            # Keep useful rejection evidence even when no complete bank exists.
            failure_path = output.with_suffix(".failed.json")
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(json.dumps({
                "error": str(error), "reset_config": reset_config,
                "accepted_per_family": builder.counts, "batches": builder.audit,
                "sampling_audit": builder.sampling_audit.summary(),
            }, indent=2), encoding="utf-8")
            print(f"SGS build failed; diagnostics: {failure_path}", flush=True)
            raise
        bank.save(output)
        audit_path = output.with_suffix(".json")
        audit_path.write_text(json.dumps(
            {"fingerprint": bank.fingerprint, "num_configs": bank.num_configs,
             "metadata": bank.metadata}, indent=2), encoding="utf-8")
        print(f"Saved {bank.num_configs} validated configurations to {output}", flush=True)
        print(f"Audit: {audit_path}; fingerprint: {bank.fingerprint}", flush=True)
        print(f"SGS total wall time including simulator startup: {time.monotonic() - started:.1f}s", flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
