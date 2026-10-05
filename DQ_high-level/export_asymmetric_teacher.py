"""Export the asymmetric actor, normalization and belief contract without Isaac Gym."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from modules.asymmetric_teacher import AsymmetricTeacherPolicy, packed_obs_dim
    from utils.asymmetric_teacher_training import feature_flags
    from utils.teacher_vision_preprocessor import PrefixRunningStandardScaler
    from utils.asymmetric_teacher_preprocessor import export_deployment

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "asymmetric_teacher_state" not in checkpoint:
        raise ValueError("Expected a full asymmetric checkpoint")
    state = checkpoint["asymmetric_teacher_state"]
    if state.get("version") != 2:
        raise ValueError("Initialize a version-2 experiment from the old checkpoint before exporting")
    settings = state["experiment_config"]["asymmetric_teacher"]
    flags = feature_flags(settings)
    policy = AsymmetricTeacherPolicy((packed_obs_dim(**flags),), (9,), "cpu", **flags)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    scaler = PrefixRunningStandardScaler(size=packed_obs_dim(**flags), device="cpu")
    scaler.load_state_dict(checkpoint["state_preprocessor"], strict=True)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = export_deployment(policy, scaler)
    payload["metadata"]["perception_config"] = settings.get("perception", {})
    torch.save(payload, output)
    print("Exported deployment Actor:", output)


if __name__ == "__main__":
    main()
