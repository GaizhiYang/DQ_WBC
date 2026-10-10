"""Export raw67 -> mean9 TorchScript for a minimal geometric teacher checkpoint."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import torch
    from utils.minimal_asymmetric_deployment import export_deployment

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    output = Path(args.output).expanduser().resolve()
    export_deployment(checkpoint, output)
    print("Exported minimal Actor TorchScript:", output)
    print("Input: raw [batch,67]; output: mean [batch,9]. Normalization and metadata.json are included.")
    print("DQ action clipping/scaling, target integration and real perception remain external.")


if __name__ == "__main__":
    main()
