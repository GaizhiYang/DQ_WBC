"""Export the SAPG leader as an ordinary single-branch teacher checkpoint.

A Scout & Relay checkpoint stores every branch: a learned condition vector
phi_j appended to the actor/critic MLP input, and per-branch log-std. The leader
(branch 0) is a fixed function of the observation, so phi_L can be folded into
the first linear layer exactly:

    W [x; phi_L] + b = W_x x + (W_phi phi_L + b)

The exported run directory loads with the unchanged single-policy model (214-d
actor input), eval_sgs_teacher.py and any existing deployment code. Optimizer
state is dropped; observation/value scalers are kept.

Usage (from DQ_high-level, no Isaac Gym needed):
    python export_sapg_leader.py --checkpoint RUN/checkpoints/agent_120000.pt --output_dir RUN_leader
"""

import argparse
import json
from pathlib import Path
import shutil

import torch
import yaml


LEADER = 0


def fold_branch_latent(state_dict, branch=LEADER, layer="net.0"):
    """Return a single-branch copy of a Policy/Value state dict."""
    if "branch_latent" not in state_dict:
        raise ValueError("state dict has no SAPG branch_latent (already single-branch?)")
    result = {key: value.clone() for key, value in state_dict.items() if key != "branch_latent"}
    latent = state_dict["branch_latent"]
    if not 0 <= branch < latent.shape[0]:
        raise ValueError(f"branch {branch} outside the {latent.shape[0]} SAPG branches")
    weight = state_dict[layer + ".weight"]
    width = latent.shape[1]
    if weight.shape[1] <= width:
        raise ValueError("first layer is narrower than the branch latent")
    result[layer + ".weight"] = weight[:, :-width].clone()
    result[layer + ".bias"] = state_dict[layer + ".bias"] + weight[:, -width:] @ latent[branch]
    log_std = state_dict.get("log_std_parameter")
    if log_std is not None and log_std.ndim == 2:
        if log_std.shape[0] != latent.shape[0]:
            raise ValueError("log_std_parameter rows do not match the branch count")
        result["log_std_parameter"] = log_std[branch].clone()
    return result


def export_checkpoint(checkpoint, branch=LEADER):
    modules = torch.load(checkpoint, map_location="cpu")
    if not isinstance(modules, dict) or "policy" not in modules or "value" not in modules:
        raise ValueError("expected a skrl agent checkpoint with policy and value modules")
    exported = {name: value for name, value in modules.items() if name != "optimizer"}
    exported["policy"] = fold_branch_latent(modules["policy"], branch)
    exported["value"] = fold_branch_latent(modules["value"], branch)
    return exported


def single_branch_run_files(source_run, output_run):
    """Copy run metadata, marking the model as one branch without relay training."""
    metadata_path = source_run / "teacher_run.json"
    config_name = "DQ_teacher.yaml"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        config_name = Path(metadata.get("config_file", config_name)).name
        arguments = metadata.setdefault("arguments", {})
        arguments["sapg_branches"] = 1
        arguments["relay"] = False
        metadata["exported_from"] = {"run": str(source_run.resolve()), "branch": LEADER,
                                     "method": "SAPG leader with phi_L folded into net.0"}
        (output_run / "teacher_run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    config_path = source_run / config_name
    if config_path.is_file():
        cfg = yaml.safe_load(config_path.read_text())
        cfg.setdefault("env", {}).setdefault("sapg", {})["branches"] = 1
        cfg["env"].setdefault("relay", {})["enabled"] = False
        with (output_run / config_name).open("w") as stream:
            yaml.safe_dump(cfg, stream, sort_keys=False)
    for name in ("bank_metadata.json",):
        if (source_run / name).is_file():
            shutil.copy2(source_run / name, output_run / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="SAPG agent_<step>.pt inside RUN/checkpoints")
    parser.add_argument("--output_dir", required=True, help="New run directory for the leader")
    parser.add_argument("--branch", type=int, default=LEADER,
                        help="Branch to export (default 0 = leader; scouts are for analysis only)")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    output_run = Path(args.output_dir)
    if output_run.exists() and any(output_run.iterdir()):
        raise SystemExit(f"Refusing to write into non-empty {output_run}")
    exported = export_checkpoint(checkpoint, args.branch)
    (output_run / "checkpoints").mkdir(parents=True, exist_ok=True)
    target = output_run / "checkpoints" / checkpoint.name
    torch.save(exported, target)
    single_branch_run_files(checkpoint.resolve().parent.parent, output_run)
    print(json.dumps({"exported": str(target), "branch": args.branch,
                      "actor_input": int(exported["policy"]["net.0.weight"].shape[1]),
                      "critic_input": int(exported["value"]["net.0.weight"].shape[1])}, indent=2))


if __name__ == "__main__":
    main()
