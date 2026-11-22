"""Gate D3: does the initial margin M0 predict which SGS configurations succeed?

No Isaac Gym. Run from DQ_high-level:

  python evaluate_margin_auc.py --bank data/sgs/sugar_box_v12.pt \
      --sgs_diagnostics DQ_teacher/<group>/<run>/sgs_diagnostics.pt \
      --reach_model data/reach/sugar_box_mlp/reach_model.pt --output outputs/d3_margin_auc.json

sgs_diagnostics.pt is written at the end of every SGS training run (also with
--stop_after). A mid-training run is more informative than a converged one:
when nearly every configuration succeeds, the AUC has few negatives.

For the learned T and the analytic baseline (B3), per family and overall:
  auc_feasible   AUC of M0 for "ever succeeded" (rate > 0) versus never
  auc_half       AUC of M0 for rate >= 0.5
  spearman       rank correlation of M0 and the success rate
  calibration    mean success rate in M0 quantile bins (proposition P4)
D3 passes when the learned auc_half on full_task exceeds 0.6.
"""

import argparse
import json
from pathlib import Path

D3_AUC = 0.6


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--sgs_diagnostics", type=Path, required=True)
    p.add_argument("--reach_model", type=Path, required=True)
    p.add_argument("--object_name", default="sugar_box")
    p.add_argument("--config", type=Path, default=Path("data/cfg/DQ_teacher.yaml"))
    p.add_argument("--gripper_body_index", type=int, default=None,
                   help="EE rigid-body index; defaults to the reach model's metadata")
    p.add_argument("--remaining_seconds", type=float, default=15.0,
                   help="Episode length at reset: 100 steps x 0.15 s with --rand_control")
    p.add_argument("--horizon_step", type=float, default=0.3)
    p.add_argument("--horizon_count", type=int, default=16)
    p.add_argument("--candidates", type=int, default=30)
    p.add_argument("--min_visits", type=int, default=5)
    p.add_argument("--bins", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, default=None)
    return p


def grasp_template(object_name, config_path, candidates, device):
    import yaml
    from PredictPiontTransform.predictpoint_mul import PredictPoint
    from utils.grasp_candidates import get_predict_point

    assets = yaml.safe_load(config_path.read_text())["env"]["asset"]["asset_multi"]
    if object_name not in assets:
        raise SystemExit(f"Unknown object {object_name}")
    camera, grasps, initial = get_predict_point(
        cube_predict_info_path="contact_grasp_info_mul",
        cube_root_states_info_path="30all_nomove_cube_root_states5.pt", num_env=1,
        intervel=23, delta_height=0.1, object_indices=[assets[object_name]["dict_idx"]])
    return PredictPoint(camera, grasps, initial, device=device).T_grasp_in_obj[0, :candidates]


def evaluate(m0, rates, families, family_names, bins):
    import torch
    from utils.margin_curriculum import roc_auc, spearman

    def block(mask):
        if int(mask.sum()) == 0:
            return {"configs": 0}
        x, y = m0[mask], rates[mask]
        out = {"configs": int(mask.sum()), "success_rate_mean": y.mean().item(),
               "m0_mean": x.mean().item(), "m0_nonnegative_fraction": (x >= 0).float().mean().item(),
               "auc_feasible": roc_auc(x, y > 0), "auc_half": roc_auc(x, y >= 0.5),
               "spearman": spearman(x, y)}
        edges = torch.quantile(x.double(), torch.linspace(0, 1, bins + 1, dtype=torch.float64))
        table = []
        for low, high in zip(edges[:-1], edges[1:]):
            part = (x >= low) & (x <= high)
            if part.any():
                table.append({"m0_low": low.item(), "m0_high": high.item(), "configs": int(part.sum()),
                              "success_rate_mean": y[part].mean().item()})
        out["calibration"] = table
        return out

    result = {"all": block(torch.ones_like(rates, dtype=torch.bool))}
    for index, name in enumerate(family_names):
        result[name] = block(families == index)
    return result


def main():
    args = parser().parse_args()
    for name in ("bank", "sgs_diagnostics", "reach_model", "config", "output"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    import os
    # get_predict_point reads PredictPiontTransform/ relative to the working directory.
    os.chdir(str(Path(__file__).resolve().parent))
    import torch
    from modules.reachability import TimeToReach, load_reach_model, margin_horizons, reach_state_dim
    from utils.margin_curriculum import initial_margins
    from utils.sgs import ConfigurationBank

    device = torch.device(args.device)
    bank = ConfigurationBank.load(str(args.bank), device=device)
    diagnostics = torch.load(args.sgs_diagnostics, map_location="cpu", weights_only=False)
    if diagnostics.get("bank_fingerprint") != bank.fingerprint:
        raise SystemExit("sgs_diagnostics.pt belongs to a different bank (fingerprint mismatch)")
    model, metadata = load_reach_model(args.reach_model, device=device)
    gripper_index = args.gripper_body_index if args.gripper_body_index is not None else metadata.get("gripper_body_index")
    if gripper_index is None:
        raise SystemExit("The reach model has no gripper_body_index; pass --gripper_body_index")
    num_gripper_dofs = bank.tensors["gripper_target"].shape[1]
    if model.state_dim != reach_state_dim(num_gripper_dofs):
        raise SystemExit("The reach model's state size does not match the bank's gripper DOFs")
    horizons = margin_horizons(0.0, args.horizon_step, args.horizon_count)
    template = grasp_template(args.object_name, args.config, args.candidates, device)

    models = {"learned": model, "analytic": TimeToReach(model.state_dim, kind="analytic").to(device)}
    m0 = {name: initial_margins(value, bank.tensors, template, horizons, args.remaining_seconds,
                                gripper_index, num_gripper_dofs)[:, 0].cpu()
          for name, value in models.items()}
    rates = diagnostics["success_rate"].float()
    visits = diagnostics["history_count"]
    families = diagnostics["family_id"]
    names = list(bank.family_names)
    keep = visits >= args.min_visits
    results = {name: evaluate(values[keep], rates[keep], families[keep], names, args.bins)
               for name, values in m0.items()}
    learned_full = results["learned"].get("full_task", {}).get("auc_half")
    passed = learned_full is not None and learned_full > D3_AUC
    report = {"gate": "D3", "passed": passed, "criterion": f"learned full_task auc_half > {D3_AUC}",
              "configs_total": int(rates.numel()), "configs_used": int(keep.sum()),
              "min_visits": args.min_visits, "diagnostics_step": diagnostics.get("global_step"),
              "horizons": horizons, "remaining_seconds": args.remaining_seconds,
              "results": results}
    print(json.dumps({"gate": "D3", "passed": passed, "learned_full_task_auc_half": learned_full,
                      "analytic_full_task_auc_half": results["analytic"].get("full_task", {}).get("auc_half"),
                      "configs_used": int(keep.sum())}, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.save({name: values for name, values in m0.items()}, args.output.with_suffix(".m0.pt"))


if __name__ == "__main__":
    main()
