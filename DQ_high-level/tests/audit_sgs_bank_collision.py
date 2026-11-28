"""CPU-only audit of saved table/chassis box overlap; never starts a simulator.

The body index must come from the matching environment's env.chassis_idx.
The bank does not save body names. Inferring the trunk from the actor root or
the nearest body would be incorrect: the URDF root is a separate base link.

Example (replace INDEX with the verified env.chassis_idx):
    python tests/audit_sgs_bank_collision.py --bank data/sgs/sugar_box_level03_v2.pt \
        --chassis-body-index INDEX --index-source 'matching builder env.chassis_idx'

This checks the saved frame using the production separating-axis calculation
and its 0.01 mm touching tolerance. Complete chassis contact forces and the
shape rest-offset pair are absent from the bank, so the force-dependent
near-contact branch and subsequent simulation ticks are explicitly not audited.
Exit codes: 0 = no saved geometric overlap; 1 = overlap; 2 = incomplete audit.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from utils.sgs import ConfigurationBank
from utils.table_collision import (box_separation, read_chassis_box,
                                   table_chassis_contact)


def resolved(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table_dimensions_from_source(path):
    """Read the literal create-box dimensions without importing Isaac Gym."""
    module = ast.parse(path.read_text())
    cls = next(node for node in module.body
               if isinstance(node, ast.ClassDef) and node.name == "B1Z1PickMulti")
    method = next(node for node in cls.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_create_envs")
    values = {}
    dimensions = None

    def number(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return float(node.value)
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id == "self" and node.attr in values):
            return values[node.attr]
        raise ValueError("table dimensions are not literal source values; provide verified geometry")

    for node in method.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name)
                and target.value.id == "self"):
            continue
        if target.attr == "table_dimz":
            values[target.attr] = number(node.value)
        if target.attr == "table_dims":
            call = node.value
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and isinstance(call.func.value, ast.Name) and call.func.value.id == "gymapi"
                    and call.func.attr == "Vec3" and len(call.args) == 3 and not call.keywords):
                raise ValueError("unrecognized table_dims construction")
            dimensions = [number(arg) for arg in call.args]
    if dimensions is None or not all(math.isfinite(v) and v > 0 for v in dimensions):
        raise ValueError("could not verify positive table box dimensions from production source")
    return dimensions


def chassis_world_pose(body, offset, rotation):
    q = body[:, 3:7]
    local = body.new_tensor(offset).expand(body.shape[0], -1)
    uv = torch.cross(q[:, :3], local, dim=-1)
    center = body[:, :3] + local + 2 * (q[:, 3:] * uv + torch.cross(q[:, :3], uv, dim=-1))
    delta = body.new_tensor(rotation).expand_as(q)
    xyz = (q[:, 3:] * delta[:, :3] + delta[:, 3:] * q[:, :3]
           + torch.cross(q[:, :3], delta[:, :3], dim=-1))
    w = q[:, 3:] * delta[:, 3:] - (q[:, :3] * delta[:, :3]).sum(-1, keepdim=True)
    return center, torch.cat((xyz, w), dim=-1)


def geometric_audit(tensors, family_names, chassis_index, half, offset, rotation, dimensions):
    bodies, table = tensors["rigid_body_state"], tensors["table_root"]
    if not 0 <= chassis_index < bodies.shape[1]:
        raise ValueError("chassis-body-index is outside the saved rigid_body_state")
    center, q = chassis_world_pose(bodies[:, chassis_index], offset, rotation)
    chassis_half, table_half = bodies.new_tensor(half), bodies.new_tensor(dimensions) / 2
    gap = box_separation(center, q, chassis_half, table[:, :3], table[:, 3:7], table_half)
    # Zero forces select ONLY the actual production geometric-overlap branch.
    # They do not claim that the original physical contact forces were zero.
    touching = table_chassis_contact(center, q, chassis_half, table[:, :3], table[:, 3:7],
                                     table_half, torch.zeros_like(center), 0.)
    if not bool(torch.isfinite(gap).all()):
        raise ValueError("nonfinite geometry audit result")
    families = tensors["family_id"]
    records = [{"config_id": row, "family": family_names[int(families[row])],
                "signed_sat_gap_m": float(gap[row]), "geometric_contact": bool(touching[row])}
               for row in range(len(families))]
    summary = {}
    for family, name in enumerate(family_names):
        mask = families == family
        ids = ((families == family) & touching).nonzero().flatten().tolist()
        summary[name] = {"count": int(mask.sum()), "geometric_contact_count": len(ids),
                         "geometric_contact_config_ids": ids,
                         "minimum_signed_sat_gap_m": float(gap[mask].min()) if bool(mask.any()) else None}
    failed = touching.nonzero().flatten().tolist()
    return {"audit_complete": True, "geometric_overlap_free": not failed,
            "completed_configs": len(records), "geometric_contact_config_ids": failed,
            "minimum_signed_sat_gap_m": float(gap.min()), "families": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--chassis-body-index", type=int,
                        help="verified env.chassis_idx; never guessed from bank poses")
    parser.add_argument("--index-source", default="explicit CLI value from matching env.chassis_idx")
    parser.add_argument("--asset-root", default="data/asset")
    parser.add_argument("--output", help="default: outputs/sgs_four_family_check_20261128/<bank>_collision_audit.json")
    args = parser.parse_args()
    bank_path = resolved(args.bank)
    output = (resolved(args.output) if args.output else ROOT / "outputs/sgs_four_family_check_20261128"
              / (bank_path.stem + "_collision_audit.json"))
    report = {"audit_type": "saved_frame_table_chassis_geometric_overlap",
              "created_utc": datetime.now(timezone.utc).isoformat(), "bank": str(bank_path),
              "audit_complete": False, "geometric_overlap_free": None,
              "full_training_collision_check_complete": False,
              "limitations": ["Bank omits chassis contact-force vectors and the verified shape rest-offset pair; "
                              "the force-dependent near-contact branch cannot be reconstructed.",
                              "No simulation or first-tick/trajectory collision check is performed.",
                              "Geometry is read from the current matching repository assets/source; "
                              "their SHA-256 hashes are recorded."],
              "method": {"production_module": "utils.table_collision",
                         "functions": ["box_separation", "table_chassis_contact"],
                         "geometric_touching_tolerance_m": 1e-5,
                         "gap_definition": "maximum signed separating-axis gap, not Euclidean closest distance",
                         "contact_function_force_argument": "zero to evaluate ONLY the geometric branch"}}
    try:
        bank = ConfigurationBank.load(bank_path, device="cpu")
        asset = resolved(args.asset_root) / bank.metadata["robot_asset"]
        source = ROOT / "envs/b1z1_pickmulti.py"
        geometry_source = ROOT / "utils/table_collision.py"
        half, offset, rotation = read_chassis_box(asset)
        dimensions = table_dimensions_from_source(source)
        urdf = ET.parse(asset).getroot()
        roots = {link.get("name") for link in urdf.findall("link")} - {
            joint.find("child").get("link") for joint in urdf.findall("joint")}
        report.update(bank_fingerprint=bank.fingerprint, num_configs=bank.num_configs,
                      family_names=list(bank.family_names),
                      geometry={"robot_asset": str(asset), "robot_asset_sha256": file_hash(asset),
                                "robot_root_links": sorted(roots), "chassis_link": "trunk",
                                "chassis_body_index": args.chassis_body_index,
                                "chassis_index_source": args.index_source if args.chassis_body_index is not None else None,
                                "chassis_half_extents_m": half, "chassis_local_offset_m": offset,
                                "chassis_local_rotation_xyzw": rotation, "table_dimensions_m": dimensions,
                                "table_dimensions_source": str(source), "source_sha256": file_hash(source),
                                "collision_module_sha256": file_hash(geometry_source)})
        if args.chassis_body_index is None:
            raise ValueError("Bank has no rigid-body names or verified trunk body index. Supply "
                             "--chassis-body-index from the matching environment's env.chassis_idx; "
                             "the separate robot root cannot substitute for this body.")
        report.update(geometric_audit(bank.tensors, bank.family_names, args.chassis_body_index,
                                      half, offset, rotation, dimensions))
        status = 0 if report["geometric_overlap_free"] else 1
    except (ValueError, KeyError, OSError, StopIteration) as error:
        report["error"] = str(error)
        status = 2
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report.get(key) for key in (
        "audit_complete", "geometric_overlap_free", "num_configs", "minimum_signed_sat_gap_m", "error")},
        ensure_ascii=False))
    print("Report:", output)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
