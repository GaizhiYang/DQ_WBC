"""Offline reset-bank sampling statistics, without simulator dependencies.

Rejection counters describe proposed states, while coverage describes the states
actually saved in the bank. Keeping these separate makes quota truncation and
IK/contact filtering visible. Inputs may be on CPU or CUDA; only this offline
diagnostic path transfers compact tensors to the CPU.
"""

from collections.abc import Mapping
from copy import deepcopy
import math

import torch


def _integer_vector(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(name + " must be a one-dimensional tensor")
    if value.dtype == torch.bool or value.is_complex():
        raise ValueError(name + " must contain integer values")
    detached = value.detach().cpu()
    if detached.is_floating_point() and not bool(
            (torch.isfinite(detached) & (detached == detached.round())).all()):
        raise ValueError(name + " must contain finite integer values")
    return detached.to(torch.long)


def _counts():
    return {
        "proposed": 0, "valid": 0, "stored": 0, "invalid": 0,
        "valid_not_stored": 0,
        "rejection_counts": {}, "first_rejection_counts": {},
    }


def _add_counts(destination, mask, valid, stored, failures, first_failures):
    proposed = int(mask.sum())
    passed = int((valid & mask).sum())
    saved = int((stored & mask).sum())
    destination["proposed"] += proposed
    destination["valid"] += passed
    destination["stored"] += saved
    destination["invalid"] += proposed - passed
    destination["valid_not_stored"] += passed - saved
    for key, values in (("rejection_counts", failures),
                        ("first_rejection_counts", first_failures)):
        for name, failure in values.items():
            destination[key][name] = destination[key].get(name, 0) + int((failure & mask).sum())


def _distribution(values):
    """Ten observed-range bins; explicitly account for empty/constant values."""
    values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    finite = values[torch.isfinite(values)]
    result = {
        "count": values.numel(), "finite_count": finite.numel(),
        "nonfinite_count": values.numel() - finite.numel(),
        "min": None, "max": None, "mean": None,
        "histogram": {"edges": [], "counts": [], "range": "observed_values"},
    }
    if not finite.numel():
        return result
    minimum, maximum = float(finite.min()), float(finite.max())
    low, high = minimum, maximum
    if low == high:
        # A display interval is needed for a histogram of a fixed parameter.
        # This does not expand the construction/acceptance range.
        margin = max(abs(low) * 1e-6, 1e-6)
        low, high = low - margin, high + margin
        result["histogram"]["range"] = "constant_value_display_interval"
    edges = torch.linspace(low, high, 11, dtype=torch.float64)
    counts, _ = torch.histogram(finite, bins=edges)
    result.update(min=minimum, max=maximum, mean=float(finite.mean()))
    result["histogram"].update(edges=edges.tolist(), counts=counts.to(torch.long).tolist())
    return result


class BuildAudit:
    """Count proposed/valid/stored states and report actual stored coverage.

    ``checks`` passed to :meth:`record` is an ordered mapping of independent
    acceptance conditions, with True meaning pass. The ordering must stay fixed
    within a family so mutually exclusive first-rejection counts are comparable.
    A candidate ID of -1 denotes the full-task family, without a grasp candidate.
    """

    def __init__(self, family_names):
        self.family_names = tuple(family_names)
        if (not self.family_names or len(set(self.family_names)) != len(self.family_names)
                or any(not isinstance(name, str) or not name for name in self.family_names)):
            raise ValueError("family_names must contain distinct nonempty names")
        self._families = [_counts() for _ in self.family_names]
        self._candidates = [{} for _ in self.family_names]
        self._orders = [None for _ in self.family_names]

    def record(self, family, candidate_ids, checks, stored_indices):
        if type(family) is not int or not 0 <= family < len(self.family_names):
            raise ValueError("family must identify a configured family")
        ids = _integer_vector(candidate_ids, "candidate_ids")
        if bool((ids < -1).any()):
            raise ValueError("candidate_ids must be nonnegative or -1")
        if not isinstance(checks, Mapping) or not checks:
            raise ValueError("checks must be a nonempty ordered mapping")
        order = tuple(checks)
        if any(not isinstance(name, str) or not name for name in order):
            raise ValueError("check names must be nonempty strings")
        if self._orders[family] is not None and self._orders[family] != order:
            raise ValueError("check order must remain constant within each family")
        valid = torch.ones(ids.numel(), dtype=torch.bool)
        failures, first_failures = {}, {}
        for name, check in checks.items():
            if (not isinstance(check, torch.Tensor) or check.dtype != torch.bool
                    or check.shape != ids.shape):
                raise ValueError("each check must be a Boolean vector matching candidate_ids")
            condition = check.detach().cpu()
            failures[name] = ~condition
            first_failures[name] = valid & ~condition
            valid = valid & condition
        selected = _integer_vector(stored_indices, "stored_indices")
        if selected.numel() and (bool(((selected < 0) | (selected >= ids.numel())).any())
                                 or selected.unique().numel() != selected.numel()):
            raise ValueError("stored_indices must contain unique in-range row indices")
        if selected.numel() and not bool(valid[selected].all()):
            raise ValueError("stored_indices must refer only to valid proposals")
        stored = torch.zeros_like(valid)
        stored[selected] = True

        # All validation above precedes mutation, including quota/index errors.
        self._orders[family] = order
        _add_counts(self._families[family], torch.ones_like(valid), valid, stored,
                    failures, first_failures)
        for candidate in ids[ids >= 0].unique().tolist():
            destination = self._candidates[family].setdefault(candidate, _counts())
            _add_counts(destination, ids == candidate, valid, stored, failures, first_failures)

    def summary(self, tensors=None):
        """Return JSON-safe counters, plus coverage when bank tensors exist.

        ``tensors=None`` is useful for reporting an exhausted build that has no
        complete bank. Counts in ``candidates`` include all proposed IDs, even
        those never accepted. ``stored_candidate_counts`` uses actual bank rows.
        """
        families = {}
        for index, name in enumerate(self.family_names):
            row = deepcopy(self._families[index])
            row["check_order"] = list(self._orders[index] or ())
            row["acceptance_rate"] = row["valid"] / row["proposed"] if row["proposed"] else None
            row["candidates"] = [
                dict(candidate_id=candidate, **deepcopy(counts))
                for candidate, counts in sorted(self._candidates[index].items())]
            families[name] = row
        result = {
            "version": "sgs-build-audit-v1",
            "notes": {
                "valid": "A proposal passing all checks before the family quota is applied.",
                "stored": "A valid proposal actually selected for the saved bank.",
                "rejection_counts": "Independent failed checks; one proposal may be counted more than once.",
                "first_rejection_counts": "Mutually exclusive first failed check in check_order; totals equal invalid.",
                "coverage": "Distributions of actual stored states, not the configured target ranges.",
                "histograms": "Ten bins across observed values; constant values use a small display interval.",
            },
            "totals": {key: sum(row[key] for row in self._families)
                       for key in ("proposed", "valid", "stored", "invalid", "valid_not_stored")},
            "families": families,
        }
        if tensors is not None:
            self._add_coverage(result, tensors)
        return result

    def _add_coverage(self, result, tensors):
        if not isinstance(tensors, Mapping):
            raise ValueError("bank tensors must be a mapping")
        ids = _integer_vector(tensors["family_id"], "family_id")
        if bool(((ids < 0) | (ids >= len(self.family_names))).any()):
            raise ValueError("bank family_id outside configured families")
        count = ids.numel()

        def read(name, width=None):
            value = tensors[name]
            expected = (count,) if width is None else (count, width)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                raise ValueError("invalid coverage tensor shape: " + name)
            return value.detach().to(device="cpu", dtype=torch.float64)

        robot, obj = read("robot_root", 13), read("object_root", 13)
        delta = obj[:, :2] - robot[:, :2]
        x, y, z, w = robot[:, 3:7].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        heading = yaw - torch.atan2(delta[:, 1], delta[:, 0])
        heading = torch.remainder(heading + math.pi, 2 * math.pi) - math.pi
        metrics = {
            "base_x_m": delta[:, 0], "base_y_m": delta[:, 1],
            "base_distance_xy_m": delta.norm(dim=1),
            "base_heading_offset_deg": torch.rad2deg(heading),
            "table_height_m": read("table_height"),
            "gripper_opening": read("gripper_opening"),
            "object_yaw_offset_deg": read("object_yaw_offset_deg"),
            "validation_peak_contact_force_n": read("validation_peak_contact_force_n"),
            "validation_max_object_displacement_m": read("validation_max_object_displacement_m"),
        }
        if "lift_height" in tensors:
            metrics["lift_height"] = read("lift_height")
        # These records describe a near-goal validation window. Placeholder
        # zeros in the other families are not evidence of a verified grasp.
        grasp_metrics = {name: read(name) for name in (
            "near_goal_relative_slip_m", "near_goal_relative_rotation_deg",
            "near_goal_contact_fraction", "near_goal_object_speed_m_s",
            "near_goal_restore_verified",
        ) if name in tensors}
        offsets = read("candidate_local_offset", 3)
        rotations = read("ee_rotation_offset_deg", 3)
        candidate_ids = (_integer_vector(tensors["grasp_candidate_index"], "grasp_candidate_index")
                         if "grasp_candidate_index" in tensors else None)
        if candidate_ids is not None and candidate_ids.shape != ids.shape:
            raise ValueError("grasp_candidate_index must match family_id")
        for index, name in enumerate(self.family_names):
            mask = ids == index
            row = result["families"][name]
            row["bank_count"] = int(mask.sum())
            row["coverage"] = {key: _distribution(values[mask]) for key, values in metrics.items()}
            if name == "near_goal":
                row["coverage"].update({key: _distribution(values[mask])
                                        for key, values in grasp_metrics.items()})
            if name != "full_task":
                row["coverage"].update({
                    "ee_candidate_distance_m": _distribution(offsets[mask].norm(dim=1)),
                    **{"candidate_local_{}_m".format(axis): _distribution(offsets[mask, j])
                       for j, axis in enumerate(("x", "y", "z"))},
                    **{"ee_rotation_{}_deg".format(axis): _distribution(rotations[mask, j])
                       for j, axis in enumerate(("roll", "pitch", "yaw"))},
                })
            if candidate_ids is not None:
                chosen = candidate_ids[mask & (candidate_ids >= 0)]
                keys, counts = chosen.unique(return_counts=True)
                stored_counts = [{"candidate_id": int(key), "count": int(value)}
                                 for key, value in zip(keys.tolist(), counts.tolist())]
                row["stored_candidate_count_source"] = "bank_grasp_candidate_index"
            else:
                stored_counts = [{"candidate_id": candidate, "count": values["stored"]}
                                 for candidate, values in sorted(self._candidates[index].items())
                                 if values["stored"]]
                row["stored_candidate_count_source"] = "recorded_stored_indices"
            row["stored_candidate_counts"] = stored_counts
            row["stored_candidate_count"] = len(stored_counts)
