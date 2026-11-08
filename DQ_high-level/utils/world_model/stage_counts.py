"""Observed high-step task events, independent of model fitting and sampling.

These counters use captured real states, including their rigid-body EE cache.
They never infer contact, call FK, consume RNG, or contribute a training loss.
"""

import torch

from utils.pick_success import lifted_grasp_mask


STAGES = ("lifted_proxy", "held_proxy", "completed")


@torch.no_grad()
def captured_stage_flags(adapter, state, action, context):
    # Bypass adapter.get('ee'): fixed8 derives imagined EE geometry using FK,
    # whereas these records already contain the real rigid-body measurement.
    get = lambda name: state[:, adapter.slices[name]]
    obj, ee = get("object"), get("ee")
    lifted = lifted_grasp_mask(
        obj[:, 2], get("table_height")[:, 0], context[:, 2],
        torch.linalg.vector_norm(obj[:, :3] - ee[:, :3], dim=-1),
        action[:, 6], adapter.lift_threshold)
    return {"lifted_proxy": lifted, "held_proxy": get("hold")[:, 0] > 0,
            "completed": get("completed")[:, 0] > .5}


class RealStageTransitionCounts:
    """GPU int64 counters by SGS reset family, not by inferred execution stage.

    Entries/exits compare a single valid high PRE/POST pair, never adjacent
    replay rows across reset. A recrossed threshold counts again. A family
    label describes the episode's reset distribution, not its current stage.
    """

    columns = ("valid_transitions", "failure_terminals") + tuple(
        name + suffix for name in STAGES for suffix in ("_eligible", "_entries", "_exits"))

    def __init__(self, family_names, device):
        names = tuple(family_names)
        if not names or len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Stage counters require distinct named SGS reset families")
        self.family_names = names
        self.labels = names + ("unknown_family",)
        self.counts = torch.zeros(len(self.labels), len(self.columns), dtype=torch.int64, device=device)

    def reset(self):
        self.counts.zero_()

    @torch.no_grad()
    def update(self, family, valid, before, after, failure):
        family, valid = family.reshape(-1).long(), valid.reshape(-1).bool()
        rows = [valid, valid & failure.reshape(-1).bool()]
        for name in STAGES:
            pre, post = before[name].reshape(-1).bool(), after[name].reshape(-1).bool()
            rows.extend((valid & ~pre, valid & ~pre & post, valid & pre & ~post))
        values = torch.stack(rows, dim=-1).long()
        group = torch.where((family >= 0) & (family < len(self.family_names)), family,
                            torch.full_like(family, len(self.family_names)))
        self.counts.scatter_add_(0, group[:, None].expand_as(values), values)

    def state_dict(self):
        return {
            "schema": "dq-real-high-stage-events-v1",
            "scope": "selected_replay_environments_train_and_holdout_since_initialization_or_load",
            "family_names": list(self.family_names), "group_labels": list(self.labels),
            "columns": list(self.columns), "counts": self.counts.detach().clone(),
            "definitions": {
                "lifted_proxy": "existing height/proximity/closing-opening rule on captured rigid bodies; not contact truth",
                "held_proxy": "captured consecutive holding counter > 0; not contact truth",
                "completed": "captured completion latch or actual success flag; count false-to-true only",
                "failure_terminals": "actual terminated and not success/completed/truncated",
                "entry_rate": "entries / valid transitions whose PRE flag is false; not episode success rate",
                "failure_rate": "failure terminal events / valid high transitions; not episode failure rate",
                "recrossings": "each valid threshold re-entry counts, including repeated entries in one episode",
            },
        }

    def metrics(self):
        # One small transfer at a PPO boundary; collection performs no host sync.
        values = self.counts.detach().cpu().tolist()
        values.append([sum(row[column] for row in values) for column in range(len(self.columns))])
        result = {}
        for label, row in zip(self.labels + ("all",), values):
            prefix = "real_stage/" + label + "/"
            result.update({prefix + name: float(count) for name, count in zip(self.columns, row)})
            result[prefix + "failure_rate"] = row[1] / max(row[0], 1)
            for index, name in enumerate(STAGES):
                eligible, entries = row[2 + 3*index:4 + 3*index]
                result[prefix + name + "_entry_rate"] = entries / max(eligible, 1)
        return result
