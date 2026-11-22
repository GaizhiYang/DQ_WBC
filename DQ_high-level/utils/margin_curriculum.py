"""Dynamic SGS from the initial interception margin M0 (main line A, curriculum).

Pure torch, no Isaac Gym: the environment and evaluate_margin_auc.py (gate D3)
share these functions.

- M0(c): the margin of configuration c at its initial state, with the true
  table motion. M0 >~ 0 is hard but feasible; M0 << 0 cannot succeed.
- Prior: SGS estimates each configuration's success rate from its history.
  A Beta-style prior p0 = sigmoid((M0 - delta) / temperature) with
  `strength` pseudo-episodes separates "hard" (sampled while it has few
  outcomes) from "impossible" (down-weighted before wasting episodes).
- delta decreases linearly over training, admitting harder configurations.
- Frontier: a full_task reset may speed its table up by the largest factor
  whose M0 still reaches delta, generating configurations beyond the bank.
"""

import math

import torch

from modules.reachability import (interception_margin, planar_carrier_future, query_frame,
                                  reach_state_features, grasp_positions_world, to_query_frame,
                                  yaw_from_quat)

CURRICULUM_MODES = ("disabled", "prior", "frontier")


def margin_delta(step, total_steps, start, end):
    """Linear schedule from start (step 0) to end (total_steps and after)."""
    if total_steps <= 0:
        return float(end)
    fraction = min(max(float(step) / float(total_steps), 0.0), 1.0)
    return float(start) + fraction * (float(end) - float(start))


def prior_success_rates(initial_margin, delta, temperature):
    return torch.sigmoid((initial_margin - delta) / temperature)


def bank_reach_states(state, gripper_index, num_gripper_dofs):
    """Reach-state features of bank initial states (commands are zero at reset)."""
    robot = state["robot_root"]
    body = state["rigid_body_state"][:, gripper_index]
    g = num_gripper_dofs
    dof_pos, dof_vel = state["dof_pos"], state["dof_vel"]
    commands = robot.new_zeros((robot.shape[0], 2))
    return reach_state_features(robot[:, :3], robot[:, 3:7], robot[:, 10:13], body[:, :3], body[:, 3:7],
                                dof_pos[:, -(6 + g):-g], dof_vel[:, -(6 + g):-g], dof_pos[:, -g:],
                                state["ee_goal_cart"], state["ee_goal_orn_rpy"], commands)


@torch.no_grad()
def initial_margins(model, state, grasp_in_object, horizons, remaining, gripper_index,
                    num_gripper_dofs, speed_scales=None, chunk=1024):
    """M0 [N, K] of bank-format states, one column per table speed scale.

    state: ConfigurationBank.batch(...) fields. grasp_in_object [C, 4, 4] or
    [N, C, 4, 4]. horizons [H] seconds (first 0). remaining: seconds left at
    the initial state. speed_scales [K] multiplies the table velocity and yaw
    rate together, which keeps a circular path's radius (default: [1]).
    """
    num = state["robot_root"].shape[0]
    device = state["robot_root"].device
    scales = (torch.ones(1, device=device) if speed_scales is None
              else torch.as_tensor(speed_scales, dtype=torch.float32, device=device))
    horizons = torch.as_tensor(horizons, dtype=torch.float32, device=device)
    if grasp_in_object.dim() == 3:
        grasp_in_object = grasp_in_object.unsqueeze(0).expand(num, -1, -1, -1)
    rotation = torch.where(state["rotation_enabled"], state["table_yaw_rate"],
                           torch.zeros_like(state["table_yaw_rate"]))
    out = torch.empty((num, scales.numel()), device=device)
    for start in range(0, num, chunk):
        part = {name: value[start:start + chunk] for name, value in state.items()
                if isinstance(value, torch.Tensor) and value.dim() > 0 and value.shape[0] == num}
        n = part["robot_root"].shape[0]
        features = bank_reach_states(part, gripper_index, num_gripper_dofs)
        origin, yaw = query_frame(part["robot_root"][:, :3], part["robot_root"][:, 3:7])
        obj, table = part["object_root"], part["table_root"]
        table_yaw = yaw_from_quat(table[:, 3:7])
        left = torch.full((n,), float(remaining), device=device)
        for k, scale in enumerate(scales):
            velocity = torch.stack((torch.zeros_like(part["table_vy"]), part["table_vy"] * scale), dim=-1)
            pos, quat = planar_carrier_future(obj[:, :3], obj[:, 3:7], table[:, :3], table_yaw, velocity,
                                              rotation[start:start + n] * scale, horizons)
            goals = to_query_frame(grasp_positions_world(pos, quat, grasp_in_object[start:start + n]),
                                   origin, yaw)
            h, c = goals.shape[1], goals.shape[2]
            times = model.evaluate(features, goals.reshape(n, h * c, 3)).reshape(n, h, c)
            out[start:start + n, k] = interception_margin(times, horizons, left)["margin"]
    return out


def choose_frontier(margins, delta):
    """Index [N] of the largest scale whose margin reaches delta; 0 if none does."""
    ok = margins >= delta
    index = torch.arange(margins.shape[1], device=margins.device).expand_as(margins)
    return torch.where(ok, index, torch.zeros_like(index)).amax(dim=1)


def roc_auc(scores, labels):
    """Area under the ROC curve with ties counted as one half; None without both classes."""
    scores = torch.as_tensor(scores, dtype=torch.float64).flatten()
    labels = torch.as_tensor(labels).flatten().bool()
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    ranks = torch.empty_like(scores)
    # Average ranks over ties (1-based).
    unique, inverse, counts = torch.unique_consecutive(sorted_scores, return_inverse=True,
                                                       return_counts=True)
    ends = torch.cumsum(counts, 0).to(torch.float64)
    average = ends - (counts.to(torch.float64) - 1) / 2
    ranks[order] = average[inverse]
    rank_sum = ranks[labels].sum().item()
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def spearman(x, y):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    y = torch.as_tensor(y, dtype=torch.float64).flatten()
    if x.numel() < 3:
        return None

    def rank(v):
        order = torch.argsort(v)
        r = torch.empty_like(v)
        _, inverse, counts = torch.unique_consecutive(v[order], return_inverse=True, return_counts=True)
        ends = torch.cumsum(counts, 0).to(torch.float64)
        r[order] = (ends - (counts.to(torch.float64) - 1) / 2)[inverse]
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return torch.corrcoef(torch.stack((rx, ry)))[0, 1].item()


def validate_curriculum_mode(mode):
    if mode not in CURRICULUM_MODES:
        raise ValueError(f"env.margin.curriculum must be one of {CURRICULUM_MODES}")
    return mode


def finite_positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)
