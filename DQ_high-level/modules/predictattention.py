import torch
import torch.nn as nn
import numpy as np


def teacher_gfm_kwargs(env):
    """Use the saved GFM layout; configs predating this flag keep the old Query."""
    enabled = env.cfg["env"].get("posePrediction", {}).get("gfmFuturePoses", False)
    if not isinstance(enabled, bool):
        raise ValueError("env.posePrediction.gfmFuturePoses must be a boolean")
    return {
        "num_future_pose_obs": getattr(env, "num_future_pose_obs", 0) if enabled else 0,
        "num_tsil_obs": getattr(env, "num_tsil_obs", 0),
    }


class PredictAttentionSelector(nn.Module):
    def __init__(self, d_model=64, obj_feat_dim=128, future_pose_dim=0):
        super().__init__()
        if future_pose_dim < 0 or future_pose_dim % 6:
            raise ValueError("GFM future_pose_dim must be 6 * horizons (poses only)")
        self.future_pose_dim = future_pose_dim
        self.key_proj = nn.Linear(6, d_model)
        self.value_proj = nn.Linear(6, d_model)
        self.query_proj = nn.Linear(obj_feat_dim + 6 + future_pose_dim, d_model)
        self.output_proj = nn.Linear(d_model, 6)  # 输出最终抓取点

    def attention_forward(self, grasps, obj_feat, pose, future_pose_obs=None):
        context = [obj_feat, pose]
        if self.future_pose_dim:
            if future_pose_obs is None or future_pose_obs.shape[-1] != self.future_pose_dim:
                raise ValueError("GFM requires the configured future poses without a validity channel")
            # These channels have the same normalization as the current pose.
            context.append(future_pose_obs)
        context = torch.cat(context, dim=-1)  # 146 with two future poses
        Q = self.query_proj(context).unsqueeze(1)      # [B, 1, d_model]
        K = self.key_proj(grasps)                      # [B, N, d_model]
        V = self.value_proj(grasps)                    # [B, N, d_model]

        attn = torch.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(K.size(-1)), dim=-1)  # [B, 1, N]
        weighted = attn @ V      # [B, 1, d_model]
        selected_grasp = self.output_proj(weighted.squeeze(1))  # [B, 6]
        return selected_grasp
