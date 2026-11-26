import torch
import torch.nn as nn
import numpy as np


def teacher_gfm_kwargs(env):
    """Teacher Policy/Value layout kwargs from the environment.

    Use the saved GFM layout; configs predating this flag keep the old Query.
    The critic-only episode-progress index is added only when that channel exists.
    """
    enabled = env.cfg["env"].get("posePrediction", {}).get("gfmFuturePoses", False)
    if not isinstance(enabled, bool):
        raise ValueError("env.posePrediction.gfmFuturePoses must be a boolean")
    kwargs = {
        "num_future_pose_obs": getattr(env, "num_future_pose_obs", 0) if enabled else 0,
        "num_tsil_obs": getattr(env, "num_tsil_obs", 0),
    }
    critic_time_index = getattr(env, "critic_time_obs_index", None)
    if critic_time_index is not None:
        kwargs["critic_time_index"] = critic_time_index
    # Critic-only hidden reward state (--critic_reward_state), likewise removed by the actor.
    critic_state_index = getattr(env, "critic_state_obs_index", None)
    if critic_state_index is not None:
        kwargs["critic_state_index"] = critic_state_index
        kwargs["critic_state_width"] = env.num_critic_state_obs
    history_frames = getattr(env, "actor_history_frames", 1)
    if history_frames != 1:
        kwargs["actor_history_frames"] = history_frames
    critic_frames = getattr(env, "critic_history_frames", 1)
    if critic_frames != 1:
        kwargs["critic_history_frames"] = critic_frames
    # Main line A: actor-visible margin block (scalars, then per-horizon tokens).
    margin_index = getattr(env, "margin_obs_index", None)
    if margin_index is not None:
        from modules.reachability import MARGIN_SCALAR_NAMES, MARGIN_TOKEN_DIM
        kwargs.update(margin_index=margin_index, margin_tokens=env.num_margin_tokens,
                      margin_token_dim=MARGIN_TOKEN_DIM, margin_scalars=len(MARGIN_SCALAR_NAMES))
    return kwargs


class PredictAttentionSelector(nn.Module):
    def __init__(self, d_model=64, obj_feat_dim=128, future_pose_dim=0, token_dim=0, num_tokens=0):
        super().__init__()
        if future_pose_dim < 0 or future_pose_dim % 6:
            raise ValueError("GFM future_pose_dim must be 6 * horizons (poses only)")
        if (token_dim == 0) != (num_tokens == 0) or token_dim < 0 or num_tokens < 0:
            raise ValueError("GFM token_dim and num_tokens must both be positive or both zero")
        self.future_pose_dim = future_pose_dim
        self.token_dim, self.num_tokens = token_dim, num_tokens
        self.key_proj = nn.Linear(6, d_model)
        self.value_proj = nn.Linear(6, d_model)
        self.query_proj = nn.Linear(obj_feat_dim + 6 + future_pose_dim, d_model)
        self.output_proj = nn.Linear(d_model, 6)  # 输出最终抓取点
        if token_dim:
            # Spatiotemporal tokens (main line A): one per margin horizon. Constant
            # channels such as the horizon vanish in the observation standardizer,
            # so a learned per-slot embedding tells attention each token's time.
            self.token_key_proj = nn.Linear(token_dim, d_model)
            self.token_value_proj = nn.Linear(token_dim, d_model)
            self.token_slot = nn.Parameter(torch.zeros(num_tokens, d_model))

    def attention_forward(self, grasps, obj_feat, pose, future_pose_obs=None, tokens=None):
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
        if self.token_dim:
            if tokens is None or tuple(tokens.shape[-2:]) != (self.num_tokens, self.token_dim):
                raise ValueError("GFM requires the configured margin tokens [B, H, token_dim]")
            # Attend jointly over the 30 current candidates and the H future tokens.
            K = torch.cat((K, self.token_key_proj(tokens) + self.token_slot), dim=1)
            V = torch.cat((V, self.token_value_proj(tokens) + self.token_slot), dim=1)

        attn = torch.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(K.size(-1)), dim=-1)  # [B, 1, N]
        weighted = attn @ V      # [B, 1, d_model]
        selected_grasp = self.output_proj(weighted.squeeze(1))  # [B, 6]
        return selected_grasp
