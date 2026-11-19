import torch
import torch.nn as nn
import numpy as np

from modules.grasp_tokens import split_grasp_tokens, token_horizons


def teacher_gfm_kwargs(env):
    """Teacher Policy/Value layout kwargs from the environment.

    Use the saved GFM layout; configs predating this flag keep the old Query.
    The critic-only episode-progress index is added only when that channel exists.
    The grasp-token layout (direction 8) is added only when tokens are enabled.
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
    token_layout = getattr(env, "grasp_token_layout", None)
    if token_layout is not None:
        kwargs["grasp_token_layout"] = token_layout
    return kwargs


class PredictAttentionSelector(nn.Module):
    # Prefix of the rollout diagnostics; the critic uses "GFM critic".
    attention_tag = "GFM"

    def __init__(self, d_model=64, obj_feat_dim=128, future_pose_dim=0, token_layout=None, enabled=True):
        super().__init__()
        if future_pose_dim < 0 or future_pose_dim % 6:
            raise ValueError("GFM future_pose_dim must be 6 * horizons (poses only)")
        self.future_pose_dim = future_pose_dim
        self.token_layout = token_layout
        self.gfm_enabled = enabled
        self.last_attention = None
        # Rollout diagnostics (see track_attention); the trainer sets the tracker.
        self.attention_tracker = None
        self._previous_attention_argmax = None
        if not enabled:
            # Ablation: the model ignores the grasp block and has no GFM parameters.
            self.selected_dim = 0
            return
        if token_layout is None:
            self.key_proj = nn.Linear(6, d_model)
            self.value_proj = nn.Linear(6, d_model)
            self.query_proj = nn.Linear(obj_feat_dim + 6 + future_pose_dim, d_model)
            self.output_proj = nn.Linear(d_model, 6)  # 输出最终抓取点
            self.selected_dim = 6
        else:
            # G1: tokens keep their full d_model embedding instead of a 6-dim pose.
            # The token's time offset h is one more input feature of Key/Value.
            self.key_proj = nn.Linear(token_layout.feature_dim + 1, d_model)
            self.value_proj = nn.Linear(token_layout.feature_dim + 1, d_model)
            self.query_proj = nn.Linear(obj_feat_dim + 6 + future_pose_dim, d_model)
            self.register_buffer("token_h", token_horizons(token_layout)[:, None], persistent=False)
            # "shared" normalization: the embedding enters the MLP at unit scale.
            if token_layout.normalization == "shared":
                self.output_norm = nn.LayerNorm(d_model)
            self.selected_dim = d_model

    def grasp_attention_input(self, states):
        """Key/Value input: the 30 old [pos, euler] candidates, or the raw token block.

        Uses the owning model's ``notrain_obs_num``, ``num_actions`` and ``cube_object_num``.
        """
        grasp_block = states[..., -self.notrain_obs_num - self.num_actions:-self.num_actions]
        if self.token_layout is not None:
            return grasp_block
        num = self.cube_object_num
        positions = grasp_block[..., :3 * num].view(-1, num, 3)
        rotations = grasp_block[..., 3 * num:].view(-1, num, 3)
        return torch.cat([positions, rotations], dim=-1)

    def attention_forward(self, grasps, obj_feat, pose, future_pose_obs=None):
        context = [obj_feat, pose]
        if self.future_pose_dim:
            if future_pose_obs is None or future_pose_obs.shape[-1] != self.future_pose_dim:
                raise ValueError("GFM requires the configured future poses without a validity channel")
            # These channels have the same normalization as the current pose.
            context.append(future_pose_obs)
        context = torch.cat(context, dim=-1)  # 146 with two future poses
        Q = self.query_proj(context).unsqueeze(1)      # [B, 1, d_model]
        if self.token_layout is None:
            K = self.key_proj(grasps)                      # [B, N, d_model]
            V = self.value_proj(grasps)                    # [B, N, d_model]
            attn = torch.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(K.size(-1)), dim=-1)  # [B, 1, N]
            self.last_attention = attn.squeeze(1).detach()
            weighted = attn @ V      # [B, 1, d_model]
            selected_grasp = self.output_proj(weighted.squeeze(1))  # [B, 6]
            return selected_grasp
        # Token mode: ``grasps`` is the raw token block (it bypasses the state scaler).
        tokens, future_active = split_grasp_tokens(grasps, self.token_layout)
        h = self.token_h.to(tokens.dtype).expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((tokens, h), dim=-1)                                   # [B, H*N, F+1]
        K = self.key_proj(tokens)
        V = self.value_proj(tokens)
        logits = (Q @ K.transpose(-2, -1)) / np.sqrt(K.size(-1))                  # [B, 1, H*N]
        # Without a valid forecast, or once the object is lifted, only h = 0 tokens remain.
        future = self.token_h[:, 0] > 0
        masked = future[None, :] & (future_active[:, None] < 0.5)
        logits = logits.masked_fill(masked[:, None, :], float("-inf"))
        attn = torch.softmax(logits, dim=-1)
        self.last_attention = attn.squeeze(1).detach()
        embedding = (attn @ V).squeeze(1)                                         # [B, d_model]
        norm = getattr(self, "output_norm", None)
        return embedding if norm is None else norm(embedding)

    def track_attention(self):
        """Log attention entropy, argmax switches and (with future tokens) the mass on h > 0.

        Only rollout passes are logged (no gradients). The critic's bootstrap value at
        each update is also a no-grad pass, so its switch rate counts one repeated
        state every rollout (about 4 % of the pairs with 24 rollout steps).
        """
        attn = self.last_attention
        if self.attention_tracker is None or attn is None or torch.is_grad_enabled():
            return
        tag = self.attention_tag
        entropy = -(attn * torch.log(attn.clamp_min(1e-12))).sum(-1).mean()
        argmax = attn.argmax(-1)
        previous = self._previous_attention_argmax
        if previous is not None and previous.shape == argmax.shape:
            self.attention_tracker(f"{tag} / argmax switch rate", (argmax != previous).float().mean().item())
        self._previous_attention_argmax = argmax
        self.attention_tracker(f"{tag} / attention entropy", entropy.item())
        if self.token_layout is not None and self.token_layout.num_horizons > 1:
            future_mass = attn[:, self.token_h[:, 0] > 0].sum(-1).mean()
            self.attention_tracker(f"{tag} / future token mass", future_mass.item())
