"""Gaussian teacher policy with a continuous, sigmoid-bounded gripper opening.

``compute`` returns the nine latent Gaussian means and log standard deviations.
Only coordinate 6 is transformed: the action stored by PPO and executed by the
environment is ``opening = sigmoid(latent_gripper)``. Its likelihood includes
the inverse transform and Jacobian in the default public API. FinitePPO can
opt into retaining the sampled latent and evaluating Gaussian likelihoods for
replay, avoiding the lossy inverse of saturated floating-point openings.
The other eight coordinates retain the existing Gaussian action semantics.
"""

import math

import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Normal

from skrl.models.torch import GaussianMixin


class ContinuousGripperGaussianMixin(GaussianMixin):
    """A local skrl extension; no changes to the vendored Gaussian mixin.

    ``distribution()`` retains skrl's latent Normal distribution for diagnostics.
    Use ``act`` for executed actions and their transformed likelihoods. Entropy
    is returned per coordinate; the gripper term uses deterministic 64-point
    Gauss-Hermite quadrature for E[log(sigmoid'(Z))]. This avoids injecting extra
    sampling noise and includes the transform when entropy regularization is on.

    At floating-point saturation, openings use the same interior endpoint guard
    as PyTorch's sigmoid transform. Default rollout and replay log probabilities
    are evaluated from the stored opening, so an unchanged policy has ratio one
    even when sigmoid inversion loses precision. Endpoint densities are thus a
    finite-precision approximation to the continuous logistic-normal density.
    With ``inputs['latent_ppo']``, the extra ``latent_actions`` and
    ``latent_log_prob`` outputs expose the original Gaussian sample/density.
    Replay must supply ``taken_latent_actions`` alongside ``taken_actions``.
    Its public log probability is the continuous transformed density at that
    actual latent, using a stable sigmoid Jacobian. It does not reconstruct a
    latent from rounded or endpoint-clipped openings. The action mapping,
    default baseline interface and deterministic BPTT mapping are unchanged.
    """

    gripper_action_semantics = "continuous_opening_v1"
    gripper_index = 6
    supports_latent_ppo = True

    def __init__(self, clip_actions=False, clip_log_std=True, min_log_std=-20,
                 max_log_std=2, reduction="sum", role="", transform_func=None,
                 deterministic=False):
        if transform_func is not None:
            raise ValueError("Continuous gripper policy does not support an additional tanh transform")
        if clip_actions:
            raise ValueError("Continuous gripper Gaussian likelihood requires clip_actions=False")
        if self.num_actions != 9:
            raise ValueError("Continuous gripper teacher policy requires nine action coordinates")
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std,
                               reduction=reduction, role=role, transform_func=None,
                               deterministic=deterministic)
        if not hasattr(self, "_gripper_action_semantics_version"):
            self.register_buffer("_gripper_action_semantics_version",
                                 torch.tensor(1, dtype=torch.int64, device=self.device))
            nodes, weights = np.polynomial.hermite.hermgauss(64)
            self.register_buffer("_gripper_entropy_nodes",
                                 torch.as_tensor(nodes, device=self.device), persistent=False)
            self.register_buffer("_gripper_entropy_weights",
                                 torch.as_tensor(weights / math.sqrt(math.pi), device=self.device),
                                 persistent=False)

    @staticmethod
    def _interior_opening(opening):
        # Use dtype-specific limits so log/log1p remain finite on CPU and CUDA.
        limits = torch.finfo(opening.dtype)
        return opening.clamp(min=limits.tiny, max=1. - limits.eps)

    def mean_to_action(self, mean):
        """Map latent means to deterministic execution, retaining input gradients.

        This is sigmoid(mu), the logistic-normal median, not E[sigmoid(Z)].
        BPTT intentionally uses this map and never consults standard deviation.
        """
        opening = self._interior_opening(torch.sigmoid(mean[..., 6:7]))
        return torch.cat((mean[..., :6], opening, mean[..., 7:]), dim=-1)

    def _role_value(self, name, role):
        options = getattr(self, name)
        return options[role] if role in options else options[""]

    def act(self, inputs, role=""):
        mean, log_std, outputs = self.compute(inputs, role)
        if self._role_value("_g_clip_log_std", role):
            log_std = log_std.clamp(self._role_value("_g_log_std_min", role),
                                    self._role_value("_g_log_std_max", role))
        self._g_log_std[role] = log_std
        self._g_num_samples[role] = mean.shape[0]
        distribution = Normal(mean, log_std.exp(), validate_args=False)
        self._g_distribution[role] = distribution
        latent = mean if self.deterministic else distribution.rsample()
        actions = self.mean_to_action(latent)

        reduction = self._role_value("_g_reduction", role)
        outputs = dict(outputs)
        outputs["mean_actions"] = self.mean_to_action(mean)
        outputs["latent_mean_actions"] = mean
        if inputs.get("latent_ppo", False):
            # Opt-in training protocol. A saturated float32 opening cannot be
            # inverted to recover the sampled Gaussian latent. Retain that
            # actual sample and evaluate its density directly during PPO.
            if "taken_actions" in inputs and "taken_latent_actions" not in inputs:
                raise ValueError("Latent PPO replay requires taken_latent_actions")
            evaluated_latent = inputs.get("taken_latent_actions", latent)
            if evaluated_latent.shape != mean.shape:
                raise ValueError("Latent PPO actions must match the nine-dimensional policy output")
            latent_coordinates = distribution.log_prob(evaluated_latent)
            outputs["latent_actions"] = evaluated_latent
            outputs["latent_log_prob"] = (latent_coordinates if reduction is None else
                reduction(latent_coordinates, dim=-1, keepdim=True))
            z = evaluated_latent[..., 6:7]
            log_jacobian = -F.softplus(-z) - F.softplus(z)
            outputs["latent_log_abs_det_jacobian"] = log_jacobian
            # Compute the transformed density at the SAME sampled/replayed
            # latent. Inverting a saturated opening here can place a Gaussian
            # evaluation arbitrarily far from its mean and overflow, even
            # though the actual sampled Gaussian log probability is finite.
            coordinate_log_prob = torch.cat((latent_coordinates[..., :6],
                latent_coordinates[..., 6:7] - log_jacobian,
                latent_coordinates[..., 7:]), dim=-1)
        else:
            # Preserve the baseline API exactly: both collection and replay
            # evaluate the density recovered from the represented opening.
            evaluated = inputs.get("taken_actions", actions)
            opening = self._interior_opening(evaluated[..., 6:7])
            log_open = torch.log(opening)
            log_closed = torch.log1p(-opening)
            inverse = torch.cat((evaluated[..., :6], log_open - log_closed,
                                 evaluated[..., 7:]), dim=-1)
            coordinate_log_prob = distribution.log_prob(inverse)
            coordinate_log_prob = torch.cat((coordinate_log_prob[..., :6],
                coordinate_log_prob[..., 6:7] - log_open - log_closed,
                coordinate_log_prob[..., 7:]), dim=-1)
        log_prob = (coordinate_log_prob if reduction is None else
                    reduction(coordinate_log_prob, dim=-1, keepdim=True))
        return actions, log_prob, outputs

    def get_entropy(self, role=""):
        distribution = self._role_value("_g_distribution", role)
        if distribution is None:
            return torch.tensor(0., device=self.device)
        entropy = distribution.entropy()
        mean = distribution.loc[..., 6:7]
        scale = distribution.scale[..., 6:7]
        nodes = self._gripper_entropy_nodes.to(dtype=mean.dtype)
        weights = self._gripper_entropy_weights.to(dtype=mean.dtype)
        samples = mean + math.sqrt(2.) * scale * nodes
        log_jacobian = -F.softplus(-samples) - F.softplus(samples)
        gripper_entropy = entropy[..., 6:7] + (log_jacobian * weights).sum(-1, keepdim=True)
        return torch.cat((entropy[..., :6], gripper_entropy, entropy[..., 7:]), dim=-1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Shapes are unchanged from the old binary policy, so regular strict
        # shape checks alone cannot detect incompatible action semantics.
        marker = state_dict.get(prefix + "_gripper_action_semantics_version")
        if not torch.is_tensor(marker) or marker.shape != torch.Size([]) or int(marker.item()) != 1:
            raise RuntimeError("Continuous gripper action semantics require a version-1 policy "
                               "checkpoint; legacy binary gripper policies cannot be loaded implicitly")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
