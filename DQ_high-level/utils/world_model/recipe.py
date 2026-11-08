"""SuperMimic-aligned defaults for the fixed-eight DQ teacher.

The 20 ms clock defines replay, warmup and the update-to-data ratio. Updates
are executed after a completed PPO rollout, so collecting a PPO rollout never
changes Actor weights halfway through it. No compilation/distributed changes.
"""

import math


SUPERMIMIC_CONFIG = {
    "recipe": "supermimic",
    "replay_envs": 4096,
    "replay_capacity": 256,
    "low_replay_capacity": 1000,
    "compact_low_replay": True,
    "window": 8,
    "hidden": [2048, 2048, 2048],
    "batch_size": 2048,
    "batches_per_update": 10,
    "state_loss_reduction": "sum",
    "auxiliary_weight": 0.0,
    "action_coordinates": "normalized_commands",
    "initial_action_std": 0.1,
    "actor_last_gain": 0.01,
    "action_rate_mode": "squared_all",
    "action_rate_weight": -0.002,  # SuperMimic -0.1 action difference weight * 0.02 s.
    # Keep the logical minibatch together on the CUDA 0 / 48 GiB launcher.
    # Smaller overrides remain available for devices with less free memory.
    "model_microbatch_size": 2048,
    "validation_batch_size": 256,
    "imagination_batch": 4096,
    "imagination_chunk_size": 4096,
    "horizon": 4,
    "warmup_steps": 5,  # 40 low steps; first Actor update still waits for PPO.
    "updates_per_low_step": 2.0,  # INCREASED from 0.4 to match SuperMimic training intensity
    "learning_rate": 2e-4,
    "actor_learning_rate": 2e-4,
    "auxiliary_strength": 1.0,
    "actor_betas": [0.9, 0.95],
    "max_grad_norm": 1.0,
    "grad_clip_value": 0.1,
    "kl_rollback": False,
    "value_source": "imagined",
    "critic_hidden": [256, 128],
    "critic_learning_rate": 2e-4,
    "critic_epochs": 5,
    "critic_minibatches": 4,
    "critic_gae_lambda": 0.95,
    "critic_max_grad_norm": 1.0,
    "balance_families": False,
    "recent_fraction": 0.0,
    "lr_schedule": "cosine",
    "lr_final_ratio": 0.03,
    "lr_decay_low_steps": 250000,
    "fixed_preprocessing": True,
    "observation_noise_scale": 1.0,
    # Probes are optional diagnostics, never activation/rollback gates.
    "probe_interval": 0,
    # Disable overly aggressive gating for initial training
    "disable_action_probes": True,
    "loose_error_gates": True,
}


# Separate from SUPERMIMIC_CONFIG: saved legacy recipes keep their UTD behavior.
MANIPULATION_CONFIG = {
    **SUPERMIMIC_CONFIG,
    "recipe": "manipulation",
    "batches_per_update": 1,
    "updates_per_low_step": 0.0,
    "wm_epochs": 1.0,
    "wm_max_batches": 512,
    "wm_prepare_boundaries": 4,
    "value_prepare_boundaries": 2,
    "value_prepare_batches": 8,
    "actor_stage1_boundaries": 4,
    "actor_stage2_boundaries": 4,
    "actor_stage1_updates": 1,
    "actor_stage2_updates": 2,
    "actor_max_updates": 4,
    "holdout_mode": "environment",
    "holdout_fraction": 0.05,
    "allow_partial_windows": True,
    "diagnostics_interval": 1,
    "diagnostics_batch_size": 128,
    "critic_memory": False,
    "layer_norm": False,
    "running_normalization": False,
    "amp": "none",
    "task_contract": "manipulation_v2",
    "task_aux_alpha": 0.1,
    "task_aux_config": {},
    "raw_action_bound_weight": 0.0,
    "raw_gripper_latent_bound": 3.0,
}


def cosine_ratio(low_steps, decay_low_steps=250000, final_ratio=0.03):
    """Same cosine shape as the reproduction, indexed by collected low steps."""
    fraction = min(max(float(low_steps) / float(decay_low_steps), 0.), 1.)
    return final_ratio + (1. - final_ratio) * 0.5 * (1. + math.cos(math.pi * fraction))
