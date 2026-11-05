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
    "model_microbatch_size": 256,
    "validation_batch_size": 256,
    "imagination_batch": 4096,
    "imagination_chunk_size": 128,
    "horizon": 4,
    "warmup_steps": 5,  # 40 low steps; first Actor update still waits for PPO.
    "updates_per_low_step": 0.4,
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
}


def cosine_ratio(low_steps, decay_low_steps=250000, final_ratio=0.03):
    """Same cosine shape as the reproduction, indexed by collected low steps."""
    fraction = min(max(float(low_steps) / float(decay_low_steps), 0.), 1.)
    return final_ratio + (1. - final_ratio) * 0.5 * (1. + math.cos(math.pi * fraction))
