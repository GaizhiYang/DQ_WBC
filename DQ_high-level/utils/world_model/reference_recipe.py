"""Pure SuperMimic-manipulation defaults; all collection clocks are explicit.

One real high step contains eight 20 ms controller/model steps. The reference
runner's 0.4 update rate is applied to HIGH decisions, while model epochs count
new valid LOW training transitions. This is the necessary fixed8 adaptation.
"""

from .recipe import MANIPULATION_CONFIG


REFERENCE_CONFIG = {
    **MANIPULATION_CONFIG,
    "recipe": "reference",
    "mode": "hybrid",
    "fixed8": True,
    "contact_hidden": [128, 128],
    "model_grad_clip": 1.,
    "contact_weight": .1,
    "recent_steps": 32,
    "collection_updates_per_high_step": .4,
    "collection_warmup_high_steps": 64,
    "world_model_only_iterations": 1000,
    "policy_update_interval": 1,
    "gamma": .99,
    "wm_epochs": 4.,
    "wm_max_batches": 0,  # zero means no cap; explicit caps discard/log excess
    "lr_decay_iterations": 10000,
    "action_noise": .1,
    "horizon": 16,
    "imagination_chunk_size": 256,
    "hidden": [2048, 2048, 2048, 2048, 2048],
    "actor_hidden": [2048, 1024, 512, 256],
    "layer_norm": True,
    "running_normalization": True,
    "dynamics_target": "physical_increment",
    "state_loss_reduction": "reference",
    "amp": "bf16",
    "policy_amp": "bf16",
    "actor_running_normalization": True,
    "critic_running_normalization": True,
    "actor_objective_reduction": "sum",
    "critic_activation": "elu",
    "fixed_preprocessing": False,
    "observation_noise_scale": 0.,
    "critic_features_mode": "reference_full",
    "imagined_reward_mode": "reference_task",
    "reference_task_config": {},
    "task_aux_alpha": 0.,
    "raw_action_bound_weight": 1.,
    "diagnostics_interval": 50,
}
