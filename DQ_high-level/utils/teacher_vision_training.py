"""PPO assembly/checkpoint support without importing Isaac Gym.

The PPO implementation and action convention are the original teacher's.
Only model inputs, preprocessing, and checkpoint bookkeeping differ.
"""
from copy import deepcopy
from pathlib import Path
import re

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveLR

from modules.teacher_vision import (
    TeacherVisionPolicy, TeacherVisionValue, load_legacy_teacher_weights,
)
from utils.teacher_vision_preprocessor import (
    PrefixRunningStandardScaler, FrozenRunningStandardScaler,
)


def legacy_step(path, explicit_step=None):
    """Do not guess the curriculum position of best_agent.pt."""
    if explicit_step is not None:
        if explicit_step < 0:
            raise ValueError("--teacher_initial_step must be non-negative")
        return explicit_step
    match = re.fullmatch(r"agent_(\d+)\.pt", Path(path).name)
    if match is None:
        raise ValueError("Specify --teacher_initial_step for a legacy checkpoint without an agent_<step>.pt name")
    return int(match.group(1))


def make_agent(env, settings, vision_mode, experiment_dir, experiment_name,
               deterministic=False, wandb=False, wandb_project="isaacgym"):
    """Use the stock PPO with one packed observation buffer and frozen scalers."""
    rollouts = int(settings["rollouts"])
    batch_size = int(settings["minibatch_size"])
    epochs = int(settings["learning_epochs"])
    samples = rollouts * env.num_envs
    if min(rollouts, batch_size, epochs) <= 0 or (not deterministic and samples % batch_size):
        raise ValueError("rollouts * num_envs must be divisible by the positive minibatch_size; epochs must be positive")
    interval = int(settings["checkpoint_interval"])
    if interval < 0 or (interval and interval % rollouts):
        raise ValueError("checkpoint_interval must be 0 or a multiple of rollouts")
    models = {
        "policy": TeacherVisionPolicy(env.observation_space, env.action_space, env.rl_device,
                                      vision_mode=vision_mode, deterministic=deterministic),
        "value": TeacherVisionValue(env.observation_space, env.action_space, env.rl_device),
    }
    cfg = deepcopy(PPO_DEFAULT_CONFIG)
    cfg.update({
        "rollouts": rollouts,
        "learning_epochs": epochs,
        "mini_batches": max(1, samples // batch_size),
        "discount_factor": 0.99,
        "lambda": 0.95,
        "learning_rate": float(settings["learning_rate"]),
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
        "random_timesteps": 0,
        "learning_starts": 0,
        "grad_norm_clip": 1.0,
        "ratio_clip": 0.2,
        "value_clip": 0.2,
        "clip_predicted_values": True,
        "value_loss_scale": 1.0,
        "kl_threshold": 0,
        "rewards_shaper": None,
        "state_preprocessor": PrefixRunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.rl_device, "freeze": True},
        "value_preprocessor": FrozenRunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": env.rl_device, "freeze": True},
    })
    cfg["experiment"].update({
        "write_interval": 0 if deterministic else rollouts,
        "checkpoint_interval": 0 if deterministic else interval,
        "directory": str(experiment_dir),
        "experiment_name": experiment_name,
        "wandb": wandb and not deterministic,
    })
    if wandb and not deterministic:
        cfg["experiment"]["wandb_kwargs"] = {
            "project": wandb_project, "tensorboard": False, "name": experiment_name,
        }
    memory = None if deterministic else RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.rl_device)
    agent = PPO(models=models, memory=memory, cfg=cfg, observation_space=env.observation_space,
                action_space=env.action_space, device=env.rl_device)
    # Stock PPO saves the optimizer but does not register its LR scheduler.
    agent.checkpoint_modules["scheduler"] = agent.scheduler
    return agent


def initialize_from_teacher(agent, checkpoint):
    """Load all teacher parameters/statistics, deliberately leaving Adam fresh."""
    if "teacher_vision_state" in checkpoint:
        raise ValueError("This is a visual-teacher checkpoint; restore it with --checkpoint")
    required = ("policy", "value", "state_preprocessor", "value_preprocessor")
    missing = set(required) - checkpoint.keys()
    if missing:
        raise ValueError("Teacher checkpoint is missing: " + ", ".join(sorted(missing)))
    load_legacy_teacher_weights(agent.policy, checkpoint["policy"])
    load_legacy_teacher_weights(agent.value, checkpoint["value"])
    agent.checkpoint_modules["state_preprocessor"].load_state_dict(checkpoint["state_preprocessor"], strict=True)
    agent.checkpoint_modules["value_preprocessor"].load_state_dict(checkpoint["value_preprocessor"], strict=True)


class TeacherVisionTrainingState:
    """Checkpoint experiment settings and environment curriculum position.

    Simulation state and partial rollouts are not serialized. A restored run
    starts fresh episodes/rollout, while retaining weights, Adam, LR scheduler,
    normalization, completed interaction count, and reward schedule position.
    """
    def __init__(self, env, config, environment_options, origin_step):
        self.env = env
        self.config = deepcopy(config)
        self.environment_options = deepcopy(environment_options)
        self.origin_step = int(origin_step)

    def state_dict(self):
        global_step = int(self.env.global_step_counter)
        return {
            "version": 1,
            "origin_step": self.origin_step,
            "global_step": global_step,
            "completed_steps": global_step - self.origin_step,
            "experiment_config": deepcopy(self.config),
            "environment_options": deepcopy(self.environment_options),
        }

    def load_state_dict(self, state):
        if state.get("version") != 1:
            raise ValueError("Unsupported visual-teacher checkpoint version")
        self.origin_step = int(state["origin_step"])
        # Wrapper __setattr__ does not forward to its underlying environment.
        self.env._env.global_step_counter = int(state["global_step"])


def restore_experiment(agent, checkpoint, evaluation=False):
    required = ["policy", "value", "state_preprocessor", "value_preprocessor", "teacher_vision_state"]
    if not evaluation:
        required += ["optimizer", "scheduler"]
    missing = set(required) - checkpoint.keys()
    if missing:
        raise ValueError("Visual-teacher checkpoint is missing: " + ", ".join(sorted(missing)))
    for key in required:
        agent.checkpoint_modules[key].load_state_dict(checkpoint[key])
