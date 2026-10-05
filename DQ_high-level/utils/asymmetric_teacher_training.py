"""Direct asymmetric PPO with optional M1 task state and M2 perception."""
from copy import deepcopy
import math

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.resources.schedulers.torch import KLAdaptiveLR

from modules.asymmetric_teacher import (
    AsymmetricTeacherPolicy, AsymmetricTeacherValue, initialize_policy_from_teacher,
    initialize_policy_from_asymmetric, initialize_value_from_teacher,
)
from utils.teacher_vision_preprocessor import PrefixRunningStandardScaler, FrozenRunningStandardScaler
from utils.asymmetric_scratch_preprocessor import (
    RolloutPrefixRunningStandardScaler, IdentityValuePreprocessor,
)
from utils import asymmetric_distributed as distributed


def feature_flags(settings):
    mode = settings.get("mode", "m2")
    if mode not in ("m0", "m1", "m2"):
        raise ValueError("mode must be m0, m1 or m2")
    return {"m1": mode != "m0", "m2": mode == "m2"}


def validate_settings(settings):
    """Budgets are vector steps and all training boundaries are full rollouts."""
    if settings.get("normalization", "frozen") not in ("frozen", "rollout"):
        raise ValueError("normalization must be frozen or rollout")
    if settings.get("initialization", "teacher") not in ("teacher", "scratch"):
        raise ValueError("initialization must be teacher or scratch")
    rollout = int(settings["rollouts"])
    if min(rollout, int(settings["minibatch_size"]), int(settings["learning_epochs"])) <= 0:
        raise ValueError("rollouts, minibatch_size and learning_epochs must be positive")
    total = int(settings["total_steps"])
    if total <= 0 or total % rollout:
        raise ValueError("total_steps must be positive and a multiple of rollouts")
    flags = feature_flags(settings)
    for key in ("checkpoint_interval", "eval_interval"):
        interval = int(settings.get(key, 0))
        if interval < 0 or interval % rollout:
            raise ValueError(key + " must be zero or a multiple of rollouts")
    if int(settings.get("eval_interval", 0)) and int(settings.get("eval_steps", 0)) <= 0:
        raise ValueError("eval_steps must be positive when periodic evaluation is enabled")
    if settings.get("time_limit_semantics", "task_deadline") != "task_deadline":
        raise ValueError("Asymmetric training uses task deadlines, not collection-truncation bootstrap")
    reward = settings.get("perception_reward", {})
    for name, default in (("weight", 0.02), ("sigma_m", 0.1), ("prediction_horizon_s", 0.15)):
        value = float(reward.get(name, default))
        if not math.isfinite(value) or value < 0 or (name == "sigma_m" and value == 0):
            raise ValueError("Invalid perception_reward." + name)
    if flags["m2"]:
        from utils.asymmetric_perception import PerceptionConfig
        perception = PerceptionConfig(**settings.get("perception", {}))
        if perception.max_depth_m > 3.0:
            raise ValueError("Simulation target depth is saturated at 3 m; max_depth_m must not exceed 3")
    return total


class AsymmetricTeacherTrainingState:
    """Count training independently of evaluation and the old reward curriculum.

    Physics, camera/filter histories and partial rollouts are not serialized.
    """
    def __init__(self, env, config, environment_options, origin_step):
        self.env = env
        self.config = deepcopy(config)
        self.environment_options = deepcopy(environment_options)
        self.origin_step = int(origin_step)
        if self.origin_step < 0:
            raise ValueError("origin_step must be non-negative")
        self.settings = self.config["asymmetric_teacher"]
        self.total_steps = validate_settings(self.settings)
        self.completed_steps = 0
        self.best_eval_score = -math.inf
        self.last_eval_step = 0

    def mark_completed(self, completed_steps):
        step = int(completed_steps)
        if step < self.completed_steps or step > self.total_steps:
            raise ValueError("Invalid completed training budget")
        if step % int(self.settings["rollouts"]):
            raise ValueError("Only completed PPO rollouts may be checkpointed")
        self.completed_steps = step
        raw = getattr(self.env, "_env", self.env)
        raw.global_step_counter = self.origin_step + step

    def state_dict(self):
        return {
            "version": 2,
            "origin_step": self.origin_step,
            "global_step": self.origin_step + self.completed_steps,
            "completed_steps": self.completed_steps,
            "best_eval_score": self.best_eval_score,
            "last_eval_step": self.last_eval_step,
            "experiment_config": deepcopy(self.config),
            "environment_options": deepcopy(self.environment_options),
        }

    def load_state_dict(self, saved):
        if saved.get("version") != 2:
            raise ValueError("Older asymmetric checkpoints require --teacher_init_checkpoint for weights initialization")
        if saved["experiment_config"]["asymmetric_teacher"] != self.settings:
            raise ValueError("Restore the saved PPO settings and total budget")
        self.origin_step = int(saved["origin_step"])
        self.completed_steps = 0
        self.mark_completed(int(saved["completed_steps"]))
        if int(saved["global_step"]) != self.origin_step + self.completed_steps:
            raise ValueError("Inconsistent checkpoint reward curriculum step")
        self.best_eval_score = float(saved.get("best_eval_score", -math.inf))
        self.last_eval_step = int(saved.get("last_eval_step", 0))


def make_agent(env, settings, experiment_dir, experiment_name, *,
               deterministic=False, wandb=False, wandb_project="isaacgym"):
    validate_settings(settings)
    rollouts = int(settings["rollouts"])
    samples = rollouts * env.num_envs
    batch_size = int(settings["minibatch_size"])
    if not deterministic and samples % batch_size:
        raise ValueError("num_envs * rollouts must be divisible by minibatch_size")
    models = {
        "policy": AsymmetricTeacherPolicy(env.observation_space, env.action_space, env.rl_device,
                                          deterministic=deterministic, **feature_flags(settings)),
        "value": AsymmetricTeacherValue(env.observation_space, env.action_space, env.rl_device, **feature_flags(settings)),
    }
    if settings.get("initialization") == "scratch":
        # Zero adapters preserve a migrated teacher; a fresh visual policy has
        # no such dependency and should train its CNN on the very first update.
        models["policy"].actor.visual_adapter.reset_parameters()
        for owner, name in ((models["policy"].actor, "belief_adapter"),
                            (models["value"], "task_adapter"),
                            (models["value"], "belief_adapter")):
            adapter = getattr(owner, name, None)
            if adapter is not None:
                adapter.reset_parameters()
    rollout_normalization = settings.get("normalization", "frozen") == "rollout"
    cfg = deepcopy(PPO_DEFAULT_CONFIG)
    cfg.update({
        "rollouts": rollouts, "learning_epochs": int(settings["learning_epochs"]),
        "mini_batches": max(1, samples // batch_size), "discount_factor": 0.99, "lambda": 0.95,
        "learning_rate": float(settings["learning_rate"]),
        "learning_rate_scheduler": KLAdaptiveLR,
        "learning_rate_scheduler_kwargs": {"kl_threshold": 0.008},
        "random_timesteps": 0, "learning_starts": 0, "grad_norm_clip": 1.0,
        "ratio_clip": 0.2, "value_clip": 0.2, "clip_predicted_values": True,
        "value_loss_scale": 1.0, "kl_threshold": 0, "rewards_shaper": None,
        "time_limit_bootstrap": False,
        "state_preprocessor": RolloutPrefixRunningStandardScaler if rollout_normalization else PrefixRunningStandardScaler,
        "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.rl_device, "freeze": True},
        "value_preprocessor": IdentityValuePreprocessor if rollout_normalization else FrozenRunningStandardScaler,
        "value_preprocessor_kwargs": {"size": 1, "device": env.rl_device, "freeze": True},
    })
    cfg["experiment"].update({
        "write_interval": rollouts if not deterministic and distributed.is_main() else 0,
        # Trainer saves after updating its own metadata and selects by eval GSR.
        "checkpoint_interval": 0,
        "directory": str(experiment_dir), "experiment_name": experiment_name,
        "wandb": wandb and not deterministic and distributed.is_main(),
    })
    if wandb and not deterministic:
        cfg["experiment"]["wandb_kwargs"] = {
            "project": wandb_project, "tensorboard": False, "name": experiment_name,
        }
    memory = None if deterministic else RandomMemory(memory_size=rollouts, num_envs=env.num_envs, device=env.rl_device)
    agent_class = PPO
    if distributed.active():
        from learning.asymmetric_distributed_ppo import DistributedAsymmetricPPO
        agent_class = DistributedAsymmetricPPO
    agent = agent_class(models=models, memory=memory, cfg=cfg, observation_space=env.observation_space,
                action_space=env.action_space, device=env.rl_device)
    agent.checkpoint_modules["scheduler"] = agent.scheduler
    return agent


def initialize_from_teacher(agent, checkpoint, *, allow_legacy=False):
    required = {"policy", "value", "state_preprocessor", "value_preprocessor"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError("Teacher checkpoint is missing: " + ", ".join(sorted(missing)))
    metadata = checkpoint.get("teacher_vision_state")
    if metadata and metadata["experiment_config"]["teacher_vision"].get("vision_mode") != "images":
        raise ValueError("Asymmetric migration requires an images visual teacher")
    if "asymmetric_teacher_state" in checkpoint:
        initialize_policy_from_asymmetric(agent.policy, checkpoint["policy"])
    else:
        initialize_policy_from_teacher(agent.policy, checkpoint["policy"], allow_legacy=allow_legacy)
    initialize_value_from_teacher(agent.value, checkpoint["value"])
    for key in ("state_preprocessor", "value_preprocessor"):
        agent.checkpoint_modules[key].load_state_dict(checkpoint[key], strict=True)


def restore_experiment(agent, checkpoint, *, evaluation=False):
    required = ["policy", "value", "state_preprocessor", "value_preprocessor", "asymmetric_teacher_state"]
    if not evaluation:
        required.extend(("optimizer", "scheduler"))
    missing = set(required) - checkpoint.keys()
    if missing:
        raise ValueError("Asymmetric checkpoint is missing: " + ", ".join(sorted(missing)))
    for key in required:
        agent.checkpoint_modules[key].load_state_dict(checkpoint[key])
