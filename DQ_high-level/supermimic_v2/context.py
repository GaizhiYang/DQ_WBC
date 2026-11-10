"""
Context Information for SuperMimic

Context represents environment invariants that don't change during an episode
but are useful for the world model and policy.
"""
import torch
from typing import Dict, Any

from .state import environment_root_state, episode_length


class Context:
    """
    Context captures environment invariants for SuperMimic.

    These are features that stay constant within an episode but may vary
    across different task instances or resets.
    """

    def __init__(self, env, device: str = "cuda:0"):
        """
        Args:
            env: DQ_WBC environment
            device: Torch device
        """
        self.env = env
        self.device = torch.device(device)
        self.num_envs = env.num_envs
        self.context_dim = 16

        # Initialize context buffers
        self._init_context()

    def _init_context(self):
        """Initialize context features from environment"""
        self.context_buffer = torch.zeros(
            self.num_envs,
            self.context_dim,
            device=self.device
        )

        # Populate with environment-specific invariants
        self._update_context()

    def _per_env(self, value, name):
        value = torch.as_tensor(value, device=self.device, dtype=self.context_buffer.dtype).detach()
        if value.numel() == 1:
            return value.reshape(1).expand(self.num_envs)
        if value.numel() != self.num_envs:
            raise ValueError(f"{name} must be scalar or have one value per environment")
        return value.reshape(self.num_envs)

    def _object_masses(self, env_ids):
        mass = getattr(self.env, 'object_mass', None)
        if mass is not None:
            return self._per_env(mass, 'object_mass')[env_ids]
        # B1Z1 does not publish a mass tensor. Query the actual actor properties
        # only at initialization/reset, never during capture or imagination.
        if all(hasattr(self.env, name) for name in ('gym', 'envs', 'cube_handles')):
            masses = [sum(body.mass for body in self.env.gym.get_actor_rigid_body_properties(
                self.env.envs[index], self.env.cube_handles[index])) for index in env_ids.tolist()]
            return torch.tensor(masses, device=self.device, dtype=self.context_buffer.dtype)
        raise AttributeError("Context requires object_mass or the object's Isaac Gym rigid-body properties")

    @torch.no_grad()
    def _update_context(self, env_ids=None):
        """Capture reset-time task parameters for the selected environments."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if env_ids.numel() == 0:
            return
        goal = getattr(self.env, 'goal_pos', None)
        if goal is not None:
            goals = torch.as_tensor(goal, device=self.device, dtype=self.context_buffer.dtype)
            if goals.shape == (3,):
                goals = goals.expand(self.num_envs, 3)
            if goals.shape != (self.num_envs, 3):
                raise ValueError("goal_pos must have shape (3,) or (num_envs, 3)")
        else:
            # Native success is lift relative to this episode's table/object,
            # including the randomized SGS reset height.
            if not all(hasattr(self.env, name) for name in (
                    'table_heights', 'init_height', 'lifted_success_threshold')):
                raise AttributeError("Context requires goal_pos or table/object heights and lifted_success_threshold")
            goals = environment_root_state(self.env, 'object')[:, :3].to(self.device).clone()
            goals[:, 2] = (self._per_env(self.env.table_heights, 'table_heights')
                           + self._per_env(self.env.init_height, 'init_height')
                           + self._per_env(self.env.lifted_success_threshold, 'lifted_success_threshold'))
        self.context_buffer[env_ids, 0:3] = goals[env_ids]

        gravity = getattr(getattr(self.env, 'sim_params', None), 'gravity', None)
        if gravity is None:
            gravity = getattr(self.env, 'cfg', {}).get('sim', {}).get('gravity')
        if gravity is None:
            raise AttributeError("Context requires sim_params.gravity or cfg['sim']['gravity']")
        if all(hasattr(gravity, axis) for axis in ('x', 'y', 'z')):
            gravity = [gravity.x, gravity.y, gravity.z]
        gravity = torch.as_tensor(gravity, device=self.device, dtype=self.context_buffer.dtype)
        if gravity.shape != (3,):
            raise ValueError("Simulation gravity must have three components")
        self.context_buffer[env_ids, 3:6] = torch.nn.functional.normalize(gravity, dim=0)
        self.context_buffer[env_ids, 6] = self._object_masses(env_ids) / 10.0
        robot_base = environment_root_state(self.env, 'robot')[:, :3].to(self.device)
        self.context_buffer[env_ids, 7:10] = robot_base[env_ids]

        # The native lift predicate uses 0.2 m EE proximity (pick_success.py).
        threshold = self._per_env(getattr(self.env, 'success_threshold', 0.2), 'success_threshold')
        self.context_buffer[env_ids, 10] = threshold[env_ids]
        self.context_buffer[env_ids, 11] = goals[env_ids, 2]
        self.context_buffer[env_ids, 12] = episode_length(self.env) / 1000.0
        self.context_buffer[env_ids, 13] = 1.0
        self.context_buffer[env_ids, 14:16] = 0.0

    def get_context(self) -> torch.Tensor:
        """
        Get current context features.

        Returns:
            context: [num_envs, context_dim] context tensor
        """
        # A later partial reset must not mutate a transition already collected.
        return self.context_buffer.clone()

    def reset_context(self, env_ids: torch.Tensor):
        """
        Reset context for specific environments (in case of parameter changes).

        Args:
            env_ids: Indices of environments to reset
        """
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        self._update_context(env_ids)

    def to_dict(self) -> Dict[str, Any]:
        """
        Export context as dictionary for logging.

        Returns:
            Dictionary of context parameters
        """
        return {
            "target_pos": self.context_buffer[0, 0:3].cpu().tolist(),
            "gravity": self.context_buffer[0, 3:6].cpu().tolist(),
            "object_mass": self.context_buffer[0, 6].item(),
            "robot_base": self.context_buffer[0, 7:10].cpu().tolist(),
            "success_threshold": self.context_buffer[0, 10].item(),
            "target_height": self.context_buffer[0, 11].item(),
        }
