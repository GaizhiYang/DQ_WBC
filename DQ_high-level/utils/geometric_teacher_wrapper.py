"""Reuse single-target teacher packing/reset semantics with geometric scoring."""
import torch

from modules.geometric_grasp_selector import GeometricGraspSelector
from .grasp_geometry import TeacherGraspGeometry
from .karl_teacher_wrapper import KarlTeacherWrapper


class GeometricTeacherWrapper(KarlTeacherWrapper):
    DIAGNOSTICS = ("no_valid_grasp", "valid_count", "grasp_switched", "locked",
                   "horizontal_distance", "topdown_angle_deg", "score",
                   "object_rejected_count", "table_rejected_count")

    def __init__(self, env, settings=None, visualize_grasp=False, grasp_vis_envs=8):
        self.geometry = TeacherGraspGeometry(env)
        selector = GeometricGraspSelector(env.num_envs, env.rl_device, settings)
        super().__init__(env, visualize_grasp=visualize_grasp,
                         grasp_vis_envs=grasp_vis_envs, selector=selector,
                         selector_name="geometric")
        self.diagnostics_callback = None
        self._diagnostic_sum = None
        self._diagnostic_steps = 0

    def _select(self, poses, ee_pose):
        return self.selector.select(poses, ee_pose, self.geometry.context())

    def step(self, actions):
        result = super().step(actions)
        if self.diagnostics_callback is not None:
            metrics = result[-1]
            means = torch.stack([metrics["geometric_" + key].float().mean()
                                 for key in self.DIAGNOSTICS])
            self._diagnostic_sum = means if self._diagnostic_sum is None else self._diagnostic_sum + means
            self._diagnostic_steps += 1
            # One small CPU transfer per rollout, not one per candidate/metric.
            if self._diagnostic_steps == 24:
                values = (self._diagnostic_sum / self._diagnostic_steps).cpu().tolist()
                for key, value in zip(self.DIAGNOSTICS, values):
                    self.diagnostics_callback("Grasp geometric / " + key, value)
                self._diagnostic_sum = None
                self._diagnostic_steps = 0
        return result
