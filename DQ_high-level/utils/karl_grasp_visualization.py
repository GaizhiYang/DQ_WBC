"""Viewer-only overlay of the grasp actually packed into the KARL observation."""
import numpy as np
import torch

from modules.karl_grasp_selector import rpy_to_quaternion


def selected_grasp_world_pose(selected, base_quaternion, arm_base):
    """Invert the teacher's base-frame observation transform (xyzw quaternions).

    Position is relative to arm_base, while orientation uses the full robot
    base rotation. DQ's "world" coordinates here are per-environment, before
    the viewer's environment layout offset. Preserve the existing clipping.
    """
    q = base_quaternion
    xyz, w = q[:, :3], q[:, 3:]
    t = 2 * torch.cross(xyz, selected[:, :3], dim=-1)
    position = arm_base + selected[:, :3] + w * t + torch.cross(xyz, t, dim=-1)
    local_q = rpy_to_quaternion(selected[:, 3:])
    local_xyz, local_w = local_q[:, :3], local_q[:, 3:]
    quaternion = torch.cat((
        w * local_xyz + local_w * xyz + torch.cross(xyz, local_xyz, dim=-1),
        w * local_w - (xyz * local_xyz).sum(-1, keepdim=True),
    ), dim=-1)
    return torch.cat((position, quaternion), dim=-1)


class KarlGraspVisualizer:
    """Cache one policy-step target per displayed environment, without selecting.

    Construct only with a live viewer. Isaac Gym imports and GPU-to-CPU copies
    are deliberately absent from the normal headless training path.
    """
    def __init__(self, env, max_envs=8, selector_name="karl"):
        from isaacgym import gymapi, gymutil

        self.env = env
        self.selector_name = selector_name
        self.label = "[%s grasp view]" % selector_name.upper()
        self.gymapi, self.gymutil = gymapi, gymutil
        self.count = min(env.num_envs, max_envs)
        self.axes = gymutil.AxesGeometry(0.12)
        self.center = gymutil.WireframeSphereGeometry(0.015, 8, 8, None, color=(1, 1, 0))
        self.world_poses = None
        self.valid = np.zeros(self.count, dtype=bool)
        self.indices = np.full(self.count, -2, dtype=np.int64)
        self._camera_initialized = False
        print(self.label + " Yellow: selected position; RGB: X/Y/Z axes (12 cm). "
              "Showing first %d environments; candidate indices are 0-based." % self.count)

    @torch.no_grad()
    def update(self, selected, metrics):
        selected = selected[:self.count].detach().to(self.env.device)
        self.world_poses = selected_grasp_world_pose(
            selected, self.env._robot_root_states[:self.count, 3:7],
            self.env.arm_base[:self.count],
        ).cpu().numpy()
        self.valid = (~metrics[self.selector_name + "_no_valid_grasp"][:self.count]).cpu().numpy()
        indices = metrics[self.selector_name + "_grasp_index"][:self.count].cpu().numpy().copy()
        indices[~self.valid] = -1
        changed = np.flatnonzero(indices != self.indices)
        if len(changed):
            print(self.label + " " + "; ".join(
                "env %d -> candidate %d" % (i, indices[i]) if self.valid[i]
                else "env %d -> no valid candidate (EE fallback, marker hidden)" % i
                for i in changed))
            if self.selector_name == "geometric":
                for i in changed[self.valid[changed]]:
                    values = [metrics["geometric_" + key][i].item() for key in
                              ("horizontal_distance", "topdown_angle_deg", "score", "valid_count")]
                    print(self.label + " env %d: center offset %.1f mm, down angle %.1f deg, "
                          "score %.3f, valid %d/30" % (i, values[0] * 1000, *values[1:]))
        self.indices = indices

        # The environment's generic camera starts far from small batches.
        # Focus env 0 once; leave subsequent user camera navigation untouched.
        if not self._camera_initialized and self.valid[0]:
            p = self.world_poses[0, :3]
            camera = self.gymapi.Vec3(float(p[0] - 1.5), float(p[1] - 2.0), float(p[2] + 1.3))
            target = self.gymapi.Vec3(float(p[0]), float(p[1]), float(p[2]))
            self.env.gym.viewer_camera_look_at(self.env.viewer, self.env.envs[0], camera, target)
            self._camera_initialized = True

    def reset(self, env_ids=None):
        # Hide old episode targets during the environment's reset-time renders.
        if env_ids is None:
            self.valid[:] = False
            self.indices[:] = -2
        else:
            ids = env_ids.detach().cpu().numpy()
            ids = ids[ids < self.count]
            self.valid[ids] = False
            self.indices[ids] = -2

    def draw(self):
        if self.world_poses is None:
            return
        for i in np.flatnonzero(self.valid):
            # These already use DQ's per-environment world frame. The viewer
            # applies the layout offset when envs[i] is supplied; subtracting
            # get_env_origin() would incorrectly pile all markers into env 0.
            p = self.world_poses[i, :3]
            q = self.world_poses[i, 3:]
            pose = self.gymapi.Transform(self.gymapi.Vec3(*map(float, p)),
                                         self.gymapi.Quat(*map(float, q)))
            self.gymutil.draw_lines(self.axes, self.env.gym, self.env.viewer, self.env.envs[i], pose)
            self.gymutil.draw_lines(self.center, self.env.gym, self.env.viewer, self.env.envs[i], pose)
