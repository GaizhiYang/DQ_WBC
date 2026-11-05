"""Real gripper target equations and task intent on CPU/CUDA without Isaac Gym."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from utils.gripper_control import closing_command, opening_to_joint
from utils.pick_success import lifted_grasp_mask


class GripperControlTests(unittest.TestCase):
    def devices(self):
        return ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])

    def test_production_controller_interpolates_targets_and_keeps_legacy_mode_explicit(self):
        source = Path(__file__).resolve().parents[1] / "envs/b1z1_base.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "B1Z1Base")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "set_gripper")
        namespace = {"torch": torch, "opening_to_joint": opening_to_joint}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
        for device in self.devices():
            for dofs in (1, 2):
                opening = torch.tensor([0., .25, .5, 1.], device=device)
                env = SimpleNamespace(continuous_gripper=True, num_gripper_dof=dofs,
                    actions=torch.zeros(4, 9, device=device),
                    gripper_dof_pos=torch.zeros(4, dofs, device=device),
                    dof_limits_lower=torch.full((dofs,), -math.pi / 2, device=device),
                    dof_limits_upper=torch.zeros(dofs, device=device))
                env.actions[:, 6] = opening
                namespace["set_gripper"](env)
                expected = (-math.pi / 2 * opening)[:, None].expand(-1, dofs)
                torch.testing.assert_close(env.gripper_dof_pos, expected)
                self.assertEqual(env.gripper_dof_pos.device, opening.device)
                if dofs != 1:
                    continue  # The current teacher uses the one-DOF asset.
                env.continuous_gripper = False
                env.actions[:, 6] = torch.tensor([-1., -.01, 0., 1.], device=device)
                namespace["set_gripper"](env)
                expected = torch.tensor([0., 0., -math.pi / 2, -math.pi / 2], device=device)
                torch.testing.assert_close(env.gripper_dof_pos, expected[:, None].expand(-1, dofs))

    def test_opening_has_joint_target_gradient_and_only_closing_side_is_grasp_eligible(self):
        for device in self.devices():
            opening = torch.tensor([0., .25, .499, .5, .75, 1.], device=device, requires_grad=True)
            target = opening_to_joint(opening, 0., -math.pi / 2)
            target.sum().backward()
            torch.testing.assert_close(opening.grad, torch.full_like(opening, -math.pi / 2))
            expected = [True, True, True, False, False, False]
            self.assertEqual(closing_command(opening).tolist(), expected)
            mask = lifted_grasp_mask(torch.full_like(opening, .8), torch.full_like(opening, .5),
                                     torch.full_like(opening, .045), torch.full_like(opening, .05),
                                     opening, .2)
            self.assertEqual(mask.tolist(), expected)


if __name__ == "__main__":
    unittest.main()
