"""Exercise graphics requests without importing Isaac Gym or creating a GPU sim."""

import ast
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace
import unittest

import torch

from utils.sgs import SUPPORTED_FAMILY_PROFILES


ROOT = Path(__file__).resolve().parents[1]


def source_node(path, name, owner=None):
    tree = ast.parse((ROOT / path).read_text())
    if owner:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name == owner)
    return deepcopy(next(node for node in tree.body if getattr(node, "name", None) == name))


def compile_nodes(nodes, namespace):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "<production graphics methods>", "exec"), namespace)


class GraphicsSelectionTests(unittest.TestCase):
    def select_graphics_device(self, *, sim_id=0, headless=True,
                               enable_camera=False, camera_test=False,
                               external_camera=None):
        class SimulatorParent:
            def create_sim(self, compute_device, graphics_device, engine, params):
                self.graphics_request = (compute_device, graphics_device)
                return object()

        method = source_node("envs/b1z1_base.py", "create_sim", "B1Z1Base")
        namespace = {"SimulatorParent": SimulatorParent, "os": os,
                     "__file__": str(ROOT / "envs/b1z1_base.py"),
                     "gymapi": SimpleNamespace(DOMAIN_SIM=0)}
        cls = ast.ClassDef(name="B1Z1Base", bases=[ast.Name(id="SimulatorParent", ctx=ast.Load())],
                           keywords=[], body=[method], decorator_list=[])
        compile_nodes([cls], namespace)
        env = namespace["B1Z1Base"]()
        env.sim_id, env.headless = sim_id, headless
        env.enable_camera, env.camera_test = enable_camera, camera_test
        env.cfg = {"env": {"terrainType": "plane"}}
        if external_camera is not None:
            env.cfg["enableCameraSensors"] = external_camera
        env.physics_engine, env.sim_params = object(), object()
        env._create_grond_plane = lambda: None
        env._create_envs = lambda: None
        env.envs, env.cube_handles = [object()], [object()]
        env.gym = SimpleNamespace(get_actor_rigid_body_index=lambda *args: 0,
                                  prepare_sim=lambda sim: None)
        env.randomize = False
        with redirect_stdout(io.StringIO()):
            env.create_sim()
        self.assertEqual(env.graphics_request[0], sim_id)
        return env.graphics_request[1]

    def test_pose_only_headless_training_keeps_graphics_disabled(self):
        for flag in (None, False):
            with self.subTest(external_camera=flag):
                self.assertEqual(self.select_graphics_device(external_camera=flag), -1)

    def test_external_preview_camera_gets_graphics_without_vision_observations(self):
        for sim_id in (0, 1):
            with self.subTest(sim_id=sim_id):
                self.assertEqual(self.select_graphics_device(
                    sim_id=sim_id, external_camera=True), sim_id)

    def test_builtin_camera_camera_test_and_viewer_keep_graphics(self):
        for option in ({"enable_camera": True}, {"camera_test": True}, {"headless": False}):
            with self.subTest(option=option):
                self.assertEqual(self.select_graphics_device(sim_id=1, **option), 1)


class BuilderGraphicsRequestTests(unittest.TestCase):
    def build_environment(self, device="cuda:0", **kwargs):
        config = {"env": {"D1_bench_task_level": "Level02", "asset": {
            "assetFileRobot": "b1z1-col/urdf/b1z1.urdf",
            "asset_multi": {"sugar_box": {"dict_idx": 0}}}}, "sensor": {}}
        calls = []

        def construct(**arguments):
            calls.append(arguments)
            return SimpleNamespace(device="cpu", reset=lambda: None)

        function = source_node("utils/sgs_bank_builder.py", "create_builder_environment")
        # Replace import boundaries only; execute the actual factory's decisions.
        function.body = [node for node in function.body
                         if not isinstance(node, (ast.Import, ast.ImportFrom))]
        namespace = {"deepcopy": deepcopy, "torch": torch,
                     "B1Z1PickMulti": construct, "B1Z1RoughCfg": lambda: object(),
                     "set_seed": lambda seed: None, "load_cfg": lambda path: config,
                     "get_predict_point": lambda **args: (None, None, None),
                     "_environment_arguments": nullcontext}
        compile_nodes([function], namespace)
        namespace["create_builder_environment"](num_envs=1, device=device, **kwargs)
        self.assertFalse(calls[0]["cfg"]["sensor"]["enableCamera"])
        self.assertTrue(calls[0]["headless"])
        self.assertNotIn("enableCameraSensors", config)
        return calls[0]

    def test_bank_building_does_not_enable_graphics_by_default(self):
        arguments = self.build_environment()
        self.assertFalse(arguments["cfg"]["enableCameraSensors"])
        self.assertEqual(arguments["graphics_device_id"], -1)

    def test_requested_preview_graphics_uses_selected_gpu(self):
        for index in (0, 1):
            with self.subTest(device=index):
                arguments = self.build_environment(device=f"cuda:{index}", enable_graphics=True)
                self.assertTrue(arguments["cfg"]["enableCameraSensors"])
                self.assertEqual(arguments["graphics_device_id"], index)

    def test_interactive_and_headless_previews_both_request_graphics(self):
        families = ("full_task", "pregrasp", "at_grasp", "near_goal")
        bank = SimpleNamespace(metadata={"family_names": families, "physical_validation": {
            "validation_profile": "geometry_grasp_v2", "near_goal_grasp_verified": True}},
            tensors={"family_id": torch.arange(4)}, num_configs=4)

        class EnvironmentRequested(Exception):
            pass

        calls = []

        def request_environment(**kwargs):
            calls.append(kwargs)
            raise EnvironmentRequested()

        method = source_node("visualize_sgs_bank.py", "__init__", "BankPreview")
        method.body = [node for node in method.body
                       if not isinstance(node, (ast.Import, ast.ImportFrom))]
        namespace = {"torch": torch, "gymapi": object(), "gymtorch": object(),
                     "ConfigurationBank": SimpleNamespace(load=lambda *a, **kw: bank),
                     "create_builder_environment": request_environment,
                     "FAMILIES": families, "random": random,
                     "SUPPORTED_FAMILY_PROFILES": SUPPORTED_FAMILY_PROFILES}
        compile_nodes([method], namespace)
        for profile, profile_families in SUPPORTED_FAMILY_PROFILES.items():
            bank.metadata["family_names"] = profile_families
            bank.metadata["physical_validation"]["validation_profile"] = profile
            bank.tensors["family_id"] = torch.arange(len(profile_families))
            bank.num_configs = len(profile_families)
            for headless in (False, True):
                with self.subTest(headless=headless, profile=profile):
                    args = SimpleNamespace(bank="unused.pt", device="cuda:1", family="full_task",
                                           sample_index=None, seed=43, samples_per_family=1,
                                           config_id=None, headless=headless)
                    preview = SimpleNamespace()
                    with self.assertRaises(EnvironmentRequested):
                        namespace["__init__"](preview, args)
                    self.assertEqual(preview.families, profile_families)
                    self.assertTrue(calls[-1]["enable_graphics"])
                    self.assertEqual(calls[-1]["device"], "cuda:1")

    def test_viewer_numeric_keys_follow_saved_family_order(self):
        method = source_node("visualize_sgs_bank.py", "interactive", "BankPreview")
        bindings = {}
        gym = SimpleNamespace(
            create_viewer=lambda *args: object(),
            subscribe_viewer_keyboard_event=lambda viewer, key, name: bindings.update({key: name}),
            query_viewer_has_closed=lambda viewer: True)
        key_names = ("SPACE", "N", "R", "RIGHT", "LEFT", "F", "C", "O", "S", "ESCAPE",
                     "1", "2", "3", "4")
        gymapi = SimpleNamespace(CameraProperties=SimpleNamespace,
                                 **{"KEY_" + name: name for name in key_names})
        namespace = {"json": json, "time": time}
        compile_nodes([method], namespace)
        for profile, families in SUPPORTED_FAMILY_PROFILES.items():
            with self.subTest(profile=profile):
                bindings.clear()
                preview = SimpleNamespace(
                    gymapi=gymapi, env=SimpleNamespace(gym=gym, sim=object()), families=families,
                    args=SimpleNamespace(config_id=None), load=lambda **kwargs: None,
                    sample_indices=[[0]] * len(families),
                    family_ids=[[index] for index in range(len(families))], measurements=lambda: {})
                with redirect_stdout(io.StringIO()):
                    namespace["interactive"](preview)
                self.assertEqual({key: name for key, name in bindings.items() if key.isdigit()},
                                 {str(index + 1): name for index, name in enumerate(families)})
                if profile == "geometry_grasp_v3":
                    self.assertEqual(bindings["3"], "near_goal")
                    self.assertNotIn("at_grasp", bindings.values())


if __name__ == "__main__":
    unittest.main()
