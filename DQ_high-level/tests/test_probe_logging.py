"""Probe logger isolation through the teacher entry and real SKRL writer.

Extract the production entry function so these CPU regressions do not import
Isaac Gym. Only simulator/model construction is replaced; argument processing,
logger configuration, Agent.init and TensorBoard event writing are real.
"""

import ast
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "third_party" / "skrl"))

from skrl.agents.torch import Agent
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from utils import config
from utils.world_model.action_checkpoint import TaskContractMetadata


def _entry_fixture(directory, name="main", wandb=True, sgs=False):
    args = config.get_params([
        "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
        "--num_envs", "2", "--timesteps", "10000", "--headless",
        "--future_pose_obs", "--no_velocity_obs",
    ])
    args.experiment_dir = str(directory)
    args.wandb_name, args.wandb = name, wandb
    args.cfg_file = str(Path(directory) / (name.replace("/", "_") + ".yaml"))
    if sgs:
        bank_path = Path(directory) / "bank.pt"
        bank_path.touch()
        args.sgs_mode, args.sgs_bank = "sgs", str(bank_path)
    source_config = {"env": {
        "numEnvs": 2, "maxEpisodeLength": 777,
        "robotStartDistanceRange": [2., 2.],
        "asset": {"asset_multi": {"sugar_box": {"dict_idx": 0}}},
    }}
    import yaml
    Path(args.cfg_file).write_text(yaml.safe_dump(source_config))
    raw_env = SimpleNamespace()
    env = SimpleNamespace(
        _env=raw_env, rl_device="cpu", num_envs=2, num_agents=1,
        observation_space=SimpleNamespace(shape=(4,)), action_space=SimpleNamespace(shape=(9,)),
        sgs_bank=SimpleNamespace(fingerprint="fixture-bank", num_configs=3),
        hold_steps=10, sgs_hold_compatibility="exact", close=Mock(),
    )

    def create_env(**kwargs):
        env.constructed_args = deepcopy(kwargs["args"])
        env.constructed_config = deepcopy(kwargs["cfg"])
        return env

    def make_agent(**kwargs):
        # TensorBoard/W&B/checkpoint initialization belongs to this real base
        # class; the irrelevant PPO networks/optimizer are not needed here.
        return Agent({}, memory=kwargs["memory"], cfg=kwargs["cfg"], device="cpu")

    def make_trainer(**kwargs):
        kwargs["cfg"]["close_environment_at_exit"] = False
        return SequentialTrainer(**kwargs)

    module = ModuleType("teacher_logging_fixture")
    module.__dict__.update({
        "__file__": str(ROOT / "train_multistate_DQ_teacher.py"),
        "get_params": lambda: deepcopy(args), "deepcopy": deepcopy,
        "set_seed": Mock(), "teacher_config_path": lambda *a, **k: args.cfg_file,
        "Path": Path, "os": os, "json": json, "cprint": Mock(),
        "load_cfg": config.load_cfg, "configure_teacher_experiment": config.configure_teacher_experiment,
        "apply_pose_prediction_config": config.apply_pose_prediction_config,
        "get_predict_point": lambda **kwargs: (None, None, None),
        "create_env": create_env, "RandomMemory": Mock(return_value=object()),
        "Policy": Mock(return_value=None), "Value": Mock(return_value=None),
        "PPO_DEFAULT_CONFIG": PPO_DEFAULT_CONFIG, "KLAdaptiveRL": object,
        "RunningStandardScaler": object, "PPO": make_agent,
        "teacher_run_end": config.teacher_run_end, "SequentialTrainer": make_trainer,
        "copy_cfg": config.copy_cfg,
        "TaskContractMetadata": TaskContractMetadata,
    })
    path = ROOT / "train_multistate_DQ_teacher.py"
    node = next(node for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.FunctionDef) and node.name == "get_trainer")
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), module.__dict__)
    return module, env


class ProbeLoggingTests(unittest.TestCase):
    def test_default_training_keeps_tensorboard_wandb_and_checkpoint_cadence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"wandb": Mock()}):
            teacher, env = _entry_fixture(directory)
            trainer = teacher.get_trainer()
            agent = trainer.agents
            self.addCleanup(agent.writer.close)
            self.assertEqual(agent.write_interval, 240)
            self.assertEqual(agent.checkpoint_interval, 5000)
            self.assertEqual(agent.cfg["experiment"]["steps_per_iteration"], 24)
            self.assertEqual(env._env.sgs_log_interval, 240)
            sys.modules["wandb"].init.assert_called_once()
            self.assertTrue(sys.modules["wandb"].init.call_args.kwargs["sync_tensorboard"])
            agent.track_data("fixture/metric", 1.)
            agent.post_interaction(239, 10000)
            agent.post_interaction(4999, 10000)
            agent.writer.flush()
            self.assertTrue((Path(agent.experiment_dir) / "checkpoints/agent_5000.pt").is_file())
            self.assertEqual(len(list(Path(directory).rglob("events.out.tfevents.*"))), 1)
            events = EventAccumulator(agent.experiment_dir).Reload()
            self.assertEqual([event.value for event in events.Scalars("iteration")], [10.])

    def test_no_logging_creates_no_event_file_or_checkpoint_and_keeps_training_task(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"wandb": Mock()}), \
                patch("skrl.agents.torch.base.SummaryWriter") as writer:
            teacher, env = _entry_fixture(directory, name="probe", wandb=True, sgs=True)
            trainer = teacher.get_trainer(is_eval=False, enable_logging=False)
            agent = trainer.agents
            self.assertEqual(agent.write_interval, 0)
            self.assertEqual(agent.checkpoint_interval, 0)
            self.assertFalse(agent.cfg["experiment"]["wandb"])
            self.assertFalse(trainer.teacher_args.wandb)
            self.assertFalse(env.constructed_config["env"]["wandb"])
            self.assertFalse(env.constructed_args.eval)
            self.assertEqual(env.constructed_config["env"]["sgs"]["mode"], "sgs")
            self.assertEqual(env.constructed_config["env"]["maxEpisodeLength"], 777)
            self.assertEqual(env.constructed_config["env"]["robotStartDistanceRange"], [2., 2.])
            self.assertTrue(env.constructed_config["env"]["posePrediction"]["enabled"])
            self.assertTrue(env.constructed_config["env"]["noVelocityObs"])
            self.assertIsNotNone(agent.memory)
            agent.track_data("fixture/metric", 1.)
            for step in (239, 4999, 9999):
                agent.post_interaction(step, 10000)
            writer.assert_not_called()
            sys.modules["wandb"].init.assert_not_called()
            sys.modules["wandb"].save.assert_not_called()
            self.assertEqual(list(Path(directory).rglob("events.out.tfevents.*")), [])
            self.assertFalse((Path(agent.experiment_dir) / "checkpoints").exists())

    def test_nested_probe_does_not_replace_main_event_file_at_iteration_100(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"wandb": Mock()}):
            teacher, _ = _entry_fixture(directory, wandb=False)
            main = teacher.get_trainer().agents
            self.addCleanup(main.writer.close)
            main.track_data("fixture/metric", 9.)
            main.post_interaction(2159, 10000)
            main.writer.flush()
            before = set(Path(directory).rglob("events.out.tfevents.*"))
            probe, _ = _entry_fixture(directory, name="main/world_model/probe.artifacts/runs/plan")
            probe.get_trainer(is_eval=False, enable_logging=False)
            main.track_data("fixture/metric", 10.)
            main.post_interaction(2399, 10000)
            main.writer.flush()
            self.assertEqual(set(Path(directory).rglob("events.out.tfevents.*")), before)
            events = EventAccumulator(main.experiment_dir).Reload()
            self.assertEqual([event.value for event in events.Scalars("iteration")], [90., 100.])
            self.assertEqual([event.value for event in events.Scalars("fixture/metric")], [9., 10.])

    def test_probe_session_explicitly_disables_logging_without_eval_task_overrides(self):
        spec = importlib.util.spec_from_file_location("probe_logging_entry", ROOT / "probe_teacher_world_model.py")
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        fake_teacher = ModuleType("train_multistate_DQ_teacher")
        original_get_params = Mock()
        fake_teacher.get_params = original_get_params
        captured = {}
        raw_env = SimpleNamespace(sgs_enabled=False)
        agent = SimpleNamespace(policy=Mock(), value=Mock(), set_running_mode=Mock())
        trainer = SimpleNamespace(agents=agent, env=SimpleNamespace(_env=raw_env, reset=Mock()))

        def get_trainer(*args, **kwargs):
            captured.update(kwargs=kwargs, args=fake_teacher.get_params())
            return trainer

        fake_teacher.get_trainer = get_trainer
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict("sys.modules", {"train_multistate_DQ_teacher": fake_teacher}), \
                patch("utils.wm_env.TeacherWorldModelAdapter", return_value=object()), \
                patch("utils.world_model.probes.IsaacGymProbeSession", return_value=SimpleNamespace()):
            cli = SimpleNamespace(num_envs=2, seed=43, device="cpu", workdir=directory,
                                  worker="base", family_sampling="original")
            bundle = {
                "teacher_arguments": {"wandb": True, "sgs_mode": "sgs", "future_pose_obs": True,
                                      "no_velocity_obs": True, "timesteps": 10000},
                "teacher_config": {"env": {"maxEpisodeLength": 777}},
                "teacher": {"policy": {"actor": 1}, "value": {"critic": 2}},
            }
            probe._new_session(cli, bundle, torch)
            self.assertEqual(captured["kwargs"], {"is_eval": False, "enable_logging": False})
            self.assertFalse(captured["args"].eval)
            self.assertFalse(captured["args"].wandb)
            self.assertEqual(captured["args"].sgs_mode, "sgs")
            self.assertTrue(captured["args"].future_pose_obs)
            self.assertTrue(captured["args"].no_velocity_obs)
            self.assertEqual(captured["args"].world_model, "off")
            self.assertIs(fake_teacher.get_params, original_get_params)
            agent.policy.load_state_dict.assert_called_once_with({"actor": 1})
            agent.value.load_state_dict.assert_called_once_with({"critic": 2})
            trainer.env.reset.assert_called_once()


if __name__ == "__main__":
    unittest.main()
