"""Exercise the actual v2 teacher entry and loop without starting Isaac Gym."""
import ast
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.config import get_params, load_cfg
from test_supermimic_teacher import FakeEnv


def load_trainer_module():
    spec = importlib.util.spec_from_file_location("v2_teacher_under_test", ROOT / "simple_supermimic_trainer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class V2Env(FakeEnv):
    """Observation storage is reused in-place, just like the real wrapper."""

    def __init__(self, num_envs, events, terminal_steps, *, dict_obs=False, four_tuple=False,
                 interrupt_at=None):
        super().__init__(num_envs, events, terminal_steps, interrupt_at)
        self.dict_obs = dict_obs
        self.four_tuple = four_tuple
        self._env.state = torch.zeros(num_envs, 34)
        self._env.num_envs = num_envs
        self._env.num_actions = self.num_actions
        self._env.num_obs = 6
        self._env.max_episode_length = 100
        self._env.context_data = torch.zeros(num_envs, 16)
        self._env.ee_pos = torch.zeros(num_envs, 3)
        self._env.ee_orn = torch.zeros(num_envs, 4)
        self._env.ee_orn[:, 3] = 1
        self.obs = torch.zeros(num_envs, 6)

    def _obs(self):
        self.obs[:, 0] = self._env.state[:, 0]
        return {"obs": self.obs} if self.dict_obs else self.obs

    def reset(self):
        ids = self.pending.nonzero(as_tuple=False).flatten()
        super().reset()
        self._env.context_data[ids] += 1
        return self._obs(), {}

    def step(self, action):
        _, reward, terminated, truncated, infos = super().step(action)
        if self.four_tuple:
            return self._obs(), reward, terminated | truncated, infos
        return self._obs(), reward, terminated, truncated, infos


class V2Harness:
    def __init__(self, directory, *, timesteps=13, stop_after=None, num_envs=3,
                 terminal_steps=(1, 10), wandb=True, dict_obs=False, four_tuple=False,
                 interrupt_at=None):
        options = ["--task", "B1Z1PickMulti", "--rl_device", "cpu", "--headless",
                   "--timesteps", str(timesteps), "--num_envs", str(num_envs),
                   "--experiment_dir", directory, "--wandb_name", "v2-entry-smoke"]
        if stop_after is not None:
            options += ["--stop_after", str(stop_after)]
        if wandb:
            options += ["--wandb"]
        self.args = get_params(options)
        self.cfg = load_cfg(ROOT / "data/cfg/DQ_teacher.yaml")
        self.cfg["env"]["numEnvs"] = num_envs
        self.events = []
        self.env = V2Env(num_envs, self.events, terminal_steps, dict_obs=dict_obs,
                         four_tuple=four_tuple, interrupt_at=interrupt_at)
        self.logs = []
        self.init_calls = []
        self.sdk_calls = []
        self.definitions = []
        self.module = load_trainer_module()
        harness = self

        class Capture:
            def __init__(self, env, device):
                self.env = env
                self.timesteps = torch.zeros(num_envs)
                self.prev_action = torch.zeros(num_envs, env.num_actions)

            def capture(self):
                harness.events.append(("capture", harness.env.steps))
                state = self.env.state.clone()
                state[:, 24:33] = self.prev_action
                state[:, 33] = self.timesteps
                return state

            def update_action(self, action):
                self.prev_action = action.clone()

            def increment_timestep(self, done=None):
                self.timesteps += 1
                if done is not None:
                    assert done.shape == (num_envs,)
                    self.timesteps[done.bool()] = 0

            def reset_episodes(self, ids):
                self.timesteps[ids] = 0
                self.prev_action[ids] = 0

        class Context:
            def __init__(self, env, device):
                self.env = env
                self.buffer = env.context_data.clone()

            def get_context(self):
                return self.buffer

            def reset_context(self, ids):
                self.buffer[ids] = self.env.context_data[ids]

        class MDP:
            state_dim, action_dim, context_dim, obs_dim = 47, 9, 16, 6

            def __init__(self, env, device):
                self.env = env

            def capture_state(self, physical, obs):
                assert physical.shape == (num_envs, 34)
                assert obs.shape == (num_envs, self.obs_dim)
                return torch.cat((physical, self.env.ee_pos, self.env.ee_orn, obs), dim=-1)

            def reward(self, state, action, next_state, context):
                return next_state[:, 0] - state[:, 0], torch.zeros(num_envs)

            def contract(self):
                return {"state_dim": self.state_dim, "obs_dim": self.obs_dim}

        class Agent:
            def __init__(self, mdp, device="cpu", **kwargs):
                self.mdp = mdp
                self.kwargs = kwargs
                self.transitions = []
                self.updates = 0
                self.world_model_updates = 0
                self.policy_updates = 0
                self.warmup_iterations = kwargs.get("warmup_iterations", 1000)
                self.replay_capacity = kwargs.get("replay_capacity", 1000000)
                self.checkpoints = []

            def select_action(self, obs, deterministic=False):
                assert obs.shape == (num_envs, 6)
                return torch.full((num_envs, 9), .25)

            def observe_transition(self, **transition):
                harness.events.append(("observe", harness.env.steps))
                self.transitions.append({key: value.clone() for key, value in transition.items()})

            def update(self):
                self.updates += 1
                self.world_model_updates += 1
                return {"phase": "warmup", "world_model_loss": .2}

            def get_stats(self):
                return {"phase": "warmup", "world_model_updates": self.world_model_updates,
                        "policy_updates": self.policy_updates,
                        "buffer_stats": {"buffer_size": len(self.transitions) * num_envs,
                                         "buffer_utilization": .01}}

            def save(self, path):
                self.checkpoints.append(path)

        def init(**kwargs):
            self.sdk_calls.append("init")
            self.init_calls.append(deepcopy(kwargs))
            return SimpleNamespace(url="https://wandb.ai/test/v2/runs/fake", offline=False)

        def log(info, step=None):
            self.sdk_calls.append("log")
            self.logs.append((deepcopy(info), step))

        def define_metric(*args, **kwargs):
            self.sdk_calls.append("define_metric")
            self.definitions.append((args, kwargs))

        self.wandb = SimpleNamespace(init=init, log=log, define_metric=define_metric,
                                     finish=lambda: self.sdk_calls.append("finish"))
        self.components = {"StateCapture": Capture, "Context": Context,
                           "DQRobotMDP": MDP, "SuperMimicAgentV2": Agent}

    def run(self):
        with ExitStack() as stack:
            stack.enter_context(patch.multiple(self.module, **self.components))
            stack.enter_context(patch.dict(sys.modules, {"wandb": self.wandb}))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return self.module.run_supermimic_training(self.env, self.args, self.cfg)


class SuperMimicV2LoopTests(unittest.TestCase):
    def test_vector_steps_and_stop_after_do_not_divide_by_environment_count(self):
        for timesteps, stop_after, expected in ((3, None, 3), (7, 2, 2), (3, 9, 3)):
            with self.subTest(timesteps=timesteps, stop_after=stop_after), tempfile.TemporaryDirectory() as directory:
                harness = V2Harness(directory, timesteps=timesteps, stop_after=stop_after,
                                    num_envs=8, terminal_steps=(), wandb=False)
                agent = harness.run()
                self.assertEqual(harness.env.steps, expected)
                self.assertEqual(agent.updates, expected)
                self.assertEqual(harness.args.timesteps, timesteps)
                self.assertTrue(harness.env.closed)
                self.assertEqual(Path(agent.checkpoints[-1]).name, f"agent_{expected * 8}")

    def test_terminal_capture_reset_context_clock_and_observation_aliasing(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = V2Harness(directory, timesteps=2, terminal_steps=(1,), wandb=False)
            agent = harness.run()
            first, second = agent.transitions
            self.assertEqual(first["state"].shape, (3, 47))
            for transition in agent.transitions:
                torch.testing.assert_close(transition["state"][:, 41:], transition["obs"])
                torch.testing.assert_close(transition["next_state"][:, 41:], transition["next_obs"])
            self.assertEqual(first["done"].shape, (3,))
            self.assertEqual(first["done"].dtype, torch.bool)
            self.assertEqual(first["done"].tolist(), [True, True, False])
            self.assertEqual(first["reward"].shape, (3,))
            self.assertEqual(first["reward"].tolist(), [1, 2, 3])
            self.assertEqual(first["obs"][:, 0].tolist(), [0, 10, 20])
            self.assertEqual(first["next_obs"][:, 0].tolist(), [1, 11, 21])
            self.assertEqual(first["next_state"][:, 0].tolist(), [1, 11, 21])
            self.assertEqual(first["next_state"][:, 33].tolist(), [1, 1, 1])
            self.assertEqual(second["state"][:, 0].tolist(), [-200, -201, 21])
            self.assertEqual(second["obs"][:, 0].tolist(), [-200, -201, 21])
            self.assertEqual(second["state"][:, 33].tolist(), [0, 0, 1])
            self.assertEqual(first["context"][:, 0].tolist(), [1, 1, 1])
            self.assertEqual(second["context"][:, 0].tolist(), [2, 2, 1])
            self.assertLess(harness.events.index(("observe", 1)), harness.events.index(("reset", 1)))

    def test_wrapped_dict_observations_and_legacy_four_value_steps(self):
        for four_tuple in (False, True):
            with self.subTest(four_tuple=four_tuple), tempfile.TemporaryDirectory() as directory:
                harness = V2Harness(directory, timesteps=2, dict_obs=True, four_tuple=four_tuple,
                                    terminal_steps=(1,), wandb=False)
                agent = harness.run()
                self.assertEqual(len(agent.transitions), 2)
                self.assertEqual(agent.transitions[0]["obs"].shape, (3, 6))
                self.assertEqual(agent.transitions[0]["done"].tolist(), [True, True, False])

    def test_preserves_reward_success_curves_and_final_partial_window(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = V2Harness(directory)
            harness.run()
            self.assertEqual([info["iteration"] for info, _ in harness.logs], [10, 13])
            self.assertEqual([info["global_step"] for info, _ in harness.logs], [30, 39])
            first, final = [info for info, _ in harness.logs]
            self.assertEqual(first["Reward / Instantaneous reward (mean)"], 6.5)
            self.assertEqual(first["Reward components / approach (mean)"], 7.5)
            self.assertEqual(first["Reward components / penalty (mean)"], -1.)
            self.assertEqual(first["Reward components / disabled (mean)"], 0.)
            self.assertEqual(first["episode_reward_mean"], 30.)
            self.assertEqual(first["episode_count"], 4)
            self.assertEqual(first["SuccessRate / sugar_box"], .25)
            self.assertEqual(first["SuccessRate / near_goal"], .5)
            self.assertEqual(final["Reward / Instantaneous reward (mean)"], 13.)
            self.assertEqual(final["episode_count"], 0)
            self.assertIn((("*",), {"step_metric": "iteration", "step_sync": True}), harness.definitions)
            config = harness.init_calls[0]["config"]
            self.assertEqual(config["arguments"]["timesteps"], 13)
            self.assertEqual(config["num_envs"], 3)
            self.assertIn("supermimic", config)
            self.assertIn("logging", config)
            self.assertEqual(harness.sdk_calls.count("finish"), 1)

    def test_no_wandb_calls_and_no_fabricated_success_before_any_episode(self):
        for wandb in (False, True):
            with self.subTest(wandb=wandb), tempfile.TemporaryDirectory() as directory:
                harness = V2Harness(directory, timesteps=3, terminal_steps=(), wandb=wandb)
                harness.run()
                if not wandb:
                    self.assertEqual(harness.sdk_calls, [])
                else:
                    self.assertEqual(len(harness.logs), 1)
                    info = harness.logs[0][0]
                    self.assertFalse(any(key.startswith("SuccessRate / ") for key in info))
                self.assertTrue(harness.env.closed)

    def test_interrupt_flushes_collected_steps_and_closes_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            harness = V2Harness(directory, timesteps=20, interrupt_at=14)
            with self.assertRaises(KeyboardInterrupt):
                harness.run()
            self.assertEqual([info["iteration"] for info, _ in harness.logs], [10, 13])
            self.assertEqual(harness.sdk_calls.count("finish"), 1)
            self.assertTrue(harness.env.closed)


class SuperMimicV2EntryTests(unittest.TestCase):
    def test_standalone_entry_delegates_to_the_shared_training_loop(self):
        tree = ast.parse((ROOT / "train_supermimic_v2.py").read_text())
        entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "train_supermimic_v2")
        delegated = [node for node in ast.walk(entry) if isinstance(node, ast.Return)
                     and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Name)
                     and node.value.func.id == "run_supermimic_training"]
        self.assertEqual(len(delegated), 1)
        self.assertEqual([node.id for node in delegated[0].value.args],
                         ["env", "args", "experiment_cfg"])
        direct_components = {node.func.id for node in ast.walk(entry)
                             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
        self.assertTrue({"StateCapture", "SuperMimicAgentV2"}.isdisjoint(direct_components))

    def test_actual_main_branch_imports_config_helpers_and_preserves_sgs_ekf(self):
        path = ROOT / "train_multistate_DQ_teacher.py"
        tree = ast.parse(path.read_text())
        # Execute the real config imports, not an injected apply_tsil_config:
        # omitting that import must reproduce the reported NameError.
        imports = [node for node in tree.body
                   if isinstance(node, ast.ImportFrom) and node.module == "utils.config"]
        main = next(node for node in tree.body if isinstance(node, ast.If)
                    and isinstance(node.test, ast.Compare)
                    and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")
        namespace = {"__name__": "__main__", "Path": Path, "torch": torch, "deepcopy": deepcopy}
        exec(compile(ast.Module(body=imports, type_ignores=[]), str(path), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            bank = Path(directory) / "bank.pt"
            bank.touch()
            args = get_params(["--use_supermimic", "--task", "B1Z1PickMulti", "--object_name", "sugar_box",
                               "--sgs_mode", "sgs", "--sgs_bank", str(bank), "--num_envs", "3",
                               "--timesteps", "20", "--stop_after", "2", "--future_pose_obs",
                               "--future_pose_horizons", "0.6", "0.9", "--no_velocity_obs"])
            predict = Mock(return_value=("camera", "grasps", "cube"))
            environment = object()
            create = Mock(return_value=environment)
            run = Mock(return_value=object())
            namespace.update(get_params=lambda: args, set_seed=lambda seed: None,
                             cprint=lambda *args, **kwargs: None,
                             teacher_config_path=lambda *args, **kwargs: str(ROOT / "data/cfg/DQ_teacher.yaml"),
                             get_predict_point=predict, create_env=create)
            with patch.dict(sys.modules, {"simple_supermimic_trainer": SimpleNamespace(run_supermimic_training=run)}):
                exec(compile(ast.Module(body=[main], type_ignores=[]), str(path), "exec"), namespace)
            run.assert_called_once()
            passed_env, passed_args, config = run.call_args[0]
            self.assertIs(passed_env, environment)
            self.assertIs(passed_args, args)
            self.assertEqual(config["env"]["numEnvs"], 3)
            self.assertEqual(config["env"]["sgs"]["mode"], "sgs")
            self.assertTrue(config["env"]["posePrediction"]["enabled"])
            self.assertEqual(config["env"]["posePrediction"]["horizons"], [.6, .9])
            self.assertFalse(config["env"]["tsil"]["enabled"])
            self.assertTrue(config["env"]["noVelocityObs"])
            self.assertEqual(predict.call_args.kwargs["object_indices"], [1])
            self.assertEqual(args.intervel, 23)


if __name__ == "__main__":
    unittest.main()
