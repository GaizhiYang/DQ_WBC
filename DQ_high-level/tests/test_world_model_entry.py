"""Entry configuration, pre-reset collection, and isolated probe lifecycle."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.wm_entry import (add_world_model_arguments, configure_world_model,
                             install_probe_callback, run_probe_subprocess,
                             save_world_model_artifacts)
from utils.wrapper import IsaacGymPreview3Wrapper


def arguments(**changes):
    defaults = dict(world_model=None, wm_config=None, task="B1Z1PickMulti",
                    resume=False, rl_device="cuda:0", sim_device="cuda:0")
    defaults.update(changes)
    return SimpleNamespace(**defaults)


def configuration(**settings):
    result = {"env": {"tsil": {"enabled": False}, "sgs": {"mode": "sgs"},
                       "world_model": settings}, "reward": {"pick_up": 50}}
    return result


class ConfigurationTests(unittest.TestCase):
    def test_supermimic_does_not_add_or_rewrite_execution_clip(self):
        for mode in ("fit", "hybrid"):
            for supplied in (None, float("inf"), .25, 1., 2.5):
                with self.subTest(mode=mode, clip=supplied):
                    cfg = configuration(mode=mode)
                    if supplied is not None:
                        cfg["env"]["clipActions"] = supplied
                    configure_world_model(cfg, arguments())
                    if supplied is None:
                        self.assertNotIn("clipActions", cfg["env"])
                    else:
                        self.assertEqual(cfg["env"]["clipActions"], supplied)

    def test_off_legacy_and_evaluation_do_not_rewrite_execution_bounds(self):
        for mode, recipe, evaluation in (("off", "supermimic", False),
                                         ("hybrid", "supermimic", True),
                                         ("hybrid", "gated", False)):
            for supplied in (None, float("inf"), .3, 1.):
                with self.subTest(mode=mode, recipe=recipe, evaluation=evaluation, clip=supplied):
                    cfg = configuration(mode=mode, recipe=recipe)
                    if supplied is not None:
                        cfg["env"]["clipActions"] = supplied
                    configure_world_model(cfg, arguments(), is_eval=evaluation)
                    if supplied is None:
                        self.assertNotIn("clipActions", cfg["env"])
                    else:
                        self.assertEqual(cfg["env"]["clipActions"], supplied)

    def test_default_recipe_and_explicit_legacy_recipe(self):
        current = configure_world_model(configuration(mode="hybrid"), arguments())
        self.assertEqual(current["recipe"], "supermimic")
        self.assertEqual(current["hidden"], [2048, 2048, 2048])
        self.assertEqual((current["batch_size"], current["batches_per_update"], current["imagination_batch"]),
                         (2048, 10, 4096))
        self.assertEqual((current["replay_envs"], current["low_replay_capacity"]), (4096, 1000))
        self.assertEqual(current["value_source"], "imagined")
        self.assertEqual(current["warmup_steps"] * 8, 40)
        self.assertEqual(current["updates_per_low_step"], .4)
        self.assertTrue(current["fixed_preprocessing"] and current["compact_low_replay"])
        self.assertFalse(current["kl_rollback"] or current["balance_families"])
        self.assertEqual(current["state_loss_reduction"], "sum")
        self.assertEqual(current["auxiliary_weight"], 0.)
        self.assertEqual(current["action_coordinates"], "normalized_commands")
        self.assertEqual(current["initial_action_std"], .1)
        self.assertEqual(current["actor_last_gain"], .01)
        self.assertEqual(current["action_rate_mode"], "squared_all")
        self.assertEqual(current["action_rate_weight"], -.002)
        old = configure_world_model(configuration(mode="hybrid"), arguments(wm_recipe="gated"))
        self.assertEqual(old["recipe"], "gated")
        self.assertEqual(old["value_source"], "ppo")
        self.assertEqual(old["hidden"], [512, 512, 512])
        self.assertTrue(old["kl_rollback"])
        self.assertEqual(old["state_loss_reduction"], "mean")
        self.assertEqual(old["auxiliary_weight"], 1.)
        self.assertEqual(old["action_coordinates"], "physical")
        self.assertIsNone(old["initial_action_std"])
        self.assertIsNone(old["actor_last_gain"])
        self.assertEqual(old["action_rate_mode"], "base_l2")
        self.assertIsNone(old["action_rate_weight"])
        # Saved resolved configs keep their preprocessing even when evaluation
        # turns model training off; no model buffers are needed to run Actor.
        evaluated = configure_world_model(configuration(**current), arguments(), is_eval=True)
        self.assertEqual(evaluated["mode"], "off")
        self.assertTrue(evaluated["fixed_preprocessing"])

    def test_off_and_evaluation_preserve_baseline_without_gpu_or_training_dependencies(self):
        cfg = configuration()
        before = deepcopy(cfg)
        settings = configure_world_model(cfg, arguments(task="Other", rl_device="cpu", sim_device="cpu"))
        self.assertEqual(settings, {"mode": "off"})
        before["env"]["world_model"] = {"mode": "off"}
        self.assertEqual(cfg, before)
        cfg = configuration(mode="hybrid", candidate_checkpoint="missing.pt")
        settings = configure_world_model(cfg, arguments(rl_device="cpu", sim_device="cpu", resume=True),
                                          is_eval=True)
        self.assertEqual(settings["mode"], "off")

    def test_yaml_then_cli_precedence_and_scientific_notation_conversion(self):
        parser = argparse.ArgumentParser()
        add_world_model_arguments(parser)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text("world_model:\n  mode: fit\n  learning_rate: 2e-4\n  batch_size: 64\n"
                            "  state_loss_reduction: mean\n  auxiliary_weight: 2e-1\n")
            parsed = parser.parse_args(["--wm_config", str(path), "--world_model", "hybrid",
                                        "--wm_batch_size", "12", "--wm_warmup_steps", "0",
                                        "--wm_state_loss_reduction", "sum", "--wm_auxiliary_weight", "0.3"])
            args = arguments(**vars(parsed))
            settings = configure_world_model(configuration(mode="off", batch_size=32), args)
        self.assertEqual(settings["mode"], "hybrid")
        self.assertEqual(settings["batch_size"], 12)
        self.assertEqual(settings["warmup_steps"], 0)
        self.assertIsInstance(settings["learning_rate"], float)
        self.assertEqual(settings["learning_rate"], 2e-4)
        self.assertTrue(settings["fixed8"])
        self.assertEqual(settings["window"], 8)
        self.assertEqual(settings["horizon"], 4)
        self.assertEqual(settings["state_loss_reduction"], "sum")
        self.assertEqual(settings["auxiliary_weight"], .3)

    def test_explicit_model_loss_overrides_recipe_defaults(self):
        settings = configure_world_model(configuration(mode="fit", state_loss_reduction="mean",
                                                        auxiliary_weight="2e-1"), arguments())
        self.assertEqual(settings["state_loss_reduction"], "mean")
        self.assertEqual(settings["auxiliary_weight"], .2)
        settings = configure_world_model(configuration(mode="fit", recipe="gated",
            state_loss_reduction="sum", auxiliary_weight=0), arguments())
        self.assertEqual(settings["state_loss_reduction"], "sum")
        self.assertEqual(settings["auxiliary_weight"], 0.)

    def test_normalized_action_settings_and_reward_override_survive_evaluation(self):
        cfg = configuration(mode="hybrid")
        cfg["reward"]["scales"] = {"action_rate": -.001, "pick_up": 50.}
        settings = configure_world_model(cfg, arguments())
        self.assertEqual(cfg["reward"]["scales"], {"action_rate": -.002, "pick_up": 50.})
        saved = deepcopy(settings)
        settings = configure_world_model(cfg, arguments(), is_eval=True)
        self.assertEqual(settings["mode"], "off")
        for field in ("action_coordinates", "initial_action_std", "actor_last_gain",
                      "action_rate_mode", "action_rate_weight"):
            self.assertEqual(settings[field], saved[field])
        self.assertEqual(cfg["reward"]["scales"]["action_rate"], -.002)

        parser = argparse.ArgumentParser()
        add_world_model_arguments(parser)
        parsed = parser.parse_args(["--wm_action_rate_weight", "-0.004",
                                    "--wm_initial_action_std", "0.2", "--wm_actor_last_gain", "0.03"])
        cfg = configuration(mode="fit")
        settings = configure_world_model(cfg, arguments(**vars(parsed)))
        self.assertEqual(settings["initial_action_std"], .2)
        self.assertEqual(settings["actor_last_gain"], .03)
        self.assertEqual(cfg["reward"]["scales"]["action_rate"], -.004)

    def test_legacy_and_explicit_base_action_rate_preserve_existing_reward(self):
        for options in ({"recipe": "gated"}, {"action_rate_mode": "base_l2"}):
            cfg = configuration(mode="fit", **options)
            cfg["reward"]["scales"] = {"action_rate": -.007}
            configure_world_model(cfg, arguments())
            self.assertEqual(cfg["reward"]["scales"]["action_rate"], -.007)

    def test_gpu_and_mainline_boundaries_fail_before_environment_creation(self):
        for overrides, text in (({"rl_device": "cpu"}, "GPU"),
                                ({"sim_device": "cuda:1"}, "same CUDA"),
                                ({"rl_device": "cuda-invalid"}, "GPU"),
                                ({"task": "Other"}, "B1Z1PickMulti"),
                                ({"resume": True}, "resume")):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, text):
                configure_world_model(configuration(mode="fit"), arguments(**overrides))
        cfg = configuration(mode="hybrid")
        cfg["env"]["tsil"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "TSIL"):
            configure_world_model(cfg, arguments())
        settings = configure_world_model(configuration(mode="fit"), arguments(rl_device="cuda", sim_device="cuda:0"))
        self.assertEqual(settings["probe_device"], "cuda:0")

    def test_bad_model_settings_and_unpaired_report_are_rejected(self):
        bad = ({"wrong_option": 1}, {"window": True}, {"learning_rate": "nan"},
               {"holdout_fraction": 1.}, {"recent_fraction": -0.1},
               {"hidden": [32, 0]}, {"probe_timeout_seconds": 0},
               {"window": 9, "low_replay_capacity": 8}, {"use_tail_value": "false"},
               {"recipe": "unknown"}, {"actor_betas": [.9, 1.]},
               {"lr_schedule": "bad"}, {"model_microbatch_size": -1},
               {"updates_per_low_step": 0}, {"critic_gae_lambda": 2},
               {"fixed_preprocessing": "false"}, {"observation_noise_scale": -.1},
               {"state_loss_reduction": "invalid"}, {"state_loss_reduction": None},
               {"auxiliary_weight": -1.}, {"auxiliary_weight": True},
               {"auxiliary_weight": "nan"}, {"auxiliary_weight": float("inf")},
               {"action_coordinates": "bad"}, {"action_rate_mode": "bad"},
               {"initial_action_std": 0}, {"initial_action_std": "nan"},
               {"actor_last_gain": -1}, {"actor_last_gain": True},
               {"action_rate_weight": .01}, {"action_rate_weight": float("-inf")})
        for invalid in bad:
            with self.subTest(settings=invalid), self.assertRaises(ValueError):
                configure_world_model(configuration(mode="fit", **invalid), arguments())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.pt"
            path.touch()
            with self.assertRaisesRegex(ValueError, "matching probe report"):
                configure_world_model(configuration(mode="hybrid", validated_checkpoint=str(path)), arguments())


class FixtureAdapter:
    def __init__(self, environment):
        self.env = environment

    def capture(self, ids):
        context = torch.zeros(len(ids), 4, device=self.env.device)
        context[:, 0] = ids
        context[:, 3] = self.env.context_gate
        return self.env.state[ids].clone(), context

    def task_outcome(self, previous, state, action, context):
        return {"timeout": self.env.timeout[context[:, 0].long()]}


class FixtureEnvironment:
    """A vector environment with explicit deferred resets, like this trainer."""

    def __init__(self, device="cpu", enabled=False):
        self.device, self.num_envs, self.num_states = device, 3, 0
        self.state = torch.zeros(3, 1, device=device)
        self.actions = torch.zeros(3, 9, device=device)
        self.reset_buf = torch.zeros(3, device=device, dtype=torch.bool)
        self.timeout = self.reset_buf.clone()
        self.pick_success = self.reset_buf.clone()
        self.active_config_id = torch.tensor([2, 0, 1], device=device)
        self.sgs_bank = SimpleNamespace(tensors={"family_id": torch.tensor([0, 1, 2], device=device)})
        self.dt, self.control_freq_low = .02, 7
        self.context_gate = 0.
        if enabled:
            self.world_model_adapter = FixtureAdapter(self)
            self.world_model_env_ids = torch.tensor([0, 2], device=device)

    def step(self, actions):
        self.actions.copy_(actions.clamp(-1., 1.))
        self.state.add_(1.)
        self.context_gate += 1.
        self.reset_buf[:] = torch.tensor([True, False, True], device=self.device)
        self.timeout[:] = torch.tensor([False, False, True], device=self.device)
        self.pick_success[:] = torch.tensor([True, False, False], device=self.device)
        return {"obs": self.state.clone()}, torch.tensor([2., 3., 4.], device=self.device), self.reset_buf, {"baseline": 1}

    def reset(self):
        self.state[self.reset_buf] = 100.
        self.active_config_id[self.reset_buf] = 0
        self.reset_buf.zero_()
        return {"obs": self.state.clone()}


class CollectorTests(unittest.TestCase):
    def test_collector_keeps_terminal_state_family_context_and_original_ppo_semantics(self):
        devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                baseline = IsaacGymPreview3Wrapper(FixtureEnvironment(device))
                raw = FixtureEnvironment(device, enabled=True)
                wrapper = IsaacGymPreview3Wrapper(raw)
                action = torch.full((3, 9), 3., device=device)
                expected = baseline.step(action)
                actual = wrapper.step(action)
                for left, right in zip(expected[:4], actual[:4]):
                    torch.testing.assert_close(left, right)
                transition = actual[-1]["world_model"]
                torch.testing.assert_close(transition["family"], torch.tensor([2, 1], device=device))
                torch.testing.assert_close(transition["state"], torch.zeros(2, 1, device=device))
                torch.testing.assert_close(transition["next_state"], torch.ones(2, 1, device=device))
                torch.testing.assert_close(transition["context"][:, 3], torch.zeros(2, device=device))
                torch.testing.assert_close(transition["actions"], torch.ones(2, 9, device=device))
                torch.testing.assert_close(transition["dt"], torch.full((2, 1), .14, device=device))
                self.assertEqual(transition["terminated"].tolist(), [True, False])
                self.assertEqual(transition["truncated"].tolist(), [False, True])
                # PPO retains baseline all-done/zero-truncation semantics; only
                # model replay distinguishes the natural terminal from timeout.
                self.assertEqual(actual[2].flatten().tolist(), [True, False, True])
                self.assertFalse(bool(actual[3].any()))
                wrapper.reset()
                self.assertEqual(wrapper._world_model_episodes.tolist(), [1, 0, 1])
                second = wrapper.step(action)[-1]["world_model"]
                torch.testing.assert_close(second["state"], torch.full((2, 1), 100., device=device))
                torch.testing.assert_close(second["episode_id"], torch.ones(2, device=device, dtype=torch.long))
                torch.testing.assert_close(second["family"], torch.zeros(2, device=device, dtype=torch.long))
                # Simulator reset/reuse cannot mutate a recorded terminal target.
                torch.testing.assert_close(transition["next_state"], torch.ones(2, 1, device=device))


class ProbeLifecycleTests(unittest.TestCase):
    def test_fit_and_manual_modes_do_not_install_automatic_gpu_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode, interval in (("fit", 10), ("hybrid", 0)):
                agent = SimpleNamespace(wm_config={"mode": mode, "probe_interval": interval})
                install_probe_callback(agent, directory, 43)
                self.assertIsNone(agent.maybe_validate)
            self.assertFalse((Path(directory) / "world_model").exists())

    @unittest.skipUnless(os.name == "posix", "Process-group cleanup uses POSIX sessions")
    def test_probe_timeout_kills_its_worker_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            child_file = Path(directory) / "child.pid"
            script = (
                "import pathlib, subprocess, sys, time; "
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(30)"
            )
            with open(os.devnull, "w") as stream:
                with self.assertRaises(subprocess.TimeoutExpired):
                    run_probe_subprocess([sys.executable, "-c", script, str(child_file)],
                                         directory, stream, timeout=1.)
            self.assertTrue(child_file.is_file())
            pid = int(child_file.read_text())
            status = Path("/proc") / str(pid) / "stat"
            # A reparented worker may briefly remain as a zombie until init
            # reaps it, but must not remain executing or holding a CUDA context.
            if status.exists():
                self.assertEqual(status.read_text().split()[2], "Z")

    def test_artifact_json_handles_nonfinite_failure_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = []
            agent = SimpleNamespace(wm_candidate=object(), device=torch.device("cpu"),
                wm_status="heldout_error_rejected", wm_adapter=SimpleNamespace(state_dim=1000),
                wm_replay=SimpleNamespace(steps=100), wm_trainer=SimpleNamespace(updates=5),
                wm_has_validated_snapshot=True,
                last_world_model_metrics={"bad_nan": float("nan"), "bad_inf": float("inf"), "good": .2},
                save_candidate=lambda path: saved.append(path))
            trainer = SimpleNamespace(agents=agent,
                teacher_args=SimpleNamespace(experiment_dir=directory, wandb_name="run"))
            save_world_model_artifacts(trainer)
            status = json.loads((Path(directory) / "run/world_model/status.json").read_text())
            self.assertEqual(status["metrics"], {"bad_nan": None, "bad_inf": None, "good": .2})
            self.assertEqual(set(status["nonfinite_metrics"]), {"bad_nan", "bad_inf"})
            self.assertEqual(len(saved), 1)


if __name__ == "__main__":
    unittest.main()
