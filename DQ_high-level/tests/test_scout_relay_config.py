"""Scout & Relay configuration and launcher checks (no torch or Isaac Gym needed)."""

from pathlib import Path
import subprocess
import unittest

import yaml

from utils.config import RELAY_DEFAULTS, SAPG_DEFAULTS, apply_scout_relay_config, get_params


ROOT = Path(__file__).resolve().parents[1]


def training_cfg(mode="sgs", tsil=False, **env):
    return {"env": dict({"sgs": {"mode": mode}, "tsil": {"enabled": tsil}}, **env)}


class ScoutRelayConfigTests(unittest.TestCase):
    def test_defaults_are_the_single_policy_sgs_teacher(self):
        cfg = training_cfg()
        sapg, relay = apply_scout_relay_config(cfg, get_params([]))
        self.assertEqual(sapg, SAPG_DEFAULTS)
        self.assertEqual(relay, RELAY_DEFAULTS)
        self.assertEqual(sapg["branches"], 1)
        self.assertFalse(relay["enabled"])
        self.assertIs(cfg["env"]["sapg"], sapg)
        self.assertIs(cfg["env"]["relay"], relay)

    def test_yaml_blocks_match_the_code_defaults(self):
        env = yaml.safe_load((ROOT / "data/cfg/DQ_teacher.yaml").read_text())["env"]
        self.assertEqual(env["sapg"], SAPG_DEFAULTS)
        self.assertEqual(env["relay"], RELAY_DEFAULTS)

    def test_command_line_overrides_yaml(self):
        cfg = training_cfg(sapg={"branches": 3, "latent_dim": 8}, relay={"capacity": 64})
        args = get_params(["--sapg_branches", "2", "--relay", "--relay_fractions", "0.6", "0.2",
                           "--scout_need_beta", "0", "--relay_alpha_max", "1",
                           "--scout_target_success", "0.3", "--relay_controller_state"])
        sapg, relay = apply_scout_relay_config(cfg, args)
        self.assertEqual(sapg["branches"], 2)
        self.assertEqual(sapg["latent_dim"], 8)
        self.assertEqual(sapg["scout_need_beta"], 0)
        self.assertEqual(sapg["scout_target_success"], [0.3])
        self.assertTrue(relay["enabled"])
        self.assertTrue(relay["controller_state"])
        self.assertEqual(relay["fractions"], [0.2, 0.6])
        self.assertEqual(relay["alpha_max"], 1.0)
        self.assertEqual(relay["capacity"], 64)
        self.assertFalse(apply_scout_relay_config(training_cfg(), get_params(["--no_relay"]))[1]["enabled"])

    def test_training_requirements(self):
        with self.assertRaisesRegex(ValueError, "sgs_mode"):
            apply_scout_relay_config(training_cfg("disabled"), get_params(["--sapg_branches", "2"]))
        with self.assertRaisesRegex(ValueError, "TSIL"):
            apply_scout_relay_config(training_cfg(tsil=True), get_params(["--relay", "--relay_source", "leader"]))
        with self.assertRaisesRegex(ValueError, "sapg_branches"):
            apply_scout_relay_config(training_cfg(), get_params(["--relay"]))
        # Self-relay without branches (variant G) and SAPG without relay (C) are valid.
        apply_scout_relay_config(training_cfg(), get_params(["--relay", "--relay_source", "leader"]))
        apply_scout_relay_config(training_cfg("uniform"), get_params(["--sapg_branches", "2"]))
        # Evaluation keeps the model settings without the training requirements.
        sapg, _ = apply_scout_relay_config(training_cfg("disabled"), get_params(["--sapg_branches", "2"]),
                                           is_eval=True)
        self.assertEqual(sapg["branches"], 2)

    def test_invalid_values_are_rejected(self):
        invalid = (
            ["--sapg_branches", "0"], ["--sapg_leader_fraction", "1"], ["--sapg_leader_fraction", "0"],
            ["--sapg_latent_dim", "0"], ["--sapg_off_policy_coef", "-1"], ["--sapg_mu_max", "nan"],
            ["--scout_need_beta", "-0.5"], ["--scout_target_success", "1.5"],
            ["--relay_alpha_max", "0"], ["--relay_alpha_max", "1.2"], ["--relay_tau_need", "-0.1"],
            ["--relay_fractions", "0.5", "0.5"], ["--relay_fractions", "1.0"], ["--relay_capacity", "0"],
            ["--relay_snapshot_interval", "0"], ["--relay_max_chains_per_config", "0"],
            ["--relay_capacity", "10"],  # 32 new chains x 4 fractions exceed the capacity
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    apply_scout_relay_config(training_cfg(), get_params(["--sapg_branches", "2"] + arguments
                                                                        if arguments[0] != "--sapg_branches"
                                                                        else arguments))
        for key, value in (("source", "both"), ("trigger", "timeout"), ("sampling", "softmax"),
                           ("quarantine_steps", -1), ("enabled", "yes")):
            with self.subTest(**{key: value}):
                with self.assertRaises(ValueError):
                    apply_scout_relay_config(training_cfg(relay={key: value}), get_params([]))

    def test_eval_restores_the_model_shape_arguments(self):
        source = (ROOT / "train_multistate_DQ_teacher.py").read_text()
        restore = source[source.index("def restore_eval_arguments"):source.index("def teacher_config_path")]
        self.assertIn('"sapg_branches"', restore)
        self.assertIn('"sapg_latent_dim"', restore)


class ScoutRelayLauncherTests(unittest.TestCase):
    def dry_run(self, variant):
        result = subprocess.run(
            ["bash", str(ROOT / "train_teacher_scout_relay.sh"), "--dry-run", "--no-wandb"],
            env={"VARIANT": variant, "PATH": "/usr/bin:/bin"}, capture_output=True, text=True)
        return result

    def test_variants_map_to_the_design_arms(self):
        expected = {
            "A": ["--sgs_mode", "uniform"],
            "B": [],
            "C": ["--sapg_branches", "2", "--scout_need_beta", "0"],
            "D": ["--sapg_branches", "2", "--scout_need_beta", "0", "--relay", "--relay_source", "leader"],
            "E": ["--sapg_branches", "2", "--scout_need_beta", "0", "--relay", "--relay_source", "scouts"],
            "F": ["--sapg_branches", "2", "--scout_need_beta", "1", "--relay", "--relay_source", "scouts"],
            "G": ["--relay", "--relay_source", "leader"],
        }
        for variant, arguments in expected.items():
            with self.subTest(variant=variant):
                result = self.dry_run(variant)
                self.assertEqual(result.returncode, 0, result.stderr)
                command = result.stdout.splitlines()[-1].split()
                tail = command[command.index("--headless") + 1:]
                self.assertEqual(tail, arguments)
                # Every arm shares the T1 baseline configuration.
                for flag in ("--action_scaling", "--critic_time_obs", "--future_pose_obs", "--no_velocity_obs"):
                    self.assertIn(flag, command)
        self.assertNotEqual(self.dry_run("X").returncode, 0)


if __name__ == "__main__":
    unittest.main()
