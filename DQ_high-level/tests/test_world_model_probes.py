"""Finite-action-effect validation protocol on CPU and CUDA.

The simulator here is deliberately labelled Toy, so its reports cannot pass
the production IsaacGym gate. Actual IsaacGym evidence needs an independent
simulator run; these tests verify replay/control/hash and statistics logic.
"""

from copy import deepcopy
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.world_model.probes import (ACTION_SEMANTICS, CONTINUOUS_INDICES, REPORT_SCHEMA,
                                      ActionEffectProbeConfig, ProbeStep,
                                      analyze_action_effects, balanced_family_config_ids,
                                      bounded_probe_action, configuration_fingerprint,
                                      load_verified_report, model_fingerprint,
                                      random_continuous_directions,
                                      run_action_effect_probe, write_report)


class ToyAdapter:
    def reward(self, previous, next_state, action, context):
        return -(next_state[:, 0]-context[:, 0]).square()


class ToyProbeModel(nn.Module):
    def __init__(self, adapter, gain=1.):
        super().__init__()
        self.spec = adapter
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self.register_buffer("weights", torch.tensor([1., .7, -.3, .4, -.5, .6, .3, .8, -.2]))

    def step(self, state, action, context, dt):
        if hasattr(self, "probe_actions"):
            self.probe_actions.append(action.detach().clone())
        movement = (action*self.weights).sum(-1, keepdim=True)*dt*self.gain
        return torch.cat((state[:, :1]+movement, state[:, 1:]+1), -1)


class ToySession:
    isolated = True
    simulator_name = "Toy"

    def __init__(self, factory, drift=0., gain=1.):
        self.factory, self.adapter = factory, ToyAdapter()
        self.state = torch.cat((torch.randn(factory.batch, 1, device=factory.device)*.02,
                                torch.zeros(factory.batch, 1, device=factory.device)), -1)
        self.context = torch.full((factory.batch, 1), .2, device=factory.device)
        self.model = ToyProbeModel(self.adapter, gain=gain).to(factory.device)
        self.drift = drift

    def capture(self):
        return self.state.clone(), self.context.clone()

    def step(self, action):
        self.factory.real_actions.append(action.detach().clone())
        dt = torch.full((self.factory.batch, 1), random.choice([.14, .16]), device=action.device)
        previous = self.state
        self.state = self.model.step(previous, action, self.context, dt)
        reward = self.adapter.reward(previous, self.state, action, self.context)+self.drift
        self.factory.devices.append(self.state.device.type)
        return ProbeStep(self.state.clone(), self.context.clone(), reward,
                         dt, torch.zeros(self.factory.batch, device=action.device, dtype=torch.bool))

    def close(self):
        self.factory.closed += 1


class Factory:
    def __init__(self, device, batch=128, drift=False):
        self.device, self.batch = device, batch
        self.calls, self.closed = 0, 0
        self.devices, self.real_actions, self.predicted_actions = [], [], []
        self.drift = drift

    def __call__(self, seed):
        self.calls += 1
        return ToySession(self, drift=self.calls*.001 if self.drift else 0.)


class ProbeChecks:
    device = "cpu"

    def execute(self, gain=1., directions=None, drift=False, base=None):
        if base is None:
            base = torch.zeros(128, 9, device=self.device)
            base[:, 6] = .5
        prefix = torch.zeros(3, 128, 9, device=self.device)
        prefix[:, :, 6] = .5
        factory = Factory(self.device, drift=drift)
        if directions is None:
            generator = torch.Generator(device=self.device).manual_seed(99)
            directions = random_continuous_directions(base, generator,
                amplitudes=[.02, .02, .02, .03, .03, .03, .05, .04, .04])
        def model_factory(adapter):
            model = ToyProbeModel(adapter, gain=gain).to(self.device)
            model.probe_actions = factory.predicted_actions
            return model
        report = run_action_effect_probe(factory, model_factory,
                                         prefix, base, directions, seed=123,
                                         configuration={"task": "toy", "duration": "7_or_8_cycles"})
        return report, factory

    def test_true_effects_pass_and_all_branches_are_independent_and_closed(self):
        report, factory = self.execute()
        self.assertTrue(report["passed"], report["reasons"])
        self.assertEqual(report["metrics"]["valid_pairs"], 128)
        self.assertGreater(report["metrics"]["selected_gain_lcb95"], 0)
        self.assertAlmostEqual(report["metrics"]["sign_agreement"], 1.)
        self.assertEqual(factory.calls, 4)
        self.assertEqual(factory.closed, 4)
        self.assertEqual(set(factory.devices), {torch.device(self.device).type})
        self.assertEqual(report["schema"], REPORT_SCHEMA)
        self.assertEqual(report["metadata"]["directions_cover"], list(range(9)))
        self.assertTrue(report["metadata"]["gripper_perturbed"])
        self.assertEqual(report["metadata"]["action_semantics"], ACTION_SEMANTICS)

    def test_low_prediction_error_is_not_enough_when_action_effect_sign_is_wrong(self):
        report, _ = self.execute(gain=-1.)
        self.assertFalse(report["passed"])
        self.assertIn("action_effect_sign_agreement", report["reasons"])
        self.assertIn("no_positive_real_gain_lower_bound", report["reasons"])

    def test_zero_effects_and_nonrepeatable_simulator_cannot_enable_bptt(self):
        report, _ = self.execute(directions=torch.zeros(128, 9, device=self.device))
        self.assertFalse(report["passed"])
        self.assertIn("insufficient_nonzero_action_effects", report["reasons"])
        self.assertIsNone(report["metrics"]["selected_gain_lcb95"])
        noisy_report, _ = self.execute(drift=True)
        self.assertFalse(noisy_report["passed"])
        self.assertEqual(noisy_report["metrics"]["rejected_repeatability"], 128)

    def test_probe_restores_all_caller_rng_streams(self):
        python_state, numpy_state = random.getstate(), np.random.get_state()
        cpu_state = torch.get_rng_state().clone()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        self.execute()
        self.assertEqual(python_state, random.getstate())
        after_numpy = np.random.get_state()
        self.assertEqual(numpy_state[0], after_numpy[0])
        np.testing.assert_array_equal(numpy_state[1], after_numpy[1])
        self.assertEqual(numpy_state[2:], after_numpy[2:])
        torch.testing.assert_close(cpu_state, torch.get_rng_state(), rtol=0, atol=0)
        for before, after in zip(cuda_state, torch.cuda.get_rng_state_all()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_report_gate_checks_model_config_simulator_integrity_and_age(self):
        report, _ = self.execute()
        configuration = {"task": "toy", "duration": "7_or_8_cycles"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"probe.json"
            write_report(path, report)
            accepted, reasons, _ = load_verified_report(path, report["model_sha256"], configuration,
                                                         required_simulator="Toy")
            self.assertTrue(accepted, reasons)
            accepted, reasons, _ = load_verified_report(path, report["model_sha256"], configuration)
            self.assertFalse(accepted)
            self.assertIn("wrong_validation_simulator", reasons)
            accepted, reasons, _ = load_verified_report(path, "different_weights", configuration,
                                                         required_simulator="Toy")
            self.assertFalse(accepted)
            self.assertIn("model_snapshot_changed", reasons)
            accepted, reasons, _ = load_verified_report(path, report["model_sha256"], {"task": "other"},
                                                         required_simulator="Toy")
            self.assertFalse(accepted)
            self.assertIn("task_configuration_changed", reasons)
            accepted, reasons, _ = load_verified_report(path, report["model_sha256"], configuration,
                max_age_seconds=10, required_simulator="Toy", now=report["created_unix"]+11)
            self.assertFalse(accepted)
            self.assertIn("report_expired_or_future_dated", reasons)
            report["passed"] = False
            write_report(path, report)
            accepted, reasons, _ = load_verified_report(path, report["model_sha256"], configuration,
                                                         required_simulator="Toy")
            self.assertFalse(accepted)
            self.assertIn("report_checksum_mismatch", reasons)

    def test_model_hash_changes_and_gripper_only_probe_is_measured_but_not_full_coverage(self):
        model = ToyProbeModel(ToyAdapter()).to(self.device)
        before = model_fingerprint(model)
        with torch.no_grad():
            model.gain.add_(.01)
        self.assertNotEqual(before, model_fingerprint(model))
        directions = torch.zeros(128, 9, device=self.device)
        directions[:, 6] = .01
        report, _ = self.execute(directions=directions)
        self.assertGreater(report["metrics"]["informative_pairs"], 0)
        self.assertAlmostEqual(report["metrics"]["sign_agreement"], 1.)
        self.assertEqual(report["metadata"]["directions_cover"], [6])
        self.assertTrue(report["metadata"]["gripper_perturbed"])
        self.assertFalse(report["passed"])
        self.assertIn("incomplete_continuous_action_coverage", report["reasons"])

    def test_default_direction_includes_finite_gripper_opening_perturbations(self):
        base = torch.zeros(128, 9, device=self.device)
        direction = random_continuous_directions(base, torch.Generator(device=self.device).manual_seed(7))
        self.assertTrue(bool((direction[:, 6] != 0).all()))
        self.assertGreater(float(direction[:, 6].abs().max()), .01)
        self.assertLessEqual(float(direction[:, 6].abs().max()), .05)

    def test_boundary_openings_are_bounded_identically_for_real_and_model_branches(self):
        base = torch.zeros(128, 9, device=self.device)
        base[1::2, 6] = 1.
        directions = random_continuous_directions(base, torch.Generator(device=self.device).manual_seed(7))
        directions[:, 6] = .05
        report, factory = self.execute(base=base, directions=directions)
        self.assertEqual(report["metadata"]["directions_cover"], list(range(9)))
        self.assertTrue(report["metadata"]["gripper_perturbed"])
        for i, multiplier in enumerate((0., 1., -1., 0.)):
            real = factory.real_actions[i * 4 + 3]
            predicted = factory.predicted_actions[i]
            torch.testing.assert_close(real, predicted, rtol=0, atol=0)
            torch.testing.assert_close(real, bounded_probe_action(base, directions, multiplier), rtol=0, atol=0)
            self.assertTrue(bool(((real[:, 6] >= 0) & (real[:, 6] <= 1)).all()))
        self.assertAlmostEqual(float(factory.real_actions[7][0, 6]), .05)
        self.assertAlmostEqual(float(factory.real_actions[11][1, 6]), .95)

    def test_correctly_signed_legacy_or_eight_coordinate_reports_cannot_enable_bptt(self):
        report, _ = self.execute()
        configuration = report["configuration"]
        cases = []
        legacy = deepcopy(report)
        legacy["schema"] = "dq-action-effect-probe-v1"
        legacy["metadata"]["directions_cover"] = [0, 1, 2, 3, 4, 5, 7, 8]
        legacy["metadata"]["gripper_perturbed"] = False
        cases.append((legacy, "report_schema_mismatch"))
        incomplete = deepcopy(report)
        incomplete["metadata"]["directions_cover"] = [0, 1, 2, 3, 4, 5, 7, 8]
        cases.append((incomplete, "incomplete_continuous_action_coverage"))
        no_gripper = deepcopy(report)
        no_gripper["metadata"]["gripper_perturbed"] = False
        cases.append((no_gripper, "continuous_gripper_not_probed"))
        old_semantics = deepcopy(report)
        old_semantics["metadata"].pop("action_semantics")
        cases.append((old_semantics, "gripper_action_semantics_mismatch"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.json"
            for changed, expected_reason in cases:
                changed.pop("report_sha256")
                changed["report_sha256"] = configuration_fingerprint(changed)
                write_report(path, changed)
                accepted, reasons, _ = load_verified_report(
                    path, report["model_sha256"], configuration, required_simulator="Toy")
                self.assertFalse(accepted)
                self.assertIn(expected_reason, reasons)
                self.assertNotIn("report_checksum_mismatch", reasons)

    def test_probe_balances_existing_families_without_dropping_small_contact_pools(self):
        bank = torch.cat((torch.zeros(900, device=self.device, dtype=torch.long),
                          torch.ones(40, device=self.device, dtype=torch.long),
                          torch.full((10,), 2, device=self.device, dtype=torch.long)))
        indices = balanced_family_config_ids(bank, 64, 9)
        counts = torch.bincount(bank[indices], minlength=3)
        self.assertLessEqual(int(counts.max()-counts.min()), 1)
        self.assertTrue(bool((counts >= 21).all()))
        torch.testing.assert_close(indices, balanced_family_config_ids(bank, 64, 9), rtol=0, atol=0)

    def test_family_coverage_is_reported_without_requiring_nonzero_reward_in_every_family(self):
        real = torch.linspace(-1., 1., 96, device=self.device)
        families = torch.arange(96, device=self.device) % 3
        real[families == 2] = 0.  # A valid but locally flat near-goal reward.
        zero = torch.zeros_like(real)
        alive = torch.ones_like(real, dtype=torch.bool)
        metadata = {"family_names": ["full_task", "pregrasp", "near_goal"],
                    "directions_cover": list(CONTINUOUS_INDICES), "gripper_perturbed": True,
                    "action_semantics": ACTION_SEMANTICS}
        arguments = (real, -real, zero, zero, real, -real, zero, zero, zero,
                     alive, "weights", {"task": "toy"})
        report = analyze_action_effects(*arguments, metadata=metadata, families=families)
        self.assertTrue(report["passed"], report["reasons"])
        self.assertEqual(report["metrics"]["reset_families"]["near_goal"]["valid_pairs"], 32)
        self.assertEqual(report["metrics"]["reset_families"]["near_goal"]["informative_pairs"], 0)
        report = analyze_action_effects(*arguments, metadata=metadata, families=families.clamp_max(1))
        self.assertFalse(report["passed"])
        self.assertIn("missing_reproducible_reset_family:near_goal", report["reasons"])


class CPUChecks(ProbeChecks, unittest.TestCase):
    pass


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CUDAChecks(ProbeChecks, unittest.TestCase):
    device = "cuda:0"


if __name__ == "__main__":
    unittest.main()
