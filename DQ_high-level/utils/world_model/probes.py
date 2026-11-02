"""Isolated finite-action-effect validation; never step/restore the PPO simulator.

Isaac Gym exposes root/DOF state setters, but not a complete PhysX contact
solver snapshot. Restoring tensors, histories and RNG is therefore NOT a
reliable same-state branch of the training simulator. This module instead
creates a fresh dedicated simulator for each branch and replays an identical
action prefix with common random numbers. It rejects mismatched start states,
physical contexts, durations, and excessive baseline repeatability error.

The report validates finite short-term reward effects on the probed states.
It does not prove long-term success improvement or Critic input-gradient
accuracy. Reports are bound to model weights and the supplied task contract;
a changed model must be probed again, or BPTT must use the validated snapshot.
"""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch


REPORT_SCHEMA = "dq-action-effect-probe-v1"
CONTINUOUS_INDICES = (0, 1, 2, 3, 4, 5, 7, 8)


@dataclass
class ActionEffectProbeConfig:
    min_pairs: int = 64
    min_informative_pairs: int = 32
    min_effect: float = 1e-5
    min_model_effect: float = 1e-6
    min_sign_agreement: float = 0.70
    min_correlation: float = 0.25
    min_selected_gain_lcb95: float = 0.0
    max_start_error: float = 1e-5
    max_context_error: float = 1e-6
    max_dt_error: float = 1e-7
    max_repeat_reward_error: float = 1e-4
    noise_multiplier: float = 5.0


@dataclass
class ProbeStep:
    state: torch.Tensor
    context: torch.Tensor
    reward: torch.Tensor
    dt: torch.Tensor
    terminated: torch.Tensor
    executed_action: torch.Tensor = None


def balanced_family_config_ids(family_ids, count, seed):
    """Uniform-within-family reset IDs for a dedicated probe world only.

    No sampler history or training probability is changed. Every nonempty
    bank family gets a slot when count permits; its deterministic generator
    does not consume the simulator's global RNG stream.
    """
    if family_ids.ndim != 1 or count < 0:
        raise ValueError("Expected one-dimensional family IDs and nonnegative count")
    if count == 0:
        return torch.empty(0, device=family_ids.device, dtype=torch.long)
    families = family_ids.unique(sorted=True)
    if not len(families):
        raise ValueError("Cannot probe an empty reset bank")
    generator = torch.Generator(device=family_ids.device).manual_seed(int(seed))
    allocation = torch.arange(count, device=family_ids.device) % len(families)
    result = torch.empty(count, device=family_ids.device, dtype=torch.long)
    for index, family in enumerate(families):
        pool = (family_ids == family).nonzero(as_tuple=False).flatten()
        destinations = (allocation == index).nonzero(as_tuple=False).flatten()
        choices = torch.randint(len(pool), (len(destinations),), device=family_ids.device, generator=generator)
        result[destinations] = pool[choices]
    return result[torch.randperm(count, device=family_ids.device, generator=generator)]


def configuration_fingerprint(configuration):
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_fingerprint(model):
    """Hash weights once per validation/checkpoint, outside the training hot path."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


@contextmanager
def preserve_rng():
    """Isolated sessions also leave Python, NumPy and Torch caller RNG intact."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def seed_probe(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def random_continuous_directions(base_actions, generator=None, amplitudes=None):
    """Finite perturbations in the eight continuous coordinates; grip unchanged.

    Defaults are action units: 2 mm position increments, 0.01 rad orientation,
    and 0.025 base command. The task's action clipping still applies in both
    real and model execution; saturated/no-effect samples fail the evidence
    count instead of being mistaken for correct gradients.
    """
    if base_actions.ndim != 2 or base_actions.shape[-1] != 9:
        raise ValueError("Expected teacher actions with shape [batch, 9]")
    scales = base_actions.new_tensor(amplitudes if amplitudes is not None else
                                    (.002, .002, .002, .01, .01, .01, 0., .025, .025))
    if scales.shape != (9,) or float(scales[6]) != 0 or bool((scales < 0).any()):
        raise ValueError("Probe amplitudes must be nine nonnegative values with zero gripper amplitude")
    direction = torch.randn(base_actions.shape, device=base_actions.device,
                            dtype=base_actions.dtype, generator=generator)
    direction[:, 6] = 0
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return direction * scales


class IsaacGymProbeSession:
    """Adapter for a newly constructed, dedicated raw B1Z1PickMulti instance.

    The factory must reproduce task configuration and seed, construct its own
    simulator/SGS sampler, reset it, and provide its own adapter. Never wrap
    the environment used by PPO. Running the factory in a separate command
    also isolates logger side effects from the training process.
    """

    isolated = True
    simulator_name = "IsaacGym"

    def __init__(self, env, adapter, training_env=None):
        if env is training_env or adapter.env is not env:
            raise ValueError("A probe must own a dedicated simulator and its matching adapter")
        self.env, self.adapter = env, adapter
        self.closed = False

    def capture(self):
        state, context = self.adapter.capture()
        return state.detach().clone(), context.detach().clone()

    @torch.no_grad()
    def step(self, action):
        result = self.env.step(action.detach().clone())
        state, context = self.capture()
        reward = result[1].detach().clone().reshape(-1)
        terminated = result[2].detach().clone().bool().reshape(-1)
        dt = torch.full((len(state), 1), self.env.control_freq_low * self.env.dt,
                        device=state.device, dtype=state.dtype)
        return ProbeStep(state, context, reward, dt, terminated,
                         self.env.actions.detach().clone())

    def close(self):
        if not self.closed:
            if getattr(self.env, "viewer", None) is not None:
                self.env.gym.destroy_viewer(self.env.viewer)
            self.env.gym.destroy_sim(self.env.sim)
            self.closed = True


def _max_row_error(first, second):
    if first.shape != second.shape:
        raise ValueError("Probe branches changed state/context shape")
    return (first - second).abs().flatten(1).amax(-1)


@torch.no_grad()
def run_action_effect_probe(session_factory, model_factory, prefix_actions,
                            base_actions, directions, seed, configuration, config=None):
    """Run base/+/-/base-repeat in four fresh, seed-identical simulators.

    ``session_factory(seed)`` creates an isolated, initialized session exposing
    ``adapter, capture(), step(action), close(), isolated, simulator_name``.
    ``model_factory(adapter)`` reconstructs the same model weights with *that
    session's* adapter. This is essential: the context slot indexes physical
    lookup tables owned by its adapter, not the training environment's tables.

    ``prefix_actions`` is [length, batch, 9], fixed before any branch. No reset
    is allowed inside it: a terminal prefix invalidates that row. Inference
    and action-effect comparisons stay on the base_actions device; only final
    report serialization crosses to CPU.
    """
    cfg = config or ActionEffectProbeConfig()
    if (cfg.min_pairs < 1 or cfg.min_informative_pairs < 2 or not 0 <= cfg.min_sign_agreement <= 1
            or not -1 <= cfg.min_correlation <= 1 or cfg.noise_multiplier < 0):
        raise ValueError("Invalid action-effect validation thresholds")
    if base_actions.ndim != 2 or base_actions.shape[-1] != 9 or directions.shape != base_actions.shape:
        raise ValueError("Base actions and directions must both have shape [batch, 9]")
    if prefix_actions.ndim != 3 or prefix_actions.shape[1:] != base_actions.shape:
        raise ValueError("Prefix actions must have shape [length, batch, 9]")
    if bool((directions[:, 6] != 0).any()):
        raise ValueError("Action-effect probes cannot perturb the binary gripper coordinate")
    if not bool(torch.isfinite(base_actions).all() & torch.isfinite(directions).all() & torch.isfinite(prefix_actions).all()):
        raise ValueError("Probe actions must be finite")
    base_actions, directions = base_actions.detach(), directions.detach()
    active_directions = [index for index in CONTINUOUS_INDICES if bool((directions[:, index] != 0).any())]
    batch = base_actions.shape[0]
    branches = {}
    reference_hash, simulator = None, None
    with preserve_rng():
        for name, multiplier in (("base", 0.), ("plus", 1.), ("minus", -1.), ("repeat", 0.)):
            seed_probe(seed)
            session = session_factory(int(seed))
            try:
                if not getattr(session, "isolated", False):
                    raise ValueError("Probe session must own a fresh isolated simulator")
                if simulator is None:
                    simulator = session.simulator_name
                elif simulator != session.simulator_name:
                    raise ValueError("Probe branches changed simulator type")
                prefix_alive = torch.ones(batch, device=base_actions.device, dtype=torch.bool)
                prefix_durations = []
                for action in prefix_actions:
                    transition = session.step(action)
                    prefix_alive &= ~transition.terminated
                    prefix_durations.append(transition.dt)
                state, context = session.capture()
                if state.device != base_actions.device or context.device != base_actions.device:
                    raise ValueError("Probe state, actions, context and model must use the same device")
                if state.shape[0] != batch:
                    raise ValueError("Probe factory batch size differs from the provided actions")
                action = base_actions + multiplier * directions
                normalize_context = getattr(session.adapter, "normalize_context", None)
                physical_context = normalize_context(context) if normalize_context is not None else context
                # Build/evaluate the learned model after the real transition:
                # model construction can consume RNG and must not alter dt or
                # other simulator randomness before step(action).
                transition = session.step(action)
                executed_action = transition.executed_action if transition.executed_action is not None else action
                model = model_factory(session.adapter)
                model.eval()
                fingerprint = model_fingerprint(model)
                if reference_hash is None:
                    reference_hash = fingerprint
                elif reference_hash != fingerprint:
                    raise ValueError("Model weights changed between probe branches")
                predicted = model.step(state, executed_action, context, transition.dt)
                predicted_reward = session.adapter.reward(state, predicted, executed_action, context).reshape(batch)
                branches[name] = {
                    "start": state.detach().clone(), "context": context.detach().clone(),
                    "physical_context": physical_context.detach().clone(),
                    "reward": transition.reward.detach().clone(), "dt": transition.dt.detach().clone(),
                    "model_reward": predicted_reward.detach().clone(), "alive": prefix_alive.clone(),
                    "prefix_dt": (torch.stack(prefix_durations) if prefix_durations else
                                  torch.empty(0, batch, 1, device=base_actions.device)),
                }
                del model
            finally:
                session.close()
    return summarize_replayed_branches(branches, reference_hash, configuration, cfg,
        metadata={"simulator": simulator, "device": str(base_actions.device), "seed": int(seed),
                  "prefix_length": len(prefix_actions), "protocol": "independent_simulator_prefix_replay",
                  "directions_cover": active_directions, "gripper_perturbed": False})


@torch.no_grad()
def summarize_replayed_branches(branches, model_sha256, configuration, config=None, metadata=None):
    """Aggregate either in-process isolated sessions or separately launched workers."""
    cfg = config or ActionEffectProbeConfig()
    baseline = branches["base"]
    batch = len(baseline["reward"])
    start_error = torch.zeros(batch, device=baseline["reward"].device)
    context_error, dt_error = start_error.clone(), start_error.clone()
    alive = baseline["alive"].clone()
    for name in ("plus", "minus", "repeat"):
        branch = branches[name]
        start_error = torch.maximum(start_error, _max_row_error(baseline["start"], branch["start"]))
        context_error = torch.maximum(context_error, _max_row_error(baseline["context"], branch["context"]))
        context_error = torch.maximum(context_error, _max_row_error(baseline["physical_context"], branch["physical_context"]))
        dt_error = torch.maximum(dt_error, _max_row_error(baseline["dt"], branch["dt"]))
        if baseline["prefix_dt"].shape[0]:
            dt_error = torch.maximum(dt_error, (baseline["prefix_dt"] - branch["prefix_dt"]).abs().amax((0, 2)))
        alive &= branch["alive"]
        if "families" in baseline:
            if "families" not in branch or not torch.equal(baseline["families"], branch["families"]):
                raise ValueError("Reset-family assignments changed between probe branches")
    return analyze_action_effects(
        branches["plus"]["reward"], branches["minus"]["reward"], baseline["reward"], branches["repeat"]["reward"],
        branches["plus"]["model_reward"], branches["minus"]["model_reward"], start_error, context_error, dt_error, alive,
        model_sha256, configuration, cfg, metadata=metadata, families=baseline.get("families"))


@torch.no_grad()
def analyze_action_effects(real_plus, real_minus, real_base, real_repeat, predicted_plus, predicted_minus,
                           start_error, context_error, dt_error, prefix_alive, model_sha256,
                           configuration, config=None, metadata=None, families=None):
    """Evaluate finite perturbations, including real benefit over the baseline action."""
    cfg = config or ActionEffectProbeConfig()
    values = [value.reshape(-1) for value in (real_plus, real_minus, real_base, real_repeat,
                                             predicted_plus, predicted_minus, start_error, context_error, dt_error)]
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("All action-effect evidence must have the same batch shape")
    real_plus, real_minus, real_base, real_repeat, predicted_plus, predicted_minus, start_error, context_error, dt_error = values
    real_delta, model_delta = real_plus-real_minus, predicted_plus-predicted_minus
    repeat_error = (real_base-real_repeat).abs()
    finite = torch.stack(values).isfinite().all(0)
    valid = (finite & prefix_alive.bool().reshape(-1) & (start_error <= cfg.max_start_error)
             & (context_error <= cfg.max_context_error) & (dt_error <= cfg.max_dt_error)
             & (repeat_error <= cfg.max_repeat_reward_error))
    noise_floor = torch.maximum(torch.full_like(repeat_error, cfg.min_effect), repeat_error * cfg.noise_multiplier)
    informative = valid & (real_delta.abs() > noise_floor) & (model_delta.abs() > cfg.min_model_effect)
    selected_reward = torch.where(model_delta > 0, real_plus, real_minus)
    selected_gain = selected_reward-real_base
    count = int(informative.sum())
    valid_count = int(valid.sum())
    if count:
        real, prediction, gains = real_delta[informative], model_delta[informative], selected_gain[informative]
        agreement = float((real.sign() == prediction.sign()).float().mean())
        real_centered, pred_centered = real-real.mean(), prediction-prediction.mean()
        denominator = real_centered.norm()*pred_centered.norm()
        correlation = float((real_centered*pred_centered).sum()/denominator.clamp_min(1e-12))
        mean_gain = float(gains.mean())
        stderr = float(gains.std(unbiased=True)/math.sqrt(count)) if count > 1 else float("inf")
        lcb = mean_gain-1.96*stderr
    else:
        agreement = correlation = mean_gain = 0.
        lcb = float("-inf")
    reasons = []
    if valid_count < cfg.min_pairs:
        reasons.append("insufficient_reproducible_pairs")
    if count < cfg.min_informative_pairs:
        reasons.append("insufficient_nonzero_action_effects")
    if agreement < cfg.min_sign_agreement:
        reasons.append("action_effect_sign_agreement")
    if correlation < cfg.min_correlation:
        reasons.append("action_effect_correlation")
    if lcb <= cfg.min_selected_gain_lcb95:
        reasons.append("no_positive_real_gain_lower_bound")
    family_metrics = {}
    if families is not None:
        family_names = (metadata or {}).get("family_names", [])
        if not family_names:
            raise ValueError("Family-labelled probes require family_names metadata")
        if families.shape != valid.shape:
            raise ValueError("Probe family labels do not match the batch")
        for family_id, name in enumerate(family_names):
            members = families == family_id
            family_metrics[name] = {"assigned_pairs": int(members.sum()),
                                    "valid_pairs": int((members & valid).sum()),
                                    "informative_pairs": int((members & informative).sum())}
            # Coverage only, not a new family-specific statistical threshold:
            # task reward can legitimately be locally flat in a reset family.
            if len(valid) >= len(family_names) and not bool((members & valid).any()):
                reasons.append("missing_reproducible_reset_family:"+name)
    # Never encode nonstandard JSON Infinity/NaN in a failed report.
    safe_lcb = lcb if math.isfinite(lcb) else None
    def finite_summary(value, reduction="mean"):
        finite_values = value[torch.isfinite(value)]
        if not finite_values.numel():
            return None
        return float(finite_values.max() if reduction == "max" else finite_values.mean())

    report = {
        "schema": REPORT_SCHEMA, "created_unix": time.time(), "passed": not reasons, "reasons": reasons,
        "model_sha256": str(model_sha256), "configuration_sha256": configuration_fingerprint(configuration),
        "configuration": configuration, "thresholds": asdict(cfg), "metadata": metadata or {},
        "metrics": {"pairs": len(real_delta), "valid_pairs": valid_count, "informative_pairs": count,
                    "sign_agreement": agreement, "correlation": correlation, "selected_gain_mean": mean_gain,
                    "selected_gain_lcb95": safe_lcb,
                    "real_base_reward_mean": finite_summary(real_base),
                    "real_plus_reward_mean": finite_summary(real_plus),
                    "real_minus_reward_mean": finite_summary(real_minus),
                    "model_plus_reward_mean": finite_summary(predicted_plus),
                    "model_minus_reward_mean": finite_summary(predicted_minus),
                    "real_action_effect_abs_mean": finite_summary(real_delta.abs()),
                    "model_action_effect_abs_mean": finite_summary(model_delta.abs()),
                    "start_error_max": finite_summary(start_error, "max"),
                    "context_error_max": finite_summary(context_error, "max"),
                    "duration_error_max": finite_summary(dt_error, "max"),
                    "repeat_reward_error_max": finite_summary(repeat_error, "max"),
                    "reset_families": family_metrics,
                    "rejected_start_or_context": int(((start_error > cfg.max_start_error) | (context_error > cfg.max_context_error)).sum()),
                    "rejected_duration": int((dt_error > cfg.max_dt_error).sum()),
                    "rejected_repeatability": int((repeat_error > cfg.max_repeat_reward_error).sum()),
                    "rejected_terminal_prefix": int((~prefix_alive.bool()).sum())},
        "scope": "finite_one_step_reward_effects_on_replayed_real_states",
        "limitations": ["Does not validate long-term success gains", "Does not validate Critic input gradients",
                        "Does not establish a classical derivative at contact transitions",
                        "Reset families label initial-state categories, not measured contact modes",
                        "Valid only for the exact model snapshot and recorded task contract"],
    }
    report["report_sha256"] = configuration_fingerprint(report)
    return report


def write_report(path, report):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(destination.parent),
                                         prefix=destination.name+".", suffix=".tmp", delete=False) as handle:
            temporary = handle.name
            handle.write(payload)
        os.replace(temporary, str(destination))
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_verified_report(path, model_sha256, configuration, max_age_seconds=None,
                         required_simulator="IsaacGym", now=None):
    """Return (accepted, reasons, report); stale/different weights cannot enable BPTT."""
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)
    reasons = []
    copied = dict(report)
    checksum = copied.pop("report_sha256", None)
    if checksum != configuration_fingerprint(copied):
        reasons.append("report_checksum_mismatch")
    if report.get("schema") != REPORT_SCHEMA:
        reasons.append("report_schema_mismatch")
    if report.get("passed") is not True:
        reasons.append("real_action_effect_validation_failed")
        # Keep the numerical gate that failed visible in the trainer's status
        # file; the generic rejection alone hides actionable probe evidence.
        reasons.extend("probe:" + reason for reason in report.get("reasons", [])
                       if isinstance(reason, str))
    if report.get("model_sha256") != model_sha256:
        reasons.append("model_snapshot_changed")
    if report.get("configuration_sha256") != configuration_fingerprint(configuration):
        reasons.append("task_configuration_changed")
    if required_simulator is not None and report.get("metadata", {}).get("simulator") != required_simulator:
        reasons.append("wrong_validation_simulator")
    if report.get("metadata", {}).get("protocol") != "independent_simulator_prefix_replay":
        reasons.append("unsupported_validation_protocol")
    current_time = time.time() if now is None else float(now)
    age = current_time-float(report.get("created_unix", 0))
    if age < -60 or (max_age_seconds is not None and age > max_age_seconds):
        reasons.append("report_expired_or_future_dated")
    return not reasons, reasons, report
