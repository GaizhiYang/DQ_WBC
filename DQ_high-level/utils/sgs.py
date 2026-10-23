"""Success-guided reset sampling, independent of Isaac Gym.

The kernel follows SGS equations (1)--(2), including Appendix B.2's weight
floor and minimum temperature. Empty histories use the project's explicit
``p_init=0`` convention; they are not observations of failure.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch


def _mode(value):
    if value == "uniform_bank":
        value = "uniform"
    if value not in ("uniform", "sgs"):
        raise ValueError("sampler mode must be 'uniform' or 'sgs'")
    return value


def _integer_vector(value, device, name):
    result = torch.as_tensor(value, device=device)
    if result.ndim != 1:
        raise ValueError("{} must be a one-dimensional tensor".format(name))
    if result.numel() and result.dtype not in (
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError("{} must contain integer IDs".format(name))
    return result.to(dtype=torch.long)


def _assert_device(condition, message):
    # Unlike bool(cuda_tensor), this does not copy a scalar to the host on every
    # reset. An invalid CUDA input is a programming error and may poison its
    # CUDA context, just like an invalid gather index.
    torch._assert_async(condition, message)


def _validate_metadata_subset(actual, expected, prefix=""):
    for name, value in expected.items():
        key = "{}.{}".format(prefix, name) if prefix else name
        if name not in actual:
            raise ValueError("bank metadata mismatch: " + key)
        if isinstance(value, dict):
            if not isinstance(actual[name], dict):
                raise ValueError("bank metadata mismatch: " + key)
            _validate_metadata_subset(actual[name], value, key)
        elif actual[name] != value:
            raise ValueError("bank metadata mismatch: " + key)


class SGSSampler:
    """Maintain per-configuration rolling outcomes and sample reset IDs.

    ``observe`` preserves the input event order within each configuration.
    Callers must submit each completed episode exactly once, before replacing
    its active configuration ID. All regular update/sampling work stays on the
    chosen device, including batches with repeated IDs or more than H outcomes
    for one configuration.
    """

    STATE_VERSION = 1
    _TENSOR_STATE = (
        "history", "count", "head", "success_sum", "visits_total", "last_seen",
    )

    def __init__(self, num_configs, history_size=100, target_success_rate=0.5,
                 concentration=1.0, temperature=2.0, epsilon=1e-4, p_init=0.0,
                 device="cpu", seed=43, mode="sgs"):
        if isinstance(num_configs, bool) or int(num_configs) != num_configs or num_configs <= 0:
            raise ValueError("num_configs must be a positive integer")
        if isinstance(history_size, bool) or int(history_size) != history_size or history_size <= 0:
            raise ValueError("history_size must be a positive integer")
        values = (target_success_rate, concentration, temperature, epsilon, p_init)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("sampler parameters must be finite")
        if not 0 <= target_success_rate <= 1 or not 0 <= p_init <= 1:
            raise ValueError("success probabilities must be in [0, 1]")
        if concentration < 0 or temperature <= 0 or epsilon <= 0:
            raise ValueError("concentration must be nonnegative; temperature/epsilon positive")
        self.num_configs = int(num_configs)
        self.history_size = int(history_size)
        self.target_success_rate = float(target_success_rate)
        self.concentration = float(concentration)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.p_init = float(p_init)
        self.device = torch.device(device)
        self.mode = _mode(mode)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.history = torch.zeros(
            self.num_configs, self.history_size, dtype=torch.bool, device=self.device)
        self.count = torch.zeros(self.num_configs, dtype=torch.long, device=self.device)
        self.head = torch.zeros_like(self.count)
        self.success_sum = torch.zeros_like(self.count)
        self.visits_total = torch.zeros_like(self.count)
        self.last_seen = torch.full_like(self.count, -1)

    @property
    def valid_count(self):
        return self.count

    @property
    def write_ptr(self):
        return self.head

    def _config_ids(self, value):
        ids = _integer_vector(value, self.device, "config_ids")
        if ids.numel():
            _assert_device(((ids >= 0) & (ids < self.num_configs)).all(),
                           "configuration ID outside the fixed bank")
        return ids

    @torch.no_grad()
    def observe(self, config_ids, successes, step=0):
        """Append a batch of valid episode outcomes in their supplied order."""
        ids = self._config_ids(config_ids)
        outcomes = torch.as_tensor(successes, device=self.device)
        if outcomes.ndim != 1 or outcomes.shape != ids.shape:
            raise ValueError("successes must have the same one-dimensional shape as config_ids")
        if int(step) != step or step < 0:
            raise ValueError("step must be a nonnegative integer")
        if not ids.numel():
            return
        _assert_device(((outcomes == 0) | (outcomes == 1)).all(),
                       "episode outcomes must be Boolean or 0/1")
        outcomes = outcomes.to(torch.bool)

        # Stable sort gives each configuration contiguous events while retaining
        # their original order. No per-configuration Python loop is necessary.
        order = torch.argsort(ids, stable=True)
        sorted_ids = ids[order]
        sorted_outcomes = outcomes[order]
        positions = torch.arange(ids.numel(), device=self.device)
        starts = torch.ones_like(sorted_ids, dtype=torch.bool)
        starts[1:] = sorted_ids[1:] != sorted_ids[:-1]
        group_start = torch.cummax(
            torch.where(starts, positions, torch.zeros_like(positions)), dim=0).values
        rank = positions - group_start
        additions = torch.zeros_like(self.count)
        additions.scatter_add_(0, sorted_ids, torch.ones_like(sorted_ids))

        # Only the final H outcomes of an oversized group survive. Their slot
        # indices include the skipped prefix, matching a sequential ring write.
        keep = rank >= (additions[sorted_ids] - self.history_size)
        kept_ids = sorted_ids[keep]
        slots = (self.head[kept_ids] + rank[keep]) % self.history_size
        new_values = sorted_outcomes[keep]
        old_values = self.history[kept_ids, slots]
        delta = torch.zeros_like(self.success_sum)
        delta.scatter_add_(0, kept_ids, new_values.long() - old_values.long())
        self.history[kept_ids, slots] = new_values
        self.success_sum.add_(delta)
        self.count.copy_(torch.clamp(self.count + additions, max=self.history_size))
        self.head.copy_((self.head + additions) % self.history_size)
        self.visits_total.add_(additions)
        self.last_seen.copy_(torch.where(additions > 0, int(step), self.last_seen))

    def record(self, config_ids, outcomes, env_ids=None, global_step=0):
        """Compatibility wrapper; event order is the supplied order, not env_id."""
        self.observe(config_ids, outcomes, step=global_step)

    @torch.no_grad()
    def success_rates(self):
        rates = self.success_sum.to(torch.float32) / self.count.clamp_min(1)
        return torch.where(self.count > 0, rates, self.p_init)

    @torch.no_grad()
    def probabilities(self, mode=None):
        if _mode(self.mode if mode is None else mode) == "uniform":
            return torch.full(
                (self.num_configs,), 1.0 / self.num_configs,
                dtype=torch.float32, device=self.device)
        rates = self.success_rates()
        log_weight = (
            self.concentration * self.target_success_rate * torch.log(rates + self.epsilon)
            + self.concentration * (1.0 - self.target_success_rate)
            * torch.log(1.0 - rates + self.epsilon)
        )
        log_epsilon = torch.full_like(log_weight, math.log(self.epsilon))
        logits = torch.logaddexp(torch.maximum(log_weight, log_epsilon), log_epsilon)
        return torch.softmax(logits / max(self.temperature, 1.0), dim=0)

    @torch.no_grad()
    def sample(self, num_resets, mode=None):
        if isinstance(num_resets, bool) or int(num_resets) != num_resets or num_resets < 0:
            raise ValueError("num_resets must be a nonnegative integer")
        if num_resets == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        return torch.multinomial(self.probabilities(mode), int(num_resets),
                                 replacement=True, generator=self.generator)

    @torch.no_grad()
    def metrics(self, step=0, family_id=None):
        """Return scalar device tensors; logging may transfer them in one batch."""
        probabilities = self.probabilities()
        visited = self.count > 0
        num_visited = visited.sum().clamp_min(1)
        ages = torch.where(visited, (int(step) - self.last_seen).clamp_min(0), 0)
        rates = self.success_rates()
        result = {
            "visited_fraction": visited.float().mean(),
            "unknown_count": (~visited).sum(),
            "history_count_mean": self.count.float().mean(),
            "history_count_min": self.count.min(),
            "history_count_max": self.count.max(),
            "visits_total": self.visits_total.sum(),
            "success_rate_visited_mean": (rates * visited).sum() / num_visited,
            "history_success_rate": (
                self.success_sum.sum().float() / self.count.sum().clamp_min(1)),
            "last_visit_age_mean": ages.sum().float() / num_visited,
            "last_visit_age_max": ages.max(),
            "sampling_entropy": -(probabilities * probabilities.clamp_min(
                torch.finfo(probabilities.dtype).tiny).log()).sum(),
            "effective_configs": 1.0 / probabilities.square().sum(),
            "probability_min": probabilities.min(),
            "probability_max": probabilities.max(),
        }
        if family_id is not None:
            families = _integer_vector(family_id, self.device, "family_id")
            if families.shape != self.count.shape:
                raise ValueError("family_id must have one entry per configuration")
            # Bank schema defines three families. This fixed-size loop is for
            # logging only; it never iterates over environments or configurations.
            for family in range(3):
                mask = families == family
                result["family_{}_probability".format(family)] = probabilities[mask].sum()
                result["family_{}_visits".format(family)] = self.visits_total[mask].sum()
        return result

    def _parameters(self):
        return {
            "num_configs": self.num_configs, "history_size": self.history_size,
            "target_success_rate": self.target_success_rate,
            "concentration": self.concentration, "temperature": self.temperature,
            "epsilon": self.epsilon, "p_init": self.p_init, "mode": self.mode,
        }

    def state_dict(self):
        """A detached CPU snapshot, including the sampler's own RNG."""
        return {
            "state_version": self.STATE_VERSION,
            "parameters": self._parameters(),
            "tensors": {name: getattr(self, name).detach().cpu().clone()
                        for name in self._TENSOR_STATE},
            "rng_device_type": self.device.type,
            "rng_state": self.generator.get_state().cpu().clone(),
        }

    def load_state_dict(self, state):
        """Validate a matching snapshot before replacing any sampler state."""
        if state.get("state_version") != self.STATE_VERSION:
            raise ValueError("unsupported sampler state version")
        if state.get("parameters") != self._parameters():
            raise ValueError("sampler checkpoint parameters do not match")
        if state.get("rng_device_type") != self.device.type:
            raise ValueError("exact RNG restore requires the same device type (CPU/CUDA)")
        loaded = {}
        for name in self._TENSOR_STATE:
            value = state.get("tensors", {}).get(name)
            reference = getattr(self, name)
            if not isinstance(value, torch.Tensor) or value.shape != reference.shape:
                raise ValueError("invalid sampler checkpoint tensor: " + name)
            if value.dtype != reference.dtype:
                raise ValueError("invalid sampler checkpoint dtype: " + name)
            loaded[name] = value.detach().cpu().clone()
        count, head = loaded["count"], loaded["head"]
        if not bool(((count >= 0) & (count <= self.history_size)).all()):
            raise ValueError("invalid history counts")
        if not bool(((head >= 0) & (head < self.history_size)).all()):
            raise ValueError("invalid ring pointers")
        if not torch.equal(loaded["history"].sum(dim=1), loaded["success_sum"]):
            raise ValueError("history success sums do not match")
        if not bool((loaded["success_sum"] <= count).all()):
            raise ValueError("more successes than valid history outcomes")
        if not bool((loaded["visits_total"] >= count).all()):
            raise ValueError("invalid cumulative visit counts")
        if not torch.equal(loaded["visits_total"].remainder(self.history_size), head):
            raise ValueError("ring pointers do not match cumulative visit counts")
        if not torch.equal(loaded["visits_total"].clamp(max=self.history_size), count):
            raise ValueError("history counts do not match cumulative visit counts")
        if not bool(((count == 0) == (loaded["last_seen"] == -1)).all()):
            raise ValueError("last_seen does not match visited configurations")
        rng = torch.Generator(device=self.device)
        rng.set_state(state["rng_state"].cpu())
        for name, value in loaded.items():
            getattr(self, name).copy_(value.to(self.device))
        self.generator.set_state(rng.get_state())


class ConfigurationBank:
    """Versioned, tensor-only physical reset bank.

    Validation checks storage integrity and simple state invariants; it does not
    replace simulator collision/contact validation. Treat tensors/metadata as
    immutable after construction so a configuration ID retains its meaning.
    """

    SCHEMA_VERSION = 1
    REQUIRED_METADATA = (
        "object_name", "robot_asset", "num_dofs", "num_bodies", "hold_steps",
        "lifted_threshold", "version",
    )
    REQUIRED_TENSORS = (
        "family_id", "robot_root", "object_root", "table_root", "rigid_body_state",
        "ee_jacobian",
        "dof_pos", "dof_vel", "ee_goal_cart", "ee_goal_orn_rpy",
        "gripper_target", "table_vy", "rotation_enabled",
        "table_yaw_rate", "table_height",
    )

    def __init__(self, tensors, metadata, device=None):
        if not isinstance(tensors, dict) or not isinstance(metadata, dict):
            raise ValueError("bank tensors and metadata must be dictionaries")
        self.metadata = json.loads(json.dumps(metadata, sort_keys=True, allow_nan=False))
        self.tensors = {}
        for name, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError("bank field must be a tensor: " + name)
            self.tensors[name] = value.detach().clone().to(
                device=device if device is not None else value.device)
        if "family_id" not in self.tensors or self.tensors["family_id"].ndim != 1:
            raise ValueError("family_id must be a one-dimensional tensor")
        self.num_configs = self.tensors["family_id"].numel()
        self.device = self.tensors["family_id"].device
        self.validate()
        self.fingerprint = self._fingerprint()

    def validate(self, expected_metadata=None):
        missing = set(self.REQUIRED_METADATA) - self.metadata.keys()
        if missing:
            raise ValueError("missing bank metadata: " + ", ".join(sorted(missing)))
        missing = set(self.REQUIRED_TENSORS) - self.tensors.keys()
        if missing:
            raise ValueError("missing bank tensors: " + ", ".join(sorted(missing)))
        if self.num_configs <= 0:
            raise ValueError("configuration bank must not be empty")
        for name in ("object_name", "robot_asset"):
            if not isinstance(self.metadata[name], str) or not self.metadata[name]:
                raise ValueError("invalid bank metadata: " + name)
        for name in ("num_dofs", "num_bodies", "hold_steps"):
            if type(self.metadata[name]) is not int or self.metadata[name] <= 0:
                raise ValueError("bank metadata must be a positive integer: " + name)
        if not self.metadata["version"]:
            raise ValueError("bank version must be nonempty")
        threshold = self.metadata["lifted_threshold"]
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("lifted_threshold must be finite and positive")
        if expected_metadata:
            _validate_metadata_subset(self.metadata, expected_metadata)

        shapes = {
            "family_id": (self.num_configs,),
            "robot_root": (self.num_configs, 13),
            "object_root": (self.num_configs, 13),
            "table_root": (self.num_configs, 13),
            "rigid_body_state": (self.num_configs, self.metadata["num_bodies"], 13),
            "ee_jacobian": (self.num_configs, 6, 6),
            "dof_pos": (self.num_configs, self.metadata["num_dofs"]),
            "dof_vel": (self.num_configs, self.metadata["num_dofs"]),
            "ee_goal_cart": (self.num_configs, 3),
            "ee_goal_orn_rpy": (self.num_configs, 3),
            "table_vy": (self.num_configs,),
            "rotation_enabled": (self.num_configs,),
            "table_yaw_rate": (self.num_configs,),
            "table_height": (self.num_configs,),
            "table_change_interval": (self.num_configs,),
        }
        for name, value in self.tensors.items():
            if value.device != self.device:
                raise ValueError("all bank tensors must be on the same device")
            if value.ndim == 0 or value.shape[0] != self.num_configs:
                raise ValueError("bank field has inconsistent configuration count: " + name)
            if name in shapes and tuple(value.shape) != shapes[name]:
                raise ValueError("invalid bank tensor shape: " + name)
            if name in ("family_id", "rotation_enabled"):
                expected_dtype = torch.long if name == "family_id" else torch.bool
                if value.dtype != expected_dtype:
                    raise ValueError("invalid bank tensor dtype: " + name)
            elif name == "table_change_interval":
                if value.dtype == torch.bool or value.is_complex():
                    raise ValueError("table_change_interval must contain integers in [40, 60)")
                valid = (torch.isfinite(value) & (value >= 40) & (value < 60)
                         & (value == value.long()))
                if not bool(valid.all()):
                    raise ValueError("table_change_interval must contain integers in [40, 60)")
            elif not value.is_floating_point() or not bool(torch.isfinite(value).all()):
                raise ValueError("bank state must contain finite floating values: " + name)
        gripper = self.tensors["gripper_target"]
        if gripper.ndim != 2 or not 0 < gripper.shape[1] <= self.metadata["num_dofs"]:
            raise ValueError("gripper_target must have shape [N, G], with 0 < G <= D")
        families = self.tensors["family_id"]
        if not bool(((families >= 0) & (families <= 2)).all()):
            raise ValueError("family_id values must be 0, 1 or 2")
        for name in ("robot_root", "object_root", "table_root", "rigid_body_state"):
            norms = self.tensors[name][..., 3:7].norm(dim=-1)
            if not bool(torch.isclose(norms, torch.ones_like(norms), atol=1e-3, rtol=1e-3).all()):
                raise ValueError("root quaternion must be normalized: " + name)
        physical = self.metadata.get("physical_validation", {})
        if not isinstance(physical, dict):
            raise ValueError("physical_validation must be a metadata dictionary")
        geometry_profile = physical.get("validation_profile") == "geometry_v1"
        if geometry_profile:
            for name, expected in (("passed", True), ("feet_near_ground", True),
                                   ("collision_checked", True), ("object_attachment", False)):
                if physical.get(name) is not expected:
                    raise ValueError("invalid geometry_v1 validation flag: " + name)
            if physical.get("grounded_initial_states", False):
                raise ValueError("geometry_v1 does not certify stable grounded_initial_states")
            if self.metadata.get("family_names") != ["full_task", "pregrasp", "at_grasp"]:
                raise ValueError("geometry_v1 requires full_task/pregrasp/at_grasp family_names")
        if geometry_profile or physical.get("grounded_initial_states", False):
            from utils.sgs_grounding import (
                APPROX_GROUND_LIMITS, FOOT_RADIUS, approx_grounded_mask, grounded_mask)
            grounding = physical.get("grounding", {})
            if not isinstance(grounding, dict):
                raise ValueError("invalid grounding geometry metadata")
            indices = grounding.get("foot_body_indices", [])
            if (not isinstance(indices, list) or len(indices) != 4
                    or any(type(i) is not int or not 0 <= i < self.metadata["num_bodies"]
                           for i in indices) or len(set(indices)) != 4
                    or grounding.get("foot_radius_m") != FOOT_RADIUS):
                raise ValueError("invalid grounding geometry metadata")
            bottom = self.tensors["rigid_body_state"][:, indices, 2] - FOOT_RADIUS
        if geometry_profile:
            if grounding.get("limits") != APPROX_GROUND_LIMITS:
                raise ValueError("invalid geometry_v1 grounding limits")
            if not bool(approx_grounded_mask(bottom).all()):
                raise ValueError("bank contains initial feet outside approximate ground limits")
        elif physical.get("grounded_initial_states", False):
            forces = self.tensors.get("initial_foot_contact_force")
            if forces is None or tuple(forces.shape) != (self.num_configs, 4, 3):
                raise ValueError("grounded bank requires initial foot contact forces")
            if not bool(grounded_mask(self.tensors["robot_root"], bottom, forces).all()):
                raise ValueError("bank contains ungrounded or unsettled initial states")
        return self

    def _fingerprint(self):
        digest = hashlib.sha256()
        digest.update(json.dumps(
            {"schema_version": self.SCHEMA_VERSION, "metadata": self.metadata},
            sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
        for name in sorted(self.tensors):
            tensor = self.tensors[name].detach().cpu().contiguous()
            digest.update(json.dumps(
                [name, str(tensor.dtype), list(tensor.shape)], separators=(",", ":")).encode("utf-8"))
            digest.update(tensor.numpy().tobytes())
        return digest.hexdigest()

    def batch(self, config_ids):
        ids = _integer_vector(config_ids, self.device, "config_ids")
        if ids.numel():
            _assert_device(((ids >= 0) & (ids < self.num_configs)).all(),
                           "configuration ID outside the fixed bank")
        return {name: values[ids] for name, values in self.tensors.items()}

    def state_dict(self):
        self.validate()
        fingerprint = self._fingerprint()
        if fingerprint != self.fingerprint:
            raise ValueError("configuration bank was mutated after construction")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "metadata": json.loads(json.dumps(self.metadata)),
            "tensors": {name: values.detach().cpu().clone()
                        for name, values in self.tensors.items()},
            "fingerprint": fingerprint,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.state_dict()
        descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path, device="cpu", expected_metadata=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("unsupported configuration bank schema")
        bank = cls(payload["tensors"], payload["metadata"], device=device)
        if bank.fingerprint != payload.get("fingerprint"):
            raise ValueError("configuration bank fingerprint mismatch")
        bank.validate(expected_metadata)
        return bank
