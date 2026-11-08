"""Autoregressive fitting on true actions, with episode-held-out validation.

Adapted from SuperMimic-Reproduced ``world_model/wm_trainer.py`` and
``world_model/world_model.py`` (MIT); see THIRD_PARTY_LICENSES.md. No Actor or
Critic is part of this optimizer. Predicted states are never teacher-forced
between steps, while true future states are used only as supervised targets.
"""

from dataclasses import asdict, dataclass, fields
import math

import torch
from torch.nn import functional as F


@dataclass
class WMTrainerConfig:
    window: int = 3
    batches_per_update: int = 4
    batch_size: int = 256
    learning_rate: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    termination_weight: float = 0.1
    contact_weight: float = 0.1
    auxiliary_weight: float = 1.0
    state_loss_reduction: str = "mean"
    recent_fraction: float = 0.5
    recent_steps: int = 32
    validation_batch_size: int = 256
    balance_families: bool = True
    microbatch_size: int = 0


class WorldModelTrainer:
    def __init__(self, model, config=None):
        self.model = model
        self.cfg = config or WMTrainerConfig()
        if min(self.cfg.window, self.cfg.batches_per_update, self.cfg.batch_size,
               self.cfg.validation_batch_size) < 1:
            raise ValueError("World-model training dimensions must be positive")
        if self.cfg.learning_rate <= 0 or self.cfg.grad_clip <= 0:
            raise ValueError("World-model learning rate and grad clip must be positive")
        if self.cfg.microbatch_size < 0:
            raise ValueError("World-model microbatch size must be nonnegative")
        if self.cfg.state_loss_reduction not in ("mean", "sum", "reference"):
            raise ValueError("World-model state_loss_reduction must be mean, sum or reference")
        if self.cfg.state_loss_reduction == "reference" and (
                getattr(model, "dynamics_target", None) != "physical_increment"):
            raise ValueError("Reference WM loss requires physical_increment coordinates")
        if (isinstance(self.cfg.auxiliary_weight, bool)
                or not math.isfinite(float(self.cfg.auxiliary_weight))
                or self.cfg.auxiliary_weight < 0):
            raise ValueError("World-model auxiliary_weight must be finite and nonnegative")
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=self.cfg.learning_rate,
                                          weight_decay=self.cfg.weight_decay)
        self.updates = 0

    def loss_on(self, batch):
        if self.cfg.state_loss_reduction == "reference":
            return self._reference_loss_on(batch)
        if getattr(batch, "valid_mask", None) is not None:
            return self._masked_loss_on(batch)
        steps = batch.actions.shape[0]
        predicted = batch.states[0]
        errors, state_terms, term_losses, auxiliary, term_accuracy = [], [], [], [], []
        contact_losses, contact_accuracy = [], []
        extra_loss = (getattr(self.model.spec, "auxiliary_loss", None)
                      if self.cfg.auxiliary_weight else None)
        physical_error = None
        if self.cfg.state_loss_reduction == "sum":
            physical_error = getattr(self.model.spec, "physical_error_vector", None)
            if physical_error is None:
                physical_error = self.model.spec.error_vector
        low_step = getattr(self.model, "low_step", None)
        for index in range(steps):
            if low_step is not None:
                # Only initial context is available in an imagined trajectory.
                # Future measured table poses must not leak into this rollout.
                predicted, contact_logits = low_step(
                    predicted, batch.actions[index], batch.context[0],
                    batch.dt[index], return_logits=True)
                labels = self.model.spec.get(batch.states[index + 1], "foot_contacts")
                contact_losses.append(F.binary_cross_entropy_with_logits(
                    contact_logits, labels.to(contact_logits.dtype), reduction="none").mean(-1))
                contact_accuracy.append(((contact_logits > 0) == labels.bool()).float().detach())
            else:
                predicted = self.model(predicted, batch.actions[index],
                                       batch.context[index], batch.dt[index])
            # No detach and no ground-truth replacement here: model fitting
            # differentiates the whole open-loop multi-step prediction error.
            if physical_error is not None:
                # SuperMimic world_model.py:95/103 sums the unweighted,
                # unit-normalized physical residuals before batch/time means.
                # NSE remains a mean over dimensions, for diagnostic reporting.
                squared = physical_error(predicted, batch.states[index + 1]).square()
                state_terms.append(squared.sum(-1))
                errors.append(squared.mean(-1))
            else:
                # Preserve the legacy model-defined, possibly group-weighted
                # mean error used by the gated recipe and existing clients.
                error = self.model.normalized_sq_error(predicted, batch.states[index + 1])
                state_terms.append(error)
                errors.append(error)
            if extra_loss is not None:
                auxiliary.append(extra_loss(predicted, batch.states[index + 1]))
            if self.model.term_head is not None:
                logits = self.model.termination_logits(predicted, batch.context[index])
                labels = batch.terminated[index].float()
                term_losses.append(F.binary_cross_entropy_with_logits(logits, labels,
                                                                      reduction="none"))
                term_accuracy.append(((logits > 0) == labels.bool()).float().detach())
        error = torch.stack(errors)
        state_loss = torch.stack(state_terms).mean()
        term_loss = (torch.stack(term_losses).mean() if term_losses else
                     state_loss.new_zeros(()))
        extra = (torch.stack(auxiliary).mean() if auxiliary else state_loss.new_zeros(()))
        contact_loss = (torch.stack(contact_losses).mean() if contact_losses else
                        state_loss.new_zeros(()))
        loss = (state_loss + self.cfg.termination_weight * term_loss
                + self.cfg.auxiliary_weight * extra + self.cfg.contact_weight * contact_loss)
        metrics = {"state_loss": state_loss.detach(), "termination_bce": term_loss.detach(),
                   "auxiliary_loss": extra.detach(), "nse_1step": error[0].mean().detach(),
                   "nse_kstep": error[-1].mean().detach(), "loss": loss.detach()}
        if term_accuracy:
            metrics["termination_accuracy"] = torch.stack(term_accuracy).mean()
        if contact_accuracy:
            metrics["contact_bce"] = contact_loss.detach()
            metrics["contact_accuracy"] = torch.stack(contact_accuracy).mean()
        return loss, metrics

    def _reference_loss_on(self, batch):
        """Source objective: mean physical axes, sum valid times / window batch.

        Increments and endpoint errors share physical units and PRE-heading
        axes. The dimensional error floor limits numerical amplification; it
        is not an acceptance threshold or a claim of real task accuracy.
        """
        steps, width = batch.actions.shape[:2]
        mask = getattr(batch, "valid_mask", None)
        mask = (torch.ones((steps, width), dtype=torch.bool, device=batch.states.device)
                if mask is None else mask.bool())
        if mask.shape != (steps, width) or bool((mask[1:] & ~mask[:-1]).any()):
            raise ValueError("Window validity must be a [steps,batch] contiguous prefix")
        count = mask.sum()
        if not bool(count):
            zero = batch.states.new_zeros(())
            return zero, {key: zero for key in ("loss", "state_loss", "state_step_loss",
                         "termination_bce", "auxiliary_loss", "contact_bce", "contact_accuracy",
                         "nse_1step", "nse_kstep", "valid_count", "valid_1step_count", "valid_kstep_count")}
        row = mask[0].long().argmax().reshape(1)
        pad_state = batch.states[0].index_select(0, row)
        pad_target = batch.states[1].index_select(0, row)
        pad_action = batch.actions[0].index_select(0, row)
        pad_context = batch.context[0].index_select(0, row)
        pad_dt = batch.dt[0].index_select(0, row)
        predicted = batch.states[0]
        scale = self.model.reference_error_scale()
        errors, contacts, contact_correct, auxiliary = [], [], [], []
        extra_loss = (getattr(self.model.spec, "auxiliary_loss", None)
                      if self.cfg.auxiliary_weight else None)
        for step in range(steps):
            active = mask[step, :, None]
            pre = torch.where(active, predicted, pad_state)
            action = torch.where(active, batch.actions[step], pad_action)
            context = torch.where(active, batch.context[0], pad_context)
            duration = torch.where(active, batch.dt[step], pad_dt)
            target = torch.where(active, batch.states[step + 1], pad_target)
            predicted, logits = self.model.low_step(pre, action, context, duration, return_logits=True)
            error = self.model.reference_error_vector(predicted, target, pre)
            error = torch.where(active, error, torch.zeros_like(error))
            errors.append((error / scale).square().mean(-1))
            labels = self.model.spec.get(target, "foot_contacts")
            contacts.append(F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype),
                                                               reduction="none").mean(-1))
            contact_correct.append(((logits > 0) == labels.bool()).float().mean(-1))
            if extra_loss is not None:
                auxiliary.append(extra_loss(predicted, target))
            predicted = torch.where(active, predicted, pad_state)

        def total(values):
            if not values:
                return predicted.new_zeros(())
            values = torch.stack(values)
            return torch.where(mask, values, torch.zeros_like(values)).sum()

        state_total, contact_total = total(errors), total(contacts)
        state_loss, contact_loss = state_total / width, contact_total / width
        extra = total(auxiliary) / width
        loss = state_loss + self.cfg.contact_weight * contact_loss + self.cfg.auxiliary_weight * extra
        errors = torch.stack(errors)
        return loss, {"loss": loss.detach(), "state_loss": state_loss.detach(),
                      "state_step_loss": (state_total / count).detach(),
                      "contact_bce": contact_loss.detach(),
                      "contact_accuracy": (total(contact_correct) / count).detach(),
                      "termination_bce": loss.new_zeros(()), "auxiliary_loss": extra.detach(),
                      "valid_count": count.detach().float(),
                      "valid_1step_count": mask[0].sum().float(),
                      "valid_kstep_count": mask[-1].sum().float(),
                      "nse_1step": (torch.where(mask[0], errors[0], 0.).sum() / mask[0].sum().clamp_min(1)).detach(),
                      "nse_kstep": (torch.where(mask[-1], errors[-1], 0.).sum() / mask[-1].sum().clamp_min(1)).detach()}

    def _masked_loss_on(self, batch):
        """Autoregressive valid prefixes; sanitize padding before all arithmetic."""
        mask = batch.valid_mask.bool()
        if mask.shape != batch.actions.shape[:2]:
            raise ValueError("Window valid_mask must have shape [steps,batch]")
        if bool((mask[1:] & ~mask[:-1]).any()):
            raise ValueError("Window validity must be a contiguous prefix")
        count = mask.sum()
        if not bool(count):
            zero = batch.states.new_zeros(())
            return zero, {key: zero for key in ("loss", "state_loss", "termination_bce",
                         "auxiliary_loss", "nse_1step", "nse_kstep", "valid_count",
                         "valid_1step_count", "valid_kstep_count")}
        # A real first transition is a safe, finite padding source. Never pass
        # reset/invalid tail quaternions, context IDs, actions or dt to a model.
        row = mask[0].long().argmax().reshape(1)
        pad_state = batch.states[0].index_select(0, row)
        pad_target = batch.states[1].index_select(0, row)
        pad_action = batch.actions[0].index_select(0, row)
        pad_context = batch.context[0].index_select(0, row)
        pad_dt = batch.dt[0].index_select(0, row)
        predicted = batch.states[0]
        low_step = getattr(self.model, "low_step", None)
        physical_error = None
        if self.cfg.state_loss_reduction == "sum":
            physical_error = getattr(self.model.spec, "physical_error_vector", self.model.spec.error_vector)
        extra_loss = (getattr(self.model.spec, "auxiliary_loss", None)
                      if self.cfg.auxiliary_weight else None)
        errors, state_terms, contacts, contact_accuracy = [], [], [], []
        terms, term_accuracy, auxiliary = [], [], []
        for index in range(len(mask)):
            active = mask[index, :, None]
            predicted = torch.where(active, predicted, pad_state)
            action = torch.where(active, batch.actions[index], pad_action)
            context = torch.where(active, batch.context[0 if low_step is not None else index], pad_context)
            dt = torch.where(active, batch.dt[index], pad_dt)
            target = torch.where(active, batch.states[index + 1], pad_target)
            if low_step is not None:
                predicted, logits = low_step(predicted, action, context, dt, return_logits=True)
                labels = self.model.spec.get(target, "foot_contacts")
                contacts.append(F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype),
                                                                   reduction="none").mean(-1))
                contact_accuracy.append(((logits > 0) == labels.bool()).float().mean(-1).detach())
            else:
                predicted = self.model(predicted, action, context, dt)
            # Inactive output is not a target or a future physical state.
            predicted = torch.where(active, predicted, pad_target)
            if physical_error is not None:
                squared = physical_error(predicted, target).square()
                state_terms.append(squared.sum(-1))
                errors.append(squared.mean(-1))
            else:
                error = self.model.normalized_sq_error(predicted, target)
                state_terms.append(error)
                errors.append(error)
            if extra_loss is not None:
                auxiliary.append(extra_loss(predicted, target))
            if self.model.term_head is not None:
                logits = self.model.termination_logits(predicted, context)
                labels = torch.where(mask[index], batch.terminated[index], False).float()
                terms.append(F.binary_cross_entropy_with_logits(logits, labels, reduction="none"))
                term_accuracy.append(((logits > 0) == labels.bool()).float().detach())

        def average(values):
            if not values:
                return predicted.new_zeros(())
            stacked = torch.stack(values)
            return torch.where(mask, stacked, torch.zeros_like(stacked)).sum() / count

        state_loss, contact_loss = average(state_terms), average(contacts)
        term_loss, extra = average(terms), average(auxiliary)
        loss = (state_loss + self.cfg.contact_weight * contact_loss
                + self.cfg.termination_weight * term_loss + self.cfg.auxiliary_weight * extra)
        error = torch.stack(errors)
        metrics = {"state_loss": state_loss.detach(), "termination_bce": term_loss.detach(),
                   "auxiliary_loss": extra.detach(), "loss": loss.detach(),
                   "valid_count": count.detach().float(),
                   "valid_1step_count": mask[0].sum().float(),
                   "valid_kstep_count": mask[-1].sum().float(),
                   "nse_1step": torch.where(mask[0], error[0], 0.).sum().detach() / mask[0].sum().clamp_min(1),
                   "nse_kstep": torch.where(mask[-1], error[-1], 0.).sum().detach() / mask[-1].sum().clamp_min(1)}
        if contacts:
            metrics.update(contact_bce=contact_loss.detach(), contact_accuracy=average(contact_accuracy))
        if terms:
            metrics["termination_accuracy"] = average(term_accuracy)
        return loss, metrics

    def update(self, replay, *, batches=None, progress_callback=None, progress_interval=32):
        batches = self.cfg.batches_per_update if batches is None else batches
        if isinstance(batches, bool) or not isinstance(batches, int) or batches < 1:
            raise ValueError("World-model update batches must be a positive integer")
        if (isinstance(progress_interval, bool) or not isinstance(progress_interval, int)
                or progress_interval < 1):
            raise ValueError("World-model progress interval must be a positive integer")
        if not replay.can_sample(self.cfg.window, "train"):
            return {"skipped_no_windows": 1.0, "updates": float(self.updates)}
        self.model.train()
        aggregate = {}
        accepted = 0
        rejected = 0
        skipped_empty = 0
        for batch_index in range(batches):
            if progress_callback is not None and batch_index and batch_index % progress_interval == 0:
                progress_callback(batch_index, batches)
            batch = replay.sample_windows(self.cfg.batch_size, self.cfg.window,
                                          split="train", balance_families=self.cfg.balance_families,
                                          recent_fraction=self.cfg.recent_fraction,
                                          recent_steps=self.cfg.recent_steps)
            self.optimizer.zero_grad(set_to_none=True)
            width = batch.states.shape[1]
            chunk = self.cfg.microbatch_size or width
            valid_mask = getattr(batch, "valid_mask", None)
            total_valid = (valid_mask.sum() if valid_mask is not None else
                           batch.states.new_tensor(batch.actions.shape[0] * width))
            if not bool(total_valid):
                skipped_empty += 1
                continue
            metrics, finite = {}, True
            for begin in range(0, width, chunk):
                end = min(width, begin + chunk)
                values = {}
                for field in fields(batch):
                    value = getattr(batch, field.name)
                    # Temporal payloads have [K,B,...]; IDs have [B].
                    values[field.name] = (None if value is None else value[:, begin:end] if field.name in (
                        "states", "actions", "context", "dt", "rewards", "terminated",
                        "truncated", "success", "valid_mask") else value[begin:end])
                chunk_mask = values.get("valid_mask")
                chunk_valid = (chunk_mask.sum() if chunk_mask is not None else
                               total_valid * ((end - begin) / width))
                if not bool(chunk_valid):
                    continue
                loss, chunk_metrics = self.loss_on(type(batch)(**values))
                if not bool(torch.isfinite(loss)):
                    finite = False
                    break
                reference_loss = self.cfg.state_loss_reduction == "reference"
                weight = ((end - begin) / width if reference_loss else chunk_valid / total_valid)
                (loss * weight).backward()
                for key, value in chunk_metrics.items():
                    metric_weight = weight
                    if key in ("valid_count", "valid_1step_count", "valid_kstep_count"):
                        metric_weight = 1.
                    elif reference_loss and key in ("state_step_loss", "contact_accuracy"):
                        metric_weight = chunk_valid / total_valid
                    elif valid_mask is not None and key in ("nse_1step", "nse_kstep"):
                        index = 0 if key == "nse_1step" else -1
                        metric_weight = chunk_mask[index].sum() / valid_mask[index].sum().clamp_min(1)
                    metrics[key] = metrics.get(key, 0.) + value * metric_weight
            if not finite:
                self.optimizer.zero_grad(set_to_none=True)
                rejected += 1
                continue
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            if not bool(torch.isfinite(norm)):
                self.optimizer.zero_grad(set_to_none=True)
                rejected += 1
                continue
            self.optimizer.step()
            accepted += 1
            self.updates += 1
            metrics["grad_norm"] = norm.detach()
            for key, value in metrics.items():
                aggregate[key] = aggregate.get(key, 0.0) + value
        # Logging only: payloads and optimization never leave their device.
        if progress_callback is not None:
            progress_callback(batches, batches)
        result = {key: float(value / max(1, accepted)) for key, value in aggregate.items()}
        result.update(updates=float(self.updates), skipped_nonfinite=float(rejected),
                      skipped_no_valid_steps=float(skipped_empty))
        result.update(self.validate(replay))
        return result

    @torch.no_grad()
    def validate(self, replay):
        if not replay.can_sample(self.cfg.window, "validation"):
            # Missing validation must not be interpreted as zero model error.
            return {"validation_available": 0.0}
        was_training = self.model.training
        self.model.eval()
        try:
            batch = replay.sample_windows(self.cfg.validation_batch_size, self.cfg.window,
                                          split="validation", balance_families=self.cfg.balance_families)
            _, metrics = self.loss_on(batch)
            result = {"validation_" + key: float(value) for key, value in metrics.items()}
            result["validation_available"] = float(result.get("validation_valid_count", 1.) > 0)
            return result
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def start_error(self, starts):
        """Real one-step prediction error for conservative imagination filtering."""
        if (hasattr(self.model, "low_step")
                and starts.action.shape[-1] == self.model.spec.low_action_dim):
            predicted = self.model.low_step(starts.state, starts.action,
                                            starts.context, starts.dt)
        else:
            predicted = self.model(starts.state, starts.action, starts.context, starts.dt)
        return self.model.normalized_sq_error(predicted, starts.next_state)

    def state_dict(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "updates": self.updates, "config": asdict(self.cfg)}

    def load_state_dict(self, state):
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.updates = int(state.get("updates", 0))
