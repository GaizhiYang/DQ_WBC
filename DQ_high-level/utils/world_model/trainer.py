"""Autoregressive fitting on true actions, with episode-held-out validation.

Adapted from SuperMimic-Reproduced ``world_model/wm_trainer.py`` and
``world_model/world_model.py`` (MIT); see THIRD_PARTY_LICENSES.md. No Actor or
Critic is part of this optimizer. Predicted states are never teacher-forced
between steps, while true future states are used only as supervised targets.
"""

from dataclasses import asdict, dataclass, fields

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
        self.optimizer = torch.optim.Adam(model.parameters(),
                                          lr=self.cfg.learning_rate,
                                          weight_decay=self.cfg.weight_decay)
        self.updates = 0

    def loss_on(self, batch):
        steps = batch.actions.shape[0]
        predicted = batch.states[0]
        errors, term_losses, auxiliary, term_accuracy = [], [], [], []
        contact_losses, contact_accuracy = [], []
        extra_loss = getattr(self.model.spec, "auxiliary_loss", None)
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
            errors.append(self.model.normalized_sq_error(predicted,
                                                          batch.states[index + 1]))
            if extra_loss is not None:
                auxiliary.append(extra_loss(predicted, batch.states[index + 1]))
            if self.model.term_head is not None:
                logits = self.model.termination_logits(predicted, batch.context[index])
                labels = batch.terminated[index].float()
                term_losses.append(F.binary_cross_entropy_with_logits(logits, labels,
                                                                      reduction="none"))
                term_accuracy.append(((logits > 0) == labels.bool()).float().detach())
        error = torch.stack(errors)
        state_loss = error.mean()
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

    def update(self, replay):
        if not replay.can_sample(self.cfg.window, "train"):
            return {"skipped_no_windows": 1.0, "updates": float(self.updates)}
        self.model.train()
        aggregate = {}
        accepted = 0
        rejected = 0
        for _ in range(self.cfg.batches_per_update):
            batch = replay.sample_windows(self.cfg.batch_size, self.cfg.window,
                                          split="train", balance_families=self.cfg.balance_families,
                                          recent_fraction=self.cfg.recent_fraction,
                                          recent_steps=self.cfg.recent_steps)
            self.optimizer.zero_grad(set_to_none=True)
            width = batch.states.shape[1]
            chunk = self.cfg.microbatch_size or width
            metrics, finite = {}, True
            for begin in range(0, width, chunk):
                end = min(width, begin + chunk)
                values = {}
                for field in fields(batch):
                    value = getattr(batch, field.name)
                    # Temporal payloads have [K,B,...]; IDs have [B].
                    values[field.name] = (value[:, begin:end] if field.name in (
                        "states", "actions", "context", "dt", "rewards", "terminated",
                        "truncated", "success") else value[begin:end])
                loss, chunk_metrics = self.loss_on(type(batch)(**values))
                if not bool(torch.isfinite(loss)):
                    finite = False
                    break
                weight = (end - begin) / width
                (loss * weight).backward()
                for key, value in chunk_metrics.items():
                    metrics[key] = metrics.get(key, 0.) + value * weight
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
        result = {key: float(value / max(1, accepted)) for key, value in aggregate.items()}
        result.update(updates=float(self.updates), skipped_nonfinite=float(rejected))
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
            result["validation_available"] = 1.0
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
