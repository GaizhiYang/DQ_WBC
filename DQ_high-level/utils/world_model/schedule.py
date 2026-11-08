"""PPO-boundary budgets for manipulation: fit data first, then value, then policy.

This is a DQ adaptation, not the reference runner's iteration/UTD schedule.
Only fractional WM credit survives a boundary. Actor work never accrues debt.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ScheduleConfig:
    batch_size: int = 2048
    wm_epochs: float = 1.0
    wm_max_batches: int = 512
    wm_prepare_boundaries: int = 4
    value_prepare_boundaries: int = 2
    value_prepare_batches: int = 8
    actor_stage1_boundaries: int = 4
    actor_stage2_boundaries: int = 4
    actor_stage1_updates: int = 1
    actor_stage2_updates: int = 2
    actor_max_updates: int = 4

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "wm_epochs":
                if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                    raise ValueError("wm_epochs must be positive and finite")
            elif isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError(name + " must be a positive integer")
        if not self.actor_stage1_updates <= self.actor_stage2_updates <= self.actor_max_updates <= 4:
            raise ValueError("Actor budgets must increase monotonically and never exceed four per PPO boundary")

    @classmethod
    def from_mapping(cls, values):
        return cls(**{name: values[name] for name in cls.__dataclass_fields__ if name in values})


@dataclass(frozen=True)
class BoundaryPlan:
    phase: str
    new_transitions: int
    model_batches: int
    discarded_batches: int
    value_batches: int
    actor_updates: int


class ManipulationSchedule:
    def __init__(self, config=None):
        self.cfg = config or ScheduleConfig()
        self.reset_replay()

    def reset_replay(self):
        """Replay is not checkpointed: its watermarks and preparation restart."""
        self.seen_transitions = 0
        self.credit = 0.0
        self.completed = {name: 0 for name in ("W", "V", "B1", "B2", "B3")}
        self.boundaries = 0
        self.discarded_batches = 0

    @property
    def phase(self):
        for name, required in (("W", self.cfg.wm_prepare_boundaries),
                               ("V", self.cfg.value_prepare_boundaries),
                               ("B1", self.cfg.actor_stage1_boundaries),
                               ("B2", self.cfg.actor_stage2_boundaries)):
            if self.completed[name] < required:
                return name
        return "B3"

    def plan(self, total_valid_train_transitions, *, can_sample=True):
        total = int(total_valid_train_transitions)
        if total < self.seen_transitions:
            raise ValueError("Replay transition counter moved backwards; reset the schedule with new replay")
        new = total - self.seen_transitions
        self.seen_transitions = total
        credit = self.credit + self.cfg.wm_epochs * new / self.cfg.batch_size
        whole = math.floor(credit + 1e-12)
        self.credit = max(0., credit - whole)
        batches = min(whole, self.cfg.wm_max_batches) if can_sample else 0
        discarded = whole - batches
        self.discarded_batches += discarded
        self.boundaries += 1
        phase = self.phase
        actor = {"B1": self.cfg.actor_stage1_updates,
                 "B2": self.cfg.actor_stage2_updates,
                 "B3": self.cfg.actor_max_updates}.get(phase, 0)
        # No new fitting work means no stale-data Actor or value preparation.
        return BoundaryPlan(phase, new, batches, discarded,
                            self.cfg.value_prepare_batches if phase == "V" and batches else 0,
                            actor if batches else 0)

    def complete(self, plan, *, model_updates, value_batches=0, actor_updates=0):
        if plan.phase != self.phase:
            raise ValueError("Complete each boundary before preparing the next phase")
        complete = plan.model_batches > 0 and model_updates == plan.model_batches
        if plan.phase == "V":
            complete &= value_batches == plan.value_batches and value_batches > 0
        elif plan.phase.startswith("B"):
            complete &= actor_updates == plan.actor_updates and actor_updates > 0
        if complete:
            self.completed[plan.phase] += 1
        return bool(complete)

    def state_dict(self):
        return {"config": asdict(self.cfg), "seen_transitions": self.seen_transitions,
                "credit": self.credit, "completed": dict(self.completed),
                "boundaries": self.boundaries, "discarded_batches": self.discarded_batches}
