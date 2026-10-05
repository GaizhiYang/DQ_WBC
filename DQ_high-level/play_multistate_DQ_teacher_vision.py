"""Deterministic evaluation of a saved visual-teacher experiment."""
from train_multistate_DQ_teacher_vision import get_trainer


if __name__ == "__main__":
    get_trainer(is_eval=True).eval()
