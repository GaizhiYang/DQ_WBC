"""Evaluate the deployable asymmetric actor using mean actions."""
from train_multistate_DQ_asymmetric_teacher import get_trainer
from utils import asymmetric_distributed as distributed


if __name__ == "__main__":
    try:
        get_trainer(is_eval=True).eval()
    finally:
        distributed.shutdown()
