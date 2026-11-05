"""Shared normalized gripper control, independent of Isaac Gym."""

GRIPPER_ACTION_MODE = "continuous_opening_v1"
CLOSING_OPENING_THRESHOLD = 0.5


def opening_to_joint(opening, closed, opened):
    """0 is closed and 1 is open; preserve gradients inside the valid range."""
    return closed + opening.clamp(0., 1.) * (opened - closed)


def closing_command(opening):
    """Closing-side intent for task bookkeeping, not a contact-force test."""
    return opening < CLOSING_OPENING_THRESHOLD
