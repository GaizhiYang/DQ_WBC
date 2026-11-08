"""Wall-clock metric snapshots while a model block pauses real collection.

The model-update axis counts actual WM optimizer steps. Environment progress
is kept separately: hundreds of model updates must not invent physics steps.
These helpers do not change optimization, replay or accumulated episode data.
"""

import time
import warnings


def _enabled(agent):
    return (getattr(agent, "write_interval", 0) > 0
            and agent.cfg.get("experiment", {}).get("world_model_live_logging", False))


def flush_before_model_update(agent, high_step):
    """Publish collected rewards/PPO losses before entering a long model block."""
    if _enabled(agent) and agent.tracking_data:
        agent.write_tracking_data(int(high_step), 0)


def log_model_snapshot(agent, metrics, *, force=False):
    """Publish at most once per ten seconds, plus explicit block endpoints."""
    if not _enabled(agent):
        return False
    now = time.monotonic()
    interval = float(agent.cfg["experiment"].get("world_model_log_seconds", 10.))
    last = getattr(agent, "_wm_last_metric_write", None)
    if not force and last is not None and now - last < interval:
        return False
    update = int(agent.wm_trainer.updates)
    high_step = int(agent.wm_replay.steps)
    payload = {"World model live / " + key: float(value) for key, value in metrics.items()}
    payload["world_model_update"] = update
    payload["World model live / collected_high_steps"] = high_step
    payload["World model live / model_updates"] = update
    agent.write_scalar_snapshot(payload, update, global_step=high_step)
    agent._wm_last_metric_write = now
    return True


def close_teacher_logging(agent, exit_code=0):
    """Flush local events and finish this run, including after Ctrl+C.

    Logging cleanup must not replace the original training exception.
    """
    writer = getattr(agent, "writer", None)
    run = getattr(agent, "_wandb_run", None)
    if writer is not None:
        try:
            writer.close()
        except Exception as error:
            warnings.warn("Could not close teacher TensorBoard writer: {}".format(error))
    if run is not None:
        try:
            run.finish(exit_code=exit_code)
        except Exception as error:
            warnings.warn("Could not finish teacher W&B upload: {}".format(error))
