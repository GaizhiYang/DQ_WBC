"""Train the time-to-reach model T from a teacher's own rollouts (main line A, gate D2).

Run from DQ_high-level with the flags used for evaluation (saved runs restore
their observation flags from teacher_run.json; no SGS flags):

  python train_reachability.py --checkpoint DQ_teacher/<group>/<run>/checkpoints/agent_120000.pt \
      --task B1Z1PickMulti --object_name sugar_box --headless --rand_control \
      --num_envs 2048 --reach_output data/reach/sugar_box_mlp

The teacher acts on the ordinary 2 m Level02 task, stochastically by default
for coverage. Every post-step state enters a hindsight buffer; pairs
(s_t, EE position at t+k) labeled with the elapsed time train T with a low
expectile. Environments with id % 10 == 0 are held out for validation.

Outputs in --reach_output (a new directory):
  reach_model.pt      pass to training with --reach_model
  reach_report.json   D2: held-out error of T versus the analytic estimate and floor
  environment.yaml    the configuration the rollouts used
"""

import argparse
import json
import sys
import time
from pathlib import Path

OWN = argparse.ArgumentParser(add_help=False)
OWN.add_argument("--reach_output", type=Path, required=True, help="New output directory")
OWN.add_argument("--reach_kind", choices=("mlp", "quasimetric"), default="mlp")
OWN.add_argument("--reach_steps", type=int, default=3000, help="High-level environment steps")
OWN.add_argument("--reach_updates_per_step", type=int, default=2)
OWN.add_argument("--reach_batch", type=int, default=8192)
OWN.add_argument("--reach_lr", type=float, default=3e-4)
OWN.add_argument("--reach_expectile", type=float, default=0.1)
OWN.add_argument("--reach_window", type=int, default=48)
OWN.add_argument("--reach_max_pair_steps", type=int, default=32)
OWN.add_argument("--reach_validate_every", type=int, default=100)
OWN.add_argument("--reach_validation_batch", type=int, default=65536)
OWN.add_argument("--reach_deterministic", action="store_true",
                 help="Deterministic teacher actions (default: sample for state coverage)")
# D2 passes when T clearly beats the analytic estimate (doc section 3.6).
D2_MAE_RATIO = 0.8


def rollout_config(rest, own, output):
    """The run's YAML plus an online-learner margin section, written to the output."""
    import yaml
    from utils.config import get_params
    from train_multistate_DQ_teacher import restore_eval_arguments, teacher_config_path

    args = get_params(rest)
    if not args.checkpoint:
        raise SystemExit("--checkpoint is required")
    if args.sgs_mode not in (None, "disabled"):
        raise SystemExit("train_reachability.py uses ordinary resets; omit --sgs_mode/--sgs_bank")
    restore_eval_arguments(args)
    path = teacher_config_path(args, is_eval=True)
    cfg = yaml.safe_load(Path(path).read_text())
    margin = cfg["env"].get("margin", {}) or {}
    if margin.get("tokens", False):
        raise SystemExit("The checkpoint's actor uses margin tokens from its own T; collect T data "
                         "with a teacher trained without --margin_tokens")
    cfg["env"]["margin"] = dict(
        reachModel="new:" + own.reach_kind, online=True, reward=False, tokens=False,
        curriculum="disabled", onlineBatch=own.reach_batch, onlineLearningRate=own.reach_lr,
        expectile=own.reach_expectile, window=own.reach_window, maxPairSteps=own.reach_max_pair_steps)
    cfg["reward"].setdefault("scales", {})["margin_shaping"] = 0.0
    # The saved training YAML names its SGS bank; evaluation-mode rollouts use ordinary resets.
    cfg["env"].setdefault("sgs", {})["mode"] = "disabled"
    written = output / "environment.yaml"
    written.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return written


def main():
    own, rest = OWN.parse_known_args()
    for name in ("reach_steps", "reach_updates_per_step", "reach_validate_every", "reach_validation_batch"):
        if getattr(own, name) <= 0:
            raise SystemExit(f"--{name} must be positive")
    import os
    os.chdir(str(Path(__file__).resolve().parent))
    own.reach_output = own.reach_output.resolve()
    own.reach_output.mkdir(parents=True, exist_ok=False)
    # Isaac Gym must be imported before torch: the teacher entry point does that.
    import train_multistate_DQ_teacher as teacher
    import torch
    from modules.reachability import save_reach_model

    config = rollout_config(rest, own, own.reach_output)
    sys.argv = [sys.argv[0]] + rest + ["--cfg_file", str(config)]
    trainer = teacher.get_trainer(is_eval=True)
    env, agent = trainer.env, trainer.agents
    raw = env._env
    learner = raw.margin_learner
    if learner is None:
        raise RuntimeError("the environment did not create the online reach learner")
    agent.set_mode("eval")
    agent.policy.deterministic = own.reach_deterministic
    for name in ("_state_preprocessor", "_value_preprocessor"):
        preprocessor = getattr(agent, name, None)
        if hasattr(preprocessor, "eval"):
            preprocessor.eval()

    history, train_metrics = [], {}
    started = time.time()
    states, _ = env.reset()
    with torch.no_grad():
        for step in range(1, own.reach_steps + 1):
            actions = agent.act(states, timestep=step, timesteps=own.reach_steps)[0]
            states, _, terminated, truncated, _ = env.step(actions)
            if torch.logical_or(terminated, truncated).any():
                states, _ = env.reset()
            # Long pairs exist once the buffer holds a full window.
            if step >= 2:
                train_metrics = learner.update(own.reach_updates_per_step) or train_metrics
            if step % own.reach_validate_every == 0 or step == own.reach_steps:
                quality = learner.validate(own.reach_validation_batch)
                entry = {"step": step, "updates": learner.updates, "seconds": time.time() - started,
                         "train": train_metrics, "validation": quality}
                history.append(entry)
                print(json.dumps({"step": step, "updates": learner.updates,
                                  "loss": train_metrics.get("loss"),
                                  "learned_mae": quality.get("learned_mae"),
                                  "analytic_mae": quality.get("analytic_mae")}), flush=True)

    final = learner.validate(own.reach_validation_batch)
    learned, analytic = final.get("learned_mae"), final.get("analytic_mae")
    passed = learned is not None and analytic is not None and learned <= D2_MAE_RATIO * analytic
    metadata = {
        "gripper_body_index": int(raw.gripper_idx), "source": "train_reachability.py",
        "checkpoint": str(Path(trainer.teacher_args.checkpoint).resolve()),
        "object": list(raw.obj_list), "table_level": raw.table_level,
        "kind": own.reach_kind, "expectile": own.reach_expectile, "updates": learner.updates,
        "env_steps": own.reach_steps, "num_envs": raw.num_envs,
        "deterministic_policy": own.reach_deterministic,
        "step_seconds": raw.margin_step_seconds,
    }
    save_reach_model(own.reach_output / "reach_model.pt", learner.target, metadata)
    report = {"gate": "D2", "passed": passed,
              "criterion": f"held-out learned MAE <= {D2_MAE_RATIO} x analytic MAE",
              "final_validation": final, "history": history, "metadata": metadata,
              "notes": ("Labels are hindsight upper bounds of the minimum time. Low "
                        "above_label_fraction plus low MAE means a tight lower envelope.")}
    (own.reach_output / "reach_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"gate": "D2", "passed": passed, "learned_mae": learned,
                      "analytic_mae": analytic, "output": str(own.reach_output)}, indent=2))


if __name__ == "__main__":
    main()
