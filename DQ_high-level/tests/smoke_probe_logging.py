"""Offline W&B/TensorBoard regression across a real isolated CUDA probe.

Run manually in the Isaac Gym environment (not part of unit-test discovery)::

    python tests/smoke_probe_logging.py --checkpoint /path/to/candidate.pt

Writes only to a temporary directory unless --output_dir is supplied. The
probe's model-quality decision is irrelevant here: all five real simulator
workers must finish without creating a TensorBoard writer, and the main
writer's scalars must still reach the local offline W&B history afterward.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


def offline_history(path):
    from wandb.proto import wandb_internal_pb2
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(path))
    records = []
    try:
        while True:
            data = store.scan_data()
            if data is None:
                break
            record = wandb_internal_pb2.Record()
            record.ParseFromString(data)
            if record.WhichOneof("record_type") == "history":
                records.append({item.key or "/".join(item.nested_key):
                                json.loads(item.value_json)
                                for item in record.history.item})
    finally:
        store.close()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir")
    parser.add_argument("--poll_seconds", type=float, default=12.,
                        help="Wait beyond W&B's 10-second TensorBoard polling period")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error("checkpoint does not exist")
    if args.poll_seconds < 11.:
        parser.error("poll_seconds must be at least 11 for the real SDK reader")
    output = Path(args.output_dir or tempfile.mkdtemp(prefix="dq_probe_logging_fixed_"))
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    run_dir = output / "teacher_run"
    report_path = run_dir / "world_model" / "probe.json"
    report_path.parent.mkdir(parents=True)
    print("Artifacts: {}".format(output), flush=True)

    # Explicit offline mode is required even if the caller exports an API key
    # or online mode. No metrics or artifacts are uploaded by this diagnostic.
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DISABLE_CODE"] = "true"
    import wandb
    run = wandb.init(project="dq-probe-logging-regression", mode="offline",
                     dir=str(output), sync_tensorboard=True,
                     settings=wandb.Settings(console="off"))
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(run_dir))
    command = [sys.executable, str(Path(__file__).resolve().parents[1] /
                                  "probe_teacher_world_model.py"),
               "--checkpoint", str(checkpoint), "--output", str(report_path),
               "--num_envs", "6", "--prefix_steps", "2", "--device", args.device]
    environment = os.environ.copy()
    environment["PATH"] = str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    environment["LD_LIBRARY_PATH"] = str(Path(sys.executable).parent.parent / "lib") + os.pathsep + environment.get("LD_LIBRARY_PATH", "")
    probe_returncode = None
    try:
        for step in (1, 2):
            writer.add_scalar("iteration", step, step * 24)
            writer.add_scalar("Reward", step * 11, step * 24)
            writer.flush()
            time.sleep(args.poll_seconds)
        with (output / "probe.log").open("w") as stream:
            result = subprocess.run(command, cwd=str(Path(__file__).resolve().parents[1]),
                                    env=environment, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=900.)
            probe_returncode = result.returncode
        print("Probe exit code: {}".format(probe_returncode), flush=True)
        for step in (3, 4):
            writer.add_scalar("iteration", step, step * 24)
            writer.add_scalar("Reward", step * 11, step * 24)
            writer.flush()
            time.sleep(args.poll_seconds)
    finally:
        writer.close()
        run.finish()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise AssertionError("Expected exactly one main event file: {}".format(event_files))
    tensorboard = EventAccumulator(str(event_files[0])).Reload()
    tb_iterations = [int(event.value) for event in tensorboard.Scalars("iteration")]
    histories = []
    for file in sorted(output.glob("wandb/offline-run-*/*.wandb")):
        histories.extend(offline_history(file))
    wb_iterations = sorted({int(row["iteration"]) for row in histories if "iteration" in row})
    wb_rewards = sorted({int(row["Reward"]) for row in histories if "Reward" in row})
    probe_events = sorted(str(path) for path in report_path.parent.rglob("events.out.tfevents.*"))
    probe_report = json.loads(report_path.read_text()) if report_path.exists() else None
    result = {"wandb_version": wandb.__version__, "mode": "offline",
              "checkpoint": str(checkpoint), "device": args.device,
              "command": command, "probe_returncode": probe_returncode,
              "probe_report_exists": probe_report is not None,
              "probe_model_quality_passed": probe_report.get("passed") if probe_report else None,
              "probe_event_files": probe_events,
              "tensorboard_iterations": tb_iterations, "wandb_iterations": wb_iterations,
              "wandb_rewards": wb_rewards, "wandb_history": histories}
    result["passed"] = (probe_returncode == 0 and probe_report is not None and
                        not probe_events and tb_iterations == [1, 2, 3, 4] and
                        wb_iterations == [1, 2, 3, 4] and wb_rewards == [11, 22, 33, 44])
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "wandb_history"}, indent=2))
    if not result["passed"]:
        raise AssertionError("Probe logging regression failed; inspect {}/result.json".format(output))


if __name__ == "__main__":
    main()
