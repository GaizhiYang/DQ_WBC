"""Reproduce teacher microbenchmarks without importing Isaac Gym.

python doc/assets/dq_teacher_karl/benchmark.py --device cuda:1
Numbers exclude simulation, normalization, candidate transforms and optim.step.
"""
import argparse
import ast
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "DQ_high-level"))
sys.path.insert(0, str(ROOT / "third_party" / "skrl"))
from skrl.models.torch import Model, GaussianMixin, DeterministicMixin
from modules.predictattention import PredictAttentionSelector
from modules.karl_teacher import KarlTeacherPolicy, KarlTeacherValue
from modules.karl_grasp_selector import KarlGraspSelector


def original_models():
    source = ROOT / "DQ_high-level" / "train_multistate_DQ_teacher.py"
    classes = [node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name in ("Policy", "Value")]
    namespace = dict(torch=torch, nn=nn, np=np, Model=Model,
                     GaussianMixin=GaussianMixin, DeterministicMixin=DeterministicMixin,
                     PredictAttentionSelector=PredictAttentionSelector, cprint=lambda *a, **k: None)
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["Policy"], namespace["Value"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=40)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(43)
    device = torch.device(args.device)
    policy, value = original_models()
    old = [cls((1276,), (9,), device, 1024, 128, None, None, None).to(device) for cls in (policy, value)]
    new = [cls((1102,), (9,), device).to(device) for cls in (KarlTeacherPolicy, KarlTeacherValue)]

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def measure(fn):
        for _ in range(10):
            fn()
        synchronize()
        samples = []
        for _ in range(5):
            start = time.perf_counter()
            for _ in range(args.iterations):
                fn()
            synchronize()
            samples.append((time.perf_counter() - start) * 1000 / args.iterations)
        return round(statistics.median(samples), 4)

    rows = []
    for batch in (1, 128, 4096):
        raw = torch.randn(batch, 1276, device=device)
        selector = KarlGraspSelector(batch, device)

        def pack():
            poses = torch.cat((raw[:, -189:-99].reshape(-1, 30, 3), raw[:, -99:-9].reshape(-1, 30, 3)), -1)
            selected, _ = selector.select(poses, raw[:, 1030:1036])
            return torch.cat((raw[:, :-189], selected, raw[:, -9:]), -1)

        packed = pack()

        def forward(models, states):
            return models[0].compute({"states": states}, "policy")[0], models[1].compute({"states": states}, "value")[0]

        def update(models, states):
            for model in models:
                model.zero_grad(set_to_none=True)
            action, prediction = forward(models, states)
            (action.square().mean() + prediction.square().mean()).backward()

        with torch.no_grad():
            baseline = measure(lambda: forward(old, raw))
            selected = measure(pack)
            rollout = measure(lambda: forward(new, pack()))
        rows.append({"batch": batch, "gfm_actor_critic_forward_ms": baseline,
                     "karl_selection_pack_ms": selected,
                     "karl_selection_pack_actor_critic_ms": rollout,
                     "gfm_forward_backward_ms": measure(lambda: update(old, raw)),
                     "karl_forward_backward_ms": measure(lambda: update(new, packed))})
    report = {"torch": torch.__version__, "device": str(device),
              "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
              "timing": "median of 5 repeats; 10 warmups; synchronized wall time; milliseconds",
              "iterations_per_repeat": args.iterations,
              "excluded": "simulation, raw pose transforms, standard scaler, optimizer step",
              "parameters": {"gfm": [sum(p.numel() for p in m.parameters()) for m in old],
                             "karl": [sum(p.numel() for p in m.parameters()) for m in new]},
              "rows": rows}
    output = Path(__file__).with_name("benchmark_results.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
