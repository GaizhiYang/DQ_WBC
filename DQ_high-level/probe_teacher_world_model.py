#!/usr/bin/env python3
"""Validate candidate action effects using independent Isaac Gym worker processes.

Example (activate dqwbc first)::

    python probe_teacher_world_model.py --checkpoint candidate.pt \
        --output probe.json --num_envs 128 --prefix_steps 4 --seed 43

The candidate bundle includes the model, matching teacher/scalers and task
configuration. A fixed deterministic-mean Actor prefix is collected once.
Base/+/-/base-repeat workers then create fresh simulators and replay that
prefix with identical seeds. A failed statistical validation still writes an
honest report and exits normally; worker/runtime errors exit nonzero.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def parser():
    arguments = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    arguments.add_argument("--checkpoint", required=True, help="Single candidate bundle saved by WorldModelPPO.save_candidate")
    arguments.add_argument("--output", required=True, help="JSON action-effect report")
    arguments.add_argument("--num_envs", type=int, default=128)
    arguments.add_argument("--prefix_steps", type=int, default=4)
    arguments.add_argument("--seed", type=int, default=43)
    arguments.add_argument("--device", default="cuda:0")
    arguments.add_argument("--worker_timeout", type=float, default=600.)
    arguments.add_argument("--min_pairs", type=int, default=64)
    arguments.add_argument("--min_informative_pairs", type=int, default=32)
    arguments.add_argument("--family_sampling", choices=("balanced", "original"), default="balanced",
                           help="Sampling only in isolated probe worlds; main SGS remains unchanged")
    arguments.add_argument("--workdir", default=None, help="Evidence/log directory; defaults to OUTPUT.artifacts")
    arguments.add_argument("--worker", choices=("plan", "base", "plus", "minus", "repeat"),
                           default=None, help=argparse.SUPPRESS)
    arguments.add_argument("--bundle_sha256", default=None, help=argparse.SUPPRESS)
    return arguments


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bundle(torch, path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if bundle.get("schema") != "dq-world-model-candidate-v1":
        raise ValueError("Expected a dq-world-model-candidate-v1 checkpoint")
    for name in ("model", "metadata", "config", "teacher", "teacher_config", "teacher_arguments"):
        if not bundle.get(name):
            raise ValueError("Candidate bundle is missing %s; re-export it with the matching teacher/configuration" % name)
    return bundle


def _new_session(cli, bundle, torch):
    # Isaac Gym must be imported by the worker before PyTorch (see _worker).
    import yaml
    import train_multistate_DQ_teacher as teacher
    from utils.config import get_parser
    from utils.wm_env import TeacherWorldModelAdapter
    from utils.wm_fixed8 import Fixed8TeacherWorldModelAdapter
    from utils.world_model.probes import (IsaacGymProbeSession, configuration_fingerprint,
                                         balanced_family_config_ids)

    args = get_parser().parse_args([])
    for name, value in bundle["teacher_arguments"].items():
        if hasattr(args, name):
            setattr(args, name, value)
    # These changes affect diagnostics/resources, not the training task.
    args.num_envs, args.seed = cli.num_envs, cli.seed
    args.rl_device = args.sim_device = cli.device
    args.headless, args.debug, args.debugvis = True, False, False
    args.wandb, args.record_video = False, False
    args.checkpoint, args.resume, args.eval = "", False, False
    args.tsil = False
    for name in vars(args):
        if name.startswith("wm_"):
            setattr(args, name, None)
    args.world_model = "off"
    args.experiment_dir = str(Path(cli.workdir)/"runs")
    args.wandb_name = cli.worker
    args.stop_after = None
    cfg = deepcopy(bundle["teacher_config"])
    cfg["env"]["numEnvs"] = cli.num_envs
    cfg["env"]["totalTimesteps"] = int(args.timesteps)
    cfg["env"]["wandb"] = False
    cfg["env"].setdefault("tsil", {})["enabled"] = False
    cfg["env"].setdefault("world_model", {})["mode"] = "off"
    cfg["env"]["world_model"]["fixed8"] = bool(bundle.get("config", {}).get("fixed8", False))
    # Keep reward/curriculum time aligned with the exported training policy.
    cfg["env"]["globalStepCounter"] = int(bundle.get("global_step", bundle.get("collected_steps", 0)))
    cfg_path = Path(cli.workdir)/(cli.worker+"_teacher.yaml")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    args.cfg_file = str(cfg_path)
    original_get_params = teacher.get_params
    original_argv = sys.argv
    try:
        teacher.get_params = lambda: deepcopy(args)
        # B1Z1PickMulti separately reads --timesteps during construction.
        # Do not leak the probe CLI into that legacy parser.
        sys.argv = [str(ROOT/"train_multistate_DQ_teacher.py"), "--timesteps", str(args.timesteps)]
        # Keep training physics/rewards while suppressing all agent loggers.
        # Disabling W&B alone still creates a SummaryWriter, whose nested
        # event file can divert the parent run's TensorBoard synchronization.
        trainer = teacher.get_trainer(is_eval=False, enable_logging=False)
    finally:
        teacher.get_params = original_get_params
        sys.argv = original_argv
    agent = trainer.agents
    weights = bundle["teacher"]
    agent.policy.load_state_dict(weights["policy"])
    agent.value.load_state_dict(weights["value"])
    for name, attribute in (("state_preprocessor", "_state_preprocessor"),
                            ("value_preprocessor", "_value_preprocessor")):
        if weights.get(name) is not None:
            getattr(agent, attribute).load_state_dict(weights[name])
    agent.set_running_mode("eval")
    adapter_class = (Fixed8TeacherWorldModelAdapter if bundle.get("config", {}).get("fixed8", False)
                     else TeacherWorldModelAdapter)
    adapter = adapter_class(trainer.env._env)
    if hasattr(adapter, "metadata"):
        actual_metadata = adapter.metadata()
        if configuration_fingerprint(actual_metadata) != configuration_fingerprint(bundle["metadata"]):
            raise ValueError("Probe task/adapter schema does not match candidate metadata")
    raw_env = trainer.env._env
    sgs_enabled = getattr(raw_env, "sgs_enabled", False)
    fingerprint = raw_env.sgs_bank.fingerprint if sgs_enabled else None
    if "sgs_bank_fingerprint" in bundle and bundle["sgs_bank_fingerprint"] != fingerprint:
        raise ValueError("Probe reset bank differs from the exported teacher's bank fingerprint")
    if sgs_enabled and cli.family_sampling == "balanced":
        original_sample = raw_env.sgs_sampler.sample
        try:
            raw_env.sgs_sampler.sample = lambda count, mode=None: balanced_family_config_ids(
                raw_env.sgs_bank.tensors["family_id"], count, cli.seed+821)
            trainer.env.reset()
        finally:
            raw_env.sgs_sampler.sample = original_sample
    else:
        trainer.env.reset()
    session = IsaacGymProbeSession(trainer.env._env, adapter)
    session.trainer = trainer
    session.probe_families = (raw_env.sgs_bank.tensors["family_id"][raw_env.active_config_id].clone()
                              if sgs_enabled else torch.zeros(cli.num_envs, device=cli.device, dtype=torch.long))
    session.probe_family_names = list(raw_env.sgs_family_names) if sgs_enabled else ["full_task"]
    session.bank_fingerprint = fingerprint
    return session


def _worker(cli):
    # Import order is required by the installed Isaac Gym extension.
    from isaacgym import gymapi  # noqa: F401
    import torch
    from utils.world_model import HierarchicalDynamics, ResidualDynamics
    from utils.world_model.probes import (bounded_probe_action, model_fingerprint,
                                         random_continuous_directions, seed_probe)

    if not cli.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Real teacher probes require CUDA simulation and CUDA model execution")
    torch.set_num_threads(1)
    seed_probe(cli.seed)
    if cli.bundle_sha256 is not None and _file_sha256(cli.checkpoint) != cli.bundle_sha256:
        raise ValueError("Candidate bundle changed after probe launch; model and teacher must be one immutable snapshot")
    bundle = _load_bundle(torch, cli.checkpoint)
    session = _new_session(cli, bundle, torch)
    try:
        with torch.no_grad():
            if cli.worker == "plan":
                prefix = []
                for _ in range(cli.prefix_steps):
                    raw_obs = session.env.obs_buf.clamp(-session.env.clip_obs, session.env.clip_obs)
                    obs = session.trainer.agents._state_preprocessor(raw_obs, train=False)
                    action, _, _ = session.trainer.agents.policy.compute({"states": obs}, role="policy")
                    action = session.trainer.agents.policy.mean_to_action(action)
                    prefix.append(action.detach().clone())
                    session.step(action)
                raw_obs = session.env.obs_buf.clamp(-session.env.clip_obs, session.env.clip_obs)
                obs = session.trainer.agents._state_preprocessor(raw_obs, train=False)
                base, _, _ = session.trainer.agents.policy.compute({"states": obs}, role="policy")
                # compute returns a latent Gaussian mean; the environment
                # consumes sigmoid-transformed continuous gripper opening.
                base = session.trainer.agents.policy.mean_to_action(base).detach().clone()
                generator = torch.Generator(device=cli.device).manual_seed(cli.seed+1943)
                directions = random_continuous_directions(base, generator=generator)
                state, context = session.capture()
                plan = {"prefix": (torch.stack(prefix) if prefix else
                                   torch.empty(0, cli.num_envs, 9, device=cli.device)),
                        "base": base, "directions": directions, "start": state, "context": context,
                        "families": session.probe_families}
                torch.save({key: value.cpu() for key, value in plan.items()}, Path(cli.workdir)/"plan.pt")
                print("Recorded fixed mean-policy prefix: %d steps, %d CUDA environments" % (cli.prefix_steps, cli.num_envs), flush=True)
                return
            plan = torch.load(Path(cli.workdir)/"plan.pt", map_location=cli.device, weights_only=False)
            alive = torch.ones(cli.num_envs, device=cli.device, dtype=torch.bool)
            prefix_dt = []
            for action in plan["prefix"]:
                transition = session.step(bounded_probe_action(action))
                alive &= ~transition.terminated
                prefix_dt.append(transition.dt)
            state, context = session.capture()
            multiplier = {"base": 0., "plus": 1., "minus": -1., "repeat": 0.}[cli.worker]
            action = bounded_probe_action(plan["base"], plan["directions"], multiplier)
            physical_context = session.adapter.normalize_context(context)
            transition = session.step(action)
            action = transition.executed_action
            config = bundle["config"]
            if config.get("fixed8", False):
                model = HierarchicalDynamics(
                    session.adapter, hidden=tuple(config.get("hidden", (512, 512, 512))),
                    contact_hidden=tuple(config.get("contact_hidden", (128, 128))),
                    nominal_dt=.16, device=cli.device)
            else:
                model = ResidualDynamics(
                    session.adapter, hidden=tuple(config.get("hidden", (512, 512, 512))),
                    term_hidden=tuple(config.get("term_hidden", (64, 64))),
                    nominal_dt=float(config.get("nominal_dt", .15)), device=cli.device)
            model.load_state_dict(bundle["model"])
            model.eval()
            predicted = model.step(state, action, context, transition.dt)
            predicted_reward = session.adapter.reward(state, predicted, action, context).reshape(-1)
            record = {"start": state, "context": context, "physical_context": physical_context,
                      "reward": transition.reward, "dt": transition.dt, "model_reward": predicted_reward,
                      "alive": alive, "prefix_dt": (torch.stack(prefix_dt) if prefix_dt else
                                        torch.empty(0, cli.num_envs, 1, device=cli.device)),
                      "next_state": transition.state, "predicted_state": predicted,
                      "actions": action, "families": session.probe_families}
            record = {name: value.detach().cpu() for name, value in record.items()}
            record["model_sha256"] = model_fingerprint(model)
            record["configuration"] = bundle["metadata"]
            record["family_names"] = session.probe_family_names
            record["sgs_bank_fingerprint"] = session.bank_fingerprint
            torch.save(record, Path(cli.workdir)/(cli.worker+".pt"))
            print("%s: real reward mean %.8g; model reward mean %.8g; dt %.4g; device %s" %
                  (cli.worker, float(transition.reward.mean()), float(predicted_reward.mean()),
                   float(transition.dt.mean()), state.device), flush=True)
    finally:
        writer = getattr(session.trainer.agents, "writer", None)
        if writer is not None:
            writer.close()
        session.close()


def _run(cli):
    # Parent never creates a simulator. Independent processes also isolate
    # PhysX lifetime, CUDA extension global state, loggers and SGS statistics.
    import torch
    from utils.world_model.probes import (ActionEffectProbeConfig,
        configuration_fingerprint, summarize_replayed_branches, write_report)

    bundle = _load_bundle(torch, cli.checkpoint)
    bundle_hash = _file_sha256(cli.checkpoint)
    directory = Path(cli.workdir)
    directory.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for worker in ("plan", "base", "plus", "minus", "repeat"):
        # ROOT was resolved before main() changes cwd. Python 3.8 may retain a
        # relative __file__, whose later resolve() would duplicate DQ_high-level.
        command = [sys.executable, str(ROOT / "probe_teacher_world_model.py"), "--checkpoint", cli.checkpoint,
                   "--output", cli.output, "--num_envs", str(cli.num_envs),
                   "--prefix_steps", str(cli.prefix_steps), "--seed", str(cli.seed),
                   "--device", cli.device, "--workdir", cli.workdir, "--worker", worker,
                   "--family_sampling", cli.family_sampling, "--bundle_sha256", bundle_hash]
        logfile = directory/(worker+".log")
        print("Launching isolated Isaac Gym %s worker; log: %s" % (worker, logfile), flush=True)
        worker_environment = dict(os.environ)
        worker_environment["PATH"] = str(Path(sys.prefix)/"bin") + os.pathsep + worker_environment.get("PATH", "")
        library_directory = str(Path(sys.prefix)/"lib")
        library_path = worker_environment.get("LD_LIBRARY_PATH", "")
        worker_environment["LD_LIBRARY_PATH"] = library_directory + (os.pathsep+library_path if library_path else "")
        with logfile.open("w", encoding="utf-8") as stream:
            result = subprocess.run(command, cwd=str(ROOT), stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=cli.worker_timeout, env=worker_environment)
        if result.returncode:
            tail = "\n".join(logfile.read_text(encoding="utf-8", errors="replace").splitlines()[-35:])
            raise RuntimeError("Isaac Gym %s worker exited %d\n%s" % (worker, result.returncode, tail))
    branches = {name: torch.load(directory/(name+".pt"), map_location=cli.device, weights_only=False)
                for name in ("base", "plus", "minus", "repeat")}
    hashes = {value.pop("model_sha256") for value in branches.values()}
    if len(hashes) != 1:
        raise ValueError("Candidate weights changed across independent probe workers")
    expected_contract = configuration_fingerprint(bundle["metadata"])
    family_names = branches["base"]["family_names"]
    bank_fingerprint = branches["base"]["sgs_bank_fingerprint"]
    for branch in branches.values():
        if configuration_fingerprint(branch.pop("configuration")) != expected_contract:
            raise ValueError("Probe task configuration changed between workers")
        if branch.pop("family_names") != family_names or branch.pop("sgs_bank_fingerprint") != bank_fingerprint:
            raise ValueError("Probe reset families/bank changed between workers")
    config = ActionEffectProbeConfig(min_pairs=cli.min_pairs,
                                    min_informative_pairs=cli.min_informative_pairs)
    report = summarize_replayed_branches(branches, next(iter(hashes)), bundle["metadata"], config,
        metadata={"simulator": "IsaacGym", "device": cli.device, "seed": cli.seed,
                  "prefix_length": cli.prefix_steps, "protocol": "independent_simulator_prefix_replay",
                  "process_isolation": True, "candidate": cli.checkpoint,
                  "candidate_bundle_sha256": bundle_hash,
                  "family_names": family_names, "family_sampling": cli.family_sampling,
                  "sgs_bank_fingerprint": bank_fingerprint,
                  "evidence_directory": cli.workdir, "elapsed_seconds": time.time()-start})
    write_report(cli.output, report)
    print(json.dumps({"report": cli.output, "passed": report["passed"],
                      "reasons": report["reasons"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


def main():
    cli = parser().parse_args()
    if cli.num_envs < 1 or cli.prefix_steps < 0 or cli.worker_timeout <= 0:
        raise ValueError("Probe batch size/timeout must be positive and prefix_steps nonnegative")
    if cli.min_pairs < 1 or cli.min_informative_pairs < 2:
        raise ValueError("Probe requires min_pairs >= 1 and min_informative_pairs >= 2")
    cli.checkpoint = str(Path(cli.checkpoint).expanduser().resolve())
    cli.output = str(Path(cli.output).expanduser().resolve())
    cli.workdir = str(Path(cli.workdir or (cli.output+".artifacts")).expanduser().resolve())
    Path(cli.workdir).mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    if cli.worker is not None:
        _worker(cli)
    else:
        _run(cli)


if __name__ == "__main__":
    main()
