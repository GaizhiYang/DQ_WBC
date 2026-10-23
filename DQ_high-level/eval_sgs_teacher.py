"""Evaluate a teacher on the original 2 m Level02 task with hold success.

Run from DQ_high-level, using the usual teacher flags plus --checkpoint and
--eval_episodes. New runs restore their saved observation/control arguments;
older checkpoints require the same model flags used during their training.
Evaluation uses synchronized waves with fixed reset seeds. Each environment
contributes at most one episode per wave, so early failures do not displace
slower test episodes. No transition is recorded and no PPO update is called.
"""

import json
from pathlib import Path


def evaluate_teacher(trainer, episodes):
    import random
    import numpy as np
    import torch

    if episodes <= 0:
        raise ValueError("The evaluation episode count must be positive")
    env, agent = trainer.env, trainer.agents
    if env.cfg["env"].get("successMode") != "hold":
        raise ValueError("Strict evaluation requires successMode=hold")
    if env.cfg["env"].get("sgs", {}).get("mode", "disabled") != "disabled":
        raise ValueError("Strict evaluation must not sample or update the training bank")
    agent.set_mode("eval")
    agent.policy.deterministic = True
    scaler_states = {}
    for name in ("_state_preprocessor", "_value_preprocessor"):
        preprocessor = getattr(agent, name, None)
        if hasattr(preprocessor, "eval"):
            preprocessor.eval()
        if hasattr(preprocessor, "state_dict"):
            scaler_states[name] = {key: value.detach().clone()
                                   for key, value in preprocessor.state_dict().items()}

    device = agent.device
    eval_seed = trainer.teacher_args.seed
    total, n_success, n_lifts = 0, 0, 0
    waves = (episodes + env.num_envs - 1) // env.num_envs
    steps = 0
    with torch.no_grad():
        for wave in range(waves):
            # Reset randomness depends on the test case, not the previous policy's
            # episode lengths. Keep seed, num_envs and episodes fixed across runs.
            wave_seed = (eval_seed + wave) % (2 ** 32)
            random.seed(wave_seed)
            np.random.seed(wave_seed)
            torch.manual_seed(wave_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(wave_seed)
            env.reset_buf.fill_(1)
            states, _ = env.reset()
            active = torch.arange(env.num_envs, device=device) < min(env.num_envs, episodes - total)
            completed = ~active
            episode_lifted = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            for _ in range(env.max_episode_length):
                actions = agent.act(states, timestep=steps, timesteps=trainer.timesteps)[0]
                actions[completed] = 0
                states, _, terminated, truncated, _ = env.step(actions)
                if not trainer.headless:
                    env.render()
                episode_lifted |= env.lifted_object.to(device) & ~completed
                finished = torch.logical_or(terminated, truncated).reshape(-1).to(device)
                counted = finished & ~completed
                n_success += int(env.episode_success.to(device)[counted].sum().item())
                n_lifts += int(episode_lifted[counted].sum().item())
                total += int(counted.sum().item())
                completed |= finished
                steps += 1
                if completed.all().item():
                    break
            if not completed.all().item():
                raise RuntimeError("An evaluation episode exceeded its configured time limit")

    if total != episodes:
        raise RuntimeError(f"Only {total}/{episodes} evaluation episodes completed within their configured time limits")
    for name, before in scaler_states.items():
        after = getattr(agent, name).state_dict()
        if before.keys() != after.keys() or any(not torch.equal(value, after[key]) for key, value in before.items()):
            raise RuntimeError(f"Evaluation modified the frozen scaler {name}")
    return {"episodes": total, "hold_successes": n_success,
            "hold_success_rate": n_success / total,
            "ever_lifted_episodes": n_lifts,
            "num_envs": env.num_envs, "control_steps": steps,
            "reset_protocol": "synchronized_waves", "waves": waves,
            "wave_seed": "eval_seed + wave_index",
            "deterministic_policy": agent.policy.deterministic,
            "scalers_unchanged": True,
            "success_mode": "hold", "hold_steps": env.hold_steps,
            "start_distance": 2.0, "task_level": "Level02",
            "sampler_mode": "disabled"}


def main():
    # Isaac Gym must be imported before torch; the teacher entrypoint does that.
    from train_multistate_DQ_teacher import get_trainer

    trainer = get_trainer(is_eval=True, strict_eval=True)
    args = trainer.teacher_args
    result = evaluate_teacher(trainer, args.eval_episodes)
    result.update({"checkpoint": str(Path(args.checkpoint).resolve()), "seed": args.seed})
    if args.eval_output:
        output = Path(args.eval_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
