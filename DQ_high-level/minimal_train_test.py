"""Bounded simulation diagnostic using the shared SuperMimic v2 training loop.

Defaults to 256 environments and 64 vector steps. Explicit --num_envs and
--stop_after arguments override these diagnostic limits. This is normally a
world-model warmup check; it does not claim that joint training was exercised.
"""
import sys


def minimal_train():
    original_argv = sys.argv[:]
    options = original_argv[1:]
    for name, default in (("--num_envs", "256"), ("--stop_after", "64")):
        if not any(option == name or option.startswith(name + "=") for option in options):
            options += [name, default]
    sys.argv = [original_argv[0], *options]
    try:
        # The training entry imports Isaac Gym before PyTorch, as required by
        # the simulator. Keep this import inside the callable for test discovery.
        from train_supermimic_v2 import train_supermimic_v2
        agent = train_supermimic_v2()
        stats = agent.get_stats()
        world_updates = stats["world_model_updates"]
        policy_updates = stats["policy_updates"]
        print(f"Diagnostic optimizer updates: world model={world_updates}, policy={policy_updates}")
        if world_updates == 0:
            print("Collection completed, but no optimizer update ran; the replay buffer may be below its training threshold.")
        elif policy_updates == 0:
            print("World-model warmup check passed. Joint policy training was not exercised by this run.")
        else:
            print("World-model and joint policy optimizer updates completed.")
        return agent
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    minimal_train()
