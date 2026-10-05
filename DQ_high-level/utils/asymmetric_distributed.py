"""torchrun lifecycle and collectives for synchronous asymmetric PPO.

Keep torch imports lazy: Isaac Gym must be imported first by the entrypoint.
"""
import os
from datetime import timedelta


def launch_world_size():
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    return size


def active():
    import torch.distributed as dist
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank():
    import torch.distributed as dist
    return dist.get_rank() if active() else 0


def world_size():
    import torch.distributed as dist
    return dist.get_world_size() if active() else 1


def is_main():
    return rank() == 0


def configure_devices(args):
    """Choose one visible CUDA device per local worker before constructing Gym."""
    size = launch_world_size()
    if size > 1:
        local_rank = int(os.environ["LOCAL_RANK"])
        local_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(size)))
        if not 0 <= local_rank < local_size:
            raise ValueError("Invalid LOCAL_RANK / LOCAL_WORLD_SIZE")
        if args.sim_device not in ("cuda", "cuda:0") or args.rl_device not in ("cuda", "cuda:0"):
            raise ValueError("torchrun assigns CUDA devices by LOCAL_RANK; select GPUs with CUDA_VISIBLE_DEVICES")
        if args.graphics_device_id != -1:
            raise ValueError("Use --graphics_device_ids with torchrun, not a shared --graphics_device_id")
        args.sim_device = args.rl_device = "cuda:%d" % local_rank
        graphics = getattr(args, "graphics_device_ids", None)
        if graphics:
            ids = [int(item) for item in graphics.split(",")]
            if len(ids) != local_size or min(ids) < 0 or len(set(ids)) != len(ids):
                raise ValueError("--graphics_device_ids requires one distinct nonnegative Vulkan index per local rank")
            args.graphics_device_id = ids[local_rank]
        else:
            args.graphics_device_id = local_rank
    elif getattr(args, "graphics_device_ids", None):
        raise ValueError("--graphics_device_ids is only used with multi-process torchrun")


def initialize(args):
    """Called after importing Isaac Gym; also select the current GPU for Gym interop."""
    import torch
    import torch.distributed as dist
    if args.sim_device.startswith("cuda"):
        device = torch.device(args.sim_device)
        torch.cuda.set_device(device)
    if launch_world_size() > 1:
        if not torch.cuda.is_available():
            raise ValueError("Isaac Gym distributed training requires CUDA")
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))


def shutdown():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    import torch.distributed as dist
    if active():
        dist.barrier()


def mean(tensor):
    import torch.distributed as dist
    result = tensor.detach().clone()
    if active():
        dist.all_reduce(result)
        result /= world_size()
    return result


def moments(samples):
    """Global population moments; samples are [N,D] and counts may differ."""
    import torch
    import torch.distributed as dist
    samples = samples.to(dtype=torch.float64)
    count = torch.tensor(float(samples.shape[0]), device=samples.device, dtype=torch.float64)
    total = samples.sum(0)
    if active():
        dist.all_reduce(count)
        dist.all_reduce(total)
    average = total / count
    # A centred second pass avoids cancellation for features with large means.
    squared = (samples - average).square().sum(0)
    if active():
        dist.all_reduce(squared)
    return average, squared / count, int(count.item())


def synchronize_agent(agent):
    import torch
    import torch.distributed as dist
    if not active():
        return
    # Optimizers are either new or loaded identically on every rank. Models
    # and scaler buffers may have consumed different initialization RNGs.
    for name in ("policy", "value", "state_preprocessor", "value_preprocessor"):
        with torch.no_grad():
            for tensor in agent.checkpoint_modules[name].state_dict().values():
                dist.broadcast(tensor, src=0)


def aggregate_evaluation(metrics, perception_counts):
    """Combine completed-episode metrics with their actual denominators."""
    import torch.distributed as dist
    if not active():
        return metrics
    gathered = [None] * world_size()
    dist.all_gather_object(gathered, (metrics, perception_counts))
    results = [item[0] for item in gathered]
    combined = dict(metrics)
    for name in ("transitions", "completed_episodes", "successes"):
        combined[name] = (sum(item[name] for item in results)
                          if all(item[name] is not None for item in results) else None)

    def weighted(field, denominator):
        values = [(item[field], item[denominator]) for item in results
                  if item[field] is not None and item[denominator]]
        count = sum(weight for _, weight in values)
        return sum(value * weight for value, weight in values) / count if count else None

    for field, denominator in (("gsr", "completed_episodes"), ("ossr", "completed_episodes"),
                               ("mean_episode_return", "completed_episodes"),
                               ("mean_reward", "transitions"), ("tsc", "successes")):
        combined[field] = weighted(field, denominator)
    keys = sorted(set().union(*(item[1] for item in gathered)))
    combined["perception"] = {}
    for key in keys:
        count = sum(counts.get(key, 0) for _, counts in gathered)
        if count:
            combined["perception"][key] = sum(
                result["perception"].get(key, 0.) * counts.get(key, 0)
                for result, counts in gathered) / count
    ranges = [item["initial_base_x_range"] for item in results if item["initial_base_x_range"] is not None]
    combined["initial_base_x_range"] = ([min(item[0] for item in ranges), max(item[1] for item in ranges)]
                                       if ranges else None)
    combined["world_size"] = world_size()
    return combined
