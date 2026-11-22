"""Gate D1: does the teacher chase the object's current position or intercept it?

Run like eval_sgs_teacher.py (strict 2 m Level02 task, synchronized waves):

  python diagnose_pursuit.py --checkpoint DQ_teacher/<group>/<run>/checkpoints/agent_120000.pt \
      --task B1Z1PickMulti --object_name sugar_box --headless --rand_control \
      --eval_episodes 800 --eval_output outputs/d1_pursuit.json

Per episode, over the approach phase (from the start until the base is within
0.6 m of the object or the object is first lifted):
  los_rate         mean |d lambda / dt| (rad/s); lambda: bearing from base to object
  heading_error    mean |wrap(base yaw - lambda)| (rad)
  lead_reference   mean asin(v_perp / 0.45 m/s): the constant-bearing lead angle
  path_ratio       base path length / straight distance from its start to its end
Pure pursuit keeps heading_error ~ 0 while los_rate stays large; constant-bearing
interception keeps los_rate ~ 0 with heading_error ~ lead_reference.
"""

import json
import math
from pathlib import Path

APPROACH_END_M = 0.6
BASE_SPEED = 0.45


def wrap(angle):
    import torch
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def summarize(rows):
    import statistics
    keys = ("los_rate", "heading_error", "lead_reference", "path_ratio", "approach_seconds")
    def block(subset):
        out = {"episodes": len(subset)}
        if not subset:
            return out
        out["success_rate"] = sum(r["success"] for r in subset) / len(subset)
        for key in keys:
            values = [r[key] for r in subset if r[key] is not None and math.isfinite(r[key])]
            if values:
                out[key + "_mean"] = statistics.fmean(values)
                out[key + "_median"] = statistics.median(values)
        return out
    result = {"all": block(rows)}
    for motion in ("linear", "circle"):
        result[motion] = block([r for r in rows if r["motion"] == motion])
    for outcome, flag in (("successful", True), ("failed", False)):
        result[outcome] = block([r for r in rows if r["success"] == flag])
    return result


def diagnose(trainer, episodes):
    import torch

    env, agent = trainer.env, trainer.agents
    raw = env._env
    agent.set_mode("eval")
    agent.policy.deterministic = True
    for name in ("_state_preprocessor", "_value_preprocessor"):
        preprocessor = getattr(agent, name, None)
        if hasattr(preprocessor, "eval"):
            preprocessor.eval()
    device = raw.device
    rows, total, steps = [], 0, 0
    waves = (episodes + raw.num_envs - 1) // raw.num_envs
    seed = trainer.teacher_args.seed
    with torch.no_grad():
        for wave in range(waves):
            torch.manual_seed(seed + wave)
            raw.reset_buf.fill_(1)
            states, _ = env.reset()
            n = raw.num_envs
            active = torch.arange(n, device=device) < min(n, episodes - total)
            completed = ~active
            approaching = active.clone()
            base = raw._robot_root_states[:, :2].clone()
            start = base.clone()
            obj = raw._cube_root_states[:, :2]
            bearing = torch.atan2(obj[:, 1] - base[:, 1], obj[:, 0] - base[:, 0])
            sums = {name: torch.zeros(n, device=device, dtype=torch.float64)
                    for name in ("los", "heading", "lead", "path", "seconds", "count")}
            end = base.clone()
            motion = raw.lin_circle_select.clone()  # circle (True) or straight line
            for _ in range(raw.max_episode_length):
                actions = agent.act(states, timestep=steps, timesteps=trainer.timesteps)[0]
                actions[completed] = 0
                states, _, terminated, truncated, _ = env.step(actions)
                steps += 1
                dt = float(raw.control_freq_low * raw.dt)
                new_base = raw._robot_root_states[:, :2]
                obj = raw._cube_root_states[:, :2]
                offset = obj - new_base
                new_bearing = torch.atan2(offset[:, 1], offset[:, 0])
                yaw = torch.atan2(2 * (raw._robot_root_states[:, 6] * raw._robot_root_states[:, 5]
                                       + raw._robot_root_states[:, 3] * raw._robot_root_states[:, 4]),
                                  1 - 2 * (raw._robot_root_states[:, 4] ** 2 + raw._robot_root_states[:, 5] ** 2))
                unit = offset / offset.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                velocity = raw._cube_root_states[:, 7:9]
                perpendicular = (velocity[:, 0] * -unit[:, 1] + velocity[:, 1] * unit[:, 0]).abs()
                lead = torch.asin((perpendicular / BASE_SPEED).clamp(max=1.0))
                done = (terminated | truncated).reshape(-1).to(device)
                ending = (offset.norm(dim=-1) < APPROACH_END_M) | raw.lifted_object | done
                weight = (approaching & ~completed).double()
                sums["los"] += weight * wrap(new_bearing - bearing).abs().double()
                sums["heading"] += weight * wrap(yaw - new_bearing).abs().double() * dt
                sums["lead"] += weight * lead.double() * dt
                sums["path"] += weight * (new_base - base).norm(dim=-1).double()
                sums["seconds"] += weight * dt
                sums["count"] += weight
                end = torch.where(approaching[:, None] & ~completed[:, None], new_base, end)
                approaching &= ~ending
                base, bearing = new_base.clone(), new_bearing
                finished = done & ~completed
                if finished.any():
                    for i in finished.nonzero().flatten().tolist():
                        seconds = sums["seconds"][i].item()
                        straight = (end[i] - start[i]).norm().item()
                        rows.append({
                            "success": bool(raw.episode_success[i].item()),
                            "motion": "circle" if motion[i].item() else "linear",
                            "table_speed": abs(raw.vel_line_forward_y_level02[i].item()),
                            "approach_seconds": seconds,
                            "los_rate": sums["los"][i].item() / seconds if seconds else None,
                            "heading_error": sums["heading"][i].item() / seconds if seconds else None,
                            "lead_reference": sums["lead"][i].item() / seconds if seconds else None,
                            "path_ratio": sums["path"][i].item() / straight if straight > 1e-3 else None,
                        })
                    total += int(finished.sum().item())
                completed |= done
                if completed.all().item():
                    break
    return rows


def main():
    # Isaac Gym must be imported before torch; the teacher entry point does that.
    from train_multistate_DQ_teacher import get_trainer

    trainer = get_trainer(is_eval=True, strict_eval=True)
    args = trainer.teacher_args
    rows = diagnose(trainer, args.eval_episodes)
    result = {"gate": "D1", "checkpoint": str(Path(args.checkpoint).resolve()),
              "approach_end_m": APPROACH_END_M, "base_speed_reference": BASE_SPEED,
              "summary": summarize(rows),
              "reading": ("pursuit: heading_error << lead_reference with large los_rate; "
                          "interception: los_rate ~ 0, heading_error ~ lead_reference")}
    print(json.dumps(result, indent=2))
    if args.eval_output:
        output = Path(args.eval_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result["episodes"] = rows
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
