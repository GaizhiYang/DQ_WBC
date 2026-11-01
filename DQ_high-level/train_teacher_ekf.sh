#!/usr/bin/env bash
# Run from any directory after activating the dqwbc environment.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash DQ_high-level/train_teacher_ekf.sh [--dry-run] [--no-wandb] [teacher arguments...]

Train the sugar_box EKF baseline with ordinary resets and the original PPO.
SGS and TSIL are explicitly disabled; no SGS bank is required.
Defaults: 8192 environments, 120000 steps, seed 43, CUDA 0, headless, W&B enabled.
EKF: horizons 0.60/0.90 seconds, history 5, Q scale 1.0, R scale 0.01.

Environment overrides:
  PYTHON_BIN       Python from the activated dqwbc environment (default: python)
  NUM_ENVS         Parallel environments (default: 8192)
  TIMESTEPS        Planned training steps (default: 120000)
  SEED             Random seed (default: 43)
  RL_DEVICE        Policy device (default: cuda:0)
  SIM_DEVICE       Simulation device (default: same as RL_DEVICE)
  EXPERIMENT_DIR   Output directory, relative to DQ_high-level or absolute
                   (default: DQ_teacher/b1-pick-sugar-box-ekf)
  RUN_NAME         W&B/run directory name (default: timestamped name)
  WANDB_PROJECT    W&B project (default: isaacgym)

Examples:
  bash DQ_high-level/train_teacher_ekf.sh
  bash DQ_high-level/train_teacher_ekf.sh --dry-run
  NUM_ENVS=64 RUN_NAME=ekf_smoke bash DQ_high-level/train_teacher_ekf.sh --no-wandb --stop_after 48

Extra arguments override other teacher defaults. This entrypoint fixes
SGS off, TSIL off and EKF on, including when a custom YAML is supplied.
EOF
}

dry_run=false
use_wandb=true
teacher_args=()
for arg in "$@"; do
    case "$arg" in
        --help|-h) usage; exit 0 ;;
        --dry-run) dry_run=true ;;
        --no-wandb) use_wandb=false ;;
        *) teacher_args+=("$arg") ;;
    esac
done

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$script_dir"

python_bin="${PYTHON_BIN:-python}"
seed="${SEED:-43}"
rl_device="${RL_DEVICE:-cuda:0}"
run_name="${RUN_NAME:-ekf_sugar_box_seed${seed}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"

cmd=(
    "$python_bin" train_multistate_DQ_teacher.py
    --cfg_file data/cfg/DQ_teacher.yaml
    --task B1Z1PickMulti --object_name sugar_box
    --rl_device "$rl_device" --sim_device "${SIM_DEVICE:-$rl_device}"
    --num_envs "${NUM_ENVS:-8192}" --timesteps "${TIMESTEPS:-120000}"
    --roboinfo --observe_gait_commands --small_value_set_zero
    --rand_control --no_velocity_obs --success_mode hold
    --future_pose_horizons 0.60 0.90
    --ekf_history 5 --ekf_q_scale 1.0 --ekf_r_scale 0.01
    --experiment_dir "${EXPERIMENT_DIR:-DQ_teacher/b1-pick-sugar-box-ekf}"
    --wandb_project "${WANDB_PROJECT:-isaacgym}" --wandb_name "$run_name"
    --seed "$seed" --headless
)

if "$use_wandb"; then
    cmd+=(--wandb)
fi

cmd+=("${teacher_args[@]}")
# Keep the method explicit even if a custom config enables SGS or TSIL.
cmd+=(--sgs_mode disabled --no_tsil --future_pose_obs)

printf 'Working directory: %s\nCommand: ' "$script_dir"
printf '%q ' "${cmd[@]}"
printf '\n'
if "$dry_run"; then
    exit 0
fi
exec "${cmd[@]}"
