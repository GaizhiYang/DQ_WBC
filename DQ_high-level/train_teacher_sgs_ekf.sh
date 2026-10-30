#!/usr/bin/env bash
# Run from any directory after activating the dqwbc environment.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash DQ_high-level/train_teacher_sgs_ekf.sh [--dry-run] [--no-wandb] [teacher arguments...]

Train the sugar_box teacher with both SGS and EKF future-pose observations.
Defaults: 16384 environments, 120000 steps, seed 43, CUDA 0, headless, W&B enabled.
EKF: horizons 0.60/0.90 seconds, history 5, Q scale 1.0, R scale 0.01.

Environment overrides:
  PYTHON_BIN       Python from the activated dqwbc environment (default: python)
  NUM_ENVS         Parallel environments (default: 16384)
  TIMESTEPS        Planned training steps (default: 120000)
  SEED             Random seed (default: 43)
  RL_DEVICE        Policy device (default: cuda:0)
  SIM_DEVICE       Simulation device (default: same as RL_DEVICE)
  SGS_BANK         Bank path, relative to DQ_high-level or absolute
                   (default: data/sgs/sugar_box_v10.pt)
  EXPERIMENT_DIR   Output directory, relative to DQ_high-level or absolute
                   (default: DQ_teacher/b1-pick-sugar-box-sgs-ekf)
  RUN_NAME         W&B/run directory name (default: timestamped name)
  WANDB_PROJECT    W&B project (default: isaacgym)

Examples:
  bash DQ_high-level/train_teacher_sgs_ekf.sh
  bash DQ_high-level/train_teacher_sgs_ekf.sh --dry-run
  NUM_ENVS=64 RUN_NAME=sgs_ekf_smoke bash DQ_high-level/train_teacher_sgs_ekf.sh --no-wandb --stop_after 48

Extra arguments are passed to train_multistate_DQ_teacher.py after the defaults.
SGS training starts from scratch; --resume and --checkpoint are not supported.
EOF
}

dry_run=false
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
bank_path="${SGS_BANK:-data/sgs/sugar_box_v10.pt}"
run_name="${RUN_NAME:-sgs_ekf_sugar_box_seed${seed}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$bank_path" ]]; then
    printf 'SGS bank not found: %s\nSet SGS_BANK to an existing validated bank.\n' "$bank_path" >&2
    exit 1
fi

cmd=(
    python train_multistate_DQ_teacher.py
    --cfg_file data/cfg/DQ_teacher.yaml
    --task B1Z1PickMulti --object_name sugar_box
    --rl_device "$rl_device" --sim_device "${SIM_DEVICE:-$rl_device}"
    --num_envs "${NUM_ENVS:-8192}" --timesteps "${TIMESTEPS:-120000}"
    --roboinfo --observe_gait_commands --small_value_set_zero
    --rand_control --no_velocity_obs
    --sgs_mode sgs --sgs_bank "$bank_path" --success_mode hold
    --future_pose_obs --future_pose_horizons 0.60 0.90
    --ekf_history 5 --ekf_q_scale 1.0 --ekf_r_scale 0.01
    --experiment_dir "${EXPERIMENT_DIR:-DQ_teacher/b1-pick-sugar-box-sgs-ekf}"
    --wandb_project "${WANDB_PROJECT:-isaacgym}" --wandb_name "$run_name"
    --seed "$seed" --headless --wandb
)

cmd+=("${teacher_args[@]}")

printf 'Working directory: %s\nCommand: ' "$script_dir"
printf '%q ' "${cmd[@]}"
printf '\n'
if "$dry_run"; then
    exit 0
fi
exec "${cmd[@]}"
