#!/usr/bin/env bash
# Complete SGS + EKF + TSIL teacher, sharing the existing teacher launch defaults.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Usage: bash DQ_high-level/train_teacher_sgs_ekf_tsil.sh [--dry-run] [--no-wandb] [teacher arguments...]

Train the complete SGS+EKF+TSIL method: adaptive per-configuration time goals,
fastest-success/high-return memory, full-configuration positive-gap replay,
joint PPO/self-imitation, frozen observation/reward/value normalization.
Defaults: 8192 environments, 120000 steps, seed 43, EKF 0.60/0.90 s.

Supports the same environment overrides as train_teacher_sgs_ekf.sh, including
PYTHON_BIN, NUM_ENVS, TIMESTEPS, SGS_BANK, RL_DEVICE, EXPERIMENT_DIR, RUN_NAME.
Replay chunk overrides: --tsil_micro_batch_size 2048 --tsil_screen_batch_size 4096.
Chunk sizes do not reduce the replay candidate set or change optimizer steps.
Checkpoints contain temporal state for strict evaluation; training resume is unsupported.

Example:
  PYTHON_BIN=/home/hehui/miniconda3/envs/dqwbc/bin/python \
    bash DQ_high-level/train_teacher_sgs_ekf_tsil.sh --no-wandb
EOF
    exit 0
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-DQ_teacher/b1-pick-sugar-box-sgs-ekf-tsil}"
export RUN_NAME="${RUN_NAME:-sgs_ekf_tsil_sugar_box_seed${SEED:-43}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
exec bash "$script_dir/train_teacher_sgs_ekf.sh" --tsil "$@"
