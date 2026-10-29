#!/usr/bin/env bash
# 727 server: expose physical GPU 2, which becomes cuda:0 inside Python.
# Reuse the shared launcher so EKF, SGS and training settings stay in sync.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

printf '727 server: CUDA_VISIBLE_DEVICES=2 (training uses logical cuda:0)\n'
exec env CUDA_VISIBLE_DEVICES=2 RL_DEVICE=cuda:0 SIM_DEVICE=cuda:0 \
    bash "$script_dir/train_teacher_sgs_ekf.sh" "$@"
