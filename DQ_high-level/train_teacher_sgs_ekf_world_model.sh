#!/usr/bin/env bash
# Real SGS/EKF/PPO with a GPU world model and validated short-horizon BPTT.
set -euo pipefail
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        cat <<'EOF'
Usage: bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh [--no-wandb] [teacher arguments...]

SGS + EKF + real PPO with a CUDA world model and short BPTT.
Default: hybrid mode; isolated action-effect probes every 2400 vector steps.
BPTT starts only after an exact model snapshot passes real action-effect and
held-out prediction checks. An unqualified model leaves real PPO running.

  --world_model off|fit|hybrid  Baseline / fit only / validated auxiliary BPTT
  --wm_probe_device cuda:1     Run isolated probes on another GPU
  --wm_probe_interval 0        Disable automatic probes (manual reports only)
  --wm_config PATH            Optional YAML overrides
  --dry-run                   Print the resolved training command

Environment: PYTHON_BIN, NUM_ENVS (4096), RL_DEVICE (cuda:0), SIM_DEVICE,
TIMESTEPS (120000), SEED (43), RUN_NAME, EXPERIMENT_DIR, SGS_BANK,
WORLD_MODEL_MODE (hybrid), WM_PROBE_INTERVAL (2400).

Details: doc/SuperMimic_SGS_EKF_implementation.md
EOF
        exit 0
    fi
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-DQ_teacher/b1-pick-sugar-box-sgs-ekf-world-model}"
export RUN_NAME="${RUN_NAME:-sgs_ekf_wm_seed${SEED:-43}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
exec bash "$script_dir/train_teacher_sgs_ekf.sh" \
    --world_model "${WORLD_MODEL_MODE:-hybrid}" \
    --wm_probe_interval "${WM_PROBE_INTERVAL:-2400}" "$@"
