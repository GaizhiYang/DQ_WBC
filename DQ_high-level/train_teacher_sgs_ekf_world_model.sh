#!/usr/bin/env bash
# Real SGS/EKF/PPO with GPU dynamics and independent imagined Actor-Critic.
set -euo pipefail
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        cat <<'EOF'
Usage: bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh [--no-wandb] [teacher arguments...]

SGS + EKF + real PPO with a CUDA world model and short BPTT.
Each high action lasts 8 x 20 ms, with a frozen low policy in every model step.
Default model fitting window: 8 low steps; BPTT: 4 high decisions (0.64 s).
Default: SuperMimic recipe, hybrid mode, one current 2048 x 3 dynamics model.
Model: 10 x 2048 windows; imagination: 4096 starts, independent 256/128 critic.
Updates: 0.4 rounds per collected LOW vector step, after a 40-low-step warmup,
executed at PPO rollout boundaries. Chunking limits activation memory.
Probes are optional diagnostics; they do not gate the SuperMimic recipe.

  --world_model off|fit|hybrid  Baseline / fit only / auxiliary BPTT
  --wm_recipe supermimic|gated New recipe / older gated auxiliary recipe
  --wm_probe_device cuda:1     Run isolated probes on another GPU
  --wm_probe_interval 0        Disable optional diagnostic probes (default)
  --wm_config PATH            Optional YAML overrides
  --wm_window 8               Autoregressive fitting window in LOW steps
  --wm_horizon 4              BPTT horizon in HIGH decisions
  --wm_low_replay_capacity 1000 Number of low-step vector transitions in replay
  --wm_imagination_chunk_size 128  Gradient accumulation chunk, not logical batch
  --wm_model_microbatch_size 256   Model gradient accumulation chunk
  --dry-run                   Print the resolved training command

Environment: PYTHON_BIN, NUM_ENVS (4096), RL_DEVICE (cuda:0), SIM_DEVICE,
TIMESTEPS (40000 high steps = 320000 low steps), SEED (43), RUN_NAME,
EXPERIMENT_DIR, SGS_BANK, WORLD_MODEL_MODE (hybrid), WM_PROBE_INTERVAL (0).

Details: doc/SuperMimic_alignment_implementation.md
EOF
        exit 0
    fi
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-DQ_teacher/b1-pick-sugar-box-sgs-ekf-world-model}"
export RUN_NAME="${RUN_NAME:-sgs_ekf_wm_seed${SEED:-43}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
export TIMESTEPS="${TIMESTEPS:-40000}"
exec bash "$script_dir/train_teacher_sgs_ekf.sh" \
    --world_model "${WORLD_MODEL_MODE:-hybrid}" \
    --wm_recipe supermimic \
    --wm_probe_interval "${WM_PROBE_INTERVAL:-0}" "$@"
