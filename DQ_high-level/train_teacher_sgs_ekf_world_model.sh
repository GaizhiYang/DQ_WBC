#!/usr/bin/env bash
# SGS/EKF collection and pure SuperMimic-manipulation BPTT on CUDA.
set -euo pipefail
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
        cat <<'EOF'
Usage: bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh [--no-wandb] [teacher arguments...]

SGS + EKF + CUDA world model + pure BPTT Actor (no PPO).
Each high action lasts 8 x 20 ms, with the frozen low policy in every model step.
Default: reference recipe, hybrid mode, 3 x 2048 SiLU dynamics, BF16 MLPs.
Fitting: 8-low-step real-action windows; physical increments and RunningNorm.
Imagination: 4 high decisions = 32 low steps = 0.64 s, logical batch 4096.
Actor: continuous gripper, differentiable all 9 action dimensions.
Real exploration: Gaussian latent with learnable std (PPO only updates std).
Critic: Imagined Critic with 256/128 ELU, full physics/task features.
Update budget: INCREASED to 2.0 updates per low step (was 0.4).
Schedule: collect 64 high steps; then alternate based on update budget.
WM budget: 2.0 * newly collected valid train LOW transitions per iteration.
Differentiable rewards: Straight-through predicates + continuous shaping.
CUDA 0 is the default. Model microbatches and imagination chunks limit activations.
The progress bar pauses while models update; console reports iteration progress.
Training scalars are committed every 24 high steps, before a model update block.
World model live / curves also update about every 10 seconds during model rounds,
plus block endpoints; their horizontal axis is world_model_update.

  --world_model off|fit|hybrid  Policy-only inference / fit only / full BPTT
  --wm_recipe reference|manipulation|supermimic|gated  Pure BPTT / legacy hybrid
  --wm_probe_device cuda:1     Run optional isolated probes on another GPU
  --wm_probe_interval 0        Disable optional diagnostic probes (default)
  --wm_config PATH            Optional YAML overrides
  --wm_window 8               Fitting window in LOW steps
  --wm_horizon 4              BPTT horizon in HIGH decisions (= 32 low steps)
  --wm_collection_warmup_high_steps 64
  --wm_collection_updates_per_low_step 2.0  # INCREASED from 0.4
  --wm_low_replay_capacity 1000
  --wm_imagination_chunk_size 256  Gradient accumulation, logical batch unchanged
  --wm_model_microbatch_size 2048
  --wm_batches_per_update 10      # Full 10 batches per round
  --wm_amp bf16               MLP-only mixed precision (kinematics/EKF stay FP32)
  --dry-run                   Print the resolved training command

Environment: PYTHON_BIN, NUM_ENVS (4096), RL_DEVICE (cuda:0), SIM_DEVICE,
TIMESTEPS (40000 high steps = 320000 low steps), SEED (43), RUN_NAME,
EXPERIMENT_DIR, SGS_BANK, WORLD_MODEL_MODE (hybrid), WM_PROBE_INTERVAL (0).
WANDB_MODE=offline keeps W&B logging local if online uploads fail;
--no-wandb disables W&B while retaining local training logs.

Details: doc/SuperMimic_correction_plan.md
EOF
        exit 0
    fi
done
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export RL_DEVICE="${RL_DEVICE:-cuda:0}"
export SIM_DEVICE="${SIM_DEVICE:-$RL_DEVICE}"
export EXPERIMENT_DIR="${EXPERIMENT_DIR:-DQ_teacher/b1-pick-sugar-box-sgs-ekf-world-model}"
export RUN_NAME="${RUN_NAME:-sgs_ekf_wm_seed${SEED:-43}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
export TIMESTEPS="${TIMESTEPS:-40000}"
# Use supermimic recipe by default (with increased update budget)
exec bash "$script_dir/train_teacher_sgs_ekf.sh" \
    --world_model "${WORLD_MODEL_MODE:-hybrid}" \
    --wm_recipe supermimic \
    --wm_probe_interval "${WM_PROBE_INTERVAL:-0}" "$@"
