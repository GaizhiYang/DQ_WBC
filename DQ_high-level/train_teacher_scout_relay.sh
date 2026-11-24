#!/usr/bin/env bash
# Run from any directory after activating the dqwbc environment.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash DQ_high-level/train_teacher_scout_relay.sh [--dry-run] [--no-wandb] [teacher arguments...]

Scout & Relay teacher (doc/SGS_SAPG_scout_relay_design.md) on top of the
SGS + EKF sugar_box teacher. The shared base is train_teacher_sgs_ekf.sh:
8192 environments, 120000 steps, EKF 0.60/0.90 s, critic time input,
normalized actions (T1). Only VARIANT changes between experiment arms.

VARIANT (design doc section 6.2; default F):
  A  PPO,  uniform sampling over the same bank
  B  PPO,  SGS (current method)
  C  SAPG (leader + 1 scout), leader SGS, scout plain SGS, no relay
  D  C + self-relay (leader's own rare successes become relay starts)
  E  C + scout relay
  F  SAPG, need-weighted scout curriculum, scout relay (full method)
  G  PPO + self-relay (single policy)

Environment overrides:
  VARIANT          A, B, C, D, E, F or G (default: F)
  PYTHON_BIN       Python from the activated dqwbc environment (default: python)
  NUM_ENVS         Parallel environments, all branches together (default: 8192)
  TIMESTEPS        Planned training steps (default: 120000)
  SEED             Random seed (default: 43)
  RL_DEVICE        Policy device (default: cuda:0)
  SIM_DEVICE       Simulation device (default: same as RL_DEVICE)
  SGS_BANK         Bank path, relative to DQ_high-level or absolute
                   (default: data/sgs/sugar_box_v12.pt)
  EXPERIMENT_DIR   Output directory (default: DQ_teacher/b1-pick-sugar-box-scout-relay)
  RUN_NAME         W&B/run directory name (default: variant + seed + timestamp)
  WANDB_PROJECT    W&B project (default: isaacgym)

Useful extra arguments (appended last, so they override the variant):
  --sapg_branches 3              leader + 2 scouts (leader keeps 50% of envs)
  --sapg_off_policy_coef 0       ablation: no IS channel
  --relay_sampling uniform       ablation: relay without SGS
  --relay_fractions 0.8          ablation: fixed relay point
  --relay_trigger failure        ablation: relay failed episodes
  --relay_alpha_max 0.15         relay share cap (0.3 by default; 1 = no cap)
  --relay_controller_state       ablation: also restore gait/history buffers
  --scout_target_success 0.3     scout SGS target

Examples:
  VARIANT=F bash DQ_high-level/train_teacher_scout_relay.sh
  VARIANT=C SEED=44 bash DQ_high-level/train_teacher_scout_relay.sh --dry-run
  VARIANT=F NUM_ENVS=64 RUN_NAME=relay_smoke bash DQ_high-level/train_teacher_scout_relay.sh --no-wandb --stop_after 48

Evaluate with eval_sgs_teacher.py as usual; it runs the leader (branch 0) only.
Training starts from scratch; --resume and --checkpoint are not supported.
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

variant="${VARIANT:-F}"
case "$variant" in
    A) variant_args=(--sgs_mode uniform) ;;
    B) variant_args=() ;;
    C) variant_args=(--sapg_branches 2 --scout_need_beta 0) ;;
    D) variant_args=(--sapg_branches 2 --scout_need_beta 0 --relay --relay_source leader) ;;
    E) variant_args=(--sapg_branches 2 --scout_need_beta 0 --relay --relay_source scouts) ;;
    F) variant_args=(--sapg_branches 2 --scout_need_beta 1 --relay --relay_source scouts) ;;
    G) variant_args=(--relay --relay_source leader) ;;
    *) printf 'Unknown VARIANT %q (expected A-G)\n' "$variant" >&2; exit 1 ;;
esac

python_bin="${PYTHON_BIN:-python}"
seed="${SEED:-43}"
rl_device="${RL_DEVICE:-cuda:0}"
bank_path="${SGS_BANK:-data/sgs/sugar_box_v12.pt}"
run_name="${RUN_NAME:-scout_relay_${variant}_sugar_box_seed${seed}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"

if [[ ! -f "$bank_path" ]]; then
    printf 'SGS bank not found: %s\nSet SGS_BANK to an existing validated bank.\n' "$bank_path" >&2
    exit 1
fi

cmd=(
    "$python_bin" train_multistate_DQ_teacher.py
    --cfg_file data/cfg/DQ_teacher.yaml
    --task B1Z1PickMulti --object_name sugar_box
    --rl_device "$rl_device" --sim_device "${SIM_DEVICE:-$rl_device}"
    --num_envs "${NUM_ENVS:-8192}" --timesteps "${TIMESTEPS:-120000}"
    --roboinfo --observe_gait_commands --small_value_set_zero
    --rand_control --no_velocity_obs
    --sgs_mode sgs --sgs_bank "$bank_path" --success_mode hold
    --future_pose_obs --future_pose_horizons 0.60 0.90
    --ekf_history 5 --ekf_q_scale 1.0 --ekf_r_scale 0.01
    --critic_time_obs
    --action_scaling normalized
    --experiment_dir "${EXPERIMENT_DIR:-DQ_teacher/b1-pick-sugar-box-scout-relay}"
    --wandb_project "${WANDB_PROJECT:-isaacgym}" --wandb_name "$run_name"
    --seed "$seed" --headless
)

if "$use_wandb"; then
    cmd+=(--wandb)
fi

cmd+=("${variant_args[@]}")
cmd+=("${teacher_args[@]}")

printf 'Variant: %s\nWorking directory: %s\nCommand: ' "$variant" "$script_dir"
printf '%q ' "${cmd[@]}"
printf '\n'
if "$dry_run"; then
    exit 0
fi
exec "${cmd[@]}"
