#!/usr/bin/env bash
# 同时启用 SGS 和 EKF，从头训练 sugar_box 教师策略。
# 追加参数可覆盖数值、路径等默认值，例如：
#   ./train_teacher_sgs_ekf.sh --num_envs 256 --wandb_name sgs_ekf_run2
#   ./train_teacher_sgs_ekf.sh --wandb
# 仅查看命令，不启动训练：
#   ./train_teacher_sgs_ekf.sh --dry-run
# 当前 SGS 不支持 --resume / --checkpoint。

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# 非交互式 shell 也能自动激活项目环境。
if [[ "${CONDA_DEFAULT_ENV:-}" != "dqwbc" ]]; then
    conda_exe="${CONDA_EXE:-${HOME}/miniconda3/bin/conda}"
    if [[ ! -x "$conda_exe" ]]; then
        conda_exe="$(command -v conda || true)"
    fi
    if [[ -z "$conda_exe" || ! -x "$conda_exe" ]]; then
        printf '找不到 Conda，请先激活 dqwbc 环境，或设置 CONDA_EXE。\n' >&2
        exit 1
    fi
    source "$("$conda_exe" info --base)/etc/profile.d/conda.sh"
    conda activate dqwbc
fi

# Isaac Gym 需要找到当前环境中的 libpython3.8.so。
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
    shift
fi

train_command=(
    "$CONDA_PREFIX/bin/python" -u train_multistate_DQ_teacher.py
    --task B1Z1PickMulti
    --object_name sugar_box
    --rl_device cuda:0
    --sim_device cuda:0
    --num_envs 512
    --timesteps 120000
    --roboinfo
    --observe_gait_commands
    --small_value_set_zero
    --rand_control
    --no_velocity_obs
    --sgs_mode sgs
    --sgs_bank data/sgs/sugar_box_v10.pt
    --future_pose_obs
    --future_pose_horizons 0.30 0.60
    --ekf_history 5
    --ekf_q_scale 1.0
    --ekf_r_scale 0.01
    --experiment_dir experiments
    --wandb_name sgs_ekf_sugar_box_v10_seed43
    --seed 43
    --headless
    "$@"
)

printf '工作目录：%s\n' "$PWD"
printf '启动命令：'
printf '%q ' "${train_command[@]}"
printf '\n'
if [[ "$dry_run" == true ]]; then
    exit 0
fi

exec "${train_command[@]}"
