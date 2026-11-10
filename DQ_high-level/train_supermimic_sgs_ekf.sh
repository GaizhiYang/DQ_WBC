#!/bin/bash

# SuperMimic Training Script for DQ_WBC
# Must run in conda environment with correct Python version

# Check if in correct conda environment
if [ "$CONDA_DEFAULT_ENV" != "dqwbc" ]; then
    echo "Error: Must activate dqwbc conda environment first!"
    echo "Run: conda activate dqwbc"
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Add Isaac Gym Python path
export PYTHONPATH=/home/hehui/DQ_WBC/third_party/isaacgym/python:$PYTHONPATH

# Training parameters
NUM_ENVS=4096
TIMESTEPS=120000  # Total timesteps (same as original)
RL_DEVICE="cuda:0"
SIM_DEVICE="cuda:0"

# SGS (State Generation System) configuration
SGS_BANK="data/sgs/bank_dq_teacher.pt"

# EKF (Extended Kalman Filter) configuration for dynamic prediction
# Use --future_pose_horizons for EKF prediction
EKF_HORIZON1=0.6  # First prediction horizon (seconds)
EKF_HORIZON2=0.9  # Second prediction horizon (seconds)

# Use conda environment's Python explicitly
PYTHON_CMD="python"

echo "=========================================="
echo "  SuperMimic Training for DQ_WBC"
echo "=========================================="
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "Num envs: $NUM_ENVS"
echo "Timesteps: $TIMESTEPS"
echo "=========================================="
echo ""

# Run SuperMimic training
$PYTHON_CMD train_supermimic_DQ.py \
    --task B1Z1PickMulti \
    --cfg_file data/cfg/DQ_teacher.yaml \
    --rl_device ${RL_DEVICE} \
    --sim_device ${SIM_DEVICE} \
    --graphics_device_id 0 \
    --headless \
    --timesteps ${TIMESTEPS} \
    --num_envs ${NUM_ENVS} \
    --sgs_mode sgs \
    --sgs_bank ${SGS_BANK} \
    --future_pose_obs \
    --future_pose_horizons ${EKF_HORIZON1} ${EKF_HORIZON2} \
    --roboinfo \
    --observe_gait_commands \
    --wandb \
    --seed 42

echo ""
echo "=========================================="
echo "  SuperMimic training completed!"
echo "=========================================="
