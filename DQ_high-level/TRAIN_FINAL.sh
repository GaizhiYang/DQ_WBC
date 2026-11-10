#!/bin/bash
# Final training script with all fixes applied

set -e

echo "========================================="
echo "  🚀 SuperMimic Training - FINAL VERSION"
echo "========================================="
echo ""
echo "✅ Applied fixes:"
echo "  - Reference-based networks (gradient-safe)"
echo "  - Reward weight optimization"
echo "  - Training phase management"
echo "  - WandB real-time logging (30+ metrics)"
echo ""

cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level

# Apply all fixes
echo "Applying all fixes..."
cp supermimic_v2/mdp_FIXED.py supermimic_v2/mdp.py
cp simple_supermimic_trainer_FIXED.py simple_supermimic_trainer.py
echo "✓ All fixes applied"

echo ""
echo "Starting training..."
echo "Watch for WandB link in output!"
echo ""

USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh "$@"
