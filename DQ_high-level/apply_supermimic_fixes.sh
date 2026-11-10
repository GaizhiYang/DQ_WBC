#!/bin/bash
# Apply SuperMimic fixes and run test

set -e

echo "========================================="
echo "  Applying SuperMimic Fixes"
echo "========================================="

cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level

# 1. Backup original files
echo "Step 1: Backing up original files..."
cp supermimic_v2/agent_v2.py supermimic_v2/agent_v2_ORIGINAL_BACKUP.py
cp supermimic_v2/mdp.py supermimic_v2/mdp_ORIGINAL_BACKUP.py
cp simple_supermimic_trainer.py simple_supermimic_trainer_ORIGINAL_BACKUP.py
echo "✓ Backups created"

# 2. Apply fixes
echo ""
echo "Step 2: Applying fixes..."
cp supermimic_v2/agent_v2_FIXED.py supermimic_v2/agent_v2.py
cp supermimic_v2/mdp_FIXED.py supermimic_v2/mdp.py
cp simple_supermimic_trainer_FIXED.py simple_supermimic_trainer.py
echo "✓ Fixes applied"

# 3. Show what changed
echo ""
echo "Step 3: Summary of changes:"
echo "-------------------------------------------"
echo "✅ agent_v2.py:"
echo "   - Open-loop World Model training (8-step rollout)"
echo "   - Fixed sample_windows() for temporal structure"
echo "   - Added sample_starts() for policy training"
echo "   - Better phase management"
echo ""
echo "✅ mdp.py:"
echo "   - w_distance: 1.0 → 2.0"
echo "   - w_approach: 0.5 → 3.0"
echo "   - w_alignment: 0.3 → 1.5"
echo "   - w_grasp: 2.0 → 4.0"
echo "   - w_height: 1.5 → 2.0"
echo ""
echo "✅ simple_supermimic_trainer.py:"
echo "   - Warmup phase: 4x WM updates/iter (first 1000 iters)"
echo "   - Joint phase: 2x WM + 1x Policy updates/iter"
echo "   - Better logging"
echo "-------------------------------------------"

# 4. Verify imports
echo ""
echo "Step 4: Verifying Python imports..."
python -c "
from supermimic_v2 import DQRobotMDP, StateCapture, Context, SuperMimicAgentV2
print('✓ All imports successful')
" || {
    echo "✗ Import failed! Check for syntax errors"
    exit 1
}

echo ""
echo "========================================="
echo "  ✅ Fixes Applied Successfully!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Quick test (100 iterations):"
echo "   USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh --no-wandb --stop_after 100"
echo ""
echo "2. Full training:"
echo "   USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh"
echo ""
echo "3. To revert changes:"
echo "   cp supermimic_v2/agent_v2_ORIGINAL_BACKUP.py supermimic_v2/agent_v2.py"
echo "   cp supermimic_v2/mdp_ORIGINAL_BACKUP.py supermimic_v2/mdp.py"
echo "   cp simple_supermimic_trainer_ORIGINAL_BACKUP.py simple_supermimic_trainer.py"
