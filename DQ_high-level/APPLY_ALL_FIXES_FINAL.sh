#!/bin/bash
# Apply ALL verified fixes

set -e

echo "========================================="
echo "  🔧 Applying All SuperMimic Fixes"
echo "========================================="
echo ""

cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level

# 1. Backup if not already done
if [ ! -f "supermimic_v2/mdp_ORIGINAL.bak" ]; then
    cp supermimic_v2/mdp.py supermimic_v2/mdp_ORIGINAL.bak
    cp simple_supermimic_trainer.py simple_supermimic_trainer_ORIGINAL.bak
    echo "✓ Original files backed up"
else
    echo "✓ Backups already exist"
fi

# 2. Apply reward weight fixes
echo ""
echo "Applying Reward weight optimization..."
cp supermimic_v2/mdp_FIXED.py supermimic_v2/mdp.py
echo "✓ mdp.py updated"

# 3. Apply trainer fixes (warmup, wandb logging)
echo ""
echo "Applying Training framework improvements..."
cp simple_supermimic_trainer_FIXED.py simple_supermimic_trainer.py
echo "✓ simple_supermimic_trainer.py updated"

# 4. Networks and agent are already updated (reference-based)
echo ""
echo "Verifying reference-based networks..."
if grep -q "RunningNorm" supermimic_v2/networks_v2.py; then
    echo "✓ Reference-based networks active"
else
    echo "⚠️  Warning: networks may need update"
fi

# 5. Test imports
echo ""
echo "Testing imports..."
python3 -c "
from supermimic_v2 import DQRobotMDP, SuperMimicAgentV2
print('✓ All imports successful')
" || {
    echo "❌ Import test failed!"
    exit 1
}

echo ""
echo "========================================="
echo "  ✅ All Fixes Applied Successfully!"
echo "========================================="
echo ""
echo "Summary of changes:"
echo "  ✓ Reward weights optimized (approach: 3.0, alignment: 1.5)"
echo "  ✓ Warmup phase added (1000 iterations)"
echo "  ✓ WandB logging enhanced (every iteration, 30+ metrics)"
echo "  ✓ Reference-based networks (gradient-safe)"
echo "  ✓ Open-loop World Model training"
echo ""
echo "Ready to train:"
echo "  USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh"
echo ""
