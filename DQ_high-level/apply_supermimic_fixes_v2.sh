#!/bin/bash
# Apply SuperMimic fixes - UPDATED VERSION

set -e

echo "========================================="
echo "  Applying SuperMimic Fixes v2"
echo "========================================="

cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level

# 1. Backup original files
echo "Step 1: Backing up original files..."
if [ ! -f "supermimic_v2/agent_v2_ORIGINAL_BACKUP.py" ]; then
    cp supermimic_v2/agent_v2.py supermimic_v2/agent_v2_ORIGINAL_BACKUP.py
    echo "✓ Backed up agent_v2.py"
else
    echo "✓ agent_v2.py already backed up"
fi

if [ ! -f "supermimic_v2/mdp_ORIGINAL_BACKUP.py" ]; then
    cp supermimic_v2/mdp.py supermimic_v2/mdp_ORIGINAL_BACKUP.py
    echo "✓ Backed up mdp.py"
else
    echo "✓ mdp.py already backed up"
fi

if [ ! -f "simple_supermimic_trainer_ORIGINAL_BACKUP.py" ]; then
    cp simple_supermimic_trainer.py simple_supermimic_trainer_ORIGINAL_BACKUP.py
    echo "✓ Backed up simple_supermimic_trainer.py"
else
    echo "✓ simple_supermimic_trainer.py already backed up"
fi

if [ ! -f "supermimic_v2/networks_v2_ORIGINAL_BACKUP.py" ]; then
    cp supermimic_v2/networks_v2.py supermimic_v2/networks_v2_ORIGINAL_BACKUP.py
    echo "✓ Backed up networks_v2.py"
else
    echo "✓ networks_v2.py already backed up"
fi

# 2. Apply fixes
echo ""
echo "Step 2: Applying fixes..."
cp supermimic_v2/agent_v2_FIXED.py supermimic_v2/agent_v2.py
echo "✓ Applied agent_v2_FIXED.py → agent_v2.py"

cp supermimic_v2/mdp_FIXED.py supermimic_v2/mdp.py
echo "✓ Applied mdp_FIXED.py → mdp.py"

cp simple_supermimic_trainer_FIXED.py simple_supermimic_trainer.py
echo "✓ Applied simple_supermimic_trainer_FIXED.py → simple_supermimic_trainer.py"

# Note: networks_v2.py was already fixed in place (Policy.forward deterministic parameter)
echo "✓ networks_v2.py already fixed (Policy.forward accepts deterministic)"

# 3. Test imports
echo ""
echo "Step 3: Testing imports..."
python3 << 'EOF'
import sys
try:
    from supermimic_v2 import DQRobotMDP, StateCapture, Context, SuperMimicAgentV2
    from supermimic_v2.networks_v2 import Policy
    import torch

    # Quick test
    policy = Policy(100, 10, (256, 128))
    obs = torch.randn(2, 100)
    action = policy(obs, deterministic=True)

    print('✓ All imports and tests successful')
except Exception as e:
    print(f'❌ Import/test failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Import test failed! Rolling back..."
    cp supermimic_v2/agent_v2_ORIGINAL_BACKUP.py supermimic_v2/agent_v2.py
    cp supermimic_v2/mdp_ORIGINAL_BACKUP.py supermimic_v2/mdp.py
    cp simple_supermimic_trainer_ORIGINAL_BACKUP.py simple_supermimic_trainer.py
    exit 1
fi

# 4. Show summary
echo ""
echo "========================================="
echo "  ✅ Fixes Applied Successfully!"
echo "========================================="
echo ""
echo "Changes applied:"
echo "-------------------------------------------"
echo "✅ agent_v2.py:"
echo "   - Open-loop World Model training (8-step rollout)"
echo "   - Fixed sample_windows() and sample_starts()"
echo "   - Better phase management (warmup/joint)"
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
echo "   - WandB logging every iteration (not every 10)"
echo "   - 30+ metrics tracked"
echo ""
echo "✅ networks_v2.py:"
echo "   - Policy.forward() accepts deterministic parameter"
echo "-------------------------------------------"
echo ""
echo "Next steps:"
echo "1. Quick test (10 min, no WandB):"
echo "   USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh --no-wandb --stop_after 100"
echo ""
echo "2. Full training (with WandB):"
echo "   USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh"
echo ""
echo "3. To revert changes:"
echo "   cp supermimic_v2/agent_v2_ORIGINAL_BACKUP.py supermimic_v2/agent_v2.py"
echo "   cp supermimic_v2/mdp_ORIGINAL_BACKUP.py supermimic_v2/mdp.py"
echo "   cp simple_supermimic_trainer_ORIGINAL_BACKUP.py simple_supermimic_trainer.py"
echo ""
