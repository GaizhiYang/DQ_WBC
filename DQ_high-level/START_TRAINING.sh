#!/bin/bash
# One-command training starter

echo "========================================="
echo "  🚀 Starting SuperMimic Training"
echo "========================================="
echo ""

cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level

# Check if fixes are applied
if ! grep -q "Open-loop multi-step" supermimic_v2/agent_v2.py 2>/dev/null; then
    echo "⚠️  Fixes not applied yet!"
    echo ""
    echo "Applying fixes now..."
    bash apply_supermimic_fixes_v2.sh
    echo ""
fi

# Run tests
echo "Running pre-flight checks..."
bash test_supermimic_fixes.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Pre-flight checks failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "  ✅ All Systems Go!"
echo "========================================="
echo ""
echo "Starting training with WandB..."
echo "Watch for the WandB link in the output!"
echo ""

# Start training
USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh

