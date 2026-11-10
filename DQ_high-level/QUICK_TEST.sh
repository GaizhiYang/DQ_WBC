#!/bin/bash
# Quick 100-iteration test before full training

echo "========================================="
echo "  🧪 Quick Test (100 iterations)"
echo "========================================="
echo ""
echo "This will run for ~5 minutes to verify everything works"
echo ""

cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level

USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh --no-wandb --stop_after 100

echo ""
echo "========================================="
echo "  ✅ Quick Test Complete!"
echo "========================================="
echo ""
echo "If you saw:"
echo "  - [Iter XX] WARMUP | WM loss: X.XXXX"
echo "  - No errors"
echo "  - WM loss decreasing"
echo ""
echo "Then you're ready for full training:"
echo "  USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh"
echo ""
