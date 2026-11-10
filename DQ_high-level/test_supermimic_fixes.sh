#!/bin/bash
# Test SuperMimic fixes before full training

set -e

echo "========================================="
echo "  Testing SuperMimic Fixes"
echo "========================================="

cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level

# Test 1: Python imports
echo ""
echo "Test 1: Checking Python imports..."
python3 << 'EOF'
import sys
import torch

# Test basic imports
try:
    from supermimic_v2 import DQRobotMDP, StateCapture, Context, SuperMimicAgentV2
    print("✓ All imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Test Policy accepts deterministic parameter
try:
    from supermimic_v2.networks_v2 import Policy
    policy = Policy(100, 10, (256, 128))
    obs = torch.randn(4, 100)
    action = policy(obs, deterministic=True)
    print("✓ Policy accepts deterministic parameter")
except Exception as e:
    print(f"❌ Policy test failed: {e}")
    sys.exit(1)

# Test agent instantiation
try:
    class MockEnv:
        num_envs = 10
        num_actions = 9
        num_obs = 1284
        num_gripper_joints = 2
        ee_pos = torch.zeros(10, 3)
        ee_orn = torch.zeros(10, 4)
        dof_limits_lower = torch.zeros(20)
        dof_limits_upper = torch.ones(20)

    env = MockEnv()
    mdp = DQRobotMDP(env, device='cpu')
    agent = SuperMimicAgentV2(mdp=mdp, device='cpu', warmup_iterations=10)
    print("✓ Agent instantiation successful")
    print(f"  State dim: {mdp.state_dim}, Action dim: {mdp.action_dim}")
except Exception as e:
    print(f"❌ Agent instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✓ All import and instantiation tests passed!")
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Tests failed! Please check the errors above."
    exit 1
fi

# Test 2: Check file structure
echo ""
echo "Test 2: Checking file structure..."
files_ok=true

if [ ! -f "supermimic_v2/agent_v2_FIXED.py" ]; then
    echo "❌ supermimic_v2/agent_v2_FIXED.py not found"
    files_ok=false
fi

if [ ! -f "supermimic_v2/mdp_FIXED.py" ]; then
    echo "❌ supermimic_v2/mdp_FIXED.py not found"
    files_ok=false
fi

if [ ! -f "simple_supermimic_trainer_FIXED.py" ]; then
    echo "❌ simple_supermimic_trainer_FIXED.py not found"
    files_ok=false
fi

if [ "$files_ok" = true ]; then
    echo "✓ All fixed files present"
fi

# Test 3: Check if fixes have been applied
echo ""
echo "Test 3: Checking if fixes are applied..."

if grep -q "agent_v2_FIXED" supermimic_v2/agent_v2.py 2>/dev/null; then
    echo "⚠️  Fixes NOT applied yet (agent_v2.py still original)"
    echo "   Run: bash apply_supermimic_fixes.sh"
    APPLIED=false
else
    if grep -q "Open-loop multi-step" supermimic_v2/agent_v2.py 2>/dev/null; then
        echo "✓ Fixes already applied to agent_v2.py"
        APPLIED=true
    else
        echo "⚠️  Cannot determine if fixes are applied"
        APPLIED=false
    fi
fi

echo ""
echo "========================================="
if [ "$APPLIED" = true ]; then
    echo "  ✅ All Tests Passed - Ready to Train!"
    echo "========================================="
    echo ""
    echo "Start training with:"
    echo "  USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh"
else
    echo "  ⚠️  Tests Passed But Fixes Not Applied"
    echo "========================================="
    echo ""
    echo "Apply fixes first:"
    echo "  bash apply_supermimic_fixes.sh"
    echo ""
    echo "Then start training:"
    echo "  USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh"
fi
echo ""
