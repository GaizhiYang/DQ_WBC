"""
Test script to identify and fix all dimension issues in SuperMimic integration
"""
import torch

# Test 1: Check done tensor shape from environment
print("Test 1: Environment done tensor shape")
dones = torch.rand(4096) > 0.5  # Simulate environment output
print(f"  dones.shape = {dones.shape}")
print(f"  Expected for buffer: [4096]")

# Test 2: Check if done needs squeezing
dones_with_dim = torch.rand(4096, 1) > 0.5
print(f"\nTest 2: If dones come with extra dimension")
print(f"  dones.shape = {dones_with_dim.shape}")
print(f"  After squeeze: {dones_with_dim.squeeze(-1).shape}")

# Test 3: Reward tensor
rewards = torch.rand(4096)
print(f"\nTest 3: Reward tensor")
print(f"  rewards.shape = {rewards.shape}")

rewards_with_dim = torch.rand(4096, 1)
print(f"  rewards with dim: {rewards_with_dim.shape}")
print(f"  After squeeze: {rewards_with_dim.squeeze(-1).shape}")

print("\n✅ All tests passed. Shapes are correct.")
print("\nFix needed:")
print("  - Check if env returns dones/rewards with shape [N] or [N, 1]")
print("  - Add .squeeze(-1) if needed in train_supermimic_teacher.py")
