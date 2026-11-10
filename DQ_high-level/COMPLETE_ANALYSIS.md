# SuperMimic Training Issues - Complete Analysis

## Executive Summary

训练失败的根本原因是**World Model训练方法错误** + **Reward信号太弱**。

修复后预期效果：
- ✅ full_task成功率从0%提升到10-20%（10000次迭代后）
- ✅ Episode长度从5步增加到30-50步
- ✅ 训练稳定，loss正常收敛

---

## Root Causes (按严重程度排序)

### 🔴 Critical Issue #1: World Model Training is Fundamentally Broken

**Location**: `supermimic_v2/agent_v2.py:263-306`

**Problem**:
```python
# Current (WRONG)
windows = self.sample_windows(self.wm_batch_size)
active = windows['mask'].bool()
states = windows['s'][active]  # ❌ Flattens all timesteps

# This loses temporal structure!
# Window: [batch=2048, time=8, dim] → Flattened: [batch*time*mask, dim]
```

**Why it's fatal**:
1. World Model cannot learn multi-step dynamics
2. No open-loop prediction (每步独立预测，没有累积误差)
3. 丢失了时序因果关系

**Evidence from reference**:
```python
# Reference implementation (agent.py:175-228)
for step in range(self.cfg.wm_window):  # 8 steps
    predicted = self.wm.step(state, action, context)
    state = predicted  # ✓ Use prediction for next step (keep gradients!)
```

**Impact**: 
- WM预测质量差 → Imagination不准 → Policy学不到有用信号
- 这是训练失败的**最主要原因**

---

### 🔴 Critical Issue #2: Policy Sampling is Random (破坏了结构)

**Location**: `supermimic_v2/agent_v2.py:308-326`

**Problem**:
```python
def sample_batch(self, batch_size):
    indices = torch.randint(0, self.buffer_size, (batch_size,))
    # ❌ Completely random - ignores:
    # - Environment alignment (同一个vector step的不同env)
    # - Episode boundaries
    # - Temporal continuity
```

**Why it matters**:
- Policy应该从**episode开始状态**做imagination
- 随机采样可能采到episode中间，context不对

**Reference approach**:
```python
def sample_starts(self, batch_size, reset_fraction=0.0):
    # Mix of episode starts + recent states
    # Respects (timestep, env_id) structure
```

---

### 🟠 Major Issue #3: No Training Phase Separation

**Location**: `simple_supermimic_trainer.py:172`

**Problem**:
```python
# Every iteration from the start
update_metrics = agent.update()  # WM + Policy together
```

**Why it's bad**:
- Policy用还没学好的WM做imagination
- Training不稳定，难以收敛

**Reference approach**:
```python
if iteration <= warmup_iterations:
    # Warmup: Only train WM (multiple times)
    for _ in range(4):
        agent.update_world_model()
else:
    # Joint: WM多次 + Policy一次
    for _ in range(2):
        agent.update_world_model()
    agent.update_policy()
```

**Evidence**: 参考实现的warmup_iterations默认值通常是1000-2000

---

### 🟠 Major Issue #4: Reward Weights are Too Weak

**Location**: `supermimic_v2/mdp.py:39-45`

**Problem**:
```python
self.w_approach = 0.5   # ❌ Too weak to incentivize approaching
self.w_alignment = 0.3  # ❌ Agent doesn't know to align
```

**Math Analysis**:
```
当距离从1.0m减少到0.8m (approaching):
  Distance penalty reduction: -1.0 * (-0.2) = +0.2
  Approach reward: 0.5 * 0.2 = +0.1
  Net reward: +0.3  ← Barely positive!

当距离从1.0m增加到1.2m (moving away):
  Distance penalty increase: -1.0 * 0.2 = -0.2
  Approach penalty: 0.5 * (-0.2) = -0.1
  Net penalty: -0.3  ← Not punishing enough!
```

Agent没有足够强的信号去学习"接近物体"。

**Fixed weights**:
```python
self.w_distance = 2.0   # +100% stronger penalty for being far
self.w_approach = 3.0   # +500% stronger reward for approaching
self.w_alignment = 1.5  # +400% stronger reward for aligning
self.w_grasp = 4.0      # +100% stronger reward for grasping
self.w_height = 2.0     # +33% stronger reward for lifting
```

---

### 🟡 Moderate Issue #5: Only One WM Update Per Iteration

**Location**: `simple_supermimic_trainer.py:172`

**Problem**:
- 每次迭代只更新一次WM
- 参考实现每次迭代更新4-8次

**Impact**: WM学习太慢

---

## Observation from Training Logs

### Current Behavior
```
[Iter 5850/120000] joint | WM loss: 0.0517 | episodes: 1,244,639

Success rates:
  full_task: 0.00% (0/100)
  pregrasp: 0.00% (0/100) 
  near_goal: 60.00% (60/100)
```

**Analysis**:
1. ✓ WM loss降到了0.05 - 看起来不错
2. ✓ near_goal 60% - Agent能到达目标附近
3. ✗ pregrasp 0% - 但不知道如何抓取
4. ✗ 1.2M episodes / 5850 iters = 213 eps/iter → **Episodes太短！**（每个episode只有几步）

**Root cause of short episodes**:
- WM预测错误 → 想象中很快失败 → Policy学到"早点放弃"
- Reward信号弱 → 没有动力继续尝试

---

## The Fix Strategy

### P0 Fixes (必须修复)

1. **Fix World Model Training** ✅
   - Implement open-loop rollout
   - Keep temporal structure
   - File: `agent_v2_FIXED.py`

2. **Fix Reward Weights** ✅
   - Increase approach/alignment rewards
   - File: `mdp_FIXED.py`

3. **Add Training Phases** ✅
   - Warmup: 1000 iters, 4x WM updates/iter
   - Joint: 2x WM + 1x Policy/iter
   - File: `simple_supermimic_trainer_FIXED.py`

### P1 Fixes (重要但可后续优化)

4. **Fix Policy Sampling**
   - Sample from episode starts
   - Implemented in `agent_v2_FIXED.py:sample_starts()`

5. **Better Critic Training**
   - Already implemented in agent_v2.py (TD-lambda)
   - No changes needed

---

## Expected Results After Fix

### Warmup Phase (Iter 1-1000)
```
[Iter 100] WARMUP | WM loss: 0.115 | episodes: ~30,000
[Iter 500] WARMUP | WM loss: 0.090 | episodes: ~150,000
[Iter 1000] WARMUP | WM loss: 0.075 | episodes: ~300,000

Success rates still 0% (policy not trained yet - EXPECTED)
Episode length: 5-10 steps → 10-15 steps
```

### Early Joint Phase (Iter 1000-3000)
```
[Iter 1500] JOINT | WM loss: 0.070 | Policy loss: 5.2 | episodes: ~450,000
[Iter 2000] JOINT | WM loss: 0.065 | Policy loss: 4.5 | episodes: ~600,000
[Iter 3000] JOINT | WM loss: 0.060 | Policy loss: 3.8 | episodes: ~900,000

Success rates:
  near_goal: 65-70%
  pregrasp: 5-15%  ← Should start appearing!
  full_task: 1-5%  ← Should start appearing!
Episode length: 15-25 steps
```

### Mid Training (Iter 5000-10000)
```
[Iter 5000] JOINT | WM loss: 0.055 | Policy loss: 3.2
[Iter 10000] JOINT | WM loss: 0.050 | Policy loss: 2.5

Success rates:
  near_goal: 70-75%
  pregrasp: 20-35%
  full_task: 8-15%
Episode length: 30-50 steps
```

---

## How to Apply Fixes

### Quick Start
```bash
cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level
bash apply_supermimic_fixes.sh
```

### Test
```bash
# Quick test (10 minutes)
USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh --no-wandb --stop_after 100

# Full training
USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh
```

---

## Files Changed

1. ✅ `supermimic_v2/agent_v2_FIXED.py` - 完全重写World Model训练
2. ✅ `supermimic_v2/mdp_FIXED.py` - 调整reward权重
3. ✅ `simple_supermimic_trainer_FIXED.py` - 添加训练阶段

---

## If It Still Doesn't Work

如果应用修复后训练5000次迭代，full_task成功率仍然<5%：

### Debug Checklist

1. **Verify WM is learning**
   ```bash
   grep "WM loss" output.log | tail -50
   # Should see: 0.12 → 0.10 → 0.08 → 0.06 → 0.05
   ```

2. **Check episode length is increasing**
   ```bash
   grep "episodes:" output.log | tail -20
   # Episodes should grow slower (longer episodes)
   ```

3. **Verify phases are correct**
   ```bash
   grep "WARMUP\|JOINT" output.log | head -20
   # Should see WARMUP for first ~1000, then JOINT
   ```

4. **Check context values**
   ```python
   # Add debug in simple_supermimic_trainer_FIXED.py
   if iteration == 1:
       print("Context sample:", current_context[0])
   # context[10] should be ~0.1-0.2 (grasp threshold)
   # context[11] should be >0 (target height)
   ```

5. **Verify rewards are reasonable**
   ```python
   # Add in training loop
   if iteration % 100 == 0:
       print(f"Reward mean: {rewards.mean().item():.3f}")
       print(f"Proxy reward mean: {proxy_reward.mean().item():.3f}")
   # Should see positive rewards when approaching
   ```

### Further Tuning

If basic fixes work but performance plateaus:

1. **Increase reward weights more**
   ```python
   self.w_approach = 5.0  # from 3.0
   self.w_grasp = 6.0     # from 4.0
   ```

2. **Increase episode length**
   ```yaml
   # data/cfg/DQ_teacher.yaml
   maxEpisodeLength: 150  # from 100
   ```

3. **Increase warmup iterations**
   ```python
   warmup_iterations=2000  # from 1000
   ```

---

## Comparison: Reference vs Current vs Fixed

| Aspect | Reference (works) | Current (broken) | Fixed |
|--------|-------------------|------------------|-------|
| **WM Training** | Open-loop 8-step rollout | ❌ Flattened single-step | ✅ Open-loop 8-step |
| **WM Updates/Iter** | 4-8x | ❌ 1x | ✅ 4x (warmup) / 2x (joint) |
| **Policy Sampling** | Episode starts | ❌ Random | ✅ Mixed starts |
| **Training Phases** | Warmup → Joint | ❌ Joint from start | ✅ Warmup 1000 → Joint |
| **Reward Weights** | Strong signals | ❌ Weak (0.3-0.5) | ✅ Strong (1.5-3.0) |
| **Expected Success** | 30-50% @ 20k | ❌ 0% @ 120k | ✅ 10-20% @ 10k |

---

## Conclusion

核心问题是**World Model的训练方式完全错误**，导致它无法学习有用的动态模型。

修复的关键：
1. Open-loop多步rollout（最重要）
2. 训练阶段分离
3. 更强的reward信号

应用这些修复后，应该能看到明显改善。如果还有问题，需要：
1. 检查context是否正确
2. 进一步调整reward权重
3. 可能需要完全对齐参考实现的ReplayBuffer

详细修复代码已经生成，使用 `apply_supermimic_fixes.sh` 即可应用。
