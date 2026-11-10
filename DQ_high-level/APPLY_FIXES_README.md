# SuperMimic 修复应用指南

## 快速开始

### 方法1: 使用自动脚本（推荐）

```bash
cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level
bash apply_supermimic_fixes.sh
```

这个脚本会：
1. 自动备份原始文件
2. 应用所有修复
3. 验证导入是否正常

### 方法2: 手动应用

```bash
cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level

# 备份
cp supermimic_v2/agent_v2.py supermimic_v2/agent_v2_ORIGINAL.py
cp supermimic_v2/mdp.py supermimic_v2/mdp_ORIGINAL.py
cp simple_supermimic_trainer.py simple_supermimic_trainer_ORIGINAL.py

# 应用修复
cp supermimic_v2/agent_v2_FIXED.py supermimic_v2/agent_v2.py
cp supermimic_v2/mdp_FIXED.py supermimic_v2/mdp.py
cp simple_supermimic_trainer_FIXED.py simple_supermimic_trainer.py
```

## 测试修复

### 快速测试（约10分钟）

```bash
USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh --no-wandb --stop_after 100
```

**期望看到的变化：**
1. 前几次迭代：`phase: warmup`
2. World Model loss开始下降
3. Episode长度逐渐增加（从~10步增加到20-30步）

### 完整训练

```bash
USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh
```

**训练阶段观察指标：**

#### Warmup阶段（Iter 1-1000）
```
[Iter 50/120000] WARMUP | WM loss: 0.1200 | Policy loss: 0.0000 | episodes: XXX
[Iter 100/120000] WARMUP | WM loss: 0.1150 | Policy loss: 0.0000 | episodes: XXX
...
[Iter 1000/120000] WARMUP | WM loss: 0.0800 | Policy loss: 0.0000 | episodes: XXX
```

✓ WM loss应该从~0.12降到~0.08
✓ Policy loss应该是0（还没开始训练）
✓ Episode数量增长较慢

#### Joint阶段（Iter 1001+）
```
[Iter 1050/120000] JOINT | WM loss: 0.0750 | Policy loss: 5.2000 | episodes: XXX
[Iter 2000/120000] JOINT | WM loss: 0.0600 | Policy loss: 4.8000 | episodes: XXX
[Iter 5000/120000] JOINT | WM loss: 0.0500 | Policy loss: 3.5000 | episodes: XXX
```

✓ WM loss继续下降并稳定在~0.05
✓ Policy loss出现并开始下降
✓ 成功率开始上升

#### 成功率预期

```
Iter 1000-3000:
  near_goal: 50-60% (保持)
  pregrasp: 0% → 5-10%
  full_task: 0% → 1-3%

Iter 3000-10000:
  near_goal: 60-70%
  pregrasp: 10-30%
  full_task: 3-10%

Iter 10000+:
  near_goal: 70-80%
  pregrasp: 30-50%
  full_task: 10-20%
```

## 监控训练

### 使用WandB
```bash
# 查看训练日志
tail -f DQ_teacher/b1-pick-sugar-box-sgs-ekf/supermimic_*/wandb/run-*/files/output.log

# 关键指标
grep "WARMUP\|JOINT" DQ_teacher/b1-pick-sugar-box-sgs-ekf/supermimic_*/wandb/run-*/files/output.log | tail -20
grep "SuccessRate" DQ_teacher/b1-pick-sugar-box-sgs-ekf/supermimic_*/wandb/run-*/files/output.log | tail -10
```

### 检查Episode长度
```python
# 在Python中快速查看
import re
with open('path/to/output.log') as f:
    for line in f:
        if 'episodes:' in line:
            # 提取episode数量，计算平均长度
            pass
```

## 故障排除

### 问题1: Import错误
```
ImportError: cannot import name 'SuperMimicAgentV2'
```

**解决**:
```bash
cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level
python -c "from supermimic_v2 import SuperMimicAgentV2; print('OK')"
```

### 问题2: WM loss不下降

**可能原因**:
1. Buffer还没填满（需要等min_buffer_size=10000）
2. 学习率太大/太小

**调试**:
```bash
# 查看buffer状态
grep "buffer_size" output.log | tail -5

# 查看WM loss趋势
grep "WM loss" output.log | tail -20
```

### 问题3: Episode仍然太短

**可能原因**:
1. Reward权重还需要调整
2. World Model预测不准
3. Policy学习太慢

**进一步调试**:
```python
# 增加reward权重
# 编辑 supermimic_v2/mdp.py
self.w_approach = 5.0  # 从3.0增加到5.0
self.w_grasp = 6.0     # 从4.0增加到6.0
```

### 问题4: 成功率还是0%

如果训练5000次迭代后full_task仍然是0%：

1. **检查Context**
```python
# 在simple_supermimic_trainer_FIXED.py中添加调试
print("Context sample:", current_context[0])
# 确保context[10]（threshold）和context[11]（target_height）有合理值
```

2. **检查Reward**
```python
# 打印实际reward值
print("Reward mean:", rewards.mean().item())
print("Proxy reward mean:", proxy_reward.mean().item())
```

3. **降低难度**
```bash
# 可以临时增加episode长度
# 编辑 data/cfg/DQ_teacher.yaml
maxEpisodeLength: 150  # 从100增加到150
```

## 恢复原始代码

如果修复后效果更差，可以恢复：

```bash
cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level

cp supermimic_v2/agent_v2_ORIGINAL_BACKUP.py supermimic_v2/agent_v2.py
cp supermimic_v2/mdp_ORIGINAL_BACKUP.py supermimic_v2/mdp.py
cp simple_supermimic_trainer_ORIGINAL_BACKUP.py simple_supermimic_trainer.py
```

## 核心修复总结

### 修复1: Open-Loop World Model训练
**文件**: `agent_v2.py` line 263-352

**改动**:
```python
# 之前：打平所有时间步，单步预测
active = windows['mask'].bool()
states = windows['s'][active]

# 之后：保持时序结构，多步rollout
for step in range(self.wm_window):
    predicted = self.wm.step(state, action, context)
    state = predicted  # 用预测继续预测
```

### 修复2: Reward权重调整
**文件**: `mdp.py` line 39-45

**改动**:
```python
# 之前
w_approach = 0.5  # 太小
w_alignment = 0.3  # 太小

# 之后
w_approach = 3.0  # 增加6倍
w_alignment = 1.5  # 增加5倍
```

### 修复3: 训练阶段分离
**文件**: `simple_supermimic_trainer.py` line 172-200

**改动**:
```python
# 之前：每次迭代WM和Policy一起更新
update_metrics = agent.update()

# 之后：分阶段更新
if iteration <= warmup_iterations:
    # Warmup: 只更新WM，4次/iter
    for _ in range(4):
        agent.update_world_model()
else:
    # Joint: 先2次WM，再1次Policy
    for _ in range(2):
        agent.update_world_model()
    agent.update_policy()
```

## 性能对比

| 指标 | 原始版本 | 修复版本（预期） |
|------|---------|----------------|
| Iter 5000 full_task成功率 | 0% | 5-10% |
| Iter 10000 full_task成功率 | 0% | 10-15% |
| Episode平均长度 | 5-10步 | 30-50步 |
| WM loss收敛值 | 0.10-0.12 | 0.05-0.06 |
| 训练稳定性 | 不稳定 | 稳定 |

## 下一步优化（可选）

如果基础修复work，可以进一步尝试：

1. **实现完整的ReplayBuffer类**（参考实现的buffer.py）
2. **添加Compile优化**（torch.compile）
3. **调整网络大小**（如果显存充足）
4. **实验不同的超参数**

详细计划见 `SUPERMIMIC_FIX_PLAN.md`
