# SuperMimic训练问题诊断报告

## 问题概述
训练120000个vector steps后，full_task成功率始终保持0%，虽然near_goal能达到50-60%。

## 关键观察

### 1. Episode终止过快
- **5850次迭代产生1,244,639个episodes**
- 平均每次迭代 ≈ 213 episodes
- 4096个环境，每次迭代约 0.052 episodes/env
- **这意味着episodes非常短暂，agent无法完成任务**

### 2. 成功率分析
```
near_goal: 50-60% ✓
pregrasp: 0%      ✗
full_task: 0%     ✗
```
- Agent能够接近目标（near_goal），但：
  - 无法完成抓取（pregrasp）
  - 无法完成完整任务（full_task）

### 3. 日志显示的问题
```
SuccessWindow / completed: 1000
SuccessWindow / successes: 0
```
- 完成了1000个episodes但成功次数为0

## 根本原因分析

### 对比参考实现 vs 当前实现

| 方面 | 参考实现 (supermimic_manipulation) | 当前实现 | 问题 |
|------|-----------------------------------|----------|------|
| **更新频率** | 每次迭代：WM多次更新 + Policy更新 | 每次迭代：WM一次 + Policy一次 | ❌ WM训练不足 |
| **Replay Buffer** | 专门的`SuperMimicBuffer`类，支持窗口采样 | 简单的字典存储 | ❌ 采样效率低 |
| **World Model训练** | Open-loop预测，多步展开（wm_window=8） | 每个成员独立训练 | ❌ 预测质量差 |
| **归一化** | 输入/输出独立归一化（`in_norm`, `out_norm`） | 简单的running normalization | ❌ 特征尺度不匹配 |
| **Error Floor** | 每个维度不同的error floor | 全局统一 | ❌ 不同状态维度重要性不同 |
| **Critic训练** | 独立的TD(λ)训练循环，多个epochs | 直接从imagination中计算 | ❌ Value估计不准 |
| **Phase管理** | 明确的warmup/joint phase | 简单的iteration计数 | ❌ 训练不稳定 |

## 关键问题

### 问题1: World Model训练不充分
```python
# 当前实现 - agent_v2.py:263-306
def update_world_model(self) -> Dict[str, float]:
    windows = self.sample_windows(self.wm_batch_size)
    active = windows['mask'].bool()
    states, actions = windows['s'][active], windows['a'][active]
    # 每个member独立训练，但没有open-loop rollout
    for member in range(self.wm.ensemble_size):
        predicted_delta = model(features)
        loss = (error / scale).square().mean()
```

**参考实现 - agent.py:175-228**
```python
def _world_model_loss(self, windows, member=None):
    # Open-loop多步展开
    for step in range(self.cfg.wm_window):
        predicted = self.wm.step(state, action, context)
        error = self.mdp.state_error(predicted, target)
        # 保持梯度，继续用predicted预测下一步
        state = torch.where(rows, predicted, padding_state)
```

### 问题2: 采样策略错误
```python
# 当前实现 - agent_v2.py:308-327
def sample_batch(self, batch_size: int):
    indices = torch.randint(0, self.buffer_size, (batch_size,), device=self.device)
    return self.replay_buffer['state'][indices]
```

这是**随机采样**，破坏了时序结构！

**参考实现 - buffer.py**
- 支持`sample_windows()`: 连续的时间窗口
- 支持`sample_starts()`: episode开始状态
- 保持环境索引对齐

### 问题3: 没有分离WM训练和Policy训练阶段
```python
# 当前实现 - simple_supermimic_trainer.py:172
update_metrics = agent.update()  # WM和Policy一起更新
```

**参考实现 - runner.py**
```python
# Warmup阶段：只训练World Model
for _ in range(config.world_model_batches):
    agent.update_world_model(buffer)

# Joint阶段：WM多次 + Policy一次
if iteration > warmup:
    for _ in range(batches_per_iteration):
        agent.update_world_model(buffer)
    agent.update_policy(buffer)
```

### 问题4: Reward函数可能不合理
```python
# mdp.py:172-176
reward = (-self.w_distance * distance
          + self.w_approach * (previous_distance - distance)
          + self.w_alignment * alignment
          + self.w_grasp * proximity * closure
          + self.w_height * proximity * torch.tanh(10.0 * height_error))
```

权重设置：
- `w_distance = 1.0` - 距离惩罚
- `w_approach = 0.5` - 接近奖励（太小）
- `w_alignment = 0.3` - 对齐奖励（太小）
- `w_grasp = 2.0` - 抓取奖励
- `w_height = 1.5` - 高度奖励

**问题**: `w_approach`和`w_alignment`太小，agent没有足够动力去接近和对齐！

### 问题5: Episode太短
观察到平均每个episode只有几步，可能的原因：
1. Termination条件太严格：`distance > 2.0`
2. Agent随机探索导致快速失败
3. Reward signal不够强

## 改进方案

### 立即修复（Critical）

1. **修复World Model训练** - 实现open-loop rollout
2. **修复采样策略** - 使用窗口采样而非随机采样
3. **分离训练阶段** - Warmup只训练WM，之后才加入Policy
4. **调整Reward权重** - 增加approach和alignment的权重

### 次要优化

5. **实现正确的Critic训练** - 独立的TD(λ)训练
6. **改进归一化** - 使用参考实现的normalizer
7. **调整终止条件** - 放宽distance > 2.0的限制
8. **增加exploration** - 在warmup阶段使用更多随机action

## 推荐的修复优先级

### P0 (必须修复，影响训练核心逻辑)
1. World Model的open-loop训练
2. 窗口采样替代随机采样  
3. 分离Warmup和Joint阶段

### P1 (重要，影响训练效果)
4. Reward权重调整
5. Critic独立训练

### P2 (可选优化)
6. 归一化改进
7. 终止条件调整

## 下一步行动

我将依次实现这些修复，从P0开始。
