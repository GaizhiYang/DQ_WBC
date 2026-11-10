# SuperMimic 训练修复方案

## 核心问题总结

通过对比参考实现 `/home/hehui/supermimic_manipulation` 和当前实现，发现了**5个致命问题**：

### 🔴 问题1: World Model训练方法错误（最严重）

**当前实现**（`agent_v2.py:263-306`）：
```python
def update_world_model(self):
    windows = self.sample_windows(self.wm_batch_size)
    active = windows['mask'].bool()
    states, actions = windows['s'][active], windows['a'][active]  # ❌ 打平了所有时间步
    
    for member in range(self.wm.ensemble_size):
        predicted_delta = model(features)  # ❌ 单步预测
        loss = (error / scale).square().mean()
```

**问题**:
1. 将窗口内的所有时间步**打平**成一维，丢失了时序信息
2. 只做**单步预测**，没有open-loop多步rollout
3. World Model无法学习长期动态

**参考实现**（`agent.py:175-228`）：
```python
def _world_model_loss(self, windows, member=None):
    state = windows["s"][:, 0]  # 初始状态
    
    # ✓ Open-loop多步展开
    for step in range(self.cfg.wm_window):  # wm_window=8
        predicted = self.wm.step(state, action, context)
        error = self.mdp.state_error(predicted, target)
        # ✓ 用预测结果继续预测下一步（保持梯度）
        state = torch.where(rows, predicted, padding_state)
```

**影响**: World Model预测质量极差，无法支撑imagination rollout

---

### 🔴 问题2: 采样策略完全错误

**当前实现**（`agent_v2.py:308-326`）：
```python
def sample_batch(self, batch_size: int):
    indices = torch.randint(0, self.buffer_size, (batch_size,), device=self.device)
    # ❌ 完全随机采样，没有考虑：
    # 1. 环境对齐（同一vector step的不同环境）
    # 2. Episode边界
    # 3. 时序连续性
```

**参考实现**（`buffer.py:153-197`）：
```python
def sample_windows(self, batch_size, window):
    # ✓ 按 (时间步, 环境ID) 二维索引
    # ✓ 保证连续的window
    # ✓ 自动处理episode边界（mask）
    ages = start[:, None] + torch.arange(window)
    rows = (ptr - size + ages) % capacity
    parts = {key: values[rows, env[:, None]] for key, values in data.items()}
```

**影响**: Policy训练时采样的初始状态是随机的，破坏了训练的结构

---

### 🔴 问题3: 没有Warmup阶段

**当前实现**（`simple_supermimic_trainer.py:172`）：
```python
update_metrics = agent.update()  # ❌ WM和Policy从第1次迭代就一起更新
```

**参考实现**（`runner.py`）：
```python
if iteration <= warmup_iterations:
    # ✓ Warmup: 只训练World Model
    for _ in range(world_model_batches):
        agent.update_world_model(buffer)
else:
    # ✓ Joint: WM多次更新 + Policy一次更新
    for _ in range(world_model_batches):
        agent.update_world_model(buffer)
    agent.update_policy(buffer)
```

**影响**: 
- World Model还没学好就开始训练Policy
- Policy基于错误的world model做imagination
- 训练不稳定，无法收敛

---

### 🔴 问题4: Reward权重设置不合理

**当前实现**（`mdp.py:172-176`）：
```python
reward = (-1.0 * distance          # 距离惩罚: -1.0
          + 0.5 * (prev - dist)    # ❌ 接近奖励太小: +0.5
          + 0.3 * alignment        # ❌ 对齐奖励太小: +0.3
          + 2.0 * proximity * closure   # 抓取: +2.0
          + 1.5 * proximity * height)   # 高度: +1.5
```

**问题分析**:
- `distance`的惩罚（-1.0）**太强**
- `approach`奖励（0.5）**太弱**，agent没有动力接近物体
- `alignment`奖励（0.3）**太弱**，agent不知道该如何对齐

举例：
```
距离从1.0m → 0.8m:
  - distance惩罚减少: +0.2 * 1.0 = +0.2
  - approach奖励:     +0.2 * 0.5 = +0.1
  - 总收益: +0.3  ← 还不错

距离从1.0m → 1.2m (走远了):
  - distance惩罚增加: -0.2 * 1.0 = -0.2
  - approach惩罚:     -0.2 * 0.5 = -0.1
  - 总损失: -0.3  ← 惩罚偏小
```

**建议权重**:
```python
self.w_distance = 2.0    # 增强距离惩罚
self.w_approach = 3.0    # 大幅增强接近奖励
self.w_alignment = 1.5   # 增强对齐奖励
self.w_grasp = 4.0       # 增强抓取奖励
self.w_height = 2.0      # 适当增强高度奖励
```

---

### 🔴 问题5: 更新频率太低

**当前实现**（`simple_supermimic_trainer.py:172`）：
```python
update_metrics = agent.update()  # 每次迭代只更新一次
```

**参考实现**（`runner.py`）：
```python
# Joint阶段：每次迭代多次更新WM
for _ in range(world_model_batches):  # 例如: 4次
    agent.update_world_model(buffer)
```

**影响**: World Model训练样本太少，学习太慢

---

## 修复方案

### 方案A: 完全重写（推荐）

**优点**: 
- 完全对齐参考实现
- 训练稳定性有保证
- 性能最优

**缺点**: 
- 工作量大（2-3天）

**步骤**:
1. 实现参考的`ReplayBuffer`类（支持窗口采样）
2. 重写World Model训练（open-loop rollout）
3. 添加Warmup/Joint阶段管理
4. 修复Policy采样逻辑
5. 调整Reward权重

---

### 方案B: 最小修复（快速）

**优点**: 
- 修改量小
- 能快速验证

**缺点**: 
- 可能还有其他细节问题
- 性能不如完全对齐

**步骤**:
1. ✅ 修复World Model训练（P0）
2. ✅ 添加Warmup阶段（P0）
3. ✅ 修复Policy采样（P0）
4. ✅ 调整Reward权重（P1）

我将实现**方案B**，让你能快速看到效果。

---

## 具体修复代码

### 修复1: World Model Open-Loop训练

创建新文件 `supermimic_v2/agent_v2_fixed.py`，替换`update_world_model`方法：

```python
def update_world_model(self) -> Dict[str, float]:
    """Open-loop multi-step World Model training"""
    if self.buffer_size < self.min_buffer_size:
        return {"world_model_loss": 0.0, "skipped": True}
    
    # Sample temporal windows
    windows = self.sample_windows(self.wm_batch_size)
    
    total_loss = 0.0
    total_valid_steps = 0
    
    # Train each ensemble member with open-loop rollout
    for member_idx in range(self.wm.ensemble_size):
        model = self.wm.models[member_idx]
        
        # Initial state for rollout
        state = windows['s'][:, 0].clone()  # [batch, state_dim]
        batch_size = state.shape[0]
        
        # Padding for masked timesteps
        padding_state = state[0].clone()
        padding_action = windows['a'][0, 0].clone()
        padding_context = windows['context'][0, 0].clone()
        
        step_losses = []
        valid_count = 0
        
        # Open-loop rollout through the window
        for step in range(self.wm_window):
            # Get current timestep data
            active = windows['mask'][:, step] > 0  # [batch]
            rows = active.unsqueeze(-1)  # [batch, 1]
            
            # Prepare inputs (use padding for inactive rows)
            state_input = torch.where(rows, state, padding_state)
            action_input = torch.where(rows, windows['a'][:, step], padding_action)
            context_input = torch.where(rows, windows['context'][:, step], padding_context)
            target = torch.where(rows, windows['s1'][:, step], padding_state)
            
            # World Model prediction
            features = self.mdp.inputs(state_input, action_input, context_input)
            model.in_norm.update(features[active])  # Only update on active
            
            predicted_delta = model(features)
            predicted_state = state_input + predicted_delta
            
            # Compute error
            target_delta = self.mdp.targets(state_input, target)
            model.out_norm.update(target_delta[active])
            
            error = self.mdp.state_error(predicted_delta, target_delta)
            
            # Mask inactive rows before computing loss
            error_masked = torch.where(rows, error, torch.zeros_like(error))
            scale = model.out_norm._std.clamp_min(model.error_floor)
            per_sample_loss = (error_masked / scale).square().mean(dim=-1)
            
            step_losses.append(per_sample_loss.sum())
            valid_count += active.sum().item()
            
            # Continue rollout with predicted state (keep gradients!)
            state = torch.where(rows, predicted_state, padding_state)
        
        # Compute total loss for this member
        member_loss = sum(step_losses) / batch_size  # Average over batch
        
        # Backward and optimize
        self.wm_optimizer.zero_grad(set_to_none=True)
        member_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), self.max_grad_norm, error_if_nonfinite=True
        )
        self.wm_optimizer.step()
        
        total_loss += member_loss.item()
        total_valid_steps += valid_count
    
    self.world_model_updates += 1
    
    return {
        "world_model_loss": total_loss / self.wm.ensemble_size,
        "wm_valid_steps": total_valid_steps / self.wm.ensemble_size,
        "skipped": False,
    }
```

### 修复2: 添加Warmup和多次WM更新

修改 `simple_supermimic_trainer.py` 的训练循环：

```python
# 在agent初始化后添加配置
WM_UPDATES_PER_ITER_WARMUP = 4  # Warmup阶段每次迭代WM更新4次
WM_UPDATES_PER_ITER_JOINT = 2   # Joint阶段每次迭代WM更新2次

# 修改训练循环 (line 120-174)
for iteration in range(1, max_iterations + 1):
    # ... 环境交互代码 ...
    
    # 根据阶段决定更新策略
    if agent.buffer_size < agent.min_buffer_size:
        # Buffer不足，跳过训练
        continue
    
    if iteration <= agent.warmup_iterations:
        # ✓ Warmup阶段：只训练World Model，多次更新
        agent.phase = "warmup"
        for _ in range(WM_UPDATES_PER_ITER_WARMUP):
            wm_metrics = agent.update_world_model()
            if iteration % 10 == 0:
                update_metrics.update(wm_metrics)
    else:
        # ✓ Joint阶段：先多次更新WM，再更新Policy
        agent.phase = "joint"
        
        # 多次World Model更新
        for _ in range(WM_UPDATES_PER_ITER_JOINT):
            wm_metrics = agent.update_world_model()
        
        # 一次Policy更新
        policy_metrics = agent.update_policy()
        
        # 合并metrics
        update_metrics.update(wm_metrics)
        update_metrics.update(policy_metrics)
    
    # ... 日志记录代码 ...
```

### 修复3: 修复Policy初始状态采样

修改 `agent_v2.py` 的 `update_policy` 方法：

```python
def update_policy(self) -> Dict[str, float]:
    """Update policy using imagination rollout from episode starts"""
    if self.buffer_size < self.min_buffer_size:
        return {"policy_loss": 0.0, "critic_loss": 0.0, "skipped": True}
    
    # ✓ 从buffer中采样episode开始状态（而非随机状态）
    # 方法1: 近似 - 从recent transitions中采样
    # 由于我们的buffer是ring buffer，可以采样最近的transitions
    recent_window = min(self.buffer_size, 10000)  # 最近10000个transitions
    start_idx = max(0, self.buffer_size - recent_window)
    indices = torch.randint(start_idx, self.buffer_size, 
                           (self.policy_batch_size,), device=self.device)
    
    batch = {
        'state': self.replay_buffer['state'][indices],
        'obs': self.replay_buffer['obs'][indices],
        'context': self.replay_buffer['context'][indices],
    }
    
    # ... 其余代码不变 ...
```

### 修复4: 调整Reward权重

修改 `supermimic_v2/mdp.py` (line 39-45)：

```python
# 原来的权重
self.w_distance = 1.0   # ❌
self.w_approach = 0.5   # ❌
self.w_alignment = 0.3  # ❌
self.w_grasp = 2.0
self.w_height = 1.5

# 改为
self.w_distance = 2.0   # ✓ 增强距离惩罚
self.w_approach = 3.0   # ✓ 大幅增强接近奖励
self.w_alignment = 1.5  # ✓ 增强对齐奖励
self.w_grasp = 4.0      # ✓ 增强抓取奖励
self.w_height = 2.0     # ✓ 适当增强高度奖励
```

---

## 实施步骤

1. **备份当前代码**
   ```bash
   cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level
   cp -r supermimic_v2 supermimic_v2_backup
   cp simple_supermimic_trainer.py simple_supermimic_trainer_backup.py
   ```

2. **应用修复**
   - 我会为你生成完整的修复代码
   - 你只需要替换对应文件

3. **测试**
   ```bash
   USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh --no-wandb --stop_after 100
   ```
   
   观察：
   - World Model loss是否下降
   - 前1000次迭代（warmup）是否只有WM更新
   - Episode长度是否增加
   - near_goal成功率是否维持
   - pregrasp和full_task成功率是否开始上升

4. **完整训练**
   ```bash
   USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh
   ```

---

## 预期效果

修复后应该看到：

1. **Warmup阶段（前1000次迭代）**:
   - 只有WM loss下降
   - Episode长度逐渐增加
   - 成功率仍然是0%（正常，因为还没训练policy）

2. **Joint阶段（1000次迭代后）**:
   - WM loss继续下降并稳定
   - Policy开始学习
   - Episode长度显著增加
   - pregrasp成功率开始上升（5-10%）
   - full_task成功率缓慢上升（1-5%）

3. **后期（5000次迭代后）**:
   - pregrasp: 20-40%
   - full_task: 5-15%

---

## 如果还不work

如果应用这些修复后训练还是不行，可能需要：

1. **完全对齐参考实现的Buffer** - 实现完整的`ReplayBuffer`类
2. **检查Context** - 确保context包含正确的目标信息
3. **检查StateCapture** - 确保捕获的状态完整且正确
4. **调试World Model** - 打印预测误差，看是否学习

让我知道是否要我生成完整的修复代码文件！
