# SuperMimic 修正实施总结

**实施日期**: 2024-11-08  
**基于**: `/home/hehui/supermimic_manipulation` 参考实现  

---

## 已实施的关键修正

### ✅ P0: 训练预算增加 (recipe.py)

**问题**: 更新数据比 0.4 太低，模型训练不足
**修正**: 
- `updates_per_low_step`: 0.4 → **2.0**
- 每采集一个20ms低层步，执行2轮完整模型更新（10 batches × 2048 samples）
- 这匹配SuperMimic的训练强度

**文件**: `DQ_high-level/utils/world_model/recipe.py`

```python
"updates_per_low_step": 2.0,  # 从 0.4 增加到 2.0
"batches_per_update": 10,     # 保持完整10批次
```

---

### ✅ P0: 可微奖励适配器 (manipulation_reward.py)

**问题**: 硬成功判据没有梯度，BPTT无法学习
**修正**: 实现SuperMimic的 **straight-through estimators**

**新文件**: `DQ_high-level/utils/world_model/manipulation_reward.py`

**核心模式**（来自参考实现）:
```python
def straight_through(hard, soft):
    """前向：硬决策；反向：平滑梯度"""
    return hard.to(soft.dtype) + (soft - soft.detach())

def _smooth_gate(self, margin):
    """二值判据的可微代理"""
    hard = (margin >= 0).float()
    soft = torch.sigmoid(margin / self.smooth_temperature)
    return straight_through(hard, soft)
```

**奖励组件**（全部可微）:
1. **Reaching**: `1 - tanh(distance)` - 连续
2. **Lifting**: `height_progress * near_object` - 连续（门控detach）
3. **Placement**: `exp(-height_error)` - 连续
4. **Stability**: `smooth_gate(threshold - velocity)` - ST梯度
5. **Gripper**: `should_close * (1 - opening)` - 梯度通过opening
6. **Success**: 计数成功 + ST + 梯度缩放（0.02）
7. **Action costs**: 全部平方和，全可微

**关键**: 所有连续项提供梯度，二值判据使用ST保证平滑反向传播

---

### ✅ P1: 集成到适配器 (wm_fixed8.py)

**新增方法**:

```python
def enable_imagination_reward(self, config=None):
    """启用可微奖励进行BPTT"""
    from utils.world_model.manipulation_reward import ManipulationRewardMDP
    self.imagination_reward_mdp = ManipulationRewardMDP(self, config)
    return self.imagination_reward_mdp

def imagination_reward(self, previous_state, state, action, context):
    """计算想象转移的可微奖励
    
    在BPTT展开期间调用，提供平滑梯度。
    所有奖励组件通过预测状态保持可微性。
    """
    if self.imagination_reward_mdp is None:
        self.enable_imagination_reward()
    
    # 从预测状态计算几何
    geometry = self._geometry(state)
    outcome = self.task_outcome(previous_state, state, action, context, geometry=geometry)
    
    # 增强outcome添加可微几何信息
    outcome.update(...)
    
    # 计算可微奖励和终止
    return self.imagination_reward_mdp.reward(...)
```

**文件**: `DQ_high-level/utils/wm_fixed8.py`

---

### ✅ P1: 更新训练脚本 (train_teacher_sgs_ekf_world_model.sh)

**修改**:
- 默认使用 `supermimic` recipe（而非reference）
- 文档更新反映新的训练预算
- 移除过时的reference配置引用

**关键变化**:
```bash
--wm_recipe supermimic  # 使用增强的训练预算
--wm_horizon 4          # 4高层步 = 32低层步 = 0.64s
```

---

### ✅ 验证: 梯度测试 (test_manipulation_reward_gradients.py)

**新测试文件**: `DQ_high-level/tests/test_manipulation_reward_gradients.py`

**测试内容**:
1. ✓ Straight-through梯度平滑性
2. ✓ 奖励为动作提供梯度
3. ✓ 全部9个动作维度有梯度路径
4. ✓ 梯度数值健康（非零、有限）

**运行测试**:
```bash
cd DQ_high-level
python tests/test_manipulation_reward_gradients.py
```

---

## 关键设计决策

### 遵循SuperMimic原则

1. **显式奖励函数**（非学习头）
   - ✓ 所有奖励项都是状态的显式函数
   - ✓ 无神经网络奖励头

2. **Straight-through估计器**
   - ✓ 前向：精确硬决策（保持语义）
   - ✓ 反向：平滑sigmoid梯度（可优化）

3. **连续塑形主导**
   - ✓ 距离、高度、速度等连续项
   - ✓ 二值判据仅用于门控，不阻断梯度

4. **选择性梯度阻断**
   - ✓ 使用 `.detach()` 控制梯度流
   - ✓ 门控变量detach，被门控内容保持梯度

### 任务适配

保留的合理差异:
- ✓ 分层控制（高层/低层）
- ✓ 冻结低层策略
- ✓ 移动底座（腿部控制）
- ✓ 任务特定几何（高度阈值、距离、开度）

这些是**任务需要的适配**，非实现错误。

---

## 预期效果

### 短期 (1-2天)
- ✓ 世界模型损失应**稳定下降**
- ✓ Actor梯度范数应**非零且有限**
- ✓ 想象更新应**频繁运行**（不再被门控阻断）

### 中期 (1周)
- ✓ 想象回报应随训练**增加**
- ✓ 真实任务性能显示**渐进改善**
- ✓ 世界模型NSE达到 **< 0.05**

### 长期 (2-4周)
- ✓ 成功率超过 **仅PPO基线**
- ✓ 样本效率提升（更少环境步数收敛）
- ✓ 策略展现**预见性**行为（使用模型预测）

---

## 训练监控

### 关键指标

**世界模型**:
```python
wm_loss          # 应该稳定下降
nse_1step        # < 0.01 优秀, < 0.05 可接受
nse_kstep        # 略高于1-step但仍应 < 0.1
contact_accuracy # > 0.9
```

**BPTT**:
```python
actor_grad_norm       # 应该 > 1e-6, < 100
imagination_return    # 应该增加
imagination_updates   # 应该 >> rejected_updates
```

**真实任务**:
```python
success_rate     # 最终目标，应超过PPO基线
episode_length   # 成功时应合理
reward_mean      # 应增加
```

### 异常检测

**如果世界模型loss不下降**:
- 检查 `updates_per_low_step` 是否 = 2.0
- 检查 `batches_per_update` 是否 = 10
- 查看 `state_loss_reduction` = "sum"

**如果actor_grad_norm总是0**:
- 运行梯度测试: `python tests/test_manipulation_reward_gradients.py`
- 检查 `imagination_reward_mdp` 是否初始化
- 验证 `value_source` = "imagined"

**如果rejected_updates >> accepted_updates**:
- 门控太严格（但在supermimic recipe中已禁用）
- 检查是否误用了gated recipe

---

## 下一步

### 立即 (今天)
```bash
# 1. 运行梯度测试
cd DQ_high-level
python tests/test_manipulation_reward_gradients.py

# 2. 启动训练（小规模验证）
NUM_ENVS=512 TIMESTEPS=1000 \
bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh --no-wandb

# 3. 监控前100步
# - 世界模型损失应下降
# - actor_grad_norm 应非零
# - 无NaN/Inf
```

### 本周
```bash
# 完整训练运行
bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh

# 监控:
tensorboard --logdir logs/ --port 6006
```

### 下周
- 与PPO-only基线对比
- 消融研究（禁用BPTT，仅模型拟合）
- 调整奖励权重（如需要）

---

## 参考

- **参考实现**: `/home/hehui/supermimic_manipulation/src/clab/tasks/supermimic.py`
- **详细方案**: `doc/SuperMimic_correction_plan.md`
- **原始文档**: `doc/SuperMimic_alignment_implementation.md`
- **论文**: `/home/hehui/supermimic_manipulation/docs/supermimic.pdf`

---

## 检查清单

在运行训练前验证:

- [ ] `utils/world_model/recipe.py`: `updates_per_low_step` = 2.0
- [ ] `utils/world_model/manipulation_reward.py`: 文件存在
- [ ] `utils/wm_fixed8.py`: `imagination_reward` 方法存在
- [ ] `train_teacher_sgs_ekf_world_model.sh`: 使用 `supermimic` recipe
- [ ] 梯度测试通过: `python tests/test_manipulation_reward_gradients.py`

全部✓后，即可开始训练！

---

**最重要**: 这些修正解决的是**配置和梯度信号**问题，不是架构问题。你的基础设施是正确的，只是训练不够充分，梯度信号不够强。现在应该可以正常训练了。
