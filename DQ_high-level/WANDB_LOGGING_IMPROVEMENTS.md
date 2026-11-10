# WandB日志改进说明

## 改进内容

### ✅ 1. 上传频率大幅提升
**之前**: 每10次迭代上传一次  
**现在**: **每次迭代都上传** - 实时监控训练进度

### ✅ 2. 新增重要指标

#### Episode统计（每次有episode完成时）
- `episode/reward_mean` - 平均episode奖励
- `episode/reward_std` - 奖励标准差
- `episode/reward_max` - 最大奖励
- `episode/reward_min` - 最小奖励
- `episode/length_mean` - 平均episode长度 ⭐ 关键指标
- `episode/length_std` - 长度标准差
- `episode/length_max` - 最长episode
- `episode/length_min` - 最短episode
- `episode/completed_this_window` - 窗口内完成的episode数
- `episode/avg_length` - 全局平均长度（总步数/总episode数）
- `episode/total_count` - 总episode数

#### 训练指标（每次迭代）
- `world_model_loss` - WM平均loss（多次更新的平均）
- `wm_valid_steps` - WM训练的有效步数
- `wm_grad_norm` - WM梯度范数
- `num_wm_updates_this_iter` - 本次迭代的WM更新次数
- `policy_loss` - Policy loss
- `critic_loss` - Critic loss
- `imagination_reward_mean` - 想象中的平均奖励
- `imagined_survival` - 想象中的存活率
- `ensemble_disagreement` - 模型分歧度
- `policy_grad_norm` - Policy梯度范数

#### Buffer状态
- `buffer/size` - Buffer中的样本数
- `buffer/utilization` - Buffer使用率

#### 训练阶段
- `phase` - "warmup" 或 "joint"
- `phase/is_warmup` - 是否在warmup（0或1）
- `phase/is_joint` - 是否在joint（0或1）
- `training/world_model_updates` - 累计WM更新次数
- `training/policy_updates` - 累计Policy更新次数

#### 其他
- `iteration` - 迭代次数
- `global_step` - 全局步数

### ✅ 3. 更好的指标组织

所有指标都使用了**分组前缀**：
- `episode/xxx` - Episode相关
- `buffer/xxx` - Replay buffer相关
- `training/xxx` - 训练步数相关
- `phase/xxx` - 训练阶段相关

这样在WandB界面中可以很方便地折叠/展开查看。

---

## 如何使用

### 查看训练曲线

1. **训练开始时**会打印WandB链接：
   ```
   [SuperMimic v2 FIXED] W&B run: https://wandb.ai/YOUR_USERNAME/isaacgym/runs/xxxxx
   ```

2. **点击链接**打开WandB界面

3. **关键曲线推荐监控**：

#### 必看曲线
- ✅ `episode/length_mean` - **最重要！**应该持续增长
- ✅ `world_model_loss` - 应该下降并稳定
- ✅ `policy_loss` - 在joint阶段应该出现并下降
- ✅ 成功率曲线（由SuperMimicMetrics提供）

#### 辅助曲线
- `episode/reward_mean` - 观察奖励趋势
- `wm_grad_norm` - 检查梯度是否正常
- `buffer/utilization` - Buffer是否填满
- `phase/is_warmup` vs `phase/is_joint` - 验证阶段切换

### 创建自定义Dashboard

在WandB界面：
1. 点击 "Add panel"
2. 选择你关心的指标
3. 推荐布局：

```
┌─────────────────────────────────────────────┐
│  Episode Length Mean (折线图)                │
├─────────────────────────────────────────────┤
│  WM Loss │ Policy Loss │ Success Rate       │
├──────────┼─────────────┼────────────────────┤
│  Episode Reward Mean                         │
├─────────────────────────────────────────────┤
│  Buffer Size │ Phase                         │
└─────────────────────────────────────────────┘
```

---

## 训练效果判断

### ✅ 训练正常的信号

**Warmup阶段（Iter 1-1000）**：
```
✓ world_model_loss: 0.12 → 0.08（下降）
✓ episode/length_mean: 10 → 15（增长）
✓ policy_loss: 0（还没开始训练）
✓ phase/is_warmup: 1
```

**Early Joint阶段（Iter 1000-3000）**：
```
✓ world_model_loss: 0.08 → 0.06（继续下降）
✓ episode/length_mean: 15 → 25（显著增长）
✓ policy_loss: 出现并从5.0降到4.0
✓ phase/is_joint: 1
✓ 成功率开始上升（1-5%）
```

**Mid Training（Iter 5000+）**：
```
✓ world_model_loss: ~0.05（稳定）
✓ episode/length_mean: 30-50（稳定在较高值）
✓ policy_loss: ~3.0（持续下降）
✓ 成功率: 10-20%
```

### ❌ 训练异常的信号

```
❌ episode/length_mean 不增长或下降
❌ world_model_loss 不下降或上升
❌ policy_loss = NaN 或爆炸
❌ wm_grad_norm > 100（梯度爆炸）
❌ episode/length_mean 始终 < 10步
```

---

## 对比原始训练

| 指标 | 原始训练 | 修复后预期 |
|------|---------|-----------|
| Episode平均长度 | ~5步 | 30-50步 |
| WM loss稳定值 | 0.10-0.12 | 0.05-0.06 |
| Full task成功率 @ 10k | 0% | 10-20% |
| 数据上传频率 | 每10次迭代 | **每次迭代** |
| 监控指标数量 | ~10个 | **30+个** |

---

## 实时监控命令

### 查看WandB链接
```bash
grep "W&B run:" DQ_teacher/b1-pick-sugar-box-sgs-ekf/supermimic_*/wandb/run-*/files/output.log
```

### 快速检查训练状态
```bash
tail -50 DQ_teacher/b1-pick-sugar-box-sgs-ekf/supermimic_*/wandb/run-*/files/output.log | grep "Iter.*JOINT\|Iter.*WARMUP"
```

---

## 高级功能

### 1. WandB Alerts（可选设置）

在WandB界面可以设置告警：
- Episode长度 < 10（训练可能失败）
- WM loss > 0.15（模型不收敛）
- Policy loss = NaN（训练崩溃）

### 2. 比较不同运行

在WandB的Workspace中：
1. 选择多个runs
2. 点击"Compare"
3. 并排查看曲线

### 3. 下载数据分析

```python
import wandb
api = wandb.Api()
run = api.run("YOUR_USERNAME/isaacgym/RUN_ID")
history = run.history()
print(history['episode/length_mean'])
```

---

## 应用改进

修复已经包含在 `simple_supermimic_trainer_FIXED.py` 中。

使用 `apply_supermimic_fixes.sh` 自动应用所有修复。

---

## 预期WandB界面

训练开始后，你将看到：

### Charts页面
- 实时更新的损失曲线
- Episode长度变化
- 成功率曲线

### System页面
- GPU使用率
- 内存占用
- 系统资源

### Logs页面
- 控制台输出
- 训练日志

### Overview页面
- 运行配置
- 超参数
- 运行状态

---

**总结**：现在你可以实时看到所有关键指标，上传频率从10次迭代降低到每次迭代，数据更详细，监控更方便！🚀
