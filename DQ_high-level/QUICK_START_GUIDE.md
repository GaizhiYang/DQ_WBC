# SuperMimic 训练修复 + WandB日志改进 - 快速指南

## 🎯 改进内容

### 1. 核心训练修复
- ✅ World Model open-loop训练（8步rollout）
- ✅ 训练阶段分离（Warmup 1000次 → Joint）
- ✅ Reward权重优化（approach: 0.5→3.0, alignment: 0.3→1.5）
- ✅ 多次WM更新per迭代（Warmup: 4次, Joint: 2次）

### 2. WandB日志增强
- ✅ **上传频率**: 每10次迭代 → **每次迭代**
- ✅ **指标数量**: ~10个 → **30+个**
- ✅ **Episode统计**: 增加长度、奖励的均值/标准差/最大/最小
- ✅ **梯度监控**: WM和Policy的梯度范数
- ✅ **阶段追踪**: Warmup/Joint阶段标识

---

## 🚀 一键应用

```bash
cd /home/hehui/DQ_WBC_test/DQ_WBC/DQ_high-level
bash apply_supermimic_fixes.sh
```

---

## 📊 WandB关键指标

### 必看曲线（判断训练是否正常）

1. **`episode/length_mean`** ⭐ 最重要
   - 应该持续增长：10 → 15 → 25 → 40
   - 如果不增长 = 训练失败

2. **`world_model_loss`**
   - Warmup: 0.12 → 0.08
   - Joint: 0.08 → 0.05
   - 应该平滑下降

3. **`policy_loss`**
   - Warmup: 0（不训练）
   - Joint: 出现并从~5.0降到~3.0

4. **成功率**（由环境提供）
   - `full_task`: 0% → 1% → 5% → 10-20%
   - `pregrasp`: 0% → 5% → 20% → 30-40%
   - `near_goal`: 应该保持50-70%

### WandB Dashboard推荐布局

```
┌─────────────────────────────────┬─────────────────────────┐
│  Episode Length Mean (大图)      │  WM Loss                │
│                                 │  Policy Loss            │
├─────────────────────────────────┼─────────────────────────┤
│  Success Rate (full_task)        │  Episode Reward Mean    │
├─────────────────────────────────┴─────────────────────────┤
│  Phase (warmup/joint)   │  Buffer Size  │  Total Episodes │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 训练状态检查

### 正常训练的信号

#### Warmup阶段（前1000次迭代）
```
✓ [Iter 100] WARMUP | WM loss: 0.115 | Policy loss: 0.0000
✓ [Iter 500] WARMUP | WM loss: 0.090 | Policy loss: 0.0000
✓ [Iter 1000] WARMUP | WM loss: 0.075 | Policy loss: 0.0000
✓ episode/length_mean: 10 → 15
✓ phase/is_warmup: 1
```

#### Joint阶段（1000次后）
```
✓ [Iter 1100] JOINT | WM loss: 0.070 | Policy loss: 5.200
✓ [Iter 2000] JOINT | WM loss: 0.065 | Policy loss: 4.500
✓ [Iter 5000] JOINT | WM loss: 0.055 | Policy loss: 3.200
✓ episode/length_mean: 15 → 25 → 35
✓ phase/is_joint: 1
✓ full_task成功率开始上升
```

### ❌ 异常训练的信号

```
❌ episode/length_mean 不增长（停在5-10）
❌ world_model_loss 不下降或上升
❌ policy_loss = NaN
❌ wm_grad_norm > 100（梯度爆炸）
❌ 所有成功率持续0%
```

---

## 📈 预期效果对比

| 指标 | 原始训练 | 修复后 |
|------|---------|-------|
| **Iter 1000** | | |
| - Episode长度 | ~5步 | ~15步 |
| - WM loss | 0.12 | 0.08 |
| - Full task成功率 | 0% | 0%（正常，还没训练policy） |
| **Iter 3000** | | |
| - Episode长度 | ~5步 | ~25步 |
| - WM loss | 0.10 | 0.06 |
| - Full task成功率 | 0% | 1-5% |
| **Iter 10000** | | |
| - Episode长度 | ~5步 | 30-50步 |
| - WM loss | 0.10 | 0.05 |
| - Full task成功率 | 0% | 10-20% |

---

## 🎬 训练流程

### 1. 应用修复
```bash
bash apply_supermimic_fixes.sh
```

### 2. 快速测试（10分钟）
```bash
USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh --no-wandb --stop_after 100
```

**检查点**:
- 是否显示 "WARMUP"
- WM loss是否下降
- 无报错

### 3. 完整训练（有WandB）
```bash
USE_SUPERMIMIC=1 bash train_teacher_sgs_ekf.sh
```

**训练开始时会显示**:
```
[SuperMimic v2 FIXED] W&B run: https://wandb.ai/YOUR_USERNAME/isaacgym/runs/xxxxx
```

### 4. 监控训练

#### 方法1: WandB网页（推荐）
- 点击上面的链接
- 实时查看曲线

#### 方法2: 命令行
```bash
# 查看最新状态
tail -50 DQ_teacher/b1-pick-sugar-box-sgs-ekf/supermimic_*/wandb/run-*/files/output.log

# 过滤关键信息
tail -200 DQ_teacher/b1-pick-sugar-box-sgs-ekf/supermimic_*/wandb/run-*/files/output.log | grep "WARMUP\|JOINT"
```

---

## 🐛 故障排查

### 问题1: WandB链接没有显示
```bash
# 检查是否启用了wandb
grep "wandb" train_teacher_sgs_ekf.sh
# 应该看到 cmd+=(--wandb)
```

### 问题2: Episode长度不增长
**可能原因**:
1. Buffer还没填满（等到iter 50-100）
2. Reward权重还需要调整

**解决**:
```python
# 进一步增加reward权重
# 编辑 supermimic_v2/mdp.py
self.w_approach = 5.0  # 从3.0
self.w_grasp = 6.0     # 从4.0
```

### 问题3: WM loss不下降
**调试**:
```bash
# 查看是否在做多次WM更新
grep "num_wm_updates_this_iter" output.log

# Warmup应该是4，Joint应该是2
```

### 问题4: 数据没有上传到WandB
**检查**:
```bash
# 1. 网络连接
ping api.wandb.ai

# 2. WandB登录状态
wandb login

# 3. 查看wandb日志
cat DQ_teacher/b1-pick-sugar-box-sgs-ekf/supermimic_*/wandb/debug.log
```

---

## 📚 相关文档

- `WANDB_LOGGING_IMPROVEMENTS.md` - WandB改进详细说明
- `SUMMARY.md` - 完整问题诊断和修复总结
- `SUPERMIMIC_FIX_PLAN.md` - 详细修复方案
- `COMPLETE_ANALYSIS.md` - 深入技术分析

---

## ✅ 检查清单

训练开始前：
- [ ] 已执行 `apply_supermimic_fixes.sh`
- [ ] Python导入无错误
- [ ] WandB已登录（`wandb login`）

训练开始后（前100次迭代）：
- [ ] 看到WandB链接并能打开
- [ ] 显示 "WARMUP" 阶段
- [ ] WM loss开始下降
- [ ] Episode长度开始增长

训练中期（1000-3000次迭代）：
- [ ] 切换到 "JOINT" 阶段
- [ ] Policy loss出现
- [ ] Episode长度显著增长（>20步）
- [ ] 成功率开始上升

---

## 🎯 成功标准

**短期目标（Iter 3000）**:
- ✅ Episode长度 > 20步
- ✅ Full task成功率 > 1%
- ✅ WM loss < 0.07

**中期目标（Iter 10000）**:
- ✅ Episode长度 > 30步
- ✅ Full task成功率 > 10%
- ✅ WM loss稳定在~0.05

---

**现在开始训练吧！记得打开WandB链接实时监控！** 🚀
