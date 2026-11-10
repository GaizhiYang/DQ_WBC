# SuperMimic 完全重写计划 - 方案 B

## 📋 架构对齐清单

### 阶段 1: 核心 MDP 接口（最关键）
- [ ] `DQRobotMDP` - 完整的 MDP 接口类
  - [ ] `inputs()` - 世界模型输入特征（state + action + context）
  - [ ] `targets()` - 世界模型训练目标（delta）
  - [ ] `state_error()` - 特定于任务的误差度量
  - [ ] `critic_features()` - Critic 输入特征
  - [ ] `observe()` - 从物理状态生成策略观测
  - [ ] `reward()` - 可微分的代理奖励
  - [ ] `done()` - 可微分的终止条件
  - [ ] `advance_obs()` - 基于想象状态更新观测

### 阶段 2: 物理状态定义
- [ ] `DQStateAdapter` - 重写为真实物理状态
  - [ ] 机器人关节位置/速度 (14D)
  - [ ] 物体位姿/速度 (9D: pos 3 + six_rotation 6)
  - [ ] 夹爪状态 (2D)
  - [ ] 上一个动作 (7D)
  - [ ] 时钟 + 成功计数器 (2D)
  - [ ] **总计：~34D 物理状态**

### 阶段 3: 上下文信息
- [ ] `Context` - 环境不变量
  - [ ] 机器人基座位姿
  - [ ] 目标物体位姿
  - [ ] 重力
  - [ ] 物体质量/惯性
  - [ ] 摩擦参数

### 阶段 4: 网络重写
- [ ] `WorldModel` - 预测 delta + 输入/输出标准化
- [ ] `WorldModelEnsemble` - 5 个模型 + 不确定性估计
- [ ] `Policy` - 带观测标准化
- [ ] `Critic` - 带输入标准化

### 阶段 5: Agent 重写
- [ ] `SuperMimicAgent` - 完整的训练循环
  - [ ] 世界模型训练（with state_error）
  - [ ] 想象展开（with disagreement penalty）
  - [ ] 策略优化（with survival weighting）
  - [ ] Critic 训练（with TD(λ) + survival weighting）

### 阶段 6: 训练脚本重写
- [ ] `train_supermimic_teacher.py` - 完整的数据流
  - [ ] 正确的状态/上下文捕获
  - [ ] 集成到训练循环
  - [ ] 诊断日志

---

## 🚀 实现策略

### 核心原则
1. **完全对齐原始实现** - 不做简化
2. **保持 DQ_WBC 环境不变** - 通过适配器层对接
3. **渐进式验证** - 每个阶段都测试

### 关键差异总结

| 组件 | 当前实现 | 原始 SuperMimic | 影响 |
|------|----------|----------------|------|
| 状态 | 51D 观测特征 | 34D 物理状态 | **致命** |
| 世界模型目标 | 绝对状态 | Delta + 误差度量 | **致命** |
| 想象展开 | 简单奖励累积 | 不确定性惩罚 + survival | **严重** |
| 策略优化 | 简单目标 | Survival weighting | **严重** |
| Critic | 简单 TD | TD(λ) + survival | **严重** |
| 标准化 | 无 | 每个组件独立 | **重要** |
| MDP 接口 | 部分 | 完整 8 个方法 | **致命** |

---

## 📝 实现顺序

### Day 1: 核心基础（今天）
1. ✅ 问题分析完成
2. ⬜ 创建新的 `supermimic_v2/` 目录
3. ⬜ 实现 `mdp.py` - 完整的 MDP 接口
4. ⬜ 实现 `state.py` - 物理状态定义
5. ⬜ 实现 `context.py` - 上下文信息
6. ⬜ 测试：状态/上下文能否正确捕获

### Day 2: 网络和世界模型
7. ⬜ 实现 `networks_v2.py` - 带标准化的网络
8. ⬜ 实现 `world_model_v2.py` - Delta 预测 + 集成
9. ⬜ 测试：世界模型能否学习

### Day 3: Agent 和训练
10. ⬜ 实现 `agent_v2.py` - 完整的训练逻辑
11. ⬜ 实现 `train_supermimic_v2.py` - 数据流集成
12. ⬜ 全流程测试和调试

---

## 🎯 成功标准

### 短期（1-2 天）
- [ ] 世界模型 loss < 0.1（当前 ~0.5）
- [ ] 想象奖励 > 10（当前 ~1.3）
- [ ] near_goal 成功率保持 70%+

### 中期（3-5 天）
- [ ] 任务成功率 > 20%（当前 0%）
- [ ] Episode reward > 50（当前 ~29）

### 长期（1-2 周）
- [ ] 任务成功率 > 60%（对齐原始论文）
- [ ] 样本效率提升 30-50% vs PPO

---

## 📂 新文件结构

```
supermimic_v2/
├── __init__.py
├── mdp.py              # DQRobotMDP - 完整 MDP 接口
├── state.py            # 物理状态定义和捕获
├── context.py          # 上下文信息
├── networks_v2.py      # 带标准化的网络
├── world_model_v2.py   # Delta 预测 + 集成
├── agent_v2.py         # 完整的 SuperMimic agent
└── utils.py            # 工具函数

train_supermimic_v2.py  # 新的训练脚本
```

---

开始实现！
