**固定 8 次低层闭环的世界模型：实现与验证**

> 后续更新：默认入口已采用单模型、独立想象 Critic 和 SuperMimic 参数配方。本文中的双快照／门控段落保留为旧版记录；最新配置与验证见 [对齐实现说明](SuperMimic_alignment_implementation.md)。

更新日期：2026-11-05。对应[当前方案](SuperMimic_SGS_EKF_fixed8_world_model_plan.md)。当前训练入口已使用本实现，旧宏步适配器保留供历史回归测试使用。

**已经接入的训练路径**

真实 PPO 每 160 ms 采样一次随机高层动作；每个动作固定执行 8 次低层决策，每次对应 4 次 5 ms 物理推进。开启世界模型时强制关闭随机 7／8 周期，原 SGS 基线关闭世界模型时不受影响。

模型训练使用真实记录的 20 ms 转移。模型输入包括预测物理状态的 heading 坐标表示、12 维实际腿部命令、持有的末端世界目标、夹爪关节目标和物理条件。网络只输出 **62 维物理残差＋4 维足接触 logits**，积分得到下一份 **64 维机器人＋物体状态**。接触分类前向转为原低层接口需要的 0／1 信号。

策略想象每个高层周期显式运行 8 次冻结 low_level。每次从预测状态生成 71 维当前观测，拼接 710 维历史，重新计算腿部反馈并推进模型。历史、步态和命令按真实规则维护；机械臂每个 20 ms 内部的 IK／位置控制及接触效果由转移模型学习，不另行假装运行四次真实物理。

末端位姿和速度来自 GPU 可微 URDF FK。实际 Isaac 根状态的位置与线速度分别对应 link origin 与 COM，FK 和基座积分已用真实根刚体 COM 修正。桌面按已知名义规则推进，不设置桌面输出、残差或损失。物体受到桌面支撑和摩擦的影响仍由条件动力学学习。

**状态维度的区别**

| 用途 | 当前维度／含义 |
|---|---|
| 学习的连续物理状态 | 64：基座 13、关节位置／速度 38、物体 13 |
| 物理残差输出 | 62：两个刚体的旋转各用 3 维增量 |
| 足接触辅助输出 | 4：分类，不作为连续残差积分 |
| 高层策略原始观测 | 1284，保持原接口 |
| 低层网络输入 | 781＝71 当前观测＋710 历史 |
| 完整展开快照 | 922，包含控制历史、任务记忆、EKF、名义桌面和目标缓存 |

**922 维是回放／想象需要携带的快照，不是神经网络预测的状态维度。** 其中保留的旧末端缓存仅供真实记录和诊断；运行时观测、奖励及关系损失使用 FK。710 维历史不进入动力学 MLP，不接受历史回归损失。两个新的缓存分别是持有的末端世界目标 7 维和夹爪目标 1 维。

**拟合、BPTT 与终止**

- 高层与微步使用独立 GPU 回放。高层回放保存想象起点及动作效果验证；微步回放保存实际 20 ms 执行命令，专门用于模型训练。
- 默认拟合窗口为 8 个微步；从真实起点出发自回归，后续不替换成真实状态或真实桌面轨迹。物理损失按基座、物体、腿、臂、夹爪分组，另有 FK 手物关系损失和足接触 BCE。
- 默认想象为 4 个高层决策，即 32 次模型调用、0.64 s。`--wm_horizon` 的单位始终是高层决策；`--wm_window` 的单位是模型微步。
- 真实 PPO 随机采样；BPTT 使用均值经执行映射后的动作，夹爪保持连续 opening。标准差只由 PPO 更新。
- BPTT 冻结世界模型、低层策略、Critic 参数和归一化统计，保留其输入梯度，包括低层历史和因果 EKF 的梯度。
- 成功、失败和超时分别按任务规则吸收，保留终止转移奖励，后续奖励和尾值清零。第一版没有混合成功／失败的 soft-done 头，也没有新增软失败辅助头。
- 保留原有 candidate／validated 两份模型及独立动作效果验证。生产 BPTT 仍须通过对应快照的验证；GPU 冒烟报告不激活此门控。新物理与关系误差门限默认改为 0.1，是待实验校准的保守起值，不是已证明适用于所有训练阶段的阈值。

**主要代码**

| 文件 | 职责 |
|---|---|
| [wm_fixed8.py](../DQ_high-level/utils/wm_fixed8.py) | 64 维物理状态、两层观测、显式低层反馈、名义桌面与高层边界更新 |
| [hierarchical.py](../DQ_high-level/utils/world_model/hierarchical.py) | 20 ms 动力学、足接触头、固定 8 次展开 |
| [kinematics.py](../DQ_high-level/utils/world_model/kinematics.py) | 纯 Torch 可微 FK 与速度运动学 |
| [b1z1_base.py](../DQ_high-level/envs/b1z1_base.py)／[wrapper.py](../DQ_high-level/utils/wrapper.py) | 真实固定周期、8 条微步采集、重置前终态和回合标签 |
| [wm_ppo.py](../DQ_high-level/utils/wm_ppo.py)／[trainer.py](../DQ_high-level/utils/world_model/trainer.py) | 双回放、模型拟合、可信快照与 PPO 衔接 |
| [imagination.py](../DQ_high-level/utils/world_model/imagination.py) | 均值动作 BPTT、硬终止吸收、KL 检查与回滚 |

旧宏步世界模型权重不能加载到本模型；新元数据包含 fixed8／64／62、低层权重和资产指纹，检查点不匹配会明确拒绝。不要把旧模型的动作验证报告用在新模型上。

**运行**

在仓库根目录，使用已有的 Isaac Gym 环境：

```bash
conda activate dqwbc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
RL_DEVICE=cuda:1 SIM_DEVICE=cuda:1 \
bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh --no-wandb
```

默认 `hybrid` 保留真实 PPO、模型拟合及通过验证后的 BPTT。只验证拟合可加 `--world_model fit`；`--wm_probe_interval 0` 关闭自动探测，但不会绕过 BPTT 的验证条件。主模拟器与学习器必须在同一 CUDA 设备。

小规模短训练命令：

```bash
NUM_ENVS=12 RL_DEVICE=cuda:1 SIM_DEVICE=cuda:1 \
bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh \
  --no-wandb --world_model fit --wm_probe_interval 0 \
  --wm_replay_envs 12 --wm_batch_size 32 --stop_after 48
```

`--stop_after` 保留原计划训练时长的课程配置，仅提前结束此次运行。未自动启动用户规模的长时间训练。

**已完成的实际 GPU 验证**

[完整 GPU 诊断报告](SuperMimic_fixed8_gpu_validation.json)使用 GPU 1、12 个真实 Isaac Gym 环境，检查采样、FK、观测、模型拟合和 BPTT：

- 12 个高层向量步对应 96 个低层向量步；每周期恰好 8 条转移。
- 高层观测最大绝对误差约 `1.87e-4`；低层观测／历史约 `5.96e-7`；低层实际动作约 `1.43e-6`。
- 末端 FK 位置最大误差约 `4.62e-6 m`，速度误差约 `2.13e-6 m/s`。
- 世界模型完成 8 次 Adam 更新；32 个模型子步的 BPTT 完成一次 Actor 更新，梯度有限，夹爪动作到物理后果及低层输入到反馈动作的梯度均非零。
- BPTT 中低层策略、世界模型、Critic、归一化统计和 `log_std` 均保持不变。
- 单个高层周期的桌面名义位置与真实读回偏差最大约 `0.247 mm`，仅作为近似诊断，不加入桌面监督损失。

另通过原训练启动脚本完成 48 个真实高层向量步：2 次 PPO 更新、384 条低层向量转移、4 次模型更新，退出码为 0，留出集验证能够运行，未出现非有限训练指标。该运行使用 `fit` 模式，没有启用未经验证模型的生产 BPTT。

回归共执行 574 项：572 项通过，2 项因未安装可选 Paparazzo 数据集跳过。真实独立探测的 5 个进程也已完成；刚经过 4 次模型更新的候选未通过动作效果检验，生产门控保持关闭。探测中起点、上下文、控制时长及重复基线奖励均一致。入口、回归和探测记录见[验证汇总](SuperMimic_fixed8_entry_validation.json)。

这些结果证明当前代码可在 GPU 上运行，且关键观测、时序和梯度路径成立。它们尚不证明长期模型精度、梯度方向正确性、完整规模吞吐或抓取成功率提升。

复查命令：

```bash
PYTHONPATH=DQ_high-level:DQ_high-level/tests CUDA_VISIBLE_DEVICES=1 \
python -m unittest discover -s DQ_high-level/tests -p 'test_world_model*.py' -q

python DQ_high-level/tests/smoke_fixed8_world_model_gpu.py --device cuda:1
```
