# SGS + EKF + TSIL 教师训练实现

已按[完整适配方案](TSIL_SGS_EKF_teacher_implementation_plan.md)接入同一条训练链路。新增入口为 [train_teacher_sgs_ekf_tsil.sh](../DQ_high-level/train_teacher_sgs_ekf_tsil.sh)，从训练开始同时启用时间条件、软时间奖励、双榜记忆、全配置正优势采样及联合优化。

## 直接运行

在已有 `dqwbc` 环境中，从仓库根目录执行：

```bash
conda activate dqwbc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
bash DQ_high-level/train_teacher_sgs_ekf_tsil.sh --no-wandb
```

环境数量沿用公共启动脚本的当前设置（4096），120000 个高层交互步、seed 43、CUDA 0、v10 bank；EKF 使用 0.60/0.90 秒、history=5、Q=1、R=0.01。省略 `--no-wandb` 可沿用原入口的 W&B 记录。`NUM_ENVS`、`RUN_NAME`、`SGS_BANK`、`EXPERIMENT_DIR` 等环境变量用法与原脚本一致。训练设备为 CUDA 时，TSIL 自动使用同一 GPU 完成轨迹记录、双榜维护、回报重标、筛选与采样，无须增加开关。

```bash
# 仅显示完整命令
bash DQ_high-level/train_teacher_sgs_ekf_tsil.sh --dry-run --no-wandb

# 显存中的 replay 微批大小；仍遍历相同的全量候选与抽样集合
bash DQ_high-level/train_teacher_sgs_ekf_tsil.sh --no-wandb \
  --tsil_micro_batch_size 2048 --tsil_screen_batch_size 4096
```

输出默认放在 `DQ_high-level/DQ_teacher/b1-pick-sugar-box-sgs-ekf-tsil/<run_name>`。原 `train_teacher_sgs_ekf.sh` 保持 SGS+EKF 模式；同时修正了该脚本原先未实际使用 `PYTHON_BIN`、`--no-wandb` 的问题及帮助中的环境数量。

## 算法与代码对应

| 代码 | 实现内容 |
|---|---|
| [b1z1_pickmulti.py](../DQ_high-level/envs/b1z1_pickmulti.py) | 每回合固定的配置目标、实际秒数、两维时间观测、重置身份与终止前信息；保持 EKF 因果历史 |
| [reward_vec_task.py](../DQ_high-level/envs/reward_vec_task.py)、[b1z1_base.py](../DQ_high-level/envs/b1z1_base.py) | `50 × success × [1+min(D/T,1)]`，负奖励保留，记录门控前 command 分量 |
| [tsil.py](../DQ_high-level/utils/tsil.py) | 时间目标与 5-NN 查询、设备通用的回放批次、用于对照的 CPU 归档实现 |
| [tsil_replay_gpu.py](../DQ_high-level/utils/tsil_replay_gpu.py) | GPU 完整回合缓存、每配置最快成功 Top-5／高回报 Top-5 联合存储、成功优先补缺、目标与回报重标、全配置正 gap 抽样 |
| [tsil_ppo.py](../DQ_high-level/utils/tsil_ppo.py) | PPO 与 `0.05 × (SI_actor + 0.05 × SI_value)` 同一步优化；在线／回放共享价值输出适配；PPO 值损失采用未裁剪与裁剪误差的较大者 |
| [tsil_normalization.py](../DQ_high-level/utils/tsil_normalization.py) | 时间通道豁免、物理观测快照、逐环境 raw reward trace、在线 returns 更新价值统计及输出仿射补偿、跨轮价值单位转换 |
| [tsil_checkpoint.py](../DQ_high-level/utils/tsil_checkpoint.py)、[教师入口](../DQ_high-level/train_multistate_DQ_teacher.py) | 配套模型／目标／描述符／归一化／EKF 配置保存与加载 |
| [eval_sgs_teacher.py](../DQ_high-level/eval_sgs_teacher.py) | 原始 2 m 完整任务、冻结目标和归一化、实际成功用时及失败惩罚时间指标 |

当前 1284 维教师输入增加为 1286 维。时间位于 `[-191:-189]`，前面的 13 维仍是 EKF，最后 189 维仍是 grasp180＋action9。记录的是环境处理后、RunningStandardScaler 前策略实际收到的观测；回放只替换目标时间，不重算 EKF。

持久状态按 `config_id` 归属，回合身份用 `(env_id, episode_id)`。完整轨迹可以跨越多个 24 步 rollout；成功、物理失败和 100 步预算结束均为有限任务终止，回放尾值为零。行政重置丢弃未完成片段，不增加 SGS 失败，也不进入回放。

每轮扫描所有有记忆的配置。每配置最多选 5 条轨迹，每轨迹按当前正 gap 不放回选最多 32 个 transition；整套索引只生成一次，5 个 epoch 复用。当前 6144 配置的理论上限为 983040 个 transition。没有额外候选数上限或 family 配额。每 epoch 保留 6 个 PPO 优化器更新；微批只改变内存占用。

轨迹数据驻留在训练 GPU。当前糖盒的固定 1024 维特征经逐次精确相等检查后共享存储，其余观测、动作与奖励分量保留 float32；没有对 EKF 量化。每配置预留 `2 × Top-K` 个共享轨迹槽，两个榜单只保存槽索引；同时进入两榜的轨迹不复制两份。仍在榜内的旧轨迹保留原槽，只复制新入榜的完整轨迹。

6144 配置、Top-K=5、最长 100 步时，双榜轨迹槽的有效载荷约 6.27 GiB；4096 环境的在途轨迹约 0.42 GiB，8192 环境约 0.84 GiB。满额 983040 条采样的压缩观测、动作、回报及元数据另需约 1.03 GiB。上述数值不含仿真、在线 rollout、网络、梯度、CUDA 缓存与筛选临时张量；实际峰值见运行验证。静态特征若无法共享，存储会按完整观测维度增长。

GPU 采样采用指数随机数竞赛：对每条轨迹的正 gap `w` 生成独立 `E ~ Exp(1)`，按 `E / w` 从小到大取最多 32 项，零 gap 不参与。这与按权重不放回采样具有相同分布。CPU 与 CUDA 的随机数流、并行运算舍入不同，不能要求同一 seed 产生逐项相同的训练轨迹。每轮仍只采样一次，并在所有 PPO epoch 中复用；没有削减配置覆盖、完整回合长度或回放样本预算。

SI 更新在第一个微批遍历中计算 Critic 梯度，并缓存 detached 的正 gap × 效率权重；在整个逻辑 minibatch 的正权重总体标准差确定后，第二遍只计算 Actor。每条样本因此只需一次 Critic 前向，保留共享标准差、权重上限和同一次联合 optimizer step。标准差使用 float64 累积；损失日志在 GPU 汇总后再传回 CPU，避免逐微批逐指标同步。

Python 调度、日志和显式调试导出仍在 CPU；训练热路径不再把整条轨迹或采样集合往返搬运。CPU `TemporalReplay` 保留为对照实现，以及纯 CPU 单元测试的后端。检查归档用的 `iter_selected`、`fastest`、`high_return` 导出接口会同步数据到 CPU，不应放进正常训练计时循环。

价值统计变化使用明确的坐标补偿。记 Critic 原网络输出为 \(f_\psi(o)\)，实际归一化预测为 \(v=a f_\psi(o)+b\)，本轮奖励缩放单位中的反归一化预测为 \(V=\sigma v+\mu\)。在线 GAE returns 更新 RMS 后，保持网络参数不动，同时更新：

\[
a'=a\frac{\sigma}{\sigma'},\qquad
b'=\frac{\sigma b+\mu-\mu'}{\sigma'}.
\]

这样，仅更新统计不会改变 V；转换到新坐标的缓存旧预测与当前预测在首次 optimizer step 前一致。`prediction_scale`、`prediction_bias` 是 checkpoint 中的非训练缓冲，不改网络参数和 Adam 动量。采集缓存、bootstrap、回放筛选、权重标准差、SI 和 PPO 损失都经过同一个可微适配入口。价值 scaler 使用同一 σ 的可逆仿射正／逆变换，不沿用观测 scaler 的 ±5 裁剪。

真实配置继续开启 `clip_predicted_values=True`、`value_clip=0.2`，在线 Critic 使用 \(c_V\operatorname{mean}\max[(v-y)^2,(v_{old}+\operatorname{clip}(v-v_{old},-0.2,0.2)-y)^2]\)。这保留作者的 PPO 值裁剪规则与 DQ 的外层损失系数，替换了原先只对硬裁剪预测求 MSE 的写法。奖励除数跨轮变化时，RMS 均值、方差和标准差数值下限分别按 q、q²、q 换算，适配系数 a、b 不变；即使方差接近零，预测也随奖励单位同步变换。

## 作者源码复用

依据 `/home/hehui/TSIL` 的 `137e3ab8d6f1eccf9cf6c611f957d106c2ff000c`：移植 `sampling.py` 的最快成功优先、按回报补缺及去重规则，`memory.py` 的双榜排序／回报定义，以及 `loss.py` 的正 gap、效率权重、正权重总体标准差与 cap。联合优化沿用 `updater.py` 的损失组合语义，PPO 值裁剪依据 `algo/utils.py::compute_value_loss`。输出仿射补偿属于保证 DQ 归一化时序正确性的适配，不宣称是论文额外提出的算法。

配置索引、GPU 存储与批量采样、逐环境 trace、完整配置正优势预筛及评估适配按本仓库重写。作者主路径中与论文不同的固定成功比例、每 minibatch 重抽和非负奖励裁剪没有引入。来源与修改说明见 [NOTICE](../third_party/tsil/NOTICE.md)，上游 Apache-2.0 许可证保存在 [LICENSE](../third_party/tsil/LICENSE)。

## 评估与 checkpoint

```bash
cd DQ_high-level
python eval_sgs_teacher.py \
  --checkpoint /path/to/run/checkpoints/agent_5000.pt \
  --num_envs 128 --eval_episodes 2000 --headless --seed 91 \
  --eval_output /path/to/evaluation.json
```

TSIL checkpoint 自带构建观测所需的参数、配置、目标表、成功标记、训练几何描述符及尺度，无须重新载入 SGS bank。测试只查询成功 `full_task` 配置的最多 5 个邻居。描述符 schema 为 `dq_full_task_planar_v2`，包含 5 维：机器人 yaw 坐标下的物体相对 x/y、物体局部 x 轴在该平面的单位投影 cos/sin、桌高。这些量对应 full_task bank 的实际任务配置；机器人落地高度、身体倾斜和关节初始瞬态不再决定近邻支持域。训练 bank 与普通严格 reset 使用同一个描述符函数，不读取隐藏速度，不预热 EKF。

距离按训练 full_task 的各维范围归一化；范围不超过 `1e-4` 的近恒定维不参与距离，但仍按 `1e-4` 原单位容差检查支持域，其他维使用训练范围加 5% 边缘。无成功参考、超支持域或无效描述符均回退 `T_ref`；物体局部 x 轴的水平投影长度不超过 `1e-6` 时朝向无定义，属于无效查询。评估分别记录 `nearest`、`no_success`、`out_of_support`、`invalid` 的实际次数，不能把固定算法标签当成命中证据。目标在 reset 领取，整个回合固定。评估代码验证目标表和三类归一化状态未被修改。

新 TSIL checkpoint 使用 metadata/state version 2，并标明 `affine_output_v2` 价值变换和 `dq_full_task_planar_v2` 几何 schema；三个价值缓冲 `prediction_scale`、`prediction_bias`、`standard_deviation_floor` 必须完整。缺失或部分缺失的新状态会明确报错。

旧 SGS+EKF checkpoint 仍走原 PPO 路径。旧 TSIL v1 checkpoint 可显式迁移用于 **Actor 评估**：保留 Actor 和观测归一化，从旧 32 维描述符确定性提取这 5 维并重算训练支持域；旧 Critic 的裁剪语义不能由新仿射适配恢复，因此禁止其价值预测和训练加载，并发出说明。迁移来源随再次保存保留，重新加载也不会解除限制。该兼容路径不能补救旧模型此前受到的训练影响。

当前 SGS 训练的“从头训练”限制保留：这些 checkpoint 用于评估，尚不提供 archive／SGS／仿真状态的完整断点续训。现有学生蒸馏入口会明确拒绝 TSIL 教师，避免把 1286 维时间条件教师误当成原教师；学生蒸馏不属于本次教师训练实现。

TensorBoard／W&B 新增 `TSIL Archive`、`TSIL Selection`、`TSIL Replay`、`TSIL Targets`、`TSIL PPO`、`TSIL Reference` 指标。其中正优势质量按真实完整回合归因，跨 rollout 累计；动作似然诊断使用同一组成功历史动作，最多 1024 个诊断样本，不改变训练采样范围。

2026-11-01 起，SGS+EKF 与 SGS+EKF+TSIL 的总体成功率 `SuccessRate / sugar_box` 只统计最近 **1000 个已完成的 `full_task` 回合**，以便和 EKF baseline 的同名、同窗口长度曲线比较。其他配置类别的成功或失败都不进入主窗口，且不挤占窗口位置；`SuccessWindow / completed`、`SuccessWindow / successes` 使用相同范围。失败和超时计入分母，主动取消或未完成回合不计入；无完整任务完成时不生成该成功率。各类 `SuccessRate / <family>` 仍使用各自最近 100 回合，仅作分类诊断；SGS 更新和 TSIL 归档仍接收所有类别的有效回合。训练结束和中断均刷新主曲线及分类曲线。该口径对重新启动的训练生效，旧日志不回写。

## 已执行验证

**GPU 迁移验证（2026-11-01）：**

- 最终完整仓库测试运行 417 项：415 项通过、2 项跳过；报告为 `/tmp/tsil_gpu_migration_20261101/tests_final.log`。
- [GPU 归档对照测试](../DQ_high-level/tests/test_tsil_replay_gpu.py)：31 项通过，包含 CPU 参考与实际 CUDA 运行。覆盖双榜并集、严格平分准入、同配置同时结束、异步回合、缓冲所有权、原始 EKF 观测、行政重置、完整 MC 回报、正 gap 加权无放回采样及 GPU 驻留。另验证时间目标状态与回放位于不同设备时的数据归属。
- [PPO 测试](../DQ_high-level/tests/test_tsil_ppo.py)：22 项通过。CUDA 微批大小 2／100、混合／全零正 gap、非平凡价值坐标及已有在线梯度均与原完整 SI 损失的梯度和指标一致；确认每微批只调用一次 Critic。
- 真实 Isaac Gym 检查：64 环境 × 144 步，通过 106 项断言；6 次联合更新、180 次 optimizer step，132 个真实完整回合，最长 100 步。每轮回放集合固定，所有样本每 epoch 恰好使用一次，归档保持收到的 EKF 与时间通道，checkpoint 保存通过。Critic 参数最大变化为 `0.0166186`；优化前价值坐标最大误差为 `2.98e-7`。归档和采样张量确认为 `cuda:0`，PyTorch 峰值分配为 6.51 GiB。该短跑未产生成功回合，不能据此推断学习效果。

同一台机器、同一张 GPU 0、相同 4096 环境／seed 43／144 步／正式 100 步回合预算，分别运行修改前 CPU 回放与 GPU 版，无 W&B：

| 计时范围 | 修改前 | GPU 版 |
|---|---:|---:|
| 144 步训练，不含初始化 | 63.50 s | 58.50 s |
| 其中 6 次联合优化累计 | 10.19 s | 5.39 s |
| 其中仿真 `env.step` 累计 | 50.28 s | 50.55 s |
| 同一归档与冻结 Critic 的单次回放准备，预热后 3 次均值 | 1.211 s | 0.114 s |

最后一行是严格固定输入的独立比较：8865 条轨迹、417711 个候选 transition，两端均筛得 187293 个正 gap，采样 104113 条。它衡量完整重标记、Critic 筛选与采样，约加速 10.65 倍。前三行是两个独立训练短跑；随机数流改变后训练轨迹与回放样本数会不同，不能把联合优化约 47% 的耗时下降当作固定计算量的精确加速比，也不能外推为 120000 步整体提速。短跑总训练耗时下降约 7.9%，仿真仍占主要时间。

4096 环境验证进程的 PyTorch 峰值分配为 10.77 GiB，包含训练后的固定归档基准；不等于 `nvidia-smi` 的完整进程占用。GPU 归档按完整容量预留，显存成本高于旧 CPU 回放。验证报告位于 `/tmp/tsil_gpu_migration_20261101/`：`smoke_result.json`、`baseline_cpu/tsil.json`、`profile/tsil.json`；计时复现脚本为同目录 `profile_cpu.py`／`profile_gpu.py`，原实现快照保存在 `original/`。

**本次两项 P1 修复的回归验证（2026-10-30）：**

- 本机 `dqwbc` 环境先导入 Isaac Gym 后，完整测试发现运行 380 项，378 项通过、2 项跳过；其中 TSIL 相关 51 项。`test_tsil_ppo.py` 的 20 项默认开启与训练相同的值裁剪，覆盖空回放在线梯度、非平凡适配下的采集／bootstrap／筛选／SI、原始价值保持、Adam 状态不变、极小方差单位转换及 checkpoint 兼容限制。
- 原审查反例中，修复前 6 次 Critic 梯度均为 0；同 seed、空回放、开启值裁剪时，修复后 6 次梯度范数为 0.9797–0.9953，最大 Critic 参数变化为 0.00600043。RMS 更新后首次优化前，当前预测与缓存旧预测的新坐标误差为 0。
- 修复版复现脚本与数据是 `/tmp/audit_tsil_critic_fixed.py`、`/tmp/audit_tsil_critic_fixed.json`，原失败数据保留。
- [真实 GPU 联合训练检查](../DQ_high-level/tests/smoke_tsil_gpu.py)：64 环境 × 144 步，6 次联合更新、180 次 optimizer step；每轮首个在线 minibatch 在加入 SI 梯度前的 Critic 梯度范数分别为 7.15177、1.54543、2.03912、2.83837、3.60430、1.35530。RMS 更新后当前／旧预测新坐标最大误差为 `2.98023e-7`，Critic 最大参数变化为 0.0171256。
- 该训练检查归档 137 个真实完成回合，包含 100 步完整失败回合。配套 checkpoint 切换到 8 个评估环境，完成 16 个严格任务回合；训练未产生成功参考，因此查询正确统计为 `no_success=16`，目标和归一化冻结检查通过。
- [真实 GPU 近邻检查](../DQ_high-level/tests/smoke_tsil_lookup_gpu.py)：迁移旧 v1 checkpoint，使用普通严格 reset 的 32 环境 × 3 波初态。在内存中注入 `full_task` 成功参考及 `D=8秒` 以单独检验目标查询，96/96 次进入近邻分支并返回 8 秒；原审查同类条件为 0/96。人工参考只用于回归探针，不计为训练成功，也不据此报告策略成功率。

本次训练报告：`/tmp/dq_tsil_fixes_20261030/full144/smoke_tsil_result.json`；严格评估：同目录 `strict_eval.json`；近邻探针：`/tmp/dq_tsil_fixes_20261030/lookup.json`。

**首次接入记录的解释更正：**修复前曾运行 360 项通过／2 项跳过的测试及 64 环境 × 144 步 smoke，报告保留于 `/tmp/dq_tsil_smoke_20261030/full144/`。其合并 Actor/Critic 参数变化检查漏掉了在线 Critic 零梯度；没有成功参考的评估也未检验真实成功近邻命中。这些旧结果不能作为两项 P1 的验收依据，问题分析见[源码审查](TSIL_SGS_EKF_code_review.md)。

本次短程训练归档的 137 个回合均失败，严格评估成功率仍为 0；修复验证的是 Critic 学习信号、坐标一致性和目标查询实际路径，不代表已经提高任务成功率或完成速度。尚未开展 120000 步完整训练。
