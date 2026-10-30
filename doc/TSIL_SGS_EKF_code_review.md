**SGS+EKF+TSIL 教师训练源码审查**

**2026-10-30 修复复验：下述两项 P1 已修复，原始审查记录保留在后文。**

当前实现与数学细节见 [实现说明](TSIL_SGS_EKF_implementation.md)。本轮修复包括：

- **Critic 坐标一致性**：ValueScaler 保存非训练的输出仿射补偿，returns RMS 更新同时补偿当前预测，保持原始价值不变；全部 online/bootstrap/replay 路径共用补偿，不修改网络参数和 Adam 动量。价值归一化去除非可逆的 ±5 裁剪，保留 PPO `clip_predicted_values=True`，值损失采用作者的 `max(unclipped², clipped²)`。奖励单位变换同时缩放标准差下限，避免极小方差破坏一致性。
- **近邻查询有效性**：配置描述符由 32 维改为稳定的 5 维：物体在机器人偏航坐标系中的相对 x/y、物体局部 x 轴的平面单位方向、桌高。保留 bank 实际随机化的物体朝向，排除机器人落地高度、倾斜与关节瞬态；不根据测试结果扩大支持域阈值。评估逐回合记录 `nearest/no_success/out_of_support/invalid`，只统计实际参与评估的环境。
- **checkpoint 一致性**：新格式持久化仿射补偿和描述符 schema；旧 v1 的 32 维几何可直接迁移，无须重建 bank。旧模型只支持 Actor 评估，不能恢复有缺陷的 Critic；这个限制在再次保存、加载后仍保留。缺失部分补偿状态或元数据与描述符版本不匹配会明确报错。

| 修复验收 | 结果 |
|---|---|
| 原生产 PPO 反例，clip=True、空 replay | 原来 6 次 Critic 梯度全 0；现在全部为 0.9797–0.9953，最大参数变化 0.00600043 |
| 原反例 RMS 更新后、优化前的新旧预测坐标误差 | 0 |
| 全量测试，先导入 Isaac Gym | 380 项：378 通过、2 跳过；其中 TSIL 51 项全部通过 |
| 真实 GPU 训练，64 环境 × 144 步 | 6 轮更新、180 次联合 optimizer step，全链路通过 |
| GPU 每轮首 minibatch 的纯在线 Critic 梯度范数 | 7.15177、1.54543、2.03912、2.83837、3.60430、1.35530；均在 SI backward 之前测量 |
| GPU 每轮 RMS 更新后的坐标误差 / Critic 最大参数变化 | 最大 2.98023e-7 / 0.0171256 |
| 旧 checkpoint 实际无成功参考时的真实严格 reset | 32/32 正确标记 no_success，回退约 16 秒 |
| 内存合成成功参考 D=8 秒，3 波 × 32 个真实严格 reset | 原 0/96 命中，修复后 96/96 命中且返回 8 秒 |
| 真实距离、桌高、物体朝向越界及 NaN 查询 | 分别标记 out_of_support / invalid，并回退 T_ref |
| 新 v2 checkpoint 的严格评估，8 环境、16 回合 | 加载及冻结检查通过；16 个目标均明确标记 no_success 回退 |

**验证边界**：短程训练自然完成 137 个回合，均未成功；严格评估 16 回合也均未成功。96/96 是已有成功参考条件下的查询命中率，参考只在探针内存中构造，未修改 checkpoint、回合成功标签或训练成绩。这些结果证明两处机制修复与集成运行，不证明学习效果提高；没有启动正式 120000 步训练。原有 baseline 归一化可比性问题仍需在研究实验中单独处理。

可重复验证脚本为 [训练 GPU 检查](../DQ_high-level/tests/smoke_tsil_gpu.py)、[严格 reset 近邻检查](../DQ_high-level/tests/smoke_tsil_lookup_gpu.py) 及 `tests/test_tsil*.py`。本地证据保存在：

- `/tmp/audit_tsil_critic_fixed.json`，原失败数据 `/tmp/audit_tsil_critic.json` 保留。
- `/tmp/dq_tsil_fixes_20261030/full144/smoke_tsil_result.json` 与同目录 `strict_eval.json`。
- `/tmp/dq_tsil_fixes_20261030/lookup.json`。
- `/tmp/dq_tsil_fixes_cpu.log`。全量测试需先导入 Isaac Gym 再导入 torch；直接按字母顺序发现测试会触发既有的 Isaac Gym 导入顺序限制。

**以下为修复前的原始审查记录，代码行号和“当前”均指修复前快照，不代表修复后的状态。**

审查日期：2026-10-30。对象是当前工作区的新 TSIL 训练入口及其实际依赖，依据本地 [TSIL.pdf](papers/TSIL.pdf) 正文 §3、附录 A.1–A.2、表 5，以及 `/home/hehui/TSIL` 的 `137e3ab8d6f1eccf9cf6c611f957d106c2ff000c` 源码。此次未修改训练代码，另用独立探针验证关键路径。

**结论：核心机制已经接入，但目前还不能认定“完整且正确地实现了 TSIL”。确认有两处应优先修正的问题：价值归一化与 PPO 值裁剪会造成 Critic 零梯度；严格评估的描述符失配使 5-NN 时间目标推断实际退化。**

上一轮“完整实现”的结论过于乐观。短程训练、checkpoint 加载和已有测试通过，能够证明链路可运行，不能替代对 Critic 学习信号和成功近邻实际命中的验证。

**P1：Value RMS 的更新与 PPO value clipping 不一致，可使 Critic 有损失却完全不更新。**

代码定位：

- [tsil_ppo.py:261](../DQ_high-level/utils/tsil_ppo.py#L261)：先用本轮 returns 更新 Value RMS，再将缓存的旧预测和 returns 变换到新尺度。
- [tsil_ppo.py:303](../DQ_high-level/utils/tsil_ppo.py#L303)：当前网络直接输出预测，随后仅计算 `old_values + clamp(values-old_values, -0.2, 0.2)` 的 MSE。
- [教师入口:398](../DQ_high-level/train_multistate_DQ_teacher.py#L398)：实际配置明确开启 `clip_predicted_values=True`。

设采集时归一化统计为 `(μ_old, σ_old)`，网络输出为 `v`，缓存的反归一化预测是 `V_old=σ_old v+μ_old`。更新 RMS 后，代码将缓存改写为：

\[
v_{old,new}=(V_{old}-\mu_{new})/\sigma_{new}.
\]

但尚未进行任何参数更新的当前网络仍输出 `v`。因此，仅统计变化就能使当前预测与旧预测相差超过 0.2。此时 `clamp` 对当前预测的导数为零，PPO 的 value loss 无法修正这个误差。

这直接影响 TSIL 的关键依据：论文 §3.3 与附录 A.2 要求用可信的当前价值估计构造正回报差。Critic 无法正常学习时，正 gap 的筛选和辅助监督也会受到影响，不能只检查辅助损失函数本身有梯度。

独立复现直接调用生产 `TSILPPO`，使用小网络、相同 seed、4 个环境、2 步正奖励、未结束回合，因此回放为空；对照只改变值裁剪开关：

| 量 | 实际采用的裁剪开启 | 关闭裁剪对照 |
|---|---:|---:|
| 更新前 Value RMS mean / variance | 0 / 1 | 0 / 1 |
| 用 returns 更新后的 mean / variance | 1.306889 / 0.521170 | 相同 |
| 网络同参数输出 | 0 | 0 |
| 旧预测转换到新尺度 | -1.810293 | 相同 |
| 6 次 optimizer step 的 Critic 梯度范数 | **全部为 0** | 0.209–0.868 |
| Critic 最大参数变化 | **0** | 0.001518 |
| 记录的平均 Value loss | **3.797328** | 0.475733 |

这是确定性的正确性反例，不是在断言正式训练的每个 minibatch 都必然零梯度。空回放是合法的训练初始状态；有回放时，辅助 Critic 损失也不能替代正常的在线价值学习。

此硬裁剪写法继承自仓库 vendored skrl，不能把它归因于 TSIL 论文。作者 `core/training/algo/utils.py::compute_value_loss` 使用未裁剪与裁剪损失的最大值，而当前仅对硬裁剪后的预测计算损失。不过，只换成 `max(unclipped_loss, clipped_loss)` 仍不足以解决上述统计坐标变化，因本例可能仍选择零梯度的裁剪分支。

修正要求是统一价值统计变化和预测裁剪的语义：在进入优化前，让旧预测与当前预测在同一价值坐标下保持一致。可采用适当的输出仿射补偿／PopArt 类处理并考虑优化器状态，或重新安排归一化快照和裁剪基准。PopArt 不是论文强制要求；关键验收条件是，**不能仅因 RMS 更新而将未更新参数的 Critic 推入裁剪死区**。直接关闭裁剪可作为定位对照，但属于明确的 PPO 行为变更，不应假称已经保持原 PPO 完全不变。

现有测试为什么漏掉：

- [test_tsil_ppo.py:62](../DQ_high-level/tests/test_tsil_ppo.py#L62) 的测试 agent 没有开启真实入口使用的值裁剪，继承了默认 `False`。
- [smoke_tsil_gpu.py:240](../DQ_high-level/tests/smoke_tsil_gpu.py#L240) 对 Actor/Critic 合并检查“至少有参数变化”；Actor 变化即可通过，不能证明 Critic 在在线损失下学习。
- 单独检查 `positive_gap_losses` 的辅助 Critic 梯度，覆盖不到这里的在线 PPO 值裁剪。

复现脚本：`/tmp/audit_tsil_critic.py`；数据：`/tmp/audit_tsil_critic.json`。运行：

```bash
/home/hehui/miniconda3/envs/dqwbc/bin/python /tmp/audit_tsil_critic.py
```

**P1：严格评估中的 5-NN 公式存在，但当前真实初态几乎必然被支持域检查拒绝。**

这里的“几乎必然”有明确结构原因；实际验证的 96 个 reset 全部被拒绝，并非已经穷举所有可能 reset。

代码定位：

- [tsil.py:51](../DQ_high-level/utils/tsil.py#L51)：描述符包含机器人高度及全部 19 个关节，包含腿部关节。
- [tsil.py:139](../DQ_high-level/utils/tsil.py#L139)：所有维度都必须位于训练范围加容差内，任一维越界立即回退 `T_ref`。
- [b1z1_pickmulti.py:65](../DQ_high-level/envs/b1z1_pickmulti.py#L65)：训练参考取自 bank 内保存的物理状态。
- [b1z1_base.py:1032](../DQ_high-level/envs/b1z1_base.py#L1032)：普通严格评估 reset 将机器人高度置为 0.60 m；随后 [b1z1_pickmulti.py:708](../DQ_high-level/envs/b1z1_pickmulti.py#L708) 仅执行一个物理 tick，就在该 reset 中查询目标。

训练 bank 的 `full_task` 状态已经过物理稳定，而严格评估在刚重置后查询。把落地瞬态和腿关节微小差异都当成配置支持域约束，会拒绝原本在目标任务参数范围内的测试初态。

本次用真实 Isaac Gym 严格评估 reset 做了 3 波、每波 32 个环境的探针。为单独检测近邻分支，只在内存中将 2048 个 `full_task` 训练参考标为成功、目标设为 8 秒；未修改 checkpoint，也未将这些人工标记当作训练成功：

| 检查项 | 结果 |
|---|---|
| 查询数 | 96 |
| 近邻推断正常返回 8 秒 | **0** |
| 回退最大时限约 16 秒 | **96** |
| 相对物体 x/y、桌高处于训练支持域 | **96/96** |
| 训练机器人高度范围 | 0.51436–0.51546 m |
| 评估机器人高度范围 | 0.59980–0.59986 m |

仅机器人高度这一维就足以拒绝全部查询，另有多个姿态、关节维度同样拒绝。bank `full_task` 的构建参数覆盖 2 m 起点及比严格评估更宽的偏航范围，因此问题不是距离、方位和桌高真正超出训练任务覆盖。

论文附录 A.2 的核心是：存在同任务成功参考时，从相近训练配置迁移时间目标。当前 DQ 设计增加的超支持域回退本身可以合理，但描述符的初始化时相不一致，使这项迁移在当前严格评估协议中无法正常生效。这属于适配未完成，不能以“回退分支正确执行”代替“新配置目标推断已验证”。

修正应统一参考与查询的配置含义和初始化时相，采用稳定、可观测的任务配置描述，避免腿部瞬态主导支持域；不能简单读取隐藏速度，也不能仅依据测试数据放宽阈值。应分别记录正常近邻、无成功参考、超支持域回退的数量。

[eval_sgs_teacher.py:126](../DQ_high-level/eval_sgs_teacher.py#L126) 当前无条件输出 `target_lookup=successful_full_task_5nn`，并不是实际命中证明。上一轮 smoke checkpoint 没有成功配置，所有目标本来就应回退，因而没有覆盖“已有成功参考时”的真实集成路径。

探针脚本：`/tmp/audit_tsil_eval_descriptors.py`；每波、每维原始数据：`/tmp/audit_tsil_eval_descriptors.json`。

**论文核心机制逐项核对。**

以下“已实现”指对应代码规则与数据流成立，不表示在上述 Critic 问题影响下已经证明训练效果。

| 论文关键内容 | 当前实现 | 审查结论 |
|---|---|---|
| 同一次快速成功同时提供时间目标与可模仿轨迹 | `TemporalState` 与每配置 archive 由真实结束回合更新 | 已接入核心思想 |
| 式 1：每配置历史最快成功时间、单调收紧 | `observe_success` 对重复 config ID 做最小值归约；联合优化后 `commit_targets` | 已实现 |
| 目标在下次 reset 领取，不改写在线 old log-prob | `tsil_episode_D` 回合内固定，pending 更新不改在线数据 | 已实现 |
| elapsed 与 D 同时输入 Actor/Critic | EKF 后插入两维，除以 T_ref，末尾 grasp/action 切片保持 | 已实现 |
| 式 2：软时间成功奖励 | 50×success×[1+min(D/T,1)]，不增加截止失败，不放松 10 步 hold | 已实现 |
| 每配置最快成功 Top-5 | 按实际秒数排序，按唯一回合 ID 去重 | 已实现 |
| 同配置高回报补缺 | 成功优先，剩余槽位从 return pool 补充，无 50/50 固定比例 | 已实现 |
| 完整回合跨 rollout 保存 | env 槽只追踪在途回合，真正结束后按旧 config 归档 | 已实现 |
| 当前目标重标 | 只改历史 D 通道；重新计算成功奖励和 MC returns | 已实现 |
| 不泄漏未来 EKF 信息 | 原始因果 EKF 输入逐步保存，不用轨迹后续状态重算过去预测 | 已实现 |
| 全配置正 gap 筛选 | 遍历所有被选择轨迹，按 gap 概率无放回选最多 32 条 | 采样规则已实现；gap 的可靠性受 P1 影响 |
| update 内固定 replay set，跨 E epochs 复用 | `prepare` 一次，之后仅 shuffle | 已实现 |
| 效率权重 w=1+success×min(D/T,1) | Actor 权重包含 w，失败 w=1 | 已实现 |
| 正权重 std、不减均值、cap=1 | 对整个逻辑 batch 的正权重求总体标准差 | 已实现 |
| 辅助 Critic 单边平方回归 | 0.5×relu(G_norm−V_norm)²，保留梯度，不乘 w | 损失函数已实现 |
| PPO 与 SI 同一次优化 | 0.05×(SI_actor+0.05×SI_value)，共同 optimizer、共同梯度裁剪 | 已实现 |
| 微批不改变损失定义 | 共用逻辑 batch 的 std，按样本数比例累计 | 已实现 |
| 逐 env trace、按 task 汇总 reward scale | terminal reward 先进入统计再清 trace；replay 只读尺度 | 已实现 |
| 物理观测与时间不同归一化 | 时间豁免 RMS；采集和对应优化使用同一物理观测快照 | 已实现 |
| 在线/回放/旧值/当前 Critic 的统一价值尺度 | 目标和旧值已变换，但与当前网络及值裁剪缺少一致处理 | **未正确完成，P1** |
| 未见配置按成功参考 5-NN 推断 D | 算法和 checkpoint 数据存在；真实严格 reset 全部被支持域拒绝 | **未有效完成，P1** |
| 评估冻结目标、SGS 与归一化 | 模型配套状态恢复、SGS 关闭、状态前后检查 | 已实现 |

**这些差异属于已经明确的 DQ 适配，不应误报为漏实现。**

- 原论文支持外部截断轨迹的 value bootstrap。当前 DQ 把 100 步预算结束定义为任务失败，因此完整回放使用零终态尾值；24 步 PPO rollout 边界仍正常 bootstrap，未被伪装成完整 fallback。这与已确定的有限任务定义一致。
- `command_reward` 的训练课程导致任务奖励随阶段改变，代码记录门控前分量并按当前开关重建；这是显式的 DQ 扩展。approaching/lifting 等有状态历史增量没有重新调用奖励函数。
- SGS 会在 reset 更换配置，因此使用 config ID 代替作者源码的固定 env ID 是必要改动。
- 时间用实际 0.14/0.16 秒累计，包含 EKF 冷启动和完整 hold；没有把 EKF horizon 当成目标总时限。
- 学生蒸馏适配和 archive/SGS/仿真状态完整续训当前明确不支持，不能说已经支持，但不属于 TSIL 教师核心算法缺失。

**另一个研究实验口径问题：TSIL 与原 SGS+EKF baseline 的归一化并不一致。**

论文附录 A.2 要求方法之间共享归一化及 PPO 主设置。当前只有 TSIL 路径新增逐环境 reward trace、冻结 observation RMS、每轮仅由 returns 更新一次 Value RMS；原 SGS+EKF 仍走旧 skrl PPO，奖励未使用同一 trace 缩放，观测／价值统计更新时序也不同。

这不使上述 TSIL 机制凭空失效，也不是第三个训练阻断问题；但若直接拿旧 SGS+EKF 训练结果与此 TSIL 结果比较，性能变化包含归一化改动的影响。要研究 TSIL 本身的贡献，需要提供共享数值处理的对照口径，或把这些数值处理变化明确列为独立实验变量。相同 PPO 超参数并不自动等于相同优化基线。

**本次独立验证及其边界。**

- 重新运行现有 34 项 `test_tsil*.py`，全部通过；这与发现上述问题不矛盾，说明缺少对应回归场景。
- 独立生成 7000 个回合，把当前双榜结果与保存全量历史后重新排序的 oracle 比较，所有配置一致；23 配置的全遍历采样得到 460 条不重复 transition。脚本：`/tmp/audit_tsil_archive_oracle.py`。
- 运行上述生产 `TSILPPO` 最小 Critic 反例，确认有损失但零梯度，以及只改变 clipping 的有效对照。
- 真实 GPU 严格 reset 的 96 个初态探针，确认成功近邻分支被描述符支持域整体拒绝。
- 本次没有进行长程学习效果实验，也没有据这些探针声称成功率必然增加或降低。

后续修正应覆盖两个验收条件：真实训练配置下在线 Critic 能得到有效梯度，且统计更新本身不制造值裁剪偏移；真实严格 reset 在存在合适成功参考时确实进入近邻分支，真正无参考／超覆盖时才回退。在这两项完成前，更准确的状态是：**TSIL 的训练主体已经实现，关键数值细节与评估泛化适配仍需修正。**
