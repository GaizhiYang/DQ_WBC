# SuperMimic＋SGS＋EKF 教师训练：实现与运行

实现日期：2026-11-02。对应[方法方案](SuperMimic_SGS_EKF_teacher_method_proposal.md)和[网络数据流图](SuperMimic_SGS_EKF_training_flow.html)。

已实现 **Isaac Gym 真实 PPO＋共享高层世界模型＋短期闭环 BPTT**。继续使用当前四足机械臂环境和冻结低层控制器。世界模型用于训练期辅助 Actor；Critic 始终用真实 PPO 数据学习。新增模块在 CUDA 上运行，不需要 MuJoCo Warp。

## 1. 启动训练

在仓库根目录，激活现有环境：

~~~bash
conda activate dqwbc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh --no-wandb
~~~

新入口复用原有 SGS＋EKF 脚本，默认 4096 个环境、120000 步、GPU 0、hybrid 模式。去掉 --no-wandb 可恢复原入口的 W&B 日志。NUM_ENVS、TIMESTEPS、SEED、RUN_NAME、EXPERIMENT_DIR、SGS_BANK 等原有环境变量继续有效。

主训练的模拟器与学习器必须在同一张 GPU。可以将独立验证放到另一张卡：

~~~bash
RL_DEVICE=cuda:0 SIM_DEVICE=cuda:0 \
bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh \
  --no-wandb --wm_probe_device cuda:1
~~~

验证会暂停当前训练更新，在独立进程中完成后继续。单卡也能运行，此时要为训练场景和验证场景同时预留显存。

原入口仍可直接使用新参数：

~~~bash
bash DQ_high-level/train_teacher_sgs_ekf.sh --no-wandb \
  --world_model hybrid --wm_probe_interval 2400
~~~

三组对照用同一入口切换：

| 参数 | 行为 |
|---|---|
| --world_model off | 原 SGS＋EKF＋PPO |
| --world_model fit | 真实 PPO＋世界模型拟合，不做辅助 Actor 更新、不自动验证 |
| --world_model hybrid | 真实 PPO＋拟合；通过验证后增加短期 BPTT |

未传入世界模型参数时，原脚本默认仍为 off。新增脚本默认 hybrid。

## 2. 实际训练顺序

1. SGS 选择真实初态，Actor 在 Isaac Gym 中执行动作。
2. Wrapper 在重置前保存完整终态、实际执行动作、实际控制时长、奖励、成功／失败／超时与回合编号。
3. 原 PPO 完成一次真实数据更新：Actor 和 Critic 保持原训练目标。
4. 世界模型从 GPU replay 采样同回合连续窗口，沿记录的真实动作自回归展开，更新模型参数。
5. 到验证间隔时，导出候选快照并在独立 Isaac Gym 进程中检验动作效果。
6. 合格快照通过留出集误差检查后，从可信真实状态展开 H 步闭环想象，通过短期奖励与终点真实 Critic 价值更新 Actor。

想象数据不进入 PPO rollout，也不进入 SGS 成功历史。模型拟合、真实重置和想象起点各自采样。

模型参数分成两份：**持续拟合的 candidate** 和 **已经验证、保持固定的 validated snapshot**。只有报告覆盖的精确权重才能用于 BPTT。候选继续拟合不会悄悄改变已经批准的梯度模型；新候选失败时，旧快照仍须通过当前真实数据的误差检查才能继续使用。

## 3. 状态、动作与网络

当前指定 sugar_box／SGS／EKF／no_velocity_obs 入口下，完整递归状态为 **914 维**，Actor 仍使用原有 **1284 维观测**。维度由适配器从环境推导，改变配置后不应手工沿用这些数字。

| 组成 | 维数／处理 |
|---|---|
| 机器人根、物体、台面、末端刚体状态 | 各13维：位置、四元数、线速度、角速度 |
| 19个关节位置、速度和上次速度 | 57维 |
| 冻结低层控制器的10帧历史 | 710维 |
| 上次低层动作、步态和足接触记忆 | 29维 |
| 累积末端目标、底盘命令、上次高层动作 | 18维 |
| EKF位姿历史、时间间隔和有效历史数 | 41维 |
| 奖励历史、保持计数、成功记录、步数、台面高度 | 7维 |

初版选择显式保存低层历史，使后续 Actor、奖励、EKF 和控制记忆可以递归重建。这是方案中“补充完整控制状态”的实现选择，尚未做历史编码压缩。

动力学网络为三层512宽 SiLU MLP，输入为归一化物理／控制特征、9维有效动作、物理条件和实际 dt；残差输出 **844维**，末层零初始化。刚体姿态用三维旋转增量更新并归一化四元数。物理随机化和物体特征通过 GPU 表查询；不把摩擦、质量等训练侧特权信息添加到 Actor 观测。

末端状态、关节状态、低层记忆分别预测，当前没有以 FK 强制它们完全一致。台面也作为受条件约束的状态预测，尚未把全部台面运动解析替换掉。累积控制目标、EKF历史、任务计数与奖励历史按已知规则推进。

高层动作包含8个连续分量和夹爪命令。输入模型前显式处理真实执行中的增量限幅、工作空间约束、命令死区和夹爪符号，避免网络对饱和命令给出虚假的物理导数。上次原始动作仍保留在观测和动作变化奖励中；该历史不作为物理 MLP 的输入。

## 4. 模型拟合与短期 BPTT

模型拟合使用同回合的三步真实动作窗口，后一步输入前一步预测状态，不插入真实未来状态。损失包含：

- 按物理／控制字段平衡的归一化状态误差，避免710维历史掩盖物体和末端误差。
- 物体相对末端的位置、姿态、线速度、角速度，以及离台面高度的五组辅助误差。
- 自然终止分类的 BCE；超时按任务时钟单独处理。

位置、旋转、线速度、角速度的尺度分别为0.05 m、0.1 rad、0.5 m/s、1 rad/s。相对误差姿态项对四元数正负号等价。

BPTT 的完整可微路径为：

~~~text
真实起点（固定）
  → 因果观测／EKF → Actor → 连续动作
  → 世界模型 → 预测状态
  → 下一次因果观测／EKF → 同一个 Actor → …
  → 短期累计奖励 ＋ 末端 Critic 价值
  → 沿全部预测步反向传播 → Actor
~~~

世界模型、Critic 和归一化器的参数／统计在此期间冻结，但保留它们对输入的导数。特别处理了 skrl RunningStandardScaler 默认关闭输入梯度的行为，并把 Critic 输出反归一化为真实奖励单位。

夹爪开闭采用事先固定的参考原始命令序列，不对符号切换使用直通梯度。PPO 继续学习夹爪；共享 Actor 参数的辅助更新可能间接改变夹爪分布，因此 KL 检查覆盖全部9维动作。

默认 H=3，每一步独立采样真实控制周期0.14／0.16秒，对应约0.42～0.48秒。每个终止步仍计入该步奖励，之后停止累计；有限任务超时不追加尾值，与当前真实 PPO 的目标一致。

辅助更新采用独立 Adam，实际步长为 actor_learning_rate × auxiliary_strength。默认是 1e-5 × 0.05 = 5e-7；只缩放 loss 会被 Adam 的归一化大幅抵消，因此这里直接缩放学习率。辅助步之后在真实观测锚点上检查全动作高斯 KL，超过0.005或出现非有限结果时，完整恢复 Actor 和 Adam 状态。

## 5. 何时允许 BPTT

hybrid 不意味着第一步就通过未训练模型更新 Actor。需要同时满足：

1. 已完成 replay 预热。
2. 有对应精确模型权重与任务配置的真实动作效果验证报告。
3. 留出回合中的每个预测步通过置信误差检查。
4. 当前想象起点的真实一步误差足够小。
5. 本次 Actor 提议通过梯度有限性和 KL 检查。

置信误差为逐样本 max(物理归一化误差，相对关系辅助误差)。留出集先对样本取均值，再检查所有预测步；起点权重为 max(0, 1 − 置信误差／起点阈值)，权重不参与求导。

训练／验证按完整回合划分，避免相邻时间窗泄漏。当前划分不等价于未见过物体或物理参数上的泛化验证。

独立动作效果验证使用5个新进程：生成动作前缀，以及基准、正扰动、负扰动、重复基准四条真实分支。所有分支重建相同随机种子／前缀，不恢复训练环境的 PhysX 内部状态。验证场景在三类 SGS 初态之间均衡抽样，不修改主训练 SGS。

默认至少64条可复现配对和32条有效动作效果；检查符号一致率≥0.70、相关系数≥0.25、模型选优动作的真实一步奖励收益95%下界≥0，并检查重复仿真的噪声。报告约束模型权重、任务配置、文件内容和模拟器类型。

**这验证的是一步有限动作扰动效果，不是长期成功率，也不保证 Critic 输入梯度完全可信。** 长期收益仍需多种子、完整任务成功率及 off／fit／hybrid 对照验证。门限是本实现的保守起始设置，并非论文已为 DQ 标定的超参数。

## 6. 可调参数

| 参数 | 默认值 |
|---|---|
| --wm_replay_envs | 128；固定记录前 min(128, num_envs) 个环境的连续轨迹 |
| --wm_replay_capacity | 每个记录环境256步，默认约234 MiB replay |
| --wm_window／--wm_horizon | 3／3 |
| --wm_batch_size／--wm_batches_per_update | 128／2 |
| --wm_warmup_steps | 96个向量环境步 |
| --wm_imagination_batch | 128 |
| --wm_learning_rate | 2e-4 |
| --wm_actor_learning_rate／--wm_auxiliary_strength | 1e-5／0.05 |
| --wm_max_validation_error／--wm_max_start_error | 0.25／0.25 |
| --wm_max_kl | 0.005 |
| --wm_probe_interval | 新启动脚本2400；直接 Python／原入口默认0 |
| --wm_probe_envs／--wm_probe_prefix_steps | 128／8 |
| --wm_probe_device | 与学习器相同，可指定另一张 CUDA 卡 |

间隔和预热按向量环境步计数，不乘环境数量；检查发生在 PPO 更新边界。当前 PPO 每24步更新一次。

其他参数可通过 --wm_config 传入 YAML，命令行覆盖 YAML。例如：

~~~yaml
world_model:
  hidden: [512, 512, 512]
  term_hidden: [64, 64]
  holdout_fraction: 0.1
  recent_fraction: 0.5
  recent_steps: 32
  use_tail_value: true
  probe_timeout_seconds: 900
  report_max_age_seconds: null
~~~

关闭尾值是 use_tail_value: false。--wm_probe_interval 0 仅关闭自动验证，不代表绕过验证。

## 7. 输出、检查点与手动验证

运行目录下 world_model/ 包含：

- candidate_XXXXXXXX.pt：某次真实验证覆盖的候选快照，包含模型、Teacher、scalers 和配置。
- probe_XXXXXXXX.json、对应日志和 .json.artifacts/：真实验证结论与分支证据。
- last_probe_status.json：最近一次自动验证状态、具体失败原因和耗时。
- candidate_latest.pt、status.json：正常结束或 Ctrl+C 时导出的最新候选及状态。

TensorBoard/W&B 中新增 World model / 前缀的模型误差、GPU replay、验证和 BPTT 指标。候选拟合误差与固定验证快照误差分别记录。常见状态 waiting_for_probe 表示继续 PPO＋拟合，heldout_error_rejected 表示暂不信任当前快照，hybrid_updated 表示完成了被接受的 BPTT 辅助步。

手动验证：

~~~bash
cd DQ_high-level
python probe_teacher_world_model.py \
  --checkpoint /path/to/world_model/candidate_latest.pt \
  --output /path/to/probe.json --num_envs 128 --device cuda:1
~~~

命令正常退出表示成功生成报告；是否合格以 JSON 的 passed 为准。加载已验证快照时需要成对传入 --wm_validated_checkpoint 和 --wm_probe_report。--wm_candidate_checkpoint 仅初始化候选世界模型，不恢复 Teacher／PPO 或 SGS 训练进度。

完整 Teacher checkpoint 同时保存模型优化器、辅助 Actor 优化器和已验证快照／报告；评估仍使用原 Actor 接口。当前入口沿用原 SGS 不支持精确续训的限制，拒绝 --resume；replay、模拟器状态和 SGS 完整进度不在新增续训协议中。不能把候选初始化称为断点续训。

## 8. 已进行的 GPU 验证

测试机器：RTX4090，dqwbc 环境 Python3.8.20、PyTorch2.4.1+cu121。

- 全部回归测试共501项：499通过、2跳过、无失败，包含新增 CPU／CUDA 测试。
- 真实入口：32环境×48步，GPU0 完成两轮 PPO 和两轮候选拟合，保存 checkpoint。
- 自动真实验证：GPU1、96环境、三类 SGS 各32条，全部配对可复现；起点、条件、时长和重复基准奖励误差均为0。
- 此次只有两次更新的候选，符号一致率0.7222、相关性0.241628。相关性未达到0.25，正确拒绝启用辅助 BPTT，保持 waiting_for_probe。
- 独立机制诊断：真实 Teacher／Critic／scalers＋三步 BPTT，Actor 梯度有限且非零；世界模型、Critic 和 scaler 数值不变。放大提议步长后完整回滚 Actor＋Adam。
- 真实状态适配核对覆盖成功、失败和原100步时限的自然超时；默认奖励误差小于1e-6，观测误差小于1e-4。

这些结果验证了实现与 CUDA 执行，不构成抓取成功率提升的实验结论。测试报告摘要保存在 [GPU 验证记录](SuperMimic_SGS_EKF_gpu_validation.json)。

复现小规模完整训练：

~~~bash
NUM_ENVS=32 RUN_NAME=wm_smoke EXPERIMENT_DIR=/tmp/dq_wm_smoke \
bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh --no-wandb \
  --stop_after 48 --wm_probe_interval 48 --wm_probe_envs 96 \
  --wm_probe_prefix_steps 2 --wm_probe_device cuda:1 \
  --wm_warmup_steps 24 --wm_replay_envs 16 --wm_replay_capacity 64 \
  --wm_batch_size 16 --wm_batches_per_update 1 --wm_imagination_batch 16
~~~

仅验证模型拟合入口可设置 --wm_probe_interval 0。单卡可把 --wm_probe_device 改为 cuda:0。

独立检查真实教师 BPTT、KL 回滚和自然超时：

~~~bash
python DQ_high-level/tests/smoke_world_model_gpu.py \
  --device cuda:0 --output /tmp/dq_world_model_gpu.json
~~~

运行回归测试时 Isaac Gym 必须先于 torch 导入：

~~~bash
PYTHONPATH=DQ_high-level python -c 'import isaacgym; import unittest; result=unittest.TextTestRunner().run(unittest.defaultTestLoader.discover("DQ_high-level/tests", pattern="test_*.py")); raise SystemExit(not result.wasSuccessful())'
~~~

## 9. 代码导航与复用来源

| 文件 | 职责 |
|---|---|
| [wm_env.py](../DQ_high-level/utils/wm_env.py) | 完整状态、物理条件、可微 EKF／观测／控制／奖励／终止 |
| [world_model/](../DQ_high-level/utils/world_model/) | 残差动力学、GPU replay、多步拟合、BPTT、动作效果验证 |
| [wm_ppo.py](../DQ_high-level/utils/wm_ppo.py) | 真实 PPO 后的拟合、快照升级和可信 BPTT |
| [wm_entry.py](../DQ_high-level/utils/wm_entry.py) | 参数、独立验证进程、导出与状态记录 |
| [wrapper.py](../DQ_high-level/utils/wrapper.py) | 重置前真实状态采集 |
| [probe_teacher_world_model.py](../DQ_high-level/probe_teacher_world_model.py) | 独立 Isaac Gym 动作效果验证入口 |

MLP、残差初始化、GPU replay 组织、自回归拟合、冻结模型的 BPTT 目标借鉴并改写自本机 SuperMimic-Reproduced。DQ 状态、SGS、PPO、EKF 和夹爪／动作约束进行了任务适配，不能直接照搬 G1 跟踪接口。具体来源文件、提交号和完整 MIT 许可见 [THIRD_PARTY_LICENSES.md](../DQ_high-level/utils/world_model/THIRD_PARTY_LICENSES.md)。这是独立复现仓库的代码迁移，不标称为 SuperMimic 官方实现。
