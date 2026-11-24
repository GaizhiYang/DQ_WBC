# 动态接触过渡课程：仓库级实现方案

日期：2026-11-23  
项目：DQ_WBC-1888ce7  
研究依据：[Dynamic_contact_transition_curriculum_design.md](Dynamic_contact_transition_curriculum_design.md)  
架构图：[Dynamic_contact_transition_architecture.html](Dynamic_contact_transition_architecture.html)

**本文是实现设计，不代表下述新模块已经存在。此次交付为文档与架构图，没有修改训练源码或运行训练。** 代码位置对应当前工作区，后续以函数名为准；所有新增入口和配置示例均须完成对应阶段后才能执行。

## 1. 实现目标与首版范围

首版实现一条可重复、可审计的离线链：

**已有教师 → 完整任务事件采集 → 动态前驱生成 → 闭爪窗口探测 → 恢复及物理验收 → 冻结 reset bundle → 原 SGS／PPO 从头训练 → 完整任务评估。**

研究对象先限定为“未接触／单侧接触到有效双侧接触”。离桌和最终 hold 是下游验证，不增加阶段奖励，不改变部署观测，不把生成器轨迹加入 PPO。

采用以下工程决策：

| 决策 | 首版做法 | 原因 |
|---|---|---|
| 生成方式 | 有限批量前向 shooting，两个层次分别生成前驱、测时序 | 能明确区分生成与筛选，不必先建设通用 RRT |
| 执行隔离 | collector 独立进程与模拟器 | 筛选时 simulate 不推进正在训练的其他环境 |
| 恢复方式 | 显式 context codec；continuation／practice 两种协议 | 不能通过复制姿态或 restore 所有 tensor 实现正确续接 |
| 存储 | 新 TransitionBankBundle 组合格式；旧 ConfigurationBank schema 1 保留 | 动态态不能继承静态 bank 的验证语义 |
| 训练接口 | 原 --sgs_bank 接受 bundle 的 manifest.json 文件，经新 loader 分派 | 现有参数和文件存在性检查可复用 |
| 采样器 | 原 SGSSampler，成功仍为最终 hold | 第一轮隔离候选生成的贡献 |
| 运行时更新 | 不支持在线替换 bank | 先验证离线候选价值，避免引入延迟结算和版本切换复杂度 |
| 模型恢复 | collector 支持冻结推理；新 bank 训练仍从头开始 | 现有 SGS 训练明确拒绝 checkpoint／resume |

需要单独排期的后续功能：在线 bank manager、训练效用分叉实验、学习式外层分配、通用接触图规划、真机在线接触估计。它们不属于首版完成标准。

## 2. 当前代码的实际调用链与改动约束

下表中路径均相对仓库根目录。

| 现有位置 | 当前职责 | 实现时必须保留的语义 |
|---|---|---|
| DQ_high-level/train_teacher_sgs_ekf.sh | 设置 v12、SGS、EKF、normalized action 等默认参数 | 不改变旧命令默认行为；显式指定 manifest 才进入新路径 |
| DQ_high-level/utils/config.py，configure_teacher_experiment，约 100–120 行 | SGS、严格评估与 bank 文件检查 | strict eval 仍关闭 SGS、2 m、Level02、100 步；SGS resume 仍拒绝 |
| DQ_high-level/train_multistate_DQ_teacher.py，get_trainer，约 341 行 | 配置、模型、scaler、memory 和 trainer | 抽取共享推理构建能力，避免 collector 复制一份易漂移逻辑 |
| DQ_high-level/utils/wrapper.py，IsaacGymPreview3Wrapper | raw Actor history、step／reset 适配 | history 属于 Wrapper；恢复不能只处理底层 env |
| DQ_high-level/envs/b1z1_base.py，step，约 1965 行 | 高层动作、低层控制、物理推进 | 事件观察器只读；不得额外推进 gait 或模拟器 |
| 同文件，floating_table_control，约 1950–1962 行 | pending root／DOF 的批量提交 | reset 内不 simulate，继续复用 deferred setter |
| DQ_high-level/envs/b1z1_pickmulti.py，_init_sgs／_reset_sgs_envs，约 124／209 行 | bank 加载、旧回合结算、新起点恢复 | 先结算旧 ID，后采新 ID；取消回合不伪造失败 |
| DQ_high-level/modules/pose_prediction.py，RollingSE3Predictor | 过去位姿窗口与 EKF 外推 | 导入历史需重定位时间戳，并只修改目标 env |
| DQ_high-level/utils/sgs.py，ConfigurationBank／SGSSampler | 静态物理 bank、最终成功率采样 | 原 bank 不可变，保留 fingerprint 与旧校验 |
| DQ_high-level/utils/sgs_bank_builder.py | IK、几何、物理验证、冷恢复 | 可复用工具；create_builder_environment 不等同于教师动态环境 |
| DQ_high-level/eval_sgs_teacher.py，evaluate_teacher | 同步 waves 的完整任务严格评估 | 不采训练 bank、不更新策略和 scaler |
| third_party/skrl/skrl/trainers/torch/base.py，single_agent_train，约 148–205 行 | step → record_transition → post_interaction → reset | 先记录 terminal observation，再 reset；未结束回合跨 rollout 继续 |

当前 Wrapper 将 reset_buf 返回为 terminated，truncated 恒零；另有 infos["time_outs"]。仓库中的 PPO 默认 time_limit_bootstrap=False。**本方案不顺手修改 timeout／GAE 语义**，否则会把算法改动混入 reset 方法。

实际运行可能使用安装环境中的 skrl。新运行记录必须保存实际导入的 PPO／trainer 文件路径及版本，GPU 验收时确认调用顺序，不仅检查仓库副本。

## 3. 目标模块结构

以下路径都是**拟新增**，已有文件的改动另列于第 12 节。

| 拟新增模块 | 主要对象／函数 | 输入 → 输出 | 依赖边界 |
|---|---|---|---|
| utils/teacher_inference.py | TeacherRuntimeSpec、resolve_teacher_runtime、build_frozen_teacher | checkpoint＋保存配置 → 冻结推理 runtime | 共享现有 Policy、Value、GFM、scaler 和环境工厂 |
| utils/contact_transition/config.py | TransitionConfig、校验与签名 | YAML → 类型化配置、hash | 纯 Python；创建 GPU 前校验 |
| utils/contact_transition/events.py | ContactEventObserver、EventBatch | 刷新后的物理张量 → 接触证据、事件、置信度 | 启用时只读；不写任务成功或奖励 |
| utils/contact_transition/snapshot.py | TransitionSnapshotCodec | env＋Wrapper → ContextBatch；反向恢复 | 使用显式字段注册，不遍历复制所有属性 |
| utils/contact_transition/probe.py | BranchRunner、ActionIntervention、CostLedger | 父状态＋动作族 → 前驱、窗口、分支记录 | 独立 collector；无 PPO memory |
| utils/contact_transition/selection.py | select_candidates | 已验候选池 → 固定 row index | 分层、去重、配额；不修改 SGS 权重公式 |
| utils/contact_transition/bank.py | TransitionBankBundle、ResetSource、load_reset_source | legacy .pt／manifest → 统一只读 reset 源 | 新旧格式分派；不修改旧 schema |
| collect_contact_transitions.py | 采集入口 | checkpoint＋完整任务条件 → source archive | 全任务轨迹、事件锚点、高层 context |
| build_contact_transition_bank.py | 探测与建库入口 | source archive＋配置 → candidate archive＋draft bundle | 前向生成、探测、选择；未验收不得训练 |
| audit_contact_transition_bank.py | 审计入口 | draft bundle＋审计配置 → 报告／可发布 bundle | GPU 恢复检查、跨条件复验、文件完整性 |

共同约束：

- utils/contact_transition 的配置、事件状态机、选择与格式验证可在无 Isaac Gym 环境测试。
- Isaac Gym 相关导入放 runtime／入口层，保持 Isaac Gym 先于 torch 的现有要求。
- 不让 bank.py 导入训练入口，也不让纯数据格式模块创建模拟器。
- 入口负责 IO 与进程生命周期；模块负责逻辑，避免把生成算法全写在脚本 main 内。
- 初版不拆成网络服务，不需要数据库、消息队列或在线 RPC。

## 4. 共享教师运行规格：防止“看似同一教师，实际环境不同”

### 4.1 可复用函数与需要的抽取

现有 train_multistate_DQ_teacher.py 中：

- restore_eval_arguments：恢复部分观测／控制参数；
- teacher_config_path：寻找检查点旁的保存 YAML；
- get_predict_point：构建对象与抓取候选输入；
- create_env：环境和 Wrapper；
- Policy／Value：模型定义；
- get_trainer：最终组装与 agent.load。

拟新增以下 API；首版可保留模型定义原位置，通过共享工厂调用，不要求一次搬动整个训练文件：

~~~python
resolve_teacher_runtime(
    checkpoint_path, *,
    source_overrides,
    purpose,                 # full_task_collection | probe | strict_eval
) -> TeacherRuntimeSpec

build_frozen_teacher(
    spec, *, num_envs, device, headless=True
) -> FrozenTeacherRuntime
~~~

让原 get_trainer 可接收已解析参数／spec，或抽出其构建子函数；旧 CLI 无参数调用行为保持不变。新增入口单独解析自身选项，不依赖偷偷改写 sys.argv。

### 4.2 RuntimeSpec 必须记录

- 最终生效 YAML 与 CLI 覆盖；不能只保存 create_env 之前的配置副本；
- Actor history、实际输入宽度、critic time、GFM 参数；
- EKF horizon、history、Q/R、物体观测坐标约定；
- action_scaling、动作限幅、控制频率、rand_control；
- 资产、低层控制器、物体网格与抓取候选数据的身份；
- hold_steps、抬升阈值、100 步时限、任务 Level；
- source checkpoint hash／step、planned_timesteps 和奖励阶段；
- Python／torch 随机种子、实际 skrl 导入路径；
- 物理 dt、低层周期、相关动力学分布签名。

允许覆盖并行数量、设备、headless、输出目录和源轨迹种子。改变动作映射、感知输入、控制频率或资产必须显式记录为不同实验，不能静默兼容。

旧 checkpoint 缺少必要 metadata 时要求提供完整的运行配置；不要静默使用当前默认 YAML。此处是程序报错约束，不要求现在向用户补配置，因为本轮不执行采集。

### 4.3 冻结推理不绕过 scaler

collector 使用 agent.act，沿用 state preprocessor；不得裸调 Policy.compute 替代真实推理路径。

构建时设 eval、torch.no_grad，策略模式默认 deterministic；不调用 record_transition 或 post_interaction，memory=None。采集前后校验策略、scaler 状态未变化。随机参考策略作为可选实验，所有分支共用相同模式并记录。

完整源轨迹采用严格完整起点；probe runtime 则允许恢复 context。后者不是普通 --eval 模式，需要新的受控构建分支，不能绕过现有“评估不采训练 bank”的限制。

## 5. 事件采集与源轨迹缓存

### 5.1 插入点

在 b1z1_base.step 的每次：

~~~python
gym.simulate(sim)
_refresh_sim_tensors()
~~~

之后增加可选只读 observer hook。这里约为 2030–2031 行，张量已经更新。禁用时仅有轻量分支，不进行 CPU 同步和日志逐帧写盘。

高层快照在 post_physics_step 完成、Wrapper 追加当前 raw frame 后、下一次动作前捕获。源回合已经 terminal 的状态可作为诊断结局，不作为未完成目标事件的普通练习起点。

不要通过调用 _compute_low_level_observations 读取特征：该函数会改变 gait／history。观察器只读取当前张量，所需派生量用纯计算获取。

### 5.2 EventBatch 最小字段

~~~python
ContactEvidence:
    finger_mask: uint8[N]       # none/left/right/both；另有 known_mask
    known_mask: bool[N]
    table_support: int8[N]      # unknown/unsupported/supported
    between_fingers: bool[N]
    clearance_m: float32[N]
    relative_twist: float32[N, 6]
    aperture: float32[N, G]
    confidence: float32[N]
    event_id: int16[N]
    event_time_s: float64[N]
~~~

表中均为拟议类型。真实接触来自“净力＋邻近几何＋时序”的证据，不能把净力直接当手指—物体接触对。unknown 独立编码；允许 F→C2，不要求必经 C1。

事件状态机包含去抖、滞回、短期持续检查，阈值通过人工核对的校准集确定。阶段目标若在快照时已完成，拒绝“零时刻成功”的窗口记录。

### 5.3 源缓存策略

- 只对选定 collector env 保存 context，避免给 8192 训练环境常驻大 ring buffer。
- 高层 ring 保存事件前若干步；物理步只存紧凑证据和必要事件时间。
- 成功轨迹以新事件发生为锚点；失败轨迹以首次进入捕获区／首次可信接触为锚点。
- episode 结束后再输出事件片段，避免每个物理 tick 写文件。
- 每个片段记录 trajectory_id、env_id、step、source_kind、source checkpoint、dynamics profile 和控制周期。
- 第一版 source archive 分 chunk 写入，保留失败／取消原因；不要只保存成功样本。

相对速度按明确坐标计算，不能将 arm-frame EKF 的导数直接当世界系速度。单位、四元数顺序、history oldest→newest 都写入 schema。

## 6. ContextBatch：字段、所有权和恢复协议

### 6.1 字段分组

设 N 为批次数，D 为 DOF 数，B 为刚体数，F 为 Actor history 帧数，O 为 raw 单帧宽度。具体维度由运行规格声明。

| 组 | 内容 | 来源／恢复原则 |
|---|---|---|
| physical | robot/object/table root [N,13]；dof_pos／vel [N,D] | root 已含线角速度；有限值、四元数及单位校验 |
| targets | EE cart／RPY、gripper target | 保存真实控制参考，不能重设为当前姿态 |
| controller | actions、last_actions、last_low_actions、commands、动作／命令 history 等实际消费字段 | 显式注册 shape；区分 raw 与 clipped action |
| locomotion | low_obs_history、gait_indices、clock_inputs、gait_wait_timer、is_walking | 恢复时不额外推进 gait |
| sensor_history | 上一次足部接触观测及 age | 仅为过去观测；下一物理刷新替换，不代表施加接触力 |
| table_control | table_commands、table height、vy、yaw rate、rotation flag、rand_change_step、模式计数 | 保存控制相位，不仅是当前 root velocity |
| kinematic_cache | rigid_body_state [N,B,13]、ee_jacobian [N,6,6] | 与物理快照同源，供首帧观察和 IK |
| estimator | EKF poses、relative_times、count，必要 valid／预测缓存 | 保存因果过去；导入后局部重算或校验缓存 |
| wrapper_history | raw_frames [N,F,O]、有效标记 | 由 Wrapper 导出／导入，不保存归一化后的帧 |
| source_progress | progress、hold／奖励历史、事件状态等 | continuation 使用；practice 按新起点清理 |
| provenance | UUID、trajectory／parent、source policy、动力学身份 | 存储与审计；不进入 Actor |

不要保存 simulator handle、指针或不透明 contact solver 状态。不将 force_sensor_tensor／净力缓存直接当作新模拟器当前接触；所需过去观测另设字段和时间语义。

action_data_history_buf／states_history_buf 当前主要用于显示或日志，应审计实际消费者后决定是否需要，不把每个 tensor 都列为 Markov 状态。

沿用旧 bank 的坐标约定 existing_reset_tensor_coordinates_no_env_origin_offset，并用跨 env_id 复制检查确认；不要无依据自动加减 env origin。

### 6.2 拟新增接口

~~~python
codec.capture(
    wrapped_env, env_ids, *,
    provenance,
) -> TransitionContextBatch

codec.restore(
    wrapped_env, env_ids, batch, *,
    protocol,                 # continuation | practice
    now_s,
) -> RestoreReceipt
~~~

RestoreReceipt 至少包含恢复 ID、协议、准备提交状态、历史导入状态、运行签名与失败原因。它不表示物理验收已经通过。

对源码做精确字段注册，不以 getattr 扫描环境复制所有 tensor。全局 global_step_counter、奖励训练阶段、整个进程 RNG 和模拟器时钟不属于单个 env 的可恢复字段。

### 6.3 continuation 与 practice

| 事项 | continuation | practice |
|---|---|---|
| 用途 | 与未中断轨迹配对验收、分支复现 | 正式 SGS 练习 |
| 物理／控制／运动相位 | 恢复 | 恢复 |
| 因果观测历史 | 恢复 | 恢复物理历史，处理任务时间字段 |
| 原 episode elapsed | 恢复 | 重置为 0 |
| hold／成功锁存 | 恢复供等价性比较 | 重新开始，不继承接近成功进度 |
| 距离／高度奖励历史 | 恢复 | 以新起点初始化，避免虚假奖励 |
| 剩余时限 | 原回合剩余 | 新的 100 高层步 |
| SGS 结算 | collector 不写 SGS | 自然结束后只写最终 hold |

continuation 验收不等于 practice 任务等价性；重给 100 步改变了局部练习的有限时域条件，所有候选方法使用同一口径。

### 6.4 三处必要的上下文修复

**低层历史初始化。** 当前 _compute_low_level_observations 以 progress_buf < 1 判断首帧。新增 low_history_needs_init: bool[N]，冷 reset 为 true，上下文恢复为 false。否则 practice 把 progress 清零后，刚恢复的 history 会被下一步重新铺平。

**EKF 时间。** 新增 export_history(env_ids)、import_history(env_ids, state, now)、predict(env_ids=None)。将原时间映射为：

$$
\tau'_j=t_{\mathrm{now}}+(\tau_j-\tau_{\mathrm{last}}).
$$

保持 float64 时间，保留 count；不重复 push 同一 timestamp。只重算选中 env，不能由一次局部 reset 改写所有环境的预测。

**Wrapper history。** 给 ObservationHistory／Wrapper 增加 subset export/import；导入后不能再次被 wrapper.reset 的 reset_mask 广播为当前帧。用一次性的已恢复 history mask 区分 cold 与 context reset。不得原地改已返回给 learner 的 states／next_states。

当前每个 raw frame 都含 critic time；Policy 逐帧删除该通道，Value 只消费最新帧且保留 time。明确导入规则：continuation 保留原时间；practice 将导入历史的 time 列统一置 0，并用重算的新起点 raw observation 替换最后一帧，其余真实物理历史保持。源时间只留 provenance。这样最新帧对应新回合进度，且不会把未来改成消费多帧的 Critic 与旧回合时间意外耦合。

首版按实际 F=1 验证，但格式要显式检查 F；如果尚未实现多帧恢复，应明确拒绝 F>1 bundle，不能静默清零。

### 6.5 恢复提交顺序

1. 按自然完成语义结算旧 SGS 回合；collector 的直接恢复不走此结算。
2. 获取新 ResetBatch，保存该回合 immutable source identity。
3. 清理本次任务进度；按 restore_kind 分派 cold 或 context。
4. 写 root／DOF／控制目标、桌面和步态上下文。
5. 填入同源 kinematic cache，重算确定性派生量。
6. 标记 pending root／DOF，沿用现有 table-control 批量提交机制。
7. 导入 EKF，局部生成 raw 当前观测。
8. 在 Wrapper 层恢复／重建 raw history，返回新 observation。
9. 下一次正常 step 的第一个物理周期提交状态并刷新测量。

reset 内没有额外 simulate，不做全局等待。若离线审计发现必须预热，只能显式把预热后状态定义为新候选并重新探测，计入成本；不能在训练异步 reset 中偷偷预热。

## 7. 动力学与分支探测

### 7.1 两层分支

**进入层。** 从合法父快照执行受限前缀，真实生成不同相对运动的前驱。初始建议 1–3 个高层步，可配置扩展；禁止独立随机赋予速度后拼接状态。

**窗口层。** 从同一候选按闭爪延迟和跟踪偏置探测，外加完全未干预策略分支。起始可用 delay_steps=[0,1,2,3]、2 种跟踪偏置、2 次粗重复、8 高层步 horizon。少量重复仅用于筛查，关键窗口需独立复验。

开始探测前确认目标事件尚未完成；已有 C2 的快照不进入“形成 C2”窗口池。事件必须在干预开始后新发生并通过持续检查。C2→L 需要另设目标与动作族；失败源轨迹仍按捕获区进入／首次可信接触锚点采集，不要求其曾成功转换。

每个探测批次按**候选数量**计费。32 候选的 4×2×2×8 干预粗探测为 4096 个单环境高层动作步，另计未干预对照、进入前缀、恢复与审计。

### 7.2 夹爪等待不是 a6=0

set_gripper 使用绝对开度：

$$
q_{\mathrm{target}}=q_{\mathrm{closed}}
+\frac{\mathrm{clip}(a_6,-1,1)+1}{2}
(q_{\mathrm{open}}-q_{\mathrm{closed}}).
$$

因此等待闭爪时应保持保存的目标，可从一致的原执行 a6 获取，或反算：

$$
a_{\mathrm{hold}}=
2\frac{q_{\mathrm{target}}-q_{\mathrm{closed}}}
{q_{\mathrm{open}}-q_{\mathrm{closed}}}-1.
$$

多夹爪 DOF 的反算值须一致；不能偷偷取均值。a6=0 是半开，不是保持。C1 的等待不得主动张爪；释放重抓是独立动作族。

拟新增 ActionIntervention 只复制并修改动作，仍经原 pre_physics_step 映射。位置偏置用米、角度用弧度声明后转为 normalized action 单位；记录 raw、clipped unit、最终目标，不能将 0.01 m 直接加到单位动作。

### 7.3 控制周期与随机数

现有每个高层 step 的 7／8 低层周期由 Python random 选择，且该步全环境共享。第一版增加可选、运行级 ControlPeriodSchedule：

- collector 配对分支共用预生成周期序列；
- 普通训练不开启 override，保持原随机行为；
- 同时保存实际 elapsed_s，不以固定 0.15 s 替代所有时长；
- 确认动作噪声、观测噪声等实际随机源；仅恢复 torch RNG 不足以配对。

在 probe 内同批复制候选需要**相同物理参数的环境副本**。本项目创建环境时的 shape 摩擦和夹爪增益可能不同；不能将同一快照复制到任意 env_id 就称为配对。

### 7.4 与原训练 DR 的衔接

将两种验证区分：

1. **等参数续跑验证**：源环境与恢复副本匹配动力学，诊断 codec 本身的误差。
2. **原分布兼容验证**：固定一组独立抽样的原训练动力学配置，检查候选在不同参数下的恢复和练习有效性。

保存相关每个 rigid shape 的摩擦、夹爪增益等身份，不能只记录一个“环境摩擦”。candidate 的 source dynamics 只是来源，不是训练时应重新设置的属性。

正式训练继续使用原环境 DR，不按 candidate 覆盖参数，也不在全局 SGS 抽样后暗中按隐藏参数拒绝重抽。否则会改变有效采样分布并产生起点—参数关联。所有候选生成基线共享兼容验证集合与成本口径。

若候选只在来源参数下有效，不进入宣称保持原 DR 的主实验。可另做明确标记的固定动力学可行性实验，但不能与原任务混报。

### 7.5 窗口证据结构

每个候选记录：

~~~python
ProbeEvidence:
    candidate_id
    target_event
    action_family_version
    delay_steps
    delay_times_s
    tracking_biases
    successes_and_trials       # 每个组合独立计数
    natural_policy_trials
    local_event_time
    full_hold_result           # 若进行独立长续跑
    contact_confidence
    restore_audit_id
    policy_hash
    source_dynamics_id
    control_schedule_id
    compute_cost
~~~

窗口可不连通，保存离散掩码和计数，不自动压成一个区间。全部探测失败记“未发现成功”，不记物理不可达。局部事件和 T_loss 不写正式 episode done／reward／SGS。

## 8. 组合式 bank：不改旧 schema，新增统一 ResetSource

### 8.1 为什么不能直接追加一个旧 profile

当前 ConfigurationBank.validate 将 SUPPORTED_FAMILY_PROFILES 中的 profile 视为几何 bank，执行旧脚部、near_goal 等校验；validate_task_hold_steps 对已知几何 profile 有 15→10 兼容规则。

因此，新增动态 profile 若直接塞进该集合，会错误继承旧证据与 hold 兼容性。首版采用组合格式及显式分派，不修改 ConfigurationBank.SCHEMA_VERSION=1。

### 8.2 Bundle 文件布局

| 文件 | 内容 |
|---|---|
| manifest.json | format/version、active row map、签名、hash、校验状态 |
| anchors.pt | 原 v12 全文件原样复制；保留原 fingerprint 与所有静态证据 |
| transitions.pt | 动态 physical＋typed context tensor 字典；不含任意 Python 对象 |
| evidence.jsonl | 窗口分支与来源、验证证据 |
| runtime.json | 解析后的教师运行规格、控制与资产身份 |
| audit.json | 恢复、隔离、跨参数验证与成本摘要 |

保存完整 anchors.pt 并不增加运行时容量。manifest 只启用其中 1152 行：

- 512 full_task；
- 512 near_goal；
- 128 pregrasp。

再加 384 动态行，**active 总容量仍为 1536**。运行时三组为 full_task=512、near_goal=512、curriculum=128+384。stored rows 与 active rows 分别记录，防止公平性误报。

### 8.3 Manifest 示意

以下只展示关键结构，省略完整 row 列表和校验数值：

~~~json
{
  "format": "dq_contact_transition_bundle",
  "schema_version": 1,
  "bundle_version": "contact_window_v1",
  "active_count": 1536,
  "family_names": ["full_task", "near_goal", "curriculum"],
  "family_counts": [512, 512, 512],
  "task": {"object": "sugar_box", "hold_steps": 10, "episode_steps": 100},
  "context_schema": "b1z1_context_v1",
  "validation": {"status": "draft", "audit_file": "audit.json"},
  "files": {
    "anchors": {"path": "anchors.pt", "sha256": "<computed>"},
    "transitions": {"path": "transitions.pt", "sha256": "<computed>"},
    "runtime": {"path": "runtime.json", "sha256": "<computed>"},
    "evidence": {"path": "evidence.jsonl", "sha256": "<computed>"},
    "audit": {"path": "audit.json", "sha256": "<filled-after-audit>"}
  },
  "rows": [
    {
      "slot": 0,
      "candidate_id": "<stable-id>",
      "family": "full_task",
      "storage": "anchors",
      "row": 0,
      "restore_kind": "legacy_cold"
    }
  ],
  "bundle_fingerprint": "<computed>"
}
~~~

每个 slot 必须恰好对应一行，不能遗漏、重复、越界。UUID／稳定身份保存在 manifest 字符串中；GPU 热路径使用 dense slot，不以 float 编码 ID。

动态 context 支持 int64／int16／uint8／bool 等明确 dtype，单独校验。任务 hold_steps 对动态证据严格匹配；anchors 的 15→10 兼容只沿用原静态 bank 规则，并单独记录，不推广到整个 bundle。

### 8.4 Fingerprint 与发布

- 文件 hash 包含实际字节；manifest 使用规范化 JSON、保留 rows 顺序计算 fingerprint，排除 fingerprint 自身字段。
- audit.json 记录 subject_digest，绑定实际被审的 physical/context、runtime、row map、任务／控制签名与校准配置。subject_digest 不含 audit 自身和最终 fingerprint；最终 manifest 再绑定 audit 文件 hash，避免循环摘要。draft 中未完成的 audit 字段不得被当成有效证据。
- 路径只允许 bundle 内的规范相对路径；验证文件存在、hash、shape、类型与 count 后再分配 GPU。
- 首版采用“最后提交 manifest”协议：build 在全新输出目录写入完整成员和 draft manifest；audit 写完审计报告并核验全部成员，最后在同一文件系统原子替换 manifest 为 passed，作为唯一发布提交点。提交前不存在可训练的 passed manifest；失败保留 draft／failed 与诊断，不额外采用另一套目录改名协议。
- 训练拒绝 draft／failed／部分文件缺失的 bundle。
- 禁止在发布后原位覆盖 transitions 或 rows；新内容生成新版本和 fingerprint。
- anchors.pt 本身保持原 ConfigurationBank 校验，不因为 bundle 的 passed 字段就跳过。
- 保存当前受支持的张量／JSON格式，不引入任意对象反序列化。

### 8.5 ResetSource 统一协议

~~~python
class ResetSource:
    num_configs: int
    family_names: tuple[str, ...]
    family_ids: Tensor         # int64，dense slot 对应
    fingerprint: str
    task_signature: dict

    def batch(self, slots: Tensor) -> ResetBatch: ...
    def diagnostics_metadata(self) -> dict: ...

class ResetBatch:
    physical: dict[str, Tensor]
    context: dict[str, Tensor] | None
    context_row_indices: Tensor  # 动态子批对应physical批次的位置，int64
    restore_kind: Tensor
    source_ids: Tensor         # 到manifest稳定身份的索引
~~~

实现 LegacyBankSource 包装原 ConfigurationBank；TransitionBundleSource 组合 anchors＋动态数据。load_reset_source 根据格式标记分派，不能让任意 JSON 都成为合法 bank。

batch(slots) 保持输入顺序并支持重复 ID。physical／restore_kind／source_ids 长度均为请求批次大小；context 只存动态子批时，必须用 context_row_indices 显式对齐，不能按内部排序猜测对应关系。纯 legacy 批次的该索引为空。

在环境中逐步将 self.sgs_bank 的直接结构读取替换为 self.reset_source；SGSSampler 不改。无需用一个伪装的 tensors 字典模拟所有旧字段。

替换位置不仅是 _init_sgs 和 reset，还包括：

- family 配额检查；
- 旧回合所属 family；
- _record_sgs_metrics；
- save_sgs_diagnostics；
- 完整任务 success_window 的 full_task 过滤，约 633 行。

最后一项是完整任务结果统计，不是新的奖励。遗漏会使新 bundle 的训练日志失真。

### 8.6 混合恢复的实验约束

anchors 为 legacy_cold，动态行为状态为 context，必须逐行标注。

B2 普通轨迹回收、B3 随机分支、O1 窗口生成的可变槽位都采用相同 context 恢复。B1 几何候选需经过标准前向过程获得真实上下文后进入共同流程；或单独组织所有方法均为 cold 的实验。

不能只让新方法带完整历史，而让基线全部清零，然后把收益全归给时序窗口。

## 9. 候选选择与离线产物

第一版使用可解释规则：

1. 剔除格式错误、恢复失败、目标事件已完成、动作族不兼容的记录。
2. 将物理有效候选分为容易、存在成功窗口但自然策略错过、未知前沿。
3. 384 个动态槽位暂按 96／192／96 配额；该值配置化、需消融。
4. 每层按事件、入射方向、相对速度分桶；限制相邻帧重复，量纲归一化后选覆盖。
5. 不足时明确减少动态有效数或回填已验 anchor，并生成新的合法 row map；记录实际比例。
6. 新方法与基线共享总容量、anchors、恢复标准、审计配置及成本账本。

导出时保存 selection_config_hash 和每个被拒绝候选的原因计数。筛选器不能在读取最终测试集成绩后改变阈值。

原设计中的“能力差距”仅用于候选诊断／分层；不将探测成功率写入 SGSSampler.observe。新候选开始时历史为空，不继承来源状态或被替换槽位的成绩。

## 10. 正式训练接入与异步 reset

### 10.1 训练时序保持现状

~~~python
states = env.reset()
for timestep in training_loop:
    actions = agent.act(states)
    next_states, rewards, terminated, truncated, infos = env.step(actions)

    agent.record_transition(
        states, actions, rewards, next_states, terminated, truncated, infos
    )
    agent.post_interaction()  # 可能在24步边界更新PPO

    if any_done:
        # env内部只重置reset_buf对应子集，先结算旧ID
        states = env.reset()
    else:
        states = next_states
~~~

新代码只扩展 env.reset 的来源加载／恢复分派，不能把自动 reset 移入 env.step。100 步回合会跨越多个 24 步 rollout，不能每次 PPO 更新重置所有环境。

### 10.2 统计隔离

| 数据 | 接收方 | 不允许写入 |
|---|---|---|
| 当前策略从正式 reset 采集的新 rollout | PPO memory | 不用旧探测轨迹补齐 |
| 正式回合最终 hold | SGSSampler.observe | 不用局部转换标签代替 |
| 窗口／接触／错过诊断 | 独立 event archive、metrics | 不写 reward、episode_success、done |
| source／probe 的中断与取消 | collector 结局与成本日志 | 不记 SGS 失败 |
| 完整任务严格评估 | eval 输出 | 不回写课程最终测试选择 |

不可变 bank 首版仍可用 active_config_id 作为 dense slot，但日志和检查点 metadata 同时保存 bundle fingerprint 与 row→candidate 身份映射。后续若做 resume，必须检查映射一致；仅相同 num_configs 不够。

### 10.3 不扩大首版修改面

- 原 SGS 公式、history size 和 PPO 超参数保持。
- 暂不增加 full_task 最低采样混合；若后续增加，所有基线共同启用并单独消融。
- 不给 collector 配置写入默认 teacher YAML。
- 新特性默认关闭；指定旧 .pt 完全走 LegacyBankSource＋原 cold reset。
- 生成器动作偏置只作用于 collector，部署策略接口不变。

## 11. 配置、命令与运行产物

### 11.1 拟新增配置示例

路径：DQ_high-level/data/cfg/contact_transition.yaml。下例是配置草案，非已经可运行的文件。

~~~yaml
version: 1
target_event: form_bilateral_contact
restore_contract: context
practice_episode_steps: 100
reference_policy_mode: deterministic

collection:
  source_kind: full_task
  ring_high_steps: 8
  capture_before_steps: [1, 2, 4]
  include_failed_approaches: true

generation:
  prefix_high_steps: [1, 2, 3]
  probe_batch_candidates: 32
  closure_delay_steps: [0, 1, 2, 3]
  tracking_bias_variants: 2
  coarse_repeats: 2
  horizon_high_steps: 8
  include_unmodified_policy: true
  replay_control_periods: true

selection:
  active_capacity: 1536
  full_task: 512
  near_goal: 512
  retained_pregrasp: 128
  generated: 384
  generated_strata: [96, 192, 96]

audit:
  require_continuation_check: true
  require_nonreset_env_isolation: true
  require_cross_dynamics_check: true
  require_frozen_policy_and_scalers: true
~~~

接触阈值、恢复容忍误差和复验重复数由校准报告提供并纳入配置签名，不在这里任意填写“看起来合理”的精度。配置缺失这些必需字段时，正式 audit 不得通过。

### 11.2 拟新增入口的执行顺序

以下命令是实现后的操作模板，checkpoint 路径需换成实际文件；本轮没有执行。

~~~bash
cd /home/hehui/DQ_WBC-1888ce7/DQ_high-level

python collect_contact_transitions.py \
  --checkpoint /ABS/PATH/agent_STEP.pt \
  --transition_config data/cfg/contact_transition.yaml \
  --num_envs 128 --episodes 512 \
  --output data/contact_transition/source_v1

python build_contact_transition_bank.py \
  --source_archive data/contact_transition/source_v1 \
  --anchor_bank data/sgs/sugar_box_v12.pt \
  --transition_config data/cfg/contact_transition.yaml \
  --output data/contact_transition/bundle_v1

python audit_contact_transition_bank.py \
  --bundle data/contact_transition/bundle_v1/manifest.json \
  --transition_config data/cfg/contact_transition.yaml \
  --finalize
~~~

--finalize 是**拟新增**操作：仅所有必需 audit 通过时完成发布，并拒绝覆写已发布版本。重复审计应生成独立报告或新版本，不能静默改变训练中的 fingerprint。

建库／审计入口从 runtime.json 定位源 checkpoint 并验证 hash；源 checkpoint 路径不再可用时要求显式提供匹配文件，不能悄悄换教师。

### 11.3 接入原训练入口

完成 loader 改造并通过验收后，沿用现有参数：

~~~bash
SGS_BANK=data/contact_transition/bundle_v1/manifest.json \
RUN_NAME=contact_window_v1_seed43 \
bash train_teacher_sgs_ekf.sh --no-wandb
~~~

现有 shell 和 config 检查的是文件，manifest.json 满足这一入口形态；真正的格式验证由新 loader 完成。

小规模 smoke 可使用现有 NUM_ENVS 和 --stop_after，保留 planned TIMESTEPS=120000 以免改变奖励阶段。不能把 120000 改成几十步，再把所得行为解释为原训练的前几十步。

### 11.4 完整任务评价沿用现有入口

~~~bash
python eval_sgs_teacher.py \
  --checkpoint /ABS/PATH/NEW_RUN/checkpoints/agent_STEP.pt \
  --num_envs 256 --eval_episodes 1024 \
  --seed 43 --headless \
  --eval_output /ABS/PATH/eval_contact_window_v1.json
~~~

这是现有评估入口的模板；新运行的保存配置应能恢复。比较时各方法固定种子、环境数、episodes 和感知设置。不要加普通 --eval 改成近距离起步。

输出新增过程诊断可放旁侧文件；原 hold_success_rate 等字段保持，以便旧评估脚本继续消费。

## 12. 逐文件修改清单

| 文件 | 必须改什么 | 默认／旧路径保护 |
|---|---|---|
| train_multistate_DQ_teacher.py | 抽取运行规格解析／冻结模型构建，允许注入已解析 spec | 旧 get_trainer CLI 与网络 shape 不变 |
| utils/teacher_inference.py（新） | 推理 runtime 与签名、checkpoint/scaler 固定 | 不构建训练 memory |
| envs/b1z1_base.py | 可选物理步 observer；history 初始化标记；运行级控制周期 override | 未启用 observer／schedule 时沿用旧行为 |
| envs/b1z1_pickmulti.py | load_reset_source；family读取统一；恢复分派与Receipt | legacy cold 路径和自然完成结算保留 |
| modules/pose_prediction.py | 子集导出／导入、时间重定位、predict(env_ids) | predict() 无参数继续等价旧接口 |
| utils/observation_history.py | subset export/import，避免原地污染历史输出 | 原 update 行为不变 |
| utils/wrapper.py | 导入 history mask、context restore 协调 | cold reset 继续平铺首帧 |
| utils/contact_transition/*.py（新） | 事件、codec、probe、selector、bundle | 无 GPU 的子模块可独立导入 |
| utils/sgs.py | 原 schema 和 sampler 保持；必要时只抽取公共校验 | 不扩充旧几何 profile 集合来接动态态 |
| utils/config.py | 如需，增加 observer/runtime 配置入口和 source诊断元数据 | --sgs_bank 文件校验、strict eval 和 SGS resume 保护保留 |
| eval_sgs_teacher.py | 可选附加 contact observer／报告，不修改主统计 | 不采 bank，不更新 scaler |
| 新 collector/build/audit 入口 | 独立 CLI、成本账本、输出版本 | 不更改原 build_sgs_bank.py 的默认行为 |

可能后续需要但不在首版内：checkpoint 全训练状态分叉、在线 bank manager、条件 sampler。不要为了让方案“完整”而提前改这些模块。

## 13. 验收测试与运行门槛

本轮仅写方案；以下是后续实现需要运行的测试，不是已经通过的训练测试。

### 13.1 CPU 单元与契约测试

| 测试 | 要验证的实际风险 |
|---|---|
| test_contact_transition_events.py | 去抖、unknown、F→C2、事件未完成资格、已有C2不产生T=0假成功 |
| test_transition_action_intervention.py | a6=0不是保持，多DOF一致性，normalized单位转换、原动作不被原位修改 |
| test_transition_context_schema.py | shape/dtype、NaN、count、history顺序、缺失字段、运行签名不兼容 |
| test_transition_ekf_restore.py | 时间重定位、重复timestamp、subset predict不修改其他env |
| test_transition_history.py | 冷reset与上下文导入、已交给learner的tensor不变、critic time协议 |
| test_transition_bundle.py | 文件hash、row唯一性、1152+384、draft拒绝、旧schema兼容 |
| test_transition_selection.py | 固定配额、轨迹去重、空桶回填、输出顺序与seed可复现 |
| test_transition_reset_protocol.py | 旧ID只结算一次，直接恢复为取消，24/100跨界不截断 |

复用现有 test_sgs.py、test_sgs_env.py、test_actor_history.py、test_action_scaling.py、test_teacher_entry.py、test_teacher_success_audit.py、test_teacher_rewards.py。保持原测试断言的语义，不通过放宽它们掩盖新恢复差异。

补充两项边界用例：混合 cold/context 且含重复 slot 的 gather 必须对齐；修改过去帧的 critic time 不应改变当前 Policy／Value 输出，最新帧的时间通道仅进入 Value、不进入 Policy。

### 13.2 GPU 验收按顺序执行

1. **冻结推理一致性**：同 checkpoint、同 raw observation，新 runtime 与旧评估路径动作一致，scaler 不变。
2. **观察器无侵入**：同 seed、同动作，observer 开／关不改变环境状态或最终结果。
3. **同环境续跑**：开环记录动作、闭环教师两种续接下，restore 与未中断分支达到预定误差标准。
4. **跨相同参数环境续跑**：验证 env_id、坐标、缓存和 DR 配对。
5. **异步隔离**：恢复少量 env，其余 env 的物理状态、history、EKF 和未提交 observation 不变；reset 本身不 simulate。
6. **实践计数**：practice reset 给100步、hold从0开始，history不因progress=0被覆盖；无即时成功。
7. **原DR兼容**：所有候选方法在相同独立动力学样本集复验，记录接受率与失败类型。
8. **训练 smoke**：跑过多个24步rollout及100步回合边界，terminal observation不被新起点覆盖，SGS回合计数一致。
9. **旧bank回归**：v12默认启动与加载路径继续运行，原始完整任务评估口径不变。

PhysX 内部状态可能无法精确克隆。数值误差、事件顺序与短续跑成功率的容忍范围应先在校准数据确定；不能以 bitwise equality 作为所有物理对照的唯一条件，也不能遇到差异后无限放宽阈值。

### 13.3 性能验收

记录 observer／capture／restore 每步耗时、GPU显存、磁盘体积、候选接受率、全部高层和物理步、CPU/GPU墙钟。

事件每个物理步采集不意味着每个物理步复制完整 context。先计算实际字段内存估算：

$$
\text{buffer bytes}
=
N_{\mathrm{tracked}}\times K_{\mathrm{ring}}
\times\sum_f \bigl(\mathrm{numel}(f)\,\mathrm{sizeof}(\mathrm{dtype}_f)\bigr).
$$

初始化前打印预算并校验上限；复用批量张量，避免热路径逐候选 .cpu()／.item()。不要为诊断让8192环境每步传完整状态到CPU。

## 14. 分阶段开发与依赖顺序

| 阶段／建议PR | 范围 | 可审查产物 | 退出标准 |
|---|---|---|---|
| P0：运行规格与诊断 | shared inference、只读 observer、源片段缓存 | 当前教师失败分解＋少量片段 | 冻结推理一致；observer无侵入；确认值得研究接触 |
| P1：上下文恢复 | codec、低层history标记、EKF、Wrapper | 配对续跑报告 | 同环境、跨相同参数与异步隔离通过 |
| P2：生成与探测 | 两层shooting、ActionIntervention、周期schedule | 带自然策略对照的窗口证据 | 窗口可重复；无T=0假成功；完整成本可追踪 |
| P3：冻结bundle | 格式、selector、loader、manifest审计 | 同容量可发布bundle | hash、来源、恢复协议、跨DR审计通过 |
| P4：训练接入 | ResetSource分派、统计／日志适配 | v12回归＋新bundle smoke | 24/100时序正确；旧ID只结算一次 |
| P5：方法实验 | B0/B1/B2/B3/O1、严格完整任务评价 | 曲线、逐种子结果、同池消融 | 收益超过简单回收，计入成本仍成立 |

P0、格式的纯CPU定义可部分并行；P2依赖P1通过，P4不能提前绕过审计。HTML 架构图中将在线更新和效用估计列为后续，不连入首版必经路径。

建议先完成 P0＋P1 的最小纵向切片：**一个检查点、一类事件、少量环境、一个快照的可靠捕获与续跑**。若恢复契约不成立，先修复它，不开始大规模候选筛选和多种子训练。

## 15. 实验交付标准与回退

主比较至少包含：

- B0：原 v12＋SGS；
- B1：同容量接触覆盖；
- B2：普通轨迹／失败前回收；
- B3：同预算随机前向分支；
- O1：窗口定向生成。

B1–O1 的可变槽位使用共同恢复契约；候选来源教师的训练成本或共享先验必须披露。用同一候选池做几何→速度→窗口／策略错位的信息消融，区分生成和选择的作用。

主指标仍是关闭 SGS、从完整起点开始的 hold_success_rate；另报失败惩罚时间、接触后闭合、离桌和滑脱。生成、校准、审计和课程 probe 都计入成本。

若出现以下情况，停止扩大系统：

- 主要失败不在接触，改做接近或感知诊断；
- 只能在来源动力学下恢复，先解决兼容性；
- 窗口短于现有控制分辨率，检查动作接口；
- 普通回收已经同样好，保留更简单实现；
- 只提高局部reset成功，不能宣称完整任务改进。

功能回退始终是指定原 sugar_box_v12.pt、关闭新增 observer／schedule，使用原训练及评估入口。新格式、实验产物和日志保留，不覆写原 bank。

## 附录：本轮重点核对的源码

- [训练入口](../DQ_high-level/train_teacher_sgs_ekf.sh)
- [教师组装与模型](../DQ_high-level/train_multistate_DQ_teacher.py)
- [配置检查](../DQ_high-level/utils/config.py)
- [SGS 与原 bank](../DQ_high-level/utils/sgs.py)
- [几何建库与冷恢复](../DQ_high-level/utils/sgs_bank_builder.py)
- [抓取任务 reset／统计](../DQ_high-level/envs/b1z1_pickmulti.py)
- [物理步、控制器与 deferred setter](../DQ_high-level/envs/b1z1_base.py)
- [EKF 历史](../DQ_high-level/modules/pose_prediction.py)
- [Wrapper](../DQ_high-level/utils/wrapper.py)
- [Raw observation history](../DQ_high-level/utils/observation_history.py)
- [完整任务评估](../DQ_high-level/eval_sgs_teacher.py)
- [仓库内 trainer 顺序](../third_party/skrl/skrl/trainers/torch/base.py)
- [仓库内 PPO](../third_party/skrl/skrl/agents/torch/ppo/ppo.py)

