# dynamic_papers 论文借鉴分析：面向 SGS+EKF 教师训练

> 2026-11-13。对象：[`train_teacher_sgs_ekf.sh`](../DQ_high-level/train_teacher_sgs_ekf.sh) 启动的 sugar_box 教师（PPO + SGS 初态采样 + EKF 未来位姿观测）。
> 论文目录：[`doc/papers/dynamic_papers`](papers/dynamic_papers)，共 9 篇，逐篇读了摘要、方法、关键实验与局限。
> 本文只做分析和建议，**没有改代码，也没有跑实验**；下文的收益判断都是待验证的假设。

---

## 0. 先对齐：当前教师在做什么

| 方面 | 现状（据代码核对） | 和论文对照时的关注点 |
|---|---|---|
| 任务 | B1+Z1 在 Level02 运动台面上抓 sugar_box，抬升并保持 `hold_steps`，最长 100 个高层步 | 动态物体 + 抓取接触，奖励稀疏（`pick_up: 50`） |
| 算法 | skrl PPO（TSIL 分支另有 `TSILPPO`） | 无模型、on-policy |
| 初态 | SGS：按配置近期成功率 `p` 加权，`w ∝ p^0.5(1-p)^0.5`，温度 2，ε 下限；**没有显式的均匀采样混合** | 对照 SuperMimic 的 reset 混合 |
| 物体运动信息 | `--no_velocity_obs` 去掉了速度；EKF 用最近 5 帧位姿外推 0.6/0.9 s，共 13 维（含 valid），12 维进 GFM Query | 对照“动作条件预测”“接触感知预测” |
| EKF 坐标系 | **在臂基坐标系 A 中拟合相对运动**（[`b1z1_pickmulti.py:766`](../DQ_high-level/envs/b1z1_pickmulti.py)，注释写明“包括机器人自身的平移和旋转”） | 见 §3 建议 B，这是本次最具体的发现 |
| 动作 | 9 维：EE 位置/姿态为**增量**（每步 ≤0.02 m / 0.06 rad），夹爪连续，底座 vx / yaw 为**直接速度指令** | 对照 ODE Policy |
| 接触信息 | 环境已有 `_contact_forces`（net contact force tensor），但没进观测 | 对照三篇触觉/接触论文 |
| 物理随机化 | 只作用于**机器人**：躯干附加质量 0–15 kg 与质心偏移、机器人刚体摩擦 0.5–3.0 分桶（[`b1z1_base.py:244`](../DQ_high-level/envs/b1z1_base.py)、`:876`）；物体质量/摩擦未见随机化；参数都不进观测 | 对照 AdaptiGraph / OrbiSim 的参数条件化 |

---

## 1. 一句话结论

9 篇里**能直接用在 PPO 教师上的**主要是三类思路，按投入产出排序：

1. **自监督“预测未来”的辅助损失**（Enhancing Tactile-based RL）：不改观测接口，就让编码器学会物体运动。最适合 `--no_velocity_obs` 的设置。
2. **把预测做“对”**（Contact-Aware Neural Dynamics、Act-FIGNet、AdaptiGraph 的共同点：预测要以动作/物理量为条件）：当前 EKF 在 A 系拟合相对运动，把机器人自身受策略控制的运动也当成匀速外推。真机只有相对位姿，因此不改坐标系，而是用 IMU 陀螺仪（以及可得的速度估计）把窗口内的历史帧补偿到当前基座坐标系后再拟合。
3. **接触信号与 reset 分布的小改动**：二值夹爪接触作为 critic 特权观测或辅助目标；SGS 加一部分均匀采样混合（SuperMimic D.1）。

**世界模型 + BPTT 替代 PPO**（SuperMimic / OrbiSim / DayDreamer / ODE Policy 的 RL 部分）收益最大，但和本任务的抓取接触、冻结底层控制器、稀疏成功奖励都冲突，SuperMimic 自己也把“物体交互”列为未做的局限，**现阶段不建议**。ForeTac、Act-FIGNet 的推理期引导/MPC 与 PPO 教师关系很弱。

---

## 2. 逐篇分析

### 2.1 SuperMimic（CoRL 2026，UC Berkeley）——可微世界模型训练人形全身控制

**核心思路。** 用策略自己采的仿真数据在线拟合一个确定性、可微的世界模型 `s_{t+1}=f(s_t,a_t,ξ)`（ξ 是每回合的域随机化参数），再冻结模型，从 replay 状态出发想象 H=32 步，**沿模型反传（BPTT）直接求回报对动作的解析梯度**，critic 对 H 步之外做自举。

**关键做法。**
- 模型条件于 sim context ξ（摩擦、标定偏差、质心），因此可以保持确定性；观测噪声在模型外单独加。
- 模型用 k=8 步自回归窗口训练，使其能纠正自身误差，想象 32 步仍稳定。
- 终止用可微的“生存权重” `c_{t+1}=c_t(1-\hat d_{t+1})`。
- 有 critic 时 H=16 即可，无 critic 时 H≤16 训练失败；无 critic 的策略被推后无法恢复。
- **附录 D.1 的 reset 采样**：动作时间轴分 1 s bin，每个 bin 维护折扣回报估计 `G_b`；reset 时 **50% 均匀 + 40% `softmax(-25 G_b/G_max)`（偏向低回报）+ 10% 按历史 softmax 权重**，让学会的难段仍有练习。

**结果。** 单卡 3 h 超过 128 卡训练 7 天的 SONIC，GPU 时减少 >1000×；PPO 在同一世界模型上需要约 64× 更多想象步。

**和本任务的差距。** 动作追踪奖励每步都很密集，训练稳定；本任务的成功奖励稀疏，且抓取是不连续接触，模型梯度在接触处不可靠。论文明确写了 “No terrain or object interaction … is ongoing work”。另外本任务的高层“动力学”包括冻结的底层运动控制器，可以学，但要额外建模。

**可借鉴。**
- ✅ **reset 混合分布**（低成本，见 §3 建议 D）。
- ✅ **critic 条件于 sim context ξ**：教师 critic 本就可用特权信息，把机器人摩擦/附加质量、台面 `vy`/`yaw_rate` 喂给 critic（见 §3 建议 E）。
- ⏸ 整套 BPTT：长期方向，当前不建议。

---

### 2.2 DayDreamer（CoRL 2022）——在真实机器人上在线训练 Dreamer

**核心思路。** 不用仿真，直接在 4 台真机上跑 DreamerV2：RSSM 潜空间世界模型 + 想象中的 actor-critic，数据采集与学习异步并行。A1 四足 1 小时学会翻身站立行走；UR5/XArm 用图像和稀疏奖励学会抓放。

**和本任务的关系。** 它解决的是真机样本效率，而本仓库的教师在 Isaac Gym 中用 8192 个环境训练，样本不是瓶颈。

**可借鉴。**
- 潜空间世界模型能把多模态传感“融合成状态”，从而减少手写状态估计器。这对**学生/真机**阶段有启发（用学习到的潜状态替代或补充 EKF），对教师意义不大。
- 结论：**借鉴价值低**，仅作世界模型路线的背景。

---

### 2.3 AdaptiGraph（RSS 2024，UIUC）——物理参数条件化的图神经动力学

**核心思路。** 把物体表示为粒子图，用 GNN 预测粒子运动；**在每个粒子上拼接材质类型和连续物理参数变量**（绳/布刚度、颗粒大小、刚体压力中心），一个模型覆盖多种材质。部署时让机器人主动交互（MPPI 选择使预测与观测分歧最大的“好奇”动作），用 BO/CMA-ES 最小化预测与观测的 Chamfer 距离，在线反推物理参数，再用 MPPI 规划。

**和本任务的关系。** 本任务是单个刚体，几何固定，不需要 GNN。但“预测模型要以影响运动的物理量为条件，并在线辨识”这条思路适用：sugar_box 在台面上的运动由台面 `vy`、`yaw_rate` 决定，被夹住后由摩擦/质量决定能否抬稳。

**可借鉴。**
- ✅ 教师 critic 接收物理参数（特权），见建议 E。
- ✅ 给学生做“在线辨识运动参数”的适配模块（类似 RMA）：用历史位姿回归台面 `vy`/`yaw_rate`。对 Level02 这种低维参数化运动，这比通用外推更准。
- ❌ 粒子图、主动探索、MPPI：与 PPO 教师无关。

---

### 2.4 Contact-Aware Neural Dynamics（2026，UCSD + Amazon FAR）——用接触信息做隐式 sim-to-real 对齐

**核心思路。** 学一个灵巧手-物体的前向动力学：输入 K 步历史（物体位姿、动作、关节、**二值接触**）+ 物体点云（PointNet）；**两阶段预测**——先用 MLP 预测未来 H 步接触概率（BCE），再把预测接触编码成 64 维特征，作为扩散 UNet 预测未来位姿增量的条件。先在大量仿真数据上训练，再用少量带触觉的真机数据微调。

**关键结论。**
- 加入接触条件后位姿预测误差和 ADD-S 都更好；仿真预训练 + 真机微调远好于只用真机数据（73.7% vs 52.6%）。
- 选**二值、整手级**接触而非连续力：连续力噪声大、标定敏感，二值标签在仿真（碰撞检测）和真机（力阈值）都能一致地得到。
- 局限：长时域在频繁接触切换和快速运动下仍有累积误差。

**和本任务的关系。** 本任务的位姿预测（EKF）只用于抓之前；抓的瞬间物体运动从“跟着台面走”突变为“跟着夹爪走”，恰好是接触引起的模式切换，匀速 EKF 必然在这里失效。

**可借鉴。**
- ✅ **二值夹爪-物体接触**作为特权观测/辅助目标（见建议 C）。Isaac Gym 中可从夹爪手指 link 的接触力或物体接触力直接得到。
- ✅ **“先预测离散事件，再条件化连续预测”** 的结构：辅助损失里可以同时预测“未来是否接触/是否已抬起”（BCE）和未来位姿（回归）。
- ✅ **预测以动作为条件**：EKF 不知道机器人接下来怎么动（见建议 B）。
- ❌ 扩散 UNet 预测器：在 PPO 环路里太重，没有必要。

---

### 2.5 Learning Contact Dynamics through Touching（Act-FIGNet，DFKI 2026）——动作条件 GNN 用于插孔

**核心思路。** 将工具和环境网格建成异构图，**用专门的“力旋量节点”把控制输入显式编码进图**，在物体本体坐标系中预测物体级 twist 和反作用力旋量（扭矩由逐顶点力场派生）。只用关节编码器和 F/T 读数，通过机器人随机触碰环境自监督训练；接入 iCEM-MPC 后在仿真插孔达 98%，真机位置误差比辨识过的 MuJoCo 低 45%。

**和本任务的关系。** 场景是固定环境中的力控插孔，与动态抓取差别很大。

**可借鉴。**
- ✅ 两个通用原则：**预测必须显式以控制输入为条件**；**在本体/局部坐标系中表示以获得旋转等变性**。前者正对应建议 B。
- ❌ 网格 GNN、F/T 观测模型、MPC：不适用。借鉴价值低。

---

### 2.6 ForeTac（2026）——为生成式策略提供预测性接触引导

**核心思路。** represent–predict–evaluate–guide 循环：TacVAE 编码触觉标记场；Foresight Transformer 以候选动作块为条件预测未来 16 步触觉潜变量；**用成功/失败交互训练一个“接触质量”打分器**；推理时把打分对动作的梯度反传，引导冻结的扩散/流匹配策略的采样过程（classifier guidance）。四个真机任务平均成功率从 36.3% 提到 71.3%。

**和本任务的关系。** 依赖触觉传感器和生成式策略的推理期引导，PPO 高斯策略没有对应的“采样过程”可引导。

**可借鉴（间接）。**
- 🔸 **从成功/失败数据学一个“结果预测器”**：TSIL 分支已经把完整轨迹存在 GPU 上，可以顺手训练一个 `P(最终成功 | 当前状态)` 分类器，用法有两种：(a) 作为辅助头共享编码器（属于建议 A 的扩展）；(b) 作为基于势函数的塑形 `γΦ(s')-Φ(s)`，在理论上不改变最优策略。(b) 风险较大，建议只在 (a) 有效后再尝试。
- 借鉴价值：低到中。

---

### 2.7 Enhancing Tactile-based RL（NeurIPS 2025，爱丁堡）——触觉 RL 的自监督辅助目标

**核心思路。** 只用本体感觉历史 + 17 个二值接触训练灵巧手（无视觉、无特权状态）。网络为共享编码器 `e`（1024→512→256）+ 策略头 + 价值头，**在 PPO（skrl 实现）损失之外加自监督辅助损失训练编码器**。比较了四种目标：
- TR：从潜变量重建当前接触（BCE，正样本权重 10，应对稀疏）；
- FR：重建全部观测；
- **FD：多步潜空间前向动力学**——`ẑ_{t+1}=f(z_t,a_t)` 自回归展开 n 步，对齐 EMA 目标编码器的 `z_{t+i}`（经投影头，类似 SPR）；
- TFD：FD + 从预测潜变量解码未来接触。

**关键结论。**
- **FD 最有效**，学到的表征能线性解码出物体位置和**速度**；
- 二值接触在“机器人与物体运动解耦”（低惯量物体、接触位置模糊等）时至关重要，仅靠本体历史学不到；
- **把辅助损失的数据从 on-policy rollout 中分离出来，单独存入更大的缓冲区**（多个 rollout 的数据），能稳定辅助损失并提升性能；
- 代价：训练时间增加，n 越大越慢；每个任务都重新做了超参搜索。

**和本任务的关系：9 篇中最贴近、最容易落地的一篇。**
- 同为 skrl PPO + Isaac 并行环境；
- 教师用了 `--no_velocity_obs`，而且观测没有历史，运动信息只能来自 EKF 那 13 维；
- 动态抓取正是“机器人与物体运动解耦”的典型场景。

**可借鉴。**
- ✅ **建议 A**：给教师编码器加“预测未来”的辅助损失。仿真中真值免费，可直接监督未来真值位姿，比潜空间 FD 更简单。
- ✅ **独立的辅助数据缓冲区**：TSIL 的 GPU 轨迹缓存可以复用。
- ✅ 二值接触 + BCE 正样本加权（建议 C）。

---

### 2.8 ODE Policy（NTU 2026）——学习动作的动力学而非动作本身

**核心思路。** 策略输出动作的向量场 `ȧ=f(a,s,z)`，执行时从上一个动作积分，因此执行轨迹天然连续。IL 中对动作块做 RK4 积分并监督积分路径，另有收缩扰动和多模态“多样性头”。**在 RL 中退化为一步 Euler：`a_t = a_{t-1} + f(a_{t-1}, s_t, z_t)`**，算法与 critic 不变。

**RL 结果。** 只在 SuperMimic（BPTT）上验证：训练回报早期高 37%，动作更平滑；原因是 `∂a_t/∂a_{t-1}` 含单位阵，梯度沿动作链传到早期动作不衰减。这个机制依赖 BPTT；**论文没有给出 ODE actor + PPO 的结果**（PPO 只作为无世界模型的参考基线）。

**和本任务的关系。** EE 位置/姿态已经是增量动作，相当于已经用上了该思想；底座 vx/yaw 仍是直接速度指令。

**可借鉴。**
- 🔸 **建议 F（可选）**：底座指令改为 `cmd_t = cmd_{t-1} + clip(Δ)`，用来减少底座指令抖动。但它也会降低对突发偏差的反应速度；动态拦截需要快速调整底座，在 PPO 下收益没有证据。可以做成一个开关，放在后面做消融。
- 借鉴价值：低到中。

---

### 2.9 OrbiSim（SJTU 2026）——世界模型作为可微物理引擎

**核心思路。** 以对象为中心的可微动力学核心（每个实体一个 token，Transformer 耦合，**AdaLN 用物体属性/世界参数调制**，循环记忆，T 步自回归训练）+ 以物理状态为条件的扩散渲染器。用途：(1) 沿动力学反传做策略/轨迹优化，**适合只有终局奖励的稀疏任务**；(2) 以物理参数为优化变量做梯度式系统辨识，并与控制交替优化（Real2Sim2Real）。在只给终局奖励的 robosuite Push 上，SAC/PPO/PPO+RND 基本学不到，OrbiSim 可以。

**和本任务的关系。** 本任务有密集塑形奖励（approaching、lifting 等），并非“只有终局奖励”；而且需要在 PPO 环路外训练一个高质量的抓取接触世界模型，工程量很大。

**可借鉴。**
- ✅ “**静态属性/世界参数作为一等条件信号**”的设计，与 SuperMimic 的 ξ、AdaptiGraph 的物理变量一致，支持建议 E。
- ⏸ 可微系统辨识：在需要把真机台面/物体参数辨识回仿真时才有意义。
- ⏸ 稀疏奖励下的解析梯度：长期方向。

---

## 3. 推荐清单（按建议的实施顺序）

| 编号 | 建议 | 来源 | 改动量 | 预期作用 | 是否影响学生蒸馏 |
|---|---|---|---|---|---|
| **A** | 编码器加“预测未来”的辅助损失 | Enhancing Tactile RL；Contact-Aware ND | 中 | 让教师在无速度观测下学到物体运动；可能加快早期学习 | 不影响：辅助头训练后丢弃 |
| **B** | EKF 拟合前用陀螺仪（及速度估计）补偿机器人自身运动 | Contact-Aware ND / Act-FIGNet 的“动作条件预测”原则 + Level02 运动结构 | 小 | 去掉机器人自身运动造成的外推偏差 | 学生端 EKF 需做同样补偿；只用真机可得信号 |
| **C** | 二值夹爪-物体接触：critic 特权观测 + 辅助 BCE 目标 | Enhancing Tactile RL；Contact-Aware ND | 小 | 让价值函数分清“夹到/没夹到”，改善抓取到抬升阶段的信用分配 | 只进 critic 和辅助头则不影响 |
| **D** | SGS 采样加入均匀混合 | SuperMimic D.1 | 很小 | 防止采样过度集中，保持覆盖 | 不影响 |
| **E** | critic 接收物理/运动参数（机器人摩擦/附加质量、台面 `vy`/`yaw_rate`） | SuperMimic ξ；OrbiSim；AdaptiGraph | 小 | 降低价值估计方差（非对称 actor-critic） | 不影响 |
| F | 底座指令改为增量（开关 + 消融） | ODE Policy | 很小 | 动作更平滑；可能降低反应速度 | 学生需同步动作语义 |
| G | 世界模型 + BPTT | SuperMimic / OrbiSim / DayDreamer | 很大 | 潜在的大幅样本/算力效率提升 | — |

### 建议 A：未来预测辅助损失（优先做）

> 本节为细化版。代码位置均已核对；超参数是起点，不是调好的值。

#### A.1 核心思想与一个重要前提

在 PPO 损失之外加一个监督/自监督损失：让 **Actor 的隐层特征** 预测“物体接下来会怎么动、会不会被夹住”。辅助头训练完就丢掉，策略的输入输出接口不变，学生蒸馏不受影响。仿真里未来真值是现成的，所以直接用**特权真值监督**，比 Enhancing Tactile RL 的潜空间自预测更简单、信号更强。这与 Ji et al. 2022（并行训练估计器）、DreamWaQ（CENet 估计速度并重建下一帧观测）的思路一致。

**前提：教师观测没有历史。** 当前 1284 维观测是单帧状态 + EKF 13 维，物体运动信息**只能**来自 EKF。单帧的隐层特征最多学会：
- 结合 `commands`/`last_action` 修正 EKF 中机器人自身运动带来的偏差（相当于建议 B3）；
- 在 EKF 刚重置（valid=0）、抓取接触切换时，“知道该不该信任 EKF”；
- 让价值相关的特征对物体运动更敏感。

它**无法凭空得到** EKF 之外的速度信息。如果希望辅助损失真正发挥“从历史中提取运动”的作用（Enhancing Tactile RL 的观测就是 k 步历史），建议同时把 EKF 已经在用的 **最近 3–5 帧相对位姿**（每帧 6 维）加进观测。学生端的 EKF 本就维护这段历史，可以复现。这一项建议作为 A 的变体单独消融（见 A.7 的 A3）。

#### A.2 预测什么（目标）

high-level 一步 = `controlFrequencyLow × controlFrequencyInv × dt` = 8×4×0.005 = **0.16 s**；`--rand_control` 下为 7 或 8 个底层步，即 0.14–0.16 s（[`b1z1_base.py:1974`](../DQ_high-level/envs/b1z1_base.py)）。因此目标按**步数**定义：`k ∈ {1, 2, 4, 6}`，其中 4/6 步约等于 EKF 的 0.6/0.9 s。

| 目标 | 定义 | 维度 | 损失 | 说明 |
|---|---|---|---|---|
| **T1 未来物体位姿（当前坐标系）** | `T_{A_t}^{obj}(t+k) = T_W^{A}(t)^{-1}·T_W^{obj}(t+k)`：物体在 t+k 时刻的位置，表达在**当前** t 时刻的臂基坐标系 | 每个 k：位置 3 + 旋转向量 3 | Smooth L1，目标先做标准化 | **主目标**。抓取前只取决于台面运动、与动作无关，因此有确定答案，容易学 |
| T2 当前相对速度 | 物体相对基座的平面线速度（即被 `--no_velocity_obs` 去掉的 2 维）+ 物体 yaw 角速度 | 3 | Smooth L1 | 显式“速度估计”，对应 Ji 2022 / DreamWaQ；最便宜 |
| T3 接触/抬起事件 | 未来 k 步内是否发生夹爪-物体接触、是否 `lifted_object` | 每个 k 各 2 | BCE，正样本权重 ≈10 | 与建议 C 合并；与动作强相关，所以辅助头输入必须含 `a_t` |
| T4（可选）潜空间自预测 | SPR 式：`ẑ_{t+1}=f(z_t,a_t)` 自回归展开，对齐 EMA 目标编码器 | — | 余弦/MSE | 实现更复杂，放在后面 |

为什么 T1 用“当前坐标系”而不是“未来坐标系”：未来坐标系中的相对位姿取决于机器人接下来怎么走，也就是取决于未来策略，目标会随策略变化，难以学习。表达在当前坐标系，抓取前就只由物体自身运动决定。这些目标只在训练时使用真值，**部署时不需要世界系信息**，与真机约束（只有相对位姿）不冲突。

#### A.3 网络改动：只加旁路，不动 Policy 的参数结构

[`Policy.compute`](../DQ_high-level/train_multistate_DQ_teacher.py) 中 `self.net` 是 `Linear-ELU-Linear-ELU-Linear-ELU-Linear(128→9)`。

1. 给 Policy 增加方法 `features(inputs)`：复用 `compute` 的前半部分（物体编码、GFM、拼接），返回 `h = self.net[:-1](x)`（128 维）；`compute` 改为 `self.net[-1](self.features(inputs))`。**`self.net` 的参数键名不变**，旧 checkpoint、play 脚本和学生蒸馏都能照常加载。
2. 新建独立模块 `AuxHead`（**不挂在 Policy 内部**），例如：
   ```python
   class AuxHead(nn.Module):
       def __init__(self, feat_dim=128, act_dim=9, out_reg=..., out_cls=...):
           self.trunk = nn.Sequential(nn.Linear(feat_dim + act_dim, 256), nn.ELU(),
                                      nn.Linear(256, 128), nn.ELU())
           self.reg = nn.Linear(128, out_reg)   # T1/T2
           self.cls = nn.Linear(128, out_cls)   # T3 logits
       def forward(self, h, a):
           z = self.trunk(torch.cat([h, a], -1)); return self.reg(z), self.cls(z)
   ```
   放在 Policy 外，Policy 的 `state_dict` 就不变；`AuxHead` 单独注册到 `agent.checkpoint_modules["aux_head"]`。
3. 只对 Actor 做。Critic 有自己独立的编码器和 GFM（参数不共享），不需要这项。

#### A.4 数据流：如何在 skrl PPO 里拿到“未来真值”

skrl 的 `_update` 用 `memory.sample_all(...)` **打乱时间顺序**后再分 mini-batch（[`ppo.py:382`](../third_party/skrl/skrl/agents/torch/ppo/ppo.py)），所以必须**在打乱之前**把每个 `(t, env)` 的目标算好，写成 memory 中的普通张量。

1. **环境侧**：每个 high-level 步在 `extras`（即 `infos`）中放入特权量：物体世界位姿（7）、臂基世界位姿（7）、物体相对平面速度、接触标志、`lifted_object`。这些都已经在环境里算出来，只是没有导出。
2. **Agent 侧**：新建 `AuxPPO(PPO)`（参考 [`TSILPPO`](../DQ_high-level/utils/tsil_ppo.py) 的子类化方式）：
   - `init()` 里额外 `create_tensor`：`aux_obj_w`、`aux_base_w`、`aux_vel`、`aux_events`，以及 `aux_done`（= terminated | truncated；默认 memory **不存 truncated**，必须自己存，否则跨回合的目标会混入 SGS reset 之后的新回合）；
   - `record_transition()`：调用父类之后，把 `infos` 里的特权量写进上述张量；
   - `_update()` 开头、`sample_all` 之前：对 24 步 rollout 中每个 t 和每个 k，若 `t+k < 24` 且 `aux_done[t : t+k]` 全为 False，则计算 T1/T3 目标，并置 `aux_mask=1`；否则 `aux_mask=0`。结果写入 `aux_target`、`aux_mask` 两个张量，加入 `_tensors_names`，随 mini-batch 一起被采样。
3. **代价与改进**：rollout 只有 24 步，k=6 时每个 rollout 末尾 6 步拿不到目标（约 25% 的样本被掩码）。第一版先接受这个损失。第二版按 Enhancing Tactile RL §3.3 加**独立辅助缓冲区**：保留最近 N=2–4 个 rollout 的 `(去掉 1024 维物体特征后的观测, 动作, 特权量)`，跨 rollout 拼接后再算目标。sugar_box 的物体特征是常量，不必存，按 ~260 维估算：8192 env × 96 步 × 260 × 4 B ≈ 0.8 GB。

#### A.5 损失与优化

```text
L_aux = Σ_k w_k · SmoothL1(ĥ_T1,k, y_T1,k)·mask_k            # 位姿，目标做 running 标准化
      + λ_v · SmoothL1(ĥ_T2, y_T2)                             # 速度
      + λ_e · Σ_k BCE(ĥ_T3,k, y_T3,k; pos_weight=10)·mask_k     # 事件
```

两种接入方式：

| 方式 | 做法 | 优点 | 风险 |
|---|---|---|---|
| **联合（先用这个）** | 每个 PPO mini-batch 内 `(policy_loss + entropy + value_loss + c_aux·L_aux).backward()`；`AuxHead` 参数加入同一优化器 | 改动最少 | 辅助梯度直接影响策略，可能扰动 KL 自适应学习率（`KLAdaptiveRL`, kl_threshold=0.008） |
| 分离优化器 | PPO 更新后，另一个 Adam（`lr_aux`）只更新 `AuxHead` + Policy 的 `feature_encoder`/GFM/`net[:-1]`，与 Enhancing Tactile RL 一致 | 互不干扰，可单独调学习率 | 两个优化器更新同一组参数，需要观察策略 KL |

若发现辅助损失影响策略稳定性，可进一步参考 PPG（Phasic Policy Gradient）：每 N 次 PPO 迭代做一次辅助阶段，同时加一项行为克隆约束 `KL(π_old‖π)`，保证辅助训练不改变策略输出。

**起始超参**：`c_aux ∈ {0.1, 0.5, 1.0}`；`k = {1,2,4,6}`，权重 `w_k` 相同；`λ_v = 1`，`λ_e = 0.5`；分离优化器时 `lr_aux = 1e-4~4.2e-4`；辅助梯度单独做 grad-norm 裁剪 1.0。

#### A.6 监控与验证

- **主指标**：[`eval_sgs_teacher.py`](../DQ_high-level/eval_sgs_teacher.py) 的原始 2 m 完整任务成功率、成功用时；训练早期成功率曲线（辅助损失通常主要改善早期学习速度）。
- **辅助损失本身**：各 k 的位姿误差（cm / deg），并与**同一批样本上 EKF 的误差**对比。若辅助头 4/6 步误差明显低于 EKF，说明网络学到了 EKF 之外的修正；若相当，说明受限于 A.1 的“无历史”前提。
- **线性探针**：冻结 `h`，回归物体相对速度，看 R²（Enhancing Tactile RL 用这一方法说明 FD 表征编码了速度）。
- **稳定性**：策略 KL、学习率（KLAdaptiveRL 是否被迫降低）、`log_std`。
- 至少 2–3 个 seed，因为 PPO 方差较大。

#### A.7 消融顺序

| 组 | 内容 | 要回答的问题 |
|---|---|---|
| A0 | 基线（当前脚本） | — |
| A1 | + T2 速度估计 | 最便宜的版本是否有用 |
| A2 | + T1 未来位姿（k=1,2,4,6） | 主方案 |
| A3 | A2 + 观测加入 3–5 帧相对位姿历史 | 打破“无历史”上限后收益是否更大 |
| A4 | A2 + T3 接触/抬起事件 | 是否改善抓取→抬升阶段（与建议 C 联动） |
| A5 | 最好的一组 + 独立辅助缓冲区 / 分离优化器 | 工程改进是否值得 |

#### A.8 风险

- 辅助梯度与 PPO 冲突：每个任务都要调 `c_aux`（Enhancing Tactile RL 为每个环境都单独做了超参搜索）。
- T1 在抓住物体后与动作相关，可以只在抓取前计算（用 `lifted_object`/接触标志做掩码），或者保留但降低权重。
- 计算开销：多一次 `features` 前向和一个小 MLP，与仿真开销相比很小；独立缓冲区会占用 ~0.8 GB 显存。

#### A.9 参考文献

| 文献 | 与本建议的关系 |
|---|---|
| Miller et al., *Enhancing Tactile-based Reinforcement Learning for Robotic Control*, NeurIPS 2025（本地 `papers/dynamic_papers/Enhancing_Tactile-based.pdf`） | 最直接的模板：skrl PPO + 辅助损失、多步前向动力学最有效、独立辅助缓冲区、BCE 正样本加权 |
| Ji et al., *Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion*, RA-L 2022, [arXiv:2202.05481](https://arxiv.org/abs/2202.05481) | PPO 训练策略的同时，用监督学习从仿真真值训练估计器（速度、足端高度、接触概率） |
| Nahrendra et al., *DreamWaQ*, ICRA 2023, [arXiv:2301.10602](https://arxiv.org/abs/2301.10602) | CENet：共享编码器同时估计速度并重建下一帧观测（β-VAE），对应 T1/T2 |
| Long et al., *Hybrid Internal Model (HIMLoco)*, ICLR 2024, [arXiv:2312.11460](https://arxiv.org/abs/2312.11460) | 显式速度估计 + 对比学习让隐变量对齐后继状态，只用本体感觉 |
| Schwarzer et al., *Data-Efficient RL with Self-Predictive Representations (SPR)*, ICLR 2021, [arXiv:2007.05929](https://arxiv.org/abs/2007.05929) | T4 潜空间多步自预测 + EMA 目标编码器的来源 |
| Jaderberg et al., *Reinforcement Learning with Unsupervised Auxiliary Tasks (UNREAL)*, ICLR 2017, [arXiv:1611.05397](https://arxiv.org/abs/1611.05397) | 辅助任务的经典出处；其中“奖励预测”可对应 T3 |
| Cobbe et al., *Phasic Policy Gradient*, ICML 2021, [arXiv:2009.04416](https://arxiv.org/abs/2009.04416) | 辅助训练与策略训练分阶段、加行为克隆约束，用来缓解辅助梯度干扰 |
| Jing et al., *Contact-Aware Neural Dynamics*, 2026（本地） | 先预测二值接触，再以接触为条件预测位姿，对应 T3 与 T1 的组合 |
| Amiranashvili et al., *Motion Perception in RL with Dynamic Objects*, CoRL 2018, [arXiv:1901.03162](https://arxiv.org/abs/1901.03162) | 在动态物体（接球）任务中，给 PPO 提供运动信息（光流）明显有帮助；佐证运动信息对动态任务的价值 |

### 建议 B：EKF 拟合时补偿机器人自身运动（只用真机可得的信息）

**约束（已修订）。** 真机只能得到物体相对基座的位姿，**不能在世界系中拟合 EKF**。教师也不应改用特权的世界系 EKF，否则学生无法复现同一份输入。本节第一版“世界系拟合”的建议作废，以下方案只使用相对位姿和本体感觉。

**问题仍然存在。** [`_update_object_pose_prediction`](../DQ_high-level/envs/b1z1_pickmulti.py) 把相对位姿直接推入匀速 EKF。相对运动 = 物体运动 ⊖ 机器人运动，后者由策略控制：转向、加减速、停步都会破坏匀速假设，也就是 Contact-Aware ND 和 Act-FIGNet 强调的“预测要以动作为条件”。
另一方面，Level02 台面上的物体在世界系里是恒定 SE(3) twist（[`sgs_motion.py`](../DQ_high-level/utils/sgs_motion.py)，不打滑时），本身很好预测；因此**推测误差主要来自机器人自身的运动被混入**（待第 0 步验证）。所以思路是把这部分运动从相对测量中扣掉，而不需要知道世界系位姿。

**第 0 步：先量化，不改训练。** 用 [`evaluate_object_pose_prediction.py`](../DQ_high-level/evaluate_object_pose_prediction.py) 统计当前 EKF 在 0.6/0.9 s 的误差，**按底座角速度、线加速度分桶**。只有在底座转向或变速时误差明显变大，下面的方案才值得做。

**方案 B1：用 IMU 陀螺仪补偿旋转（推荐先做）。**
- 真机一定有陀螺仪：底层控制器的观测里已经使用 `base_ang_vel`（[`b1z1_base.py:1158`](../DQ_high-level/envs/b1z1_base.py)）。
- 对 5 帧窗口内的每一帧，用陀螺积分得到“该帧基座 → 当前基座”的相对旋转 `R_k→t`，把历史相对位姿变换到**当前**基座坐标系后再拟合 EKF。窗口只有零点几秒，陀螺积分的漂移可以忽略。
- 平移暂不补偿。旋转在远距离处造成的横向偏差最大（距离 × 偏航角），补偿旋转就能消除这部分主要误差。
- 仿真中用 `base_ang_vel` + 与真机相当的噪声/偏置来积分，**不用真值根状态**，保证教师与真机条件一致。

**方案 B2：再补偿平移。**
- 窗口内的基座平移可以从以下来源估计：(a) 底层的速度估计或腿式里程计（如果真机部署链路中有）；(b) 退一步，用策略自己发出的 vx/yaw 指令积分（一定可得，但底层跟踪不完美，误差需要在仿真中注入）。
- 先确认真机上实际能拿到哪一种，再决定是否做。

**方案 B3：不改 EKF，让网络学会修正。**
- 保持 EKF 不变，把最近几步的底座指令（以及陀螺角速度）作为观测，并结合建议 A 的辅助损失，监督“未来真实相对位姿”。网络会以动作为条件学到 EKF 的残差。
- 完全不需要额外传感器，但效果依赖学习，不如 B1 直接。

**预测语义要说清楚。** 补偿后，EKF 预测的是“物体将来位于**当前**基座坐标系中的哪里”，而当前 EKF 隐含的是“若相对运动保持匀速，将来位于将来基座坐标系中的哪里”。两者语义不同：网络结构和维度不变，但要从头训练，并用新的实验目录（与之前 GFM 维度变更的处理一致）；蒸馏时学生端的 EKF 也必须做同样的补偿。

**风险。**
- 抓住物体后，物体运动不再是恒定 twist，这一段 EKF 本来就不适用；建议 C 的接触信号可以帮助网络判断何时不信任预测。
- 如果第 0 步显示误差与底座运动关系不大，这条建议就不必做。

### 建议 C：二值接触信号

- 由 `_contact_forces` 中两个夹爪手指 link 的接触力（或物体 link 的接触力 + 与夹爪距离）阈值化，得到 `c_t ∈ {0,1}`（可分左右指）。
- **放哪里**：先放进 critic（特权）和建议 A 的辅助 BCE 目标；actor 不放，以免学生无法复现。之后可以评估真机夹爪电流/开度停滞能否给出近似信号，再决定是否进 actor。
- 依据：Enhancing Tactile RL 发现二值接触在机器人-物体运动解耦时关键；Contact-Aware ND 证明二值接触稳健且在仿真和真机上一致。

### 建议 D：SGS 加均匀混合

当前 [`probabilities`](../DQ_high-level/utils/sgs.py) 只有 SGS 权重 + ε 下限。按 SuperMimic D.1 改为混合：

```python
q = α_u * uniform + (1 - α_u) * q_sgs          # 例如 α_u = 0.2–0.5
# 可选：再混入 q_sgs 的 EMA，让刚学会的配置保留练习
```

这是一行级别的改动，可直接做 `α_u ∈ {0, 0.2, 0.5}` 消融。注意 SuperMimic 偏向**低回报**，SGS 偏向**中等成功率**，两者的目标不同；混合的价值在于覆盖，而不是替换 SGS 的加权规则。

### 建议 E：critic 接收物理/运动参数

把每个环境的机器人摩擦桶值、躯干附加质量与质心偏移、台面 `vy`/`yaw_rate`/是否旋转（SGS 配置中已有）拼进 **Value** 的输入。actor 不变，学生不受影响。依据：SuperMimic 的模型和 critic 都条件于 ξ 才能保持确定性；OrbiSim/AdaptiGraph 都把物理参数作为一等条件。

另外，目前物体本身的质量和摩擦没有随机化。若后续要提高抓取后抬升的鲁棒性（AdaptiGraph 的出发点正是同类物体物理参数不同、行为不同），可以加入物体质量/摩擦随机化，并同样只把参数交给 critic。

### 不建议现在做的

- **世界模型 + BPTT 替代 PPO（G）**：抓取接触的不连续性让模型梯度不可靠；高层动力学包括冻结的底层控制器；SuperMimic 的成功依赖每步密集的追踪奖励，并明确未覆盖物体交互。工程量远大于以上 A–E 的总和。
- **扩散预测器、GNN、MPC、推理期梯度引导**（Contact-Aware ND、Act-FIGNet、AdaptiGraph、ForeTac）：服务于规划或生成式策略，与 PPO 教师的接口不匹配。

---

## 4. 建议的实验顺序

所有实验均使用相同的 SGS bank、seed 组和 120000 步预算，以 [`eval_sgs_teacher.py`](../DQ_high-level/eval_sgs_teacher.py) 的原始 2 m 完整任务成功率和成功用时作为主指标，至少 2–3 个 seed。

1. **离线量化 B**：只跑预测评估脚本，按底座角速度/加速度分桶看误差，不训练。决定 B 是否值得做。
2. **D + E**：改动最小，可以合并为一组，与基线对比。
3. **B1 陀螺仪旋转补偿**（如果第 1 步确认误差显著）；B2/B3 视真机可得信号再定。
4. **A**：按 §3 建议 A.7 的顺序（A1→A2→A3/A4→A5）；扫描 `c_aux`。
5. **F**：最后作为独立消融。

每一步都只加一个变量，并记录到 W&B 中现有的成功率和每类配置成功率面板。

---

## 附：论文速查

| 论文 | 一句话 | 对本任务的借鉴价值 |
|---|---|---|
| SuperMimic | 在线拟合可微世界模型，BPTT 训练人形追踪策略，算力降低 >1000× | 中（reset 混合、critic 条件 ξ；BPTT 本身暂不适用） |
| DayDreamer | 在 4 台真机上在线跑 Dreamer | 低 |
| AdaptiGraph | 物理参数条件化的 GNN 动力学 + 在线参数辨识 | 中低（参数条件化、学生端运动参数辨识） |
| Contact-Aware Neural Dynamics | 先预测二值接触，再以接触为条件扩散预测位姿；仿真预训练 + 真机微调 | 中（二值接触、动作条件预测） |
| Act-FIGNet | 动作条件网格 GNN，预测插孔运动和力旋量 | 低（原则性启发） |
| ForeTac | 预测触觉后果，用成功/失败学到的打分梯度引导生成式策略 | 低到中（结果预测器） |
| Enhancing Tactile-based RL | PPO + 自监督前向动力学辅助损失 + 二值接触 + 独立辅助缓冲区 | **高**（建议 A、C） |
| ODE Policy | 策略学习动作的向量场，RL 中为 `a_t=a_{t-1}+f` | 低到中（底座增量指令） |
| OrbiSim | 对象为中心、参数条件化的可微世界模型，用于稀疏奖励优化和系统辨识 | 低到中（参数条件化思想） |
