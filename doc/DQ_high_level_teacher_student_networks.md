**DQ_high-level 教师策略与学生策略：网络结构、张量流与训练关系**

本文基于当前仓库的实际代码，沿着 README 的高层训练入口梳理网络。教师使用 **物体特征编码器＋抓取候选注意力＋Actor/Critic MLP**；学生使用 **共享视觉 CNN＋双分支状态引导时序 Transformer＋主动作头＋残差动作头**。两者输出相同的 9 维高层动作，并调用已经训练好的底层控制器。

本文包含 **17 张可缩放 SVG 结构图**。图中蓝色表示输入，绿色表示含可学习参数的网络，紫色表示张量运算，橙色表示输出，灰色表示数据、非学习运算或当前路径不使用的模块，红色表示训练损失。绿色表示模块具有参数，并不表示这些参数在每个训练阶段都被更新；具体更新范围见训练关系说明。

| 阅读内容 | 对应图 |
|---|---|
| [范围、入口与总体关系](#scope) | 图 1 |
| [教师观测的精确组成](#teacher-input) | 观测维度与切片表 |
| [教师 Actor、物体编码器与抓取注意力](#teacher-actor) | 图 2—4 |
| [教师 Critic 与高斯动作分布](#teacher-critic) | 图 5—6 |
| [学生输入与整体结构](#student-input) | 图 7 |
| [共享 CNN、状态引导 Transformer 及注意力内部](#student-vision) | 图 8—11 |
| [学生主动作头与残差动作头](#student-heads) | 图 12 |
| [教师到学生的监督与分阶段训练](#training) | 图 13 |
| [高层调用的底层网络与执行路径](#low-level) | 图 14—17 |
| [参数量、实现边界与验证](#verification) | 参数表、覆盖清单 |

<a id="scope"></a>

**本文采用 README 主路径对应的配置。** 教师入口是 [`train_multistate_DQ_teacher.py`](../DQ_high-level/train_multistate_DQ_teacher.py)，学生入口是 [`train_multi_bc_deter_DQ_stu.py`](../DQ_high-level/train_multi_bc_deter_DQ_stu.py)。两个 `play` 脚本复用这些训练脚本中的模型定义。

| 条件 | 本文取值 |
|---|---|
| 任务 | `B1Z1PickMulti` |
| 机器人状态 | 启用 `--roboinfo` |
| 底层步态观测 | 启用 `--observe_gait_commands` |
| 底座与动作 | 非 `floating_base`，不启用 `--pitch_control`，9 维高层动作 |
| 物体特征 | 启用，1024 维；不启用 `--no_feature` |
| 教师动作分布 | 默认 `use_tanh=False` |
| 学生相机模式 | `full`，前向与腕部两路相机 |
| 学生图像 | 高 54、宽 96；每个视角 3 帧、每帧 2 通道 |
| 学生训练器 | 默认 `DAgger_RNN`，不启用 `--mlp_stu` |
| 学生额外输出 | 不启用 `--pred_success` |
| 观测尾部 | 上一步动作，不启用 `--last_commands` |

维度记号：`B` 为批量大小；学生视觉序列长度 `T=3`；抓取候选数 `N=30`；学生 token 特征维度 `D=64`。图像张量按 `[批量, 通道, 高, 宽]` 表示。所有 `Linear` 和卷积默认包含 bias，除非另行说明。

![图1：高层训练与执行的总体关系](assets/dq_high_level_networks/01_training_overview.svg)

*图 1：教师先通过 PPO 学习；学生随后通过教师动作监督学习。底层控制器和解析控制连接高层动作与仿真。图中回环表示环境交互，不表示对仿真动力学反向传播。*

教师训练时同时使用 Actor 和 Critic；学生训练时，教师提供动作标签，学生自身没有 Critic。学生脚本仍然构造教师 PPO 对象及其 Value 网络以加载相应模型，但 DAgger 生成动作标签实际调用的是教师 Actor。

高层运行还涉及一套预训练底层 `ActorCritic`。为了完整覆盖实际涉及的网络，本文将其历史编码器、特权编码器、Actor 双头和 Critic 双头一并绘出，并标记哪些分支在高层实际被使用。没有把仓库中仅导入、未实例化的旧网络当成当前策略的一部分。

---

<a id="teacher-input"></a>

**教师网络的原始输入是 1276 维。** 环境把物体特征、状态、抓取候选和上一动作写入 `obs_buf`。下表中的切片均采用 Python 左闭右开区间。

| `obs_buf` 切片 | 内容 | 维度 | 学生数值状态是否保留 |
|---|---|---:|---|
| `[0:1024]` | 从物体 `features.npy` 读取的特征 | 1024 | 否 |
| `[1024:1030]` | 物体位置与 RPY 姿态 | 6 | 否 |
| `[1030:1036]` | 当前末端位置与 RPY 姿态 | 6 | 是 |
| `[1036:1055]` | 全部关节位置，包含夹爪 | 19 | 是 |
| `[1055:1073]` | 不含夹爪的关节速度，乘 `0.05` | 18 | 是 |
| `[1073:1076]` | 当前底盘命令 `commands` | 3 | 是 |
| `[1076:1082]` | 当前末端目标位置与目标 RPY | 6 | 是 |
| `[1082:1085]` | 机器人局部线速度 | 3 | 否 |
| `[1085:1087]` | 物体相对机器人的平面速度 | 2 | 否 |
| `[1087:1177]` | 30 个抓取候选的位置，先整体展开 | 90 | 否 |
| `[1177:1267]` | 30 个抓取候选的 RPY，再整体展开 | 90 | 否 |
| `[1267:1276]` | 上一步高层动作 | 9 | 是 |
| **合计** | | **1276** | **保留 61 维** |

来源：[`B1Z1PickMulti._compute_observations`](../DQ_high-level/envs/b1z1_pickmulti.py#L534)、[`compute_robot_observations`](../DQ_high-level/envs/b1z1_pickmulti.py#L783)、[`_setup_obs_and_action_info`](../DQ_high-level/envs/b1z1_base.py#L209)。

这个维度也可以从环境的配置计算式得到：

```text
num_obs = 38 + 30×6 + 2 + 1024 - 1
最终观测 = num_obs + 高层动作9 + roboinfo扩展24
         = 1276
```

有三个数据来源细节直接影响网络解释。

第一，`features.npy` 是预计算资源，每个物体的文件形状为 `(1,1024)`。环境按物体类别读取这份特征，并在不同物体实例中使用。当前高层代码没有在线运行的 PointNet 或其他点云编码网络，也不能仅根据特征维度反推出离线生成器的结构。代码中真正参与高层学习的是后面的 `1024→512→128` 编码器。

第二，候选抓取由 `contact_grasp_info_mul/predictions_image_*.json` 加载。资源中有 30 份物体抓取文件，每份包含 `(30,4,4)` 的候选变换。**物体类别数 30 与每个物体的候选抓取数 30 是两个概念**。每个环境中的目标物体使用自己的 30 个候选；教师注意力的候选轴为 `[B,30,6]`。

第三，`PredictPoint` 根据物体当前位姿更新候选变换，环境再将候选转换为机器人相关坐标中的位置和姿态。这个模块执行坐标变换，没有可训练神经网络。`compute_robot_observations` 临时返回的最后 6 维物体世界位姿用于该变换，随后被候选数据替换，不能把这 6 维再次加到最终 1276 维中。物体观测本身也有坐标约定：位置的水平分量在机器人 yaw 相关坐标中表达，z 分量保留物体世界高度；候选位置则相对机械臂基座并使用机身旋转转换。见 [`PredictPoint.get_grasp_world`](../DQ_high-level/PredictPiontTransform/predictpoint_mul.py#L167)。

**送入教师网络前，全部 1276 维观测先经过 `RunningStandardScaler`。** 这包含预计算物体特征与抓取候选，并不只是机器人状态。该预处理器维护运行均值和方差，默认将标准化值裁剪到 `[-5,5]`；它不是可学习的神经网络。后面的教师网络图均以这个实际预处理顺序为准。来源：[`PPO 配置`](../DQ_high-level/train_multistate_DQ_teacher.py#L313)、[`PPO.act`](../third_party/skrl/skrl/agents/torch/ppo/ppo.py#L199)、[`RunningStandardScaler`](../third_party/skrl/skrl/resources/preprocessors/torch/running_standard_scaler.py)。

---

<a id="teacher-actor"></a>

**教师 Actor 由一个物体编码器、一个抓取候选注意力模块和一个动作均值 MLP 组成。** 类定义为 `Policy(GaussianMixin, Model, PredictAttentionSelector)`，见 [`教师 Policy`](../DQ_high-level/train_multistate_DQ_teacher.py#L58)。

![图2：教师Actor完整网络](assets/dq_high_level_networks/02_teacher_actor.svg)

*图 2：教师 Actor 的全部可学习模块。物体位姿 6 维既参与注意力 Query，也保留在通向主 MLP 的 72 维状态中。*

进入主 MLP 的维度为：

$$
d_{\mathrm{teacher\ fusion}}=(1276-1024-180)+128+6=206.
$$

其中其余状态 72 维由当前状态 63 维和上一动作 9 维组成。代码中的拼接顺序是：

```python
concat([
    states[..., 1024:1087],  # 63维状态
    states[..., -9:],        # 上一动作9
    features_encode,        # 物体编码128
    right_predict_point     # 抓取表示6
], dim=-1)                  # 206维
```

**物体编码器把 1024 维输入压缩到 128 维。** Actor 和 Critic 分别拥有一份结构相同、参数独立的编码器。

![图3：物体特征编码器](assets/dq_high_level_networks/03_teacher_feature_encoder.svg)

*图 3：`feature_encoder`。第二个 Linear 后没有 ELU 或 Tanh。*

| 顺序 | 层 | 输入形状 | 输出形状 | 参数数 |
|---:|---|---|---|---:|
| 1 | `Linear(1024,512)` | `[B,1024]` | `[B,512]` | 524,800 |
| 2 | `ELU()` | `[B,512]` | `[B,512]` | 0 |
| 3 | `Linear(512,128)` | `[B,512]` | `[B,128]` | 65,664 |
| **合计** | | | | **590,464** |

该编码 $z\in\mathbb{R}^{128}$ 一路直接进入动作 MLP，另一路与物体位姿拼接以生成注意力 Query。因此物体特征同时影响“如何汇总抓取候选”和“最终如何行动”。

**抓取注意力根据物体特征和物体位姿，对 30 个抓取候选做连续汇总。** 实现位于 [`PredictAttentionSelector`](../DQ_high-level/modules/predictattention.py#L6)。

![图4：抓取候选注意力](assets/dq_high_level_networks/04_grasp_attention.svg)

*图 4：单头、单 Query 的缩放点积注意力。四个线性投影均为可学习参数，没有额外 FFN、LayerNorm 或位置编码。*

设标准化后的物体位姿为 $p\in\mathbb{R}^{B\times6}$，候选集合为 $G\in\mathbb{R}^{B\times30\times6}$，则：

$$
c=[z,p]\in\mathbb{R}^{B\times134},
\quad Q=\operatorname{unsqueeze}(W_qc+b_q)\in\mathbb{R}^{B\times1\times64},
$$

$$
K=W_kG+b_k,\quad V=W_vG+b_v,
\quad K,V\in\mathbb{R}^{B\times30\times64},
$$

$$
\alpha=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{64}}\right),
\qquad g=W_o\operatorname{squeeze}(\alpha V)+b_o\in\mathbb{R}^{B\times6}.
$$

| 投影 | 输入→输出 | 参数数 |
|---|---|---:|
| `query_proj` | `134→64` | 8,640 |
| `key_proj` | `6→64` | 448 |
| `value_proj` | `6→64` | 448 |
| `output_proj` | `64→6` | 390 |
| **注意力模块合计** | | **9,926** |

虽然变量名为 `selected_grasp` 或 `right_predict_point`，实际计算没有 `argmax`，没有返回某个候选的索引，也没有直接抓取位姿真值损失。输出应解释为 **6 维可学习抓取表示**：它由标准化候选经注意力和线性投影得到，不保证恰好等于某个候选，也不保证自身就是合法物理位姿。这个表示与其他状态一起服务于动作决策。

**Actor 主 MLP 把融合后的 206 维信息映射成 9 维动作均值。** 它的层次为：

| 顺序 | 层 | 输出形状 |
|---:|---|---|
| 1 | `Linear(206,512) → ELU` | `[B,512]` |
| 2 | `Linear(512,256) → ELU` | `[B,256]` |
| 3 | `Linear(256,128) → ELU` | `[B,128]` |
| 4 | `Linear(128,9)` | `[B,9]` |

这里没有 RNN 或图像历史。教师的动态信息直接包含机器人速度、物体相对平面速度和上一动作。MLP 输出层不带激活，后续由动作分布与环境动作处理决定采样和限制方式。

---

<a id="teacher-critic"></a>

**教师 Critic 具有独立的同构特征处理分支，最终输出一个标量价值。** `Value(DeterministicMixin, Model, PredictAttentionSelector)` 与 Actor 分别实例化，因此物体编码器、注意力四个投影和主 MLP 都不共享参数。见 [`教师 Value`](../DQ_high-level/train_multistate_DQ_teacher.py#L106) 和 [`模型实例化`](../DQ_high-level/train_multistate_DQ_teacher.py#L291)。

![图5：教师Critic完整网络](assets/dq_high_level_networks/05_teacher_critic.svg)

*图 5：Critic 使用和 Actor 相同的教师观测，但在自己的参数下重新计算物体编码和抓取表示。*

Critic 的主 MLP 为：

```text
206 → Linear(206,512) → ELU
    → Linear(512,256) → ELU
    → Linear(256,128) → ELU
    → Linear(128,1)
```

输出形状为 `[B,1]`，没有 Sigmoid 或 Tanh。PPO 还配置了单独的价值标准化器，训练和价值使用时配合标准化/逆变换处理。虽然 Critic 类也注册了 `log_std_parameter[9]`，但其 `compute` 没有使用它；它不表示 Critic 在输出一个动作分布。

**教师的探索来自高斯输出分布，而不是另一个预测标准差的 MLP。** `log_std_parameter` 是长度为 9 的独立参数向量，初值全为 0，各动作维度的标准差对所有输入状态共享。

![图6：教师高斯输出分布](assets/dq_high_level_networks/06_teacher_distribution.svg)

*图 6：默认 `use_tanh=False` 的动作分布路径。分布中的均值依赖观测，标准差是可学习但不依赖观测的向量。*

$$
\sigma=\exp\big(\operatorname{clip}(\log\sigma,-20,2)\big),
\qquad
\pi_T(a\mid o_T)=\mathcal{N}\big(\mu_T(o_T),\operatorname{diag}(\sigma^2)\big).
$$

PPO 训练采用重参数化采样；评估以及给学生产生监督标签时设置 `deterministic=True`，返回均值动作。逐动作维度的 `log_prob` 相加得到 `[B,1]`，用于 PPO 策略更新。标准差参数在教师高斯训练中参与学习。

可选 `--use_tanh` 会进入仓库 `GaussianMixin` 的 `TanhNormal` 分支并启用相应动作裁剪，不能直接把它理解为给上述 MLP 添加一个 `nn.Tanh` 层。本文所有默认图以 README 未启用该选项的路径为准。实现见 [`GaussianMixin.act`](../third_party/skrl/skrl/models/torch/gaussian.py#L188)。

---

<a id="student-input"></a>

**学生实际消费的输入是 62269 维，由图像历史和机器人状态组成。** 类定义为 `Policy(DeterministicMixin, Model)`，没有实例化学生 Value 网络。见 [`学生 Policy`](../DQ_high-level/train_multi_bc_deter_DQ_stu.py#L30)。

![图7：学生Policy完整网络](assets/dq_high_level_networks/07_student_policy.svg)

*图 7：图中两个 CNN 框表示同一个 `shared_cnn` 的两次调用，虚线表示权重共享，不是额外的数据流。两个 Transformer 和两个投影层均各有独立参数。*

学生图像元素总数为：

$$
d_{\mathrm{image}}=2\ \text{视角}\times3\ \text{帧}\times2\ \text{通道}\times54\times96=62208.
$$

学生状态不是额外估计器的输出，而是从环境教师观测中保留机器人可用的部分：

```python
robot_state = concat([
    obs_buf[:, 1030:1082],  # 末端6 + 关节位置19 + 关节速度18 + 命令3 + 目标6
    obs_buf[:, -9:]         # 上一高层动作9
], dim=-1)                 # 52 + 9 = 61
```

因此实际输入为：

$$
o_S=[\operatorname{flatten}(I_{t-2:t}),r_t]\in\mathbb{R}^{B\times(62208+61)}
=\mathbb{R}^{B\times62269}.
$$

学生不直接接收物体特征、物体位姿、机器人线速度、物体相对速度或抓取候选。它得到的机器人状态 token 是**当前 61 维状态**，并不是三帧机器人状态；三帧历史只在视觉输入中显式保存。

有一处模型元数据容易误导阅读：训练脚本将 `env.observation_space` 赋给 `student_obs_space`，因此传入学生 `Model` 构造器的名义观测维度仍是 1276。但是 `compute` 实际接收训练器传来的 `states["states"]`，内存也按 `env.state_space` 保存学生输入，所以真实前向输入是 62269。应以 `compute` 和环境拼接为准。见 [`学生实例化`](../DQ_high-level/train_multi_bc_deter_DQ_stu.py#L304)、[`训练器观测分流`](../DQ_high-level/learning/dagger_trainer.py#L117)、[`学生状态拼接`](../DQ_high-level/envs/b1z1_base.py#L2134)。

**图像两个通道分别是目标掩码和掩码后的深度。** 环境先根据目标 segmentation id 构造 0/1 mask，再将深度裁剪、转为正值并归一化，最后计算 `normalized_depth * mask`。RGB 虽被环境读取，但没有进入这里的学生网络。默认学生也没有额外 `RunningStandardScaler`；图像归一化和关节速度缩放由环境完成。

每个时间点先构造四个通道，再按旧帧到新帧放入历史：

| 时间 | 展开通道索引 | 内容顺序 |
|---|---|---|
| `t-2` | `0,1,2,3` | 前向 mask、腕部 mask、前向目标深度、腕部目标深度 |
| `t-1` | `4,5,6,7` | 同上 |
| `t` | `8,9,10,11` | 同上 |

输入 reshape 后为 `[B,12,54,96]`。前向分支取 `[0,2,4,6,8,10]`，腕部分支取 `[1,3,5,7,9,11]`，分别得到 `[B,6,54,96]`；每个分支再 reshape 成 `[B×3,2,54,96]`，让同一帧的 mask 和深度成为两个输入通道。episode 开始时，环境用当前图像重复填充历史，之后移位更新。

来源：[`相机观测处理`](../DQ_high-level/envs/b1z1_base.py#L1352)、[`图像历史构造`](../DQ_high-level/envs/b1z1_base.py#L2154)、[`学生图像拆分`](../DQ_high-level/train_multi_bc_deter_DQ_stu.py#L80)。学生创建环境时会强制开启相机，不能只依据 YAML 中的 `enableCamera: false` 判断学生没有视觉输入。

---

<a id="student-vision"></a>

**共享 CNN 为每帧生成一个 64 维视觉 token。** 两个视角和三个时间点全部复用 `SharedCNNBackbone` 的同一套参数。它是二维空间图像编码器，时间建模发生在后续 Transformer 中。

![图8：共享CNN逐层结构](assets/dq_high_level_networks/08_shared_cnn.svg)

*图 8：`SharedCNNBackbone` 的所有层，包含卷积核大小、步长、池化和展平维度。*

| 顺序 | 层与配置 | 单帧输出形状 |
|---:|---|---|
| 0 | mask＋目标深度 | `2×54×96` |
| 1 | `Conv2d(2,16,kernel_size=5,stride=1,padding=0)` | `16×50×92` |
| 2 | `MaxPool2d(2,2)` | `16×25×46` |
| 3 | `ELU` | `16×25×46` |
| 4 | `Conv2d(16,32,kernel_size=3,stride=1,padding=0)` | `32×23×44` |
| 5 | `ELU` | `32×23×44` |
| 6 | `Flatten` | `32384` |
| 7 | `Linear(32384,128) → ELU` | `128` |
| 8 | `Linear(128,64)` | `64` |

注意第一层卷积之后先做 MaxPool，再做 ELU；这里没有 BatchNorm 或 Dropout，最终 64 维输出也没有激活。源码中的部分卷积输出注释沿用了旧通道数，本文按实际 `in_channels/out_channels` 与前向结果绘图。见 [`SharedCNNBackbone`](../DQ_high-level/modules/feature_extractor.py#L188)。

编码完成后恢复时间维，得到：

```text
arm_seq   : [B,3,64]  # 腕部
base_seq  : [B,3,64]  # 机身前向
```

CNN 的 `32384→128` 全连接层包含 4,145,280 个参数，是整个学生模型最大的单层。共享 CNN 总参数为 4,158,992；虽然被两个视角多次调用，参数只计算一份。

**每个视角随后进入独立的状态引导 Transformer。** `transformer_arm` 与 `transformer_base` 都是 `GuidedTransformerBlock(feature_dim=64)`，二者结构相同而参数不共享。见 [`GuidedTransformerBlock`](../DQ_high-level/modules/feature_extractor.py#L234)。

![图9：状态引导Transformer](assets/dq_high_level_networks/09_guided_transformer.svg)

*图 9：每个分支包含位置编码、状态投影、两个 Encoder 层和视觉 token 平均池化。*

先对三帧视觉 token 添加固定的正弦位置编码：

$$
\operatorname{PE}(t,2i)=\sin\left(t/10000^{2i/64}\right),
\qquad
\operatorname{PE}(t,2i+1)=\cos\left(t/10000^{2i/64}\right).
$$

位置编码通过 `register_buffer` 保存，不是可训练参数，预生成长度为 10。当前只取其中三帧。每个分支自己的 `state_proj: Linear(61,64)` 把机器人状态变成一个 token，然后构造：

$$
X=[\operatorname{Linear}_{61\to64}(r_t),\ f_{t-2}+\operatorname{PE}(0),\ f_{t-1}+\operatorname{PE}(1),\ f_t+\operatorname{PE}(2)]
\in\mathbb{R}^{B\times4\times64}.
$$

**位置编码只加在视觉 token 上，状态 token 不加位置编码。** 状态引导通过四个 token 的自注意力交互实现，没有额外单独的 cross-attention 子层，也没有门控网络。

两个 Encoder 层处理后，代码执行 `y[:,1:].mean(dim=1)`，只平均三个视觉 token，输出 `[B,64]`。状态 token 虽不直接进入平均，其信息已经能够通过自注意力影响视觉 token。之后原始机器人状态还会再次直连到动作融合向量。

| Transformer 配置 | 实际值 |
|---|---|
| 每个分支 Encoder 层数 | 2 |
| token 维度 `d_model` | 64 |
| 多头数 `nhead` | 2 |
| 每头维度 | 32 |
| 前馈隐藏维度 `dim_feedforward` | 2048 |
| 前馈激活 | ReLU |
| Dropout | 0.1 |
| `batch_first` | True |
| `norm_first` | False，Post-LN |
| LayerNorm epsilon | `1e-5` |
| Linear/MHA bias | True |
| 外层额外 Encoder norm | 未配置 |
| 因果遮罩、padding mask | 未传入 |

`2048`、ReLU 和 Post-LN 等没有在仓库构造调用中显式覆盖，来自所用 PyTorch `TransformerEncoderLayer` 默认值；已核对本地 PyTorch `2.4.1+cu121` 的签名。没有因果遮罩意味着四个已经可用的 token 相互可见；输入只包含当前及历史图像，不包含未来图像。

**单层 Transformer 内部包含多头注意力、前馈网络、两次残差连接和两次 LayerNorm。** 两个视角各有两层，因此学生共有四个这样的 Encoder 层。

![图10：TransformerEncoderLayer内部](assets/dq_high_level_networks/10_transformer_encoder_layer.svg)

*图 10：Post-LN 的准确顺序。注意力内部还有对注意力概率的 Dropout；该 Dropout 与后面的 `dropout1` 不同。*

用 $X\in\mathbb{R}^{B\times4\times64}$ 表示输入，计算可以写成：

$$
H=\operatorname{LN}_1\left(X+\operatorname{Dropout}_1(\operatorname{MHA}(X,X,X))\right),
$$

$$
\operatorname{FFN}(H)=W_2\operatorname{Dropout}\left(\operatorname{ReLU}(W_1H+b_1)\right)+b_2,
$$

$$
Y=\operatorname{LN}_2\left(H+\operatorname{Dropout}_2(\operatorname{FFN}(H))\right).
$$

前馈子网络为 `64→2048→64`，不能误写成和外层动作 MLP 一样的 `64→128→64`。LayerNorm 的缩放和平移参数可学习；每层在注意力路径与 FFN 路径各有一条残差连接。

**多头注意力内部也包含可学习的线性投影。** 为完整展示网络，下面把它进一步展开。

![图11：多头自注意力内部](assets/dq_high_level_networks/11_multihead_attention.svg)

*图 11：每层使用两头自注意力，每头 32 维；每个头的注意力矩阵为 `4×4`。*

PyTorch 在当前配置下用形状 `[192,64]` 的 `in_proj_weight` 和 `[192]` 的 bias 保存 Q/K/V 投影，等价于三个 `Linear(64,64)`。拆成两头后，Q/K/V 都为 `[B,2,4,32]`：

$$
\operatorname{head}_h=
\operatorname{Dropout}\left(\operatorname{softmax}\left(Q_hK_h^\top/\sqrt{32}\right)\right)V_h,
$$

$$
\operatorname{MHA}(X)=\operatorname{Linear}_{64\to64}\big([\operatorname{head}_1,\operatorname{head}_2]\big).
$$

Q/K/V 投影与最后的 `out_proj` 共 16,640 个参数；一整层 Encoder 为 281,152 个参数。每个状态引导分支包含两层 Encoder 的 562,304 个参数，加上 `state_proj` 的 3,968 个参数，共 **566,272**。

学生视觉时间建模的作用是利用连续帧中的目标位置、形状和深度变化来学习控制。代码没有显式速度预测头，也没有用物体速度标签监督 Transformer；不能把其 64 维输出直接等同于估计速度。

---

<a id="student-heads"></a>

**两路 Transformer 特征经过各自线性投影后，与机器人状态融合，联合预测全部动作。** `arm_proj` 和 `base_proj` 均为 `Linear(64,64)`，后面没有激活；分支名称表示视觉来源，不表示它们分别输出机械臂动作或底盘动作。

![图12：学生动作头与残差头](assets/dq_high_level_networks/12_student_action_heads.svg)

*图 12：残差头同时读取融合特征和主动作输出；最终动作是两者逐元素相加。两个动作 MLP 的所有层均已画出。*

$$
F=[\operatorname{arm\_proj}(h_{arm}),\operatorname{base\_proj}(h_{base}),r_t]
\in\mathbb{R}^{B\times(64+64+61)}=\mathbb{R}^{B\times189},
$$

$$
a_0=\operatorname{action\_head}(F),\qquad
\Delta a=\operatorname{residual\_mlp}([F,a_0]),\qquad
a_S=a_0+\Delta a.
$$

| 模块 | 层序列 | 输入维度 | 输出维度 | 参数数 |
|---|---|---:|---:|---:|
| `arm_proj` | `Linear(64,64)` | 64 | 64 | 4,160 |
| `base_proj` | `Linear(64,64)` | 64 | 64 | 4,160 |
| `action_head` | `Linear(189,128)→ELU→Linear(128,64)→ELU→Linear(64,9)` | 189 | 9 | 33,161 |
| `residual_mlp` | `Linear(198,128)→ELU→Linear(128,64)→ELU→Linear(64,9)` | 198 | 9 | 34,313 |

当前学生网络输出没有 Tanh，`DeterministicMixin(clip_actions=False)` 也不在策略层裁剪动作；环境继续按动作语义进行裁剪或缩放。模型的 `compute` 返回 `(actions, residual_actions, {})`，仓库修改过的 `DeterministicMixin` 会原样保留第二项。因此这里第二项是残差，不是 `log_prob`。见 [`动作头及前向`](../DQ_high-level/train_multi_bc_deter_DQ_stu.py#L60)、[`自定义 DeterministicMixin`](../third_party/skrl/skrl/models/torch/deterministic.py#L87)。

学生类虽然注册了 `log_std_parameter[9]`，但没有使用它，也不生成高斯动作分布。“确定性”指没有动作分布采样；训练模式的 Transformer 仍然有 Dropout。环境采样和评估时模型设为 eval，Dropout 关闭。

**教师和学生输出的 9 维动作具有相同语义。** 它们操作的是高层目标和命令，不是直接输出 19 个电机的目标。

| 动作索引 | 含义 | 后续处理 |
|---|---|---|
| `[0:3]` | 末端目标位置增量 | 裁剪/缩放后累加到当前目标位置 |
| `[3:6]` | 末端目标 RPY 增量 | 裁剪/缩放后累加到当前目标姿态 |
| `[6]` | 夹爪命令 | 转换为夹爪关节目标 |
| `[7]` | 底盘前进速度命令 | 写入 `commands[:,0]` |
| `[8]` | 底盘偏航角速度命令 | 写入 `commands[:,2]` |

不启用 pitch/floating 控制时，`commands[:,1]` 置零；观测仍保留完整 3 维命令。默认非 Tanh 路径中，末端位置增量限制为每轴 `±0.02`，姿态增量为每轴 `±0.06`。这里增量是相对于已有的末端目标累加，不应改写为每步直接相加到实际末端测量位置。来源：[`pre_physics_step`](../DQ_high-level/envs/b1z1_base.py#L1993)。

---

<a id="training"></a>

**教师网络通过 PPO 学习，学生通过教师动作标签学习，两者不共享参数。** 教师损失会更新 Actor 的物体编码器、注意力、动作 MLP 和 log_std，也会更新 Critic 的独立编码器、注意力和价值 MLP。学生的模仿损失不直接对齐教师特征或注意力权重；没有单独的特征蒸馏、抓取表示重建或 Transformer 速度预测损失。

![图13：学生分阶段训练和梯度范围](assets/dq_high_level_networks/13_student_training.svg)

*图 13：采集数据时两个策略都在 `no_grad` 下计算；训练学生时再对保存的观测重新前向。教师仅提供监督标签。*

训练器在同一状态下分别执行：

```python
teacher_obs = states["obs"]       # 1276维
student_obs = states["states"]    # 62269维
teacher_actions = teacher.act(teacher_obs)[0]
student_actions, residual_actions, _ = student.act(student_obs)
```

前 4000 个训练器 timestep 执行教师动作，之后执行学生动作；启用 `depth_random` 时预热为 8000 步。接管以后，教师仍对学生访问到的状态提供标签，这正是该 DAgger 训练路径的重要作用。这里 timestep 是并行环境的交互步，不是所有环境样本数的累加。

**动作控制权切换和残差参数训练切换是两个不同的时间点。** 默认训练器每收集 24 步执行一次更新，更新调用根据当前 timestep 判断阶段：

| 阶段条件 | 可训练参数 | 实际反向传播的目标 |
|---|---|---|
| `timestep < 210000` | 除 `residual_mlp` 外的学生参数；闲置 log_std 即使允许梯度也无使用路径 | `MSE(student_actions, teacher_actions)` |
| `timestep >= 210000` | 仅 `residual_mlp` | `MSE(residual_actions, teacher_actions - sampled_actions)` |

第一阶段为：

$$
L_{\mathrm{imitation}}=\operatorname{MSE}(a_0+\Delta a,a_T).
$$

冻结 `residual_mlp` 是冻结其参数，没有把残差输出 `detach`，也没有将残差置零。它始终参与前向，且固定残差函数仍可将损失梯度传回融合特征与主动作头。默认初始化也没有把最后一层置零，因此不能假设前期残差天然为零。源码虽然计算了一个 `residual_loss`，但这一阶段用于反向传播的是模仿损失。

第二阶段为：

$$
L_{\mathrm{residual}}=
\operatorname{MSE}\big(\Delta a_{\mathrm{current}},\ a_T-a_{S,\mathrm{collected}}\big).
$$

其中 `sampled_actions` 是数据采集时保存的完整学生动作 $a_{S,\mathrm{collected}}$，包含当时残差。它不是当前重新计算的主动作 $a_0$。因此该实现不能简化描述成“残差拟合教师动作减主动作”，也不能说第二阶段仍反向优化最终动作 MSE。第二阶段算出的 `imitation_loss` 只用于相关记录，实际调用的是 `residual_loss.backward()`。

第二阶段冻结的是上游参数，更新前训练器仍对学生调用 `set_mode("train")`，因此上游 Transformer 的 Dropout 仍开启；冻结参数不等于将这些模块切换为 eval。采集环境数据时则恢复 eval。

默认学生总训练时长为 README 中的 240000 步，因此会进入 210000 之后的残差训练阶段。教师基础学习率为 `4.2e-4`，学生为 `5e-5`；二者配置均为 24 rollout 步、5 epochs、6 mini-batches。学生默认使用 Adam 更新，未配置状态 RunningStandardScaler。这些是当前主路径配置，不是网络结构本身的硬性要求。

来源：[`学生训练配置及教师加载`](../DQ_high-level/train_multi_bc_deter_DQ_stu.py#L313)、[`DAgger 交互流程`](../DQ_high-level/learning/dagger_trainer.py#L114)、[`参数阶段切换`](../DQ_high-level/learning/dagger_rnn.py#L268)、[`模仿损失`](../DQ_high-level/learning/dagger_rnn.py#L343)、[`残差损失`](../DQ_high-level/learning/dagger_rnn.py#L432)。

名字 `DAgger_RNN` 不能证明这里有循环神经网络。当前 Policy 没有 GRU、LSTM，也没有声明 RNN hidden-state specification，训练器的 `_rnn` 为 False。时序结构就是显式堆叠的三帧视觉＋Transformer。`--mlp_stu` 目前只切换训练器，没有改动 Policy 的实例化，不能将它介绍为已经接好的纯 MLP 学生变体。

---

<a id="low-level"></a>

**高层控制还调用一套预训练底层网络，本文将它作为执行依赖单独展示。** 来源是 [`utils/low_level_model.py`](../DQ_high-level/utils/low_level_model.py)，高层通过 [`_load_low_level_model`](../DQ_high-level/envs/b1z1_base.py#L942) 构造并加载 checkpoint。这套模型不是前面高层 PPO 的 Actor/Critic，也不是视觉学生。

高层将底层模型设为 eval，通过 `torch.no_grad()` 和 `hist_encoding=True` 调用 `act_inference`。它不加入高层 PPO 或 DAgger 优化器；这里所谓“固定的底层策略”指不参与高层更新，代码并没有逐个把底层参数的 `requires_grad` 设置为 False。

启用 `--observe_gait_commands` 时，当前底层本体观测是 71 维：

| 内容 | 维度 |
|---|---:|
| 机身 roll、pitch | 2 |
| 机身局部角速度 | 3 |
| 不含夹爪的关节位置偏差 | 18 |
| 不含夹爪的关节速度，乘 `0.05` | 18 |
| 上一步底层腿部动作 | 12 |
| 足端接触 | 4 |
| 当前命令 `commands` | 3 |
| 当前末端目标位置 | 3 |
| `0 * curr_ee_goal_cart` 占位 | 3 |
| gait index＋4 个 clock inputs | 5 |
| **合计** | **71** |

实际部署输入为当前 71 维＋10 步历史 `10×71`，共 **781 维**，不含额外的 18 维特权状态。环境先拼接当前状态和已有历史，再把当前状态加入历史；新 episode 用当前状态重复初始化历史。来源：[`底层观测构造`](../DQ_high-level/envs/b1z1_base.py#L1129)。

![图14：底层Actor网络](assets/dq_high_level_networks/14_low_level_actor.svg)

*图 14：底层历史编码得到 20 维 latent，与当前 71 维状态拼接后，通过共享 Actor 骨干和腿、臂两个动作头。*

底层 Actor 的实际结构是：

```text
当前本体状态71 + 历史编码20 = 91
    → actor_backbone：Linear(91,128) + ELU
    ├→ 腿头：Linear(128,128)+ELU → Linear(128,128)+ELU → Linear(128,12)
    └→ 臂头：Linear(128,128)+ELU → Linear(128,128)+ELU → Linear(128,6)
拼接两个动作头 → 18维动作
```

这里腿头和臂头是实际分离的动作头，与视觉学生的 `arm/base` 特征分支含义不同。高层加载配置中 `output_tanh=False`、`adaptive_arm_gains=False`，因此末端无 Tanh，也没有额外的自适应增益输出。

**底层 `StateHistoryEncoder` 用一维卷积编码十步本体历史。** 它既不是学生视觉 CNN，也不是 RNN。

![图15：底层历史编码器](assets/dq_high_level_networks/15_low_level_history_encoder.svg)

*图 15：Conv1d 沿时间轴卷积。通道维为特征，时间长度由 10 变为 4，再变为 3。*

| 顺序 | 层/运算 | 输出形状 |
|---:|---|---|
| 0 | 历史输入 | `[B,10,71]` |
| 1 | 合并 batch 与时间，逐帧 `Linear(71,30)→ELU` | `[B×10,30]` |
| 2 | reshape＋permute | `[B,30,10]` |
| 3 | `Conv1d(30,20,kernel=4,stride=2)→ELU` | `[B,20,4]` |
| 4 | `Conv1d(20,10,kernel=2,stride=1)→ELU` | `[B,10,3]` |
| 5 | Flatten | `[B,30]` |
| 6 | `Linear(30,20)→ELU` | `[B,20]` |

两层卷积均为零 padding；最后一个输出 Linear 后仍有 ELU。模型也有 20/50 步历史对应的其他卷积配置，但高层实际实例化的是 10 步分支。来源：[`StateHistoryEncoder`](../DQ_high-level/utils/low_level_model.py#L39)。

**底层 checkpoint 容器还包含特权编码器和独立 Critic。** 它们会被实例化并加载权重，但高层实际 `hist_encoding=True` 的控制路径不调用它们，下面绘出以区分“模型中存在”和“当前执行使用”。

![图16：底层特权编码器与Critic](assets/dq_high_level_networks/16_low_level_priv_and_critic.svg)

*图 16：左侧特权编码器是 Actor 历史分支的替代分支；右侧为底层双价值头 Critic。它们不接收高层视觉输入。*

特权编码器为：

```text
privileged state 18
    → Linear(18,64) → ELU
    → Linear(64,20) → ELU
```

Actor 的 `hist_encoding=False` 使用这个 20 维 latent，`hist_encoding=True` 使用历史编码器的 20 维 latent。二者是**二选一**，不是同时拼接。高层实际只提供当前状态＋历史的 781 维布局，不应把部署输入中的第 71 维以后误当作特权状态。

底层 Critic 直接读取当前状态 71＋特权信息 18＝89 维，结构为：

```text
Linear(89,128) + ELU
    ├→ 腿价值头：128→128 + ELU →128 + ELU →1
    └→ 臂价值头：128→128 + ELU →128 + ELU →1
拼接 → [B,2]
```

Critic 不通过 Actor 的历史编码器或特权编码器；自己的两个价值头共用 Critic 骨干，但不与 Actor 共享权重。其 `[B,2]` 输出也不同于高层教师 Critic 的 `[B,1]`。底层模型另有 `[1,18]` 的标准差参数供随机 `act` 路径使用，高层采用的 `act_inference` 不使用它。来源：[`底层 Actor 分支`](../DQ_high-level/utils/low_level_model.py#L206)、[`底层 Critic`](../DQ_high-level/utils/low_level_model.py#L235)、[`底层推理接口`](../DQ_high-level/utils/low_level_model.py#L349)。

**底层网络虽然计算 18 维动作，当前高层环境中机械臂仍通过解析 IK 位置目标执行。** 这是理解整个控制结构时不可省略的连接。

![图17：高层动作到执行的完整路径](assets/dq_high_level_networks/17_action_execution.svg)

*图 17：绿色底层 Actor 是预训练神经网络；PD、目标累加、IK 和夹爪转换为解析控制，不包含需要训练的网络。*

腿部 12 维输出经关节重排、缩放和 PD 转成力矩。`_compute_torques` 显式将最后 6 个机械臂力矩置零，因此底层臂头虽然参与前向计算，其动作不会经这条力矩路径直接驱动机械臂。机械臂由目标位姿误差经阻尼最小二乘 IK 得到关节增量：

$$
\Delta q=J^\top(JJ^\top+0.05^2I)^{-1}\Delta x,
\qquad q_{\mathrm{target}}=q_{\mathrm{current}}+\Delta q.
$$

夹爪单独设置关节目标。环境分别下发腿部力矩和机械臂、夹爪位置目标。来源：[`力矩与 IK`](../DQ_high-level/envs/b1z1_base.py#L1193)、[`关节位置目标`](../DQ_high-level/envs/b1z1_base.py#L1702)、[`底层查询与仿真执行`](../DQ_high-level/envs/b1z1_base.py#L1927)。

---

<a id="verification"></a>

**参数量与维度已经根据实际类定义核对。** 高层模型使用源码原类独立实例化，避免触发 Isaac Gym 环境创建；底层模型按高层加载配置直接实例化。验证环境为本地 `dqwbc` Python、PyTorch `2.4.1+cu121`，随机输入批量 `B=2`。没有运行完整仿真训练或把随机前向当成策略性能验证。

| 网络 | 实际输入 | 核心融合维度 | 输出 | 注册参数总数 |
|---|---|---|---|---:|
| 高层教师 Actor | `[B,1276]` | 206 | 动作均值 `[B,9]`，log_std `[9]` | **871,768** |
| 高层教师 Critic | `[B,1276]` | 206 | `[B,1]` | **870,736** |
| 高层学生 Policy | `[B,62269]` | 189；残差输入 198 | 最终动作 `[B,9]`，残差 `[B,9]` | **5,367,339** |
| 底层 ActorCritic 整体 | 历史推理为 `[B,781]` | Actor 91；Critic 输入 89 | Actor 18；Critic 2 | **166,116** |

教师 Actor＋Critic 总参数为 **1,742,504**。Critic 和学生各自有 9 个未使用的 `log_std_parameter`，统计中保留它们；教师 Actor 的 9 个 log_std 是实际高斯策略参数，不应扣为闲置。

| 高层模块 | 教师 Actor | 教师 Critic | 学生 Policy |
|---|---:|---:|---:|
| 物体特征编码器 | 590,464 | 590,464 | — |
| 抓取注意力四个投影 | 9,926 | 9,926 | — |
| 主 MLP | 271,369 | 270,337 | — |
| 共享 CNN | — | — | 4,158,992 |
| 腕部 GuidedTransformerBlock | — | — | 566,272 |
| 前向 GuidedTransformerBlock | — | — | 566,272 |
| 腕部分支投影 | — | — | 4,160 |
| 前向分支投影 | — | — | 4,160 |
| 主动作头 | — | — | 33,161 |
| 残差动作头 | — | — | 34,313 |
| 声明的 log_std | 9 | 9，未用 | 9，未用 |

学生参数量大于教师 Actor，主要因为需要直接处理图像且包含两个 Transformer；这里“教师/学生”指监督关系和可用信息，不能理解为必然进行模型大小压缩。

| 底层模块 | 参数数 |
|---|---:|
| Actor 特权编码器 | 2,516 |
| Actor 历史编码器 | 5,610 |
| Actor 共享骨干 | 11,776 |
| Actor 腿头 | 34,572 |
| Actor 臂头 | 33,798 |
| Actor 合计 | **88,272** |
| Critic 骨干 | 11,520 |
| Critic 腿价值头 | 33,153 |
| Critic 臂价值头 | 33,153 |
| Critic 合计 | **77,826** |
| 标准差参数 | 18 |

独立验证确认：教师 Actor、Critic、学生的前向维度与反向传播通过；底层历史分支 `781→18`、特权分支 `89→18`、历史编码 `10×71→20` 和 Critic `89→2` 的前向维度符合本文。Actor/Critic 独立参数、学生 CNN 共享实例和两个 Transformer 独立参数均与实际构造一致。

**使用本图理解或修改代码时，应保留以下实现边界。** 教师固定假设候选数为 30，Query 输入为 134，观测尾部动作为 9；学生固定使用 61 维状态、54×96 图像的 CNN 展平尺寸，以及残差输入中的 `+9`。更改机器人、动作数、物体特征维度、相机模式或分辨率时，需要同步核对观测切片和网络层，不能认为命令行选项会自动适配全部结构。

| 容易误读的对象 | 当前代码的实际含义 |
|---|---|
| `PredictPoint` | 抓取候选坐标变换，不是神经网络 |
| `features.npy` 与抓取 JSON | 离线预计算输入；当前路径没有在线点云编码器或抓取提议网络 |
| `PredictAttentionSelector` | 单头注意力与线性投影，不是完整 Transformer，也不做 argmax |
| 教师 Actor/Critic | 同结构但不共享参数，使用同一类教师观测 |
| 学生 `arm/base` | 两个视觉特征分支，共同服务于完整 9 维动作输出 |
| 学生 `DAgger_RNN` | 支持循环网络的训练器名称，当前 Policy 没有循环状态 |
| 学生 `log_std_parameter` | 注册但未使用，不代表高斯学生 |
| `DepthFeatureExtractor`、`DepthOnlyFCBackbone*`、`CNNFeatureExtractor`、`Conv2dHeadModel` | 当前学生文件中的旧导入，不在默认 Policy 实例化路径中 |
| `--mlp_stu` | 当前只切换训练器；不是已实现的纯 MLP 网络开关 |
| 第一阶段残差头冻结 | 仅冻结参数；残差仍参与前向并传递输入梯度 |
| 底层臂动作头 | 模型中存在并前向计算，但高层执行用 IK 位置目标 |

为便于检查“涉及网络是否都画全”，各实际模块与图对应如下：

| 模块 | 覆盖位置 |
|---|---|
| 高层教师 Actor 全图 | 图 2 |
| 教师 Actor/Critic 各自的物体编码器 | 图 3；独立关系见图 2、5 |
| 抓取 Query/Key/Value/Output 四个投影 | 图 4 |
| 教师 Critic 与价值 MLP | 图 5 |
| 教师动作分布参数与输出 | 图 6 |
| 学生共享 CNN 的卷积、池化、全连接 | 图 8 |
| 两个 GuidedTransformerBlock、状态投影、位置编码 | 图 7、9 |
| Transformer FFN、LayerNorm、残差、Dropout | 图 10 |
| MultiheadAttention Q/K/V/Output 投影 | 图 11 |
| 学生两路投影、主动作头、残差头 | 图 12 |
| 底层 Actor 骨干及腿/臂双头 | 图 14 |
| 底层历史编码器逐帧 MLP、时间卷积、输出 MLP | 图 15 |
| 底层特权编码器、Critic 骨干及双价值头 | 图 16 |
| 模型与环境、解析执行模块之间的连接 | 图 1、13、17 |

**所有图都作为相对路径 SVG 嵌入本文件。** 在支持 SVG 的 Markdown 阅读器中可以直接查看，并打开原图放大；移动文档时应同时保留 `assets/dq_high_level_networks` 文件夹。图源集中保存在 [`render_diagrams.py`](assets/dq_high_level_networks/render_diagrams.py)，可用本地 Graphviz 重新生成：

```bash
python doc/assets/dq_high_level_networks/render_diagrams.py
```

脚本只依赖 Python 标准库和系统 `dot` 命令。本文的架构与解释以仓库可执行定义为依据，代码中沿用的旧维度注释没有作为结构判断依据。

---

<a id="high-level-rewards"></a>

**补充：KARL 筛选教师训练的奖励设计——论文与当前代码逐项对应**

本节针对下面这条从头训练命令，核对日期为 2026-10-06。解释依据包括 [DQ-NET 论文正文 Training Details](DQ-NET.pdf#page=5)、[附录 Reward Functions](DQ-NET.pdf#page=11)、[附录公式 (10)](DQ-NET.pdf#page=12)、[表 6：高层奖励](DQ-NET.pdf#page=13)，以及当前仓库的环境实现。论文与代码存在差异，以下明确区分两者。

```bash
python train_multistate_DQ_teacher.py \
  --task B1Z1PickMulti --grasp_selector karl \
  --karl_switch_margin_deg 30 --karl_orientation_preference none \
  --num_envs 512 --object_name green_bowl \
  --roboinfo --observe_gait_commands \
  --headless --sim_device cuda:0 --rl_device cuda:0 \
  --timesteps 80000 --seed 43 \
  --experiment_dir DQ_teacher/grasp_karl --wandb_name seed43
```

**最重要的结论是：KARL 改动替换了候选筛选及策略输入，奖励仍是 DQ 原高层任务奖励。** 训练目标是让机器人靠近物体、将物体抬高并保持，同时约束动作、底盘状态和转向；并没有新增“精确到达选中抓取位姿”的直接奖励。下面记录的是当前行为，本次文档补充没有改动奖励源码或 YAML 配置。

KARL 分支的 1102 维观测与 Actor/Critic 结构另见[实现与训练网络说明](DQ_teacher_KARL抓取筛选实现与训练网络.md)；本节关注环境奖励，不改变前文对原 GFM 网络的结构说明。

**1. 论文的设计思路，以及本命令实际采用的配置**

论文沿用 VBC 静态抓取奖励结构，并为动态全身抓取做调整：

| 论文中的类别 | 设计目的 | 主要项 |
|---|---|---|
| 任务奖励 | 把稀疏的“最终抓取成功”分解成容易探索的中间目标 | 接近、抬升进展、任务完成 |
| 辅助奖励 | 改善动作平滑性、朝向、底盘与机械臂协调 | 关节速度变化、动作变化、近目标速度、末端/底盘朝向、底盘距离与高度、偏航约束 |

论文附录描述任务奖励按阶段互斥启用，但**当前代码没有完整实现三阶段互斥开关**：尚未达到抬升成功条件时，接近与抬升进展两项可以同时为正。代码中有效的分阶段逻辑见本节第 4 部分。[R-论文][R-通用奖励][R-任务奖励]

这条命令未指定检查点，所以读取 `data/cfg/DQ_teacher.yaml`。当前文件中的任务等级已经是 **`Level01`**，不是先前架构文档生成时的 `Level00`；`--grasp_selector karl` 不会修改它。论文实验正文使用 Level 4 训练，因此这条命令本身不是论文完整训练设置的复现。[R-配置][R-入口]

| 实际配置 | 当前值 | 与奖励有关的含义 |
|---|---:|---|
| 训练环境 | `B1Z1PickMulti` | 采用该任务及其父类的奖励函数 |
| 物体 | `green_bowl` | 初始物体根位置相对桌面偏移 `init_height=0.026 m` |
| 抬升阈值 | `liftedSuccessThreshold=0.35 m` | 相对物体初始根位置的高度增量阈值 |
| 底盘距离目标 | `baseObjectDisThreshold=0.2 m` | 使用物体到机械臂基座的水平距离 |
| 目标底盘高度 | `base_height_target=0.55 m` | 抓起后鼓励底盘回到该高度 |
| 保持长度 | `holdSteps=25` | 连续满足抬升条件 25 个高层步时结束训练 episode |
| Episode 最大长度 | 100 高层步 | 超时结束 |
| 正奖励截断 | `only_positive_rewards=False` | 总奖励允许为负 |
| 相机 | `sensor.enableCamera=false` | 默认不启用相机方向/位置相关终止规则 |
| 总训练长度 | 80000 高层向量环境步 | `command_reward` 从第 40000 步起启用 |

`--num_envs 512` 表示每一个向量环境步并行收集 512 个转移，不会把每个环境的奖励乘以 512。80000 步对应约 4096 万个环境转移；第 40000 步是所有并行环境共同的训练进度，不是某个 episode 的第 40000 步。

默认仿真步长是 `0.005 s`，每低层周期执行 4 个物理步，每高层周期执行 8 个低层周期，所以一个高层步名义上是 `0.005×4×8=0.16 s`。保持 15/25 步分别约为 2.4/4.0 秒，100 步约为 16 秒。奖励每个高层步累计一次，**没有统一再乘 `0.16` 或 `0.02`**。[R-配置][R-调度]

**2. 奖励如何汇总：当前有 11 个非零配置项**

代码首先删去 YAML 中权重为零的项，再通过 `_reward_<name>()` 找到实际函数。`compute_reward()` 对所有有效项逐一计算、乘权重、相加。当前配置的完整表达式是：

\[
\begin{aligned}
r_t={}&0.5r_{\mathrm{approach}}
      +0.8r_{\mathrm{lift}}
      +3.5r_{\mathrm{completion}}\\
     &-0.001r_{\mathrm{acc}}
      +0.05r_{\mathrm{cmd}}
      -0.001r_{\mathrm{action}}\\
     &+0.01r_{\mathrm{ee\_orn}}
      +0.05r_{\mathrm{base\_dir}}
      +0.01r_{\mathrm{base\_approach}}\\
     &+0.5r_{\mathrm{base\_h}}
      +0.4r_{\mathrm{yaw}}.
\end{aligned}
\]

其中 `r_acc`、`r_action` 的函数输出是非负代价，乘负权重后产生惩罚；`r_yaw` 的函数输出已经是非正值，所以乘 **正的 0.4** 后才是惩罚。[R-汇总]

| 代码配置键 | 论文表 6 对应项 | 当前权重 | 当前实现的主要作用 |
|---|---|---:|---|
| `approaching` | `r_approach` | +0.5 | 末端到物体根位置的距离刷新历史最小时给分 |
| `lifting` | `r_lift` | +0.8 | 物体相对桌面的抬升量刷新历史最高时给分 |
| `pick_up` | `r_completion` | +3.5 | 训练时连续抬升达到 15 步后逐步给分 |
| `acc_penalty` | `r_acc` | −0.001 | 对机械臂关节速度变化构造惩罚；当前缓存时序有问题 |
| `command_reward` | `r_cmd` | +0.05 | 后半程、近目标时奖励较小的底盘前向速度命令 |
| `action_rate` | `r_action` | −0.001 | 惩罚相邻高层步的两个底盘动作分量变化 |
| `ee_orn` | `r_ee_orn` | +0.01 | 鼓励末端局部 +X 轴指向物体 |
| `base_dir` | `r_base_orn` | +0.05 | 本意是底盘朝向对齐；当前实现退化为零 |
| `base_approaching` | `r_base_approach` | +0.01 | 鼓励物体到机械臂基座的水平距离接近 0.2 m |
| `grasp_base_height` | `r_base_h` | +0.5 | 满足 `lifted_now` 时奖励底盘高度接近 0.55 m |
| `limit_yaw_rotation_penalty` | `r_yaw` | +0.4 | 偏航偏差超过 60° 时产生负奖励，超过 70° 还触发结束 |

`rad_penalty`、`base_ang_pen`、`gripper_rate` 的当前权重均为零，函数不会被奖励汇总器调用。源码中存在的 `reach`、`action_penalty` 也没有配置启用；注释掉的桌面接触惩罚不生效。不要仅凭函数名存在，就认为它参与了这条命令的优化。[R-配置][R-汇总]

**3. 理解阶段条件前，先区分两个“已经抬升”标志**

定义：

- `p_obj`：仿真中物体的根位置，不一定等于其几何中心或质量中心。
- `p_ee`：实际末端位置；`d=‖p_ee−p_obj‖₂`。
- `z_table`：当前桌面高度；`h_init`：物体根位置的初始桌面偏移。
- `h=z_obj−z_table−h_init`：物体相对于初始放置高度的抬升量。

每步在奖励计算之前，`check_termination()` 更新：

\[
L_t=\texttt{lifted\_object}
=\mathbf 1[h_t>0.35]\,\mathbf 1[d_t<0.2],
\]

\[
N_t=\texttt{lifted\_now}
=\mathbf 1[z_{obj,t}-z_{table,t}>0.015+0.35]\,\mathbf 1[d_t<0.2].
\]

二者不是同一个标志。对于 `green_bowl`，`h_init=0.026 m`，因此：

| 标志 | 物体根位置相对桌面的高度条件 | 用途 |
|---|---|---|
| `lifted_object` | 严格高于 `0.026+0.35=0.376 m`，且末端距离小于 0.2 m | 接近/抬升奖励开关、成功保持计数 |
| `lifted_now` | 严格高于 `0.015+0.35=0.365 m`，且末端距离小于 0.2 m | 抓起后的底盘高度奖励、与上一帧标志组合判定掉落 |

这属于基于高度与末端邻近程度的抓起判据，**没有在这里直接检查接触力、夹爪闭合状态或物体是否牢固夹持**。`liftedInitThreshold=0.05` 虽然在配置里，但当前启用的这些奖励/成功判断并未用它划分“开始抬升”的阶段。[R-任务奖励]

**4. 三个任务奖励的精确实现**

**4.1 接近奖励 `approaching`：奖励打破历史最小距离。**

令 `d_best` 为调用此奖励前保存的最近距离，`clip(x,0,10)` 为代码中的截断，则：

\[
r_{\mathrm{approach},t}
=(1-L_t)\tanh\!\left(10\,\operatorname{clip}(d_{best}-d_t,0,10)\right),
\qquad d_{best}\leftarrow\min(d_{best},d_t).
\]

例如历史最近距离为 0.30 m，本步到 0.28 m，得到加权奖励 `0.5×tanh(0.2)≈0.0987`。若先退到 0.40 m 再回到 0.30 m，不会重新得到接近奖励，只有比 0.30 m 更近才有进展分。这减少了在同一距离区间来回移动刷分的空间。

这里衡量的是末端到**物体根位置**的距离，物体自身移动也会改变该距离；并非末端到当前选中抓取候选的距离。子类中英文注释与实际布尔掩码不一致，应以 `reward *= ~self.lifted_object` 为准：未达到抬升成功条件时才启用。[R-通用奖励][R-任务奖励]

**4.2 抬升进展奖励 `lifting`：奖励刷新抬升高度纪录。**

令 `h_best` 为调用前保存的最高抬升量：

\[
r_{\mathrm{lift},t}
=(1-L_t)\tanh\!\left(10\,\operatorname{clip}(h_t-h_{best},0,10)\right),
\qquad h_{best}\leftarrow\max(h_{best},h_t).
\]

若本步比历史最高抬升量再高 0.01 m，加权奖励为 `0.8×tanh(0.1)≈0.0797`。保持同一高度没有进展分；下降后抬回旧纪录也没有新的进展分。

`h` 扣除了当前桌面高度和物体初始放置偏移，所以不能简单通过整张桌面抬高来增加相对抬升量。达到 `L_t=True` 的这一帧起，抬升进展奖励被置零，而不是达到阈值的那一步再额外给一笔抬升奖励。

历史缓存用负值表示未初始化，并在 `_update_curr_dist()` 中填入当前值；上面的历史纪录解释针对正常初始化后的阶段。当前实现没有增加“必须先抓住”“末端先小于某距离才开始给抬升进展”的额外开关，因而也不能将它直接解释为接触验证奖励。[R-通用奖励][R-距离更新]

**4.3 完成奖励 `pick_up`：训练时必须连续保持，但不是仅在成功时给一次奖励。**

对这条正常训练命令，记连续抬升计数为 `k_t`：

\[
k_t=\begin{cases}k_{t-1}+1,&L_t=1,\\0,&L_t=0,\end{cases}
\qquad r_{\mathrm{completion},t}=\mathbf 1[k_t\ge15].
\]

同时，当 `k_t≥holdSteps=25` 时置 `reset_buf=1`。

| 当前状态 | 接近奖励 | 抬升进展奖励 | 完成奖励的加权值 | 此条件是否要求结束 |
|---|---|---|---:|---|
| `L_t=False` | 刷新距离纪录时可为正 | 刷新高度纪录时可为正 | 0 | 否；但仍可能因其他失败条件结束 |
| `L_t=True`，保持第 1—14 步 | 0 | 0 | 0 | 否 |
| `L_t=True`，保持第 15—24 步 | 0 | 0 | 每步 +3.5 | 否 |
| `L_t=True`，保持第 25 步 | 0 | 0 | +3.5 | 是 |
| 评估模式第一次满足 `L_t=True` | 0 | 0 | +3.5 | 立即结束 |

因此，在没有其他提前终止的理想情况下，一次成功保持过程中的完成奖励会在第 15—25 步共发放 **11 次**，未折扣累计为 **38.5**。保持第 1—14 步虽然没有这三项任务奖励，仍可获得抓起后的底盘高度奖励及其他辅助奖励。

代码中的 `global_step_counter<1 or self.eval` 分支用于立即成功奖励/重置；普通训练先在 `post_physics_step()` 中将计数加到 1，故不能把它理解成“训练前几万步不需要保持”。训练中的 15/25 步规则从正常第一步就适用。[R-通用奖励][R-调度]

**5. 八个辅助项：公式、激活条件与实现差异**

**5.1 机械臂速度变化惩罚 `acc_penalty`，权重 −0.001。**

代码取 6 个机械臂关节、排除夹爪：

\[
r_{\mathrm{acc}}
=1-\exp\left(-\frac{\|\dot q_{arm}-\dot q_{arm,cache}\|_2}{\texttt{self.dt}}\right).
\]

与论文表 6 相比，它限定为机械臂关节，并多除以 `self.dt`；当前 `self.dt=4×0.005=0.02 s`，不是高层周期 0.16 s。

**但当前执行时序使这一项实际失效。** 每次 `_refresh_sim_tensors()` 都先把当前 `_dof_vel` 克隆到 `_last_dof_vel`。最后一个物理步已刷新过一次，紧接着 `post_physics_step()` 在计算奖励前再次刷新，中间没有新的物理步，于是“历史速度”被覆盖为当前速度。本机 4 环境、8 个高层步的短运行中，机械臂速度非零，但两份速度差及该奖励均为 0。不能据配置权重非零就认为当前训练已经惩罚了机械臂加速度。[R-通用奖励][R-速度缓存][R-调度]

**5.2 近目标减速奖励 `command_reward`，权重 +0.05。**

定义 `ρ=‖p_obj,xy−p_arm_base,xy‖₂`，`v_x*` 为环境执行的底盘前向速度命令。当前公式为：

\[
r_{\mathrm{cmd}}
=\mathbf 1[t_{global}\ge40000]\,
 \mathbf 1[\rho<0.2]\,\exp(-|v_x^*|).
\]

它与论文表 6 的 `−|v_x*|+0.25exp(−|v_x*|)` 不同：代码没有线性负项、没有 0.25 系数，还加了距离和训练进度门控。这实际上是一个正的减速奖励，不是单独的速度罚分；越接近停止，奖励越接近最大值 0.05。

例如 `ρ=0.1 m`、`v_x*=0.1 m/s`：第 39999 步为 0，第 40000 步起约为 `0.05×exp(−0.1)=0.04524`。这里没有新增退火，原实现是到达训练一半时硬开启；即使不传任何退火参数，此原有课程门控仍然存在。[R-通用奖励][R-入口]

**5.3 动作平滑惩罚 `action_rate`，权重 −0.001。**

\[
r_{\mathrm{action}}
=\|a_t[7:9]-a_{t-1}[7:9]\|_2.
\]

当前只约束 9 维高层动作中的第 7、8 两个零基索引，即底盘前向速度和偏航命令对应的动作分量，**不包含 6 维末端目标增量与夹爪动作**。论文的 `1−exp(−‖a_t−a_{t-1}‖)` 不是这里实际执行的公式。

这里使用环境保存的 `self.actions`，位于高层动作裁剪之后、按物理命令限幅/缩放之前；不能直接把它当作底盘实际速度变化。`last_actions` 在奖励计算之后才更新，因此这一项的前后步动作缓存与上述速度缓存问题不同。[R-通用奖励][R-调度]

**5.4 末端朝向奖励 `ee_orn`，权重 +0.01。**

令 `u_ee=R_ee[1,0,0]^T`，当末端与物体距离不少于 0.01 m 时：

\[
r_{\mathrm{ee\_orn}}
=u_{ee}\cdot\frac{p_{obj}-p_{ee}}{\|p_{obj}-p_{ee}\|_2};
\qquad d<0.01\ \text{时取 }0.
\]

它奖励末端局部 +X 轴朝向物体，值域为 `[-1,1]`，加权后在 `[-0.01,0.01]`。朝向相反时会产生负奖励。这是**单根轴与物体方向的对齐**，不是末端四元数与抓取候选四元数的完整姿态误差，也不会单独约束绕该轴的旋转。[R-通用奖励]

**5.5 底盘朝向奖励 `base_dir`，权重 +0.05，当前恒为零。**

论文表 6 的底盘朝向项权重为 0.25，当前 YAML 是 0.05。更关键的是，源码在计算物体方向后执行了：

```python
obj_dir = obj_pos - self._robot_root_states[:, :3]
obj_dir[:, :2] = 0.   # 将 x、y 清零，只保留 z
```

随后用这个竖直向量与仅含 yaw 旋转的底盘水平 +X 方向计算余弦，相互正交，因此结果为 0；高度差小于阈值的分支也直接为 0。它没有实现预期的“底盘水平朝向物体”。对多组水平朝向和目标位置的数值检查，以及本机短时 GPU 运行，都确认了这一结果。[R-通用奖励]

**5.6 底盘接近奖励 `base_approaching`，权重 +0.01。**

\[
r_{\mathrm{base\_approach}}
=1+\tanh(-10|\rho-0.2|),
\qquad\rho=\|p_{obj,xy}-p_{arm\_base,xy}\|_2.
\]

这是围绕目标水平距离的持续位置奖励，而不是刷新历史距离的进展奖励。在 `ρ=0.2 m` 时达到最大值 1，即每步加权 +0.01；太近或太远都会降低。论文表 6 写的是 0.6，当前配置是 0.2，而且代码的距离原点为**机械臂基座**，不是直接取机器人机身根位置。[R-通用奖励][R-配置]

**5.7 抓起后底盘高度奖励 `grasp_base_height`，权重 +0.5。**

\[
r_{\mathrm{base\_h}}
=N_t\exp(-|z_{base}-0.55|).
\]

指数形式对应论文，但代码多了 `lifted_now` 门控。未满足 `N_t` 时这项为零；满足后鼓励底盘高度接近 0.55 m，在目标高度处每个高层步加权 +0.5。该高度直接取仿真世界系机器人根位置的 z，并非额外估计的离地高度。[R-通用奖励][R-任务奖励]

**5.8 偏航限制 `limit_yaw_rotation_penalty`，权重 +0.4。**

令 `Δψ=|ψ_t−ψ_0|`，代码执行：

\[
r_{\mathrm{yaw}}
=-\mathbf 1[\Delta\psi>\pi/3]\tanh(\Delta\psi),
\qquad
\Delta\psi>70^\circ\ \Longrightarrow\ \texttt{reset\_buf}=1.
\]

`ψ_0` 来自创建机器人时保存的初始朝向，当前为 0；它没有在每次随机化 episode 初始 yaw 后重新赋值。代码直接对弧度作差取绝对值，没有角度环绕处理。未使用的 `delta_yaw_angle=torch.deg2rad(delta_yaw)` 不参与最终评分，不能据这一行推断实际阈值又被转换了一遍。[R-任务奖励][R-初始朝向]

有三点容易混淆：

1. **正文的“quadratic penalty”与公式并不一致。** 当前代码和附录公式 (10) 使用的是带门槛的 `−tanh`，没有平方，也不是对“超过 60° 的部分”做平方。超过 60° 时罚分直接跳到约 `−0.4tanh(π/3)≈−0.312`。
2. **论文表 6 的符号不能直接照抄成乘法。** 表中函数已经为负，却又列权重 −0.4；若二者直接相乘会变成奖励。代码用“负的函数值×正的 0.4”，得到实际负惩罚，也没有把论文中重复描述的 yaw 项再叠加一次。
3. **这与 KARL 的 30° 门槛是两回事。** KARL 门槛约束候选切换收益；这里的 60°/70° 约束机器人底盘偏航。

源码还有一个命名陷阱：`quaternion_tensor_to_euler()` 按 wxyz 解读参数，但调用传入 Isaac Gym 的 xyzw 后取第 0 项。将实际运算展开，该项恰好等于标准 xyzw 的 yaw：

\[
\operatorname{atan2}\left(2(q_xq_y+q_zq_w),\ 1-2(q_y^2+q_z^2)\right).
\]

因此不能仅凭注释“第 0 项是 roll”就断言当前算错了偏航。对 100 组随机旋转的原函数对照结果一致；59°、61°、71° 的加权奖励分别约为 `0、−0.3150、−0.3381`，仅 71° 触发该项的提前结束。[R-四元数]

**6. 终止条件也是任务设计的一部分，但不等于额外负奖励**

高层一步的关键顺序是：

```text
执行 8×4 个物理步
  → 更新高层计数、刷新状态
  → check_termination()：失败/超时，更新 lifted_now 与 lifted_object
  → compute_reward()：计算所有有效项，成功保持与 yaw 项还可设置 reset_buf
  → 更新 last_actions，生成观测
  → 包装器选择下一目标、返回 reward/done
  → PPO 存储终止转移，再重置对应环境
```

| 终止来源 | 当前触发条件 |
|---|---|
| 机身倾斜 | `|roll|>0.8 rad` 或 `|pitch|>0.8 rad`，约 45.8° |
| 机身过低 | 机器人根位置 `z<0.1 m` |
| 末端目标跟踪异常 | 机械臂局部目标 z 与实际末端局部 z 的差超过 0.2 m；这里仅检查 z 差，不是完整 IK 可达性判断 |
| 超时 | `progress_buf≥100` |
| 抬起后掉落判据 | **上一帧** `lifted_object=True`，但本帧 `lifted_now=False` |
| 物体落到桌面以下 | `z_obj+0.015<z_table` |
| 偏航过大 | `Δψ>70°`；在 yaw 奖励函数内部设置 |
| 训练成功结束 | 连续 `lifted_object=True` 达到 25 步；评估改成第一次满足就结束 |

相机方向/位置终止的开关默认取 `enableCamera`；当前教师配置是 false，因此本命令不启用那两项。`--headless` 本身不是关闭这些终止条件的依据。[R-终止][R-任务奖励]

配置没有 `termination` 奖励项，所以失败结束时不会自动追加统一的 `−1` 或其他常数惩罚，也不会自动把当步总奖励清零。提前终止主要通过截断后续收益影响优化。包装器将这些 `reset_buf` 作为 `terminated` 返回，`truncated` 为零；超时也走当前终止路径。[R-汇总][R-包装器]

另一个实现细节是：**若把 yaw 权重直接设为零，奖励准备阶段会连同该函数一起移除，函数内的 70° 终止也会随之失效。** 因此“去掉 yaw 奖励”在当前结构下同时改变了终止机制，并非纯粹只改一个奖励系数。

**7. 成功统计、环境奖励和 PPO 损失需要分开理解**

训练时，`success_counter` 在首次满足 `lifted_object=True` 且之前 `pick_counter<1` 时就增加；此时还没有达到 15 步的完成奖励门槛，更没有达到 25 步的成功结束门槛。若之后掉落，已经增加的成功计数不会因这次保持失败而撤销。因此训练日志中的成功计数/成功率**不等于“稳定保持 25 步”的完成率**。源码也保留了相关计数问题的注释。[R-通用奖励]

评估模式则首次满足抬升条件就给完成奖励并重置，不要求 25 步保持。所以比较教师效果时，应明确报告的是原评估成功率、训练计数还是持续保持成功率；奖励上升也不能单独证明抓取质量提高。

环境产生 `r_t` 后，当前 PPO 没有额外 `rewards_shaper`。PPO 用 `γ=0.99`、`λ=0.95` 计算回报和 GAE，进一步进行优势标准化、价值目标标准化及裁剪策略/价值损失计算。这里的 `r_t` 是强化学习信号，不是直接对 Actor 输出求导的监督损失。原学生主路径仍是 DAgger 的教师动作回归，环境奖励可以用于观察学生行为，但不替代其动作监督损失。[R-入口][R-PPO]

本任务调用冻结的低层策略执行底盘运动；论文表 5 中的腿部力矩、落脚、低层速度跟踪等奖励属于低层预训练，**没有被自动加到上面 11 项高层奖励中**。

**8. 这些结论对 GFM/KARL 对照实验意味着什么**

KARL 包装器只处理观测与筛选状态，原样返回环境的 `reward` 和 `done`。以下量没有直接进入奖励计算：选中候选索引、候选置信度、GFM 注意力权重、末端到选中候选的 SE(3) 距离、候选切换次数。也没有奖励梯度直接更新这个无参数选择器。[R-KARL包装器]

因此两种模型在**相同物理状态、动作及奖励历史缓存**下得到相同奖励；随着策略动作不同，后续轨迹和累计奖励当然可以不同。`ee_orn` 仍朝向物体根位置，`approaching` 仍接近物体根位置，而不是显式跟踪所选抓取。这一点有助于区分“选到了什么目标”和“奖励究竟要求机器人完成什么”。

若要判断替换 GFM 是否有效，首先应让两组保持相同任务等级、奖励实现、种子、训练总步数和评估协议。当前发现的底盘朝向项、速度缓存、成功统计等问题同时存在于两组环境中。本节先如实记录；如果后续修复，应在两组中同步使用，并将“奖励修复”与“候选筛选替换”分开做消融。

**9. 论文与代码差异速查及核验记录**

| 项目 | 论文描述/表 6 | 当前命令实际行为 |
|---|---|---|
| 任务阶段 | 三类任务奖励按阶段互斥 | 接近与抬升进展可同时启用；完成阶段有 15/25 步规则 |
| `r_acc` | 速度差的指数形式 | 6 个臂关节、额外除 `0.02 s`；重复刷新覆盖缓存，短运行实测为零 |
| `r_cmd` | `−|v|+0.25exp(−|v|)` | `exp(−|v|)`，且只在全局步数≥40000、水平距离<0.2 m 时启用 |
| `r_action` | `1−exp(−‖Δa‖)` | `‖Δa[7:9]‖₂`，仅两个底盘动作分量 |
| 底盘朝向 | 权重 0.25，方向对齐 | 权重 0.05，但水平/竖直方向构造使结果恒为零 |
| 底盘接近 | 目标距离 0.6 | 相对机械臂基座的水平距离，目标为 0.2 m |
| 底盘高度 | `exp(−|Hc−Ht|)` | 额外乘 `lifted_now`，目标 0.55 m |
| yaw 非线性 | 正文称 quadratic，公式 (10) 为 `−tanh` | 仅实现带门槛的 `−tanh` |
| yaw 权重 | 表 6 的负函数与 −0.4 存在符号歧义 | 负函数乘 +0.4，产生实际负惩罚 |
| 训练等级 | 正文报告 Level 4 训练 | 当前本地 YAML 为 `Level01` |

本次除了阅读源码，还执行了原函数的独立数值检查，没有手写另一套近似奖励替代验证：

| 检查 | 结果 |
|---|---|
| 未成功抬升，同时刷新距离/高度纪录 | 两项原始奖励同时为正，示例 `0.761594 / 0.099668` |
| 保持计数 1、14、15、24、25 | 加权完成奖励分别 `0、0、3.5、3.5、3.5`；仅第 25 步因保持条件结束 |
| 首次抬升的成功计数 | 第 1 个保持步即增加，早于完成奖励与 episode 成功结束 |
| 39999/40000 全局步，距离 0.1 m、命令 0.1 m/s | 原始近目标速度奖励由 `0` 变为 `0.904837` |
| 100 组随机四元数的 yaw 调用链 | 第 0 项与标准 xyzw yaw 计算一致 |
| 偏航 59°/61°/71° | 加权奖励约 `0/−0.3150/−0.3381`；71° 触发结束 |
| 本机 GPU：4 个 green_bowl 环境、Level01、8 高层步 | `base_dir` 与 `acc_penalty` 全程为零，臂关节速度最大绝对值约 1.58—4.72 rad/s，说明不是因为机械臂静止 |

短时 GPU 检查沿用命令的任务、奖励、80000 总步数配置与 KARL 选项，仅把环境数改为 4、设备改为本机 `cuda:1`，执行环境交互而未进行 PPO 权重更新。它用于验证奖励调用与缓存行为，不代表完整 80000 步训练的性能或成功率结论。

**本节来源索引（行号对应核验时的文件）：**

- [R-论文]：[DQ-NET.pdf](DQ-NET.pdf)，PDF 第 5 页 Training Details、第 11—12 页 Reward Functions、公式 (10)、第 13 页表 6。
- [R-配置]：[DQ_teacher.yaml](../DQ_high-level/data/cfg/DQ_teacher.yaml)，L1—33 的等级/阈值/物体设置，L224—246 的奖励权重与相机开关。
- [R-入口]：[train_multistate_DQ_teacher.py](../DQ_high-level/train_multistate_DQ_teacher.py)，`get_trainer()` 的配置读取、环境创建与 PPO 配置；[b1z1_pickmulti.py](../DQ_high-level/envs/b1z1_pickmulti.py) L34—39 的 `train_reward_strict=timesteps/2`。
- [R-通用奖励]：[reward_vec_task.py](../DQ_high-level/envs/reward_vec_task.py)，L11—74、L111—151、L175—179。
- [R-任务奖励]：[b1z1_pickmulti.py](../DQ_high-level/envs/b1z1_pickmulti.py)，L700—759 的成功标志/掉落/yaw，以及 L774—809 的高层任务奖励覆盖方法。
- [R-距离更新]：[b1z1_pickmulti.py](../DQ_high-level/envs/b1z1_pickmulti.py)，L555—562；物体 `init_height` 见 L130—141。
- [R-汇总]：[b1z1_base.py](../DQ_high-level/envs/b1z1_base.py)，L1631—1668 的奖励函数注册及求和。
- [R-速度缓存]：[b1z1_base.py](../DQ_high-level/envs/b1z1_base.py)，L1079 起的 `_refresh_sim_tensors()`。
- [R-调度]：[b1z1_base.py](../DQ_high-level/envs/b1z1_base.py)，`step()`、`post_physics_step()`，尤其 L2251—2260 的执行顺序；`self.dt` 在构造函数中设为低层周期。
- [R-终止]：[b1z1_base.py](../DQ_high-level/envs/b1z1_base.py)，L1606—1629；相机约束开关见构造函数的 `terminate_on_camera_constraint`。
- [R-初始朝向]：[b1z1_base.py](../DQ_high-level/envs/b1z1_base.py)，L903—915 的创建朝向与缓存，以及 `_reset_actors()` 的随机化处理。
- [R-四元数]：[torch_utils.py](../third_party/isaacgym/python/isaacgym/torch_utils.py)，`quaternion_tensor_to_euler()` 与 `euler_from_quat()`。
- [R-包装器]：[wrapper.py](../DQ_high-level/utils/wrapper.py)，`IsaacGymPreview3Wrapper.step()`。
- [R-KARL包装器]：[karl_teacher_wrapper.py](../DQ_high-level/utils/karl_teacher_wrapper.py)，`step()` 原样传递奖励及结束标志。
- [R-PPO]：[ppo.py](../third_party/skrl/skrl/agents/torch/ppo/ppo.py)，`record_transition()`、`compute_gae()` 与 `_update()`。

[R-论文]: DQ-NET.pdf
[R-配置]: ../DQ_high-level/data/cfg/DQ_teacher.yaml
[R-入口]: ../DQ_high-level/train_multistate_DQ_teacher.py
[R-通用奖励]: ../DQ_high-level/envs/reward_vec_task.py
[R-任务奖励]: ../DQ_high-level/envs/b1z1_pickmulti.py
[R-距离更新]: ../DQ_high-level/envs/b1z1_pickmulti.py
[R-汇总]: ../DQ_high-level/envs/b1z1_base.py
[R-速度缓存]: ../DQ_high-level/envs/b1z1_base.py
[R-调度]: ../DQ_high-level/envs/b1z1_base.py
[R-终止]: ../DQ_high-level/envs/b1z1_base.py
[R-初始朝向]: ../DQ_high-level/envs/b1z1_base.py
[R-四元数]: ../third_party/isaacgym/python/isaacgym/torch_utils.py
[R-包装器]: ../DQ_high-level/utils/wrapper.py
[R-KARL包装器]: ../DQ_high-level/utils/karl_teacher_wrapper.py
[R-PPO]: ../third_party/skrl/skrl/agents/torch/ppo/ppo.py
