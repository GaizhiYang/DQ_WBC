# 非对称视觉教师：M0、M1、M2 实现与运行

当前实现直接训练可部署 Actor，已删除退火计划、运行时 α、特权 Actor 过渡分支及对应命令行参数。默认 M2；也可单独运行 M0 或 M1。架构、字段和参考论文对应关系见[设计文档](DQ_asymmetric_teacher_design.md)及[架构图 PDF](assets/dq_asymmetric_teacher/dq_asymmetric_teacher_figures.pdf)。

## 版本与入口

| 模式 | 输入变化 | 训练变化 |
|---|---|---|
| `m0` | Actor：图像+机器人状态61；Critic：原教师特权观测 | 直接非对称 PPO |
| `m1` | Critic 追加任务状态5维 | `5→512` 适配器；迁移时零初始化 |
| `m2`（默认） | Actor/Critic 追加感知 belief16；Actor 图像使用同一感知扰动流 | 两相机延迟/丢帧/噪声、持续滤波、训练专用预测奖励 |

支持从零训练和教师权重迁移。迁移时，新适配器零初始化；原视觉教师的 CNN、Transformer、视觉适配器、控制后层、动作 log std 和 Critic 可迁移，Actor 融合首层仅保留61维可部署状态对应列。初始策略输出不保证等于原教师，需继续训练适应。从零训练则随机初始化高层网络及这些适配器。

## 运行环境

以下命令从 `DQ_high-level` 执行，使用本机已有 `dqwbc` 环境：

```bash
cd /home/hehui/DQ_WBC/DQ_high-level
conda activate dqwbc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

训练和独立仿真评估需要 Isaac Gym、GPU 渲染、现有机器人资产/低层 checkpoint。纯模型测试与部署加载不需要 Isaac Gym。

## 不指定 checkpoint：从零训练高层策略

```bash
python train_multistate_DQ_asymmetric_teacher.py \
  --num_envs 128 --object_name green_bowl --asymmetric_mode m2 \
  --headless --sim_device cuda:0 --rl_device cuda:0 \
  --roboinfo --observe_gait_commands \
  --experiment_dir DQ_teacher/asymmetric_m2 --wandb_name seed43
```

不传 `--teacher_init_checkpoint`、`--checkpoint` 或 `--resume` 时，从本地 YAML 配置开始新实验：高层 Actor、Critic、CNN、Transformer 和适配器均随机初始化，训练计数及原任务奖励课程从0开始。M0、M1同样支持。`--teacher_initial_step` 和 `--asymmetric_allow_legacy` 只用于指定了教师 checkpoint 的迁移实验。

**这里从零训练的是高层网络。** 原有低层运动控制器仍从 `data/low_model/model_44000.pt` 加载，不参加本次 PPO 更新。非对称架构本身不要求教师权重；迁移只是复用已有能力的可选初始化方式。

从零训练自动记录 `initialization: scratch`、`normalization: rollout`：第一轮使用均值0、方差1；每个完整 rollout 的 PPO 更新结束后，仅用本轮训练观测的1276维前缀更新统计，供下一轮使用。同一轮采样与全部 PPO epochs 使用同一份统计，图像、task5、belief16始终保持原量纲。value使用固定恒等变换，不将原始 returns 截断到±5。评估不更新统计，部署导出冻结当前统计快照。

## 指定教师 checkpoint：迁移训练

```bash
python train_multistate_DQ_asymmetric_teacher.py \
  --teacher_init_checkpoint DQ_teacher/vision_ablation/images_seed44/checkpoints/agent_57600.pt \
  --asymmetric_mode m2 \
  --headless --sim_device cuda:0 --rl_device cuda:0 \
  --roboinfo --observe_gait_commands \
  --experiment_dir DQ_teacher/asymmetric_m2 --wandb_name seed43
```

运行 M1 时改为 `--asymmetric_mode m1`，并使用不同实验目录；M0 同理。所有阶段从第一步就只使用可部署 Actor 输入。

不传 `--asymmetric_config` 时，继承源 checkpoint 中的环境、物体集合、任务等级、环境选项和原奖励课程长度，再附加默认非对称训练设置。源 checkpoint 的 `global_step` 决定课程起点，不从新文件名猜测。新实验可用 `--num_envs`、`--object_name`、`--vision_task_level` 调整范围。显式传 `--asymmetric_config data/cfg/DQ_asymmetric_teacher.yaml` 则采用该配置文件的环境设置；默认 Level01，研究 Level04 时应明确指定。

只有无视觉旧教师时，需要 `--asymmetric_allow_legacy`；视觉网络此时没有已训练视觉权重，不能当作视觉教师迁移对照。没有课程元数据且文件名无法确定步数时，再传 `--teacher_initial_step`。

视觉教师或旧非对称 checkpoint 继续使用原冻结统计。若源文件是采用 `normalization: rollout` 从零训练得到的非对称 checkpoint，新迁移实验继承该归一化模式与统计，以保持 value 单位一致。

## 配置

配置位于 [DQ_asymmetric_teacher.yaml](../DQ_high-level/data/cfg/DQ_asymmetric_teacher.yaml) 的 `asymmetric_teacher`：

```yaml
mode: m2
total_steps: 120000
rollouts: 24
minibatch_size: 128
learning_epochs: 5
learning_rate: 0.0001
checkpoint_interval: 2400
eval_interval: 2400
eval_steps: 1000
time_limit_semantics: task_deadline
reward_schedule_steps: 120000
perception_corruption: true
perception_corruption_eval: true
perception:
  camera_delay_frames: [1, 3]
  history_extra_frames: 2
  measurement_std_m: 0.04
  process_accel_variance: 0.5
  depth_noise_std_m: 0.003
  depth_noise_distance_scale: 0.003
  depth_noise_angular_scale: 0.002
  pixel_dropout_prob: 0.02
  frame_dropout_prob: 0.01
  frame_dropout_distance_scale: 0.01
  frame_dropout_angular_scale: 0.03
  small_target_dropout_scale: 0.05
  burst_length_min: 2
  burst_length_max: 5
perception_reward:
  weight: 0.02
  sigma_m: 0.10
  prediction_horizon_s: 0.15
```

完整可选感知字段见 `PerceptionConfig`。所有时间单位为秒，几何单位为米；延迟帧数指环境控制调用次数，原环境中步内的图像延迟还会通过真实采集时间戳体现。`process_accel_variance` 是连续白噪声加速度谱密度（m²/s³），不是每帧固定方差。

`total_steps` 是本次新实验的固定向量步预算。`--timesteps` 是本次运行希望达到的已完成步数，可分段运行，但不能超过预算，且应为完整 rollout 倍数。`num_envs*rollouts` 必须能被 minibatch_size 整除。可用 `--vision_minibatch_size`、`--vision_learning_epochs` 覆盖新实验设置。

消融时可将 `perception_reward.weight` 设为0，感知诊断仍保留。`perception_corruption: false` 同时关闭训练合成噪声、丢帧与额外延迟；需要干净评估时同时设置 `perception_corruption_eval: false`。默认评估保留同一感知困难条件，使用固定随机种子和策略均值。

原奖励课程的 `reward_schedule_steps` 仍保留，它控制原任务奖励，和已删除的 Actor 特权输入退火无关。为保持实验可比，不要把课程长度与新增预算混用。

## 多 GPU 同步训练

支持 M0/M1/M2 从零训练、权重迁移和同规模断点恢复。采用每张GPU一个进程的同步数据并行 PPO：每个进程运行独立 Isaac Gym 环境、相机与感知滤波，Actor/Critic 共同更新同一个策略。单纯把 `sim_device` 和 `rl_device` 指向不同GPU不是这种多卡训练。

下面命令用于当前工作站的两张 RTX 4090，**每卡64环境，总共128环境**，全局 minibatch 仍为128：

```bash
cd /home/hehui/DQ_WBC/DQ_high-level
conda activate dqwbc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

CUDA_VISIBLE_DEVICES=0,1 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  train_multistate_DQ_asymmetric_teacher.py \
  --num_envs 64 --vision_minibatch_size 64 \
  --object_name green_bowl --asymmetric_mode m2 \
  --headless --graphics_device_ids 0,1 \
  --roboinfo --observe_gait_commands \
  --experiment_dir DQ_teacher/asymmetric_m2 --wandb_name seed43_2gpu
```

这是 PyTorch [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html) 的模块启动形式，不需要另加 `--distributed`。各进程按 `LOCAL_RANK` 自动使用 `cuda:0`、`cuda:1`，不用分别传 `--sim_device`/`--rl_device`。如果使用这些参数，多进程入口仅接受默认的 `cuda:0`/`cuda`，然后按rank分配，避免把所有进程错误绑定到一张卡。

本机默认 Vulkan 枚举为GPU 0、软件渲染器、GPU 1。Isaac Gym 的相机张量设备编号在这种混合枚举下会出错。上面的 `VK_ICD_FILENAMES` 仅为本次运行加载 NVIDIA ICD，已实测将两张卡的渲染编号统一为0、1；不要在这条命令中使用过滤前的 `0,2`。它不修改系统驱动。其他机器需确认本地 NVIDIA ICD 文件路径和CUDA/Vulkan设备对应关系；`CUDA_VISIBLE_DEVICES`本身不会同步重排Vulkan设备。

| 参数/预算 | 含义 | 上述双卡示例 |
|---|---|---|
| `--num_envs` | 每个进程的环境数 | 每卡64，全局128 |
| `--vision_minibatch_size` | 每卡每次梯度更新的样本数 | 每卡64，全局128 |
| `rollouts` | 每次更新前收集的向量步数 | 24 |
| 每次完整 rollout 样本数 | GPU数 × 环境数/卡 × rollouts | 3072 |
| `--timesteps`/`total_steps` | 同步的训练向量步，不乘GPU数 | 默认120000 |
| 总交互数 | GPU数 × 环境数/卡 × completed_steps | 完整训练15360000 |

如果希望每张卡各128环境，改成 `--num_envs 128 --vision_minibatch_size 128`，此时全局环境数256、全局minibatch256，总交互数也增加。不会自动扩大本地配置中的环境数或训练预算。

实现细节：初始化后广播 Actor/Critic 与预处理统计；使用不同rank的采样/感知随机种子；全局标准化advantage；每个minibatch平均梯度后统一裁剪、执行Adam；跨卡平均KL使学习率和提前停止决策一致。从零训练的前缀统计使用所有rank的原始观测，并在完整 PPO 更新后统一更新。旧教师迁移的冻结统计保持原语义。

评估由所有进程同时运行，再按实际完成回合数汇总GSR/OSSR/回报/TSC；感知指标按有效记录数汇总。只有rank0写配置、TensorBoard/W&B、`evaluations.jsonl`、checkpoint及导出文件。PPO损失和评估指标是全局值，常规训练环境reward/感知诊断日志取rank0本地环境。

多卡迁移时，在上述命令追加 `--teacher_init_checkpoint <教师或非对称checkpoint>`。双卡完整恢复示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json \
OMP_NUM_THREADS=1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  train_multistate_DQ_asymmetric_teacher.py \
  --checkpoint DQ_teacher/asymmetric_m2/seed43_2gpu/checkpoints/agent_24000.pt \
  --timesteps 48000 --headless --graphics_device_ids 0,1 \
  --experiment_dir DQ_teacher/asymmetric_m2 --wandb_name seed43_2gpu
```

完整续训要求GPU进程数、每卡环境数和minibatch配置与保存时一致。改变卡数时，通过 `--teacher_init_checkpoint` 新开实验；原checkpoint仍可单卡评估和CPU导出。当前已验证单机双卡，未验证多节点或多卡带来的长期收敛/速度收益。

## 断点恢复与旧版本迁移

```bash
python train_multistate_DQ_asymmetric_teacher.py \
  --checkpoint DQ_teacher/asymmetric_m2/seed43/checkpoints/agent_24000.pt \
  --timesteps 48000 --headless --sim_device cuda:0 --rl_device cuda:0 \
  --experiment_dir DQ_teacher/asymmetric_m2 --wandb_name seed43
```

也可用 `--resume` 自动寻找指定实验目录中最大编号的 `agent_*.pt`。版本2 checkpoint 恢复 Actor/Critic、Adam、学习率调度器、归一化模式及统计、课程位置、已完成预算和最佳评估分数；不能在完整恢复中更改网络模式或 PPO 样本布局。早期版本2没有归一化模式字段时，按原冻结模式恢复。

从 M0 升级到 M1/M2，或从旧版1 checkpoint 迁移，使用 `--teacher_init_checkpoint` 开启新实验。新版加载器会丢弃旧文件中的过渡分支及 α 状态，仅迁移可用权重，并初始化新适配器/优化器。兼容加载中保留对旧键名的识别，不代表运行时仍有退火。旧版1不支持 `--checkpoint` 完整续训。

仿真物理状态、随机流、相机/滤波历史及未完成回合不保存；恢复从新回合开始，并非逐步一致的进程快照。

## 评估和日志

训练默认每2400步及训练结束执行评估。评估只用策略均值；统计 GSR、OSSR、TSC、原任务 reward/episode return，M2 的感知 bonus 不加入评估回报。最佳文件为 `best_agent.pt`，仅依据完成回合的 GSR 选择；没有完成回合时不选择最佳模型。

评估写入实验目录的 `evaluations.jsonl`。M2 还记录可见表面代理预测误差、初始化率、有效观测率、观测年龄、预测年龄、迟到拒绝计数和感知奖励均值。预测误差没有有效样本时不伪造零值。训练指标在 `Perception / ...`，评估在 `Evaluation perception / ...`。

评估期间冻结训练课程位置，保存/恢复训练统计与随机流，结束后完整重置仿真和感知状态再收集下一 rollout。原 OSSR 沿用仓库的启发式定义，TSC 单位为高层控制步。

```bash
python play_multistate_DQ_asymmetric_teacher.py \
  --checkpoint DQ_teacher/asymmetric_m2/seed43/checkpoints/best_agent.pt \
  --timesteps 5000 --headless --sim_device cuda:0 --rl_device cuda:0 \
  --experiment_dir DQ_teacher/asymmetric_m2 --wandb_name evaluation
```

独立评估不执行 PPO 更新或保存训练 checkpoint。

## 导出与部署

训练结束自动生成 `actor_N.pt`。也可不启动仿真单独导出：

```bash
python export_asymmetric_teacher.py \
  --checkpoint DQ_teacher/asymmetric_m2/seed43/checkpoints/agent_120000.pt \
  --output DQ_teacher/asymmetric_m2/seed43/actor_deploy.pt
```

导出只含 Actor、61维归一化统计、log std 元数据、feature flags、图像和 belief 协议及感知配置。没有 Critic、物体编码器、GFM或物体真值输入。

M0/M1 直接调用：

```python
from utils.asymmetric_teacher_preprocessor import load_deployment
policy = load_deployment("actor_deploy.pt", device="cuda:0")
# images: [B,12,54,96]，mask与depth/3，三帧由旧到新
# raw_proprio61: [B,61]，已按原观测约定预缩放，但未标准化
mean_actions = policy(images, raw_proprio61)
```

M2 推荐使用完整感知运行时，而非自行拼接 belief：

```python
from utils.asymmetric_runtime import M2DeploymentRuntime
runtime = M2DeploymentRuntime("actor_deploy.pt", num_envs=1, device="cuda:0")
# images: [B,2,2,54,96]，相机顺序base/wrist，模态mask/正光学深度(米)
# K: [B,2,3,3]；T_world_camera: [B,2,4,4]，使用图像采集时刻位姿
# timestamps: [B,2]；now: [B]，同一单调时钟，建议float64
# T_world_base: [B,4,4]，当前控制时刻位姿
mean_actions = runtime.step(
    images=images, T_world_camera=T_world_camera, intrinsics=K,
    timestamps=timestamps, T_world_base=T_world_base, now=now,
    raw_proprio=raw_proprio61,
)
# episode结束或切换跟踪目标时：
runtime.reset()
```

缺失相机提供零图像及它最近的采集时间戳；控制时钟继续前进。部署关闭合成扰动/延迟，实际异步延迟由采集时间戳处理。历史窗口需覆盖实际传感器时延；太旧的测量明确拒绝。完整运行时使用与训练相同的滤波、图像尺度与三帧历史。

可部署感知的边界是输入的 mask/depth；实际检测器、同步标定、机身位姿估计及机械臂 FK 仍需接入真实系统。滤波输出是可见表面代理，不能直接解释为精确物体中心。动作的执行缩放、裁剪与安全限制仍由原控制器处理。

## 主要文件与验证

| 文件 | 职责 |
|---|---|
| [modules/asymmetric_teacher.py](../DQ_high-level/modules/asymmetric_teacher.py) | 直接 Actor、M1/M2 适配器、权重迁移 |
| [utils/asymmetric_perception.py](../DQ_high-level/utils/asymmetric_perception.py) | 传感器扰动、时间戳重放、持续 KF、belief 协议 |
| [utils/asymmetric_camera.py](../DQ_high-level/utils/asymmetric_camera.py) | 采集时刻几何、世界系对齐、奖励表面参考 |
| [utils/asymmetric_teacher_wrapper.py](../DQ_high-level/utils/asymmetric_teacher_wrapper.py) | 任务5维、感知历史、GT隔离、训练奖励 |
| [utils/asymmetric_runtime.py](../DQ_high-level/utils/asymmetric_runtime.py) | 无仿真依赖的 M2 部署前端 |
| [utils/asymmetric_teacher_preprocessor.py](../DQ_high-level/utils/asymmetric_teacher_preprocessor.py) | 部署归一化及格式校验 |
| [utils/asymmetric_teacher_training.py](../DQ_high-level/utils/asymmetric_teacher_training.py) | PPO、预算、版本2恢复 |
| [utils/asymmetric_scratch_preprocessor.py](../DQ_high-level/utils/asymmetric_scratch_preprocessor.py) | 从零训练的轮次边界统计与恒等value变换 |
| [learning/asymmetric_teacher_trainer.py](../DQ_high-level/learning/asymmetric_teacher_trainer.py) | 完整更新边界、隔离评估、日志和最佳模型 |
| [learning/asymmetric_distributed_ppo.py](../DQ_high-level/learning/asymmetric_distributed_ppo.py) | 多进程全局advantage、梯度和KL同步 |
| [utils/asymmetric_distributed.py](../DQ_high-level/utils/asymmetric_distributed.py) | torchrun设备与生命周期、全局统计及评估汇总 |

从仓库根目录执行测试：

```bash
python -m unittest discover -s tests -p 'test_asymmetric*.py' -v
python -m unittest discover -s tests -p 'test_teacher_vision*.py' -v
```

验证覆盖 Actor 无 GT 输入、M1/M2 可训练且初始化保持已有分支、实际 PPO 更新与恢复、异步 KF 对照 NumPy 历史重放、连续缺测、局部重置、绝对时间戳、相机坐标、GT与奖励隔离、评估隔离及导出部署动作一致性。

2026-10-05 验证结果（含从零训练及多卡扩展）：**105项非对称相关测试、30项原视觉教师回归测试全部通过**，`git diff --check` 通过。多卡CPU测试实际启动两个Gloo进程，检查梯度平均、全局统计、模型/Adam/学习率一致性、加权评估、单进程写文件及恢复导出。

真实 Isaac Gym/CUDA 0 短测采用2个 sugar_box 环境、rollout4、1 epoch、4步任务截止：M1训练4步；M2先训练8步，保存后恢复至24步，完成周期评估、6步独立评估和两种导出路径一致性检查。新增Actor belief与Critic task/belief适配器均获得非零更新，checkpoint预算和课程计数正确。

多环境相机反投影已对照目标几何核验，修正了相机全局坐标与环境局部坐标的偏移；部署运行时与训练端的图像历史、belief和动作已逐步比较一致。早期CUDA 1相机互操作错误已在本轮定位为混合Vulkan设备枚举问题：只加载NVIDIA ICD并采用正确的渲染设备参数后，双卡相机与训练均通过，见多卡命令。

从零训练扩展另以2个 green_bowl 环境完成无高层checkpoint的8步训练、保存后恢复至16步、周期评估和导出一致性验证。课程起点为0，观测统计样本数依次为17和33（包括1样本初始先验），value保持原始reward单位。CPU测试验证视觉编码器第一次反向传播即有非零梯度，同一轮采样/PPO更新的归一化和动作概率一致。

双卡实测使用本机两张RTX 4090、NCCL、每卡2个green_bowl环境，训练8步后恢复至16步。恢复后逐个张量确认两卡的Actor、Critic、Adam状态及归一化统计完全一致；前缀统计全局计数为65（4环境×16步+1先验）。全局评估只写一条记录；多卡checkpoint的单卡6步评估及CPU导出也通过，导出权重/统计与运行中导出逐值相同。

上述短测用于工程验收，不用于评价收敛或抓取成功率；尚未启动正式长程训练。
