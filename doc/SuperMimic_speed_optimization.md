# 世界模型训练提速与 cuda:0 验证

日期：2026-11-06。入口：`DQ_high-level/train_teacher_sgs_ekf_world_model.sh`。

## 结果

**下表是最初执行提速的单块历史测量。后续已重新按源码修正损失、动作坐标和 Critic；最新结构及逐节点对照见 [对齐实现说明第0节](SuperMimic_alignment_implementation.md)，新多块运行见 [稳定性报告](SuperMimic_world_model_stability_gpu.json)。不能用单块通过代替多块稳定性或最终收敛。**

已在 **cuda:0 的 48 GB RTX 4090** 上，使用完整的 **4096 个 Isaac Gym 环境**、高层回放256步、低层回放1000步，完成首个真实 PPO rollout 及全部60轮模型／想象更新。

| 测量项 | 结果 |
|---|---:|
| 真实采集 | 24个高层步＝192个低层步 |
| 模型更新轮数 | 60 |
| 动力学 Adam 更新 | 600次，未缩减10批×2048窗口×8步预算 |
| Actor BPTT 更新 | 60次，全部接受 |
| 每轮模型拟合平均耗时 | **1.981秒** |
| 每轮Actor＋想象Critic平均耗时 | **0.797秒** |
| 两阶段平均合计 | **2.779秒／轮** |
| 60轮模型／想象更新块 | **约166.8秒** |
| 加上真实PPO的更新块 | **约167.4秒** |
| 本次真实采集＋训练调用 | **178.13秒** |
| 含环境构建及检查的诊断总时间 | 247.93秒 |
| 峰值Torch已分配显存 | **20884.89 MiB，约20.40 GiB** |
| 检查末尾设备剩余显存 | 14569.88 MiB，约14.23 GiB |

每条想象轨迹仍为4次高层决策×8次低层闭环＝32个20 ms模型步。UTD仍为0.4，首块仍执行60轮。此次提速来自执行方式的优化，没有减少学习预算或缩短想象。

原始结果：[生产规模GPU验证](SuperMimic_speed_production_gpu_validation.json)、[小场景完整批量验证](SuperMimic_speed_small_gpu_validation.json)。

这是首个完整采样／更新块的实测，尚未运行40000个高层步的完整训练，也不声称抓取成功率已提升。回放按生产容量分配，但首块只采到了192个低层步，后续填满后的索引缓存占用和任务耗时还可能变化。Torch峰值不包含PhysX、驱动等非Torch分配；设备剩余显存还受其他进程影响。

此前优化前诊断见 [流程与性能审阅](SuperMimic_training_flow_performance_review.md)。旧诊断在cuda:1上使用12环境小回放，不能拿它与本次生产规模直接计算严格的端到端加速倍数。

## 后续：多更新块数值稳定性修复

首块性能通过不等于多块交替训练稳定。后续实际运行出现过：第144步（另一运行是第384步）真实PPO结束后，整个BPTT块的Actor更新都被拒绝，恢复采集后PhysX报CUDA 700。该错误表示设备非法内存访问，并不等同于显存不足；故障后的CUDA上下文不能继续正常使用，需要重新启动训练进程。[NVIDIA CUDA错误定义](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html)

已修复两处可独立复现的数值问题，并补齐仿真入口保护：

1. **PPO概率比的无效溢出分支。** 原实现先计算 `exp(new_logp-old_logp)` 再裁剪，float32下即使最终loss有限，反向仍可能出现 `0*inf -> NaN`。本地 `utils/finite_ppo.py` 对正／负优势分别在log域执行等价裁剪再取指数。负优势下真正无界的溢出明确拒绝该minibatch，不用任意截断替换原目标。
2. **饱和夹爪opening的反演误差。** float32中多个不同Gaussian样本可能映射到同一opening；从opening逆推不能恢复原样本。世界模型入口现在在PPO专用memory额外保存实际采样latent，直接用该latent的Gaussian log-prob计算ratio。固定样本的sigmoid Jacobian在新旧比率中抵消。发送给环境的opening、真实随机采样、BPTT均值动作均保持原样；标准差仍仅由真实PPO更新。新增回放约3.38 MiB。未启用世界模型的旧PPO接口保持兼容。
3. **优化器保护。** PPO检查输入、loss、梯度及联合梯度范数；优化后若参数或Adam状态非有限，恢复该步之前的参数和Adam状态。初始已坏的状态明确报错。BPTT也检查自己的Adam状态，并打印拒绝原因；不会再拿已损坏的Actor无信息地重复跑完整更新块。
4. **进入PhysX前检查。** 原始动作在clamp前检查；每个物理子步的关节目标、力矩、将上传的root states及SGS reset DOF先统一检查。异常直接报告阶段、步骤和环境编号，阻止坏输入上传。正常路径在GPU上聚合，仅读取一个判断标量，不用 `nan_to_num` 掩盖来源。

这些改动保留2048×3动力学网络、4096环境、10×2048拟合预算、UTD=0.4、4×8=32步想象、学习率及KL gate配置；不依靠缩减训练任务绕过问题。没有修改vendored skrl。

新增 `tests/smoke_world_model_stability_gpu.py` 用完整生产参数验证多个更新块；`--timesteps=40000`保留原课程，只用`--stop_after`限制诊断长度。逐步保存输入、参数、Adam状态、CUDA位置、PPO/BPTT接受数和夹爪饱和情况。保护性拒绝单个minibatch可被记录，但整块PPO或BPTT无有效更新不能算通过。运行方式：

```bash
python DQ_high-level/tests/smoke_world_model_stability_gpu.py \
  --device cuda:0 --high-steps 168 \
  --output doc/SuperMimic_world_model_stability_gpu.json
```

实际GPU复测结果以该JSON的`passed`、完成步数与逐块记录为准；单元测试复现数值漏洞并不等于已捕获用户原运行的首次故障张量。

### 按源码重审后的结构修复

仅保留以上数值保护的第一次长测，在163个高层步时拦截到非有限公开log-prob；此前第144步后的动作均值最大绝对值已约9022。进程在进入PhysX之前停止，未触发新的700或段错误。记录保存在 [修正前动作放大证据](SuperMimic_unbounded_action_failure_gpu.json)。另一次带诊断的原代码168步运行没有复现native崩溃，见 [原路径诊断](SuperMimic_crash_diagnostic_baseline.json)；因此没有声称已找到用户原运行唯一触发CUDA700的张量。

随后重读SuperMimic源码，修正了三个相关结构问题：原始动作可以绕过已饱和的物理控制直接影响Critic；Gaussian采样与厘米级命令量纲不匹配；缺少源中的全维平方动作变化成本。同时将动力学默认损失改回源的标准化残差平方和，取消额外FK拟合分支。具体实现、剩余差异与测试见 [对齐实现说明](SuperMimic_alignment_implementation.md)。

没有启用先前试拟的默认 `clipActions=1`，原始动作仍可无界；新运行要靠正确动作尺度、物理价值输入和一致奖励保持合理行为。有限值保护用于在异常时报告故障位置，不把“没有段错误”当成学习有效的证明。

### 修正后的生产规模复测结果

[最终JSON](SuperMimic_world_model_stability_gpu.json) 的 `passed=true`、`completed=true`：cuda:0、4096环境、168高层／1344低层步、7个更新块，完成5210次动力学Adam、521/521次Actor BPTT、210/210个PPO minibatch，无保护性拒绝／回滚，未出现CUDA700或段错误。已实际穿过第144步完整模型块，再返回PhysX运行到168步。冻结低层未变，σ只由PPO更新，所有检查的参数／优化器状态有限且CUDA驻留。

模型拟合平均 **1.301秒／轮**，Actor＋想象Critic平均 **0.786秒／轮**；峰值Torch分配约 **20.36 GiB**。诊断总时长1238.7秒，含构建场景及频繁检查。与开头历史单块相比，当前损失与Critic输入已改变，因此这些数字是新实现的实测耗时，不是保持数学目标不变的严格A/B加速比。完整40000步收敛和另一运行的384步崩溃位置未由本次短测覆盖。

## 修改内容

### 1. 默认设备和批量执行

- 世界模型启动脚本默认 `RL_DEVICE=cuda:0`，`SIM_DEVICE`默认跟随RL设备。
- 动力学 `model_microbatch_size` 从256改为2048，一批无需串行拆成8块。
- 想象 `imagination_chunk_size` 从128改为4096，一次处理整个逻辑批量，无需串行执行32块。
- 保留CLI覆盖，较小显存的设备仍可使用梯度累积。
- 模型结构、逻辑批量、拟合窗口、想象长度、优化器更新次数与学习率不变；没有加入编译或多GPU训练。

### 2. 末端专用FK

`B1Z1Kinematics.end_effector(...)` 只遍历末端所需的9个关节节点；原完整FK仍保留26节点路径和四足输出。训练适配器改用末端路径。

两条路径共用运动学递推及原buffers。末端位姿、线速度、角速度以及对输入的导数与原完整路径一致，不改变checkpoint布局。

### 3. 同一终点复用几何和任务结果

- 奖励和终止共用局部几何结果。
- `finish_high` 可以同时返回状态、奖励和outcome，想象循环直接复用。
- 每个高层终点原来的4次FK降为1次；高层策略观测所需的起点FK仍独立计算。
- 没有把带梯度的结果缓存到对象上；没有跨想象步复用旧状态计算图。
- 保留成功、失败、超时、已终止行的吸收语义及全部历史递推。

### 4. 回放采样缓存

同一学习块中重复抽样时，复用对应窗口长度／split的合法窗口掩码、候选索引和当前采样参数的权重。

- 采集追加、环覆盖或索引相关Tensor被修改后失效。
- 原采样分布、`multinomial`抽样和随机数消费保持一致。
- 不改变episode隔离、终止末步保留或train／validation划分。
- 缓存占用纳入`memory_bytes`；采样参数切换不会无限积累权重版本。

## 验证

完整GPU训练诊断确认：

- 真实PPO仍更新Actor标准差和自己的Critic。
- BPTT更新Actor和独立想象Critic，未更新世界模型、真实Critic或标准差。
- 冻结低层策略始终不变。
- 策略、模型、两个Critic、FK、缩放／噪声模块和两套回放均在cuda:0。
- 高层无噪声观测重建最大误差为`3.19e-4`。
- 没有OOM，60次BPTT全部通过数值检查。

CPU回归：世界模型相关224项中，127项通过，97项CUDA专用项按无GPU模式跳过。随后在cuda:0单独运行运动学、终点复用、想象Critic、微批等价和层级动力学专项，**37项CUDA测试全部通过**。另通过启动脚本语法、默认设备dry-run和`git diff --check`。

## 运行

```bash
conda activate dqwbc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh
```

脚本现在默认使用cuda:0。若当前shell已有其他设备覆盖，可以明确指定：

```bash
RL_DEVICE=cuda:0 SIM_DEVICE=cuda:0 \
bash DQ_high-level/train_teacher_sgs_ekf_world_model.sh
```

复现生产规模检查（使用临时实验目录，关闭在线日志，结束后销毁诊断环境）：

```bash
python DQ_high-level/tests/smoke_supermimic_aligned_gpu.py \
  --device cuda:0 --production \
  --output doc/SuperMimic_speed_production_gpu_validation.json
```

如需绕过此前的W&B上传故障，启动时仍可加`--no-wandb`或设置`WANDB_MODE=offline`。该日志选项与本次计算优化独立。
