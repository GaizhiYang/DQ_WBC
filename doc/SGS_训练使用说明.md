# SGS 教师训练使用说明

适用入口：`train_multistate_DQ_teacher.py`；任务：`B1Z1PickMulti`；物体：`sugar_box`；保留 `--no_velocity_obs`。默认 `sgs.mode=disabled`，原训练命令可以继续使用。

## 1. 环境与工作目录

```bash
conda activate dqwbc
cd /home/hehui/DQ_WBC/DQ_high-level
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

本实现已在本机 `dqwbc` 环境运行。当前机器仅激活 conda 不会自动加入其动态库目录；上面的 `export` 用于解决 Isaac Gym 导入时找不到 `libpython3.8.so.1.0` 的问题。

## 2. 构建可重复抽取的固定配置库（v9）

构库入口读取 [data/cfg/sgs_bank.yaml](../DQ_high-level/data/cfg/sgs_bank.yaml)，当前为 `schema_version: 4`。四类为 `full_task`（完整任务起步）、`pregrasp`（临近抓取）、`at_grasp`（已到达抓取位姿）和 `near_goal`（已抓住、抬升 5–10 cm）。修改 YAML 后重新构库；训练只读取生成的 `.pt` 缓冲区。旧 schema 3 仍可构建原三类库。

```bash
python build_sgs_bank.py \
  --device cuda:0 \
  --num-envs 1024 \
  --per-family 512 \
  --seed 43 \
  --max-batches 512 \
  --reset-config data/cfg/sgs_bank.yaml \
  --output data/sgs/sugar_box_v9.pt
```

`--reset-config` 表示上述初态参数文件；`--config` 表示教师环境配置（默认 `data/cfg/DQ_teacher.yaml`），两者用途不同。当前配置保留用户调节的底盘范围和第三类夹爪开度：

| YAML 字段 | 含义 | 当前值 |
| --- | --- | --- |
| `full_task.base.x_range_m` / `y_range_m` | 完整任务底盘到物体的世界 x / y 有符号间隔，单位 m | `[2.0, 2.0]` / `[0.0, 0.0]` |
| `full_task.base.heading_offset_range_deg` | 底盘朝向相对物体方向的偏角，单位度 | `[-15, 15]` |
| `full_task.arm_pose` | 完整任务机械臂关节姿态 | `default` |
| `pregrasp.base.x_range_m` / `y_range_m` | 临近抓取底盘与物体间隔 | `[0.68, 0.95]` / `[-0.2, 0.2]` |
| `pregrasp.base.heading_offset_range_deg` | 临近抓取底盘朝向偏角 | `[-15, 15]` |
| `pregrasp.ee_approach_distance_range_m` | 末端到候选抓取原点的欧氏距离，单位 m | `[0.05, 0.15]` |
| `at_grasp.base.x_range_m` / `y_range_m` | 到达抓取位姿类别的底盘与物体间隔 | `[0.68, 0.85]` / `[-0.1, 0.1]` |
| `at_grasp.base.heading_offset_range_deg` | 到达抓取位姿类别的底盘朝向偏角 | `[-10, 10]` |
| `at_grasp.gripper_opening_range` | 随机夹爪开度，归一化 0 为闭合、1 为全开 | `[0.5, 1.0]` |
| `near_goal.base.x_range_m` / `y_range_m` | 已抓住类别的底盘与物体间隔 | `[0.68, 0.85]` / `[-0.1, 0.1]` |
| `near_goal.base.heading_offset_range_deg` | 已抓住类别的底盘朝向偏角 | `[-10, 10]` |
| `near_goal.object_lift_height_range_m` | 物体中心相对原桌面支撑位置的实际抬升量 | `[0.05, 0.10]` |

四类 `base` 使用相同定义：`x = object_world_x - robot_base_world_x`，`y = object_world_y - robot_base_world_y`；不是底盘局部坐标，也不是水平欧氏距离。偏角为 `wrap(robot_yaw - atan2(y, x))`，0° 表示正朝向物体，正值表示逆时针，范围须位于 `[-180, 180]` 内。

以下是当前可调扰动参数，数值是本仓库的工程设置，不是论文推荐常数：

```yaml
schema_version: 4
object:
  yaw_offset_range_deg: [-15, 15]
validation:
  physics_steps: 5
  max_object_displacement_m: 0.02
  max_contact_force_n: 5000.0
pregrasp:
  # 本片段只展示新增字段；保留原文件里的 base 和距离配置。
  ee_lateral_offset_range_m: [-0.05, 0.05]
  ee_vertical_offset_range_m: [-0.05, 0.05]
  ee_rotation_offset_range_deg:
    roll: [-5, 5]
    pitch: [-5, 5]
    yaw: [-5, 5]
```

`object.yaw_offset_range_deg` 用于物体在桌面上的起始朝向，绕世界 z 轴、相对于任务正常 reset 的 `init_quat` 旋转。临近抓取的横向 y、上下 z 和 roll/pitch/yaw 都相对于**所选候选抓取位姿的局部坐标系**，不是底盘或世界坐标。独立采样欧氏距离 `d` 和偏移 `y/z` 后，只保留 `y² + z² < d²` 的组合，再取 `x = -sqrt(d² - y² - z²)`；不可行的组合记入 `sample_geometry` 拒绝统计。增加横向扰动不会自动扩大距离，也不会悄悄缩小用户指定的 y/z 范围。配置检查只拒绝完全没有可行交集的范围；部分不可行组合会降低采样效率。每轴旋转范围限制在 `[-45, 45]` 度。

四类构造方法：

1. **完整任务起步**：正常重置，按 YAML 放置底盘、校正足底到地面附近，机械臂保持原任务 reset 使用的 `_initial_dof_pos` 默认关节姿态。
2. **临近抓取**：物体仍在桌上，夹爪张开；在候选抓取位姿的 local -x 一侧采样距离、横向/上下偏移及小角度姿态扰动，使用 IK 设置机械臂。检查实际距离、姿态、可达性、碰撞和足底几何。不执行完整抓取脚本，不以脚本抓取成功作为入库条件。
3. **已到达抓取位姿**：物体仍在桌上，使用 IK 将末端对齐候选抓取位姿，按 `gripper_opening_range` 随机设置关节位置和控制目标。不预先抬升，不要求已可靠夹持。归一化值映射到 URDF 关节上下限；`[0, 0]` 固定闭合，`[1, 1]` 固定全开。
4. **Near-Goal**：从可行抓取位姿开始，使用原低层控制器，实际闭爪、控制末端抬升并短时保持；物体依靠物理接触随夹爪运动。检查真实抬升量、夹爪接触、物体相对末端滑移/转动和末尾速度，并验证保存状态能够重新恢复后保持抓持。保存实际关节开度、持续闭爪目标及当时实际使用的末端控制目标。额外抓持过程只用于这一类。

`pregrasp.ee_approach_distance_range_m` 与末端到物体中心的距离不同。旧的 `grasped_before_lift` 类不自动映射为新第三类或第四类；schema 1/2 需迁移后再构库。新第四类的参数路径是 `near_goal.object_lift_height_range_m`，与旧第三类的同名参数区分。

四类的足底均只要求碰撞球底面大致位于地面附近，当前允许 **−5 mm 至 +20 mm**；用 **4 cm 碰撞球半径**计算。**不要求稳定站立、连续承重脚数或底盘静止速度。** 前三类运行可配置的短物理窗口，默认 5 个物理步、每步 5 ms，共 25 ms；检查窗口内物体相对候选初始位置的最大位移和峰值接触力，默认上限分别为 2 cm 和 5000 N。第四类采用专用的抬升/抓持验收，不能沿用“仍在桌上”与总位移不超过 2 cm 的条件。允许足地及合理夹爪物体接触，保留关节限位、数值合法性和保存帧参数范围检查。

目标范围先均匀采样，经过 IK 和物理筛选后，最终库内分布不保证均匀。`[a, a]` 是固定值，只有这类零宽范围允许位置和末端距离 **±5 mm**、偏角 **±1°** 的物理容差。不可达设置仍可能导致样本不足；构库不会自动扩展用户范围或复制样本补齐。

5000 N 是所有刚体接触力范数的峰值上限；另外保留非抓取部位 5 N、抓取部位 250 N 的更严格接触阈值。连续窗口只在开始时设置机器人和物体状态；浮动桌每步维持支撑位姿，避免重力导致桌面持续下落。训练恢复后按保存的参数继续执行原有 Level02 桌面运动。

成功后输出 `.pt` 缓冲区和同名 `.json` 审计文件。除配置数、指纹和有效 YAML 外，`metadata.sampling_audit` 区分提出的候选、通过筛选的候选和最终入库数量，记录拒绝原因及分布，帮助定位哪些范围过窄或难以到达。独立拒绝计数可能重叠，`first_rejection_counts` 只记每条候选首次失败的检查。通过筛选但超过某类剩余名额的候选不算物理失败。保存帧还记录所选候选 ID、候选局部位移、末端局部姿态偏差和实际物体 yaw，便于复查采样目标与实际状态的差别。采样耗尽时不保存不完整库，拒绝统计写入同名 `.failed.json`。

目标是四类各 512 条、共 2048 条。新库使用 `validation_profile: geometry_grasp_v2`、`near_goal_grasp_verified: true`，保留 `feet_near_ground: true`、`collision_checked: true`、`object_attachment: false`。v8 的三类 `geometry_v1` 库仍可加载；v1–v4 的第三类语义不同，不能直接使用。历史文件与报告保留，新训练命令使用 v9。

构建一次后即可反复训练。修改 YAML 后需要重新生成文件，SGS 和 uniform 对照应读取同一个文件和指纹。训练期间从缓冲区抽取已有初态，**不会重跑离线 IK 或物理筛选流程**。

## 3. 正式训练

先按第 2 节成功构建四类各 512 条的 v9 库，再在原命令上增加 `--sgs_mode sgs` 和 `--sgs_bank`。训练不读取 `sgs_bank.yaml`，也不接受构库入口的 `--reset-config`：

```bash
python train_multistate_DQ_teacher.py \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --timesteps 120000 \
  --task B1Z1PickMulti \
  --object_name sugar_box \
  --experiment_dir DQ_teacher/b1-pick-sugar-box-sgs \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero \
  --rand_control \
  --no_velocity_obs \
  --sgs_mode sgs \
  --sgs_bank data/sgs/sugar_box_v9.pt \
  --seed 43 \
  --headless --wandb --wandb_name no_velocity_obs
```

固定库均匀采样对照只需将 `--sgs_mode sgs` 改为 `--sgs_mode uniform`，同时为 `--experiment_dir` 使用不同目录，保留相同 bank、seed、环境数和训练步数。原始任务基线使用 `--sgs_mode disabled`，无需 bank。

默认 `bank_size: null`，从库读取实际配置数；也可显式填写整数，加载时核对一致。SGS 参数为每配置最近 100 个已完成回合、目标成功率 0.5、`kappa=1`、`temperature=2`、`epsilon=1e-4`、`p_init=0`。配置位于 `data/cfg/DQ_teacher.yaml` 的 `env.sgs`。uniform/SGS 训练使用 hold 成功标签。四类等量时，初始每类总采样概率约 25%；后续仍按每条配置的历史成功率调整，不强制各类保持等概率。

当前实现从头训练，不支持 SGS/uniform 的 `--resume` 或 `--checkpoint`。不能将 agent checkpoint 或诊断文件当作完整续训状态。

## 4. 小规模 GPU 冒烟

以下命令同样以第 2 节的 v9 库已构建成功为前提；本轮验证进展见第 14 节。

```bash
python train_multistate_DQ_teacher.py \
  --rl_device cuda:0 --sim_device cuda:0 \
  --timesteps 120000 --stop_after 96 --num_envs 32 \
  --task B1Z1PickMulti --object_name sugar_box \
  --experiment_dir DQ_teacher/sgs-smoke --wandb_name seed43 \
  --roboinfo --observe_gait_commands --small_value_set_zero \
  --rand_control --no_velocity_obs \
  --sgs_mode sgs --sgs_bank data/sgs/sugar_box_v9.pt \
  --seed 43 --headless
```

`--stop_after 96` 提前结束本次运行，`--timesteps 120000` 仍决定原奖励调度。不要为冒烟把总计划改成 96。96 个控制步对应 4 次 PPO 更新；该运行只检查执行链路，不说明训练收敛或成功率改善。提前结束时会额外保存 `checkpoints/agent_96.pt`。

## 5. 严格评估

```bash
python eval_sgs_teacher.py \
  --checkpoint DQ_teacher/b1-pick-sugar-box-sgs/no_velocity_obs/checkpoints/agent_120000.pt \
  --rl_device cuda:0 --sim_device cuda:0 \
  --num_envs 128 --eval_episodes 2048 \
  --seed 1001 --headless \
  --eval_output DQ_teacher/b1-pick-sugar-box-sgs/no_velocity_obs/eval_seed1001.json
```

评估强制使用原始 2 m 起点、Level02、100 步时限和连续保持 15 步的成功条件，关闭训练 sampler；策略取确定性动作，scaler 冻结且核对评估前后状态一致。新运行会从保存的 YAML 和 `teacher_run.json` 恢复观测与控制参数；旧 checkpoint 缺少这些文件时，需要显式提供 `--cfg_file` 及训练时的模型参数。

比较不同方法或 checkpoint 时，固定 `--seed`、`--num_envs`、`--eval_episodes`。评估按同步批次运行，每个环境每批只计第一次结束，全部结束后统一 reset；第 `wave` 批使用 `seed + wave`，避免策略结束时间改变下一批的随机配置。最后一批只计剩余所需回合。

以输出 JSON 中的 `hold_success_rate` 比较性能。需要选择最佳 checkpoint 时，固定验证集，并另外保留不同 seed 的测试集。

## 6. 输出与核对

训练输出位于 `experiment_dir/wandb_name/`：

| 文件 | 用途 |
| --- | --- |
| YAML 配置快照、`teacher_run.json` | 保存配置、入口参数与总训练计划 |
| `bank_metadata.json` | 保存 bank 路径、配置数和指纹；核对 uniform/SGS 是否使用同一库 |
| `sgs_diagnostics.pt` | 正常训练结束时导出每配置成功率、历史样本数、累计访问数、采样概率、类别统计及 bank 指纹 |
| `checkpoints/agent_*.pt` | 用于加载策略评估；不包含完整 SGS/仿真续训状态 |
| TensorBoard events | 查看训练曲线及 `SGS / ...` 统计 |

`sgs_diagnostics.pt` 明确标记为诊断文件，不能用于恢复训练。未完成回合不会被当作失败加入成功历史。

## 7. v1 历史验证结果

本节仅保留历史执行与统计检查。v1 构库缺少保存前接地门槛，完整任务和临近抓取快照存在悬空问题；不能用以下 GPU 通过结果证明其初态站稳，也不再用于新训练。

验证日期：2026-10-20；环境：`dqwbc`，PyTorch `2.4.1+cu121`，Isaac Gym GPU PhysX。

| 检查 | 实际结果 |
| --- | --- |
| 配置库 | 7 批 × 512 候选完成筛选，最终三类各 512，共 1536 条 |
| CPU / CUDA 回归 | 43 项全部通过，覆盖 SGS 公式、重复 ID、滚动历史、回合归因、异步重置和严格成功标签 |
| SGS GPU 训练 | `cuda:0`，5000 环境，最终 1536 条库，96 步、4 次 PPO 更新；6709 个完成回合正确登记 |
| SGS 状态核对 | 75 项检查通过，包括首帧缓存、局部取消不影响其他环境、延迟状态提交、有限值与模型参数实际更新 |
| Uniform GPU 训练 | `cuda:1`，32 环境，同一最终库，96 步、4 次 PPO 更新；概率保持均匀，模型参数正常 |
| 严格评估 | 加载 Uniform 冒烟 checkpoint，完成 8 回合；原始 2 m / Level02 / hold15，确定性策略，scaler 前后完全一致 |

v1 库指纹：

```text
b02cdb5def5c2fa2bae5bdc294533c9ed6c6843e800d8b6435bf98f6d141e835
```

5000 环境运行的 PyTorch 分配显存峰值为 5.38 GiB，缓存保留峰值为 5.75 GiB；这些数值不包含 Isaac Gym / PhysX 自身的显存分配。测试使用本机 `cuda:0`，可用总显存 49140 MiB。

上述短程测试验证执行和统计正确性。96 步模型在 8 回合严格评估中的保持成功率为 0/8，不代表完整训练效果；尚需通过相同预算的 Uniform / SGS 完整训练比较收益。详细结果见 [SGS_实现验证记录.json](SGS_实现验证记录.json)，构库审计见 [sugar_box_v1.json](../DQ_high-level/data/sgs/sugar_box_v1.json)。

## 8. 随机抽样可视化配置

在第 1 节的环境和目录中运行；默认读取 v9，每类随机抽取 3 条，共 12 条。读取旧 v8 库时自动展示三类、共 9 条：

```bash
python visualize_sgs_bank.py --device cuda:0 \
  --bank data/sgs/sugar_box_v9.pt --samples-per-family 3 --seed 43
```

窗口打开后暂停，终端显示抽到的配置 ID。更换 `--seed` 可换一组，`--samples-per-family` 调节数量；该抽样用于检查，与训练 SGS 采样独立。

| 按键 | 功能 |
| --- | --- |
| `1` / `2` / `3` / `4` | 完整任务起步 / 临近抓取 / 已到达抓取位姿 / Near-Goal（旧三类库无 `4`） |
| `←` / `→` | 切换当前类别的抽样配置 |
| `F` / `C` | 全景 / 末端与物体近景 |
| 空格 / `N` | 播放或暂停 / 前进一个高层控制步 |
| `R` | 重新恢复保存的初态和随机夹爪目标 |
| `O` | 人为切换开合目标，下一步生效 |
| `S` / `Esc` | 保存截图 / 退出 |

**默认播放保持库里保存的夹爪目标，包括第三类的随机开度和第四类的闭爪目标。** 预览对当前环境实例单独保留该目标，因为训练原动作中 0 表示开爪，不能用 0 代表保持；训练动作语义未改。按 `O` 后才切换为全开/全闭，按 `R` 恢复库中的值。第三类没有物体已抓稳或会保持悬空的保证。

无窗口导出图片、GIF 和测量 JSON：

```bash
python visualize_sgs_bank.py \
  --device cuda:0 --headless \
  --bank data/sgs/sugar_box_v9.pt \
  --samples-per-family 3 --seed 43 --steps 8 \
  --output ../doc/sgs_preview/v9_random_seed43
```

输出 `overview.png`、全景 `grounding_overview.png`、每条配置带 ID 的图片/动画及 `measurements.json`。复查指定条目用 `--config-id 1024`。不同 bank 中相同 ID 不代表同一个状态。

对比图显示最大足底间隙和 `feet_near_ground`，后者只检查足底碰撞几何是否在 **[−0.005, 0.020] m**。`foot_contact_force_z_N` 仅供诊断，不作为稳定站立合格条件。`gripper_opening_normalized` 是实测关节开度，`gripper_target_opening_normalized` 是当前控制目标，均为 0 闭合、1 全开；碰撞可能令实际开度偏离目标。

`measurements.json` 和交互终端增加以下实测字段：`ee_to_grasp_candidate_m` 是到保存时所选候选的实际距离；`candidate_local_offset_m` 是候选局部 `[x, y, z]` 偏移；`ee_rotation_offset_deg` 是相对候选的局部 `[roll, pitch, yaw]`；`object_yaw_offset_deg` 是相对正常 reset 姿态的物体 yaw。候选按物体当前姿态重新计算，不以另一个更近的候选替换。完整任务没有所选候选，对应字段为 `null`；物体 yaw 仍正常记录。`saved_geometry` 保存构库时实测值，用于区别恢复后物理推进产生的变化。对比图只额外标注候选距离，详细参数查看 JSON。

Near-Goal 还记录 `saved_grasp_evidence`（构库的抬升、滑移/旋转、接触帧比例、末尾速度和恢复验证标记），以及预览中的 `relative_slip_from_saved_m`、`relative_rotation_from_saved_deg`、`object_in_ee_position_m`、`finger_contact_force_n`、`both_fingers_contact`、`object_speed_m_s`、`near_goal_lift_in_range`。接触力按固定指侧、活动指侧顺序记录。当前仿真会折叠固定关节，固定指 `gripperStator` 的碰撞合并到 `link06`，因此使用 `link06` 与 `gripperMover` 的净接触力诊断；未折叠时使用独立的 `gripperStator`。这些值不是独立的指物接触对，更不能凭一帧接触证明稳定抓持，需要结合夹持几何与持续保持结果。前三类对应近目标诊断为 `null`。

恢复首帧只推进渲染所需的一个 **5 ms** 物理步，没有长时间落地等待；这与构库时默认 5 步的筛选窗口用途不同。播放采用低层行走控制和机械臂 IK，高层位姿增量与底盘指令为零，保留 Level02 桌面运动。因此短动画展示初态的物理响应，不代表训练好的教师表现。`ee_to_object_center_m` 不能直接用来核对 YAML 的候选抓取距离。

旧版本预览图留在各自目录。新版预览接受三类 `geometry_v1` 或四类 `geometry_grasp_v2` 库，避免把旧的已抓持样本误标为新第三类。

## 9. v2 历史距离调整与验证（2026-10-22）

本节及其图片仅用于历史复查；当前训练入口拒绝 v2 库。

已生成 `data/sgs/sugar_box_v2.pt`：13 批 × 512 个候选环境完成筛选，三类各保存 512 条。临近抓取入库状态的前后间隔实测为 **0.680046–0.894592 m**，中位数 **0.733268 m**。所有 512 条均处于所要求的 0.68–0.90 m 范围内；距离分布受可达性和抓取验证影响，较近样本更多。旧 v1 文件的指纹未变化。

v2 指纹：`3d334ddad4fda3e6ade1a271d0a9f48f6446363aaddd5659146ed075031a13d6`。

使用新库在 `cuda:0` 完成 32 环境、96 步、4 次 PPO 更新，75 项检查通过，22 个完成回合正常登记。该结果验证新库可用于 GPU 训练，不代表长期训练效果；见 [GPU 验证记录](sgs_v2_gpu_validation.json)和[构库审计](../DQ_high-level/data/sgs/sugar_box_v2.json)。首轮报告写入遇到终端 ASCII 编码错误，使用 `PYTHONUTF8=1` 重跑后完整通过。

每类随机 3 条的 [v2 对比图](sgs_preview/v2_random_seed43/overview.png)和短动画作为历史资料保留。旧版图片和原始数据可用于复查；当前 v5 预览入口不再接受旧类别语义的库。

**后续脚部接地审计发现的限制：** 当前 v2 的完整任务和临近抓取两类（各 512 条）在保存时四脚碰撞底面均离地约 8.2–8.4 cm。原任务重置和临近抓取构造把底盘 z 设置为 0.60 m，随后过早保存快照；原筛选检查数值、终止、物体支撑及抓取可执行性，没有要求保存时机器人已经接地。抽样 ID 39、985 在预览首帧脚部接触力均为零，推进约 0.15 s 后落地。已抓持样本 ID 1488 首帧已有脚部接触。脚的视觉球半径 3 cm、碰撞球半径 4 cm，还会带来约 1 cm 的视觉间隙，但不能解释前两类约 8 cm 的真实离地。

见[接地审计数据](sgs_preview/v2_ground_audit/report.json)、[完整任务首帧](sgs_preview/v2_ground_audit/id39_initial.png)及[推进两步后](sgs_preview/v2_ground_audit/id39_after2steps.png)。上述 GPU 冒烟通过表示训练链路可运行，不表示初态已经稳定接地。旧文件仍保留这些初态；v3 构库已加入落地稳定、保存前及恢复首步接地检查，并在稳定后重新核对底盘距离和末端抓取关系。

## 10. v3 接地修复的历史验证记录

2026-10-22 已完成实际验证。当前库为 `data/sgs/sugar_box_v3.pt`，三类各 512 条，共 **1536 条**，指纹为 `683871f797e2434df606dbf53084f8ec08c884ad0131659a64b6741e01ab325a`。完整任务状态采用更严格站立门槛重新采集；临近抓取和已抓持沿用本轮 v3 构建中已验证的 1024 条状态，来源指纹及原验证记录保存在 `reused_validated_families` 元数据中。最终库的 104 批记录只对应重新采集完整任务，不能理解为三类均重新生成 104 批。

全库独立复查使用 `cuda:0`、128 个环境、随机种子 1043。每条配置清空历史后仅推进一个 **5 ms** 物理步，不插入落地等待：

| 配置类别 | 保存帧接地检查 | 恢复首步支撑检查 | 首步最大足底间隙 |
| --- | --- | --- | --- |
| 完整任务起步 | 512 / 512 | 512 / 512 | 5.51 mm |
| 临近抓取 | 512 / 512 | 512 / 512 | 6.59 mm |
| 已抓住、待抬升 | 512 / 512 | 512 / 512 | 8.42 mm |

恢复首步所有配置至少有两脚竖直接触力超过 2 N；保存帧完整任务四脚承重，另外两类至少三脚承重。这些结果来自本次独立复查，不能解释为所有随机化动力学条件下始终四脚承重。完整任务保存时与物体的水平距离为 1.9950–2.0049 m，临近抓取的 x 方向间隔为 0.6809–0.8967 m，均符合设置。详见[全库接地审计](sgs_v3_grounding_audit.json)。复查命令（在 `DQ_high-level` 下运行）：

```bash
PYTHONUTF8=1 python tests/audit_sgs_grounding_gpu.py \
  --bank data/sgs/sugar_box_v3.pt --device cuda:0 \
  --num-envs 128 --seed 1043 \
  --output ../doc/sgs_v3_grounding_audit.json
```

同一指纹的库已完成 **32 个 GPU 环境、96 步、4 次 PPO 更新**，75 项训练链路检查全部通过，记录到 22 个已完成回合；另有 **76 项单元回归测试通过**。该短测试验证 GPU 训练可运行，不代表长期学习效果，见[GPU 验证记录](sgs_v3_gpu_validation.json)。随机抽取每类 3 条的预览也已重新导出，9 条均通过首步支撑判据，见[脚部接地全景图](sgs_preview/v3_random_seed43/grounding_overview.png)。

## 11. v4 YAML 接入的历史验证记录

2026-10-22 已验证默认 YAML 和一份自定义 YAML。两者均使用 `cuda:0`、1024 个构建环境，生成每类 4 条、合计 12 条的小样本库，分别经过 2 批和 9 批物理筛选。小样本输出在 `/tmp/sgs_yaml_default.pt`、`/tmp/sgs_yaml_custom.pt`，用于本次测试，不作为 1536 条正式训练库。

自定义测试同时改变三类底盘范围，并将完整任务 x/y 改为 `[1.6, 1.8]` / `[0.10, 0.15]` m、偏角改为 `[5, 8]` 度，临近抓取末端距离改为 4–7 cm，已抓持抬升改为 9–12 cm。保存状态的全部底盘范围检查通过；完整任务仍保持默认机械臂关节姿态，临近抓取存在距离符合范围的候选，已抓持物体的实际抬升为 10.64–11.47 cm。两份完整有效配置及逐条测量见[几何参数验证](sgs_yaml_geometry_validation.json)。

两份库各 12 条都通过独立随机种子 1043 下的保存帧及首个 5 ms 接地复查，见[默认参数接地记录](sgs_yaml_default_grounding.json)和[自定义参数接地记录](sgs_yaml_custom_grounding.json)。自定义库又完成 32 个 GPU 环境、96 步、4 次 PPO 更新，75 项训练链路检查通过，见[GPU 短训练记录](sgs_yaml_gpu_validation.json)。该测试临时将 `env.sgs.bank_size` 设为 12，以匹配测试库；正式命令仍使用三类各 512 条。另有 113 项 CPU/CUDA 回归测试通过。

本轮未覆盖既有 v3 正式库，也未替用户生成一份参数尚待调整的 1536 条 v4 正式库。按第 2 节修改 YAML 并重新构建即可。新参数可能改变可达性和采集耗时；本次小样本结果不代表任意参数都能通过，也不代表长期训练效果。


## 12. v5 历史简化版验证记录

本节只记录简化版实际运行结果，不能沿用上文 v3/v4 的站立、抓持验证结果作为 v5 证明。当时的变更包括三类新语义、schema 2 YAML、足底几何检查、随机夹爪开度及训练缓冲区恢复接口。

2026-10-22 已使用当时的 YAML 生成正式库 [sugar_box_v5.pt](../DQ_high-level/data/sgs/sugar_box_v5.pt)。使用 `cuda:0`、1024 个构建环境，7 批收齐三类各 512 条，共 **1536 条**。采样、IK 与基础物理筛查累计 **6.791 秒**；该时间不含 Isaac Gym 启动、资产加载及最后文件保存。逐批记录见[构库审计](../DQ_high-level/data/sgs/sugar_box_v5.json)。

库指纹：`8cb9d378dd36ee28435e38167577ecc5841fc00617750ba6bc41294875d52cf8`。

| 检查 | 实际结果 |
| --- | --- |
| CPU / CUDA 回归测试 | 137 项通过，包括随机开度映射、范围检查、新库加载及重置逻辑 |
| 保存帧底盘范围、足底近地 | 1536 / 1536 通过；足底距地面约 0.741–3.478 mm |
| 临近抓取候选距离 | 512 / 512 在当前 5–15 cm 范围内，实测 5.001–14.979 cm |
| 已到达抓取位姿 | 512 / 512 合格；相对所选候选的位置误差最大 4.990 mm，方向误差最大 0.0890 rad |
| 第三类夹爪开度 | 实际开度 0.000279–0.998507，实际位置与目标开度均符合 `[0, 1]` |
| GPU 教师短训练 | 32 环境、96 步、4 次 PPO 更新；75 项状态/训练链路检查通过，23 个完成回合登记 |
| 从缓冲区实际抽取 | 三类抽取次数分别为 28、13、15；核对 root、DOF、末端目标和夹爪目标恢复一致 |
| 随机预览 | 每类 3 条，共 9 条，预览首帧均足底近地；第三类保留各自随机开度 |

保存帧几何复查见 [sgs_v5_geometry_validation.json](sgs_v5_geometry_validation.json)，GPU 训练记录见 [sgs_v5_gpu_validation.json](sgs_v5_gpu_validation.json)。几何复查没有重复碰撞仿真；构库的碰撞检查是单个 PhysX 步的基础接触力筛查，不是完整网格无穿透证明。目标开度均匀采样，但经过 IK/碰撞筛选，入库开度并不均匀，报告中保留了直方图。

**独立恢复首步检查有明确边界。** 换用 128 个环境、seed 1043，逐条恢复后推进一个 5 ms 物理步，三类分别有 512/512、500/512、510/512 满足同一足底范围。14 条出现约 5–8 mm 的足底穿入，最深为 7.96 mm；所有配置首步最大离地间隙为 5.66 mm，没有明显悬空。部分异常条目同时出现较大的瞬时接触力和底盘速度，不能仅凭这次短检查判定其长期影响。该动态检查的 `passed` 保持为 `false`，详见 [sgs_v5_grounding_audit.json](sgs_v5_grounding_audit.json)，不能写成全库恢复后全部通过。保存帧均通过，也不保证不同动力学环境下首步响应完全一致。按本次简化要求，未将跨环境复放、承重或稳定站立重新设为构库准入条件。

随机样本的[脚部全景图](sgs_preview/v5_random_seed43/grounding_overview.png)、[末端近景对比](sgs_preview/v5_random_seed43/overview.png)与[逐帧测量](sgs_preview/v5_random_seed43/measurements.json)已导出。短训练与预览验证的是执行链路和初态表现；SGS 是否提高最终任务成功率仍需完整训练对照。

## 13. v8 覆盖与物理筛选改进（历史记录）

v8 保留原有三类初态及 SGS 成功率驱动采样，修改范围集中于离线配置库：增加物体朝向和临近抓取位姿扰动，运行默认 25 ms 的物理筛选窗口，并记录采样覆盖、拒绝原因和保存状态的实测几何。训练继续从同一个 GPU 缓冲区恢复初态；没有新增 Near-Goal 类，也没有恢复逐条抓取成功或稳定站立要求。

本节记录当时的 v8 实测；当前构库、训练和预览命令见第 2–4、8 节，新增 Near-Goal 见第 14 节。修改 YAML 后需要重建配置库；仅换文件名不会给旧库增加检查或新类别。

2026-10-25 使用当时的 YAML、`cuda:0`、1024 个构建环境，生成正式库 [sugar_box_v8.pt](../DQ_high-level/data/sgs/sugar_box_v8.pt)。9 批完成三类各 512 条，共 **1536 条**；采样、IK 和物理筛选累计 **3.034 秒**，包含仿真启动与文件保存的本次总用时约 **25.6 秒**。这些是本机本次实测，不是任意参数范围下的耗时保证。

正式库指纹：`781f279627a7d6397097a81c0bcfc5dfa7cd3ba15f682fad5f2b54138d97044e`。

| 类别 | 提出的候选 | 通过全部检查 | 实际入库 | 通过比例 |
| --- | ---: | ---: | ---: | ---: |
| 完整任务起步 | 1024 | 1024 | 512 | 100% |
| 临近抓取 | 9216 | 519 | 512 | 5.63% |
| 已到达抓取位姿 | 4096 | 677 | 512 | 16.53% |

已收齐的类别不再统计新的候选；完整任务模板仍会用于后续两类的构造。临近抓取最主要的首个拒绝原因是 IK 位置残差，共 8625 条。所有候选的原因及入库直方图见[正式库构建报告](../DQ_high-level/data/sgs/sugar_box_v8.json)。

| 验证项目 | 本轮结果 |
| --- | --- |
| CPU / CUDA 回归 | 169 项通过，包含瞬态碰撞/位移峰值保留、桌面支撑控制、机器人和物体连续演化、欧氏距离约束 |
| 正式库保存帧临近抓取 | 欧氏距离约 5.000–14.459 cm；局部 y/z 均在 ±1 cm 内，局部 roll/pitch/yaw 均在 ±5° 内 |
| 正式库保存帧已到达抓取位姿 | 候选位置误差最大约 4.971 mm，夹爪实际开度约 0.50016–0.99744 |
| 物体 yaw / 短窗位移 | 三类 yaw 均在 ±15° 内；已入库状态的窗口内物体位移最大约 6.96 mm |
| 正式库 GPU 教师训练 | 同一 1536 条正式库，32 个环境、96 步、4 次 PPO 更新；75 项检查通过，35 个完成回合登记 |
| 正式库三类实际抽取 | 分别为 35、16、17 次；root、DOF、末端/夹爪目标及缓存恢复核对通过 |
| 正式库随机预览 | 每类 3 条、共 9 条；恢复后首个 5 ms 物理步均满足足底近地要求 |

先使用 48 条小库完成执行链路检查，再生成正式库。最终[GPU 验证记录](sgs_v8_gpu_validation.json)对应上面的 **1536 条正式库及同一指纹**，使用原教师配置的 `bank_size=1536` 和用户训练参数（关闭在线 wandb、缩小环境数并用 `--stop_after 96` 提前停止）。该测试确认实际恢复与 PPO 更新可运行，不代表长期学习效果。

全量[保存帧几何复查](sgs_v8_geometry_validation.json)有 **161 项检查通过**，包括实际 root/足底几何、审计计数及分布与张量一致性。候选相对位姿使用构库保存的测量字段，未独立重算候选或重复碰撞仿真。物体相对四元数计算角度前先归一化，独立 SciPy yaw 计算与保存字段最大差约 `0.0000146°`。

正式库的[三类随机预览](sgs_preview/v8_random_seed43/overview.png)、[足底全景](sgs_preview/v8_random_seed43/grounding_overview.png)和[逐帧测量](sgs_preview/v8_random_seed43/measurements.json)均已保存。首帧 9/9 满足足底范围；继续运行 8 个高层零增量控制步后，ID 147、1043 各有一只脚抬至约 3.08 cm、2.28 cm，超出“所有脚近地”的几何范围。该播放使用低层策略和移动桌面，不是稳定站立认证，也未将后续持续站稳重新设为准入条件。本轮图像在 `cuda:0` 导出；本机 `cuda:1` 相机预览遇到原生渲染段错误，未将其标记为通过。

覆盖统计仍显示偏斜：临近抓取入库覆盖 21 个候选，但候选 12、23 合计占 **64.06%**；第三类覆盖 17 个候选。当前实现提供了新的连续扰动和可检查的分布，并不保证候选均衡，也尚未证明最终成功率提高。连续短窗筛查仍是 PhysX 接触力与几何检查，不是全网格无穿透证明或全库跨环境动态复放。

## 14. v9 新增 Near-Goal

v9 在原三类之外增加 `near_goal`。前三类继续采用原有轻量检查，第四类单独执行“闭爪—抬升—保持—恢复检查”。构库后保存真实机器人、物体、桌面和关节状态、末端目标、闭爪目标及重置缓存；物体没有附着约束。训练期间直接在 GPU 上抽取并恢复这些配置，策略可以自行开闭夹爪。

YAML 的第四类配置如下；这些时间和阈值是初版工程设置，可根据拒绝统计和实际播放调整：

```yaml
near_goal:
  base:
    x_range_m: [0.68, 0.85]
    y_range_m: [-0.1, 0.1]
    heading_offset_range_deg: [-10, 10]
  object_lift_height_range_m: [0.05, 0.10]
  close_time_s: 1.50
  lift_time_s: 0.60
  hold_time_s: 0.20
  max_relative_slip_m: 0.01
  max_relative_rotation_deg: 10.0
  min_finger_contact_force_n: 0.2
  min_contact_fraction: 0.8
  max_object_speed_m_s: 0.10
```

抬升定义为 `object_z - table_height - init_height`，不能直接把物体中心距桌面高度当作抬升量。保存帧必须落在指定范围，范围上界须低于任务成功阈值；当前成功仍要求超过 20 cm 并连续保持 15 个控制步，恢复后计数从零开始。短时抓持检查的 `hold_time_s` 与任务成功保持步数是不同参数。

本仓库夹爪驱动刚度较低，最初拟定的 0.2 秒不足以完成真实闭合，因此当前闭爪时间为 1.5 秒，抬升时间为 0.6 秒。构造过程使用原低层控制器；末端命令增加有界积分补偿，每个世界坐标轴最多 ±0.15 m，用来补偿增量 IK 在重力下的跟踪误差。该补偿器只在离线构造过程中运行，正式训练不新增此控制器。

保存时保留实际使用的末端控制目标，不把目标重置成当时实际末端位置，以便恢复后持续提供支撑物体所需的控制作用。恢复检查从保存状态清空控制历史后开始，在真实低层控制器及原桌面运动下保持保存的末端/闭爪目标，进行短时抓持复查；它不是教师策略自主动作的完整闭环评估。

保持期间记录物体相对末端的滑移和转动，以及两指接触满足阈值的帧比例；末尾还限制物体速度。25 ms 的原几何窗口不足以筛掉所有“暂时还没掉下来”的假抓持，因此本类使用独立保持窗口。接触力仍是仿真刚体的接触诊断，要与夹持几何、脱离桌面支撑和实际保持结果共同检查，不能把单个非零接触力当作抓稳证据。足底仍只检查大致接地，没有要求机器人稳定站立。

构库 JSON 的 `metadata.sampling_audit.families.near_goal` 汇总新类别的提议、通过、入库及拒绝原因；`coverage` 增加 `lift_height`、`near_goal_relative_slip_m`、`near_goal_relative_rotation_deg`、`near_goal_contact_fraction`、`near_goal_object_speed_m_s`、`near_goal_restore_verified`。近目标证据仅汇总在第四类，其他类别的占位零不代表抓持验证通过。旧 v8 库缺少这些字段时不补造证据。

### 本轮实际验证

2026-10-25 已用当前 YAML、`cuda:1`、1024 个构建环境和 seed 43 生成正式库 [sugar_box_v9.pt](../DQ_high-level/data/sgs/sugar_box_v9.pt)。四类各 **512 条，共 2048 条**；36 批完成，采样、IK 和物理筛查累计 **133.07 秒**，包含仿真启动和文件保存约 **155.9 秒**。Near-Goal 需要真实闭爪、抬升及两次保持，因此比前三类的几何构造更耗时。

正式库指纹：`c6c7f69d8e8a3fd67012eeab4026f86bf8d7be58c7f4edbe6b1f63b95f5498ec`。

| 类别 | 提出的候选 | 通过全部检查 | 实际入库 | 入库候选抓取位姿种类 |
| --- | ---: | ---: | ---: | ---: |
| 完整任务起步 | 1024 | 1024 | 512 | 不适用 |
| 临近抓取 | 8192 | 570 | 512 | 22 |
| 已到达抓取位姿 | 4096 | 684 | 512 | 17 |
| Near-Goal | 36864 | 522 | 512 | 15 |

详细拒绝原因和分布见[构建报告](../DQ_high-level/data/sgs/sugar_box_v9.json)。新增类通过率约 **1.42%**；最终库不保证候选位姿均衡。

| 验证项目 | 结果 |
| --- | --- |
| CPU / CUDA 回归测试 | **203 项通过** |
| 全库保存状态复查 | **279 项检查通过**，四类底盘范围、足底几何、审计统计与保存张量一致 |
| Near-Goal 实际初始抬升 | **5.026–9.307 cm**，512 条均低于任务成功高度 |
| 构库保存的抓持证据 | 最大相对滑移 **5.528 mm**、最大转动 **3.416°**、两指接触帧比例均为 1.0，原构建环境中的冷恢复检查通过 |
| 正式库 GPU 教师短训练 | `cuda:1`、32 环境、96 步、**4 次 PPO 更新**；76 项检查通过、34 个已完成回合 |
| 从正式缓冲区实际抽取 | 四类分别 **22、22、9、14 次**；恢复状态一致、模型参数有限且实际更新 |
| 随机图像预览 | `cuda:0`，每类 3 条、共 12 条；首个 5 ms 物理步均足底近地，3 条 Near-Goal 均处于指定抬升范围且具有双指接触 |

全库复查见 [sgs_v9_geometry_validation.json](sgs_v9_geometry_validation.json)，GPU 训练见 [sgs_v9_gpu_validation.json](sgs_v9_gpu_validation.json)。保存状态复查没有重复碰撞仿真；其中的抓持证据来自构建器。短训练确认 GPU 执行、采样、恢复和优化链路可运行，没有证明长期训练成功率提高。

### 跨环境抓持复查的边界

另外创建 32 个新 GPU 环境，以 seed 1043 随机抽取 32 条不同 Near-Goal，清空历史并保持保存的末端和闭爪目标约 **0.205 秒**。**32/32** 均通过夹持区域、物体离桌、相对滑移/转动、接触、足底近地及碰撞检查；最大相对滑移 **0.868 mm**、最大转动 **0.609°**，两指接触帧比例最低 **95.12%**。再次恢复同批初态并开爪约 0.805 秒后，**32/32** 失去双指接触，且物体高度均下降超过 2 cm，符合真实夹持解除后的响应。

但这次跨环境复查**仅 13/32 通过全部构库阈值**：19 条末尾世界坐标速度超过 `0.10 m/s`，最大约 `0.207 m/s`；其中 1 条在窗口内曾越出 5–10 cm 高度范围，末帧 32 条又均位于范围内。该报告的 `passed` 保持为 **false**，没有放宽阈值或把失败隐去。原构建环境中的冷恢复验证不等于任意摩擦、夹爪刚度和控制响应下都会通过同样阈值；此处观察到抓持仍在，但整体运动不一定满足构库时的速度限制。

报告见 [sgs_v9_grasp_validation.json](sgs_v9_grasp_validation.json)。复查命令：

```bash
python tests/audit_sgs_near_goal_gpu.py \
  --bank data/sgs/sugar_box_v9.pt --device cuda:1 \
  --num-envs 32 --seed 1043 \
  --output ../doc/sgs_v9_grasp_validation.json
```

随机预览见[四类末端对比](sgs_preview/v9_random_seed43/overview.png)、[足底全景](sgs_preview/v9_random_seed43/grounding_overview.png)和[逐帧测量](sgs_preview/v9_random_seed43/measurements.json)。播放约 1.2 秒后，3 条 Near-Goal 仍保有双指接触及很小的相对滑移，但其中 2 条随机械臂下降而离开初始 5–10 cm 范围。配置库保证的是经过筛查的初态与短时抓持；后续继续抬升、保持高度仍由教师策略学习，不能将零增量播放解释为长期稳定抓持认证。

## 15. v9 训练奖励与成功归因修复

2026-10-25 修复三处原有逻辑，奖励项和 YAML 权重保持不变：

1. `grasp_base_height` 使用与 `pick_up` 相同的 `lifted_object` 条件。对 sugar_box，实际抬升必须超过 20 cm 且物体距末端小于 20 cm，才获得这项站姿奖励；不再在 17–20 cm 的未成功区间持续给奖励。Near-Goal 的初始 5–10 cm 不触发该奖励。原来的掉落判定保留。
2. `base_dir` 只清除物体方向的 z 分量，使用水平指向计算底盘朝向奖励。原实现错误地清除了 x/y，导致该奖励为零。
3. yaw 超过 70° 的失败提前在 `check_termination()` 中登记到 `_failure_now`，再计算成功保持奖励；第 15 个保持步若同时偏航失败，不登记成功。yaw 奖励函数只计算超过 60° 的罚分，不再修改重置标志。角差使用最短环绕距离，参考仍为任务原始朝向；该奖励权重为零或未配置时，保留原先禁用这一终止规则的行为。

训练命令及 `sugar_box_v9.pt` 无需修改或重建。四类继续共用任务奖励，SGS 只根据完成回合的成功标签调整配置采样。比较新旧训练结果时应记录代码版本；正式的 SGS / uniform 对照应使用同一修复版本。

**210 项 CPU / CUDA 回归测试通过**。新增的 7 项[奖励行为回归](../DQ_high-level/tests/test_teacher_rewards.py)覆盖成功高度门槛、正对/侧对/背对物体、yaw 惩罚与终止门槛、角度环绕，以及最终保持步偏航失败后向 SGS 登记失败的完整统计路径。

修复后已在 `cuda:1` 使用原 2048 条正式库完成 **32 环境、96 步、4 次 PPO 更新**，76 项训练链路检查通过；40 个完成回合正确登记，四类实际抽取次数为 **24、24、9、16**，模型参数有限且实际更新。原计划仍为 `--timesteps 120000`，通过 `--stop_after 96` 提前结束；没有开启在线 wandb。记录见 [sgs_v9_reward_fix_gpu_validation.json](sgs_v9_reward_fix_gpu_validation.json)。这次短训练验证运行与统计链路，不代表长期训练成功率改善。
