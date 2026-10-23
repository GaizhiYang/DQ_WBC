# SGS 教师训练使用说明

适用入口：`train_multistate_DQ_teacher.py`；任务：`B1Z1PickMulti`；物体：`sugar_box`；保留 `--no_velocity_obs`。默认 `sgs.mode=disabled`，原训练命令可以继续使用。

## 1. 环境与工作目录

```bash
conda activate dqwbc
cd /home/hehui/DQ_WBC/DQ_high-level
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

本实现已在本机 `dqwbc` 环境运行。当前机器仅激活 conda 不会自动加入其动态库目录；上面的 `export` 用于解决 Isaac Gym 导入时找不到 `libpython3.8.so.1.0` 的问题。

## 2. 构建可重复抽取的固定配置库（v5 简化版）

构库入口读取 [data/cfg/sgs_bank.yaml](../DQ_high-level/data/cfg/sgs_bank.yaml)，`schema_version: 2`。三类分别为 `full_task`（完整任务起步）、`pregrasp`（临近抓取）、`at_grasp`（已到达抓取位姿）。修改 YAML 后重新构库；训练只读取生成的 `.pt` 缓冲区。

```bash
python build_sgs_bank.py \
  --device cuda:0 \
  --num-envs 1024 \
  --per-family 512 \
  --seed 43 \
  --max-batches 512 \
  --reset-config data/cfg/sgs_bank.yaml \
  --output data/sgs/sugar_box_v5.pt
```

`--reset-config` 表示上述初态参数文件；`--config` 表示教师环境配置（默认 `data/cfg/DQ_teacher.yaml`），两者用途不同。当前 YAML 保留用户已经调节的底盘和临近抓取范围：

| YAML 字段 | 含义 | 当前值 |
| --- | --- | --- |
| `full_task.base.x_range_m` / `y_range_m` | 完整任务底盘到物体的世界 x / y 有符号间隔，单位 m | `[2.0, 2.0]` / `[0.0, 0.0]` |
| `full_task.base.heading_offset_range_deg` | 底盘朝向相对物体方向的偏角，单位度 | `[-9.1673, 9.1673]` |
| `full_task.arm_pose` | 完整任务机械臂关节姿态 | `default` |
| `pregrasp.base.x_range_m` / `y_range_m` | 临近抓取底盘与物体间隔 | `[0.68, 0.95]` / `[-0.2, 0.2]` |
| `pregrasp.base.heading_offset_range_deg` | 临近抓取底盘朝向偏角 | `[-1.4324, 1.4324]` |
| `pregrasp.ee_approach_distance_range_m` | 末端到候选抓取位姿的距离，单位 m | `[0.05, 0.15]` |
| `at_grasp.base.x_range_m` / `y_range_m` | 到达抓取位姿类别的底盘与物体间隔 | `[0.68, 0.85]` / `[-0.1, 0.1]` |
| `at_grasp.base.heading_offset_range_deg` | 到达抓取位姿类别的底盘朝向偏角 | `[-1.4324, 1.4324]` |
| `at_grasp.gripper_opening_range` | 随机夹爪开度，归一化 0 为闭合、1 为全开 | `[0.0, 1.0]` |

三个 `base` 使用相同定义：`x = object_world_x - robot_base_world_x`，`y = object_world_y - robot_base_world_y`；不是底盘局部坐标，也不是水平欧氏距离。偏角为 `wrap(robot_yaw - atan2(y, x))`，0° 表示正朝向物体，正值表示逆时针，范围须位于 `[-180, 180]` 内。

三类构造方法：

1. **完整任务起步**：正常重置，按 YAML 放置底盘、校正足底到地面附近，机械臂保持原任务实际 reset 使用的 `_initial_dof_pos` 默认关节姿态。
2. **临近抓取**：物体仍在桌上，夹爪张开；沿候选抓取位姿的 local +x 反方向退后指定距离，使用 IK 设置机械臂。检查距离、可达性、碰撞和足底几何。**不执行完整抓取脚本，不以脚本抓取成功作为入库条件。**
3. **已到达抓取位姿**：物体仍在桌上，使用 IK 将末端对齐候选抓取位姿，按 `gripper_opening_range` 随机设置关节位置和控制目标。**不预先抬升，不要求已可靠夹持。** `[0.2, 0.8]` 采样中间开度，`[0, 0]` 固定闭合，`[1, 1]` 固定全开；归一化值映射到 URDF 关节上下限。

`pregrasp.ee_approach_distance_range_m` 描述末端到候选抓取位姿的距离，与末端到物体中心的距离不同。旧的 `grasped_before_lift` 和 `object_lift_height_range_m` 已被删除，不再控制新库。旧 schema 配置不会被悄悄解释成新语义。

三类只要求四脚碰撞球底面大致位于地面附近，当前允许 **−5 mm 至 +20 mm**；用 **4 cm 碰撞球半径**计算。**不要求稳定站立，不检查连续承重脚数、接触力或底盘静止速度。** 构库使用短物理刷新检查几何和碰撞，保留关节限位、数值合法性及参数范围检查。取消连续站立等待、跨动力学环境保持复放、抓取见证和松爪掉落测试。碰撞检查允许足地及合理夹爪物体接触，不把“无任何接触”当作合法初态的条件。

目标范围先均匀采样，经过可达性和碰撞筛选后，最终库内分布不保证均匀。`[a, a]` 是固定值，只有这类零宽范围允许位置和末端距离 **±5 mm**、偏角 **±1°** 的物理容差。不可达设置仍可能导致样本不足；构库不会自动扩展用户范围或复制假样本补齐。

成功后输出 `.pt` 缓冲区和同名 `.json` 审计文件，记录配置数、指纹、`metadata.construction.reset_config` 以及 `validation_profile: geometry_v1`、`feet_near_ground: true`、`collision_checked: true`、`object_attachment: false`。目标仍是三类各 512 条、共 1536 条。训练加载核对新类别语义和几何验证标记；v1–v4 的第三类是旧的“已抓住、待抬升”，不能当作 v5 使用。旧库与旧报告作为历史资料保留。

构建一次后即可反复训练。修改 YAML 后需要重新生成文件，SGS 和 uniform 对照应读取同一个文件和指纹。训练期间从缓冲区抽取已有初态，**不会重跑离线 IK、碰撞筛选或抓取构造流程**。

## 3. 正式训练

先按第 2 节成功构建三类各 512 条的 v5 库，再在原命令上增加 `--sgs_mode sgs` 和 `--sgs_bank`。训练不读取 `sgs_bank.yaml`，也不接受构库入口的 `--reset-config`：

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
  --sgs_bank data/sgs/sugar_box_v5.pt \
  --seed 43 \
  --headless --wandb --wandb_name no_velocity_obs
```

固定库均匀采样对照只需将 `--sgs_mode sgs` 改为 `--sgs_mode uniform`，同时为 `--experiment_dir` 使用不同目录，保留相同 bank、seed、环境数和训练步数。原始任务基线使用 `--sgs_mode disabled`，无需 bank。

默认 SGS 参数为：1536 个配置、每配置最近 100 个已完成回合、目标成功率 0.5、`kappa=1`、`temperature=2`、`epsilon=1e-4`、`p_init=0`。配置位于 `data/cfg/DQ_teacher.yaml` 的 `env.sgs`。uniform/SGS 训练使用 hold 成功标签。

当前实现从头训练，不支持 SGS/uniform 的 `--resume` 或 `--checkpoint`。不能将 agent checkpoint 或诊断文件当作完整续训状态。

## 4. 小规模 GPU 冒烟

以下命令同样以第 2 节的 v5 库已构建成功为前提；本轮实际验证结果见第 12 节。

```bash
python train_multistate_DQ_teacher.py \
  --rl_device cuda:0 --sim_device cuda:0 \
  --timesteps 120000 --stop_after 96 --num_envs 32 \
  --task B1Z1PickMulti --object_name sugar_box \
  --experiment_dir DQ_teacher/sgs-smoke --wandb_name seed43 \
  --roboinfo --observe_gait_commands --small_value_set_zero \
  --rand_control --no_velocity_obs \
  --sgs_mode sgs --sgs_bank data/sgs/sugar_box_v5.pt \
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

在第 1 节的环境和目录中运行；默认读取 v5，每类随机抽取 3 条，共 9 条：

```bash
python visualize_sgs_bank.py --device cuda:0 \
  --bank data/sgs/sugar_box_v5.pt --samples-per-family 3 --seed 43
```

窗口打开后暂停，终端显示抽到的配置 ID。更换 `--seed` 可换一组，`--samples-per-family` 调节数量；该抽样用于检查，与训练 SGS 采样独立。

| 按键 | 功能 |
| --- | --- |
| `1` / `2` / `3` | 完整任务起步 / 临近抓取 / 已到达抓取位姿 |
| `←` / `→` | 切换当前类别的抽样配置 |
| `F` / `C` | 全景 / 末端与物体近景 |
| 空格 / `N` | 播放或暂停 / 前进一个高层控制步 |
| `R` | 重新恢复保存的初态和随机夹爪目标 |
| `O` | 人为切换开合目标，下一步生效 |
| `S` / `Esc` | 保存截图 / 退出 |

**默认播放保持库里保存的夹爪目标，包括第三类的随机开度。** 预览对当前环境实例单独保留该目标，因为训练原动作中 0 表示开爪，不能用 0 代表保持随机开度；训练动作语义未改。按 `O` 后才切换为全开/全闭，按 `R` 恢复库中的值。第三类没有物体已抓稳或会保持悬空的保证。

无窗口导出图片、GIF 和测量 JSON：

```bash
python visualize_sgs_bank.py \
  --device cuda:0 --headless \
  --bank data/sgs/sugar_box_v5.pt \
  --samples-per-family 3 --seed 43 --steps 8 \
  --output ../doc/sgs_preview/v5_random_seed43
```

输出 `overview.png`、全景 `grounding_overview.png`、每条配置带 ID 的图片/动画及 `measurements.json`。复查指定条目用 `--config-id 1024`。不同 bank 中相同 ID 不代表同一个状态。

对比图显示最大足底间隙和 `feet_near_ground`，后者只检查足底碰撞几何是否在 **[−0.005, 0.020] m**。`foot_contact_force_z_N` 仅供诊断，不作为稳定站立合格条件。`gripper_opening_normalized` 是实测关节开度，`gripper_target_opening_normalized` 是当前控制目标，均为 0 闭合、1 全开；碰撞可能令实际开度偏离目标。

恢复首帧只推进渲染所需的一个 **5 ms** 物理步，没有长时间落地等待。播放采用低层行走控制和机械臂 IK，高层位姿增量与底盘指令为零，保留 Level02 桌面运动。因此短动画展示初态的物理响应，不代表训练好的教师表现。`ee_to_object_center_m` 不能直接用来核对 YAML 的候选抓取距离。

旧版本预览图留在各自的 `v1`/`v2`/`v3` 目录。新版预览要求 `geometry_v1` 库，避免把旧的已抓持样本误标为新第三类。

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


## 12. v5 简化版验证记录

本节只记录简化版实际运行结果，不能沿用上文 v3/v4 的站立、抓持验证结果作为 v5 证明。当前变更包括三类新语义、schema 2 YAML、足底几何检查、随机夹爪开度及训练缓冲区恢复接口。

2026-10-22 已使用当前 YAML 生成正式库 [sugar_box_v5.pt](../DQ_high-level/data/sgs/sugar_box_v5.pt)。使用 `cuda:0`、1024 个构建环境，7 批收齐三类各 512 条，共 **1536 条**。采样、IK 与基础物理筛查累计 **6.791 秒**；该时间不含 Isaac Gym 启动、资产加载及最后文件保存。逐批记录见[构库审计](../DQ_high-level/data/sgs/sugar_box_v5.json)。

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
