# Scout & Relay 代码修改记录

> 日期：2026-11-24。对应设计：[SGS_SAPG_scout_relay_design.md](SGS_SAPG_scout_relay_design.md) §4、§6.2、§9。
> 状态：代码已完成，**尚未在 Isaac Gym 中运行**。本机没有 torch 和 GPU，只跑通了不依赖 torch 的配置与启动脚本测试（7 项）。依赖 torch 的测试和训练冒烟测试需要在训练机上执行，见文末。

## 1. 实现了什么

- **多分支策略（SAPG）**：actor 和 critic 共享骨干，每个分支有可学习条件向量 φ_j（16 维，拼在 MLP 输入末尾）和独立 `log_std`。分支 0 是 leader，也是唯一部署的策略。
- **SAPG 更新**：scout 只用自己的数据做 PPO；leader 用自己的数据，再加等量 scout 数据的 IS 修正项（μ·min(ρA, clip(ρ)A)，A 为旧 leader 的一步 TD 残差，critic 目标 r + γV_L(s′)）。
- **侦察课程**：每个 scout 有自己的 SGS 统计，采样权重为 [(w(p_S)+ε)(1−p_L+ε)^β]^(1/T)，只在静态 bank 上采样。
- **接力库**：scout（或自接力时的 leader）的成功回合，若 leader 在该配置上成功率 ≤ 0.3，就把 hold 开始前 f·T_s（f = 0.2/0.4/0.6/0.8）处的物理快照写入接力库。leader 在 bank ∪ 接力库上运行同一个 SGS 核，接力库总概率质量上限 α_max = 0.3。接力项用 Laplace 估计成功率，并有隔离、掌握、无用、容量淘汰四种退休方式。
- **指标与诊断**：leader / scout / 能力差距 / 接力库 / IS 有效样本量（含 H1 所需的按配置 ESS 与 gap 的相关性）。
- **部署导出**：把 φ_L 精确折叠进第一层偏置，导出与原 214 维单策略网络完全等价的检查点。

默认参数（`branches: 1`、`relay.enabled: False`）下走原代码路径，原有 SGS 基线行为不变，包括采样器随机数的调用顺序。

## 2. 文件改动

### 新增

| 文件 | 内容 |
|---|---|
| `DQ_high-level/utils/scout_relay.py` | `branch_layout`（分支环境划分）、`scout_probabilities`（侦察课程）、`SnapshotBuffer`（回合内快照）、`RelayBank`（接力库：采样、上限、链记账、退休、指标）、`ISOverlapTracker`（按配置的 Kish ESS） |
| `DQ_high-level/utils/sapg_ppo.py` | `SAPGPPO`（继承 skrl PPO）：带分支 ID 的 act/记录/更新，leader 的 IS 项，按分支归一化优势，诊断日志 |
| `DQ_high-level/train_teacher_scout_relay.sh` | 新启动脚本，`VARIANT=A…G` 对应设计 §6.2 的实验组，公共底座与 `train_teacher_sgs_ekf.sh`（T1）相同 |
| `DQ_high-level/export_sapg_leader.py` | 把 SAPG 检查点导出为单分支 leader 运行目录（检查点 + YAML + teacher_run.json） |
| `DQ_high-level/tests/test_scout_relay.py` | 接力库、快照选择、侦察课程、ESS、环境重置全流程（AST 抽取真实环境方法）、导出折叠的单元测试（需 torch） |
| `DQ_high-level/tests/test_scout_relay_config.py` | 配置合并与校验、YAML 默认值、启动脚本各组参数（不需 torch，已在本机通过） |

### 修改

| 文件 | 改动 |
|---|---|
| `utils/sgs.py` | 抽出 `sgs_logits()` 核函数（`probabilities()` 数值不变）；新增 `SGSSampler.logits()`、`reset_entries()` |
| `envs/b1z1_pickmulti.py` | 新增 `_init_scout_relay`、`_reset_scout_relay_envs`、`_create_relay_chains`、`_capture_relay_snapshots`、`_relay_state_values`、`scout_relay_gap`、`_scout_relay_metrics`；把原 `_reset_sgs_envs` 的恢复部分拆为 `_restore_sgs_state`（基线路径逻辑不变）；`step` 末尾采快照；`check_termination` 记录翻倒/高度/IK 失败供隔离判定；`SuccessRate / sugar_box` 只统计 leader；`save_sgs_diagnostics` 增加 `scout_relay` 一节 |
| `train_multistate_DQ_teacher.py` | Policy/Value 增加 `num_branches`、`branch_latent_dim`；无 `branch_ids` 输入时默认用 leader；训练且分支数 > 1 时使用 `SAPGPPO`；评估恢复 `sapg_branches`、`sapg_latent_dim` |
| `utils/config.py` | `SAPG_DEFAULTS`、`RELAY_DEFAULTS`、`apply_scout_relay_config()`（合并 YAML 与命令行并校验）；新增命令行参数（见 §3） |
| `data/cfg/DQ_teacher.yaml` | 新增带注释的 `sapg:`、`relay:` 配置块（默认关闭） |
| `eval_sgs_teacher.py` | 结果中记录模型分支数和被评估分支（leader）；评估环境若带分支或接力则报错 |
| `tests/test_sgs_env.py` | 测试用环境的方法清单加入 `_restore_sgs_state`（拆分后基线测试仍走真实代码） |

`train_teacher_sgs_ekf.sh` 未改动（其中 `--action_scaling normalized` 是之前已有的未提交改动）。

## 3. 新参数

命令行参数优先于 YAML，未给出时用 YAML 或默认值。

| 参数 | 默认 | 含义 |
|---|---|---|
| `--sapg_branches` | 1 | 分支数 M；1 即普通 PPO |
| `--sapg_leader_fraction` | 0.5 | leader 的环境占比，scout 均分其余 |
| `--sapg_latent_dim` | 16 | 条件向量维度 |
| `--sapg_off_policy_coef` | 1.0 | SAPG 的 λ；0 = 关闭 IS 通道（消融） |
| `--sapg_entropy_coef` | 0 | scout j 加 coef·j·熵，leader 不加 |
| `--sapg_mu_max` | 0 | IS 权重截断；0 = 不截断（原版 SAPG） |
| `--scout_need_beta` | 1.0 | 侦察课程的需求指数；0 = 各分支独立 SGS |
| `--scout_target_success` | 同 leader | 每个 scout 的 SGS 目标成功率 |
| `--relay` / `--no_relay` | 关 | 启用接力库 |
| `--relay_source` | scouts | scouts（方法）/ leader（自接力基线） |
| `--relay_trigger` | success | success（方法）/ failure（消融） |
| `--relay_alpha_max` | 0.3 | 接力库概率质量上限；1 = 不设上限 |
| `--relay_tau_need` | 0.3 | 仅在 leader 成功率 ≤ τ 的配置上建链 |
| `--relay_fractions` | 0.2 0.4 0.6 0.8 | 接力点相对 hold 前时长的位置 |
| `--relay_sampling` | sgs | sgs（方法）/ uniform（消融：接力不用 SGS） |
| `--relay_snapshot_interval` | 5 | 快照间隔（高层步） |
| `--relay_capacity`、`--relay_max_chains_per_config` | 4096、2 | 容量与每配置活跃链数 |
| `--relay_controller_state` | 关 | 消融：恢复时同时恢复步态相位和低层历史 |

只在 YAML 中设置的项：`relay.history_size`(50)、`master_rate`/`master_count`(0.9/20)、`useless_count`(40)、`quarantine_steps`(3)、`max_new_chains_per_reset`(32)、`sapg.chunk_size`。

## 4. 实验组与运行

```bash
VARIANT=F bash DQ_high-level/train_teacher_scout_relay.sh
```

| VARIANT | 追加参数 | 设计 §6.2 |
|---|---|---|
| A | `--sgs_mode uniform` | PPO + bank 上均匀 |
| B | 无 | PPO + SGS（当前方法） |
| C | `--sapg_branches 2 --scout_need_beta 0` | SAPG + leader SGS |
| D | C + `--relay --relay_source leader` | 自接力 |
| E | C + `--relay --relay_source scouts` | scout 接力 |
| F | `--sapg_branches 2 --scout_need_beta 1 --relay --relay_source scouts` | 完整方法（默认） |
| G | `--relay --relay_source leader` | 单策略自接力 |

消融参数直接追加在命令末尾，例如 `--sapg_off_policy_coef 0`、`--relay_sampling uniform`、`--relay_fractions 0.8`。

## 5. 日志指标

| 前缀 | 内容 |
|---|---|
| `SuccessRate / sugar_box` | **仅 leader** 的 full_task 窗口（与基线可比）；scout 另记 `SuccessRate / scout_full_task` |
| `SGS / …` | leader 在 bank 上的统计（原有含义）；新增 `concurrent_copies` = leader 环境数 / exp(H(q))（P0 诊断①） |
| `Scout / …` | 各族成功率与采样比例、`leader_success_of_sampled`（scout 所练配置上 leader 的期望成功率，越低说明越聚焦 leader 弱项） |
| `Gap / …` | 最优 scout 与 leader 的成功率差（双方各 ≥ 20 回合的配置）：均值、≥ 0.3 的比例 |
| `Relay / …` | 活跃项与链数、各类退休计数、接力起步占比、上限后的质量、各 f 的成功率、`sampled_fraction_mean`（H5 的接力点前移曲线） |
| `SAPG / …` | 各分支 KL、策略损失、动作 std；μ 的均值、最大值、尾部比例；批量 ESS；按配置 ESS 的中位数（按 gap ≥ 0.3 分组）与 `corr(gap, config ESS)`（H1） |

注意：`Reward / …` 曲线由 skrl 统计，混合了所有分支。

训练结束时的 `sgs_diagnostics.pt` 新增 `scout_relay` 一节：scout 成功率与计数、按配置的 gap、接力库统计、按配置的 IS 矩。这一节只用于分析，不能用于续训。

## 6. 评估与部署

- **评估**：照常使用 `eval_sgs_teacher.py --checkpoint …`。评估时环境没有分支和接力；模型保留全部分支，但输入不带分支 ID，因此执行的是 leader。
- **导出 leader**：真机或旧版部署代码只接受单分支网络时，先导出：

  ```bash
  cd DQ_high-level && python export_sapg_leader.py --checkpoint RUN/checkpoints/agent_120000.pt --output_dir RUN_leader
  ```

  导出的检查点 actor 输入为原来的 214 维，按构造与 leader 数值等价（对应单元测试 `LeaderExportTests` 需在训练机上运行），可直接用原评估脚本评估。

## 7. 与设计稿的差异与未实现项

- on-policy 部分沿用本仓库的 GAE（原版 SAPG 用 3 步回报）；off-policy 用一步 TD，与设计 §5.3 一致。
- 每个分支的优势单独归一化；off-policy 样本的单样本权重与 on-policy 相同（M = 2 时 leader 的 on:off = 1:1）。
- 按配置的 ESS 基于参与 leader 更新的 scout 子样本（M = 2 时就是全部 scout 数据），按更新做指数平均。
- 快照恢复与 bank 一致：控制器历史清零。可用 `--relay_controller_state` 做对照。Level02 的桌面运动与模式计时无关，所以没有保存模式计时。
- **未实现**：P0 的“恢复—开环回放”一致性脚本、同配置副本间的轨迹离散度指标、带接力库的续训（与原 SGS 一样只能从头训练）。

## 8. 上 GPU 前请先做

1. **单元测试**（训练环境中，从 `DQ_high-level` 运行；`test_sgs*` 用来确认拆分后的基线没有回归）：

   ```bash
   cd DQ_high-level && python -m pytest tests/test_scout_relay.py tests/test_scout_relay_config.py tests/test_sgs_env.py tests/test_sgs.py tests/test_sgs_logging.py tests/test_teacher_entry.py
   ```

2. **冒烟测试**（小规模跑通 SAPG 更新和接力流程）：

   ```bash
   VARIANT=F NUM_ENVS=64 RUN_NAME=relay_smoke bash DQ_high-level/train_teacher_scout_relay.sh --no-wandb --stop_after 48
   ```

3. **基线一致性**：`VARIANT=B` 与 `train_teacher_sgs_ekf.sh` 的命令只差输出目录，代码路径和随机数调用顺序也相同。GPU 物理本身不完全确定，同种子曲线应落在同一范围内。
4. **开销**：M = 2 时 leader 每轮多一份 scout 样本的前向和反向，接力还要额外存快照（4096 个 scout 环境约 160 MB 显存）。请记录 GPU 小时和显存，供公平比较。
