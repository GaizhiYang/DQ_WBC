# 主线 A 实现说明：学到的时间可达性与拦截裕度

> 2026-11-22。对应 `顶会论文主线思路.md` §3。本机没有 torch / GPU，测试只能在服务器上跑（§6）。
> 第一次完整训练 ours_s44 失败，奖励塑形已改为最早拦截时间势函数，原因见 §2.1。
> 所有新功能默认关闭；不加新参数时，训练与 T1 基线完全相同。

---

## 1. 实现了什么

| 文档 §3.2 | 代码 | 开关 |
|---|---|---|
| 自身时间距离 `T(s, g)` | `modules/reachability.py`：`TimeToReach`（`mlp` / `quasimetric` / `analytic`）、`HindsightBuffer`、`ReachLearner` | `--reach_model` |
| 拦截裕度 `M = max_{c,h}[h − T(s, ĝ_c(t+h))]` | `interception_margin`；环境内每步计算两份：特权版 `M_priv` 和 EKF 版 `M_ekf` | 有 `--reach_model` 即计算并记录 |
| 奖励：势函数塑形 | `utils/margin_task.py::_reward_margin_shaping`，势函数来自最早拦截时间 τ\*（`intercept_time`），默认与 `approaching` / `base_approaching` 并存 | `--margin_reward` |
| 表征：时空 token | 每个预测时刻 h 取最优候选，组成 16 个 token；GFM 对 30 个当前候选和 16 个 token 一起做注意力 | `--margin_tokens` |
| 课程：动态 SGS | M0 可行性先验（Beta 伪计数）＋ 前沿变异（加快桌速） | `--margin_curriculum prior / frontier` |
| T 的在线更新 | EMA 目标网络，训练到 50% 后冻结，随 checkpoint 保存 | `--reach_online` |
| 关卡 D1 / D2 / D3 | `diagnose_pursuit.py` / `train_reachability.py` / `evaluate_margin_auc.py` | 独立脚本 |

### 信息边界（直接部署的前提）

| 量 | 用在哪 | 真机可得？ |
|---|---|---|
| T 的输入：末端位姿、臂关节、夹爪、末端目标积分器、底座指令、IMU 角速度和重力方向 | 全部 | 是（`REACH_STATE_LAYOUT`；不含底座线速度，与 `--no_velocity_obs` 一致） |
| 物体未来位姿：世界系 EKF 外推 | Actor 的 token 和 `M_ekf` | 是。需要里程计把检测到的位姿转到里程计坐标系，只要求 EKF 窗口内（约 0.75 s）漂移小 |
| 物体未来位姿：桌子真实运动（当前扭量外推） | 奖励 τ\*_priv、课程 M0、日志 `M_priv` | 否，只在训练中使用 |

Actor 从不接触 τ\*_priv 和 `M_priv`。

---

## 2. 关键设计（与文档 §3.2 不同或文档没写的地方）

1. **目标 g 只用位置，不含姿态。** 抓取候选的姿态约定和末端连杆不一定一致，用在 hindsight 标签上会出错；而底座走过去的时间远大于手腕转动的时间。以后需要的话，可以加上 6D 姿态。
2. **查询坐标系**：原点在臂基座，x 轴沿底座朝向，z 轴竖直，在查询时刻冻结。目标、状态、token 都用这个坐标系。
3. **标签与损失**：`(s_t, t+k 时刻的末端位置)` 的标签是实际耗时，它是最短时间的上界；用 τ = 0.1 的 expectile 回归拟合下包络。`T = 物理下界 + softplus(残差)`。物理下界对任何参数都成立：臂基座速度不超过 0.45 + 0.6×0.3 m/s，再加上手臂和转动的贡献。
4. **quasimetric 版本**是 MRN 距离 `max relu(ψ−φ) + ‖φ′−ψ′‖`。它目前只是结构上的归纳偏置，训练损失和 mlp 版相同，**还没有实现 QRL 的对偶目标**。论文如果要声称“quasimetric 学到最优时间”，需要补上这一项。
5. **势函数**（第一版失败后已修改，见 §2.1）：
   - 默认 `intercept`：`Φ = 1 − τ*/8 s ∈ [0, 1]`。τ\* 是最早拦截时间，即每个时刻的最优候选裕度 `m(h) = h − T(s, g(t+h))` 第一次 ≥ 0 的 h，在相邻预测时刻之间线性插值；
   - 物体静止时 τ\* 就等于 T(s, g)，到达抓取点时为 0，所以 Φ 一直到抓取都在变，不会饱和；
   - 4.5 s 内都拦不到时，τ\* 延续为 `T(s, g(t+4.5))`，在边界上连续；上限是 T 自身的截断值 8 s；
   - τ\* 不受回合剩余时间影响，Φ 只取决于状态和物体运动；
   - 物体抬起后 Φ = 1，终止状态（含超时，PPO 这里不在超时处自举）Φ = 0。因为 Φ ≥ 0，终止那一步的塑形奖励不会为正。折扣总和严格等于 `−Φ(s₀)`（命题 P2，有测试）；
   - 默认权重 1，且**保留** `approaching` / `base_approaching`。按最优速度前进时，Φ 每步约增加 0.15/8 ≈ 0.02，与原接近奖励每步 0.01–0.04 相当；
   - `slack`（`--margin_potential slack`）是第一版：`Φ = (clip(M, −2, 2)/2 + 1)/2`，只作消融保留。
6. **token**（未改动）：每个预测时刻一个 token，内容是该时刻裕度最大的候选：位置（3）、6D 姿态（6）、h、该时刻的裕度、是否有效，共 12 维。共 16 个时刻（0–4.5 s），加上 `M_ekf` 和最佳时刻两个标量，每帧多 194 维。
   - 观测标准化会把常数通道（如 h）变成 0，所以 GFM 为每个 token 槽位学一个位置嵌入；
   - token 只进 GFM，不进 Actor MLP；两个标量进 MLP；
   - Critic 的 MLP 看到整个块。
7. **课程先验**：SGS 估计成功率时，混入 `strength = 4` 个伪样本，伪样本的成功率为 `p₀ = sigmoid((M0 − δ)/0.5)`。
   - 只影响采样，记录的成功率仍是实测值；
   - 没访问过、M0 ≈ 0 的配置，先验约为 0.5，会被多采样（“难”）；
   - M0 ≪ 0 的配置先验接近 0，基本不被采样（“不可能”）；
   - δ 在训练中从 0.5 s 线性降到 0。
8. **前沿变异**：每次 full_task 重置时，以 25% 的概率尝试把桌子的线速度和角速度同乘一个系数 1.0–1.6（保持圆周半径不变），取 M0 仍不低于 δ 的最大系数。这类回合单独统计为 `SuccessRate / frontier`，不进 SGS 历史、family 窗口和主指标 `SuccessRate / sugar_box`。这一步也直接给出速度外推实验（0.48 m/s）。
9. **不兼容**：不能与 TSIL、相机、浮动基座同时使用；token 要求有物体特征，且 `lastCommands=False`。

### 2.1 第一次完整训练（ours_s44）为何失败

ours_s44 同时打开了奖励、token 和课程，并用了第一版塑形（`slack`，权重 10，去掉 approaching）。到第 190 轮被停止时，near_goal 成功率一直为 0（T1 第 40 轮已到 0.70）。排查的结论是：数据流没有问题，失败来自第一版塑形的设计。

- **量级**：塑形每步平均 −0.14 到 −0.53，是 T1 中 approaching（0.011–0.039）的 5–15 倍。每个回合的最后一步要付 −10Φ。势函数不变性要求 Critic 把 −10Φ(s) 学准；Critic 在第 40 轮前后 value loss 升到 6.85，之后解释方差恒为 0，即对所有状态输出同一个值；
- **近处没有信号**：approaching 被去掉；物体附近 M ≥ 2，被截断后 Φ ≡ 1；
- **M 被最远时刻主导**：物体比机器人慢时，h − T 随 h 增大，最佳时刻的均值为 3.8–4.3 s（上限 4.5 s），所以 M ≈ 4.5 − T(物体约 4 s 后的位置)。它引导的是物体 4 秒后的位置，而不是当前的抓取点。

`intercept` 势函数、权重 1、保留 approaching 这三处修改分别针对这三点。

---

## 3. 运行顺序

以下命令都在 `DQ_high-level` 下执行。`<B1>` 指当前最好的 T1 基线 checkpoint，例如 `DQ_teacher/b1-pick-sugar-box-sgs-ekf/sgs_ekf_sugar_box_seed43_20261120_220722/checkpoints/agent_120000.pt`。

### 3.1 关卡 D1：现有教师是在“追”还是在“拦”

```bash
python diagnose_pursuit.py --checkpoint <B1> --task B1Z1PickMulti --object_name sugar_box --headless --rand_control --eval_episodes 800 --eval_output outputs/d1_pursuit.json
```
先看 `summary.linear` 和 `summary.circle` 两组：

- **追踪**：`heading_error` 远小于 `lead_reference`，`los_rate` 很大；
- **拦截**：`los_rate ≈ 0`，`heading_error ≈ lead_reference`。

### 3.2 训练 T，同时完成关卡 D2

```bash
python train_reachability.py --checkpoint DQ_teacher/b1-pick-sugar-box-sgs-ekf/sgs_ekf_sugar_box_seed43_20261120_220722/checkpoints/agent_120000.pt --task B1Z1PickMulti --object_name sugar_box --headless --rand_control --num_envs 2048 --reach_output data/reach/sugar_box_mlp
```

- 默认用随机采样的动作（覆盖更多状态），跑 3000 步，每步更新 2 次；
- 编号是 10 的倍数的环境留作验证集；
- 输出 `reach_model.pt` 和 `reach_report.json`。`passed` 的标准是：留出 MAE ≤ 0.8 × 解析估计的 MAE；
- 另外看 `learned_above_label_fraction`，它应该很小，否则 T 不是下包络；
- `--reach_kind quasimetric` 训练对照版本。

### 3.3 关卡 D3：M0 能否预测配置的难度

先用 B1 的设置跑一个半程训练，得到 `sgs_diagnostics.pt`（训练收敛后几乎所有配置都成功，AUC 没有负样本）：

```bash
SEED=43 RUN_NAME=b1_d3_probe bash train_teacher_sgs_ekf.sh --action_scaling normalized --stop_after 7200
```

```bash
python evaluate_margin_auc.py --bank data/sgs/sugar_box_v12.pt --sgs_diagnostics DQ_teacher/b1-pick-sugar-box-sgs-ekf/b1_d3_probe/sgs_diagnostics.pt --reach_model data/reach/sugar_box_mlp/reach_model.pt --output outputs/d3_margin_auc.json
```

`passed` 的标准是：full_task 上 learned 的 `auc_half` > 0.6。同时输出 analytic 的结果作对照，以及 `calibration` 分箱（命题 P4）。

### 3.4 主实验：2³ 析因（每格 ≥ 3 个种子）

以下 R、T、C 分别指奖励、表征（token）、课程三个环节。所有组都加 `--action_scaling normalized`；除 B1 外，都再加 `--reach_model data/reach/sugar_box_mlp/reach_model.pt`。

**先跑 R、T、C 三个单因素组**，各约 500 轮（`--stop_after 12000`）。和 T1 对比第 40–60 轮的 near_goal、第 190 轮的 pregrasp 成功率，以及 `Value / Explained variance`。单因素都不差于 T1 之后，再跑组合和多种子。

| 组 | 追加参数 |
|---|---|
| B1（基线） | 无 |
| R | `--margin_reward` |
| T | `--margin_tokens` |
| C | `--margin_curriculum frontier` |
| R+T、R+C、T+C | 对应组合 |
| **Ours** | `--margin_reward --margin_tokens --margin_curriculum frontier` |

示例（单因素 R）：

```bash
SEED=44 RUN_NAME=r_intercept_s44 bash train_teacher_sgs_ekf.sh --action_scaling normalized --reach_model data/reach/sugar_box_mlp/reach_model.pt --margin_reward --stop_after 12000
```

R 的消融：

- **R-slack**：`--margin_reward --margin_potential slack`，第一版势函数（权重仍为 1，保留 approaching）；
- **R-replace**：`--margin_reward --no_margin_keep_tracking`，塑形作为唯一的接近信号；
- 权重：`--margin_reward_scale 0.5 / 2`。

基线对应：

- **B3 解析可达性**：`--reach_model analytic`，其余同 Ours；
- **B4 距离势函数**：`--approach_shaping potential`（T2 已实现）；
- **B5 均匀重置**：`--sgs_mode uniform`；
- **从零开始、在线学 T**：`--reach_model new:mlp --reach_online`（对应“鸡生蛋”的质疑）。

评估仍然用 `eval_sgs_teacher.py`。margin 配置保存在 run 的 YAML 里；T 的副本在 run 目录 `reach_model_initial.pt`；在线版本的 T 保存在 checkpoint 中。

### 3.5 训练中要看的曲线

- `Margin / privileged intercept time mean (s)`：τ\* 的均值，应随训练下降；
- `Value / Explained variance`：应保持为正。若长时间恰好为 0，说明 Critic 已经对所有状态输出常数（ours_s44 的症状）；
- `Margin / privileged mean (s)`、`Margin / privileged nonnegative fraction`：它们应随训练上升；
- `Margin / EKF minus privileged abs mean (s)`：EKF 预测引入的误差；
- `Curriculum / delta (s)`、`Curriculum / frontier speed scale mean`、`SuccessRate / frontier`；
- 在线版本另有 `Reach / loss`、`Reach / validation learned_mae`。

---

## 4. 计算开销

每步对每个环境，特权版和 EKF 版各计算 16 × 30 = 480 次 T，按 1024 个环境一块分块计算（`chunkEnvs`）。T 的状态编码对每个环境只算一次；每个目标只经过一个线性投影和两层 64 维的头。8192 个环境下估计每步增加约十几毫秒，**尚未实测**。显存不够时把 `env.margin.chunkEnvs` 调小。

---

## 5. 还没做 / 已知局限

- **B2 基线**（Spatiotemporal Agility 风格：预测拦截点 + 解析 t_go + 时间门控奖励）还没写；
- 主线 B（IMM 多假设）没做。目前 token 里没有 EKF 不确定性；
- 特权未来轨迹按桌子当前的扭量外推：对 Level00–02 是精确的，对 Level03/04 只在下一次随机切换之前准确；
- quasimetric 版本没有 QRL 对偶目标（见 §2.4）；
- 在线 T 在冻结之前一直在变，这段时间的塑形不严格满足势函数不变性（P2）。论文主实验建议用离线训练、固定不变的 T；
- D1 的“接近阶段”在距离 < 0.6 m 或物体首次被抬起时结束，阈值可在脚本顶部修改。

---

## 6. 测试

有 torch 的机器上运行：

```bash
cd DQ_high-level && PYTHONPATH=tests python -m unittest test_margin test_teacher_gfm test_value_learning test_critic_time_obs test_sgs_env test_sgs_logging test_pick_task test_baseline_reset_parity test_tsil_env -v
```

`test_margin` 覆盖：

- 桌子运动外推与数值积分一致；
- 裕度的剩余时间掩码；
- T ≥ 物理下界，模型保存与读取；
- hindsight 标签不跨回合，且坐标变换正确；
- 学习器能学会匀速时间，并优于解析估计；
- M0 随物体靠近、远离的单调性；
- SGS 先验只影响采样；
- 最早拦截时间：静止目标等于 T，取第一次穿越，在最后一个时刻连续，并截断到上限；
- 塑形奖励的折扣总和等于 `−Φ(s₀)`（两种势函数）；
- token 块的布局；
- 前沿变异只作用于 full_task；
- 观测块的位置不移动已有的通道；
- Actor 只经由 GFM 使用 token；
- 配置校验。

其余测试用来确认原有功能没有被改坏。
