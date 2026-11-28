# 当前变速 Level03 的 EKF 实测（2026-11-27）

当前 EKF 在变速 Level03 上仍有预测收益，但其收益取决于对照方法、预测时距和运动阶段。它显著优于“保持当前位姿”；在无观测噪声的本次测试中，最近两帧的简单匀速外推在总体位置误差上略优于 EKF。刹车时，0.6/0.9 秒 EKF 预测甚至不如保持当前位置。

## 实测设置

- 使用当前仓库的真实 Isaac Gym 环境、生产 `RollingSE3Predictor` 和当前 Level03 遥控车运动实现，未修改生产 EKF 或训练参数。
- Level02、Level03 各运行 seed 43、47；每个种子 16 个环境、500 个高层步，约 75 秒仿真时间，使用 GPU 0。
- EKF：history=5，Q scale=1.0，R scale=0.01；测试 0.3、0.6、0.9 秒。训练入口默认使用 0.6/0.9 秒，0.3 秒是额外诊断。
- 高层动作全零，原底层站立策略继续运行；SGS、TSIL、相机和额外随机物体瞬移关闭，普通 reset 保留。
- 输入与 teacher 一样，来自干净的 simulator root poses，没有另加传感器噪声。结论针对预测质量，不是对真实传感器去噪性能的证明。
- 主指标在生产 EKF 使用的随机器人运动的 arm frame 中计算：预测未来相对位姿，与未来时刻的相对真值比较。世界坐标回放仅作辅助诊断。
- 使用实际 simulation timestamps；5 帧历史跨度中位数约 0.60 秒。真值通常每 0.02 秒记录一次，reset 额外采样。未来位置线性插值、旋转用 SLERP，禁止跨 episode 或向外插值。
- 四种方法使用相同预测起点、相同已观测历史和相同未来真值。Level03 共 **14,507 个共同预测起点，160 个 episode**；不同 horizon 也共享起点。重叠窗口不能视为独立试验。

## Level03 总体结果

以下位置误差是三维欧氏距离的均值，单位 cm，越小越好。

| 方法 | 0.3 秒 | 0.6 秒 | 0.9 秒 |
|---|---:|---:|---:|
| 当前在线 EKF | 0.97 | 2.20 | 3.71 |
| 保持当前位姿 | 5.11 | 10.10 | 15.02 |
| 最近两帧匀速外推 CV2 | 0.84 | 2.06 | 3.61 |
| 同五帧窗口匀速外推 CV5 | 1.13 | 2.57 | 4.31 |

相对保持当前位姿，EKF 在 0.6/0.9 秒上的平均位置误差下降约 **78%/75%**。不过，和 CV2 相比，EKF 的位置误差仍多约 0.14/0.10 cm；不能把相对 hold 的收益全部归因于 EKF 滤波本身。

| EKF 指标 | 0.3 秒 | 0.6 秒 | 0.9 秒 |
|---|---:|---:|---:|
| 位置误差 P95（cm） | 4.10 | 8.53 | 13.47 |
| 位置误差 ≤2 cm 的比例 | 87.68% | 71.18% | 46.12% |
| 平均旋转误差（°） | 0.50 | 1.12 | 1.92 |

CV2 的平均旋转误差约为 0.51/1.15/1.99°；EKF 在总体旋转预测上略好。均值较小不代表每次预测都精确，0.9 秒预测的长尾尤其明显。

以整个 episode 等权、配对 bootstrap 2,000 次得到的“EKF−CV2”平均位置误差差值及描述性 95% 区间为：0.3 秒 +0.131 cm [0.101, 0.157]，0.6 秒 +0.143 cm [0.099, 0.184]，0.9 秒 +0.096 cm [0.032, 0.155]。正值表示本次采样中 CV2 更好；只有两个随机种子，不能把这些区间解释成新种子或新策略上的保证。

作为运动分布参照，Level02 共 14,502 个起点、161 个 episode，其在线 EKF 平均位置误差为 0.66/1.29/2.07 cm；Level03 为 0.97/2.20/3.71 cm。Level03 的确更难预测，但两者的速度方向、曲率、停走和加速度分布都不同，差异不能全归因于加速度。

世界坐标辅助回放中，Level03 的平均位置误差为 0.48/1.24/2.24 cm。这不等同于 teacher 实际使用的相对坐标结果：机器人自身运动也会改变相对位姿，而且辅助评估的可用起点可能不同。

## 变速和刹车的影响

加减速标签仅用于划分评估样本：取预测起点之前 0.10 秒的桌面指令平面速度模长变化率，以 ±0.02 m/s² 分组。这不是物体实测加速度，也没有把桌面速度输入预测器。倒车时按速度大小判断加速/减速。

| 起点状态 | 起点数 | EKF 0.6 秒（cm） | CV2 0.6 秒（cm） | EKF 0.9 秒（cm） | CV2 0.9 秒（cm） |
|---|---:|---:|---:|---:|---:|
| 速度大小增加 | 5,759 | 3.13 | 2.84 | 5.08 | 4.80 |
| 速度大小下降 | 4,044 | 2.06 | 1.77 | 3.49 | 3.15 |
| 速度近似稳定 | 4,704 | 1.18 | 1.34 | 2.23 | 2.56 |

EKF 在速度稳定组优于 CV2，在加减速组落后于 CV2。这与有限历史窗口对速度变化的响应滞后相符，但这里的分组是观察性比较，不是单独控制加速度的因果实验。

另按 Level03 driver 的 **BRAKE 阶段**分组（526 个起点）：

| 方法 | 0.3 秒（cm） | 0.6 秒（cm） | 0.9 秒（cm） |
|---|---:|---:|---:|
| EKF | 1.58 | 4.11 | 7.21 |
| 保持当前位姿 | 2.20 | 3.42 | 4.07 |
| CV2 | 1.01 | 3.03 | 5.61 |

刹车起点的远期预测明显过冲，EKF 在 0.6/0.9 秒上比 hold 更差。阶段切换还按各自预测时距独立统计，0.8 秒后发生的切换不会算作 0.3 秒预测窗口内的切换。详细数据见输出 CSV。

## 原因与使用判断

`modules/pose_prediction.py` 每次在最近 5 个测量上重新拟合，能够随新观测更新速度；但未来外推使用固定 body twist，没有加速度状态，也不知道未来随机油门/转向命令。

所以“模型假设匀速”不等于“变速就完全无效”。当前结果支持继续把 EKF 作为预测信息；同时，0.9 秒输出应被视为不确定性较大的远期参考。若下一步优化，应把 CV2 保留为对照，先比较更短时距/历史窗口，再评估加速度状态或运动阶段自适应，而不能仅以优于 hold 判定改进成功。

本次没有训练或调用高层抓取策略，**不能推出加上 EKF 后抓取成功率提高多少**。验证该结论需要匹配训练预算和种子的 EKF 开/关或预测器替换实验；直接给已训练网络删掉输入也不是公平对照。

## 脚本补全与复现

- `collect_online_ekf.py`：保留在线预测及真实五帧历史；补充密集阶段/速度指令记录、采样与源码状态元数据，以及同时间戳 reset 的保护。普通 reset 会推进物理时间，旧采集不因此被判为无效。
- `utils/ekf_motion_evaluation.py`：reset 历史边界保护、插值端点浮点容差、逐时距事件统计、加减速分组。
- `utils/ekf_world_diagnostic.py`：使用记录的 Q/R 回放生产 EKF，并检查与在线输出的一致性。
- `summarize_online_ekf.py`：实际种子/环境数/采样间隔，分种子指标，以及按整个 episode 配对 bootstrap 的描述性 95% 区间。差值为 EKF−对照，负值更好；episode 等权与总体起点等权的均值有区别，区间不能代表跨新种子的泛化保证。
- `run_online_ekf_check.py`：一条命令串联采集、评估和汇总；支持复用完整采集，显式指定的种子不会混入其他目录。

新建输出目录运行完整测试：

```bash
conda activate dqwbc
cd DQ_high-level
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
CUDA_VISIBLE_DEVICES=0 python run_online_ekf_check.py \
  --output outputs/my_level03_ekf_check \
  --seeds 43 47 --num-envs 16 --steps 500 --device cuda:0
```

复用本次采集和评估，重新汇总：

```bash
CUDA_VISIBLE_DEVICES=0 python run_online_ekf_check.py \
  --output outputs/level03_ekf_check_20261127_v2 --skip-collect
```

本次输出目录：[level03_ekf_check_20261127_v2](../DQ_high-level/outputs/level03_ekf_check_20261127_v2/)。原始数据在各 `level*_seed*/capture.npz`，主评估在 `relative_evaluation/`，回放检查在 `world_diagnostic/`；根目录的 `metrics.csv`、`seed_metrics.csv`、`paired_episode_comparisons.csv` 和 `summary.json` 保存汇总。

![预测对比图](../DQ_high-level/outputs/level03_ekf_check_20261127_v2/comparison.png)

## 回归验证

共 100 项通过，30 项跳过，无失败。包括既有 EKF、观测接入、Level03 运动和训练入口的 82 项通过，以及本次采集/评分/汇总的 18 项通过。跳过项是 28 个 CUDA 专用回归（CPU 套件主动隐藏 GPU）与 2 个缺少外部 Paparazzo 实现的测试；真实在线仿真及生产 EKF 回放在 GPU 0 上实际执行。

四组生产回放全部通过在线一致性检查：最大位置差约 3.08×10⁻⁶ m，最大旋转差约 4.84×10⁻⁵°，低于预设 10⁻⁵ m / 0.001° 容差。四组采集均完成全部 500 步，无采集异常。

新增/扩展评估测试（在 `DQ_high-level`）：

```bash
PYTHONPATH=.:tests python -m unittest -v \
  test_online_ekf_capture test_ekf_motion_evaluation \
  test_ekf_world_diagnostic test_summarize_online_ekf
```

既有 CPU 回归命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=.:tests python -m unittest -v \
  test_future_pose_observations test_object_pose_prediction \
  test_pose_prediction_tuning test_level03_motion test_sgs_motion_heading \
  test_sgs_motion_ranges test_teacher_entry
```
