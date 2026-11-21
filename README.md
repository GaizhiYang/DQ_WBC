
**[Project Page](https://kolakivy.github.io/DQ/)** | **[arXiv](https://arxiv.org/abs/2508.08328)**

**[Qiwei Liang](https://kolakivy.github.io/)**,Boyang Cai,Rongyi He, Hui Li, Tao Teng, [Haihan Duan](https://duanhaihan.github.io/), Changxin Huang, **[Runhao Zeng](https://zengrunhao.com/)***

**Association for the Advancement of Artificial Intelligence(AAAI)  2026**

 - **We design the first benchmark for evaling the whole-body coordination of dynamic object grasping with legged manipulators.**

<p align="center">
<img src="./dq_bench.png" width="100%"/>
</p>


 - **Meanwhile, to improve the performance of grasping dynamic objects, we designed DQ-Net.**

<p align="center">
<img src="./dqnet_architecture.png" width="100%"/>
</p>


**Clarification**: Our project is highly based on work--[VBC](https://github.com/Ericonaldo/visual_wholebody)

# 💻 Installation
## Set up the environment
```bash
conda create -n dqwbc python=3.8 # isaacgym requires python <=3.8
conda activate dqwbc

git clone https://github.com/YoungYNG/DQ_WBC.git

cd DQ_WBC

pip install torch torchvision torchaudio

cd third_party/isaacgym/python && pip install -e .

cd ../..
cd rsl_rl && pip install -e .

cd ..
cd skrl && pip install -e .

cd ../..
cd DQ_low-level && pip install -e .

pip install numpy pydelatin tqdm imageio-ffmpeg opencv-python wandb scipy termcolor
```

# 🛠️ Usage
### Low-level training
This part you can totally refer to [VBC's Low-level introduction](https://github.com/Ericonaldo/visual_wholebody/tree/main/low-level), we have provided our low-level model in our github repo: [low-level model of the entity B1Z1](https://github.com/YoungYNG/DQ_WBC/tree/main/DQ_high-level/data/low_model)

**Note**:We made changes to the **low-level** part of the VBC work mainly by **expanding the value ranges** of `delta_orn_r`, `delta_orn_p`, and `delta_orn_y` in `low-level/legged_gym/envs/manip_loco/b1z1_config.py`

### High-level training and eval
1. Train DQ_teacher:
```bash
   python train_multistate_DQ_teacher.py --rl_device cuda:0 --sim_device cuda:0 --timesteps 120000  --task B1Z1PickMulti --experiment_dir DQ_teacher/b1-pick-multi-DQteacher_01 --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --headless
   ```
   For a single-object experiment, add `--object_name sugar_box` (or another
   name under `env.asset.asset_multi` in `DQ_high-level/data/cfg/DQ_teacher.yaml`).
   Use a separate `--experiment_dir`, for example `DQ_teacher/sugar_box_test`.
   This selects the matching asset, feature, initial pose, and precomputed grasp
   predictions; each object still has 30 candidate grasps. Omitting the option
   uses all configured objects. Add `--num_envs 512` for a smaller parallel batch
   when testing; fewer environments also means fewer training samples per step.
   Run these commands from `DQ_high-level`. Use the same `--object_name` when
   playing the resulting checkpoint. Training saves the selected object in the
   experiment config, so `--resume` retains it.

   To train the sugar-box teacher with **both SGS and EKF**, run from the
   repository root after activating `dqwbc`:
   ```bash
   bash DQ_high-level/train_teacher_sgs_ekf.sh
   ```
   参数说明：
你标出的这些参数，控制的是**训练任务、观测内容、SGS/EKF、训练规模和日志**。

先明确“**不加**”的含义：下面说的是**从脚本中删除对应参数**。如果只是运行脚本时没有额外输入它，脚本里的参数仍然生效。具体默认值已核对[参数解析代码](/home/hehui/DQ_WBC/DQ_high-level/utils/config.py:205)和[当前 YAML 配置](/home/hehui/DQ_WBC/DQ_high-level/data/cfg/DQ_teacher.yaml:1)。

**任务和训练规模：**

| 参数 | 当前作用 | 从脚本中删除会怎样 |
|---|---|---|
| `--cfg_file data/cfg/DQ_teacher.yaml` | 选择环境、奖励、机器人等配置 | 新训练默认仍读取这个 YAML，基本相同 |
| `--task B1Z1PickMulti` | 选择抓取环境类 | 默认是空字符串，无法正常创建环境，会报错 |
| `--object_name sugar_box` | 只训练糖盒 | 普通模式会使用 YAML 中全部物体；当前 SGS 模式要求明确指定糖盒，因此会报错 |
| `--rl_device cuda:0` | 策略网络、PPO 计算使用的设备 | 默认仍为 `cuda:0` |
| `--sim_device cuda:0` | 物理仿真使用的设备 | 默认仍为 `cuda:0`；若你通过环境变量指定了其他设备，删除后就不会再跟随该设置 |
| `--num_envs 8192` | 并行运行 8192 个环境 | 使用当前 YAML 的 **4096** 个环境 |
| `--timesteps 120000` | 总共执行 120000 个高层采样步，也影响训练阶段调度 | 默认变成 **24000** 步，训练和阶段切换都会提前 |

这里的 `timesteps` 不是回合数，也不是 PPO 更新次数。当前每采集 **24 步**更新一次 PPO；8192 个环境每轮提供 `8192 × 24 = 196608` 条转移样本。

**观测和控制：**

| 参数 | 当前作用 | 从脚本中删除会怎样 |
|---|---|---|
| `--roboinfo` | 高层观测包含腿部关节位置、速度等机器人信息 | 主要去掉 12 个腿关节位置和 12 个速度，每帧观测减少 24 维，网络输入随之改变 |
| `--observe_gait_commands` | **底层控制器**额外接收步态相位和时钟，共 5 维 | 底层观测少 5 维；当前预训练底层模型按带这部分输入构建，删除可能直接导致权重维度不匹配 |
| `--small_value_set_zero` | 给底盘速度命令设置死区，帮助停止 | 小幅底盘命令仍会执行，更容易产生缓慢移动或抖动 |
| `--rand_control` | 每个高层动作随机执行 7 或 8 个底层周期，模拟控制周期波动 | 固定执行 YAML 设置的 8 个底层周期 |
| `--no_velocity_obs` | 删除高层观测中的机器人局部线速度 3 维、物体相对平面速度 2 维 | 当前 YAML 默认保留速度，因此每帧会**增加 5 维** |
| `--critic_time_obs` | 只给 Critic 添加 `当前步数 / 回合最大步数`，帮助判断剩余任务时间 | 当前 YAML 默认关闭；Critic 从 215 维回到 214 维，Actor 不受影响 |
| `--actor_history_frames 4` | Actor 使用 `[t-3,t-2,t-1,t]`，即当前帧加前 3 帧 | 当前 YAML 默认 **1 帧**，Actor 从 **856 维回到 214 维** |
| `--critic_reward_state` | 只给 Critic 加 7 维隐藏奖励状态：历史最近距离、接近差距、抬升差距、两个保持计数、是否已抬起、command_reward 当前权重 | 当前 YAML 默认关闭；Actor 不受影响 |
| `--command_reward_schedule ramp` / `constant` | `ramp`：command_reward 在前 20% 训练内线性升到全权重；`constant`：从第一步起全权重 | 默认 `step`：训练到一半时才突然打开（旧行为）；TSIL 只支持 `step` |
| `--approach_shaping potential` | 接近和抬升奖励改为势函数塑形 `γΦ(s′) − Φ(s)`，终止时 Φ = 0 | 默认 `record`：只奖励刷新历史最近距离 / 最大高度（旧行为） |
| `--no_value_clip`、`--gae_lambda 0.9` | PPO 价值损失不裁剪；GAE λ 改为 0.9（消融用） | 默认裁剪价值损失，λ = 0.95 |
| `--action_scaling normalized` | 网络输出先截断到 [-1, 1]，再乘物理上限：末端 ±0.02 m、±0.06 rad，底座 ±0.45 m/s、±0.6 rad/s；策略 std 初始为 0.5，上限为 1 | 默认 `legacy`，即旧行为：直接在物理上限截断原始输出。两种模式的 checkpoint 动作语义不同，要用不同的实验目录 |

`small_value_set_zero` 的实际规则是：当**前进速度绝对值 ≤ 0.15 m/s，并且偏航角速度绝对值 ≤ 0.35 rad/s**时，把底盘命令整体置零；它不会把所有较小的机械臂动作清零。

当前底层周期为 `0.02 s`，所以开启 `rand_control` 后，相邻高层观测通常间隔 **0.14 或 0.16 秒**。Actor 的“前 3 帧”因此不是固定时间跨度。

**SGS 和 EKF：**

| 参数 | 当前作用 | 从脚本中删除会怎样 |
|---|---|---|
| `--sgs_mode sgs` | 从状态库中自适应采样训练起点，依据各配置的成功反馈调整采样概率 | 当前 YAML 默认 `disabled`，关闭 SGS，使用普通任务重置 |
| `--sgs_bank "$bank_path"` | 指定经过验证的 SGS 状态库；目前默认 `sugar_box_v12.pt` | 如果仍保留 SGS，YAML 的库路径为空，会报错 |
| `--success_mode hold` | 使用抓住、抬高并持续保持的成功规则 | 当前默认仍为 `hold`，行为相同；`legacy` 目前也是同一规则的兼容名称 |
| `--future_pose_obs` | 将 EKF 预测的未来物体位姿加入高层观测 | 当前 YAML 默认关闭，Actor/Critic 不再接收未来位姿 |
| `--future_pose_horizons 0.60 0.90` | 预测距离当前 **0.60 秒、0.90 秒**后的物体位姿 | 若预测仍开启，改用 YAML 的 **0.30 秒、0.60 秒** |
| `--ekf_history 5` | EKF 使用最近 5 次物体位姿测量进行估计 | YAML 默认也是 5，行为相同 |
| `--ekf_q_scale 1.0` | 缩放 EKF 的过程噪声协方差 Q，表达对运动模型误差的假设 | YAML 默认也是 1.0 |
| `--ekf_r_scale 0.01` | 缩放 EKF 的测量噪声协方差 R，表达对位姿测量噪声的假设 | YAML 默认也是 0.01 |

**`ekf_history 5` 和 `actor_history_frames 4` 是两个独立缓存。** 前者用于计算未来物体位姿；后者把完整的历史观测交给 Actor。

一般来说，Q 增大表示允许运动模型有更大误差，R 增大表示认为测量更不可靠。这里的 `0.01` 是协方差的**缩放系数**，不代表“位置误差为 1 cm”。

**输出和运行方式：**

| 参数 | 当前作用 | 从脚本中删除会怎样 |
|---|---|---|
| `--experiment_dir …` | 保存模型、配置和日志的根目录 | 默认保存到 `experiments` |
| `--wandb_project …` | W&B 项目名 | 默认 `isaacgym`，与脚本当前默认相同 |
| `--wandb_name "$run_name"` | 本次运行名称，也用于本地实验子目录命名 | 默认固定为 `isaacgym`，不再自动带时间戳 |
| `--seed 43` | 设置网络初始化、采样等随机种子 | 默认仍为 43 |
| `--headless` | 不打开仿真可视化窗口 | 尝试打开窗口，需要图形环境，并增加显示开销 |

W&B 是否启用由后面的 `--wandb` 决定，仅设置项目名和运行名不会启用上传；该脚本默认会加 `--wandb`，传 `--no-wandb` 才关闭。

如果你只是想比较“有历史”和“无历史”，保持其他参数一致，直接运行：

```bash
# 当前帧 + 前3帧：856维
bash DQ_high-level/train_teacher_sgs_ekf.sh

# 仅当前帧：214维
bash DQ_high-level/train_teacher_sgs_ekf.sh --actor_history_frames 1
```

   
   
   On the **727 server**, use the dedicated launcher to select physical GPU 2:
   ```bash
   bash DQ_high-level/train_teacher_sgs_ekf_727.sh
   ```
   It sets `CUDA_VISIBLE_DEVICES=2` before launching Python and uses logical
   `cuda:0` for both the policy and simulator. Other settings and options are
   inherited from the shared launcher, including `--dry-run`.
   The launcher enables headless training and W&B, using the validated v12
   SGS bank, 8192 environments, 120000 training steps, seed 43, and EKF
   horizons of 0.60/0.90 seconds (history 5, Q scale 1.0, R scale 0.01).
   It also passes `--critic_time_obs`: the critic (not the actor) receives the
   elapsed fraction of the 100-step limit, because a timeout ends the episode
   as a true terminal. Append `--no_critic_time_obs` for the previous baseline;
   checkpoints with and without this channel have different input sizes.
   Runs get a timestamped name under
   `DQ_high-level/DQ_teacher/b1-pick-sugar-box-sgs-ekf/`.
   Use `--dry-run` to inspect the command without starting training, or
   `--help` to see device, bank, run-name, and output-directory overrides:
   ```bash
   bash DQ_high-level/train_teacher_sgs_ekf.sh --dry-run
   NUM_ENVS=64 RUN_NAME=sgs_ekf_smoke bash DQ_high-level/train_teacher_sgs_ekf.sh --no-wandb --stop_after 48
   ```
   `--stop_after` limits this invocation while retaining the 120000-step
   reward schedule. SGS runs start from scratch and do not support resuming.

```bash
cd /home/hehui/DQ_WBC/DQ_high-level

python train_multistate_DQ_teacher.py \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --timesteps 120000 \
  --task B1Z1PickMulti \
  --object_name sugar_box \
  --experiment_dir DQ_teacher/b1-pick-sugar-box \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero \
  --rand_control \
  --headless --wandb --wandb_name xxxx
```


2. Play DQ_teacher:
   ```bash
   python play_multistate_DQ_teacher.py --task B1Z1PickMulti --checkpoint "your_teacher_checkpoint_path" --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --rl_device "cuda:0" --sim_device "cuda:0"  --headless
   ```
3. Train DQ_stu:
   ```bash
python train_multi_bc_deter_DQ_stu.py \
  --task B1Z1PickMulti \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --timesteps 240000 \
  --experiment_dir DQ_stu/b1-pick-multi-DQstu_02 \
  --teacher_ckpt_path DQ_teacher/b1-pick-multi-DQteacher_01/isaacgym/checkpoints/agent_120000.pt \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero \
  --rand_control \
  --headless
   ```
   For single-object distillation, add `--object_name sugar_box`, use a separate
   output such as `--experiment_dir DQ_stu/b1-pick-sugar-box`, and point
   `--teacher_ckpt_path` to
   `DQ_teacher/b1-pick-sugar-box/isaacgym/checkpoints/agent_120000.pt`.
   The student selects the object's initial pose and grasp predictions using
   its original `dict_idx`; each object still has 30 candidate grasps.
   `--num_envs` overrides the default 200 student environments. Student runs
   save the selected objects and environment count in their experiment YAML,
   so `--resume` retains the single-object selection.

   The DQ teacher and student configs use the same terrain, table motion,
   start distance (0.85–1.1 m), lifting threshold (0.2 m), and hold time
   (15 steps). `cameraVisibilityReset: False` keeps camera-only termination
   rules from changing the student's episode distribution. The student
   enables cameras for its visual observations; the teacher uses privileged
   object features and grasp poses.


```bash
python train_multi_bc_deter_DQ_stu.py \
  --task B1Z1PickMulti \
  --object_name sugar_box \
  --rl_device cuda:0 \
  --sim_device cuda:0 \
  --timesteps 240000 \
  --experiment_dir DQ_stu/b1-pick-sugar-box \
  --teacher_ckpt_path DQ_teacher/b1-pick-sugar-box/isaacgym/checkpoints/agent_120000.pt \
  --roboinfo \
  --observe_gait_commands \
  --small_value_set_zero \
  --rand_control \
  --headless
```
4. Play DQ_stu:
   ```bash
   python play_multi_bc_deter_DQ_stu.py --task B1Z1PickMulti --checkpoint your_stu_checkpoint_path --roboinfo --observe_gait_commands --small_value_set_zero --rand_control --rl_device cuda:0 --sim_device cuda:0   --headless
   ```

### Change the DQ_Bench Level:
You can easily make this by just modifying the config file:`DQ_WBC/DQ_high-level/data/cfg/DQ_stu.yaml`,`DQ_WBC/DQ_high-level/data/cfg/DQ_teacher.yaml`


# 📝 Citation

[](https://github.com/YanjieZe/3D-Diffusion-Policy/tree/master#-citation)

If you find our work useful, please consider citing:

```
@article{liang2025whole,
  title={Whole-Body Coordination for Dynamic Object Grasping with Legged Manipulators},
  author={Liang, Qiwei and Cai, Boyang and He, Rongyi and Li, Hui and Teng, Tao and Duan, Haihan and Huang, Changxin and Zeng, Runhao},
  journal={arXiv preprint arXiv:2508.08328},
  year={2025}
}
```
