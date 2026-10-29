
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
   On the **727 server**, use the dedicated launcher to select physical GPU 2:
   ```bash
   bash DQ_high-level/train_teacher_sgs_ekf_727.sh
   ```
   It sets `CUDA_VISIBLE_DEVICES=2` before launching Python and uses logical
   `cuda:0` for both the policy and simulator. Other settings and options are
   inherited from the shared launcher, including `--dry-run`.
   The launcher enables headless training and W&B, using the validated v10
   SGS bank, 16384 environments, 120000 training steps, seed 43, and EKF
   horizons of 0.60/0.90 seconds (history 5, Q scale 1.0, R scale 0.01).
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
