# SGS 还能结合什么：动态抓取教师的补充论文与实验方案

2026-11-22 · 面向 B1Z1 移动糖盒抓取 · 训练入口 `DQ_high-level/train_teacher_sgs_ekf.sh`

本轮补充 **12 篇论文、8 组组合方向**。其中 CurrOT 的 2024 年论文是 2022 年工作的扩展，不能算作两次独立方法验证。本文区分原论文结论与面向本项目的迁移设想，组合收益尚未经过本项目训练验证。

**优先建议评估 SGS＋接触过渡状态覆盖、SGS＋连续起点课程和 SGS＋目标任务相关性。** 三者分别解决“缺少哪些练习状态”“怎样从容易起点走向完整任务”“哪些辅助练习有助于最终目标”。当前还没有 v12 教师的失败分布，应先用基线回放定位瓶颈。若主要卡在接触到闭爪，首选接触覆盖；若主要卡在接近，先做连续起点课程。不必一次全部加入。

前两轮已经讨论 PLR、ALP-GMM、RFCL、ACCEL、DemoStart、SIL/TSIL、SAPG、VDS，以及夹爪动作、残差控制、示范等。本文只简要连接这些方案，主要展开新增机制。

- [前一轮 SGS 拓展调研](SGS_extensions_literature_review.md)
- [前一轮 RL manipulation 调研](RL_manipulation_teacher_research.md)

## 1. 从当前实现看，新增方法应该补在哪里

当前 SGS 在固定配置库中，根据每个配置最近的持稳成功率决定 reset 概率。v12 有 1536 个配置，full_task、pregrasp、near_goal 各 512 个。full_task 从约 2 m 开始，pregrasp 的底盘距离约 0.8–0.95 m，near_goal 已经握住并抬高糖盒。三类配置提供不同起点，但它们没有显式保证覆盖“首次单指接触→双指接触→刚离桌”等物理转换。

因此需要分别回答三个问题。第一，库里是否有策略需要学习的状态。第二，训练这些状态是否帮助原始完整任务。第三，成功率相近的状态，其难度是否来自不同动力学条件或不同阶段。

当前入口使用 PPO、8192 个并行环境、hold 成功判据和 EKF 未来位姿，默认实际 Actor 为单帧。PPO rollout 为 24 个高层步，回合最多 100 步。后续实验需要显式固定这些设置，尤其不要让动作结构、观测历史与采样方法同时变化。[代码入口](../DQ_high-level/train_teacher_sgs_ekf.sh) · [状态库元数据](../DQ_high-level/data/sgs/sugar_box_v12.json) · [SGS 实现](../DQ_high-level/utils/sgs.py)

| 新增方向 | 主要论文 | 与 SGS 的组合位置 | 开发量 | 建议触发条件 |
|---|---|---|---|---|
| 接触模式覆盖 | CSRL，CoRL 2026 | 生成新的可行 reset | 中 | 预抓能到达，接触到闭爪经常失败 |
| 连续起点课程 | CurrOT、GRADIENT | 决定新起点向哪里扩展 | 中到高 | 近起点学会，完整起步长期停滞 |
| 目标相关性选样 | ProCuRL-Target；GradAlign 为进阶参考 | 改变 SGS 外层预算 | 中；梯度版高 | 辅助起点改善没有转化为完整任务改善 |
| 规划生成可达状态 | Sampling-based Exploration，RSS 2023 | 用动力学探索建立 bank | 高 | 独立 IK 起点与完整任务经过的状态差异明显 |
| 动力学条件课程 | Active Domain Randomization | 对 reset 与动力学联合分配练习 | 中到高 | 同一配置在不同摩擦或增益下表现差异大 |
| 辅助技能调度 | SAC-X，ICML 2018 | 同时选择从哪里开始、练哪种行为 | 高 | 回合内部探索不足，缺少连续有效动作 |
| 目标难度与 hindsight | HTRPO，IJCAI 2021 | 扩展目标条件及经验用途 | 高 | 大量失败已完成较低高度或较短持稳目标 |
| 持稳扰动课程 | Supervision via Competition；RARL | 练习抗扰动抓持 | 中；学习对手版高 | 名义成功很多，但轻微扰动即滑落 |

这里的开发量是结合本项目接口做出的估计。原论文中的操作实验、通用控制实验与跨领域方法，下面会分别说明。

## 2. SGS＋CSRL：补齐接触转换附近的状态

**论文依据。** *Combining Constrained Sampling and Reinforcement Learning for Manipulation*，Toussaint 等，CoRL 2026。CSRL 先选择接触模式，再求满足碰撞、接触、摩擦锥和准静态平衡约束的状态。其插值重置会投影到可行状态集合。论文在六类仿真操作中验证，包含 Panda 与 Allegro，默认学习器为 TD7＋HER。可迁移的部分是物理约束下的状态生成。[R1，§3.2–3.3、§4](https://arxiv.org/pdf/2602.08557)

**项目方案。** 先利用现有建库器生成一小批具有以下接触条件的状态，避免第一步就实现通用非线性约束求解器。

| 接触条件 | 具体练习 | 候选检查重点 |
|---|---|---|
| 桌面支撑、夹爪未触物 | 最后几厘米对准 | 可达、不过度穿透、目标仍在可抓范围 |
| 桌面支撑、单指接触 | 擦碰后的对准与闭合 | 物体有支撑，接触不会立即把它推出工作空间 |
| 双指接触、尚未离桌 | 形成有效握持并抬升 | 两指间位置、实际开度、桌面碰撞与相对滑移 |
| 已离桌、存在小幅滑移 | 抬升到持稳 | 滑移程度可恢复，控制目标与物体速度相容 |

这些标签描述接触状态，可以与现有 reset family 同时保存。不能用 `near_goal` 代替逐帧接触标签，也不能只凭手指净力认定手指接触了糖盒，因为力也可能来自桌面。

第一轮保持 bank 容量和 full_task 覆盖，替换少量非 full_task 槽位。SGS 仍按真正的最终 hold 结果选样。这与前一轮“动态 bank”的新增区别，是为候选状态提供明确的接触覆盖目标。

**最小实验。** 同容量、同生成预算比较原 bank、普通几何扰动 bank、接触条件覆盖 bank，三组都用同一 SGS。记录完整任务持稳率、首次闭合后的抬起率、离桌后的滑脱率，并比较新状态与完整起步真实经过状态的覆盖关系。

**边界。** CSRL 的准静态约束只为本项目提供几何候选。移动糖盒的速度、底盘运动、控制目标及 EKF 历史需要继续做动态恢复检查。悬空单指接触不应被当作稳定握持。新增模式是否有价值，要由完整任务成绩决定。

## 3. SGS＋CurrOT／GRADIENT：填补预抓与完整起步之间的空档

**论文依据。** CurrOT 的 *Curriculum Reinforcement Learning via Constrained Optimal Transport*，ICML 2022，以及扩展论文 *On the Benefit of Optimal Transport for Curriculum Reinforcement Learning*，TPAMI 2024，把课程向目标分布的推进写成最优传输问题。扩展论文强调对课程支持集中的任务施加性能约束，避免用大量容易任务掩盖难任务。其主要实验为导航和参数化行走。[R2](https://proceedings.mlr.press/v162/klink22a/klink22a.pdf) · [R3，§IV–V](https://arxiv.org/html/2309.14091v2)

GRADIENT 的 *Curriculum Reinforcement Learning using Optimal Transport via Gradual Domain Adaptation*，NeurIPS 2022，用 Wasserstein 重心构造源分布到目标分布的路径，达到当前阶段回报门槛后再推进。它包含使用 SAC 的 FetchPush 实验，改变的是二维目标位置分布。[R4，§4.3、§5.3](https://proceedings.neurips.cc/paper_files/paper/2022/file/4556f5398bd2c61bd7500e306b4e560a-Paper-Conference.pdf)

**项目方案。** 如果 near_goal 和 pregrasp 已学得较好，而从 2 m 起步仍难以完成接近，可让外层课程生成预抓与完整起点之间的桥接状态，内层继续使用 SGS。

一个可审查的容量方案是保留 512 个 full_task 和 512 个 near_goal，将原 512 个 pregrasp 槽位作为逐步替换的桥接池。这是工程起点，实际比例需要消融。先只推进未持物的接近分支，保留持物状态的独立接触模式。

距离度量可用经量纲归一化的底盘相对距离、相对朝向、目标速度和运动方向。候选生成后经 IK、碰撞及短时恢复验证。直接线性插值整条机器人状态，尤其跨越未抓住和已持物两种模式，可能产生无效状态。

**最小实验。** 先比较固定 bank＋SGS 与手工距离递进＋SGS。只有桥接起点已经显示收益，再比较 GRADIENT 与 CurrOT 生成的课程，判断是否需要更复杂的分布优化。生成和筛选的仿真步都计入预算。

**边界。** 固定 bank 上只重加权无法创造缺失的中间状态。SGS 再加权也会改变最优传输求出的分布，组合不能直接继承原方法的理论性质。应记录最终实际采样分布，并与原 OT 分布、固定比例混合做比较。GRADIENT 与下节 GradAlign 是不同方法。

## 4. SGS＋目标相关性：练习是否真正帮助完整任务

**论文依据。** *Proximal Curriculum with Task Correlations for Deep Reinforcement Learning*，Tzannetos 等，IJCAI 2024，提出 ProCuRL-Target。它从目标任务回报的一阶改善出发讨论梯度对齐，再构造学习潜力与任务相关性联合的选样机制。实验使用 PPO，涵盖 PointMass、Sparse Goal Reaching、MiniGrid、BipedalWalker，尚无机械臂操作验证。实验相关性采用任务参数距离核。[R5，§3、§4.2](https://ijcai.org/proceedings/2024/0556.pdf)

**项目方案。** 将当前 SGS 概率记为 $q_{\mathrm{SGS}}(i)$，定义候选配置对完整任务的相关性分数 $r_i$。一个简化迁移版为

$$
r_i=\frac{1}{|V|}\sum_{j\in V} z_j K(c_i,c_j),
\qquad
q_{\mathrm{rel}}(i)=\frac{q_{\mathrm{SGS}}(i)(\epsilon+r_i)}{\sum_k q_{\mathrm{SGS}}(k)(\epsilon+r_k)}.
$$

$V$ 是用于课程决策的完整任务验证配置，独立于最终测试集；$z_j$ 可先取验证成功率的 $p_j(1-p_j)$，$K$ 使用非负的任务相关性核。再将 $q_{\mathrm{rel}}$ 与一定比例原 SGS 混合，防止早期目标评分不足导致覆盖丢失。这是借鉴版，原论文对任务对进行 softmax 采样。

初版相关性应优先由目标速度、方向、姿态等共同任务条件决定。直接计算完整 reset 向量的欧氏距离，会让底盘距物体远近主导分数，可能排除本来有益的近目标练习。可先按动态条件分组，组内仍让 SGS 分配三类起点。

如果完整任务全部失败，基于 $p(1-p)$ 的目标学习潜力没有有效区分信号，应回退原 SGS。手工距离核只是迁移能力的代理，需要用后续完整任务提升验证。

**最小实验。** 原 SGS、SGS＋固定 full_task 比例、SGS＋相关性三个对照。第二组检验收益是否仅来自多练完整任务，第三组才检验选对辅助练习的价值。报告各动态条件下的完整持稳率及获得的训练步数。

**进阶参考。** *GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning*，Yang 等，2026，以目标验证集与候选训练数据的策略梯度余弦相似度选样。它在 LLM 的 GRPO 训练中验证，属于跨领域借鉴。[R6，§4–5](https://arxiv.org/html/2602.21492v2)

本项目可先只记录三类 reset 的 actor 更新方向，与完整任务成功目标梯度的对齐度，检验其是否预测随后成功率变化。成功样本太少时不宜启用。密集 shaping 回报的梯度不能直接解释为最终成功率梯度，Adam、PPO clipping 和多轮更新也会影响预测。通过离线诊断后，再研究组级采样；不建议一开始对 1536 个配置逐一估梯度。

## 5. SGS＋采样规划：让 bank 包含有动力学路径连接的状态

**论文依据。** *Sampling-based Exploration for Reinforcement Learning of Dexterous Manipulation*，Khandate 等，RSS 2023，用 RRT 探索产生状态，再把任务相关树路径上的节点用于 RL 重置。G-RRT 通过真实仿真动力学扩展，M-RRT 使用接触运动学近似。原实现接入 Isaac Gym 非对称 Actor-Critic PPO，在多指手重定向中验证并有真机实验。[R7，§III–IV](https://arxiv.org/pdf/2303.03486)

**项目方案。** 从 full_task 或已验证 pregrasp 出发，用现有九维高层动作做有限分支探索。用手物相对位姿、相对运动、实际开度与接触条件描述覆盖，把增加有效覆盖且能恢复续跑的节点加入 bank，随后由 SGS 选择练习频率。

这与 CSRL 的区别是，CSRL 帮助寻找满足状态约束的配置，动力学树还提供从已有状态到达候选的路径。从独立 pregrasp 根节点建树，只能说明候选相对于这个根节点可达。要论证完整任务可达，需要从 full_task 连续扩展，或另有到达该根节点的有效记录。对移动糖盒，优先借鉴 G-RRT。多指手 M-RRT 中的接触假设不能直接移植到平行夹爪。

**最小实验。** 在相同候选生成预算下，比较当前策略 rollout 收集、随机动作 rollout 收集、带覆盖目标的树扩展收集。再用同一批节点比较均匀采样与 SGS，分离状态覆盖和选样机制的贡献。

**边界。** 可达不等于可恢复。节点必须保存相容的目标运动相位、控制目标及必要历史，或通过明确的冷恢复验证。动目标接近态不需要静止稳定，不能照搬固定动作等待不掉落的统一检查。树构建和验证成本必须计入总训练成本。

论文中的同名基线 **SGS 指 Stable Grasp Sampler**，与本项目已引入的 SGS 不是同一个方法，不能据此写成“RRT 已证明优于当前 SGS”。

## 6. SGS＋动力学条件课程：拆开几何难度与物理难度

**论文依据。** *Active Domain Randomization*，Mehta 等，CoRL 2019，论文集发表于 2020。它利用随机化环境与参考环境中轨迹的可区分性，学习环境参数采样策略，并用 SVPG 保持多样性。实验包含机械臂推物、到达及真机迁移，学习器使用 DDPG。[R8，§3–4、附录 A](https://proceedings.mlr.press/v100/mehta20a/mehta20a.pdf)

**项目动机。** 当前环境创建时已经随机化摩擦与夹爪增益。SGS 对同一配置汇总成功结果，可能出现“在某种动力学下几乎总成功，在另一种下几乎总失败，合起来约 50%”的情况。这个现象需先测量，不能直接认定它是现有瓶颈。[环境创建](../DQ_high-level/envs/b1z1_base.py)

**第一步只做条件统计。** 将已记录的实际摩擦、夹爪增益粗分组，在固定策略与同一 reset 集上配对评估，估计 $p(\mathrm{hold}\mid i,d)$。稀疏组可向配置整体均值收缩，避免将随机波动误认为动力学差异。

如果确有结构差异，先保留原动力学边际分布，在各环境组内按条件成功率运行 SGS，形成

$$
q(i,d)=p_{\mathrm{fixed}}(d)\,q_{\mathrm{SGS}}(i\mid d).
$$

这一步借鉴条件化课程思想，不是复现 ADR。需要主动改变 $q(d)$ 时，再考虑原 ADR 的判别器和参数提出器，并保留目标随机化分布的覆盖。

**工程边界。** 当前不少物理参数在创建环境时设置。每组并行环境持续执行相同步数时，仅改变 reset 抽样无法任意改变动力学的步数边际分布。完整 ADR 需要明确的参数更新路径或环境预算分配，还要测量吞吐影响。

**最小实验。** 原 SGS 与固定动力学边际下的条件 SGS 对照，保持同样的动力学范围、环境数和交互量。报告目标分布平均成功率与预先定义的困难组成功率。判别器能区分轨迹，并不保证差异可通过训练消除；极弱夹持力等不可解条件需要独立识别。

## 7. SGS＋SAC-X 思想：选择练习行为

**论文依据。** *Learning by Playing – Solving Sparse Reward Tasks from Scratch*，Riedmiller 等，ICML 2018，提出 SAC-X。每个辅助奖励对应一个可执行 intention，高层调度器以外部任务回报为目标选择这些行为。它以 off-policy 方法共享经验，价值学习使用 Retrace，在仿真堆叠、收纳和真实机械臂操作中验证。[R9，§3、§5](https://proceedings.mlr.press/v80/riedmiller18a/riedmiller18a.pdf)

**项目方案。** SGS 选择起点，行为条件 $z$ 决定本轮练习对准、形成接触、抬升或最终持稳。与前一轮“预测交接状态是否有利”相比，这里增加的是实际执行辅助目标的策略。

保留 PPO 的第一版可以使用条件 Actor 与 Critic，单个回合内固定 $z$。全部动作概率、GAE 和价值目标都按实际执行的 $z$ 计算。SGS 的统计按 $(i,z)$ 区分，不能把辅助抬升成功写入最终 hold 的成功历史。已在 reset 时完成辅助目标的组合要排除或重新定义，否则 near_goal 会制造无意义的辅助成功。

**最小实验。** 原 PPO、条件 PPO＋固定行为比例、条件 PPO＋依据完整任务收益调整行为比例。首先检验辅助目标是否增加真正的完整成功。若测试时只执行最终 hold intention，必须单独评估它；若需要调度器才能完成任务，则将调度器纳入最终控制系统与学生蒸馏范围。

**边界。** 这个简化版不等同于原 SAC-X。原方法在回合中切换行为并跨行为复用经验，不能直接把这些数据当成当前单策略 PPO rollout。回合内调度需要处理调度状态、切换边界和行为概率一致性，开发成本较高。

## 8. SGS＋目标难度／hindsight：利用没有完全持稳的回合

**论文依据。** *Hindsight Trust Region Policy Optimization*，Zhang 等，IJCAI 2021，将 hindsight 引入 TRPO，对目标改变后的轨迹分布及信赖域约束做重要性采样修正，并筛选更接近最终任务的 hindsight goals。实验包含 Fetch Reach、Push、Slide、PickAndPlace。[R10，§3–5](https://ijcai.org/proceedings/2021/459)

**项目方案。** 在现有最终目标保持不变的前提下，定义辅助目标 $g=(h,d)$，其中 $h$ 是离桌高度门槛，$d$ 是连续持稳步数，最终目标为约 $(0.20\mathrm{m},10)$。先统计完整失败轨迹中，有多少真实达到更低高度或更短持续时间的目标。

第一阶段先做 goal-conditioned PPO 课程，每次采样起点及目标，维护 $p(\mathrm{success}\mid i,g)$，保留最终目标练习比例。较短持稳或较低抬升只是辅助任务，最终测试始终使用原判据。

第二阶段才考虑 hindsight。已经采到的失败轨迹可能对另一个目标有用，但当前 Actor 没有目标条件，不能仅改 reward 后塞回 PPO。需要完整的目标条件策略与价值函数，以及经过推导的跨目标修正，或将原论文算法作为独立研究分支。

**最小实验。** 原 SGS、SGS＋目标难度课程、同样目标课程＋校正后的 hindsight。用前两组分离“任务变容易”本身的收益，再判断经验重用是否额外有效。

**边界。** 连续 hold 是轨迹属性，重标注时必须重新计算整段连续计数与目标终止，Critic 需要必要进度和时间信息。目标速度与运动轨迹属于动力学条件，不能把盒子实际运动重标成另一条反事实轨迹。未来观测不能作为当时 Actor 的输入。HTRPO 的结论也不直接提供 PPO 组合保证。

## 9. SGS＋持稳扰动课程：把名义成功变成抗扰动能力

**论文依据。** *Supervision via Competition: Robot Adversaries for Learning Tasks*，Pinto 等，ICRA 2017，用摇动或另一只手争抢来检验并改进真实抓取。它是单步重复博弈式的自监督抓取系统。*Robust Adversarial Reinforcement Learning*，ICML 2017，则交替训练控制策略与施加扰动力的对手，主要在连续控制任务验证。[R11](https://arxiv.org/pdf/1610.01685) · [R12](https://proceedings.mlr.press/v70/pinto17a/pinto17a.pdf)

**项目方案。** 若名义 hold 很多、轻微扰动后却常掉落，可以在首次离桌后的固定阶段施加有限幅度、方向与时长可控的扰动，让 SGS 同时考虑起点与扰动档位。先做简单的随机脉冲和按能力提高强度的课程，再判断是否需要训练对手。

扰动以外力或明确的环境扰动实施时，PPO 应保存自己实际采样的动作，扰动作为环境变化记录。假如通过覆盖教师动作来制造摇动，就需要另行处理行为分布，不能当作教师原动作直接更新。

**最小实验。** 无额外扰动、固定随机扰动、强度课程三个对照。先标定使成功率有区分而非全部归零的范围，分别报告原始无扰动 hold 和独立扰动测试成绩。课程扰动条件与原条件分开保存成功统计。

**边界。** 当前成功会在连续持稳达标后锁存。如果扰动发生在成功锁存后，旧成功标志无法评价扰动后的掉落。应建立独立的抗扰动判据或调整实验分支的评估时序，同时保留可比较的原始指标。这条路线在已有一定抓取能力后更有意义。

## 10. 如何组合成一条清楚的研究主线

### 方案 A：接触过渡状态＋SGS，作为首选

先通过 CSRL 启发的状态生成，补充仍有桌面支撑的单指／双指接触状态。保持容量、训练算法和 SGS 公式不变，验证完整任务是否改善。若有效，再考虑用 G-RRT 提高这些状态与完整接近轨迹的可达一致性。

可研究的问题是，**物理可行并覆盖关键接触转换的 reset，是否比同数量的普通预抓扰动，更能提升从 2 m 完整起步的动态抓取。**

### 方案 B：连续起点课程＋SGS，适合接近阶段瓶颈

先做手工距离桥接，确认库中起点空档值得填补。随后用 CurrOT 或 GRADIENT 替换手工推进，再判断目标相关性选样是否继续有收益。这样能分别回答“桥接是否有效”“自动课程是否有效”“辅助任务是否选对”。

### 方案 C：条件 SGS＋持稳扰动，适合稳定性瓶颈

先按已有动力学条件拆分失败，再加入有限强度扰动课程。适用于已能抓住、但低摩擦或弱夹爪条件下容易滑落的教师。训练评价必须同时保留原始平均表现和困难条件表现。

SAC-X 与 hindsight 涉及策略条件、价值目标及数据使用方式的改变，建议作为后续独立分支。GradAlign 先做预测性诊断，不宜直接与多个新模块一起训练。

## 11. 代码落点与统一验收

| 需要修改或扩展的职责 | 现有位置 | 对应方向 |
|---|---|---|
| 接触候选、IK、物理恢复与过滤 | `DQ_high-level/utils/sgs_bank_builder.py` | CSRL、连续课程、规划 |
| 配置评分、条件统计、概率混合 | `DQ_high-level/utils/sgs.py` | 目标相关性、条件 SGS |
| 重置与配置 ID、历史状态的一致性 | `DQ_high-level/envs/b1z1_pickmulti.py` | 所有动态 bank 方案 |
| 环境参数记录及扰动施加 | `DQ_high-level/envs/b1z1_base.py` | ADR、抗扰动课程 |
| 条件 Actor/Critic、rollout、PPO 目标 | `DQ_high-level/train_multistate_DQ_teacher.py` | SAC-X 思想、目标条件、梯度诊断 |
| 抬升、连续持稳及成功锁存 | `DQ_high-level/utils/pick_success.py` | hindsight、抗扰动评估 |
| 原始完整任务测试 | `DQ_high-level/eval_sgs_teacher.py` | 全部方向 |

新增状态使用新配置身份与独立成功历史，并保存 bank 版本。正在运行的旧回合，结束时必须归属它实际使用的起点。动态快照需保存或验证控制历史、EKF 预热与目标运动相位的兼容性。

课程决策用的验证集与最终测试集分开。最终都从 2 m、Level02、关闭 SGS 的完整起点评估，匹配测试种子、环境数和回合数。记录 `hold_success_rate`、`ever_lifted_episodes`、`failure_penalized_time`，并按速度、运动方向及相关动力学条件分组。

reset 次数比例与训练步数比例都要报告。长回合会在 PPO batch 中占用更多步数。生成候选、筛选、课程验证、规划和额外物理预热的交互成本计入总预算，墙钟时间单独记录。最终候选至少用三个训练种子复核，列出逐种子结果与配对差值。

**若基线失败主要集中在接触到闭爪，建议第一轮只做原 bank＋SGS、几何扰动 bank＋SGS、接触覆盖 bank＋SGS 三组。** 若接触覆盖组改善完整任务，再扩展到生成方式×均匀／SGS 的对照，检验是否存在组合收益。方法命名或组合本身不足以支持新颖性，需要由具体失败机制和消融结果建立贡献。

## 12. 本轮新增论文索引

| 编号 | 论文 | 年份／出处 | 最值得读的部分 |
|---|---|---|---|
| R1 | [Combining Constrained Sampling and Reinforcement Learning for Manipulation](https://arxiv.org/abs/2602.08557) | CoRL 2026；v3 更新于 2026-10-29 | §3.2 接触约束、§3.3 投影重置 |
| R2 | [Curriculum Reinforcement Learning via Constrained Optimal Transport](https://proceedings.mlr.press/v162/klink22a.html) | ICML 2022 | 课程分布推进与可学性约束 |
| R3 | [On the Benefit of Optimal Transport for Curriculum Reinforcement Learning](https://arxiv.org/html/2309.14091v2) | IEEE TPAMI 2024；R2 扩展 | §IV-B 平均性能约束的问题、§V 近似实现 |
| R4 | [Curriculum Reinforcement Learning using Optimal Transport via Gradual Domain Adaptation](https://proceedings.neurips.cc/paper_files/paper/2022/file/4556f5398bd2c61bd7500e306b4e560a-Paper-Conference.pdf) | NeurIPS 2022 | §4.3 课程推进、§5.3 FetchPush |
| R5 | [Proximal Curriculum with Task Correlations for Deep Reinforcement Learning](https://ijcai.org/proceedings/2024/0556.pdf) | IJCAI 2024 | §3 目标迁移推导、§4.2 相关性核 |
| R6 | [GradAlign: Gradient-Aligned Data Selection for LLM Reinforcement Learning](https://arxiv.org/html/2602.21492v2) | 2026 arXiv，跨领域参考 | §4 梯度对齐、§5 适用条件与开销 |
| R7 | [Sampling-based Exploration for Reinforcement Learning of Dexterous Manipulation](https://arxiv.org/abs/2303.03486) | RSS 2023 | §III RRT 与 reset、§IV PPO 实验 |
| R8 | [Active Domain Randomization](https://proceedings.mlr.press/v100/mehta20a.html) | CoRL 2019，论文集 2020 | §3 轨迹判别与采样、§4 真机迁移 |
| R9 | [Learning by Playing – Solving Sparse Reward Tasks from Scratch](https://proceedings.mlr.press/v80/riedmiller18a.html) | ICML 2018 | intention、调度器与 off-policy 经验共享 |
| R10 | [Hindsight Trust Region Policy Optimization](https://ijcai.org/proceedings/2021/459) | IJCAI 2021 | §3 重要性采样、§4 目标过滤 |
| R11 | [Supervision via Competition: Robot Adversaries for Learning Tasks](https://arxiv.org/abs/1610.01685) | ICRA 2017 | 抓后扰动与稳定性监督 |
| R12 | [Robust Adversarial Reinforcement Learning](https://proceedings.mlr.press/v70/pinto17a.html) | ICML 2017 | 控制器与扰动策略交替训练 |

基础论文为 [A Balanced Data Diet: Addressing the Exploration Bottleneck in Mega-Scale RL for Robot Control](https://arxiv.org/abs/2610.12465)，对应本地 [SGS.pdf](papers/SGS.pdf)。本轮优先阅读顺序为 R1、R5、R2/R3、R7，其余按实际失败类型选择。
