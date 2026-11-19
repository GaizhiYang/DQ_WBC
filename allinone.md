# 教师策略方向 2 / 3 的论文依据

检索日期：2026-11-19。对应 `doc/改进思路详解_合理性分析.md:203–459`：方向 2 是到达时间与拦截可行性模型，方向 3 是剩余拦截时间的势函数奖励。

**有直接论文依据，值得做受控实验；但文档的部分数学解释需要修正。** DDL 支持从轨迹学习时间距离并用于奖励；RL-RRT 支持学习已有控制策略的到达时间；Ng et al. 与 Grześ 支持正确处理边界的势函数塑形。它们没有共同证明当前具体组合能改善 DQ_WBC。

## 检索结果

API 查询：`dynamical distance learning`；年份：1999–2026，覆盖理论基础。六源返回：Semantic Scholar=0、OpenAlex=3、arXiv=3、OpenReview=3、Crossref=3、DBLP=0；12 个独立结果，合并重复 0 个；相关性筛选保留 1 个、剔除 11 个。缺失摘要的记录补查了原页面或检索摘要。词法分数不是论文质量评分。

| # | 论文 | 日期 / 会议 | API 引用数 | 匹配分 | 检索来源 |
|---|---|---|---:|---:|---|
| 1 | [Dynamical Distance Learning for Semi-Supervised and Unsupervised Skill Discovery](https://arxiv.org/abs/1907.08225) — Kristian Hartikainen、Xinyang Geng、Tuomas Haarnoja、Sergey Levine | 预印本 2019-07-18；ICLR 2020 | 101 | 3 | OpenAlex；另核对 arXiv 原文和作者项目页 |

以下为模型知识线索及定向网络检索补充的 8 篇，均已核对一手来源；不是上述 API 命中。未获取引用数的论文不记为零。

| # | 论文 / 作者 | 年份、出处 | 对方案的支持 | 证据边界 |
|---|---|---|---|---|
| 2 | [RL-RRT: Kinodynamic Motion Planning via Learning Reachability Estimators from RL Policies](https://arxiv.org/abs/1907.04799) — Hao-Tien Lewis Chiang 等 | IEEE RA-L 2019 | 学习既定 RL 控制策略的 time-to-reach，再用于规划；与方向 2 的独立时间估计器直接相关 | 特定策略的时间不等于全局最短时间；实机导航不是动态抓取 |
| 3 | [Temporal Difference Models: Model-Free Deep RL for Model-Based Control](https://arxiv.org/abs/1802.09081) — Vitchyr Pong 等 | ICLR 2018 | 用带动作、目标和时间期限条件的价值函数，预测给定时间内能多接近目标 | 输出是有限时域目标距离价值，不直接输出最早到达秒数 |
| 4 | [Optimal Goal-Reaching Reinforcement Learning via Quasimetric Learning](https://proceedings.mlr.press/v202/wang23al.html) — Tongzhou Wang 等 | ICML 2023 | QRL 用非对称拟度量、局部代价约束学习最优目标距离，支持跨轨迹组合 | 主体推导假设确定性动力学、单位转移代价；普通时间回归不继承其保证 |
| 5 | [DHER: Hindsight Experience Replay for Dynamic Goals](https://openreview.net/forum?id=Byf5-30qFX) — Meng Fang 等 | ICLR 2019 | 动态目标下匹配失败轨迹并重组经验，支持方向 2 v2 的重标动机 | 原方法训练 off-policy 策略，没有验证文档的位置/速度倒推标签能学到最短会合时间 |
| 6 | [Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) — Andrew Y. Ng、Daishi Harada、Stuart Russell | ICML 1999 | 向原奖励增加折扣势能差，在相应条件下保留最优策略 | 不要求势函数等于真实最优价值；不保证任意删除、替换旧奖励仍保持策略不变 |
| 7 | [Reward Shaping in Episodic Reinforcement Learning](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf) — Marek Grześ | AAMAS 2017 | 分析回合结束的势能边界，支持在停止计回报的终点将势能归零 | 与文档“失败终止必须取负势能”的一般性说法相冲突 |
| 8 | [Offline Goal-conditioned Reinforcement Learning with Quasimetric Representations](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1c5956164472d6d8123d574aa75cd063-Abstract-Conference.html) — Vivek Myers、Bill Zheng、Benjamin Eysenbach、Sergey Levine | NeurIPS 2025 | 方法称 TMD；处理次优离线数据、随机动力学和轨迹拼接 | successor distance 与折扣到达价值相关，不能直接当秒数代入 `h − τ` |
| 9 | [Spatiotemporal Agility: Time-Constrained Reinforcement Learning for Vision-Guided Dynamic Quadrupedal Interception](https://arxiv.org/abs/2608.06907) — Yidong Zhu 等 | arXiv，2026-08-07；未核实同行评审录用 | 四足接球结合落点/时间预测与时间条件策略，支持显式考虑拦截时间 | 背部篮筐接球不是机械臂夹持；没有验证学习机器人 τ 或 PBRS |

## 全部结果的分析

### 1. 概况

共保留 9 篇相关文献，跨度 1999–2026。DDL、RL-RRT、Ng 1999、Grześ 2017 最直接；TDM 回答动作条件建模问题，QRL/TMD 提供更深入的距离学习路线，DHER 和四足接球论文支持动态目标环节。本轮不是穷尽性系统综述。

### 2. 方法演进

这组文献连接了势函数理论与终止边界（1999、2017）、数据驱动的时间/可达性学习（2018–2020）、拟度量最优距离与随机动力学下拼接（2023、2025）。动态目标重标与四足拦截是另外两个任务关联方向。

会议分布为 ICLR 3 篇、ICML 2 篇、RA-L/AAMAS/NeurIPS 各 1 篇、预印本 1 篇。DDL、TDM 2018、TMD 2025 均包含 Sergey Levine。此小样本不能说明整个领域的热度增长。

### 3. 核心主题及与 DQ_WBC 的对应

**时间距离可以直接学，不必先学完整下一状态。对应 [1]、[2]、[3]。**

DDL §4.1–4.3 从同一轨迹采样 `(s_i,s_j)`，用间隔 `j−i` 监督距离，再将负距离用于策略学习，提供了“轨迹 → 时间距离 → 奖励”的直接先例。原方法的负距离奖励与文档中的势能差奖励并不相同。[DDL 原文](https://arxiv.org/html/1907.08225v3)

DDL 学习策略相关、以到达样本为条件的时间。原文明确提醒，忽略未到达情形、直接按该距离贪心，可能偏好偶尔成功的快速路径。RL-RRT 也明确预测既定点到点策略的时间，并处理失败样本。因此，文档第一版宜称为“经验到达时间估计”，不能仅凭偏下 expectile 称为“机器人最短到达时间”。[RL-RRT §III-B](https://arxiv.org/html/1907.04799v2)

针对动作建模，TDM 使用 `Q(s,a,g,H)`，预测从状态 s 执行动作 a 后，在期限 H 内能够多接近目标 g。只需奖励塑形时可先估计状态到目标距离；需要比较候选动作时可考虑动作条件价值。两者均不以完整下一状态预测为必要前提。注意 H 是输入期限，不是文档的 τ 输出。[作者方法说明](https://bair.berkeley.edu/blog/2018/04/26/tdm/)

**经验时间与最优距离必须区分。对应 [2]、[4]、[8]。**

QRL 的局部代价约束、全局分离目标与三角不等式共同产生最短路径结构；仅换损失或拟度量网络不等于实现 QRL。其确定性动力学、单位代价假设不能直接套到当前随机摩擦、质量与步长的任务。[QRL §3](https://arxiv.org/html/2304.01203v2)

TMD 2025 处理随机动力学和次优数据，但距离处于折扣后继概率/价值的对数空间。若用于物理时间裕度，必须重新建立量纲与校准关系；可先考虑用于可达性表示或动作排序。[TMD §3–4](https://arxiv.org/html/2509.20478v1)

**动态目标有相关依据，会合不等于稳定抓住。对应 [3]、[5]、[9]。**

DHER 假设被追求的目标不改变环境动力学。它支持接触前虚拟动态目标重标的动机，不能直接为改变真实接触对象后的轨迹重标背书。由末端终点位置、速度倒推匀速目标并监督会合时间，是文档需要独立验证的扩展。[DHER 原文](https://openreview.net/pdf?id=Byf5-30qFX)

2026 年四足拦截论文支持空间与时间共同约束的任务动机，但其装置、控制接口和成功定义不同。仿真接球指标包含球入筐后弹出的情形，不能与本仓库抓住、抬升、保持的成功率同口径比较。[实验部分](https://arxiv.org/html/2608.06907v1)

**时间势函数有理论基础，当前解释需要修正。对应 [1]、[6]、[7]。**

正确结构为 `r'_t = r_t + F_t`，其中 `F_t = γΦ(s_{t+1}) − Φ(s_t)`。势函数可以是有用的启发式，不要求等于真实 V*。因此即使 h* 只是估计，也可以尝试 Φ=−κh*；理论依赖奖励变换及边界，而不是时间估计完全准确。[Ng et al. 1999](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf)

三个需要改正的具体表述：

1. `−h*=V*` 不能直接用于当前 PPO。无折扣、每秒代价为 1、目标严格为最短到达时间时才具有这种解释。即使每步代价为 1、确定 n 步到达，折扣价值也是 `−(1−γ^n)/(1−γ)`。当前多项抓取奖励更不符合，不能据此声称按 F 贪心即当前任务最优。
2. 停止计回报的完整回合终点宜统一设 Φ=0。长度 T 的塑形折扣和为 `Σ γ^t F_t = −Φ(s_0) + γ^T Φ(s_T)`。失败终点设为 `−κH_cap` 会留下依赖终止时间的项，可能额外偏好延迟失败。终止那一步的正塑形不等于整回合存在失败套利，必须看完整折扣和。失败惩罚应明确放在任务奖励中。训练 rollout 分段造成的非终止截断不能机械地当成真正终止。[Grześ 2017 §4.1](https://www.ifaamas.org/Proceedings/aamas2017/pdfs/p565.pdf)
3. 在原奖励上加 F，与删除旧接近奖励后换成 F，是不同实验。Ng 的不变性结论直接支持前者的正确实现；后者改变了被保留的原奖励，需要单独检验任务偏好与最终成功率。

以上针对 DQ_WBC 的建议是论文与任务定义结合后的分析，不是文献已经报告的本仓库实验结果。

### 4. 标题关键词频率

按 9 篇标题的文档出现次数统计，同篇重复只计一次；合并 RL 与 reinforcement learning、goal 的复数和连字符变体。

| 关键词 | 篇数 |
|---|---:|
| Learning | 6 |
| Reinforcement learning / RL | 6 |
| Goal / goals | 3 |
| Reward shaping | 2 |
| Quasimetric | 2 |

### 5. 已发表论文引用数

仅 DDL 的保留 API 记录有引用数，无法给出整个集合有效的 Top 5。101 是 OpenAlex 返回的预印本记录计数，不是跨库合并的完整引用数。

| 可用记录排序 | 论文 | 会议年份 | 引用数 |
|---|---|---|---:|
| 1 | Dynamical Distance Learning for Semi-Supervised and Unsupervised Skill Discovery | 2020 | 101 |

### 6. 第一作者引用数

仅汇总有计数的记录，不能解释为作者影响力排名或整个集合的 Top 5。

| 可用记录排序 | 作者 | 有引用数的论文数 | 引用数合计 |
|---|---|---:|---:|
| 1 | Kristian Hartikainen | 1 | 101 |

### 7. 建议阅读路径

1. Ng 1999 [6]：先明确何种奖励变换才保留原任务最优策略。
2. Grześ 2017 [7]：重点 §4.1，处理完整回合的终止边界。
3. RL-RRT 2019 [2]：重点 §III-B、Algorithm 1，确定预测的是哪个控制策略的到达时间。
4. DDL 2020 [1]：重点 §4.1–4.3，学习轨迹时间监督与条件偏差分析。
5. QRL 2023 [4]：当覆盖与跨轨迹泛化成为瓶颈，再研究最优距离的结构与假设，随后扩展到 TMD 2025。

官方资源：[DHER 代码](https://github.com/mengf1/DHER)、[QRL 代码](https://github.com/quasimetric-learning/quasimetric-rl)、[DDL 项目与视频](https://sites.google.com/view/dynamical-distance-learning)。本轮未运行这些实现。

## 当前实施判断

优先尝试 DDL / RL-RRT 式时间估计加正确 PBRS，暂不把 QRL/TMD 作为第一版必需项。时间距离是否优于便宜的预测点距离势函数，需要本仓库消融。

保持相同初始 checkpoint、SGS 配置和评估成功口径，对比：现有奖励；现有奖励加预测点距离 PBRS；现有奖励加学习时间 PBRS。主要看完整抓取、抬升和保持成功率，以及达到相同成功率的交互量，再看接触相对速度和高速场景失败率。τ 拟合误差降低不等于控制价值提高。

τ 只用于训练奖励，可避免部署时额外查询模型，但不能消除 sim2real gap：仿真塑形仍可能教出对时序误差敏感的动作。以上论文没有证明此组合在 DQ_WBC 上可迁移。

## 检索记录与限制

未过滤 API 结果：`/tmp/dqwbc_tau_papers_20261119.json`。数据库错误未被当作无相关文献，已用一手来源定向补核。脚本错误原文：

```text
[dblp] Error on query 'dynamical distance learning': Expecting value: line 1 column 1 (char 0)
[semantic_scholar] Error on query 'dynamical distance learning': 429 Client Error:  for url: https://api.semanticscholar.org/graph/v1/paper/search?query=dynamical+distance+learning&offset=0&limit=3&fields=title%2Cauthors%2Cyear%2Cabstract%2CcitationCount%2Curl%2Cvenue%2CpublicationDate%2CexternalIds&year=1999-2026
[semantic_scholar] rate limit detected; skipping remaining queries.
```

Ng 1999 PDF 文本提取存在乱码，网页截图接口亦失败；摘要通过原论文链接的索引文本核对，关键终止边界依据正常可读的 Grześ 原文及代数展开核对。本轮没有逐页复核 Ng 原文的全部证明。

