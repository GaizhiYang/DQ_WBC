"""Render the proposal diagrams. Requires Graphviz and Noto Sans CJK SC."""
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
HEADER = '''digraph G {
graph [rankdir=TB, bgcolor="white", pad="0.3", nodesep="0.28", ranksep="0.48", fontname="Noto Sans CJK SC", fontsize=19, labelloc=t];
node [shape=box, style="rounded,filled", fillcolor="#e6f3ea", color="#46785b", fontname="Noto Sans CJK SC", fontsize=12, margin="0.15,0.10"];
edge [color="#58677c", fontname="Noto Sans CJK SC", fontsize=10, arrowsize=0.7];
'''
BLUE = 'fillcolor="#e5effb", color="#4875a5"'
GREY = 'fillcolor="#edf0f4", color="#8994a5"'
PURPLE = 'fillcolor="#eee8fa", color="#8872aa"'
ORANGE = 'fillcolor="#fff0d8", color="#b7873a"'
RED = 'fillcolor="#fde7e6", color="#b76562"'

graphs = {}
graphs['01_evidence_and_gap'] = f'''
label="现有 DQ-NET：教师与学生之间的信息和训练差异";
subgraph cluster_t {{ label="教师：当前源码"; color="#b6c7dc";
t0 [label="当前特权观测 1276D\\n物体特征1024 + 物体位姿/速度\\n机器人状态 + 30×6 抓取候选 + 上次动作", {BLUE}];
t1 [label="冻结标准化器 + 物体编码128D\\nGFM：30个候选 → 学习表征6D", {GREY}];
t2 [label="Actor MLP 206→512→256→128→9", {GREY}];
t0 -> t1 -> t2;
}}
subgraph cluster_s {{ label="学生：当前源码"; color="#b6c7dc";
s0 [label="双视角 × 3帧 × (mask+depth)\\n54×96 + 当前本体状态61D\\n没有逐帧本体状态和实际采集时间戳", {BLUE}];
s1 [label="共享浅层CNN → 每帧64D\\n腕部/前向各一个2层Transformer\\n每路由当前本体状态引导"];
s2 [label="融合189D → 主动作9D + 残差9D"];
s0 -> s1 -> s2;
}}
gap [label="关键假设：有限历史能否恢复教师决策所需信息？\\n视觉年龄60–80ms；当前真值教师无需完成这种预测\\n高难度下位姿/速度误差会放大闭环动作误差", {RED}];
loss [label="现有在线DAgger：动作MSE\\n短暂教师预热后完全由学生rollout\\n无显式物体运动监督", {RED}];
t2 -> loss [label="当前状态的确定性教师动作"];
s2 -> loss;
t1 -> gap [style=dashed, arrowhead=none];
s1 -> gap [style=dashed, arrowhead=none];
note [label="证据边界：信息差与采样方式可由代码确认；各自贡献需消融\\n21万步残差问题只影响相应长训练，不能解释论文8万步结果", {GREY}];
gap -> note [style=invis]; loss -> note [style=invis];
'''

graphs['02_proposed_student'] = f'''
label="DQ-NET 改进学生：完整建议结构（待实现、待消融）";
images [label="双视角历史 I：B×2×4×2×108×192\\n每帧 mask + metric depth；有效像素另存mask", {BLUE}];
sync [label="与每张图同步的 p(τ)：61D\\n相机到当前机体变换估计9D + 年龄1D + 有效标记1D", {BLUE}];
now [label="当前可部署本体状态 p(t)：B×61\\n含上次提交的raw action；无物体真值", {BLUE}];
vision [label="共享2通道ResNet18 + 空间读出\\n单帧128D；全窗 B×8×128\\n详细层与尺寸见图3"];
pro [label="本体编码 61→64→64"];
meta [label="时间/外参编码 11→32→32"];
token [label="每帧拼接128+64+32=224\\nLinear224→256 + 视角embedding", {PURPLE}];
query [label="当前状态查询：61→128→256"];
seq [label="按采集时间排序8个视觉token\\n末尾附当前状态query：B×9×256\\nepisode mask + 有效帧mask", {PURPLE}];
tr [label="因果Transformer ×2\\nD=256，4 heads，FFN=512，Pre-LN\\n仅历史→当前；取末尾query输出 h(t)：256D"];
belief [label="运动状态头：256→128→10\\n当前物体p̂ 3 + 相对速度v̂ 3\\n位置log variance 3 + 可见性概率1"];
concat [label="动作条件：[h(t), p(t), b̂(t)]\\n256 + 61 + 10 = 327D", {PURPLE}];
actor [label="动作主干 MLP\\n327→256→128→64，ELU"];
ee [label="末端头64→6\\nΔ目标位置3 + Δ目标RPY3", {ORANGE}];
base [label="底盘头64→2\\nvx + yaw rate", {ORANGE}];
grip [label="夹爪头64→1\\n保持原命令符号语义", {ORANGE}];
action [label="组装：[Δp3, ΔRPY3, grip1, vx1, yaw1]\\n逆标签归一化 → 现有clip/积分/阈值适配", {ORANGE}];
ll [label="现有冻结底层腿部策略 + 机械臂IK + 夹爪控制\\n仿真周期先保持140–160ms；实际推理延迟待测", {GREY}];
geomloss [label="仅训练：当前位置/速度标签\\n监督 b̂；真值不进入actor输入", {RED}];
opt [label="可选训练头（默认关闭）\\n未来位置2×3D；教师隐变量128+6D\\n不影响部署输入接口", style="rounded,filled,dashed", fillcolor="#fde7e6", color="#b76562"];
images -> vision -> token;
sync -> pro; sync -> meta; pro -> token; meta -> token;
now -> query -> seq; token -> seq -> tr;
tr -> belief -> concat; tr -> concat; now -> concat;
concat -> actor; actor -> ee; actor -> base; actor -> grip;
ee -> action; base -> action; grip -> action; action -> ll;
geomloss -> belief [style=dashed, label="监督"];
tr -> opt [style=dashed];
{{rank=same; ee; base; grip;}}
'''

graphs['03_visual_encoder'] = f'''
label="建议视觉编码器细节：保留空间信息，并显式记录相机运动";
inp [label="单帧 B×2×108×192\\nmask∈[0,1] + 归一化有效深度", {BLUE}];
stem [label="Conv7×7：2→64，stride2，padding3\\nGN + ReLU → 64×54×96\\nMaxPool3×3，stride2 → 64×27×48"];
r1 [label="ResNet stage1：BasicBlock×2\\n64通道，stride1 → 64×27×48"];
r2 [label="stage2：BasicBlock×2\\n128通道，首块stride2 → 128×14×24"];
r3 [label="stage3：BasicBlock×2\\n256通道，首块stride2 → 256×7×12"];
r4 [label="stage4：BasicBlock×2\\n512通道，首块改stride1 → 512×7×12\\n所有BN替换GN；共享两相机权重"];
heat [label="1×1Conv 512→32\\n空间softmax × 深度有效性后再归一化\\n32张共同权重图；全无效时置0"];
xy [label="加权空间坐标\\n32×(x,y) = 64D", {PURPLE}];
dep [label="与xy相同权重下加权深度\\n每张权重图1个深度值，共32D\\n全无效时置0并传有效标记", {PURPLE}];
global [label="GlobalAvgPool 512D\\nLinear512→32 = 32D"];
out [label="concat：64 + 32 + 32 = 128D\\n不是绝对世界位姿；后续结合本体状态/外参", {PURPLE}];
note [label="迁移来源边界\\nVIRAL已启用：ResNet18 / 全局池化 / 可训练视觉特征\\n本图新增：2通道、GN、stride16、空间/深度读出\\n先做标准ResNet18+全局池化基线，再验证新增项", {GREY}];
inp -> stem -> r1 -> r2 -> r3 -> r4;
r4 -> heat; heat -> xy; heat -> dep;
inp -> heat [style=dashed, label="缩放后的有效性"];
inp -> dep [style=dashed, label="缩放后的深度"];
r4 -> global; xy -> out; dep -> out; global -> out;
out -> note [style=invis];
{{rank=same; xy; dep; global;}}
'''

graphs['04_training_loop'] = f'''
label="建议训练流程：持续混合rollout + 同状态教师标签 + 纯学生验证";
env [label="统一仿真任务分布 / episode ID / 时间戳\\n教师组与学生组使用相同物体和难度分层", {BLUE}];
priv [label="当前真实状态 s(t)\\n仅供冻结教师和训练标签", {BLUE}];
obs [label="延迟双视角历史 + 同步本体历史\\n当前本体状态（可部署观测）", {BLUE}];
teacher [label="冻结原DQ教师 + 冻结RunningStandardScaler\\nno_grad / eval / deterministic mean", {GREY}];
student [label="改进学生（图2）\\n训练时输入也只含可部署观测"];
mix [label="rollout角色选择（整回合固定）\\n约50%教师环境 + 50%学生环境\\n选择完整动作；不做动作向量平均", {PURPLE}];
step [label="现有动作适配器 → 冻结底层控制 → env.step", {ORANGE}];
buffer [label="记录每个当前状态的教师标签与学生观测\\n包括教师组、学生组、失败和恢复片段\\n当前rollout为主；少量分层旧片段为可选", {GREY}];
loss [label="L = L_action + λg L_gripper + λb L_belief\\n可选：λf L_future + λz L_latent + λΔ L_delta\\n先归一化标签，再逐项消融；原残差分支关闭", {RED}];
update [label="更新学生所有启用模块\\n单一动作主干端到端训练；不在21万步改换损失"];
eval [label="独立验证：100%学生闭环\\nteacher action mask恒为0，完整起点，固定测试种子\\n同时记录GSR / OSSR / 时间 / 分阶段失败", {ORANGE}];
reset [label="done/truncated：只清理对应环境\\n刷新reset观测、图像缓存、时间戳、历史有效标记\\n可选阶段快照仅用于训练，不用于正式成功率", {GREY}];
env -> priv -> teacher; env -> obs -> student;
teacher -> mix [label="aT"]; student -> mix [label="aS"]; mix -> step;
step -> env [constraint=false, label="环境闭环"];
teacher -> buffer [label="两组都重标注"]; obs -> buffer; priv -> buffer [label="辅助真值标签"];
buffer -> loss; student -> loss; loss -> update -> student [constraint=false];
student -> eval [style=dashed, label="checkpoint"];
step -> reset [label="结束回合"]; reset -> env [constraint=false];
'''

graphs['05_time_and_reset'] = f'''
label="时间对齐与重置：不能把旧图像当作当前物体状态";
subgraph cluster_old {{ label="现有时间关系（按当前配置推算）"; color="#b6c7dc";
i0 [label="图像 τ(t−2)\\n约340–400ms前", {BLUE}];
i1 [label="图像 τ(t−1)\\n约200–240ms前", {BLUE}];
i2 [label="最新图像 τ(t)\\n约60–80ms前", {BLUE}];
cur [label="决策时刻 t\\n当前本体状态 + 当前教师真值标签", {ORANGE}];
i0 -> i1 -> i2 -> cur;
}}
subgraph cluster_new {{ label="建议每个视觉token的数据契约"; color="#b6c7dc";
image [label="图像 I(τ) + 采集时间 τ", {BLUE}];
match [label="插值/采样本体状态 p(τ)\\nFK + 可部署里程计估计 T_B(t)C(τ)\\n禁止使用仿真真值机体轨迹", {BLUE}];
delta [label="Δt = t−τ\\n独立记录相机年龄和控制间隔", {PURPLE}];
infer [label="历史Transformer估计当前 b̂(t)\\n损失目标用 t 时刻真值\\n如做未来预测，标签须同一回合、统一坐标系"];
image -> infer; match -> infer; delta -> infer;
}}
reset [label="回合重置必须原子更新\\n新的物理状态 → 刷新渲染/本体 → 同episode缓存\\n无新图时标记invalid；不得伪装成新图\\n只清理done环境，保留其它环境的有效历史", {RED}];
cur -> image [style=invis]; infer -> reset [style=dashed];
'''

for name, body in graphs.items():
    path = OUT / f'{name}.dot'
    path.write_text(HEADER + body + '\n}\n', encoding='utf-8')
    for fmt in ('svg', 'png'):
        subprocess.run(['dot', f'-T{fmt}', str(path), '-o', str(OUT / f'{name}.{fmt}')], check=True)
    print(name)
