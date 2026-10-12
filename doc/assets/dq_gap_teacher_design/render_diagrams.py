#!/usr/bin/env python3
"""Generate five documentation figures, without implementing the proposal.

Requirements: Graphviz dot and Noto Sans CJK SC. Produces DOT, SVG, and PNG.
"""
from __future__ import annotations
import html
import json
from pathlib import Path
import struct
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
FONT = "Noto Sans CJK SC"
COLORS = {
    "input": ("#EAF3FF", "#3D77B1"),
    "learn": ("#E9F6EE", "#438B64"),
    "frozen": ("#F0F2F5", "#7C8795"),
    "output": ("#FFF0DF", "#C68035"),
    "gt": ("#FFF7CC", "#B79A35"),
    "loss": ("#FDEBEC", "#C2626B"),
    "note": ("#FAFBFD", "#A1AABA"),
}


def quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def label(title, lines=()):
    rows = [f'<TR><TD ALIGN="CENTER"><B>{html.escape(title)}</B></TD></TR>']
    rows.extend(f'<TR><TD ALIGN="CENTER">{html.escape(line)}</TD></TR>' for line in lines)
    return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">' + ''.join(rows) + '</TABLE>>'


class Diagram:
    def __init__(self, name, title, subtitle):
        self.name = name
        self.lines = ["digraph G {", f'''graph [rankdir=TB, bgcolor="white", pad="0.22", nodesep="0.32", ranksep="0.42", splines=polyline, compound=true, newrank=true, outputorder=edgesfirst, fontname={quote(FONT)}, fontsize=21, labelloc=t, labeljust=c, label={quote(title + chr(10) + subtitle)}];''',
                      f'''node [shape=box, style="rounded,filled", fontname={quote(FONT)}, fontsize=12.5, fontcolor="#233247", color="#8893A2", penwidth=1.2, margin="0.12,0.09"];
edge [fontname={quote(FONT)}, fontsize=10.5, fontcolor="#516073", color="#68798D", penwidth=1.1, arrowsize=0.72];''']
        self.edges = {}

    def node(self, key, title, lines=(), kind="frozen", **attrs):
        fill, stroke = COLORS[kind]
        options = [f"label={label(title, lines)}", f"fillcolor={quote(fill)}", f"color={quote(stroke)}"]
        options.extend(f"{k}={quote(v)}" for k, v in attrs.items())
        self.lines.append(f"{key} [{', '.join(options)}];")

    def edge(self, source, target, text="", training=False, **attrs):
        if text:
            attrs["label"] = text
        if training:
            attrs.update(style="dashed", color="#B35A66", fontcolor="#9F4C58")
        suffix = " [" + ", ".join(f"{k}={quote(v)}" for k, v in attrs.items()) + "]" if attrs else ""
        self.edges[(source, target)] = f"{source} -> {target}{suffix};"

    def cluster(self, name, title, color="#D7DEE8"):
        self.lines.append(f'subgraph cluster_{name} {{ label={quote(title)}; fontsize=15; fontcolor="#33445A"; style="rounded"; color={quote(color)}; penwidth=1.6; margin=16;')

    def end_cluster(self):
        self.lines.append("}")

    def same(self, *keys):
        self.lines.append("{ rank=same; " + "; ".join(keys) + "; }")

    def legend(self, extra=""):
        self.node("legend", "阅读约定", ["蓝：观测  绿：学习模块  灰：冻结/确定性模块",
                "橙：输出  黄：仿真训练真值  红：训练损失",
                extra or "实线：前向数据；红虚线：训练监督/更新；虚点线：状态依赖"], "note", fontsize=10.5)

    def write(self):
        dot_path = ROOT / (self.name + ".dot")
        dot_path.write_text("\n".join(self.lines + list(self.edges.values()) + ["}", ""]), encoding="utf-8")
        for extension in ("svg", "png"):
            command = ["dot", f"-T{extension}"]
            if extension == "png":
                command.append("-Gdpi=130")
            subprocess.run(command + [str(dot_path), "-o", str(ROOT / (self.name + "." + extension))], check=True)
        svg = ET.parse(ROOT / (self.name + ".svg")).getroot()
        png = (ROOT / (self.name + ".png")).read_bytes()
        width, height = struct.unpack(">II", png[16:24])
        print(f"{self.name}: PNG {width}×{height}; SVG {svg.attrib['width']}×{svg.attrib['height']}; DOT saved")


def baseline_vs_proposed():
    g = Diagram("01_baseline_vs_proposed", "图 1｜当前教师与拟议单阶段可部署教师", "设计方案 / 尚未实现；仿真和真机使用不同观测生成路径、同一个Actor")
    g.cluster("baseline", "当前训练：gfm + actor_drop_velocity_obs", "#CAD7EA")
    g.node("b_raw", "原 PPO 观测 1276D", ["对象特征1024 + 对象位姿6 + 机器人61", "+ 线速度5 + 原候选30×6"], "gt")
    g.node("b_drop", "Actor 仅删除线速度5D", ["1276 → 1271；物体位姿依赖仍在", "Critic继续读取原始1276D"], "frozen")
    g.node("b_shape", "对象特征编码", ["features.npy：1024 → 512 → 128"], "learn")
    g.node("b_gfm", "原 GFM", ["Query：对象编码128 + 对象位姿6", "原候选30×6 → 潜表示6"], "learn")
    g.node("b_fusion", "当前 Actor 融合 201D", ["对象编码128 + GFM6 + 状态67", "状态67 = 对象位姿6 + 机器人61", "201 → 512 → 256 → 128"], "learn")
    g.node("b_out", "原动作头", ["μ[9] + log_std[9]", "当前输入需对象位姿，无法直接感知部署"], "output")
    g.end_cluster()
    g.cluster("proposal", "提出方案：感知抓取集合替换Actor的物体真值依赖", "#BBDAC8")
    g.node("p_sim", "仿真训练侧", ["新LoG多视角缓存 + 仿真机器人状态", "真值只用于缓存查询/坐标放置及训练"], "input")
    g.node("p_real", "真机部署侧", ["深度 + 目标ROI → 在线冻结LoG", "标定 + FK + 里程计 → 当前坐标"], "input")
    g.node("p_contract", "两侧统一 Actor 原始观测367D", ["机器人61 + 可见表面6 + 30×抓取9 + mask30", "抓取9=t3+rot6；无对象真值/原离线特征", "同一观测包候选与表面来自同一捕获时间"], "input")
    g.node("p_encode", "简化共享 Actor 编码", ["机器人61→128；表面6→32", "抓取点编码 + 67D候选GGN → masked max256", "检测分数/宽度仅在前端筛选"], "learn")
    g.node("p_fusion", "新 Actor 融合 418D", ["256 + 128 + 32 + any_valid1 + count/30 1", "418 → 512 → 256 → 128"], "learn")
    g.node("p_out", "同一动作策略训练与导出", ["μ[9] + 独立log_std[9]；无tanh高斯", "训练：采样；部署：均值μ", "无注意力、辅助头、GRU或学生"], "output")
    g.end_cluster()
    g.node("critic", "仅仿真训练保留原独立 Critic", ["特权1276 → 对象编码128 + 原GFM6 + 状态72 = 206", "206 → 512 → 256 → 128 → V(s)；不与Actor共享参数", "物体真值可以训练价值与奖励，不作为新Actor输入"], "learn")
    g.legend("新方案需重新训练；保留DQ原高层9D动作与现有低层执行链")
    for a, b in [("b_raw", "b_drop"), ("b_drop", "b_shape"), ("b_drop", "b_gfm"), ("b_shape", "b_gfm"), ("b_shape", "b_fusion"), ("b_gfm", "b_fusion"), ("b_fusion", "b_out"), ("p_sim", "p_contract"), ("p_real", "p_contract"), ("p_contract", "p_encode"), ("p_encode", "p_fusion"), ("p_fusion", "p_out")]:
        g.edge(a, b)
    g.same("b_raw", "p_sim", "p_real")
    g.same("b_out", "p_out")
    g.edge("b_out", "critic", style="invis")
    g.edge("p_out", "critic", style="invis")
    g.edge("critic", "legend", style="invis")
    g.write()


def grasp_encoder():
    g = Diagram("02_grasp_encoder", "图 2｜共享Actor：67D GGN与418D融合的逐层架构", "设计方案 / 尚未实现；原始输入367D与融合特征418D不同；省略批维，N=30，L=20")
    g.cluster("grasp", "抓取集合编码：借鉴GAP-RL的点集表示与GGN", "#BBDAC8")
    g.node("template", "固定去均值工具模板χ", ["[20,3]；固定随机种子生成Gaussian点", "按Z1工具实际尺度设定；作为模型buffer导出"], "frozen")
    g.node("grasp", "前端筛选/时间对齐后的候选", ["当前EE系：[30,9]=t3+合法R前两列rot6", "确定性恢复R[30,3,3]；valid_mask[30]", "无效槽t=0、R=I、mask=False，有限占位"], "input")
    g.node("points", "工具点变换 + 参考组拼接", ["Pⱼ = χRⱼᵀ + tⱼ → [30,20,3]", "拼接参考组χ → [31,20,3]", "第0组为参考；第1–30组为抓取"], "frozen")
    g.node("point_mlp", "所有点/组共享同一 MLP", ["Linear 3→16 → LayerNorm16 → ReLU", "Linear 16→32 → LayerNorm32 → ReLU", "[31,20,3] → [31,20,32]"], "learn")
    g.node("groupmax", "组内max pooling（20点轴）", ["[31,20,32] → [31,32]", "参考h_ref[32]；抓取hⱼ[30,32]"], "frozen")
    g.node("center", "候选中心（EE系）", ["centerⱼ = mean(Pⱼ, 点轴) → [30,3]", "χ去均值时centerⱼ = tⱼ"], "frozen")
    g.node("token", "每个候选拼接67D", ["hⱼ32 + 广播h_ref32 + centerⱼ3", "[30,32] + [30,32] + [30,3]", "输出[30,67]"], "frozen")
    g.node("candidate_mlp", "所有候选共享同一 MLP", ["Linear 67→64 → LayerNorm64 → ReLU", "Linear 64→256 → LayerNorm256 → ReLU", "[30,67] → [30,256]"], "learn")
    g.node("max", "Masked max pooling（30候选轴）", ["无效项置−∞后，仅在有效候选上max", "至少1项有效 → z_grasp[256]", "全部无效 → 显式返回256维零向量"], "frozen")
    g.end_cluster()
    g.cluster("context", "机器人与可见表面编码", "#CAD7EA")
    g.node("robot", "机器人可获取状态61D", ["EE姿态6 + 关节位置19 + 关节速度18", "+ 底盘命令3 + 累积EE目标6 + 上次动作9"], "input")
    g.node("robot_mlp", "机器人编码128D", ["沿用关节重排/缩放；选定归一化契约", "Linear 61→128 → ELU"], "learn")
    g.node("surface", "可见表面摘要6D", ["B系可见点median3 + visible1", "+ validDepthFraction1 + age1", "来自目标ROI可见点；不是物体真值中心"], "input")
    g.node("surface_mlp", "表面编码32D", ["固定量纲归一化", "Linear 6→32 → ELU"], "learn")
    g.node("valid", "抓取有效性2D", ["any_valid ∈ {0,1}", "valid_count / 30 ∈ [0,1]"], "frozen")
    g.end_cluster()
    g.node("fusion", "融合418D", ["z_grasp256 + z_robot128 + z_surface32 + 有效性2", "concat → [418]"], "frozen")
    g.node("trunk", "Actor 主干", ["Linear 418→512 → ELU", "Linear 512→256 → ELU", "Linear 256→128 → ELU"], "learn")
    g.node("mean", "动作均值头", ["Linear 128→9 → μ[9]", "无输出tanh；保留DQ动作处理"], "learn")
    g.node("std", "独立策略方差参数", ["可学习log_std[9]，不依赖观测", "沿用裁剪[-20,2]；σ = exp(clamp(log_std))"], "learn")
    g.node("output", "高斯策略π(a|o) = Normal(μ, σ)", ["PPO训练：采样9D动作，计算log_prob与entropy", "真机：同一Actor输出确定性均值μ[9]"], "output")
    g.legend("B：机械臂安装点原点、完整机身朝向；EE：ee_gripper_link；无注意力/辅助分支")
    for a, b in [("template", "points"), ("grasp", "points"), ("points", "point_mlp"), ("point_mlp", "groupmax"), ("points", "center"), ("groupmax", "token"), ("center", "token"), ("token", "candidate_mlp"), ("candidate_mlp", "max"), ("robot", "robot_mlp"), ("surface", "surface_mlp"), ("max", "fusion"), ("robot_mlp", "fusion"), ("surface_mlp", "fusion"), ("valid", "fusion"), ("fusion", "trunk"), ("trunk", "mean"), ("mean", "output"), ("std", "output")]:
        g.edge(a, b)
    g.same("template", "grasp", "robot")
    g.same("points", "robot_mlp")
    g.same("point_mlp", "surface")
    g.same("groupmax", "center", "surface_mlp", "valid")
    g.same("mean", "std")
    g.edge("grasp", "max", "mask", style="dotted", constraint="false")
    g.edge("max", "fusion", weight="40")
    g.edge("fusion", "trunk", weight="40")
    g.edge("trunk", "mean", weight="40")
    g.edge("mean", "output", weight="40")
    g.edge("grasp", "valid", "mask统计", style="dotted", constraint="false")
    g.edge("robot_mlp", "surface", style="invis")
    g.edge("output", "legend", style="invis")
    g.write()


def training_network():
    g = Diagram("03_training_network", "图 3｜仿真训练侧：缓存生成观测与非对称PPO", "设计方案 / 尚未实现；单阶段训练同一Actor；Critic与奖励可以使用仿真真值")
    g.node("world", "4096并行DQ仿真环境", ["采集时刻τ：对象/相机状态；控制时刻t：机器人状态", "动作通过原DQ低层执行；与部署使用相同的观测过期处理"], "frozen")
    g.node("truth", "训练专用真值", ["对象6D位姿、对象速度、任务状态", "只供捕获时缓存查询/变换、Critic与reward/done"], "gt")
    g.node("cache", "新Z1适配LoG多视角缓存", ["检测候选（物体系）+ 可见表面点", "score/width + 视角/距离/采集条件", "深度ROI离线检测生成；不是原DQ30库"], "frozen")
    g.node("obsgen", "模拟感知：按采集时刻τ查询", ["按相机ID/方向/距离选最近单条记录，不合并near4", "用τ时对象位姿将候选/表面放入O系", "surface来自同记录可见点并做视锥筛选，非GT中心", "模拟漏检/假阳性/标定偏差/延迟"], "frozen")
    g.node("current", "时序适配：O → 当前EE/B", ["只补偿自身运动；age/TTL基于最后目标有效τ", "空ROI帧不刷新τ；无抓取但有效表面可刷新", "每tick一路最近合格包，候选/表面同τ", "Z1几何与宽度筛选 → 最多30候选 + mask"], "frozen")
    g.node("obs", "新Actor原始观测367D", ["robot61 + surface6 + 30×(t3+rot6) + mask30", "无对象真值、features.npy或原固定候选库"], "input")
    g.node("priv", "原Critic特权观测1276D", ["features.npy 1024 + 原状态72 + 原候选180", "状态含对象位姿6、机器人61、线速度5", "原30库由对象真值变换，仅供Critic"], "gt")
    g.node("reward", "原任务reward/done", ["读取对象/机器人真值", "保持原任务定义；仅训练使用"], "gt")
    g.node("buffer", "一次rollout的固定记录", ["Actor/Critic观测、动作、old_log_prob、old_value", "reward/done、next_critic/value、timeout bootstrap信息", "Actor归一化统计在采集与全部PPO epochs内冻结", "minibatch复用原观测，不重查缓存/更新历史/重采样模板"], "frozen")
    g.cluster("actor", "Actor参数θ：唯一导出部署的网络", "#BBDAC8")
    g.node("actor_enc", "可部署编码器（图2）", ["GGN67→64→256 → masked max256", "robot61→128；surface6→32；有效性2"], "learn")
    g.node("actor_trunk", "Actor融合与主干", ["418 → 512 → 256 → 128", "每个隐藏层ELU；无辅助头"], "learn")
    g.node("policy", "Normal(μ[9], σ[9])", ["μ：128→9；log_std独立参数[9]", "σ = exp(clamp(log_std, -20, 2))", "采样动作、log_prob、entropy"], "output")
    g.end_cluster()
    g.cluster("critic", "Critic参数φ：独立训练，不导出", "#D6CF9E")
    g.node("critic_features", "对象特征编码128D", ["原1024D → Linear512 → ELU", "→ Linear128（按原实现）"], "learn")
    g.node("critic_gfm", "保留原GFM → 6D", ["Query：对象编码128 + 对象位姿6", "Key/Value：原候选30×6"], "learn")
    g.node("critic_state", "原状态72D", ["对象位姿6 + 机器人61 + 线速度5"], "gt")
    g.node("critic_trunk", "Critic融合与主干", ["编码128 + GFM6 + 状态72 = 206", "206 → 512 → 256 → 128 → 1"], "learn")
    g.node("value", "标量价值Vφ(s)", ["仅估计训练回报；无参数共享到Actor"], "output")
    g.end_cluster()
    g.node("gae", "GAE与回报目标", ["reward/done、old_value、next_value构造", "区分timeout与terminal；固定目标stop-gradient"], "frozen")
    g.node("ppo", "PPO Actor损失", ["clipped ratio × advantage − entropy系数 × H", "仅更新θ：编码器、主干、均值头和log_std"], "loss")
    g.node("value_loss", "Critic价值损失", ["回归return（沿用现有PPO设置）", "仅更新φ；真值不进入Actor前向"], "loss")
    g.node("export", "单阶段训练产物", ["导出Actor权重 + χ + 归一化统计 + 观测/动作契约", "周期边界才更新统计；部署使用最终固定统计", "无学生网络、无第二阶段蒸馏"], "output")
    g.legend("黄框为真值边界；真值生成训练观测不等于Actor直接读取对象位姿")
    for a, b in [("world", "truth"), ("truth", "obsgen"), ("cache", "obsgen"), ("obsgen", "current"), ("current", "obs"), ("truth", "priv"), ("truth", "reward"), ("obs", "buffer"), ("priv", "buffer"), ("reward", "buffer"), ("buffer", "actor_enc"), ("buffer", "critic_features"), ("buffer", "critic_state"), ("actor_enc", "actor_trunk"), ("actor_trunk", "policy"), ("critic_features", "critic_gfm"), ("critic_features", "critic_trunk"), ("critic_gfm", "critic_trunk"), ("critic_state", "critic_trunk"), ("critic_trunk", "value"), ("policy", "ppo"), ("gae", "ppo"), ("value", "value_loss"), ("gae", "value_loss")]:
        g.edge(a, b)
    g.edge("buffer", "gae", "reward/done/old_value", constraint="false")
    g.edge("buffer", "critic_gfm", "原候选与对象pose", style="dotted")
    g.edge("ppo", "export", training=True)
    g.edge("value_loss", "export", style="invis")
    g.same("truth", "cache")
    g.same("obs", "priv", "reward")
    g.same("actor_enc", "critic_features", "critic_state")
    g.same("actor_trunk", "critic_gfm")
    g.same("policy", "critic_trunk")
    g.same("value", "gae")
    g.same("ppo", "value_loss")
    g.edge("export", "legend", style="invis")
    g.write()


def deployment_network():
    g = Diagram("04_deployment_network", "图 4｜真机部署侧：在线深度感知到原DQ控制链", "设计方案 / 尚未实现；同一Actor；无需对象6D估计、对象离线特征、Critic或学生")
    g.node("camera", "深度相机（前向/腕部，1或2台）", ["深度 + 可部署目标ROI；各自采集时间τ", "内参/外参已标定；双相机不假定同步"], "input")
    g.node("measure", "时间同步机器人测量", ["FK、关节反馈、IMU/里程计、控制器内部状态", "提供τ与当前t的坐标变换；构造robot61"], "input")
    g.node("cloud", "目标ROI → 可见深度点云", ["反投影、去噪、工作区/桌面过滤", "保留有效深度比例；不可用仿真GT分割"], "frozen")
    g.node("log", "冻结LoG + Z1抓取适配", ["可见点云 → 候选[R,t]、score、width", "工具轴转换与Z1尺度/开口/传感器几何检查", "score/width只筛选候选，不进入GGN"], "frozen")
    g.node("history", "捕获时统一到固定O系", ["Tᴼ_G(τ) = Tᴼ_C(τ) Tᶜ_G；表面点同τ存O系", "τ是最后目标ROI与深度有效的包；空ROI帧不刷新", "无抓取但目标表面有效，可更新τ；age/TTL从τ计算", "历史区域种子维护；失配后用新有效点云重初始化"], "frozen")
    g.node("align", "控制时刻t：转到当前EE/B", ["Tᴱ_G(t) = inv(Tᴼ_E(t)) Tᴼ_G(τ)", "pᴮ(t) = inv(Tᴼ_B(t)) pᴼ(τ)", "取一路最近合格包：候选/表面同τ，不混合多帧", "只补偿机器人自身运动；不预测对象运动"], "frozen")
    g.node("grasp", "抓取输入270D + mask30", ["筛选/去重/限额30个 → EE系[t3,rot6]", "过期/非法mask=0；有限占位t0、R=I", "全无候选时集合特征显式置零"], "input")
    g.node("surface", "表面摘要6D", ["B系可见点median3 + visible1", "+ validDepthFraction1 + age1", "表面可见但无抓取时，仍支持靠近目标"], "input")
    g.node("robot", "机器人状态61D", ["EE6 + joints19 + velocities18", "+ base commands3 + goal6 + last action9", "沿用训练重排、缩放和最终固定统计"], "input")
    g.node("contract", "同训练字段的Actor输入367D", ["robot61 + surface6 + grasp270 + mask30", "同一χ[20,3]、长度单位、工具轴和固定归一化", "any_valid与valid_count/30由mask计算"], "input")
    g.node("actor", "训练导出的同一Actor", ["GGN masked max256 + robot128 + surface32 + valid2", "418 → 512 → 256 → 128 → μ[9]", "冻结推理；执行确定性均值"], "frozen")
    g.node("action", "沿用DQ高层9D动作语义", ["a0:3 → 累积B系EE位置目标，逐轴±0.02m", "a3:6 → 累积目标RPY，逐轴±0.06rad", "a6≥0开夹爪；a6<0闭夹爪", "a7:9 → 底盘vx/yaw；并非GAP的EE残差动作"], "output")
    g.node("execute", "现有DQ低层执行接口", ["机械臂：IK → 关节目标 / PD；底盘：原低层/WBC接口", "过TTL目标失效：停底盘、EE目标设当前FK保持", "夹爪保留上次命令（不是a6=0）；仿真执行器同样处理", "更新累积目标/上次动作，下一周期重复感知"], "frozen")
    g.node("frames", "坐标与时间约定", ["O：固定里程计系；EE：ee_gripper_link", "B：机械臂安装点为原点 + 完整机身姿态", "实测感知/策略/执行耗时；TTL与丢帧处理训练一致", "对象运动会使旧候选失效，坐标补偿无法消除该误差"], "note")
    g.legend("LoG、Z1适配、标定和真机接口均需实现/验证；本图不是已部署系统")
    for a, b in [("camera", "cloud"), ("cloud", "log"), ("log", "history"), ("cloud", "history"), ("measure", "history"), ("history", "align"), ("align", "grasp"), ("align", "surface"), ("measure", "robot"), ("grasp", "contract"), ("surface", "contract"), ("robot", "contract"), ("contract", "actor"), ("actor", "action"), ("action", "execute")]:
        g.edge(a, b)
    g.edge("measure", "align", "当前t FK/里程计", style="dotted", constraint="false")
    g.same("camera", "measure")
    g.same("grasp", "surface", "robot")
    g.edge("execute", "frames", style="invis")
    g.edge("frames", "legend", style="invis")
    g.write()


def cache_contract():
    g = Diagram("05_cache_contract", "图 5｜新多视角缓存制作、仿真查询与真机在线生成", "设计方案 / 尚未实现；两侧共用输入契约，缓存分布与真实感知分布仍需验证")
    g.cluster("offline", "A. 离线制作新缓存：各视角运行真实抓取检测", "#D7DEE8")
    g.node("render", "渲染目标ROI深度与可见点", ["覆盖视角、距离、遮挡/采集条件", "对象坐标只用于离线储存变换"], "input")
    g.node("offline_log", "冻结LoG → Z1适配/几何检查", ["保存检测候选、score、width；允许检测为空", "保留检测器真正看见的表面点"], "frozen")
    g.node("newcache", "新Z1多视角缓存", ["物体系抓取变换 + 可见表面点", "检测元数据 + 视角/距离/采集条件", "不是features.npy，也不是原DQ固定30库"], "frozen")
    g.end_cluster()
    g.cluster("sim", "B. 仿真训练侧：缓存近似感知输出", "#E5D89B")
    g.node("sim_truth", "采集时刻τ对象/相机真值", ["选缓存视角；放置缓存抓取与可见点", "真值不作为新Actor输入字段"], "gt")
    g.node("query", "首版：最近单条视角记录", ["按相机ID/方向/距离选同一条候选与可见表面", "surface做当前视锥筛选；不合并near4多视角", "模拟漏检/假阳性/标定偏差/捕获延迟", "静态缓存不能完整表达当前动态遮挡"], "frozen")
    g.node("sim_o", "按捕获时真值放入O系", ["Tᴼ_G(τ) = Tᴼ_obj(τ) Tᵒᵇʲ_G", "pᴼ(τ) = Tᴼ_obj(τ) pᵒᵇʲ", "之后只随机器人转帧；不能改用t时刻对象真值"], "frozen")
    g.end_cluster()
    g.cluster("real", "C. 真机部署侧：在线感知产生同义字段", "#BED3E9")
    g.node("real_cloud", "真实深度 + 目标ROI点云", ["相机标定、capture timestamp、机器人FK", "无需对象6D估计或按对象查询缓存"], "input")
    g.node("online_log", "同一冻结LoG + Z1工具适配", ["在线预测候选、score、width与可见点", "保留检测失败；可更新历史区域种子"], "frozen")
    g.node("real_o", "按捕获时机器人测量放入O系", ["Tᴼ_G(τ) = Tᴼ_C(τ) Tᶜ_G", "pᴼ(τ) = Tᴼ_C(τ) pᶜ", "保存捕获时间；不使用对象真值"], "frozen")
    g.end_cluster()
    g.node("adapter", "两侧相同的时序/几何观测适配器", ["O系历史 → 当前EE候选 + B系可见表面；age/TTL判定", "τ=最后目标ROI/深度有效包；空ROI新帧不刷新τ", "分数/宽度/几何筛选 → 最多30个 + mask；不足项有限占位", "取一路最近合格包：候选/表面同τ；robot61来自当前状态"], "frozen")
    g.node("contract", "统一Actor契约：原始367D，融合418D", ["robot61；surface6=[medianᴮ3, visible1, depthfrac1, age1]", "30×[t3,rot6] + mask30；固定工具模板χ[20,3]", "67D候选token → masked max256；融合418D", "score/width仅前端使用；无对象真值或原离线特征"], "input")
    g.node("actor", "同一个Actor", ["仿真：PPO单阶段高斯采样训练", "真机：最终固定统计 + 同一权重，输出μ[9]", "不训练学生，不需要策略蒸馏"], "learn")
    g.node("validation", "验证感知分布差距", ["少量在线检测仿真rollout对比缓存训练rollout", "比较候选分布、无候选率、延迟及任务成功率", "缓存不能保证真实遮挡一致；需校准缓存与感知误差模型"], "note")
    g.legend("缓存仅训练提速；部署能力取决于真实感知能否产生同义几何输入")
    for a, b in [("render", "offline_log"), ("offline_log", "newcache"), ("newcache", "query"), ("sim_truth", "query"), ("query", "sim_o"), ("real_cloud", "online_log"), ("online_log", "real_o"), ("sim_o", "adapter"), ("real_o", "adapter"), ("adapter", "contract"), ("contract", "actor"), ("actor", "validation")]:
        g.edge(a, b)
    g.same("sim_truth", "real_cloud")
    g.same("query", "online_log")
    g.same("sim_o", "real_o")
    g.edge("validation", "legend", style="invis")
    g.write()


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    baseline_vs_proposed()
    grasp_encoder()
    training_network()
    deployment_network()
    cache_contract()
