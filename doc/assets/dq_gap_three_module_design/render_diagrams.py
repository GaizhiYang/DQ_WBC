#!/usr/bin/env python3
"""Render the proposed GRE + grasp detection + GGN documentation figures.

This script draws a design, not an implemented training/deployment pipeline.
Requires Graphviz and Noto Sans CJK SC. Outputs six DOT/SVG/PNG figure sets.
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
PALETTE = {
    "input": ("#EAF3FF", "#3D77B1"),
    "learn": ("#E9F6EE", "#438B64"),
    "fixed": ("#F0F2F5", "#7C8795"),
    "output": ("#FFF0DF", "#C68035"),
    "truth": ("#FFF7CC", "#B79A35"),
    "loss": ("#FDEBEC", "#C2626B"),
    "note": ("#FAFBFD", "#A1AABA"),
}


def quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def node_label(title, rows):
    cells = [f'<TR><TD ALIGN="CENTER"><B>{html.escape(title)}</B></TD></TR>']
    cells += [f'<TR><TD ALIGN="CENTER">{html.escape(row)}</TD></TR>' for row in rows]
    return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">' + ''.join(cells) + '</TABLE>>'


class Diagram:
    def __init__(self, name, title, subtitle):
        self.name = name
        self.lines = ["digraph G {", f'''graph [rankdir=TB, bgcolor="white", pad="0.25", nodesep="0.34", ranksep="0.40", splines=polyline, compound=true, newrank=true, outputorder=edgesfirst, fontname={quote(FONT)}, fontsize=21, labelloc=t, labeljust=c, label={quote(title + chr(10) + subtitle)}];''',
                      f'''node [shape=box, style="rounded,filled", fontname={quote(FONT)}, fontsize=12.5, fontcolor="#233247", color="#8893A2", penwidth=1.2, margin="0.12,0.09"];
edge [fontname={quote(FONT)}, fontsize=10.5, fontcolor="#516073", color="#68798D", penwidth=1.1, arrowsize=0.72];''']
        self.edges = []

    def node(self, key, title, rows=(), kind="fixed", **attrs):
        fill, stroke = PALETTE[kind]
        options = [f"label={node_label(title, rows)}", f"fillcolor={quote(fill)}", f"color={quote(stroke)}"]
        options += [f"{k}={quote(v)}" for k, v in attrs.items()]
        self.lines.append(f"{key} [{', '.join(options)}];")

    def edge(self, source, target, text="", train=False, **attrs):
        if text:
            attrs["label"] = text
        if train:
            attrs.update(style="dashed", color="#B35A66", fontcolor="#9F4C58")
        suffix = " [" + ", ".join(f"{k}={quote(v)}" for k, v in attrs.items()) + "]" if attrs else ""
        self.edges.append(f"{source} -> {target}{suffix};")

    def chain(self, *keys):
        for source, target in zip(keys, keys[1:]):
            self.edge(source, target)

    def cluster(self, key, title, color="#D7DEE8"):
        self.lines.append(f'subgraph cluster_{key} {{ label={quote(title)}; fontsize=15; fontcolor="#33445A"; style="rounded"; color={quote(color)}; penwidth=1.6; margin=16;')

    def end(self):
        self.lines.append("}")

    def same(self, *keys):
        self.lines.append("{ rank=same; " + "; ".join(keys) + "; }")

    def legend(self, rows):
        self.node("legend", "阅读约定", ["蓝：可部署观测　绿：学习网络　灰：几何/冻结模块", "黄：训练专用真值　橙：输出　红：PPO损失"] + list(rows), "note", fontsize=10.5)

    def write(self):
        path = ROOT / f"{self.name}.dot"
        path.write_text("\n".join(self.lines + self.edges + ["}", ""]), encoding="utf-8")
        for kind in ("svg", "png"):
            args = ["dot", f"-T{kind}"]
            if kind == "png":
                args += ["-Gdpi=130"]
            subprocess.run(args + [str(path), "-o", str(ROOT / f"{self.name}.{kind}")], check=True)
        svg = ET.parse(ROOT / f"{self.name}.svg").getroot()
        png = (ROOT / f"{self.name}.png").read_bytes()
        width, height = struct.unpack(">II", png[16:24])
        print(f"{self.name}: PNG {width}×{height}; SVG {svg.attrib['width']}×{svg.attrib['height']}")


def overview():
    g = Diagram("01_module_overview", "图 1｜用 GRE、Grasp Detection、Grasp Encoder 替换原 GFM", "设计 / 尚未实现；保留 DQ 的 PPO、九维动作与低层控制，单阶段训练同一部署 Actor")
    g.cluster("old", "当前基线：gfm + actor_drop_velocity_obs", "#CAD7EA")
    g.node("old_raw", "原始观测 1276D", ["features.npy 1024 + 对象位姿6 + 机器人61", "+ 基座线速度3 + 对象相对XY速度2 + 30×6候选"], "truth")
    g.node("old_feature", "原物体特征编码", ["1024 → 512 → 128"], "learn")
    g.node("old_gfm", "原 GFM", ["对象编码128 + 对象位姿6作为Query", "固定30候选经对象真值搬运 → 潜表示6"], "learn")
    g.node("old_actor", "当前 Actor 融合 201D", ["对象编码128 + GFM6 + 对象位姿6 + 机器人61", "201 → 512 → 256 → 128 → μ[9]", "删除速度5D后，仍读取对象位姿"], "learn")
    g.node("old_critic", "当前 Critic：独立参数，融合206D", ["独立对象编码128 + 独立GFM6 + 状态72", "206 → 512 → 256 → 128 → V", "仅作为独立实验基线"], "learn")
    g.end()
    g.cluster("new", "拟议方案：Actor 与 Critic 都去除 GFM / features.npy", "#BBDAC8")
    g.node("sim", "4096并行训练：缓存近似感知", ["新sugar_box视角缓存 + 捕获时仿真状态", "近似 GRE/LoG 的输出，不逐步在线检测"], "input")
    g.node("real", "真机：真实感知前端", ["深度目标ROI + 标定/FK/里程计", "GRE → 冻结LoG → Z1工具适配"], "input")
    g.node("contract", "两侧相同 Actor 观测 467D", ["机器人61 + 可见表面6", "+ 40×(位置3 + rot6) + mask40", "当前EE/B坐标；候选和表面来自同一观测包"], "input")
    g.node("encoder", "可学习 Grasp Encoder：GGN", ["参考与候选工具点 → 共享点MLP", "67D候选 → 64 → 256 → masked max256", "robot61→128；surface6→32；有效性2"], "learn")
    g.node("actor", "新 Actor：融合418D", ["256 + 128 + 32 + 2 = 418", "418 → 512 → 256 → 128 → μ[9]", "独立可学习log_std[9]；训练采样，部署均值"], "learn")
    g.node("new_critic", "新 Critic：独立复制编码器", ["同一467D观测 + 特权11 = 原始478D", "独立编码418 + 特权11 = 融合429D", "429 → 512 → 256 → 128 → V"], "learn")
    g.end()
    g.node("execution", "DQ 执行链保持不变", ["9D：末端累计目标增量6 + 夹爪1 + 底盘vx/yaw2", "原IK / 低层策略 / PD或WBC接口"], "output")
    g.legend(["Actor/Critic 不共享权重；无学生、注意力、辅助头或GRU", "单阶段高层PPO；另做低并行真实前端评测，评测不训练"])
    g.chain("old_raw", "old_feature", "old_gfm", "old_actor")
    g.edge("old_raw", "old_critic", "独立编码分支", style="dotted", constraint="false")
    g.edge("old_raw", "old_gfm", "原候选/真值", style="dotted", constraint="false")
    g.edge("sim", "contract")
    g.edge("real", "contract")
    g.chain("contract", "encoder", "actor", "execution")
    g.edge("contract", "new_critic")
    g.same("old_raw", "sim", "real")
    g.same("old_feature", "contract")
    g.same("old_gfm", "encoder")
    g.same("old_actor", "actor")
    g.same("old_critic", "new_critic", "execution")
    g.edge("execution", "legend", style="invis")
    g.edge("old_critic", "legend", style="invis")
    g.edge("new_critic", "legend", style="invis")
    g.write()


def actor_critic():
    g = Diagram("02_network_actor_critic", "图 2｜逐层网络：GGN、418D Actor 与 429D Critic", "设计 / 尚未实现；省略批维B，N=40，L=20；所有维度指单环境输入")
    g.cluster("ggn", "Actor 的 Grasp Encoder（可学习）", "#BBDAC8")
    g.node("template", "固定工具模板 χ[20,3]", ["固定种子Gaussian点、去均值", "Z1实际工具尺度；模型buffer随权重导出"], "fixed")
    g.node("grasps", "当前EE系候选与mask", ["[40,9] = t3 + R前两列rot6；mask[40]", "恢复合法R；无效槽有限占位、mask=False"], "input")
    g.node("points", "候选工具点 + 参考组", ["Pⱼ = χRⱼᵀ + tⱼ → [40,20,3]", "concat(χ, P₁,…,P₄₀) → [41,20,3]"], "fixed")
    g.node("point_mlp", "所有组/点共享点MLP", ["Linear 3→16 → LayerNorm16 → ReLU", "Linear 16→32 → LayerNorm32 → ReLU", "[41,20,3] → [41,20,32]"], "learn")
    g.node("point_max", "组内 max（20点轴）", ["[41,20,32] → [41,32]", "参考 h_ref[32] 与每候选 hⱼ[32]"], "fixed")
    g.node("tokens", "每候选67D组合", ["hⱼ32 + h_ref32 + mean(Pⱼ)3", "χ去均值：mean(Pⱼ)=tⱼ", "[40,67]；不拼score/width"], "fixed")
    g.node("token_mlp", "所有候选共享GGN MLP", ["Linear 67→64 → LayerNorm64 → ReLU", "Linear 64→256 → LayerNorm256 → ReLU", "[40,67] → [40,256]"], "learn")
    g.node("masked_max", "候选 masked max", ["有效候选上max → z_grasp[256]", "无效项屏蔽；全部无效显式返回0", "候选排列不改变集合编码"], "fixed")
    g.end()
    g.cluster("context", "Actor 的机器人/可见表面编码", "#CAD7EA")
    g.node("robot", "机器人61D", ["EE pose6 + q19 + q̇18 + base commands3", "+ 累积EE目标6 + 上次高层动作9", "原关节重排/缩放；使用导出的归一化统计"], "input")
    g.node("robot_mlp", "机器人编码128D", ["Linear 61→128 → ELU"], "learn")
    g.node("surface", "可见表面6D（捕获包摘要）", ["B系包内目标点median3 + visible1", "+ 捕获时有效深度比例1 + age1", "visible指τ时包有效", "age=(t−τ)/TTL，裁剪[0,1]", "不是对象真值中心；含无候选远处接近线索"], "input")
    g.node("surface_mlp", "表面编码32D", ["固定量纲归一化", "Linear 6→32 → ELU"], "learn")
    g.node("valid", "候选有效性2D", ["any_valid ∈ {0,1}", "valid_count/40 ∈ [0,1]"], "fixed")
    g.end()
    g.cluster("actor", "Actor参数θ：仅此网络导出部署", "#BBDAC8")
    g.node("fusion", "融合418D", ["z_grasp256 + z_robot128 + z_surface32", "+ any_valid1 + count/40 1 = 418"], "fixed")
    g.node("trunk", "Actor主干（隐藏层ELU）", ["Linear 418→512 → ELU", "Linear 512→256 → ELU", "Linear 256→128 → ELU"], "learn")
    g.node("mean", "动作均值头", ["Linear 128→9 → μ[9]", "不加输出tanh"], "learn")
    g.node("std", "独立策略方差参数", ["可学习log_std[9]", "σ=exp(clamp(log_std, −20, 2))"], "learn")
    g.node("policy", "Normal(μ, σ)", ["训练：采样9D动作、log_prob、entropy", "部署：确定性μ[9]"], "output")
    g.end()
    g.cluster("critic", "Critic参数φ：完全独立，不导出", "#D6CF9E")
    g.node("critic_raw", "Critic原始观测478D", ["Actor同包观测467 + 训练专用特权11", "特权11：对象位姿6 + 基座局部线速度3", "+ 对象相对XY速度2"], "truth")
    g.node("critic_enc", "独立复制观测编码器", ["GGN：点3→16→32；候选67→64→256", "masked max256 + robot61→128", "+ surface6→32 + 有效性2 = 418", "χ模板相同；所有学习权重与Actor独立"], "learn")
    g.node("critic_fusion", "Critic融合429D", ["独立观测编码418 + 特权11 = 429", "不含GFM或features.npy"], "fixed")
    g.node("critic_trunk", "Critic主干（隐藏层ELU）", ["Linear 429→512 → ELU", "Linear 512→256 → ELU", "Linear 256→128 → ELU", "Linear 128→1 → V(s)"], "learn")
    g.node("value", "价值V[1]", ["仅PPO训练、GAE/timeout bootstrap使用"], "output")
    g.end()
    g.legend(["原始Actor467D ≠ 融合418D；原始Critic478D ≠ 融合429D", "mask不参与点MLP数值归一化；归一化统计在整次rollout和PPO epochs内冻结"])
    g.edge("template", "points")
    g.chain("grasps", "points", "point_mlp", "point_max", "tokens", "token_mlp", "masked_max", "fusion", "trunk", "mean", "policy")
    g.chain("robot", "robot_mlp", "fusion")
    g.chain("surface", "surface_mlp", "fusion")
    g.edge("valid", "fusion")
    g.edge("std", "policy")
    g.chain("critic_raw", "critic_enc", "critic_fusion", "critic_trunk", "value")
    g.edge("critic_raw", "critic_fusion", "特权11旁路", style="dotted", constraint="false")
    g.same("template", "grasps", "robot", "critic_raw")
    g.same("point_mlp", "surface", "critic_enc")
    g.same("tokens", "surface_mlp", "valid", "critic_fusion")
    g.same("masked_max", "critic_trunk")
    g.same("mean", "std", "value")
    g.edge("robot_mlp", "surface", style="invis")
    g.edge("grasps", "masked_max", "mask", style="dotted", constraint="false")
    g.edge("grasps", "valid", "mask统计", style="dotted", constraint="false")
    g.edge("policy", "legend", style="invis")
    g.edge("value", "legend", style="invis")
    g.write()


def simulation():
    g = Diagram("03_simulation_training", "图 3｜仿真训练：先生成新缓存，再单阶段PPO", "设计 / 尚未实现；4096主训练用缓存近似感知，独立低并行在线评测检查分布差异")
    g.cluster("offline", "离线准备：运行GRE/LoG采集缓存", "#D0D7E2")
    g.node("asset", "GAP sugar_box 规范资产", ["004_sugar_box textured.obj + collision_coacd.obj", "生成Isaac URDF、物理参数与语义ID映射"], "fixed")
    g.node("capture", "独立小环境采集多视角深度", ["远/近相机、视角/距离/ROI/遮挡条件", "不依赖已有sugar_box的v3缓存"], "fixed")
    g.node("offline_front", "同一前端：GRE → 冻结LoG → Z1适配", ["固定工具轴/TCP/开口/夹爪几何兼容筛选", "当前机器人IK可达性/场景碰撞在运行时校验", "保存检测输出而非直接查询标注真值抓取"], "fixed")
    g.node("cache", "新sugar_box多视角缓存", ["候选t/R + score/width + 可见目标点（物体系）", "视角/距离/ROI条件 + 工具/相机/mesh版本", "候选和可见点绑定同一条记录"], "input")
    g.end()
    g.cluster("main", "4096并行主训练：缓存观测生成器", "#CAD7EA")
    g.node("world", "DQ B1Z1PickMulti 仿真", ["对象与机器人动态；原DQ控制频率/动作链", "sugar_box语义不变，实际加载GAP规范资产"], "fixed")
    g.node("truth", "训练专用真值", ["捕获时刻τ的对象位姿/相机视角", "当前特权状态、reward/done"], "truth")
    g.node("query", "查询单条最近缓存记录", ["按τ时视角/距离/相机ID选择一条记录", "候选和表面用同条，不混合不同时刻/视角", "τ时对象真值只用于放置该观测到固定里程计O"], "fixed")
    g.node("noise", "感知扰动与时间缓存", ["ROI/候选漏检、深度缺失、检测误差、延迟", "按可见包τ保存；空ROI帧不刷新τ", "不用当前对象真值修补旧检测"], "fixed")
    g.node("align", "当前控制时刻t的观测适配", ["O → 当前EE候选、O → 当前B包内目标点", "仅补偿机器人自身运动；age/TTL一致", "visible指捕获包有效", "age=(t−τ)/TTL，裁剪[0,1]", "最多40候选+mask；过期执行同一guard"], "fixed")
    g.node("actor_obs", "Actor观测467D", ["robot61 + surface6 + grasp40×9 + mask40", "不含物体真值 / features.npy / GFM"], "input")
    g.node("critic_obs", "Critic观测478D", ["同包Actor467 + 训练专用特权11", "对象pose6 + 基座局部线速度3 + 相对XY速度2"], "truth")
    g.node("reward", "原reward/done与timeout信号", ["可读取对象/机器人真值；不传给Actor"], "truth")
    g.end()
    g.cluster("ppo", "唯一训练阶段：非对称PPO", "#BBDAC8")
    g.node("actor", "Actor θ（图2）", ["独立GGN/机器人/表面编码 → 418D", "418→512→256→128 → Gaussian9", "采样动作按原DQ低层执行"], "learn")
    g.node("critic", "Critic φ（图2）", ["独立复制编码器，不与Actor共享参数", "418 + 特权11 → 429→512→256→128→V"], "learn")
    g.node("rollout", "固定rollout buffer", ["Actor/Critic观测、采样动作、old_log_prob、value", "reward/done、next-value及timeout bootstrap信息", "归一化统计：采集+全部PPO epochs内冻结", "minibatch不重查缓存、不重采样历史/工具模板"], "fixed")
    g.node("loss", "PPO损失", ["L_actor_clip + cV·L_value − cH·entropy", "GAE/优势估计；无辅助监督、无蒸馏损失", "按原训练超参起步，验证后再调参"], "loss")
    g.node("export", "导出同一个Actor θ", ["权重 + χ工具buffer + 固定归一化统计", "观测schema/工具/坐标/版本清单"], "output")
    g.end()
    g.node("eval", "独立低并行在线仿真评测", ["深度ROI → 真GRE → LoG → Z1适配 → 同一Actor", "检查缓存/在线候选与surface分布、成功率差距", "不更新权重；不是学生或第二训练阶段"], "note")
    g.legend(["物体真值可用于缓存生成/查询、Critic与奖励，但不会成为Actor字段", "缓存近似三模块输出，不宣称等价于完整在线GRE闭环"])
    g.chain("asset", "capture", "offline_front", "cache", "query", "noise", "align", "actor_obs", "actor", "rollout", "loss", "export")
    g.chain("world", "truth", "query")
    g.edge("truth", "critic_obs")
    g.edge("actor_obs", "critic_obs")
    g.chain("critic_obs", "critic", "rollout")
    g.edge("truth", "reward")
    g.edge("reward", "rollout")
    g.edge("loss", "actor", "更新θ", train=True, constraint="false")
    g.edge("loss", "critic", "更新φ", train=True, constraint="false")
    g.edge("actor", "world", "DQ九维动作", style="dotted", constraint="false")
    g.edge("export", "eval")
    g.same("asset", "world")
    g.same("capture", "truth")
    g.same("actor_obs", "critic_obs", "reward")
    g.same("actor", "critic")
    g.edge("eval", "legend", style="invis")
    g.write()


def explorer():
    g = Diagram("04_region_explorer", "图 4｜Graspable Region Explorer：几何跟踪与探索", "设计 / 尚未实现；GRE不学习，LoG冻结；O为固定里程计系，EE为ee_gripper_link")
    g.node("depth", "当前深度帧 + 目标ROI（捕获时刻τ）", ["完整有效场景点保留桌面/障碍上下文", "目标ROI由关联/分割提供，仅用于目标流", "无有效目标包时不刷新历史时间戳"], "input")
    g.node("calib", "捕获时标定、FK与里程计", ["深度反投影pC → pEE(τ) → pO", "使用τ时机器人位姿，不用控制t时位姿替代"], "input")
    g.node("cloud", "当前目标可见点云 pO", ["固定O系中进行历史区域匹配", "初始帧无历史则直接进入探索"], "fixed")
    g.node("scene", "完整场景上下文 scene_E(τ)", ["完整有效深度点：含桌面/障碍", "不以目标ROI裁剪LoG场景上下文", "去无效深度/异常噪点，保留遮挡与接触几何"], "fixed")
    g.node("history", "上一次有效区域记忆", ["历史可抓中心/抓取中心 cO + capture_time", "以捕获时变换写入O，不保存无时间的旧EE坐标", "几何状态，不是Actor隐状态或GRU"], "fixed")
    g.node("ball", "跟踪历史可抓区域", ["在当前pO对历史cO做球查询", "参考半径0.05m；Z1需实测确认", "丢弃无点支持或过TTL历史中心"], "fixed")
    g.node("track", "Track region：局部区域FPS", ["有当前深度支持的局部点集 → 区域中心", "保留仍可见、仍有检测意义的区域"], "fixed")
    g.node("explore", "Explore region：目标点云FPS", ["从当前ROI可见点补充新区域中心", "首帧全部来自探索；避免只盯旧区域", "GAP初始84中心仅参考，不硬套Z1预算"], "fixed")
    g.node("centers", "合并并筛选区域中心", ["Track + Explore → 去重/预算限制", "DBSCAN等去噪；保留当前点云支持", "cO → 捕获时cEE(τ)"], "fixed")
    g.node("log", "Grasp Detection：冻结LoG", ["infer_from_centers(scene_E(τ), cEE(τ))", "中心来自targetROI，场景上下文来自完整深度", "输出抓取t/R、score、width等", "LoG实现含.so，适配需实际运行验证"], "fixed")
    g.node("z1", "Z1固定工具与几何适配", ["转换工具轴、TCP、开口宽度与长度约定", "筛选固定夹爪几何不兼容候选", "IK可达性/场景碰撞由运行时另行校验", "最多40候选；不足补有限占位+mask"], "fixed")
    g.node("save", "写入下一帧区域记忆", ["新可抓中心/候选中心经τ时EE→O保存", "历史是机器人自运动补偿，不预测物体运动", "目标关联失败/过TTL时清空"], "fixed")
    g.node("out", "统一感知观测包", ["O系抓取候选 + O系目标可见点 + capture_time", "同包坐标/时间/标定/工具版本 → 图5时序适配", "score/width用于前端筛选，不输入GGN"], "output")
    g.legend(["当前深度证据修正历史区域；不需要对象规范坐标系或对象6D位姿", "GAP示例默认ws_filter；拟议DQ运行时明确启用explorer流程"])
    g.edge("depth", "cloud")
    g.edge("calib", "cloud")
    g.edge("depth", "scene")
    g.edge("calib", "scene")
    g.edge("cloud", "ball")
    g.edge("history", "ball")
    g.chain("ball", "track", "centers", "log", "z1")
    g.chain("cloud", "explore", "centers")
    g.edge("scene", "log", "完整场景上下文", style="dotted", constraint="false")
    g.edge("z1", "save")
    g.edge("z1", "out")
    g.edge("save", "history", "下一帧使用", style="dotted", constraint="false")
    g.same("depth", "calib", "history")
    g.same("cloud", "scene")
    g.same("track", "explore")
    g.same("save", "out")
    g.edge("out", "legend", style="invis")
    g.edge("save", "legend", style="invis")
    g.write()


def deployment():
    g = Diagram("05_real_deployment", "图 5｜真机部署：在线三模块前端 + 同一DQ Actor", "设计 / 尚未实现；不读取对象6D位姿或features.npy；不更换原DQ动作和低层控制")
    g.node("sensor", "选择一个有效感知包（捕获时刻τ）", ["前向远距离 / 腕部近距离深度 + 目标ROI", "每tick选择一路最近合格包；不跨时间硬拼", "相机内外参 + depth scale + 目标关联"], "input")
    g.node("fk", "时间对齐的机器人反馈", ["τ时FK/IMU/里程计；t时FK/关节/命令", "O：固定里程计系；E：ee_gripper_link", "B：机械臂安装原点 + 完整机身朝向"], "input")
    g.node("cloud", "捕获时点云构建：两条数据流", ["scene_E(τ)：完整有效场景，含桌面/障碍", "target_E(τ)：目标ROI点，仅供GRE/表面摘要", "target_E(τ)→O用于区域匹配与观测记录", "不恢复对象6D"], "fixed")
    g.node("gre", "① Graspable Region Explorer", ["O系历史区域 + 当前可见点云", "区域跟踪/球查询 + FPS新区域探索（图4）", "输出E(τ)系区域中心"], "fixed")
    g.node("gd", "② Grasp Detection", ["冻结LoG：完整scene_E(τ) + GRE中心", "目标ROI中心引导检测，保留桌面/障碍上下文", "Z1工具轴/TCP/开口/固定夹爪几何适配", "最多40个有效候选，余槽mask=False"], "fixed")
    g.node("packet", "记录 O系感知包", ["τ时EE→O转换候选，表面点也保存在O", "candidate/visible points/mask/capture_time", "ROI空帧不刷新τ；有效ROI无抓取可刷新"], "fixed")
    g.node("align", "控制时刻t的适配器", ["O→E(t)抓取；O→B(t)包内目标点", "只补偿机器人自身运动，不推断物体运动", "当前FK下候选可达性再筛，更新mask", "表面median3 + visible + depth fraction + age", "visible为τ时包有效；TTL与仿真一致", "age=(t−τ)/TTL，裁剪[0,1]"], "fixed")
    g.node("obs", "Actor原始观测467D", ["robot61 + surface6 + grasp40×9 + mask40", "robot按训练契约重排/缩放，固定导出stats", "抓取合法R用前两列rot6；无效槽有限占位"], "input")
    g.node("encoder", "③ Grasp Encoder + 状态编码", ["工具点GGN67→64→256 → masked max256", "robot61→128；surface6→32；有效性2", "融合418D（图2）"], "learn")
    g.node("actor", "训练得到的同一Actor", ["418→512→256→128 → μ[9]", "部署只取均值；不采样log_std", "无Critic、学生、辅助头或在线更新"], "learn")
    g.node("decoder", "沿用DQ九维动作解释", ["a₀…a₅更新B系累计EE目标：位置±0.02m", "RPY±0.06rad；限幅、IK和原频率沿用", "a₆ ≥ 0打开夹爪，a₆ < 0关闭", "a₇底盘vx（原幅度±0.45），a₈底盘yaw"], "output")
    g.node("guard", "感知过期 guard（训练/真机同样执行）", ["最后目标有效包超过TTL：全部候选/表面失效", "底盘vx/yaw置0；累计EE目标设当前FK保持", "维持上次夹爪命令；a₆=0不是保持", "有效ROI无抓取但未过期：允许Actor按表面接近"], "fixed")
    g.node("control", "DQ执行目标校验与原低层执行", ["动作目标IK / 场景碰撞约束在运行时校验", "执行动作IK / 原低层策略 / PD或WBC接口", "关节/末端/底盘反馈 → 下一次感知与robot61"], "output")
    g.node("required", "部署前必须验证的接口", ["LoG运行依赖 + Z1工具适配；ROI目标关联", "相机标定/时间同步/里程计误差；数据契约", "GGN导出数值一致性 + 缓存/在线分布差距"], "note")
    g.legend(["沿用DQ累计目标增量，不采用GAP的直接EE残差动作", "虚点线：姿态/有效性/反馈依赖；图示为设计，不代表模块已接入"])
    g.chain("sensor", "cloud", "gre", "gd", "packet", "align", "obs", "encoder", "actor", "decoder", "control", "required")
    g.edge("fk", "cloud", "τ时变换")
    g.edge("cloud", "gd", "完整scene_E(τ)", style="dotted", constraint="false")
    g.edge("fk", "align", "t时变换", style="dotted", constraint="false")
    g.edge("fk", "obs", "robot61", style="dotted", constraint="false")
    g.edge("align", "guard", "age/TTL状态")
    g.edge("guard", "control", "过期时覆盖命令", style="dotted")
    g.same("sensor", "fk")
    g.same("decoder", "guard")
    g.edge("required", "legend", style="invis")
    g.write()


def assets():
    g = Diagram("06_asset_and_cache", "图 6｜GAP-YCB资产复用：可直接读取文件，不可直接假定训练兼容", "设计 / 尚未实现；sugar_box 主方案使用GAP规范mesh，重新生成Z1多视角缓存")
    g.node("folder", "GAP-RL/gap_rl/data/mani_skill2_ycb", ["共享资产来源；保留原文件，新增数据写DQ侧", "语义映射：sugar_box ↔ 004_sugar_box"], "input")
    g.cluster("existing", "现有GAP文件：实际可复用内容", "#CAD7EA")
    g.node("mesh", "models/004_sugar_box", ["textured.obj：可视mesh", "collision_coacd.obj：碰撞mesh", "现有模型目录无Isaac URDF"], "input")
    g.node("info", "info_pick_v1.json", ["scale=1、density=50、bbox等", "密度/惯量/接触参数需明确设定与验证"], "input")
    g.node("v3", "info_localgrasp_v3.json", ["12物体 × 72视角；不包含sugar_box", "每抓取7D：t3 + quaternion_xyzw4", "附score/width/height/depth；并非4×4矩阵"], "input")
    g.node("v0", "info_localgrasp_v0.json", ["含sugar_box，110个4×4抓取变换", "单组抓取，并非72视角检测缓存", "不能替代目标所需的新多视角观测"], "input")
    g.node("limits", "现有缓存缺少的观测字段", ["目标可见点 / 深度ROI / 捕获条件与时间", "相机标定版本 / GRE历史轨迹 / Z1工具版本", "info_graspnet_v1.json也没有sugar_box"], "note")
    g.end()
    g.cluster("adapt", "主方案：统一规范资产与数据约定", "#BBDAC8")
    g.node("canonical", "采用GAP mesh为规范物体系", ["原DQ sugar URDF长轴x，GAP mesh长轴z", "不同mesh版本不能只改路径假定坐标一致", "相机、候选、可见点、物理尺寸用同一版本"], "fixed")
    g.node("urdf", "生成Isaac资产包装", ["GAP可视/碰撞mesh → 新URDF与物理配置", "记录尺度、质量、惯量、接触/碰撞设置", "DQ任务保留sugar_box语义与原动作控制"], "fixed")
    g.node("gen", "独立小环境重新采集", ["无需已有v3 sugar缓存即可启动", "同GRE + 冻结LoG + Z1适配生成检测", "多视角/距离/相机/ROI/遮挡条件"], "fixed")
    g.node("newcache", "新Z1-compatible sugar缓存（拟新增）", ["DQ数据目录：004_sugar_box/manifest + records", "每记录：抓取t/R、score/width、可见目标点", "规范物体系数据 + 采集条件 + 版本hash", "查询时单条记录同时产生候选与surface6"], "output")
    g.node("train", "DQ 4096缓存训练入口", ["按τ时视角选记录；只在前端用真值放置", "Actor467、Critic478；N=40，L=20", "不再加载features.npy或原GFM抓取库"], "output")
    g.end()
    g.node("reuse", "已有v3其他物体的复用边界", ["可读取作为格式参考/其他物体候选源", "先校验对象mesh、坐标轴、工具、相机与尺度", "补齐可见点/ROI字段；不能直接充当sugar缓存"], "note")
    g.legend(["新缓存路径与格式是设计建议，当前尚未生成", "GAP缓存生成脚本原路径有环境/标注依赖，主版使用独立采集器"])
    g.edge("folder", "mesh")
    g.edge("folder", "info")
    g.edge("folder", "v3")
    g.edge("folder", "v0")
    g.edge("mesh", "canonical")
    g.edge("info", "urdf")
    g.chain("canonical", "urdf", "gen", "newcache", "train")
    g.edge("v3", "reuse")
    g.edge("v0", "limits", style="invis")
    g.edge("limits", "gen", "必须补齐", style="dotted")
    g.same("mesh", "info", "canonical")
    g.same("v3", "v0", "urdf")
    g.same("limits", "gen", "reuse")
    g.edge("mesh", "v3", style="invis")
    g.edge("info", "v0", style="invis")
    g.edge("train", "legend", style="invis")
    g.edge("reuse", "legend", style="invis")
    g.write()


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    for draw in (overview, actor_critic, simulation, explorer, deployment, assets):
        draw()
