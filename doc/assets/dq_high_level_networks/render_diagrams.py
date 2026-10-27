"""Render the architecture figures with Graphviz (no Python dependencies).

Run from any directory: python /path/to/render_diagrams.py
Teacher figures describe --no_velocity_obs --future_pose_obs, horizons .30/.60 s.
"""
from pathlib import Path
import json
import subprocess

OUT = Path(__file__).resolve().parent
COLORS = {
    "input": ("#EAF3FF", "#5385B9"),
    "net": ("#EAF7F0", "#478364"),
    "op": ("#F2EFFF", "#8173AD"),
    "output": ("#FFF1DC", "#B48640"),
    "fixed": ("#F1F3F5", "#8A949E"),
    "loss": ("#FCEDEC", "#B87570"),
    "forecast": ("#DDF4F1", "#218B80"),
}


class Figure:
    def __init__(self, name, title, direction="TB"):
        self.name = name
        self.lines = [
            "digraph G {",
            f'graph [rankdir={direction}, bgcolor="white", pad="0.28", nodesep="0.35", ranksep="0.40", splines=polyline, fontname="Noto Sans CJK SC", fontsize=19, labelloc=t, label={json.dumps(title, ensure_ascii=False)}];',
            'node [shape=box, style="rounded,filled", fontname="Noto Sans CJK SC", fontsize=12, margin="0.16,0.10", penwidth=1.2];',
            'edge [color="#65758B", arrowsize=0.7, fontname="Noto Sans CJK SC", fontsize=10];',
        ]

    def node(self, key, label, kind="net"):
        fill, stroke = COLORS[kind]
        self.lines.append(f'{key} [label={json.dumps(label, ensure_ascii=False)}, fillcolor="{fill}", color="{stroke}"];')
        return key

    def edge(self, start, end, label="", dashed=False):
        attrs = []
        if label:
            attrs.append("label=" + json.dumps(label, ensure_ascii=False))
        if dashed:
            attrs.append('style=dashed')
        self.lines.append(f'{start} -> {end}' + (" [" + ", ".join(attrs) + "]" if attrs else "") + ";")

    def chain(self, *keys):
        for a, b in zip(keys, keys[1:]):
            self.edge(a, b)

    def same_rank(self, *keys):
        self.lines.append("{rank=same; " + "; ".join(keys) + ";}")

    def render(self, png=False):
        source = "\n".join(self.lines + ["}"])
        subprocess.run(["dot", "-Tsvg", "-o", str(OUT / (self.name + ".svg"))], input=source, text=True, check=True)
        if png:
            subprocess.run(["dot", "-Tpng", "-Gdpi=150", "-o", str(OUT / (self.name + ".png"))], input=source, text=True, check=True)


def overview():
    g = Figure("01_training_overview", "高层训练与执行：两阶段学习，同一套动作接口")
    for key, label, kind in [
        ("env", "B1Z1PickMulti 仿真环境", "fixed"),
        ("prediction", "环境侧：最近5帧世界位姿 + 时间戳\n固定EKF预测0.30 / 0.60秒，再转换坐标", "fixed"),
        ("ot", "教师观测 oT：[B,1284]\n包含未来位姿12 + 有效标志1\n已移除机器人线速度3与物体相对速度2", "input"),
        ("os", "学生观测 oS：[B,62269]\n双视角 × 三帧 × 双通道图像 + 状态61", "input"),
        ("t", "教师 Actor\n特征编码 + 抓取注意力 + MLP", "net"),
        ("v", "教师 Critic\n独立编码、注意力、价值MLP", "net"),
        ("ppo", "阶段一：PPO\n奖励、优势、策略损失、价值损失", "loss"),
        ("s", "学生 Policy\n共享CNN + 双Transformer + 动作与残差头", "net"),
        ("sample", "阶段一：教师高斯采样动作\n[B,9]", "output"),
        ("label", "阶段二：冻结教师的均值动作\naT：[B,9]", "output"),
        ("dag", "阶段二：DAgger\n教师动作监督；后期仅更新残差头", "loss"),
        ("as", "学生最终动作 aS：[B,9]", "output"),
        ("control", "动作执行\n预训练底层Actor + 腿部PD + 机械臂IK + 夹爪", "fixed"),
    ]:
        g.node(key, label, kind)
    g.edge("env", "ot"); g.edge("env", "os")
    g.chain("env", "prediction", "ot")
    g.edge("ot", "t"); g.edge("ot", "v"); g.edge("t", "ppo"); g.edge("v", "ppo")
    g.edge("env", "ppo", "奖励", True)
    g.chain("os", "s", "as"); g.chain("t", "label", "dag"); g.edge("as", "dag")
    g.chain("t", "sample", "control")
    g.edge("as", "control", "学生控制阶段"); g.edge("label", "control", "预热阶段")
    g.edge("control", "env", "执行后返回新观测", True)
    g.same_rank("t", "s", "v")
    return g


def teacher_actor():
    g = Figure("02_teacher_actor", "最新教师 Actor：1284维观测 → 214维融合 → 9维动作")
    for key, label, kind in [
        ("source", "环境侧生成新增13维（详见EKF流程图）\n最近5帧世界位姿 → EKF → 0.30 / 0.60秒预测\n使用当前机器人参考转换；EKF无可学习参数", "fixed"),
        ("raw", "教师观测 oT：[B,1284]\n特征1024 + 当前状态58 + 预测13 + 候选180 + 上一动作9", "input"),
        ("norm", "RunningStandardScaler\n全部1284维标准化（含有效标志），再分支", "fixed"),
        ("f", "物体特征：[B,1024]\n[0:1024]", "input"),
        ("p", "当前物体位姿：[B,6]\n[1024:1030]", "input"),
        ("gs", "当前位置候选90 + 当前姿态候选90\n[1095:1275] → G：[B,30,6]", "input"),
        ("other", "当前状态：[B,58]\n[1024:1082]，含当前物体位姿6", "input"),
        ("future", "新增预测观测：[B,13]\n[1082:1095]\n0.30秒位姿6 + 0.60秒位姿6 + valid1", "forecast"),
        ("previous", "上一高层动作：[B,9]\n[1275:1284]", "input"),
        ("enc", "feature_encoder\nLinear 1024→512 + ELU\nLinear 512→128", "net"),
        ("att", "抓取候选注意力\nQuery = 编码128 + 当前物体位姿6\nKey / Value = 当前30个抓取候选\n输出抓取表示：[B,6]", "net"),
        ("cat", "按代码顺序拼接：[当前58, 预测13, 上一动作9, 编码128, 抓取表示6]\n58 + 13 + 9 + 128 + 6 = 214", "op"),
        ("l1", "Linear 214→512 + ELU", "net"),
        ("l2", "Linear 512→256 + ELU", "net"),
        ("l3", "Linear 256→128 + ELU", "net"),
        ("mean", "Linear 128→9\n动作均值 μ：[B,9]", "net"),
        ("std", "独立可学习 log_std：[9]\n初始为0；默认限制在[-20,2]", "net"),
        ("dist", "Normal(μ, exp(log_std))\n训练：采样；评估/蒸馏：均值", "op"),
        ("out", "高层动作 aT：[B,9]", "output"),
    ]:
        g.node(key, label, kind)
    g.chain("source", "raw", "norm")
    for k in ("f", "p", "gs", "other", "future", "previous"): g.edge("norm", k)
    g.chain("f", "enc"); g.edge("enc", "att", "编码128"); g.edge("p", "att"); g.edge("gs", "att")
    g.edge("enc", "cat", "编码128"); g.edge("other", "cat"); g.edge("att", "cat", "抓取表示6")
    g.edge("future", "cat", "预测直接进入主MLP"); g.edge("previous", "cat")
    g.chain("cat", "l1", "l2", "l3", "mean", "dist", "out"); g.edge("std", "dist")
    return g


def feature_encoder():
    g = Figure("03_teacher_feature_encoder", "教师物体特征编码器：Actor与Critic各有一套独立参数", "LR")
    for key, label, kind in [
        ("f", "features.npy\n预计算特征1024\n输入已标准化", "input"),
        ("l1", "Linear\n1024→512", "net"),
        ("e", "ELU", "op"),
        ("l2", "Linear\n512→128", "net"),
        ("z", "物体编码 z\n[B,128]\n末尾无激活", "output"),
    ]: g.node(key, label, kind)
    g.chain("f", "l1", "e", "l2", "z")
    return g


def grasp_attention():
    g = Figure("04_grasp_attention", "抓取候选注意力：单头、单Query、30个Key/Value")
    for key, label, kind in [
        ("z", "物体编码 z：[B,128]", "input"), ("p", "当前物体位姿 p：[B,6]\n未来13维不进入Query", "input"),
        ("context", "concat(z,p)：[B,134]", "op"),
        ("q", "query_proj：Linear 134→64\nunsqueeze → Q：[B,1,64]", "net"),
        ("grasps", "G：[B,30,6]\n由当前物体位姿变换的抓取候选", "input"),
        ("k", "key_proj：Linear 6→64\nK：[B,30,64]", "net"),
        ("v", "value_proj：Linear 6→64\nV：[B,30,64]", "net"),
        ("score", "Q × transpose(K) / √64\n[B,1,30]", "op"),
        ("soft", "Softmax（候选维）\nα：[B,1,30]", "op"),
        ("weighted", "α × V：[B,1,64]\nsqueeze → [B,64]", "op"),
        ("out", "output_proj：Linear 64→6\n抓取表示 g：[B,6]", "net"),
        ("note", "连续注意力汇总 + 线性投影\n没有argmax、候选索引或位姿监督损失", "fixed"),
    ]: g.node(key, label, kind)
    g.edge("z", "context"); g.edge("p", "context"); g.chain("context", "q", "score", "soft", "weighted", "out", "note")
    g.edge("grasps", "k"); g.edge("grasps", "v"); g.edge("k", "score"); g.edge("v", "weighted")
    return g


def teacher_critic():
    g = Figure("05_teacher_critic", "最新教师 Critic：同样接收预测13维，参数与Actor独立")
    for key, label, kind in [
        ("o", "教师观测：[B,1284]\n经过同一PPO状态标准化器", "input"),
        ("e", "Critic独立 feature_encoder\n1024→512 + ELU→128", "net"),
        ("g", "当前物体位姿6 + 当前抓取候选30×6", "input"),
        ("a", "Critic独立 PredictAttentionSelector\n134/6→64→6", "net"),
        ("r", "当前状态58 + 上一动作9\n物体当前位姿包含在58维中", "input"),
        ("future", "新增预测观测：[B,13]\n未来0.30秒6 + 0.60秒6 + valid1", "forecast"),
        ("c", "concat([当前58, 预测13, 上一动作9, 编码128, 抓取表示6])\n[B,214]", "op"),
        ("l1", "Linear 214→512 + ELU", "net"),
        ("l2", "Linear 512→256 + ELU", "net"),
        ("l3", "Linear 256→128 + ELU", "net"),
        ("l4", "Linear 128→1", "net"),
        ("v", "价值输出：[B,1]\nPPO配合价值标准化/逆变换使用", "output"),
        ("unused", "注册的 log_std：[9]\n未参与 Critic 前向", "fixed"),
    ]: g.node(key, label, kind)
    for k in ("e", "g", "r", "future"): g.edge("o", k)
    g.edge("future", "c")
    g.edge("e", "a"); g.edge("g", "a"); g.edge("r", "c"); g.edge("e", "c"); g.edge("a", "c")
    g.chain("c", "l1", "l2", "l3", "l4", "v")
    return g


def pose_prediction():
    g = Figure("18_teacher_pose_prediction", "EKF观测生成：世界系预测 → 原教师坐标表示 → 新增13维")
    for key, label, kind in [
        ("world", "当前物体世界位姿\n位置xyz + 四元数xyzw：[B,7]", "input"),
        ("time", "真实仿真时间戳\ngym.get_sim_time()", "input"),
        ("history", "每个环境独立的滚动缓存\n最近5帧世界位姿 [B,5,7] + 时间戳 [B,5]", "fixed"),
        ("ekf", "RollingSE3Predictor：固定SE(3) EKF\n每次用5帧重新初始化并更新\nQ scale=1；R scale=0.01；使用实际Δt\n不读仿真真值速度，不参与PPO反向传播", "fixed"),
        ("forecast", "预测未来世界位姿：[B,2,7]\nt + 0.30秒、t + 0.60秒", "output"),
        ("robot", "当前时刻 t 的机器人位置和姿态\n计算机械臂基座世界位置、机器人yaw\n两组未来位姿共用这一个参考", "input"),
        ("convert", "object_pose_in_teacher_frame\nxy：机械臂基座相对位置，经机器人yaw逆旋转\nz：直接保留预测的世界高度\nRPY：预测姿态去掉机器人yaw后转欧拉角", "op"),
        ("pose", "两组未来位姿：[B,2,6]\n每组 [x,y,z,roll,pitch,yaw]", "forecast"),
        ("valid", "有效标志 valid：[B,1]\n5帧齐全且预测数值正常才为1\n不是准确率或置信概率", "fixed"),
        ("reset", "环境reset：只清空对应环境历史\n历史不足或数值异常：valid=0\n两组未来位姿都填当前物体观测6维", "fixed"),
        ("block", "拼接新增观测：[B,13]\n0.30秒位姿6 | 0.60秒位姿6 | valid1", "forecast"),
        ("obs", "放在当前状态之后、抓取候选之前\n完整观测1284 → 标准化 → Actor / Critic主MLP", "output"),
    ]: g.node(key, label, kind)
    g.edge("world", "history"); g.edge("time", "history")
    g.chain("history", "ekf", "forecast", "convert", "pose", "block", "obs")
    g.edge("robot", "convert"); g.edge("ekf", "valid"); g.edge("valid", "block")
    g.edge("reset", "history", "清空", True); g.edge("reset", "block", "回退规则", True)
    return g


def gaussian():
    g = Figure("06_teacher_distribution", "教师动作输出：默认高斯策略与确定性教师标签")
    for key, label, kind in [
        ("m", "Actor MLP均值 μ(s)\n[B,9]", "input"),
        ("l", "log_std 参数 [9]\n不依赖当前观测", "net"),
        ("s", "Clamp(-20,2) → exp\nσ：[9]，广播到batch", "op"),
        ("d", "逐动作维 Normal(μ,σ)\n对角协方差", "op"),
        ("train", "训练：rsample\na = μ + σ ⊙ ε", "output"),
        ("eval", "评估 / 学生训练中的教师\ndeterministic=True → a=μ", "output"),
        ("log", "log_prob：逐维计算后sum\n[B,1]；供PPO使用", "op"),
    ]: g.node(key, label, kind)
    g.chain("l", "s", "d"); g.edge("m", "d"); g.edge("d", "train"); g.edge("m", "eval"); g.edge("d", "log")
    return g


def student():
    g = Figure("07_student_policy", "学生 Policy：共享视觉编码、独立时序分支、联合动作与残差")
    for key, label, kind in [
        ("raw", "学生输入：[B,62269]", "input"),
        ("img", "图像62208 → [B,12,54,96]\n2视角 × 3帧 × 2通道", "op"),
        ("r", "当前机器人状态 r：[B,61]", "input"),
        ("arm", "腕部 mask+目标深度\n[B×3,2,54,96]", "input"),
        ("base", "前向 mask+目标深度\n[B×3,2,54,96]", "input"),
        ("ca", "shared_cnn 同一个实例\n逐帧 2×54×96→64", "net"),
        ("cb", "shared_cnn 同一个实例\n逐帧 2×54×96→64", "net"),
        ("ta", "transformer_arm（独立参数）\n3×64视觉 + 状态token → 64", "net"),
        ("tb", "transformer_base（独立参数）\n3×64视觉 + 状态token → 64", "net"),
        ("pa", "arm_proj：Linear 64→64", "net"),
        ("pb", "base_proj：Linear 64→64", "net"),
        ("f", "fusion = concat(arm64,base64,r61)\n[B,189]", "op"),
        ("a", "action_head\n189→128 ELU→64 ELU→9\n初始动作 a0", "net"),
        ("res", "residual_mlp\nconcat(fusion,a0)：198\n198→128 ELU→64 ELU→9", "net"),
        ("sum", "aS = a0 + Δa\n最终动作：[B,9]", "output"),
        ("unused", "log_std 参数 [9] 未使用\n没有学生Critic / 循环隐藏状态", "fixed"),
    ]: g.node(key, label, kind)
    g.edge("raw", "img"); g.edge("raw", "r"); g.edge("img", "arm"); g.edge("img", "base")
    g.chain("arm", "ca", "ta", "pa", "f"); g.chain("base", "cb", "tb", "pb", "f")
    g.edge("ca", "cb", "共享全部CNN权重", True)
    for k in ("ta", "tb", "f"): g.edge("r", k)
    g.chain("f", "a", "sum"); g.edge("f", "res"); g.edge("a", "res"); g.edge("res", "sum", "Δa")
    g.same_rank("ca", "cb"); g.same_rank("ta", "tb"); g.same_rank("pa", "pb")
    return g


def cnn():
    g = Figure("08_shared_cnn", "SharedCNNBackbone：所有视角、所有帧共享参数")
    layers = [
        ("x", "单个视角的帧批量\n[B×3,2,54,96]", "input"),
        ("c1", "Conv2d 2→16\nkernel=5, stride=1, padding=0\n[B×3,16,50,92]", "net"),
        ("pool", "MaxPool2d kernel=2, stride=2\n[B×3,16,25,46]", "op"),
        ("e1", "ELU", "op"),
        ("c2", "Conv2d 16→32\nkernel=3, stride=1, padding=0\n[B×3,32,23,44]", "net"),
        ("e2", "ELU", "op"),
        ("flat", "Flatten：32×23×44=32384\n[B×3,32384]", "op"),
        ("fc1", "Linear 32384→128 + ELU", "net"),
        ("fc2", "Linear 128→64\n末尾无激活", "net"),
        ("seq", "reshape → 时序视觉特征 [B,3,64]", "output"),
    ]
    for key, label, kind in layers: g.node(key, label, kind)
    g.chain(*(x[0] for x in layers))
    return g


def guided():
    g = Figure("09_guided_transformer", "GuidedTransformerBlock：机器人状态作为第一个token")
    for key, label, kind in [
        ("f", "视觉特征序列 [B,3,64]", "input"),
        ("pe", "固定 sin/cos 位置编码\nPE：[1,3,64]（buffer上限10帧）", "fixed"),
        ("add", "逐元素相加\n[B,3,64]", "op"),
        ("r", "当前机器人状态 [B,61]", "input"),
        ("rp", "state_proj：Linear 61→64\nunsqueeze → [B,1,64]", "net"),
        ("cat", "concat([状态token, 视觉token序列])\n[B,4,64]；状态token不加位置编码", "op"),
        ("l1", "TransformerEncoderLayer 1\nd_model=64, heads=2, FFN=2048", "net"),
        ("l2", "TransformerEncoderLayer 2\nd_model=64, heads=2, FFN=2048", "net"),
        ("slice", "取 y[:,1:] → [B,3,64]\n排除状态token", "op"),
        ("pool", "mean(dim=1)\n输出 [B,64]", "output"),
    ]: g.node(key, label, kind)
    g.edge("f", "add"); g.edge("pe", "add"); g.chain("r", "rp", "cat")
    g.chain("add", "cat", "l1", "l2", "slice", "pool")
    return g


def encoder_layer():
    g = Figure("10_transformer_encoder_layer", "单层 TransformerEncoderLayer：Post-LN，两个残差连接")
    layers = [
        ("x", "X：[B,4,64]", "input"),
        ("mha", "多头自注意力 MHA(X,X,X)\n三路输入同为X；内部独立Q/K/V投影\n2头 × 32维；注意力概率Dropout=0.1", "net"),
        ("d1", "dropout1：p=0.1", "op"),
        ("add1", "Add：X + attention_output", "op"),
        ("ln1", "norm1：LayerNorm(64)\neps=1e-5；可学习γ、β", "net"),
        ("f1", "linear1：Linear 64→2048", "net"),
        ("relu", "ReLU", "op"),
        ("drop", "dropout：p=0.1", "op"),
        ("f2", "linear2：Linear 2048→64", "net"),
        ("d2", "dropout2：p=0.1", "op"),
        ("add2", "Add：norm1_output + FFN_output", "op"),
        ("ln2", "norm2：LayerNorm(64)\n输出 [B,4,64]", "net"),
    ]
    for key, label, kind in layers: g.node(key, label, kind)
    g.chain(*(x[0] for x in layers)); g.edge("x", "add1", "残差1"); g.edge("ln1", "add2", "残差2")
    return g


def mha():
    g = Figure("11_multihead_attention", "学生自注意力内部：4个token，两头，每头32维")
    for key, label, kind in [
        ("x", "X：[B,4,64]\n状态token + 3帧视觉token", "input"),
        ("qkv", "in_proj_weight：[192,64]，bias：[192]\n等价于三个独立 Linear 64→64\n得到 Q、K、V", "net"),
        ("split", "拆成2个头\n每个Q/K/V：[B,2,4,32]", "op"),
        ("score", "Qh × transpose(Kh) / √32\n[B,2,4,4]", "op"),
        ("soft", "Softmax（Key维） + Dropout(0.1)\n无因果mask / 无padding mask", "op"),
        ("weighted", "每头权重 × Vh\n[B,2,4,32]", "op"),
        ("merge", "合并两个头 → [B,4,64]", "op"),
        ("out", "out_proj：Linear 64→64\n[B,4,64]", "net"),
    ]: g.node(key, label, kind)
    g.chain("x", "qkv", "split", "score", "soft", "weighted", "merge", "out")
    g.edge("split", "weighted", "Vh")
    return g


def heads():
    g = Figure("12_student_action_heads", "学生联合动作头与残差头：残差始终参与前向")
    for key, label, kind in [
        ("arm", "腕部Transformer输出64", "input"), ("base", "前向Transformer输出64", "input"),
        ("ap", "arm_proj：Linear 64→64\n无激活", "net"), ("bp", "base_proj：Linear 64→64\n无激活", "net"),
        ("r", "机器人状态61", "input"), ("f", "融合 F：[B,189]", "op"),
        ("a1", "Linear 189→128 + ELU", "net"), ("a2", "Linear 128→64 + ELU", "net"),
        ("a0", "Linear 64→9\na0：[B,9]", "net"),
        ("rc", "concat(F,a0) → [B,198]", "op"),
        ("r1", "Linear 198→128 + ELU", "net"), ("r2", "Linear 128→64 + ELU", "net"),
        ("delta", "Linear 64→9\nΔa：[B,9]", "net"),
        ("sum", "aS = a0 + Δa\n[B,9]，无输出Tanh", "output"),
    ]: g.node(key, label, kind)
    g.chain("arm", "ap", "f"); g.chain("base", "bp", "f"); g.edge("r", "f")
    g.chain("f", "a1", "a2", "a0", "sum"); g.edge("f", "rc"); g.edge("a0", "rc")
    g.chain("rc", "r1", "r2", "delta", "sum")
    return g


def training():
    g = Figure("13_student_training", "学生默认DAgger_RNN训练路径：动作监督与参数更新范围")
    for key, label, kind in [
        ("obs", "同一环境状态\n分别构造教师/学生观测", "input"),
        ("t", "教师Actor：eval + no_grad\n输出均值动作 aT", "fixed"),
        ("s", "学生Policy：采样时eval + no_grad\naS = a0 + Δa", "net"),
        ("memory", "Memory\n学生观测、aT、采样时学生最终动作\naS_old（常量监督目标的一部分）", "fixed"),
        ("early", "t < 210000\n冻结 residual_mlp 参数\n更新CNN、Transformer、投影、action_head", "net"),
        ("le", "L = MSE(aS_current, aT)\n冻结的残差分支仍传递输入梯度", "loss"),
        ("late", "t ≥ 210000\n仅 residual_mlp 参数可训练\n其余模块被冻结", "net"),
        ("ll", "L = MSE(Δa_current, aT − aS_old)\n目标不是 aT − 当前a0", "loss"),
        ("step", "环境控制权\n前4000步：教师\n之后：学生\ndepth_random时预热8000步", "fixed"),
    ]: g.node(key, label, kind)
    g.edge("obs", "t"); g.edge("obs", "s"); g.edge("t", "memory"); g.edge("s", "memory")
    g.edge("t", "step"); g.edge("s", "step")
    g.chain("memory", "early", "le"); g.chain("memory", "late", "ll")
    return g


def low_actor():
    g = Figure("14_low_level_actor", "高层调用的预训练底层Actor：hist_encoding=True，整个调用no_grad")
    for key, label, kind in [
        ("o", "部署输入 [B,781]\n当前本体状态71 + 历史10×71", "input"),
        ("p", "当前本体状态 [B,71]", "input"),
        ("h", "历史 [B,10,71]", "input"),
        ("he", "StateHistoryEncoder\n逐帧MLP + 两层Conv1d + MLP\n输出 [B,20]", "net"),
        ("c", "concat(当前71, 历史编码20)\n[B,91]", "op"),
        ("trunk", "actor_backbone\nLinear 91→128 + ELU", "net"),
        ("leg", "actor_leg_control_head\nLinear 128→128 + ELU\nLinear 128→128 + ELU\nLinear 128→12", "net"),
        ("arm", "actor_arm_control_head\nLinear 128→128 + ELU\nLinear 128→128 + ELU\nLinear 128→6", "net"),
        ("out", "Actor输出：concat → 18维动作均值", "output"),
        ("pd", "环境重排关节顺序\n腿部12维用于PD力矩\n机械臂6维对应力矩被置零\n机械臂实际通过IK目标执行", "fixed"),
    ]: g.node(key, label, kind)
    g.edge("o", "p"); g.edge("o", "h"); g.chain("h", "he", "c"); g.edge("p", "c")
    g.chain("c", "trunk"); g.edge("trunk", "leg"); g.edge("trunk", "arm")
    g.edge("leg", "out"); g.edge("arm", "out"); g.edge("out", "pd")
    return g


def low_history():
    g = Figure("15_low_level_history_encoder", "底层 StateHistoryEncoder：10步本体历史 → 20维隐变量")
    layers = [
        ("x", "历史观测 [B,10,71]", "input"),
        ("l1", "逐时间步 Linear 71→30 + ELU\n[B×10,30]", "net"),
        ("perm", "reshape + permute\n[B,30,10]（通道30，时间10）", "op"),
        ("c1", "Conv1d 30→20\nkernel=4, stride=2, padding=0\n+ ELU → [B,20,4]", "net"),
        ("c2", "Conv1d 20→10\nkernel=2, stride=1, padding=0\n+ ELU → [B,10,3]", "net"),
        ("flat", "Flatten → [B,30]", "op"),
        ("l2", "Linear 30→20 + ELU\n历史隐变量 [B,20]", "net"),
    ]
    for key, label, kind in layers: g.node(key, label, kind)
    g.chain(*(x[0] for x in layers))
    return g


def low_unused():
    g = Figure("16_low_level_priv_and_critic", "底层模型还包含的网络：被加载，但高层控制路径不调用")
    for key, label, kind in [
        ("priv", "特权信息 [B,18]\n仅对应底层特权分支", "input"),
        ("p1", "priv_encoder\nLinear 18→64 + ELU", "net"),
        ("p2", "Linear 64→20 + ELU", "net"),
        ("po", "20维隐变量\nhist_encoding=False时替代历史编码\n高层实际使用hist_encoding=True", "fixed"),
        ("pc", "本体状态71 + 特权信息18\n[B,89]", "input"),
        ("cb", "critic_backbone\nLinear 89→128 + ELU", "net"),
        ("cl", "critic_leg_control_head\n128→128 + ELU\n128→128 + ELU\n128→1", "net"),
        ("ca", "critic_arm_control_head\n128→128 + ELU\n128→128 + ELU\n128→1", "net"),
        ("cv", "concat(V_leg,V_arm) → [B,2]\n高层训练不调用此Critic", "output"),
        ("std", "底层 std 参数：[1,18]\n随机act路径使用\n高层act_inference不使用", "fixed"),
    ]: g.node(key, label, kind)
    g.chain("priv", "p1", "p2", "po"); g.chain("pc", "cb")
    g.edge("cb", "cl"); g.edge("cb", "ca"); g.edge("cl", "cv"); g.edge("ca", "cv")
    return g


def execution():
    g = Figure("17_action_execution", "高层9维动作到机器人执行：神经网络与解析控制的分工")
    for key, label, kind in [
        ("a", "教师 / 学生高层动作 [B,9]", "input"),
        ("dp", "a[0:3] 位置增量\na[3:6] RPY增量", "input"),
        ("g", "a[6] 夹爪命令", "input"),
        ("cmd", "a[7] 前进速度\na[8] 偏航角速度", "input"),
        ("feedback", "当前机器人状态、历史与雅可比\n来自环境反馈", "fixed"),
        ("goal", "裁剪 / 缩放 + 累加\n更新末端目标位姿", "fixed"),
        ("lowobs", "底层观测71 + 历史10×71\n包含底盘命令、末端目标等", "op"),
        ("low", "预训练底层Actor\n历史编码 + 共享骨干 + 双头", "net"),
        ("legs", "腿部12维动作 → PD\nτ = Kp(q_target−q) − Kd·qdot", "fixed"),
        ("ik", "末端位姿误差 + DLS IK\n得到6个机械臂关节目标", "fixed"),
        ("grip", "夹爪关节目标", "fixed"),
        ("sim", "Isaac Gym 执行与仿真\n返回下一次教师/学生观测", "output"),
    ]: g.node(key, label, kind)
    for k in ("dp", "g", "cmd"): g.edge("a", k)
    g.chain("dp", "goal", "ik", "sim"); g.chain("g", "grip", "sim")
    g.edge("goal", "lowobs"); g.chain("cmd", "lowobs", "low", "legs", "sim")
    for k in ("lowobs", "ik", "legs"): g.edge("feedback", k)
    return g


if __name__ == "__main__":
    figures = [overview(), teacher_actor(), feature_encoder(), grasp_attention(), teacher_critic(), gaussian(), student(), cnn(), guided(), encoder_layer(), mha(), heads(), training(), low_actor(), low_history(), low_unused(), execution(), pose_prediction()]
    for figure in figures:
        figure.render(png=figure.name in {"02_teacher_actor", "05_teacher_critic", "18_teacher_pose_prediction"})
    print(f"Rendered {len(figures)} SVG figures in {OUT}")
