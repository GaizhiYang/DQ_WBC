"""Reproduce the implemented M0/M1/M2 architecture diagrams with Graphviz."""
from pathlib import Path
import html
import json
import shutil
import subprocess

OUT = Path(__file__).resolve().parent
FONT = "Noto Sans CJK SC"
COLORS = {
    "sensor": ("#E9F2FD", "#5584B0"),
    "reuse": ("#EAF5EC", "#568362"),
    "new": ("#F1ECFC", "#8270B0"),
    "op": ("#FFF4DF", "#BA924B"),
    "gt": ("#FCEDEC", "#B47470"),
    "fixed": ("#F0F2F4", "#85909B"),
}


class Figure:
    def __init__(self, name, title, subtitle, direction="LR"):
        self.name = name
        label = '<<B>{}</B><BR/><FONT POINT-SIZE="13">{}</FONT>>'.format(
            html.escape(title), html.escape(subtitle).replace("\n", "<BR/>"))
        self.lines = [
            "digraph G {",
            'graph [rankdir=%s, compound=true, newrank=true, bgcolor="white", '
            'pad=".30", nodesep=".28", ranksep=".42", splines=polyline, '
            'fontname="%s", fontsize=23, labelloc=t, labeljust=l, label=%s];'
            % (direction, FONT, label),
            'node [shape=box, style="rounded,filled", fontname="%s", fontsize=13, '
            'margin=".15,.11", penwidth=1.35];' % FONT,
            'edge [fontname="%s", fontsize=10, color="#61768E", arrowsize=.72, penwidth=1.25];' % FONT,
        ]

    def node(self, key, label, kind="reuse", dashed=False):
        fill, stroke = COLORS[kind]
        attrs = dict(label=label, fillcolor=fill, color=stroke)
        if dashed:
            attrs["style"] = "rounded,filled,dashed"
        self.lines.append(key + " [" + ",".join(k + "=" + json.dumps(v, ensure_ascii=False) for k, v in attrs.items()) + "];")

    def edge(self, a, b, label="", dashed=False, **attrs):
        if label:
            attrs["label"] = label
        if dashed:
            attrs["style"] = "dashed"
        self.lines.append(a + " -> " + b + (" [" + ",".join(k + "=" + json.dumps(v, ensure_ascii=False) for k, v in attrs.items()) + "]" if attrs else "") + ";")

    def chain(self, *keys):
        for a, b in zip(keys, keys[1:]):
            self.edge(a, b)

    def rank(self, *keys):
        self.lines.append("{rank=same; " + "; ".join(keys) + ";}")

    def cluster(self, key, title, color):
        self.lines.append('subgraph cluster_%s {label=%s; color="%s"; penwidth=1.5; style="rounded"; fontname="%s"; fontsize=16; margin=16;' % (key, json.dumps(title, ensure_ascii=False), color, FONT))

    def end(self):
        self.lines.append("}")

    def render(self):
        source = "\n".join(self.lines + ["}"]) + "\n"
        (OUT / (self.name + ".dot")).write_text(source, encoding="utf-8")
        for fmt in ("svg", "png", "pdf"):
            args = ["dot", "-T" + fmt]
            if fmt == "png":
                args.append("-Gdpi=150")
            subprocess.run(args + ["-o", str(OUT / (self.name + "." + fmt))], input=source, text=True, check=True)


def final_architecture():
    g = Figure("01_final_architecture", "已实现｜DQ 非对称 Actor–Critic：M0 / M1 / M2",
               "默认 M2 = M1 + 传感器 belief；M0 / M1 可切换。蓝：部署观测　绿：继承网络　紫：新增模块　红：训练专用\n"
               "训练打包 [priv1276, images62208, task5?, belief16?]：M0 = 63484，M1 = 63489，M2 = 63505")
    g.cluster("actor", "Actor｜所有模式均只使用部署可得输入；与 Critic 参数独立", "#89A9C7")
    for args in [
        ("images", "双相机 mask + depth\n每路三帧历史\n[B,12,54,96] = 62208", "sensor"),
        ("p", "归一化本体 / 动作历史\np [B,61]\n机器人52 + 上一动作9", "sensor"),
        ("belief", "M2｜传感器滤波 belief\n[B,16]，固定单位 / 截断\n感知流程见图3", "sensor"),
        ("cnn", "共享 CNN\n每帧 2×54×96 → 64\n两相机、三帧共用", "reuse"),
        ("ta", "腕相机 GuidedTransformer\n3视觉 token + 1状态 token\n2层 / 2头 / d=64\nFF=2048，dropout=0", "reuse"),
        ("tb", "底盘 GuidedTransformer\n3视觉 token + 1状态 token\n2层 / 2头 / d=64\nFF=2048，dropout=0", "reuse"),
        ("visual", "各路池化 + Linear 64→64\n拼接 z [B,128]\n继承 visual_adapter\n128→512，无 bias", "reuse"),
        ("pa", "proprio_adapter\n61→512，含 bias\n继承原首层对应61列", "new"),
        ("ba", "M2｜belief_adapter\n16→512，无 bias\n新增时权重初始化为0", "new"),
        ("sum", "512维逐元素相加\n→ ELU", "op"),
        ("head", "继承 Actor 下游 MLP\n512→256 ELU\n256→128 ELU → 9\n输出动作均值 μ", "reuse"),
        ("normal", "继承可学习 log_std [9]\n对角高斯策略\n训练采样 / 评估取 μ", "reuse"),
        ("control", "9维高层动作\n固定原 LLP + IK + 夹爪\n保持 DQ 执行接口", "fixed"),
    ]:
        g.node(*args)
    g.chain("images", "cnn")
    g.edge("cnn", "ta"); g.edge("cnn", "tb")
    g.edge("p", "ta", "61→64 状态 token"); g.edge("p", "tb")
    g.chain("p", "pa", "sum")
    g.chain("belief", "ba", "sum")
    g.edge("ta", "visual"); g.edge("tb", "visual")
    g.chain("visual", "sum", "head", "normal", "control")
    g.rank("images", "p", "belief"); g.rank("ta", "tb")
    g.rank("visual", "pa", "ba")
    g.end()

    g.cluster("critic", "Critic｜仅训练；保留自身 GT 编码器与 GFM；不读取图像", "#CB9691")
    for args in [
        ("gt", "归一化特权状态\npriv [B,1276]\n物体、机器人、抓取候选", "gt"),
        ("cgfm", "原 Critic 编码器 + GFM\n物体特征1024→512→128\n128 + 位姿6 → Query\n30×6候选 → 抓取表示6", "reuse"),
        ("cfused", "原206维融合\n状态 / 动作72 + 编码128\n+ 抓取表示6\nLinear 206→512", "reuse"),
        ("task", "M1 / M2｜任务状态 [B,5]\n剩余时间、最近距离、最高高度\n持物计数比例、当前举起标志\n固定缩放 / 截断", "gt"),
        ("ctask", "M1 / M2｜task_adapter\n5→512，无 bias\n新增时权重初始化为0", "new"),
        ("cbelief", "M2｜belief_adapter\n同一传感器 belief16 → 512\n无 bias，新增时权重初始化为0", "new"),
        ("csum", "512维逐元素相加\n→ ELU", "op"),
        ("chead", "继承 Critic 下游 MLP\n512→256 ELU\n256→128 ELU → 1", "reuse"),
        ("value", "V(priv, task?, belief?)\nGAE / PPO value loss", "op"),
    ]:
        g.node(*args)
    g.chain("gt", "cgfm", "cfused", "csum", "chead", "value")
    g.edge("gt", "cfused", "状态 / 动作72")
    g.chain("task", "ctask", "csum")
    g.edge("cbelief", "csum")
    g.rank("gt", "task"); g.rank("cfused", "ctask", "cbelief")
    g.end()
    g.rank("images", "p", "belief", "gt", "task")
    g.rank("ta", "tb", "cgfm"); g.rank("visual", "pa", "ba", "cfused", "ctask", "cbelief")
    g.rank("sum", "csum"); g.rank("head", "chead"); g.rank("normal", "value")
    # Declaring the independent Critic component first keeps Actor on top in
    # dot's left-to-right component ordering without a misleading data edge.
    a = next(i for i, line in enumerate(g.lines) if line.startswith("subgraph cluster_actor "))
    ae = next(i for i in range(a, len(g.lines)) if g.lines[i] == "}") + 1
    c = next(i for i, line in enumerate(g.lines) if line.startswith("subgraph cluster_critic "))
    ce = next(i for i in range(c, len(g.lines)) if g.lines[i] == "}") + 1
    g.lines = g.lines[:a] + g.lines[c:ce] + g.lines[a:ae] + g.lines[ce:]
    return g


def transition():
    g = Figure("02_weight_transition", "已实现｜教师权重直接迁移 → 非对称 PPO",
               "从第一轮 rollout 起，Actor 就只接收部署输入。新增 adapter 为零初始化；已有视觉权重完整继承。")
    g.cluster("migration", "权重迁移｜源：TeacherVisionPolicy / 原教师 Critic", "#A3BDA9")
    for args in [
        ("first", "教师 Actor 首层\nLinear 206→512\n206维融合输入", "reuse"),
        ("select", "选取首层的本体列\nkeep = [6:58] + [63:72]\n并复制完整 bias", "new"),
        ("pa", "proprio_adapter\n61→512\n只有部署本体 / 历史状态", "new"),
        ("vision", "教师视觉分支\n共享 CNN + 两路 Transformer\n两路投影 + visual_adapter", "reuse"),
        ("visualcopy", "原样复制全部视觉权重\n保留已训练的 128→512\nvisual_adapter", "reuse"),
        ("downstream", "教师 Actor 下游与分布\n512→256→128→9\nlog_std [9]", "reuse"),
        ("headcopy", "原样复制下游 MLP\n和可学习 log_std", "reuse"),
        ("newbelief", "M2 新增 Actor belief16→512\n无 bias，权重0\n开始训练后可学习", "new"),
        ("actor", "纯视觉 / 本体 Actor\nM2 启用 belief16\n保持9维动作接口", "sensor"),
        ("valueold", "原教师 Critic 全部参数\n独立特征编码器 + GFM\n206→512→256→128→1", "reuse"),
        ("valuecopy", "保留 Critic 的全部原参数\nM1：新增 task5→512\nM2：再加 belief16→512\n新 adapter 无 bias、权重0", "new"),
        ("critic", "独立特权 Critic\npriv1276 + task5? + belief16?\nGFM 仅在 Critic 内运行", "gt"),
    ]:
        g.node(*args)
    g.chain("first", "select", "pa", "actor")
    g.chain("vision", "visualcopy", "actor")
    g.chain("downstream", "headcopy", "actor")
    g.edge("newbelief", "actor")
    g.chain("valueold", "valuecopy", "critic")
    g.rank("first", "vision", "downstream", "valueold")
    g.rank("select", "visualcopy", "headcopy", "valuecopy")
    g.rank("pa", "newbelief"); g.rank("actor", "critic")
    g.end()
    g.cluster("training", "直接 PPO｜模式可切换 M0 / M1 / M2（默认）", "#B2A4D2")
    for args in [
        ("ppo", "完整 rollout → GAE → PPO\n更新 Actor 和 Critic\n冻结原低层控制策略\n保持原任务奖励 / 课程", "op"),
        ("evaluate", "定期独立评估\nActor 取均值动作\n按任务成功率选择 checkpoint", "fixed"),
        ("export", "部署导出 v2\n纯 Actor + 61维归一化统计\nfeature flags + belief约定\nM2：图像62208 + raw61 + belief16", "sensor"),
    ]:
        g.node(*args)
    g.chain("ppo", "evaluate", "export")
    g.end()
    g.edge("actor", "ppo", "动作 / log_prob")
    g.edge("critic", "ppo", "V / value loss")
    g.node("stats", "迁移与恢复边界\n初始化：恢复原归一化统计，新建 PPO optimizer\n恢复训练：严格读取相同模式的 v2 checkpoint\n部署输入无需1276维 GT，也无需任务记忆5", "fixed")
    g.edge("stats", "ppo", "冻结原输入 / 价值统计", dashed=True)
    g.rank("stats", "first", "vision", "downstream", "valueold")
    return g


def training_loop():
    g = Figure("03_training_loop", "已实现｜M2 传感器感知与 PPO 训练闭环",
               "M2 默认包含 M1。感知估计来自相机帧、标定、采样时间与机器人位姿；GT 只流向 Critic 和训练奖励。\n"
               "每轮流程：传感器观测 → 执行动作 / 收集 rollout → PPO 更新 → 下一轮观测。")
    g.cluster("perception", "可部署传感器前端｜M2 启用；M0 / M1 直接沿用原图像历史", "#89A9C7")
    for args in [
        ("sensor", "底盘 + 腕部双相机\nmask / 正向深度（m）\n内参 K、采样时刻 T_world_camera\n两相机 timestamps、当前机器人位姿", "sensor"),
        ("corrupt", "训练扰动：动作影响视角\n距离 / 角速率相关噪声与漏检\n连续丢帧 + 两相机独立延迟\n部署关闭合成扰动，保留真实时间戳", "new"),
        ("history", "交付帧 → 三帧历史\nmask / depth，12×54×96\n无新帧 / 整帧丢失时填零", "sensor"),
        ("measurement", "mask + 有效 depth 反投影\n采样时刻相机位姿 → 世界系\n估计可见表面参考点", "sensor"),
        ("kf", "常速度 Kalman Filter\n按 capture time 排序\n固定滞后窗口回放更新\n漏观测时仅预测", "new"),
        ("belief", "当前 base 系 belief16\n位置3 / 绝对速度3 / 位置std3\n可见2 / 观测年龄2 / 初始化1\n预测年龄1 / 有效深度比例1", "sensor"),
    ]:
        g.node(*args)
    g.chain("sensor", "corrupt", "history")
    g.chain("corrupt", "measurement", "kf", "belief")
    g.end()

    g.cluster("learning", "独立非对称网络｜Actor 不读取目标 GT；M1 任务记忆只进入 Critic", "#A3BDA9")
    for args in [
        ("actor", "视觉 Actor\n图像历史 + 本体61 + belief16\nCNN + 双 GuidedTransformer\n输出 μ / log_std / action", "reuse"),
        ("control", "固定 LLP + IK + 夹爪\n执行9维高层动作\n改变视角与下一时刻观测", "fixed"),
        ("gt", "仅训练信息\npriv1276 + M1 task5\n物体 GT 刚体位姿 / 速度", "gt"),
        ("critic", "独立 Critic\npriv1276 + task5 + belief16\n输出状态价值 V", "reuse"),
        ("reward", "原 DQ 任务奖励 / 终止\n+ M2 预测精度奖励\n评估成绩只用原任务表现", "gt"),
        ("buffer", "Rollout buffer\n观测 / 动作 / old log_prob\nreward / done / old value", "op"),
        ("ppo", "GAE → PPO 更新\nActor：policy + entropy\nCritic：value loss\n终止按任务截止期限处理", "new"),
    ]:
        g.node(*args)
    g.edge("actor", "control")
    g.edge("gt", "critic"); g.edge("gt", "reward")
    g.edge("actor", "buffer", "动作 + log_prob")
    g.edge("critic", "buffer", "V")
    g.chain("reward", "buffer", "ppo")
    g.end()
    g.edge("history", "actor", "图像历史")
    g.edge("belief", "actor", "belief16")
    g.edge("belief", "critic", "相同 belief16")

    g.cluster("reward_reference", "M2 奖励参考｜仅训练；不反馈给传感器估计器", "#CB9691")
    for args in [
        ("clean", "干净可见表面参考点\n由未扰动的渲染帧得到\n与目标中心区分", "gt"),
        ("reference", "GT 刚体运动传播参考点\n构造0.15 s短时预测参考\n只用于计算误差", "gt"),
        ("error", "KF 短时预测 vs 参考\n误差 → 指数精度奖励\n可通过权重关闭", "op"),
    ]:
        g.node(*args)
    g.chain("clean", "reference", "error")
    g.end()
    g.edge("kf", "error", "估计的世界位置 / 速度", constraint="false")
    g.edge("error", "reward", "训练时追加", color="#B47470")
    g.rank("sensor", "clean")
    g.rank("history", "measurement", "reference")
    g.rank("belief", "error", "gt")
    g.rank("actor", "critic", "reward")
    return g


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    figures = [final_architecture(), transition(), training_loop()]
    for figure in figures:
        figure.render()
    if shutil.which("pdfunite"):
        subprocess.run(["pdfunite", *[str(OUT / (f.name + ".pdf")) for f in figures],
                        str(OUT / "dq_asymmetric_teacher_figures.pdf")], check=True)
    else:
        raise RuntimeError("pdfunite is required to generate the combined review artifact")
    print("Rendered", len(figures), "figures to", OUT)


if __name__ == "__main__":
    main()
