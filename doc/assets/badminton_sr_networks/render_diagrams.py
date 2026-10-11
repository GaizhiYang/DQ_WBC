"""Redraw the paper analysis figures: python /path/to/render_diagrams.py.

Only Graphviz's dot executable is required. Facts follow the supplied
badminton_sr_read.pdf; the last figure is explicitly a proposed DQ adaptation.
"""
import json
from pathlib import Path
import subprocess


OUT = Path(__file__).resolve().parent
COLORS = {
    "input": ("#eaf3ff", "#5487bd"),
    "net": ("#eaf7f0", "#49836a"),
    "model": ("#f2efff", "#8371ac"),
    "train": ("#fff0df", "#bd9148"),
    "loss": ("#fceceb", "#b87874"),
    "note": ("#f3f4f6", "#949ca6"),
}


class Figure:
    def __init__(self, name, title, direction="TB"):
        self.name = name
        self.lines = ["digraph G {",
            'graph [rankdir=%s, bgcolor="white", pad="0.3", nodesep="0.3", ranksep="0.45", '
            'splines=polyline, fontname="Noto Sans CJK SC", fontsize=20, labelloc=t, label=%s];'
            % (direction, json.dumps(title, ensure_ascii=False)),
            'node [shape=box, style="rounded,filled", fontname="Noto Sans CJK SC", '
            'fontsize=12, margin="0.18,0.12", penwidth=1.2];',
            'edge [color="#64748b", arrowsize=0.7, fontname="Noto Sans CJK SC", fontsize=10];']

    def node(self, key, label, kind="model"):
        fill, border = COLORS[kind]
        self.lines.append('%s [label=%s, fillcolor="%s", color="%s"];'
                          % (key, json.dumps(label, ensure_ascii=False), fill, border))

    def edge(self, a, b, label="", dashed=False):
        attrs = ["label=" + json.dumps(label, ensure_ascii=False)] if label else []
        if dashed:
            attrs.append("style=dashed")
        self.lines.append("%s -> %s%s;" % (a, b, " [" + ",".join(attrs) + "]" if attrs else ""))

    def chain(self, *keys):
        for a, b in zip(keys, keys[1:]):
            self.edge(a, b)

    def same_rank(self, *keys):
        self.lines.append("{rank=same; " + ";".join(keys) + ";}")

    def render(self):
        source = "\n".join(self.lines + ["}"])
        path = OUT / (self.name + ".dot")
        path.write_text(source, encoding="utf-8")
        for fmt in ("svg", "png"):
            # Keep SVG coordinates unscaled: Graphviz 2.43 can scale its root
            # group at higher DPI while leaving viewBox unchanged, clipping
            # the drawing. PNG still uses 150 DPI for readable raster output.
            dpi = 72 if fmt == "svg" else 150
            subprocess.run(["dot", "-T" + fmt, "-Gdpi=" + str(dpi), str(path), "-o",
                            str(OUT / (self.name + "." + fmt))], check=True)


def overview():
    g = Figure("01_asymmetric_training", "图 1｜原论文：带感知闭环的非对称 Actor–Critic 训练")
    for key, label, kind in [
        ("env", "4096 个并行仿真环境\nANYmal-D + DynaArm / 羽毛球轨迹 / 随机化", "train"),
        ("prop", "共享本体观测\n基座状态、关节状态、球场位置、上一动作", "input"),
        ("per", "回归感知模型 → 时间自适应 EKF → 拦截预测\n相机系目标观测、检测标志、拦截命令、倒计时", "model"),
        ("cat", "共享观测拼接\n注意：感知分支已经含测量误差", "model"),
        ("noise", "Actor 的附加观测噪声\n对应原 Fig. 6 的 +noise 节点", "model"),
        ("actor", "Actor MLP\ndA → 512 → 256 → 128\n部署可用的数值观测", "net"),
        ("act", "全身 18 个关节位置命令\n12 个腿部 + 6 个机械臂关节", "input"),
        ("enh", "Critic 增强状态\n无噪本体 / 末端状态 / 完美感知 / 随机化参数", "train"),
        ("future", "Critic 任务信息\n下一目标 / 发射倒计时 / 剩余次数 / 任务标志", "train"),
        ("critic", "Critic MLP\ndC → 512 → 256 → 128 → V(s)\n额外信息只用于价值估计", "net"),
        ("feedback", "仿真反馈\n指定时刻跟踪奖励 + 密集预测误差奖励\n运动正则化 + 电流约束", "loss"),
        ("rl", "N-P3O（约束 PPO）\n回报 / 优势 / 策略与价值优化", "loss"),
        ("note", "部署保留：Actor + EKF + 预测器\nCritic 与回归感知模拟器只用于训练", "note"),
    ]:
        g.node(key, label, kind)
    g.edge("env", "prop"); g.edge("env", "per")
    g.edge("prop", "cat"); g.edge("per", "cat")
    g.chain("cat", "noise", "actor", "act")
    g.edge("act", "env", "全身运动改变后续相机观测", True)
    g.edge("cat", "critic", "在最后 +noise 之前分支")
    g.edge("env", "enh"); g.edge("env", "future")
    g.edge("enh", "critic"); g.edge("future", "critic")
    g.edge("env", "feedback"); g.edge("feedback", "rl"); g.edge("critic", "rl")
    g.edge("rl", "actor", "更新策略参数", True)
    g.edge("rl", "critic", "更新价值参数", True)
    g.edge("actor", "note", dashed=True)
    g.same_rank("enh", "future", "prop", "per")
    return g


def actor():
    g = Figure("02_actor_network", "图 2｜Actor 的观测分组、MLP 和动作接口")
    for key, label, kind in [
        ("base", "基座 / 定位\nvb(3)、ωb(3)、gb(3)\n球场朝向 h（编码未披露）、基座 xy(2)", "input"),
        ("joint", "本体 / 动作记忆\nq−qdefault(18)、qdot(18)、a_prev(18)\n18 维由控制关节数对应推导", "input"),
        ("target", "感知 / 任务命令\n相机系目标位置(3)、检测标志 D(1)\n拦截 cmd（编码未披露）、距拦截时间 t(1)", "input"),
        ("cat", "拼接并使用部署条件下的带噪观测\noA：[B,dA]；论文未给出总 dA", "model"),
        ("fc1", "Linear(dA,512)\nELU\n[B,512]", "net"),
        ("fc2", "Linear(512,256)\nELU\n[B,256]", "net"),
        ("fc3", "Linear(256,128)\nELU\n[B,128]", "net"),
        ("out", "策略输出 πθ(a|oA)\n动作命令维数 18\n状态相关标准差 σ(oA)", "net"),
        ("command", "关节位置命令 qdes：[B,18]\n100 Hz；腿与臂统一控制", "input"),
        ("note", "论文给出三层隐藏宽度与 ELU\n未公布 μ/σ 头各自层数、共享方式及动作缩放\n图中输出框表示功能，不补造尾部网络", "note"),
    ]:
        g.node(key, label, kind)
    g.edge("base", "cat"); g.edge("joint", "cat"); g.edge("target", "cat")
    g.chain("cat", "fc1", "fc2", "fc3", "out", "command")
    g.edge("out", "note", dashed=True)
    g.same_rank("base", "joint", "target")
    return g


def critic():
    g = Figure("03_critic_network", "图 3｜Critic：增强状态 + 多目标任务信息")
    for key, label, kind in [
        ("shared", "共享观测旁路\n最后 +noise 节点之前的本体 / 感知 / 命令\n共享感知通路本身已包含模拟测量误差", "input"),
        ("sensor", "增强传感与状态信息\n无噪本体与理想羽毛球状态\n末端位置 / 速度 / 拍面法向\n附加基座质量 / 摩擦 / 拦截预测误差 ε / FOV", "train"),
        ("mdp", "任务 MDP 信息（不进入 Actor）\n当前及下一羽毛球发射倒计时\n下一拦截位置 / 速度 / 朝向命令\n剩余目标数量、当前及下一目标/站立标志", "train"),
        ("cat", "拼接 Critic 输入 oC：[B,dC]\n论文未给出总 dC 与部分编码维数", "model"),
        ("fc1", "Linear(dC,512) → ELU\n[B,512]", "net"),
        ("fc2", "Linear(512,256) → ELU\n[B,256]", "net"),
        ("fc3", "Linear(256,128) → ELU\n[B,128]", "net"),
        ("value", "状态价值 Vφ(s)\n[B,1]", "net"),
        ("adv", "回报 / 优势估计\n任务反馈进入 N-P3O 优化", "loss"),
        ("note", "未来任务信息用于区分相同观测下的不同回报\n部署时移除整个 Critic\n不添加论文未披露的独立 cost critic 网络", "note"),
    ]:
        g.node(key, label, kind)
    for key in ("shared", "sensor", "mdp"):
        g.edge(key, "cat")
    g.chain("cat", "fc1", "fc2", "fc3", "value", "adv")
    g.edge("mdp", "note", dashed=True)
    g.same_rank("shared", "sensor", "mdp")
    return g


def sim_perception():
    g = Figure("04_simulation_perception", "图 4｜仿真感知：真实误差回归 + 有记忆滤波 + 任务预测")
    for key, label, kind in [
        ("offline", "离线真实数据\n相机观察已知位置目标、动捕记录相机运动", "train"),
        ("fit", "线性回归：检测概率与位置测量误差\n依赖距离、角速度；论文未给出回归系数", "model"),
        ("gt", "仿真真值\n羽毛球位置 / 速度 / 预生成轨迹", "train"),
        ("camera", "机器人运动与相机位姿\n计算距离、相对运动、FOV\n水平 50° / 垂直 74°", "input"),
        ("measure", "FOV 门控 + 概率检测 + 带噪位置\nD、最新相机系位置/方向\n论文未给出完整误差分布", "model"),
        ("delay", "模拟异步传输 / 延迟\nFig.6：U(0.15,0.25) s\n滤波前完成 map 坐标与时间对齐", "model"),
        ("ekf", "时间自适应 EKF\n当前测量 + 上次滤波记忆 + 时间信息\n估计 p̂、v̂；论文未给出精确增广状态", "model"),
        ("pred", "动力学拦截预测\n训练采用真值轨迹附近的线性化偏移（S9）\n避免每步完整滚动预测", "model"),
        ("cmd", "Actor 的感知 / 任务输入\n最新相机系测量 + D、拦截命令与剩余时间\n目标位置误差以基座系表达\n目标速度 / 拍面法向按规则配置", "input"),
        ("hold", "漏检后保持最后预测\n最多 2 s；检测标志随时更新", "model"),
        ("actor", "Actor MLP → 全身运动", "net"),
    ]:
        g.node(key, label, kind)
    g.chain("offline", "fit", "measure", "delay", "ekf", "pred", "cmd", "actor")
    g.edge("gt", "measure"); g.edge("camera", "measure")
    g.edge("gt", "pred", "真值轨迹供训练近似")
    g.edge("delay", "cmd", "最新已到达测量 + D")
    g.edge("pred", "hold"); g.edge("hold", "cmd", "检测丢失时")
    g.edge("ekf", "ekf", "递归状态")
    g.edge("actor", "camera", "动作反馈改变感知质量", True)
    g.same_rank("gt", "camera", "fit")
    return g


def deployment():
    g = Figure("05_deployment_multirate", "图 5｜部署数据流：60 Hz 感知 + 100 Hz 控制 + 400 Hz 估计")
    for key, label, kind in [
        ("cam", "ZED X 双目相机\n全局快门 / 图像时间戳 / 60 Hz", "input"),
        ("hsv", "HSV 橙色阈值检测\nH<5 或 H>176、S>60、V>160\n不是神经网络检测器", "model"),
        ("stereo", "双目 3D 测量\n图像检测 → 相机系位置", "model"),
        ("local", "机器人状态估计：400 Hz\nMSF + CompSLAM / 位姿与本体状态", "model"),
        ("map", "坐标与时间对齐\n目标测量转换到稳定 map frame", "model"),
        ("ekf", "与训练一致的时间自适应 EKF\n感知估计链异步 60 Hz", "model"),
        ("pred", "解析羽毛球轨迹预测\n计算候选拦截点与命令时间", "model"),
        ("rule", "拦截规则\n轨迹通过网高 / 落地点区域\n配置挥拍高度、速度、拍面方向", "model"),
        ("obs", "Actor 观测\n最新已到达感知、检测标志、拦截命令、剩余时间\n本体状态、球场位置、上一动作\n60 Hz 测量保持到下一帧，供 100 Hz 策略使用", "input"),
        ("actor", "与训练相同的 Actor MLP\n100 Hz", "net"),
        ("drive", "18 个全身关节位置命令\n机器人驱动与物理运动", "input"),
        ("hold", "失去目标时\n最后拦截预测保持 ≤2 s", "model"),
        ("note", "部署无需 Critic 或回归噪声模拟器\n真实测量自然带有误差；EKF 与 Actor 连续运行", "note"),
    ]:
        g.node(key, label, kind)
    g.chain("cam", "hsv", "stereo", "map", "ekf", "pred", "rule", "obs", "actor", "drive")
    g.edge("local", "map"); g.edge("local", "obs"); g.edge("stereo", "obs", "最新测量 / D")
    g.edge("pred", "hold"); g.edge("hold", "obs", "漏检时")
    g.edge("drive", "cam", "改变视野 / 成像质量", True)
    g.edge("actor", "note", dashed=True)
    return g


def training():
    g = Figure("06_training_pipeline", "图 6｜训练准备、连续目标任务与约束策略优化")
    for key, label, kind in [
        ("data", "实机感知采集\n拟合轻量误差 / 检测概率模型", "train"),
        ("sysid", "系统辨识与执行器建模\n臂：CMA-ES、摩擦/阻尼/armature、皮带传动\n腿：SEA 执行器模型", "train"),
        ("pool", "预生成羽毛球轨迹池\n随机初始状态、空气动力学参数", "train"),
        ("env", "4096 环境 / 0.01 s 控制周期\n每回合 6 个目标，间隔 2 s\n10% 站立环境；域随机化与随机推扰", "train"),
        ("policy", "Actor 选择全身动作\nCritic 估计回报；未来任务信息只供 Critic", "net"),
        ("sparse", "指定挥拍时刻的单步跟踪奖励\n位置 6400 / 拍面朝向 1200 / 速度 1200", "loss"),
        ("dense", "每步感知预测误差奖励（权重 3）\nrε = 1 / (1 + ||真实拦截点−预测点||²)\n无目标时站立奖励 + 运动正则化", "loss"),
        ("cost", "硬件约束信号\n机械臂总电流 |I_total| < 8 A\n关节力矩/速度限制仍含软惩罚", "loss"),
        ("opt", "N-P3O + AdamW\nγ=0.995、GAE λ=0.975、KL 目标=0.01\n更新 Actor 和 Critic", "loss"),
        ("deploy", "部署 Actor + EKF + 预测器\n不经过教师→学生行为蒸馏", "net"),
    ]:
        g.node(key, label, kind)
    for key in ("data", "sysid", "pool"):
        g.edge(key, "env")
    g.edge("env", "policy")
    for key in ("sparse", "dense", "cost"):
        g.edge("env", key)
        g.edge(key, "opt")
    g.edge("policy", "opt"); g.edge("opt", "policy", "参数更新", True)
    g.edge("policy", "env", "环境交互", True)
    g.edge("policy", "deploy", "训练完成")
    g.same_rank("data", "sysid", "pool")
    g.same_rank("sparse", "dense", "cost")
    return g


def active_perception():
    g = Figure("07_active_perception_loop", "图 7｜主动感知为何出现：动作影响最终预测误差", "LR")
    for key, label, kind in [
        ("action", "Actor 的全身动作\n移动、转身、机身俯仰、挥臂", "net"),
        ("view", "相机位姿 / 目标距离 / 角运动\n视野和运动模糊发生变化", "input"),
        ("sensor", "检测是否成功 + 位置误差\n真实数据拟合的运动相关感知模型", "model"),
        ("belief", "多次测量的 EKF 累积\n预测最终拦截点 p̂T", "model"),
        ("task", "任务跟踪与预测误差反馈\n对比 p̂T 与真值 pT\n通过 N-P3O 学习权衡感知与运动", "loss"),
        ("note", "奖励不是单纯的 FOV/目标可见面积\n不是 EKF 协方差奖励\n也不要求穿过感知模块做可微反向传播", "note"),
    ]:
        g.node(key, label, kind)
    g.chain("action", "view", "sensor", "belief", "task")
    g.edge("task", "action", "强化学习策略更新", True)
    g.edge("task", "note", dashed=True)
    return g


def dq_proposal():
    g = Figure("08_dq_adaptation_proposal", "图 8｜DQ-NET 迁移建议（提议，非论文原网络 / 非本次代码实现）")
    for key, label, kind in [
        ("camera", "DQ 双相机的部署可得测量\nmask/depth 或已有视觉编码\n有效标志、时间戳", "input"),
        ("filter", "任务适配的持久估计器\n目标 6D 位姿/运动、漏检记忆\n训练部署共用；采用抓取物体运动模型", "model"),
        ("grasp", "候选抓取 / 接触目标\n使用估计物体位姿变换\n先验模型只在部署可得时使用", "model"),
        ("actor", "DQ Actor\n部署本体信息 + 感知估计 / 视觉特征\n保留当前 9 维高层动作接口", "net"),
        ("control", "现有低层策略 / IK / 夹爪控制\n不在第一步改成全身 18 关节联合学习", "model"),
        ("env", "DQ 抓取仿真与真实执行\n控制动作反馈改变感知质量", "train"),
        ("critic", "DQ Critic\nGT 目标位姿/速度 + 本体状态\n任务阶段 / 保持计数 / 剩余时间\n运动模式与域随机化信息", "net"),
        ("loss", "原抓取奖励 + 预测误差消融\n用真值接触/抓取目标评估预测误差\n不将可见表面均值直接当作真实中心", "loss"),
        ("rl", "直接非对称 RL\n先验证信息与训练闭环，再逐项比较", "loss"),
    ]:
        g.node(key, label, kind)
    g.chain("camera", "filter", "grasp", "actor", "control", "env")
    g.edge("filter", "actor"); g.edge("env", "camera", "运动—感知闭环", True)
    g.edge("env", "critic"); g.edge("env", "loss")
    g.edge("filter", "loss", "估计误差")
    g.edge("critic", "rl"); g.edge("loss", "rl")
    g.edge("rl", "actor", "策略更新", True)
    g.edge("rl", "critic", "价值更新", True)
    return g


if __name__ == "__main__":
    for builder in (overview, actor, critic, sim_perception, deployment,
                    training, active_perception, dq_proposal):
        builder().render()
    print("Rendered 8 diagrams as DOT, SVG and PNG in", OUT)
