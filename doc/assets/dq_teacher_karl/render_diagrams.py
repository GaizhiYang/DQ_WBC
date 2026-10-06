"""Render the implemented KARL teacher with Graphviz; SVG + PNG + DOT."""
import json
from pathlib import Path
import subprocess

OUT = Path(__file__).resolve().parent
COLORS = {"input": "#e7f1ff", "learn": "#e7f5ec", "fixed": "#eef0f4",
          "select": "#ede8fa", "output": "#fff0d9", "loss": "#ffe9e5"}


class Figure:
    def __init__(self, filename, title):
        self.filename = filename
        self.lines = ["digraph G {", "graph [rankdir=TB, bgcolor=white, pad=0.3, nodesep=0.35, ranksep=0.5, splines=polyline, fontname=\"Noto Sans CJK SC\", fontsize=21, labelloc=t, label=" + json.dumps(title, ensure_ascii=False) + "];",
                      'node [shape=box, style="rounded,filled", fontname="Noto Sans CJK SC", fontsize=13, margin="0.18,0.12", color="#678099"];',
                      'edge [color="#65758B", arrowsize=0.75, fontname="Noto Sans CJK SC", fontsize=11];']

    def node(self, key, label, kind="learn"):
        self.lines.append(key + " [label=" + json.dumps(label, ensure_ascii=False) + ', fillcolor="' + COLORS[kind] + '"];')

    def edge(self, a, b, label="", dashed=False):
        attrs = "label=" + json.dumps(label, ensure_ascii=False)
        if dashed:
            attrs += ", style=dashed, constraint=false"
        self.lines.append(a + " -> " + b + " [" + attrs + "];")

    def chain(self, *keys):
        for a, b in zip(keys, keys[1:]):
            self.edge(a, b)

    def render(self):
        source = "\n".join(self.lines + ["}"])
        (OUT / (self.filename + ".dot")).write_text(source + "\n")
        for extension in ("svg", "png"):
            subprocess.run(["dot", "-T" + extension, "-o", str(OUT / (self.filename + "." + extension))],
                           input=source, text=True, check=True)


def overview():
    f = Figure("01_training_overview", "DQ-NET 教师：用 KARL 几何筛选替换 GFM")
    f.node("bank", "离线资产（保持原 DQ）\n每物体 30 个 Contact-GraspNet 候选\n物体系固定变换 + 1024 维点云特征", "fixed")
    f.node("sim", "Isaac Gym 并行环境\n物体位姿 / 实际末端位姿 / 机器人状态", "input")
    f.node("transform", "原候选位姿变换：物体系 → 世界系 → 机器人基座系\n30 × (xyz + RPY)，位置 z 仍沿用原裁剪\nKARL 分支输出真实 RPY（修正旧 YPR 顺序）", "fixed")
    f.node("select", "KARL 筛选器（无参数、无梯度）\nSO(3) 姿态距离 → 最小代价候选\n改善超过 30° 才切换；每环境保存一个索引", "select")
    f.node("pack", "打包观测 [B,1102]\n点云 1024 + 状态 63 + 所选抓取 6 + 上一动作 9", "input")
    f.node("norm", "RunningStandardScaler 1102 → 1102\n筛选已在未归一化几何量上完成", "fixed")
    f.node("actor", "Actor（独立参数）\n点云编码 1024→512→128\n拼接 206 → MLP 512→256→128 → μ 9")
    f.node("critic", "Critic（独立参数）\n点云编码 1024→512→128\n拼接 206 → MLP 512→256→128 → V 1")
    f.node("actions", "对角高斯采样：9 维高层动作\n末端位置增量 3 + 姿态增量 3\n夹爪 1 + 底盘速度命令 2", "output")
    f.node("control", "原高层动作处理 + 冻结低层策略 / 臂 IK-PD\n物理仿真、奖励与终止规则保持原设置", "fixed")
    f.node("ppo", "原 PPO / GAE\n24 步采样 × B 环境\n5 epochs × 6 minibatches", "loss")
    f.lines += ["{rank=same; bank; sim;}", "{rank=same; actor; critic;}"]
    f.edge("bank", "transform"); f.edge("sim", "transform"); f.edge("sim", "select", "实际末端姿态")
    f.chain("transform", "select", "pack", "norm")
    f.edge("bank", "pack", "点云特征"); f.edge("sim", "pack", "状态 / 上一动作")
    f.edge("norm", "actor"); f.edge("norm", "critic")
    f.chain("actor", "actions", "control"); f.edge("control", "sim", "下一步", True)
    f.edge("pack", "ppo", "存储选择后的原始观测"); f.edge("critic", "ppo", "V / GAE")
    f.edge("actions", "ppo", "a / log π / r / done")
    f.edge("ppo", "actor", "策略梯度", True); f.edge("ppo", "critic", "价值梯度", True)
    f.render()


def networks():
    f = Figure("02_actor_critic", "Actor / Critic 详细结构（--roboinfo；B 为当前批量）")
    f.node("obs", "已归一化观测 [B,1102]\n[特征1024 | 状态63 | 目标抓取6 | 上一动作9]", "input")
    f.node("feat", "物体点云特征 [B,1024]", "input")
    f.node("context", "共享输入值，网络参数不共享\n状态 [B,63] + 上一动作 [B,9] + 抓取 [B,6]", "input")
    f.edge("obs", "feat"); f.edge("obs", "context")
    for prefix, title, output in (("a", "Actor", 9), ("v", "Critic", 1)):
        f.node(prefix+"enc1", title + " 点云编码\nLinear 1024→512 + ELU\n524,800 参数")
        f.node(prefix+"enc2", "Linear 512→128（无末端激活）\n65,664 参数")
        f.node(prefix+"cat", "Concat [状态63, 上一动作9, 编码128, 抓取6]\n[B,206]", "select")
        f.node(prefix+"h1", "Linear 206→512 + ELU\n105,984 参数")
        f.node(prefix+"h2", "Linear 512→256 + ELU\n131,328 参数")
        f.node(prefix+"h3", "Linear 256→128 + ELU\n32,896 参数")
        f.node(prefix+"out", "Linear 128→" + str(output) + "\n" + ("μ [B,9]；1,161 参数" if prefix == "a" else "V(s) [B,1]；129 参数"), "output")
        f.edge("feat", prefix+"enc1")
        f.chain(prefix+"enc1", prefix+"enc2", prefix+"cat", prefix+"h1", prefix+"h2", prefix+"h3", prefix+"out")
        f.edge("context", prefix+"cat")
    f.node("std", "Actor 独立 log_std [9]\n9 个可学习参数；截断 [-20,2]")
    f.node("gauss", "π(a|o) = Normal(μ, diag(exp(log_std)²))\n采样 a [B,9] / log_prob [B,1]\n评估使用确定性动作；可沿用 --use_tanh", "output")
    f.node("value", "价值损失输入 / GAE 自举\n沿用原 Value 的未使用 log_std [9]\n不参与 Critic 前向与梯度", "fixed")
    f.edge("aout", "gauss"); f.edge("std", "gauss"); f.edge("vout", "value")
    f.render()


def selector():
    f = Figure("03_grasp_selection", "KARL 候选筛选：每环境独立、只在采样阶段运行")
    f.node("grasp", "原 DQ 候选 [B,30,6]\n统一机器人基座系 / 米与弧度\n候选顺序在 episode 内保持不变", "input")
    f.node("ee", "实际末端位姿 [B,6]\n使用实际姿态，不是当前末端命令", "input")
    f.node("quat", "RPY → xyzw 四元数\nθᵢ = 2 acos(clamp(|qᵢ · qₑₑ|, 0, 1))\n处理 q 与 −q 等价、跨越 ±π", "select")
    f.node("cost", "默认 Jᵢ = θᵢ\n可选 --karl_orientation_preference karl：Jᵢ = θᵢ + cᵢ\ncᵢ 为原 UR5 范围启发式 0/1（未经 DQ 标定）\n非有限候选的 Jᵢ = ∞", "select")
    f.node("argmin", "j_best = argminᵢ Jᵢ\n每步重新评分全部 30 个候选", "select")
    f.node("memory", "持久索引 j_prev [B]\n初次 / 对应环境 reset 时设为 0\n其他环境保留历史索引", "fixed")
    f.node("switch", "若 J[j_prev] > J[j_best] + π/6，则切换\n否则保留；等于门槛也保留\n门槛作用于总代价，不一定只是角度", "select")
    f.node("target", "Gather 同一候选的 xyz 和 RPY\n输出真实候选 [B,6] + 新索引\n全部无效时输出当前末端位姿并置诊断标志", "output")
    f.node("buffer", "拼入 [B,1102]，Actor / Critic 共用\nPPO 缓冲区保存选中的位姿\n后续 minibatch 不访问筛选器", "fixed")
    f.edge("grasp", "quat"); f.edge("ee", "quat")
    f.chain("quat", "cost", "argmin", "switch", "target", "buffer")
    f.edge("memory", "switch"); f.edge("grasp", "target", "保留物理候选")
    f.edge("target", "memory", "更新索引", True)
    f.render()


def training():
    f = Figure("04_ppo_replay", "PPO 数据与梯度：选中目标随观测一起存储")
    f.node("raw", "环境原始 obs [B,1276]\n30 候选姿态 + 物体 / 机器人状态", "input")
    f.node("select", "KarlTeacherWrapper\n原始几何评分 + 滞回 + 部分 reset\n无神经网络 / 无梯度", "select")
    f.node("packed", "o_selected [B,1102]\n实际选中的目标抓取 6 维\n重新分配张量，避免环境原地改写旧观测", "input")
    f.node("rollout", "采样：标准化 → Actor / Critic\n高斯动作、旧 log_prob、旧 V\n与环境交互得到 r、done", "learn")
    f.node("memory", "RandomMemory：24 × B\nstates [24,B,1102] / actions [24,B,9]\nrewards / terminated / log_prob / values", "fixed")
    f.node("gae", "GAE：γ=0.99，λ=0.95\n优势 A、回报 R；价值标准化", "fixed")
    f.node("batch", "5 epochs × 6 minibatches\n打乱已存储的样本\n沿用原 RunningStandardScaler 更新规则", "input")
    f.node("net", "重新计算 Actor / Critic\n只读取样本自带的选中抓取\n不重评分、不改历史候选索引", "learn")
    f.node("loss", "原 PPO clipped policy loss + value loss\nratio_clip=0.2；value_clip=0.2\n初始 lr=4.2e−4；KLAdaptiveRL=0.008", "loss")
    f.node("opt", "Adam / 梯度裁剪 1.0\n只更新两套点云编码器与 MLP、Actor log_std\n冻结低层控制策略", "loss")
    f.node("reset", "done 的终止观测先进入缓冲区\n真正 env.reset() 时，仅清零对应环境的目标索引", "fixed")
    f.chain("raw", "select", "packed", "rollout", "memory", "gae", "batch", "net", "loss", "opt")
    f.edge("packed", "memory", "保存未归一化观测")
    f.edge("opt", "net", "更新参数", True); f.edge("reset", "select", "部分环境", True)
    f.render()


if __name__ == "__main__":
    for draw in (overview, networks, selector, training):
        draw()
