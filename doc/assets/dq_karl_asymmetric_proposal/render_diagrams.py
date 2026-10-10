#!/usr/bin/env python3
"""Render documentation-only proposal diagrams; no training code is changed."""
from pathlib import Path
import html
import subprocess

OUT = Path(__file__).resolve().parent
FONT = "Noto Sans CJK SC"
COLORS = {
    "actor": ("#E5EEFF", "#416DC2"),
    "critic": ("#F0E7FF", "#8B63B8"),
    "fixed": ("#E1F4EE", "#398474"),
    "gt": ("#FCE8E5", "#BB625A"),
    "data": ("#F0F3F7", "#6C7B91"),
    "note": ("#FFF4D9", "#BA9137"),
}


def quote(value):
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') + '"'


def table(title, rows, kind="data"):
    fill, border = COLORS[kind]
    lines = [
        '<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="7">',
        '<TR><TD ALIGN="LEFT"><B>{}</B></TD></TR>'.format(html.escape(title)),
    ]
    for row in rows:
        lines.append('<TR><TD ALIGN="LEFT">{}</TD></TR>'.format(html.escape(row)))
    lines.append('</TABLE>')
    return '<' + ''.join(lines) + '>'


def node(name, title, rows=(), kind="data"):
    fill, border = COLORS[kind]
    return '{} [label={}, fillcolor="{}", color="{}"];'.format(
        name, table(title, rows, kind), fill, border)


def edge(a, b, label=None, **attrs):
    if label:
        attrs["label"] = label
    tail = ' [' + ', '.join(k + '=' + quote(str(v)) for k, v in attrs.items()) + ']' if attrs else ''
    return '{} -> {}{};'.format(a, b, tail)


def graph(title, body, rankdir="TB"):
    return '\n'.join([
        'digraph proposal {',
        'graph [rankdir={}, bgcolor="#FFFFFF", pad="0.35", nodesep="0.45", ranksep="0.65", splines=polyline, fontname="{}", fontsize=23, fontcolor="#1E293B", labelloc=t, label={}];'.format(
            rankdir, FONT, quote(title + '\n建议方案｜尚未实现 / 训练验证')),
        'node [shape=box, style="rounded,filled", margin="0.06,0.04", penwidth=1.5, fontname="{}", fontsize=15, fontcolor="#172B42"];'.format(FONT),
        'edge [fontname="{}", fontsize=12, color="#65758B", fontcolor="#52647B", penwidth=1.5, arrowsize=0.75];'.format(FONT),
        body,
        'legend [shape=plain, style="", label=<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="10"><TR>'
        '<TD BGCOLOR="#E5EEFF">蓝：训练 / 部署 Actor</TD>'
        '<TD BGCOLOR="#F0E7FF">紫：仅训练 Critic</TD>'
        '<TD BGCOLOR="#E1F4EE">绿：固定感知 / 控制</TD></TR><TR>'
        '<TD BGCOLOR="#FCE8E5">红：仿真真值</TD>'
        '<TD BGCOLOR="#F0F3F7">灰：观测 / 计算</TD>'
        '<TD BGCOLOR="#FFF4D9">黄：设计约束</TD></TR></TABLE>>];',
        '{ rank=sink; legend; }',
        '}',
    ])


def system():
    body = '\n'.join([
        'subgraph cluster_train { label="训练侧：仿真环境"; color="#D89C96"; style="rounded,dashed";',
        node('sim', '训练：真实物理状态 sₜ', ['机器人 / 物体 / 桌面 / 接触'], 'gt'),
        node('sensor', '模拟传感器与测量包', ['时间相关噪声、延迟、丢帧、标定误差', '输出带曝光时间戳的估计测量接口'], 'fixed'),
        edge('sim', 'sensor'), '}',
        'subgraph cluster_real { label="部署侧：真实传感器"; color="#80B6A8"; style="rounded,dashed";',
        node('camera', '部署：双相机 RGB-D / mask', ['机器人编码器、IMU、状态估计器'], 'fixed'),
        node('vision', '外部预训练感知（冻结）', ['位姿初始化 / 跟踪 / 失视后重注册', 'CAD 可用性是显式部署前提'], 'fixed'),
        edge('camera', 'vision'), '}',
        node('common', '训练与部署复用相同的状态构造', ['时间对齐 → 运动补偿 → SE(3) 状态估计', '局部候选池 → 几何筛选 / 滞回 / 有效性', '所有筛选输入来自估计值'], 'fixed'),
        edge('sensor', 'common', '训练测量包'), edge('vision', 'common', '真机测量包'),
        node('obs', '可部署观测 oₜᴬ : [B,268]', ['当前机器人 Pₜ : 76', '目标历史 Gₜ₋₃:ₜ : 4 × 48 = 192'], 'data'),
        edge('common', 'obs'),
        node('actor', 'Actor πθ(a | oᴬ)', ['双分支 MLP → 9 维动作均值', '训练：高斯采样；部署：均值'], 'actor'),
        node('priv', '仅训练：80 维特权状态', ['同一候选 ID 的真实抓取位姿', '真实运动 / 接触 / 物理参数 / 任务状态'], 'gt'),
        node('critic', '独立 Critic Vφ(oᴬ, sᵖ)', ['268 维可部署观测 + 80 维特权状态', '不向 Actor 提供特征或动作'], 'critic'),
        edge('obs', 'actor'), edge('obs', 'critic'), edge('sim', 'priv', '真值仅用于训练支路', style='dashed', color='#BB625A'),
        edge('priv', 'critic'), '{ rank=same; actor; critic; }',
        node('ctrl', 'DQ 现有执行接口', ['归一化动作 → 限幅 / 物理尺度 → 目标累加',
            '累计 EE 目标 / IK → 冻结底层策略', '冻结底层全身策略 → 机器人运动', '默认：高层 6.25 Hz / 底层 50 Hz / 物理 200 Hz'], 'fixed'),
        node('ppo', '训练时 PPO / GAE', ['真实奖励与终止 → 回报 / 优势', '独立更新 Actor 与 Critic', '无蒸馏、无 BC、无 α 混合'], 'note'),
        edge('actor', 'ctrl', '唯一动作来源'), edge('critic', 'ppo', '价值估计'),
        edge('sim', 'ppo', '奖励与 done', style='dashed', color='#BB625A'),
        node('export', '部署导出边界', ['Actor 权重 + Actor 归一化统计', '固定估计器 / 历史 / 筛选状态与控制配置', 'Critic 与仿真特权状态不部署'], 'note'),
        edge('ctrl', 'export', style='invis'),
    ])
    return graph('01｜总体架构：估计几何 Actor + 特权 Critic', body)


def actor():
    body = '\n'.join([
        node('p', '当前机器人 Pₜ : [B,76]', [
            '关节位置 19 + 关节速度 18',
            '当前 EE：位置 3 + Rot6 6 = 9',
            '累计 EE 目标：位置 3 + Rot6 6 = 9',
            '底盘命令 3 + 上次实际执行的归一化命令 9',
            '重力方向 3 + 估计底盘线速度 3 + IMU 角速度 3'], 'data'),
        node('g', '单帧目标 Gₜ : [B,48]', [
            '估计抓取位姿：位置 3 + Rot6 6 = 9',
            '4 个对应抓取关键点误差：4 × 3 = 12',
            '相对抓取线速度 3 + 角速度 3',
            '位置不确定性 3 + 姿态不确定性 3',
            '两路相机有效标志 2 + 测量年龄 2',
            '已初始化 / 融合年龄 / 跟踪质量：各 1',
            '候选有效（含所需支撑面） / 本步切换：各 1',
            '两路相机视野余量：2',
            '估计支撑面法向量 3 + 平面偏移 1'], 'data'),
        '{ rank=same; p; g; }',
        node('history', '固定长度目标历史 H = 4', ['[Gₜ₋₃, Gₜ₋₂, Gₜ₋₁, Gₜ] → [B,192]', '含当前帧：3 × 0.16 s = 0.48 s 历史跨度',
            '只保存因果估计；reset 清历史', '换候选时刷新目标历史并写切换标志'], 'fixed'),
        edge('g', 'history'),
        node('pnorm', 'Actor 机器人归一化', ['[B,76]；部署固定统计量'], 'data'),
        node('gnorm', 'Actor 目标归一化', ['[B,192]；部署固定统计量', '标志 / 年龄的处理按字段显式规定'], 'data'),
        edge('p', 'pnorm'), edge('history', 'gnorm'),
        node('penc', '机器人编码器', ['Linear 76 → 128 + ELU', 'Linear 128 → 64 + ELU', '输出 [B,64]'], 'actor'),
        node('genc', '目标历史编码器', ['Linear 192 → 128 + ELU', 'Linear 128 → 64 + ELU', '输出 [B,64]'], 'actor'),
        edge('pnorm', 'penc'), edge('gnorm', 'genc'), '{ rank=same; penc; genc; }',
        node('fusion', '拼接机器人与目标特征', ['[B,64] ⊕ [B,64] → [B,128]'], 'data'),
        edge('penc', 'fusion'), edge('genc', 'fusion'),
        node('trunk', '控制 MLP', ['Linear 128 → 256 + ELU', 'Linear 256 → 128 + ELU', 'Linear 128 → 9 → μₜ [B,9]'], 'actor'),
        edge('fusion', 'trunk'),
        node('sigma', '可训练 log_std : [9]', ['与状态无关，广播到 batch'], 'actor'),
        node('action', '9 维归一化命令 uₜ', ['训练：uₜ ~ Normal(μₜ, diag(exp(2 log_std)))', 'PPO 保存原始采样 uₜ 和对应 log_prob', '部署：uₜ = μₜ → 相同执行适配器'], 'actor'),
        edge('trunk', 'action'), edge('sigma', 'action'),
        node('adapter', '部署与训练共用的执行适配器', ['归一化命令 → 限幅 / 物理尺度 → 原目标累加', 'Δp 3：0.02 m；ΔRPY 3：0.06 rad', 'vx：0.45 m/s；yaw：0.6 rad/s；夹爪：符号', '估计几何约束 / 预抓取；不构成完整碰撞规划'], 'fixed'),
        edge('action', 'adapter'),
        node('boundary', '观测边界与参数规模', ['原始 Actor 输入：76 + 4 × 48 = 268', '无图像 CNN / 1024 维离线特征 / GFM', '118,162 个可训练参数（含 log_std）'], 'note'),
        edge('adapter', 'boundary', style='invis'),
    ])
    return graph('02｜可部署 Actor：当前机器人状态 + 4 帧估计抓取历史', body)


def critic():
    body = '\n'.join([
        node('aobs', '同一条可部署观测 oₜᴬ', ['机器人状态 76 + 目标历史 192', '[B,268]；含实际估计误差 / 丢失历史', 'Critic 使用自己独立的归一化和编码器'], 'data'),
        node('privobs', '新增特权状态 sₜᵖ : [B,80]', [
            '物体真实位姿 9 + 真实 twist 6 = 15',
            '同一 Actor 候选 ID：真位姿 9 + 关键点误差 12 = 21',
            '真实底盘 twist 6 + 真实 EE twist 6',
            '接触概要 4（新增计算；非直接对象配对力）',
            '物体尺寸 3 + 质量 1 + 摩擦 1 = 5',
            '桌面位姿 9 + twist 6 = 15',
            '原任务状态 5 + 新阶段 one-hot 3 = 8'], 'gt'),
        '{ rank=same; aobs; privobs; }',
        node('aenc', 'Critic 可部署观测编码器', ['Linear 268 → 128 + ELU', 'Linear 128 → 128 + ELU', '输出 [B,128]'], 'critic'),
        node('privenc', 'Critic 特权编码器', ['独立按字段归一化', 'Linear 80 → 256 + ELU', 'Linear 256 → 128 + ELU', '输出 [B,128]'], 'critic'),
        edge('aobs', 'aenc'), edge('privobs', 'privenc'), '{ rank=same; aenc; privenc; }',
        node('cat', '拼接 [B,128] ⊕ [B,128]', ['融合输入 [B,256]'], 'data'),
        edge('aenc', 'cat'), edge('privenc', 'cat'),
        node('vnet', '价值 MLP', ['Linear 256 → 256 + ELU', 'Linear 256 → 128 + ELU', 'Linear 128 → 1'], 'critic'),
        edge('cat', 'vnet'),
        node('v', '状态价值 Vφ(oₜᴬ, sₜᵖ) : [B,1]', ['用于 GAE / 回报估计 / 价值损失', 'Critic 通过 GAE 优势提供策略训练信号'], 'critic'),
        edge('vnet', 'v'),
        node('guard', '独立参数与同一任务目标', ['原始 Critic 输入：268 + 80 = 348', '全部参数独立于 Actor；无特权特征回流', '不得给 Critic / 奖励另选一个候选 ID', '203,393 个可训练参数；训练后不部署'], 'note'),
        edge('v', 'guard', style='invis'),
    ])
    return graph('03｜仅训练 Critic：感知历史 + 特权物理摘要', body)


def perception():
    body = '\n'.join([
        node('packet', '带曝光时间戳 τ 的测量包', ['两路相机：RGB-D / mask → 位姿测量、质量、有效性', '仿真：从真值生成相关噪声 / 延迟 / 丢帧测量包', '机器人历史缓存：odom、关节、IMU、外参'], 'fixed'),
        node('capture', '1. 用曝光时刻的外参消除相机运动', ['ᵂTᴼ(τ) = ᵂTᶜ(τ) · ᶜTᴼ(τ)', 'W：局部 odom；C：相机；O：物体', '禁止用当前时刻外参代替曝光时刻外参'], 'fixed'),
        edge('packet', 'capture'),
        node('filter', '2. 在固定 odom 系 W 中维护状态', ['SE(3) 误差状态滤波 / 异常观测门控', '位置、姿态、线 / 角速度与协方差', '处理延迟到达观测后预测到当前控制时刻 t'], 'fixed'),
        edge('capture', 'filter'),
        node('visible', '有新观测', ['校验质量 / 创新残差 → 测量更新', '用真实时间间隔 Δt 更新速度'], 'fixed'),
        node('lost', '短暂失视', ['只预测，不重复把旧测量当新测量', '年龄 / 不确定性增长并传给 Actor'], 'note'),
        node('reacquire', '重现 / 长期失效', ['重注册、门控与状态重初始化', '无初始有效目标时走显式等待 / 恢复模式'], 'note'),
        edge('filter', 'visible'), edge('filter', 'lost'), edge('filter', 'reacquire'),
        '{ rank=same; visible; lost; reacquire; }',
        node('pool', '3. 当前估计物体位姿更新抓取池', ['ᵂTᴳⱼ(t) = ᵂT̂ᴼ(t) · ᴼTᴳⱼ', '局部候选只在初始化 / 失效重建时生成', 'CGN 输出必须先校准到 DQ 实际 TCP'], 'fixed'),
        edge('visible', 'pool'), edge('lost', 'pool'), edge('reacquire', 'pool'),
        node('selector', '4. 按估计几何选择候选', ['居中 / 上部 / 接近方向 / 桌面间隙', '评分滞回、近闭合保持、无候选显式标志', '不读取仿真真实物体位姿或真实桌面状态', '缺少所需支撑面估计：目标无效 / 显式保持'], 'fixed'),
        edge('pool', 'selector'),
        node('transform', '5. 转换到当前机器人参考系 A', ['ᴬTᴳ(t) = inv(ᵂTᴬ(t)) · ᵂTᴳ(t)', 'A 原点为 arm_base，轴与当前机器人 base 一致', '相对速度应补偿底盘平移 / 旋转', '换候选 / 重初始化：刷新历史，显式写切换标志'], 'fixed'),
        edge('selector', 'transform'),
        node('output', 'Gₜ [B,48] → 历史 H = 4 → Actor', ['有效性、测量年龄和不确定性跟随每帧数据', '整条感知与筛选链在训练 / 真机共用', '范围：DQ 无杂乱桌面；并非完整路径碰撞规划'], 'actor'),
        edge('transform', 'output'),
    ])
    return graph('04｜感知状态连续性：时间戳、移动坐标系与恢复', body)


def training():
    body = '\n'.join([
        node('start', '一套高层 Actor + 一套独立 Critic', ['Actor 从第一步起仅使用部署可获得的信息', '感知模块与底层全身控制保持冻结'], 'note'),
        node('collect', '并行采样 rollout', ['同一滤波器 / 候选选择器构建 oᴬ [268]', 'Actor 采样 9 维高层动作 → DQ 执行接口', '记录动作、old log_prob、oᴬ、sᵖ、奖励、终止'], 'actor'),
        edge('start', 'collect'),
        node('reward', '真实物理状态构造训练奖励', ['同一候选 ID 的抓取关键点对齐', '提升 / 保持、机器人稳定、碰撞与动作平滑', '双相机可见性与恢复，不把短遮挡等同失败'], 'gt'),
        '{ rank=same; collect; reward; }',
        edge('reward', 'collect', 'rₜ / done', constraint='false'),
        node('gae', 'Critic 估值 → GAE / return', ['正确处理超时 bootstrap 与真实终止', '固定采样期 log_prob 和实际传入的观测'], 'critic'),
        edge('collect', 'gae'),
        node('update', '同一轮 PPO 更新', ['Actor：clip 策略损失 + 熵项', 'Critic：价值回归损失', 'Actor / Critic 参数、归一化统计相互独立', '无教师动作标签、无 BC、无 α 混合'], 'note'),
        edge('gae', 'update'),
        edge('update', 'collect', '重复采样 / 更新', constraint='false', color='#416DC2'),
        'subgraph cluster_curriculum { label="同一个策略的任务难度课程（不是四个训练网络）"; color="#D3B267"; style="rounded,dashed";',
        node('c0', '难度 0：建立抓取能力', ['静态 / 较近目标；基础测量噪声', 'Actor 观测边界从开始就保持一致'], 'note'),
        node('c1', '难度 1：全身接近', ['扩大起点 / 物体 / 桌面变化', '强化移动到底盘可操作区域'], 'note'),
        node('c2', '难度 2：动态目标', ['从低速增至更快轨迹 / 角速度，模型 / 标定偏差', '检查运动估计与执行饱和'], 'note'),
        node('c3', '难度 3：感知中断', ['经标定异步延迟、短失视、异常重获与失败重试', '保留简单任务采样'], 'note'),
        edge('c0', 'c1'), edge('c1', 'c2'), edge('c2', 'c3'), '}',
        edge('update', 'c0', '分阶段独立验证窗口', style='dashed'),
        node('export', '完成训练 → 导出同一个 Actor', ['Actor 参数 / 归一化 + 固定状态构造配置', '真机运行感知 → Actor → 现有全身控制', '先验证真实感知闭环，再报告抓取成功率'], 'actor'),
        edge('c3', 'export'),
    ])
    return graph('05｜单阶段高层 PPO：独立 Actor / Critic 与持续课程', body)


def main():
    makers = [
        ('01_system', system), ('02_actor', actor), ('03_critic', critic),
        ('04_perception', perception), ('05_training', training),
    ]
    for name, maker in makers:
        dot = OUT / (name + '.dot')
        dot.write_text(maker(), encoding='utf-8')
        for fmt in ('svg', 'png'):
            command = ['dot', '-T' + fmt, str(dot), '-o', str(OUT / (name + '.' + fmt))]
            if fmt == 'png':
                command.insert(1, '-Gdpi=120')
            subprocess.run(command, check=True)
        print(name)


if __name__ == '__main__':
    main()
