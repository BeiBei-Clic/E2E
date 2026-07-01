# ADR 0001: Self-conditioning — 机制对症, 待对照实验验证

- 状态: 待实验 (2026-06-30, 第二轮 grilling 修正)
- 决策者: grilling 会话 (用户质疑 → 诊断修正)

## 背景 (Context)

原始 ELF 自带 self-conditioning (把上一步对 x0 的预测 x̂₀_prev 拼回输入做迭代精化)。
本仓库迁移时为简化暂时去掉。评估发现"表达式严重偏离真实 (结构错, R² 会骗人)"。

## 命门: t=0 纯噪声启动失败 (非"低 t 端 rmse 大")

> 修正记录: 第一轮 grilling 误归因为"低 t 端去噪瓶颈"。用户质疑"去噪开始阶段噪声大是正常的"
> 成立——低 t 单步 rmse 大是 diffusion 通性, 非 bug。起点消融实验推翻了原归因。

**决定性证据 (experiments/diag_m3_trajectory.py, best.pth step20000)**:

起点消融 (z_start=t_start·x0+(1-t_start)·noise, 积分到 t=1):
```
t_start=0.00: 终点 ‖z-x0‖=1.093  ← 纯噪声起点, 发散
t_start=0.05: 终点 ‖z-x0‖=0.454  ← 仅多 5% x0 信号, 暴跌 58%
t_start=0.10: 终点 ‖z-x0‖=0.256
t_start=0.20: 终点 ‖z-x0‖=0.197
t_start=0.30: 终点 ‖z-x0‖=0.162
```
命门精确压在 **t=0 纯噪声启动**: 模型从零 x0 信号反推能力不足, 轨迹方向错误 → 进入 OOD 区域
(整条轨迹漂浮 ‖z-x0‖≈1.0, 中段 t=0.5-0.7 最低 1.007 但远未进训练流形, 高 t 端回升发散)。
绕过 t∈[0,0.05] 模型即可收敛。

**lr 不是命门** (run.md:122 用户早先实验: 1e-4 vs 2e-3 对低 t 无影响)。val_l2=0.0053 是假象
(被高 t 主导, 掩盖 t=0 启动失败)。

## 决策 (Decision): self-cond 重新相关, 待对照实验

self-cond 机制 ("第二步起用上一步 x̂₀ 估计作先验, 迭代精化") **针对的正是 t=0 启动后的逐步精化过程**。
起点消融证明"给一点 x0 方向信息就能救回轨迹" → self-cond 提供的跨步估计机制上对症。

**但不跳跃归因**: 起点消融给的是真 x0 信号, self-cond 给的是模型自己的(粗糙)估计, 不等价。
**必须实验**: 加 self-cond 训练 → 重跑 diag_m3_trajectory.py 起点消融 → 看 t_start=0 能否降到
接近 t_start=0.05 (0.454) 水平。能则 self-cond 对症, 不能则否。

## 杠杆点备选 (Consequences)

1. **self-cond (本 ADR 主题)**: 机制对症, 待对照实验。训练每步 +1 forward。
2. 采样端: t=0 启动改进 (更多低 t 步数 / 启动策略), 改采样器不重训。
3. 训练端: 低 t 段加权 / t=0 附近过采样 (但 uniform 已含 t∈[0,0.05])。
4. 参数化: v/noise-param 改变 t=0 反推难度。

## 诊断脚本

- experiments/diag_m3_structure.py — skeleton 结构正确率三口径 (测D单步/测B ODE/测E SDE)
- experiments/diag_m3_trajectory.py — 轨迹追踪 (每步‖z-x0‖) + 起点消融 (本 ADR 核心证据)

## 附录: 采样端零成本修正 (renorm) 失败 (2026-06-30)

诊断 experiments/diag_m3_renorm.py 验证"数值偏=范数问题"假设:
- Part1: 低 t 端 x_pred std 塌缩 (t=0: 0.704 vs x0 std 1.0, 偏小 30%, "保守预测")
- Part2: 每步对 x_pred renorm (强制 mean0 std1) 后, ODE 终点 1.096→0.982 (仅降 0.11, 远未收敛)

**结论: 范数偏是次要因素。** renorm 修范数但修不了方向 (cosine 0.66 的 34° 误差仍在)。
主因 = 低 t 端方向精度不足 + OOD 漂移恶性循环 (轨迹漂浮 ‖z-x0‖≈1.0)。
**采样端标准修正不够 → 需训练端手段 (self-cond 跨步先验提升方向精度)**。

## 实验结论 (2026-07-01): self-cond 对 t=0 启动无效 — 否决

m3_self_cond/best.pth (step20000, lr1e-4, self_cond_prob0.5) 采样诊断 vs best.pth 基线:
- t=0 cosine: 0.655 vs 0.66 (一样)
- 轨迹漂浮 ‖z-x0‖ 终点: 1.091 vs 1.085 (一样, 都没收敛)
- 起点消融 t_start=0 终点: 1.086 vs 1.093 (一样); 整条曲线几乎重合
- 结构正确率: 0% vs 0% (t_start=0.5 才 ~3.5%)

**self-cond 完全未改善采样。** 验证了 grilling 标记的结构性风险: self-cond 先验链起点
= 第一步 uncond x̂₀ (cosine 0.655, 和无 self-cond 一样不准), 跨步先验救不了"起点能力不足"。
轨迹从第一步就漂出流形, 后续先验全是错的。val_l2 也未改善 (0.0067 vs 0.0053, 略高)。

**self-cond 方向否决。** t=0 纯噪声启动是 flow matching 本质难题, 依赖"上一步 x̂₀"的机制
无效 (起点 x̂₀ 本身就塌)。后续方向应转向不依赖低 t 单步反推的方案:
1. 参数化 (预测 noise / v-param, 改变低 t 反推难度)
2. 采样起点 (不从 t=0 起步 / 流形投影)
3. 训练加权 (低 t 段 min-SNR)
