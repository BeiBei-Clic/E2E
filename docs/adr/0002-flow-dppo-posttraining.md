# ADR 0002: Flow-DPPO 后训练迁移到 ELF-SR — 实验失败

- 状态: **实验失败** (2026-07-02, 诊断证实架构不匹配)
- 决策者: grilling 会话 (7 轮问答) → 实现 → 诊断否决

## 背景 (Context)

M3 流匹配预训练完成 (`checkpoints/m3/best.pth`, 无 self-cond), pmlb R² 中位 0.695。
尝试用 Flow-DPPO (UniRL, 连续流匹配 RL 后训练) 提升 BFGS-R²。
评判标准: BFGS-R² 是唯一裁判 (能拟合即正确, 见 memory `sr-no-ground-truth-form`)。

## 探索的决策 (7 项, 实现 train_flow_m3_rl.py)

1. reward = max(0, BFGS-R²), 不可解码/负=0; 整组 0 跳过。
2. 数据 = 合成训练数据 (在线生成, 复用预训练管线), pmlb 留 held-out。
3. 采样器 = Flow-SDE (当前 t:0→1, μ_θ=ODE-Euler 步, σ_t=eta·√Δt·√(1−t))。gamma-SDE 不能用 (ε 非加性)。
4. denoiser 全量更新, encoder freeze。
5. reward 瓶颈: BFGS 1.66–2.85s/样本 (常数中位 10-11); 16 进程并行, G=8/B=8/K=10。
6. 从零写 (DPPO loss 内联), 不移植 unirl 框架。
7. lr=1e-4, eta=0.7, τ=1e-3 (实测 KL floor 5e-5, τ=1e-5 不适用)。

Step 1-4 验证全通过: 加载/Flow-SDE rollout (1.4s/64样本, 合法率 81%)/reward (16进程 16s, group-std 0.247 健康)/DPPO 循环 (梯度累积修 OOM, ratio sanity check)。

## 实验结果 (2026-07-02): 失败

50 步训练 + 诊断 (固定 prompt+噪声, init vs trained):

```
reward 趋势: 无爬升, 后段下降 (step 8-17 均值 0.225 → step 31-50 均值 0.152)
KL: 5e-5 → 2e-6 (梯度信号快速衰减)
诊断: init r_mean 0.138 (>0:520/1024) | trained r_mean 0.054 (>0:226/1024) | decode 相同率 0.000
```

- decode 相同率 0.000: denoiser **确实学了** (不是梯度太弱), 50 步后 decode 完全变了。
- trained r_mean 0.054 vs init 0.138 (**降 61%**): 学到的方向**严重损害 reward**。

## 根因: latent RL 与 token reward 的架构鸿沟

denoiser 两头: **flow 头** (`final_layer`, 输出 latent x_pred) + **decode 头** (`unembed`, 输出 token logits), 共享 backbone。

DPPO loss 经 `μ_new = ODE步(x_pred)` 反传, **梯度只到 flow 头 + shared backbone, decode 头不在梯度路径**。
reward = decode 头 argmax 后骨架的 R², 在 token 空间。

- RL 优化 flow 头 (让 μ 朝高 advantage 的 z_next 靠), 但 z_next=μ_old+σ·ε 含噪。
- backbone 被带去拟合含噪点 → flow 精度破坏 → decode 头读到被带偏的 hidden → token 质量雪崩。
- **flow 头优化目标与 decode 头 reward 目标天然冲突**, 共享 backbone, RL 牺牲 decode 头迁就 flow 头。

非调参能救 (方向错非幅度错): 降 lr / 加 KL 正则只让崩得慢。诊断相同率 0 + reward 降 61% 钉死。

## 结论

**Flow-DPPO (latent policy gradient) 与当前 SR 架构 (latent flow + 独立 decode 头 + argmax) 不匹配。**
reward 在 token 空间、RL 在 latent 空间, 隔着不可微 argmax + 不共享梯度的 decode 头。
图像 RL 无此问题 (像素 reward 在生成空间)。

出路 (未采纳, 记录备查):
- **A. RL 改 token 级** (reward 经 decode logits 反传, 本质 DRPO/AR-RL, 偏离 Flow-DPPO, 重写大半)。
- **B. 接受, 换思路** (如回 ADR 0001 采样起点问题, 那才是 R² 上限的根本约束)。

代码/日志已删 (2026-07-02), 本 ADR 保留为负面教训。
