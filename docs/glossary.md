# Glossary — ELF-SR (E2E) 术语

## cross-condition (外部条件)
数值点 (x,y) → 表达式。任务给定的输入条件, 对应 ELF 翻译的"原句"(源端)。
注入方式: cross-attention (每 block 的 K/V 源)。全程固定 (同一组数值点)。
本仓库 M3 的 condition 通道。

## self-condition (自洽条件) — [[cross-condition]]
把本条轨迹**上一步对 x0 的预测 x̂₀_prev** 拼回 denoiser 输入 `[z, x̂₀_prev]` → self_cond_proj, 做迭代精化。
对应 ELF 翻译的"译文上一版草稿"(目标端自洽)。每步变化。
作用: 给"从带噪 z 反推干净 x0"一个 warm-start 先验, 减少多步采样累积误差。
**本仓库当前未启用** (见 [[adr-0001]]): 不对症低 t 端瓶颈, 且救不了采样第一步。

## low-t-denoising-bottleneck (低 t 端去噪瓶颈)
flow matching 中 t→0 时 z≈纯噪声, 从中反推干净 x0 信息论上极难 (t=0.02 时 x0 信号仅 2%)。
本仓库实测: best.pth 在 t=0.02 单步去噪 rmse≈0.64 (x0 std≈1), 且随训练/lr 调整稳定在此平台。
采样 ODE 从 t=0 起步 → 第一步在此瓶颈上误差爆炸 → 轨迹漂移 → 终点不收敛 (rmse 1.09) → 模式塌陷。
**当前 M3 的命门** (非 self-cond、非 lr)。

## mode-collapse (采样模式塌陷)
条件/无条件生成模型采样时输出坍缩到少数高频模式。本仓库表现: ODE 采样 decode 出的表达式
几乎全是 `1.4 add (14.0 mul ...)`, 常数集中在 {1.4, 14, 0.14}。源于采样终点没收敛到真实 x0,
落在 latent 空间默认区域。

## skeleton-structural-accuracy (结构正确率)
符号回归评估指标: 采出表达式的 skeleton (数值叶子替换为 CONSTANT_k 占位, 忽略具体常数) 与
真值 skeleton 完全一致的比例。区别于 R² (数值拟合) 和 token exact-match (逐 token, 含常数)。
> **立场变更 (2026-07-01)**: 用户裁定"能拟合的就是正确表达式, BFGS-R² 是唯一裁判"
> (见 [[sr-no-ground-truth-form]])。本指标**不再作为模型好坏或训练有效性的评判/终止判据**;
> 仅保留为"诊断采样过程本身有没有收敛"的工具性探测 (如 diag_m3_trajectory 起点消融)。

## SC-CFG (self-cond classifier-free guidance)
原始 ELF 的可选增强: num_self_cond_cfg_tokens 个可学习 prefix token 注入 self-cond 强度标量,
采样时做 cond/uncond 两次 forward 的 CFG。代价: 采样成本翻倍。本仓库 num_self_cond_cfg_tokens=0 (未启用)。
