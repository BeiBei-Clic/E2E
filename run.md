# ELF-SR 运行手册

> 连续流匹配符号回归各阶段的运行命令。实现计划见 `docs/elf_sr_implementation_plan.md`。

## 阶段 A：expression encoder MLM 预训练（Step 2-3）

在线生成训练数据，MLM 预训练一个双向 expression encoder（freeze 后给 denoiser 提供去噪目标空间）。**DDP 多卡加速**，收敛判据：验证集 MLM loss 早停。

```bash
# 多卡 DDP（推荐）：按机器改 CUDA_VISIBLE_DEVICES 与 --nproc_per_node
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_expression_encoder.py \
    > logs/enc_mlm.log 2>&1 &

# 监控：每 log_every(100) 步打印 train_ema/lr，每 eval_every(2000) 步打印 val/best
tail -f logs/enc_mlm.log
```

停止训练：
```bash
# 停掉（pkill 自动排除自身，安全；会杀掉 torchrun + 所有训练子进程）
pkill -f train_expression_encoder.py

# 若卡住不退（NCCL 偶尔僵死），强杀
pkill -9 -f train_expression_encoder.py

# 查看进程 / 确认是否已退出
pgrep -af train_expression_encoder.py
```
> 中途停会保留**最近一次 eval 点**的 `best.pth`/`last.pth`（每 `eval_every` 步存一次，含 model/optimizer/scheduler/step 等完整状态）。

断点续训（从 checkpoint 恢复 model/optimizer/scheduler/step/best，无缝接续）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_expression_encoder.py \
    --resume checkpoints/expression_encoder/last.pth > logs/enc_mlm_resume.log 2>&1 &
```
> 从 `last.pth` 的下一步起续训（模型权重 + 优化器动量 + lr 位置 + best/bad 全恢复）；数据序列重新在线生成（不影响续训）。也可 `--resume .../best.pth` 从最优那步续。

其它启动方式：
```bash
# 单卡走 DDP
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_expression_encoder.py
# 单卡非 DDP（最简，不走 NCCL）
python train_expression_encoder.py
# 快速缩减规模
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_expression_encoder.py \
    --max_steps 50000 --batch_size 128
```

**默认参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max_steps` | 200000 | 步数安全帽上限 |
| `--batch_size` | 256 | **每卡**在线生成训练 batch（有效 batch = 该值 × world_size） |
| `--mask_prob` | 0.15 | MLM 遮挡比例 |
| `--lr` | 1e-4 | 峰值学习率（warmup+cosine；DDP 梯度按 SUM，配合此 lr 即线性 scaling） |
| `--warmup` | 1000 | warmup 步数 |
| `--eval_every` | 2000 | 每 N 步 eval 验证集 loss（并存 best/last + 打印 val） |
| `--log_every` | 100 | 每 N 步打印 train_ema/lr（不 eval，开销可忽略，便于看进度） |
| `--patience` | 20 | val loss 无 best 更新的早停耐心 |
| `--val_size` | 1024 | 验证集表达式数（独立 rng、mask 固定，所有 rank 共享） |
| `--val_batch` | 128 | 验证集分块大小 |
| `--out_dir` | `checkpoints/expression_encoder` | checkpoint 输出目录（仅 rank0 写） |
| `--num_workers` | 4 | 每 rank 后台数据生成进程数（DataLoader 预取，overlap CPU 生成与 GPU 训练；实测 4 已达拐点，8 不更快） |
| `--resume` | "" | 从 checkpoint 恢复继续训练（如 `checkpoints/expression_encoder/last.pth`） |
| `--cpu` | off | 强制 CPU（禁用 DDP） |
| `--seed` | 0 | 随机种子（训练数据用 seed+rank+worker_id，每卡每 worker 见不同数据） |

**产出**：`{out_dir}/best.pth`（最低 val loss）、`last.pth`（最后一个 eval 点）。每个 checkpoint 含 `model/optimizer/scheduler/step/best_val/best_step/bad`，支持 `--resume` 无缝续训。
**收敛（M1）**：val loss 平台、`best` 不再更新（`bad` 涨到 `patience` 自动早停）。

> 数据/DDP 说明：`gen_tree_encoded` 只生成表达式树 + 编码、跳过 MLM 用不到的数值点（~6× 快于 `gen_expr`）；`TreeDataset` + `DataLoader(num_workers)` 在后台多进程预取，overlap CPU 生成与 GPU 训练（实测 `num_workers=4` 即达拐点，~0.8s/step）。每 rank 用 `seed+rank`、每 worker 用 `seed+rank+worker_id` 保证数据多样；验证集固定且所有 rank 相同；checkpoint 与日志仅 rank0 输出。`MLMEncoder` 把 `fwd+predict` 合并成单次 forward，满足 DDP「一次 forward 覆盖全部参数」。

---

## 阶段 B：denoiser flow matching 训练（Step 6 / M2 无条件版）

denoiser（ELF-B，移植自 ELF-pytorch）在 expression embedding 空间做 flow matching 去噪。**M2 无条件版**（不加 condition / self-cond / CFG），先验证 flow matching 本身；expression encoder freeze 提供目标 x0，归一化用 Step 4 的 mean/std。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_flow_matching.py > logs/flow_m2.log 2>&1 &
tail -f logs/flow_m2.log   # 每 log_every 步 train_ema, 每 eval_every 步 val_l2
```

停止 / 续训同阶段 A（`pkill -f train_flow_matching.py`；`--resume {out_dir}/last.pth`）。

**关键参数**（其余 `lr/warmup/eval_every/log_every/patience/val_*/num_workers/resume/cpu/seed` 同阶段 A）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--enc_ckpt` | `checkpoints/expression_encoder/best.pth` | freeze 的 expression encoder |
| `--max_length` | 128 | 表达式 pad 长度（denoiser RoPE 固定长度约束） |
| `--latent_mean` / `--latent_std` | -0.0004 / 0.9942 | Step 4 实测（归一化 x0） |
| `--p_mean` / `--p_std` | 0.8 / 0.8 | logit-normal 时间调度 |
| `--noise_scale` | 1.0 | flow matching 噪声尺度 |
| `--decoder_prob` | 0.5 | 每 example 选 decode(CE) vs denoise(MSE) 分支的概率 |
| `--t_eps` | 5e-2 | `v=(x-z)/(1-t)` 分母 clamp |
| `--out_dir` | `checkpoints/flow_m2` | checkpoint 输出 |

**产出**：`{out_dir}/{best,last}.pth`（denoiser 权重 + optimizer/scheduler/step/best，支持 `--resume`）。
**M2 达标**：val_l2（denoise MSE）收敛下降 + decode 分支 unembed 产出多样合法 expression token。

> smoke 已验证（val_l2：40 步 29.9 → 100 步 4.36，loss 明确下降）。M3 将叠 condition（数值点 cond_emb + `cond_seq_mask`，clean cond 不加噪）。

## 推理 / 评估（Step 7-8）

> TODO（待采样器与评估脚本就位后补命令）。
