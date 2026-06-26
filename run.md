# ELF-SR 运行手册

> 连续流匹配符号回归各阶段的运行命令。实现计划见 `docs/elf_sr_implementation_plan.md`。

## 阶段 A：expression encoder MLM 预训练（Step 2-3）

在线生成训练数据，MLM 预训练一个双向 expression encoder（freeze 后给 denoiser 提供去噪目标空间）。**DDP 多卡加速**，收敛判据：验证集 MLM loss 早停。

```bash
# 多卡 DDP（推荐）：按机器改 CUDA_VISIBLE_DEVICES 与 --nproc_per_node
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_expression_encoder.py \
    > logs/enc_mlm.log 2>&1 &

# 监控 val loss 曲线（每 eval 打印 train_ema / val / best / bad / lr / 耗时）
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
> 中途停会保留**最近一次 eval 点**的 `best.pth`/`last.pth`（每 `eval_every` 步存一次），不会丢太多进度。当前脚本未做断点续训，重启从 step 0 开始。

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
| `--eval_every` | 2000 | 每 N 步 eval 验证集 loss |
| `--patience` | 20 | val loss 无 best 更新的早停耐心 |
| `--val_size` | 1024 | 验证集表达式数（独立 rng、mask 固定，所有 rank 共享） |
| `--val_batch` | 128 | 验证集分块大小 |
| `--out_dir` | `checkpoints/expression_encoder` | checkpoint 输出目录（仅 rank0 写） |
| `--cpu` | off | 强制 CPU（禁用 DDP） |
| `--seed` | 0 | 随机种子（训练数据用 seed+rank，故每卡见不同数据） |

**产出**：`{out_dir}/best.pth`（最低 val loss）、`last.pth`（最后一个 eval 点）。
**收敛（M1）**：val loss 平台、`best` 不再更新（`bad` 涨到 `patience` 自动早停）。

> DDP 说明：每 rank 用 `seed+rank` 独立生成训练数据（数据并行）；验证集固定且所有 rank 相同；checkpoint 与日志仅 rank0 输出。模型用 `MLMEncoder` 包装把 `fwd+proj+CE` 合并成单次 forward，以满足 DDP「一次 forward 覆盖全部参数」的要求。

---

## 阶段 B：denoiser flow matching 训练（Step 6）

> TODO（待 Step 4-5 完成、Step 6 脚本就位后补命令）。

## 推理 / 评估（Step 7-8）

> TODO（待采样器与评估脚本就位后补命令）。
