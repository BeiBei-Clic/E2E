# ELF-SR 运行手册

> 连续流匹配符号回归各阶段的运行命令。实现计划见 `docs/elf_sr_implementation_plan.md`。

## 阶段 A：expression encoder MLM 预训练（Step 2-3）

在线生成训练数据，MLM 预训练一个双向 expression encoder（freeze 后给 denoiser 提供去噪目标空间）。收敛判据：验证集 MLM loss 早停。

```bash
# 后台启动（默认参数即推荐配置）
mkdir -p logs
nohup python train_expression_encoder.py > logs/enc_mlm.log 2>&1 &

# 监控 val loss 曲线
tail -f logs/enc_mlm.log
```

**默认参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max_steps` | 200000 | 步数安全帽上限 |
| `--batch_size` | 256 | 在线生成训练 batch |
| `--mask_prob` | 0.15 | MLM 遮挡比例 |
| `--lr` | 1e-4 | 峰值学习率（warmup+cosine） |
| `--warmup` | 1000 | warmup 步数 |
| `--eval_every` | 2000 | 每 N 步 eval 验证集 loss |
| `--patience` | 20 | val loss 无 best 更新的早停耐心 |
| `--val_size` | 1024 | 验证集表达式数（独立 rng、mask 固定） |
| `--val_batch` | 128 | 验证集分块大小 |
| `--out_dir` | `checkpoints/expression_encoder` | checkpoint 输出目录 |
| `--cpu` | off | 强制 CPU |
| `--seed` | 0 | 随机种子 |

**产出**：`{out_dir}/best.pth`（最低 val loss）、`last.pth`（最后一个 eval 点）。
**收敛（M1）**：val loss 平台、`best` 不再更新（`bad` 涨到 `patience` 自动早停）。

快速缩减规模示例：
```bash
python train_expression_encoder.py --max_steps 50000 --batch_size 128
```

---

## 阶段 B：denoiser flow matching 训练（Step 6）

> TODO（待 Step 4-5 完成、Step 6 脚本就位后补命令）。

## 推理 / 评估（Step 7-8）

> TODO（待采样器与评估脚本就位后补命令）。
