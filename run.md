# ELF-SR 运行手册

> 连续流匹配符号回归各阶段的运行命令。完整训练路径见 `docs/training_pipeline.md`。

## 阶段 A：expression encoder MLM 预训练（Step 2-3）

在线生成训练数据，MLM 预训练一个双向 expression encoder（freeze 后给 denoiser 提供去噪目标空间）。**DDP 多卡加速**，收敛判据：验证集 MLM loss 早停。

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_expression_encoder.py \
    > logs/enc_mlm.log 2>&1 &
tail -f logs/enc_mlm.log   # 每 log_every(100) 步 train_ema/lr，每 eval_every(2000) 步 val/best
```

停止训练：
```bash
pkill -f train_expression_encoder.py        # 停（自动排除自身）
pkill -9 -f train_expression_encoder.py     # NCCL 偶尔僵死时强杀
pgrep -af train_expression_encoder.py       # 确认是否已退出
```
> 中途停会保留**最近一次 eval 点**的 `best.pth`/`last.pth`（每 `eval_every` 步存一次，含 model/optimizer/scheduler/step 等完整状态）。

断点续训（从 checkpoint 恢复 model/optimizer/scheduler/step/best，无缝接续）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_expression_encoder.py \
    --resume checkpoints/expression_encoder/last.pth > logs/enc_mlm_resume.log 2>&1 &
```
> 从 `last.pth` 的下一步起续训（权重 + 优化器动量 + lr 位置 + best/bad 全恢复）；数据序列重新在线生成（不影响续训）。也可 `--resume .../best.pth` 从最优那步续。

**默认参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max_steps` | 200000 | 步数安全帽上限 |
| `--batch_size` | 256 | **每卡**在线生成训练 batch（有效 batch = 该值 × world_size） |
| `--mask_prob` | 0.15 | MLM 遮挡比例 |
| `--lr` | 1e-4 | 峰值学习率（warmup+cosine；DDP 梯度按 SUM，配合此 lr 即线性 scaling） |
| `--warmup` | 1000 | warmup 步数 |
| `--eval_every` | 2000 | 每 N 步 eval 验证集 loss（并存 best/last + 打印 val） |
| `--log_every` | 100 | 每 N 步打印 train_ema/lr（不 eval，开销可忽略） |
| `--patience` | 20 | val loss 无 best 更新的早停耐心 |
| `--val_size` | 1024 | 验证集表达式数（独立 rng、mask 固定，所有 rank 共享） |
| `--out_dir` | `checkpoints/expression_encoder` | checkpoint 输出目录（仅 rank0 写） |
| `--num_workers` | 4 | 每 rank 后台数据生成进程数（实测 4 达拐点，8 不更快） |
| `--resume` | "" | 从 checkpoint 续训 |
| `--seed` | 0 | 随机种子（数据用 seed+rank+worker_id） |

**产出**：`{out_dir}/best.pth`（最低 val loss）、`last.pth`（最后一个 eval 点），支持 `--resume` 无缝续训。
**收敛**：val loss 平台、`best` 不再更新（`bad` 涨到 `patience` 自动早停）。

> 数据/DDP：`gen_tree_encoded` 只生成表达式树 + 编码、跳过 MLM 用不到的数值点（~6× 快于 `gen_expr`）；`TreeDataset` + `DataLoader(num_workers)` 后台多进程预取，overlap CPU 生成与 GPU 训练。`MLMEncoder` 把 `fwd+predict` 合并成单次 forward，满足 DDP「一次 forward 覆盖全部参数」。

---

## 阶段 B：denoiser flow matching 条件训练（Step 6 / M3 数值点→表达式）

denoiser（ELF-B，移植自 ELF-pytorch）在 expression embedding 空间做 flow matching 去噪。M3 接入 condition：数值点 `(x,y)` → 数值点 encoder（`model.pt`，freeze）→ `cond_emb`，经 denoiser 每 block `[self-attn → cross-attn → FFN]` 的 **cross-attention** 注入（弃用 ELF prepend——prepend 破坏点数无关性，点数进主序列就要固定 `max_length`；cond 作 cross-attn 的 K/V，点数任意、不加噪、不进 loss，`max_length` 只含 target=128）。expression encoder 仍 freeze 提供去噪目标 `x0`。从头训（denoiser 结构含 cross-attention）。

```bash
# uniform 时间调度 + lr 2e-3 修通低 t 端采样瓶颈:
#   logit_normal(p_mean=0.8) 让 t<0.1 样本<0.01%, denoiser 低 t 端不学, ODE 从纯噪声出发即崩;
#   lr(1e-4 vs 2e-3) 对低 t 端无影响 (实验证明), uniform + lr2e-3 让采样合法率 0.23→0.79。
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 train_flow_m3.py \
    --lr 2e-3 --time_schedule uniform --warmup 300 --max_steps 5000 \
    --num_workers 4 --probe_every 200 --log_every 500 --eval_every 1000 \
    --out_dir checkpoints/m3 \
    > logs/m3.log 2>&1 &
tail -f logs/m3.log   # 每 log_every 步 train_ema, 每 probe_every 步 低t端rmse, 每 eval_every 步 val_l2+acc
```

停止：`pkill -f train_flow_m3.py`。

续训（从 best.pth 接着训；**关键：`--max_steps` 要调大**，否则 cosine 已到末端 lr=0、等于没训）：
```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 train_flow_m3.py \
    --lr 2e-3 --time_schedule uniform --warmup 300 --max_steps 10000 \
    --num_workers 4 --probe_every 200 --log_every 500 --eval_every 1000 \
    --resume checkpoints/m3/best.pth \
    --out_dir checkpoints/m3 \
    > logs/m3_resume.log 2>&1 &
tail -f logs/m3_resume.log
```
> `--resume` 恢复 model+optimizer+scheduler，cosine 周期随 max_steps 延长，lr 从中段衰减到 10000。

**关键参数**（其余 `lr/warmup/eval_every/log_every/patience/val_*/num_workers/resume/cpu/seed` 同阶段 A）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--point_ckpt` | `model.pt` | freeze 的数值点 encoder（embedder+encoder，18.8M；含 `mw.env.float_encoder` 编码数值点） |
| `--enc_ckpt` | `checkpoints/expression_encoder/best.pth` | freeze 的 expression encoder（提供 x0） |
| `--max_length` | 128 | **target only**（cond 经 cross-attn 不占主序列） |
| `--out_dir` | `checkpoints/m3` | checkpoint 输出 |
| `--batch_size` | 128 | 每卡 batch（denoiser 116.2M；序列 128，显存 ≈7GB/卡） |
| `--num_workers` | 4 | 每 rank 后台数据生成（worker 里跑 `gen_expr` + 数值点 `encode`+`batch`；实测 4 已完全 overlap） |
| `--lr` | 2e-3 | ELF 原版 lr 0.002；1e-4 训 16K 步 decode acc 仅 0.6、采样合法率 0.23 |
| `--time_schedule` | `uniform` | denoise 分支 t 采样：`uniform`（U[0,1]，低 t 端样本充足，修采样起点崩）替代 `logit_normal`(p_mean=0.8) |
| `--lr_schedule` | `cosine` | warmup→cosine 到 max_steps；`constant` 前期 lr 大导致 loss spike |
| `--probe_every` | 200 | >0：每 N 步测低 t 端单步去噪 rmse（t=0.02/0.1/0.5），密集监控采样瓶颈；0=关 |

> M3 **不归一化 cond**（`cond_emb` 直接喂 cross-attn 的新 `kv_proj`，尺度差 std 0.29 vs target 1 由可学投影吸收）。encoder 前向套 **bf16 autocast**（freeze+no_grad）。DDP 删掉 `self_cond_proj`（M3 不用 self-cond）配 `find_unused_parameters=False` + `static_graph` + `gradient_as_bucket_view`。
>
> **GPU 负载关键坑**：`LinearPointEmbedder.forward` 把 CPU 编码（`encode`+`batch`，`float_encoder` 逐数值转 descriptor token，~2.2s/批）和 GPU 嵌入混在一起。CPU 编码若留在主进程会卡死 GPU（worker 只能 overlap `gen_expr`，overlap 不了它）——这就是 `num_workers` 调多大都没用的根因。**解法**：worker 里跑完 `encode`+`batch`（纯 CPU，只读 `float_encoder`/`float_word2id`/`params`，不碰 GPU 权重，fork 安全），主进程 `encode_cond` 只剩 GPU `embed`+`compress`+`point_enc`（~0.5s）。

**M3 达标**：
- `val_l2`（denoise MSE）下降；
- `acc_correct`（正确 cond 的 decode acc）超过 **0.632**；
- **`Δcond = acc_correct − acc_shuffled` 显著为正**——正确 cond 比 batch 内打乱 cond 更准，直接证明生成依赖输入。

> **M3 已知局限**：真实 ODE 采样 R² 仅 ~0.07（单步 decode z≈x0 时 BFGS R²>0.9 达 55%），瓶颈是 ODE 收敛（z_final rmse 偏离 x0 → decode 结构错率上升），非 decode 头/常数优化。**阶段 C 双头 GRPO 后训练**解决此问题（pmlb R² median 0.695 → 后训练后大幅提升）。

---

## 阶段 C：双头 GRPO 后训练（token-GRPO + Flow-DPPO）

RL 后训练提升 BFGS-R²。**双头**——flow 头走 Flow-DPPO（latent logp + KL-ADV mask），decode 头走 token-GRPO（reward 经 decode logits 的 token logp 反传，decode 头接收梯度）。修复了纯 latent Flow-DPPO 失败的根因（decode 头不在梯度路径，R² 反降 61%）。reward = `max(0, BFGS-R²)`，不可解码=0。详见 `docs/adr/0002-flow-dppo-posttraining.md`。

**脚本**：`train_flow_m3_rl.py`（从零写，DPPO loss 内联抄 `_flowdppo_kl_adv_loss`，不移植 unirl 框架）。从 `m3/best.pth` 起 RL。

```bash
PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python train_flow_m3_rl.py \
    --device cuda:3 --max_steps 500 --num_updates 1 --kl_mask_threshold 1e-3
```

续训（resume 恢复 model+optimizer+step+r_ema+best_r，收敛于 step 1400 / r_ema 0.647）：
```bash
PYTHONPATH=. PYTORCH_ALLOC_CONF=expandable_segments:True .venv/bin/python train_flow_m3_rl.py \
    --device cuda:3 --resume checkpoints/m3_rl/last.pth --max_steps 1500 --num_updates 1
```

停止：`pkill -f train_flow_m3_rl.py`。

**关键参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--ckpt` | `checkpoints/m3/best.pth` | 初始 denoiser 权重（RL 起点） |
| `--point_ckpt` | `model.pt` | freeze 的数值点 encoder |
| `--out_dir` | `checkpoints/m3_rl` | checkpoint 输出 |
| `--device` | `cuda` | 用哪张卡（单卡训） |
| `--G` | 32 | group 大小（每 prompt 采样 32 条做组相对优势） |
| `--B` | 8 | 每步 prompt 数（BG = B×G = 256） |
| `--K` | 10 | Flow-SDE 步数 |
| `--lr` | 1e-4 | RL 学习率 |
| `--eta` | 0.7 | SDE 随机性（σ_t = eta·√Δt·√(1-t)） |
| `--num_updates` | 2 | KL 锚点冻结后 minibatch 次数（实际用 1 防 off-policy 偏差） |
| `--decode_temp` | 1.0 | decode 温度采样 τ（rollout logp 与训练 logp 同口径） |
| `--kl_mask_threshold` | 1e-3 | flow 头 KL-ADV mask 阈值 |
| `--reward_workers` | 32 | BFGS reward 并行进程数 |
| `--max_steps` | 500 | 最大步数 |
| `--eval_every` | 50 | 每 N 步存 best/last（best 按 r_ema） |
| `--resume` | "" | 从 ckpt 续训（model+optimizer+step+r_ema+best_r） |

> chunk=64 硬编码（flow+decode 分块 forward+backward 防 BG=256 OOM，梯度累积等价全量）。训练结束脚本自动跑诊断（固定 prompt+噪声，init vs trained R² 对比）。reward 瓶颈 = BFGS 1.66-2.85s/样本（中位数 21 个常数），用 32 进程并行池。

**产出**：`checkpoints/m3_rl/{best,last}.pth`（best r_ema 0.647 @ step 1400，诊断 trained R² 0.554 vs init 0.124，涨 4.5 倍）。

---

## pmlb 批量评估

`experiments/pmlb/pmlb_batch_inference_m3.py`：数值点 → cond_emb → ODE 批量采样 → decode → 并行 BFGS 常数优化 → rescale → R²。**自适应采样规模**（R²<阈值 `n_samples` 翻倍重试，跨 attempt 取最优、达阈值提前退出）+ **多 worker 并行 BFGS**（skeleton 去重后唯一候选用 `ProcessPoolExecutor(fork)` 并行常数优化）。维度 >10 跳过。**BFGS-R² 是唯一裁判**。

```bash
# 结果 CSV 自动按"权重名(ckpt 父目录)+噪声强度"命名: pmlb_{权重名}_{ns:g}.csv
#   如 checkpoints/m3_rl/best.pth + 0.1 -> pmlb_m3_rl_0.1.csv
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_batch_inference_m3.py \
    --device cuda:1 --noise_strength 0.1 \
    --ckpt checkpoints/m3_rl/best.pth
```

停止：`pkill -f pmlb_batch_inference_m3`。续跑：直接重跑同命令（`load_existing_results` 按 `(dataset, noise_strength)` 跳过已完成，CSV 追加写）。换权重 / 换噪声强度 → 自动落不同 CSV。

四噪声（0/0.001/0.01/0.1）三卡并行跑：noise=0 先单卡跑完，再三卡并行跑 0.001/0.01/0.1。

**关键参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | `cuda` | 用哪张卡（`cuda:0`/`cuda:3`；推理 ~1.6G，可与训练同卡共存） |
| `--noise_strength` | 0.0 | **目标噪声**：给 y 加相对噪声 `y*(1+ns·N(0,1))`；四档评估用 0/0.001/0.01/0.1 |
| `--ckpt` | `checkpoints/m3/best.pth` | 加载哪个 denoiser 权重（后训练用 `checkpoints/m3_rl/best.pth`） |
| `--point_ckpt` | `model.pt` | freeze 的数值点 encoder |
| `--n_samples` | 32 | **初始**采样规模（R²<阈值翻倍重试的起点） |
| `--r2_threshold` | 0.9 | R² 达此阈值提前退出（选优口径 = scaled 空间 vs `y_to_fit`） |
| `--max_retries` | 3 | 翻倍重试次数（总 attempt = max_retries+1，32→64→128→256） |
| `--bfgs_workers` | `os.cpu_count()` | 并行 BFGS worker 数（fork） |
| `--n_ode_steps` | 100 | ODE Euler 步数（t:0→1；多更准但慢） |
| `--max_length` | 128 | target 序列长度（同训练） |
| `--max_rows` / `--max_input_points` | 200 / 200 | pmlb 行数上限 / 喂 cond 的数值点上限 |
| `--rescale` | True | StandardScaler 标准化 X（`--no-rescale` 关） |
| `--noise_seed` / `--seed` | 0 / 0 | 噪声种子 / 采样种子 |
| `--dataset_limit` | None | 只跑前 N 个数据集（smoke，如 `--dataset_limit 2`） |
| `--output_csv` | `experiments/pmlb/results/pmlb_{权重名}_{ns:g}.csv` | 结果 CSV（含 `beam_size`/`attempt` 列） |

> 流程对齐：`apply_target_noise` 加噪→`y_to_fit`（BFGS 拟合目标）；`StandardScaler` 只标准化 X、不动 y；BFGS(Nelder-Mead) 在 scaled_X 空间拟合常数、reference=`y_to_fit`；`rescale_function` 把树里 `x_k` 包 `add(b_k,mul(a_k,x_k))`（常数不变）；报告口径=rescale 后树在原 X 求值 vs 干净 y。`refinement_type` 取 NoRef/BFGS 中 r² 较优者。try-except 仅包 BFGS（失败/非有限 → 回退 raw）。

### 结果汇总

按 `noise_strength` 分组，对 Feynman / Strogatz / Black-box 三组统计 r² 均值/方差、recovery_rate（r²>0.9 占比）、complexity、seconds（`experiments/pmlb/pmlb_results_summary.py`）：

```bash
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_results_summary.py \
    --input_csvs \
        experiments/pmlb/results/pmlb_m3_rl_0.csv \
        experiments/pmlb/results/pmlb_m3_rl_0.001.csv \
        experiments/pmlb/results/pmlb_m3_rl_0.01.csv \
        experiments/pmlb/results/pmlb_m3_rl_0.1.csv \
    --output_csv experiments/pmlb/results/pmlb_m3_rl_summary.csv
```

> 文件名含 `_{ns}.csv` 即自动推断噪声强度（`infer_noise_strength`）。结果（后训练 m3_rl）：Feynman R²~0.90、Strogatz~0.97、Black-box~0.81，对噪声稳健（0.001 几乎无损，0.1 略降）。
