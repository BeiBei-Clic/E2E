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
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_flow_matching.py \
    --num_workers 8 > logs/flow_m2.log 2>&1 &
tail -f logs/flow_m2.log   # 每 log_every 步 train_ema, 每 eval_every 步 val_l2 + decode_acc
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
| `--batch_size` | 128 | **每卡** batch（256 会 OOM；denoiser 92.6M + decode logits `(B,128,10292)` 大） |
| `--gradient_checkpointing` | 开 | denoiser 开 grad ckpt 省 activations（脚本固化，训练慢 ~30%）。OOM 时可再降 `--batch_size` 到 64 |

**显存 / GPU 负载调优**：
- OOM → 降 `--batch_size`（128→64）。当前 128 + grad_ckpt ≈ 7GB/卡。
- **GPU 负载不稳（掉）** → 多是 DataLoader 跟不上：M2 每步主进程要串行跑一遍 expr_enc 前向（57.8M），等 token 时空闲。缓解：`--num_workers 8`（脚本 `prefetch_factor=4`）。若仍掉，瓶颈是 expr_enc 前向（无法挪进 worker——CUDA 不能 fork），根本解是给 expr_enc 前向套 bf16 autocast（待加）。

**产出**：`{out_dir}/{best,last}.pth`（denoiser 权重 + optimizer/scheduler/step/best，支持 `--resume`）。
**M2 达标**：val_l2（denoise MSE）收敛下降 + decode 分支 unembed 产出多样合法 expression token。

> smoke 已验证（val_l2：40 步 29.9 → 100 步 4.36，loss 明确下降）。

---

## 阶段 C：denoiser flow matching 条件训练（Step 6 / M3 数值点→表达式）

M3 在 M2 基础上接入 condition：数值点 `(x,y)` → 数值点 encoder（`model.pt`，freeze）→ `cond_emb`，经 denoiser 的 **cross-attention** 注入。**弃用 ELF prepend**（prepend 破坏点数无关性——点数进主序列就要固定 `max_length`），改为每 block `[self-attn → cross-attn → FFN]`：cond 作 cross-attn 的 K/V（点数任意、不加噪、不进 loss），`max_length` 只含 target=128。expression encoder 仍 freeze 提供去噪目标 `x0`。从头训（denoiser 结构变，M2 权重不续）。详见计划文档 Step 6 / 项目记忆 `elf-sr-m3-cross-attention`。

```bash
# uniform 时间调度 + lr 2e-3 (ELF 原版 lr 0.002): 修通低 t 端采样瓶颈。
# 诊断: logit_normal(p_mean=0.8) 让 t<0.1 样本<0.01%, denoiser 低 t 端不学, ODE 从纯噪声出发即崩;
# lr (1e-4 vs 2e-3) 对低 t 端无影响 (实验证明, t=0.02 rmse 三次都卡 0.99~1.00)。uniform + lr2e-3 让采样合法率 0.23→0.79。
# GPU3 被占时用 3 卡 (effective batch 384); 4 卡全空时改 --nproc_per_node=4。
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 train_flow_m3.py \
    --lr 2e-3 --time_schedule uniform --warmup 300 --max_steps 5000 \
    --num_workers 4 --probe_every 200 --log_every 500 --eval_every 1000 \
    --out_dir checkpoints/lr_search/uniform_cosine_lr2e-3 \
    > logs/uniform_cosine_lr2e-3.log 2>&1 &
tail -f logs/uniform_cosine_lr2e-3.log   # 每 log_every 步 train_ema, 每 probe_every 步 低t端rmse, 每 eval_every 步 val_l2+acc
```

停止：`pkill -f train_flow_m3.py`。

续训（从 best.pth 接着训；**关键：`--max_steps` 要调大**，否则 cosine 已到末端 lr=0、等于没训）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 train_flow_m3.py \
    --lr 2e-3 --time_schedule uniform --warmup 300 --max_steps 10000 \
    --num_workers 4 --probe_every 200 --log_every 500 --eval_every 1000 \
    --resume checkpoints/lr_search/uniform_cosine_lr2e-3/best.pth \
    --out_dir checkpoints/lr_search/uniform_cosine_lr2e-3 \
    > logs/uniform_cosine_lr2e-3_resume.log 2>&1 &
tail -f logs/uniform_cosine_lr2e-3_resume.log
```

> `--resume` 恢复 model+optimizer+scheduler，从 step 5000 续到 `max_steps=10000`。cosine 周期随 max_steps 延长，lr 从中段（~1e-3）衰减到 10000（首步可能 lr=0，之后正常）。目的：让 t=0.02 端从 0.645 继续降，攻 ODE 收敛瓶颈（测B 真实 R² 仅 4%，瓶颈在 ODE 不收敛）。

**关键参数**（其余 `lr/warmup/eval_every/log_every/patience/val_*/num_workers/resume/cpu/seed` 同阶段 B）

| 参数 | 默认 | 说明 |
|---|---|---|
| `--point_ckpt` | `model.pt` | freeze 的数值点 encoder（embedder+encoder，18.8M；含 `mw.env.float_encoder` 编码数值点） |
| `--enc_ckpt` | `checkpoints/expression_encoder/best.pth` | freeze 的 expression encoder（提供 x0） |
| `--max_length` | 128 | **target only**（cond 经 cross-attn 不占主序列，序列长度回到 M2 水平） |
| `--out_dir` | `checkpoints/flow_m3` | checkpoint 输出 |
| `--batch_size` | 128 | 每卡 batch（denoiser 116.2M；序列 128，显存同 M2 ≈7GB/卡） |
| `--num_workers` | 4 | 每 rank 后台数据生成进程（worker 里跑 `gen_expr` + 数值点 `encode`+`batch`；实测 4 已完全 overlap，不必更大） |
| `--lr` | 2e-3 | ELF 原版 lr 0.002；1e-4 训 16K 步 decode acc 仅 0.6、采样合法率 0.23 |
| `--time_schedule` | `uniform` | denoise 分支 t 采样：`uniform`（U[0,1]，低 t 端样本充足，修采样起点崩）替代 `logit_normal`(p_mean=0.8)。**关键**：p_mean=0.8 让 t<0.1 样本<0.01%，denoiser 低 t 端不学 |
| `--lr_schedule` | `cosine` | warmup→cosine 到 max_steps；`constant` 实验证明前期 lr 大导致 loss spike，用正常 cosine |
| `--probe_every` | 200 | >0：每 N 步测低 t 端单步去噪 rmse（t=0.02/0.1/0.5），密集监控采样瓶颈；0=关 |

> M3 **不归一化 cond**（`cond_emb` 直接喂 cross-attn 的新 `kv_proj`，尺度差 std 0.29 vs target 1 由可学投影吸收）。**时间调度改 uniform**（替代 M2 的 `logit_normal` p_mean=0.8）：p_mean=0.8 让 t<0.1 训练样本 <0.01%，denoiser 低 t 端（采样起点）不学，ODE 从纯噪声出发即崩（z_final rmse 1.1 不收敛）；uniform 让低 t 端有样本，采样合法率 0.23→0.79。`noise_scale/t_eps/decoder_prob` 同 M2。lr 用 2e-3（ELF 配方；实验证明 lr 大小对低 t 端无影响，但加速中高 t 收敛）。encoder 前向套 **bf16 autocast**（freeze+no_grad）。
>
> **GPU 负载（关键，踩过坑）**：`LinearPointEmbedder.forward` 把 CPU 编码（`encode`+`batch`，`float_encoder` 逐数值转 descriptor token，~2.2s/批）和 GPU 嵌入（`embed`+`compress`）混在一起。CPU 编码若留在主进程会卡死 GPU（worker 只能 overlap `gen_expr`，overlap 不了它）——这就是 `num_workers` 调多大都没用的根因。**解法**：worker 里跑完 `encode`+`batch`（纯 CPU，只读 `float_encoder`/`float_word2id`/`params`，不碰 GPU 权重，fork 安全），主进程 `encode_cond` 只剩 GPU `embed`+`compress`+`point_enc`（~0.5s）。实测 `num_workers=4` 单卡 2.2s/步、DDP 4 卡 2.7s/步（warmup），GPU 4×100%。诊断用 `--profile_steps N`（单卡跑 N 步打分段计时：data/enc_t/enc_c/noise/fwd/bwd/opt）。

**M3 达标**（去风险点，区别于 M2 的关键）：
- `val_l2`（denoise MSE）下降；
- `acc_correct`（正确 cond 的 decode acc）超过 M2 水平 **0.632**；
- **`Δcond = acc_correct − acc_shuffled` 显著为正**——正确 cond 比 batch 内打乱 cond 更准，直接证明生成依赖输入（否则 cross-attn 被无视、两者持平）。

> smoke 已验证（单卡 `num_workers=0/2` 均通；val_l2 4 步 31.5→17.1 下降；可变点数 33~170 / 维度 1~10 正常）。负载优化后 4 卡 DDP 实测 2.7s/步（warmup）、GPU 4×100%、`enc_c` 2.7s→0.5s。DDP 删掉 `self_cond_proj`（M3 不用 self-cond）让所有参数 used，配 `find_unused_parameters=False` + `static_graph` + `gradient_as_bucket_view`（DDP 最佳实践）；util 周期性 100%↔30% 是 all-reduce 同步 + 各 rank forward 差异的固有低段（梯度通信仅 14ms，非瓶颈，无法消除），step 间隔 2.6s 波动<2% 即训练稳定。后续 Step 7 再加 self-cond / CFG / label-drop + 采样器（数值点 → 采样 → 表达式 → 新点 R²）。
>
> **采样 R² 评估**（`eval_flow_m3.py`，加载 checkpoint 测）：数值点 → ODE/SDE 采样 → 末步 decode → **常数优化**（Nelder-Mead 重拟合常数；拟合+R² 都用 `Node.val`——曾用 `BFGSRefinement`(sympytorch) 但 sympytorch 与 Node.val 对 inv/pow/log 数值不一致，导致 BFGS R² 反而 < raw，弃之）→ R²（reference 用 GT 表达式干净求值，弃含训练噪声的 y）。诊断：低 t 端单步去噪 rmse、ODE 收敛 rmse（z_final vs x0）、decode 合法率/多样性、R²（raw vs BFGS）。**结论（uniform+lr2e-3 5000步）**：常数优化有效（结构对的表达式 BFGS 后 R²≈1）；测D（单步 decode，z 接近 x0）BFGS R²>0.9 达 **55%**，测B（真实 ODE 采样）仅 **4%**——**瓶颈是 ODE 收敛**（t=0.02 端 0.645 不够低，z_final rmse 1.09 偏离 x0 → decode 结构错率上升），非 decode 头/常数优化；待突破（更长训练 / 更大模型 / self-cond）。

## 推理 / 评估（Step 7-8）

### pmlb 批量评估（M3 vs 端到端基线）

`experiments/pmlb/pmlb_batch_inference_m3.py`：把 `pmlb_batch_inference.py` 的符号回归模型从端到端 `SymbolicTransformerRegressor` 换成 M3 流匹配（数值点 → cond_emb → ODE 批量采样 → decode → BFGS 常数优化 → rescale → R²）。**对齐 `pmlb_adaptive_beam_inference.py` 两点做法**：① **自适应采样规模**——M3 无 beam，等价物是并行采样数 `n_samples`，R² < `r2_threshold` 则 `n_samples` 翻倍重试（跨 attempt 取最优、达阈值提前退出）；② **多 worker 并行 BFGS**——skeleton 去重后的唯一候选用 `ProcessPoolExecutor(fork)` 并行常数优化（M3 的 `tree_fit_r2` 纯 numpy、不依赖 env/GPU，task 直接传 `(tree,x,y)`）。维度 >10 跳过，其余（StandardScaler 标准化 X / BFGS-in-scaled-space / rescale_function / metrics）与端到端评估完全一致（CSV 在端到端字段上加 `beam_size`/`attempt` 两列）。端到端基线（model.pt, noise=0.1, 222 集）：r² 中位 0.783、>0.5 占 157/221。

```bash
# M3 pmlb 自适应评估 (改 --device / --noise_strength / --ckpt 即可; adaptive + 并行 BFGS 默认开)
PYTHONPATH=. .venv/bin/python experiments/pmlb/pmlb_batch_inference_m3.py \
    --device cuda:3 --noise_strength 0.1 \
    --ckpt checkpoints/lr_search/uniform_cosine_lr2e-3/best.pth \
    > logs/m3_pmlb_eval.log 2>&1 &
tail -f logs/m3_pmlb_eval.log   # 每集打印 "dataset: ok r2=... beam=N attempt=K (Ns)"
```

停止：`pkill -f pmlb_batch_inference_m3`。
续跑：直接重跑同命令（`load_existing_results` 按 `(dataset, noise_strength)` 跳过已完成，CSV 追加写）。**换权重 / 换噪声强度** → 结果落不同 CSV（默认按 `noise_strength` 自动分文件 `pmlb_m3_adaptive_noise_{ns}.csv`；换权重想保留旧结果就显式 `--output_csv` 指新文件）从头跑。

**关键参数**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--device` | `cuda` | **用哪张卡**：`cuda:0`/`cuda:3`（M3 推理 ~1.6G，可与训练同卡共存） |
| `--noise_strength` | `0.0` | **目标噪声**：给 y 加相对噪声 `y*(1+ns·N(0,1))`；对齐端到端基线用 `0.1` |
| `--ckpt` | `checkpoints/lr_search/uniform_cosine_lr2e-3/best.pth` | **加载哪个 M3 权重**（denoiser） |
| `--point_ckpt` | `model.pt` | freeze 的数值点 encoder（embedder+encoder） |
| `--n_samples` | 32 | **初始**采样规模（R²<阈值翻倍重试的起点；M3 无 beam，n_samples 个不同噪声 batch 并行采样） |
| `--r2_threshold` | 0.9 | R² 达此阈值提前退出（选优口径 = scaled 空间 vs `y_to_fit`） |
| `--max_retries` | 3 | R² 未达阈值的翻倍重试次数（总 attempt = max_retries+1，即 32→64→128→256） |
| `--bfgs_workers` | `os.cpu_count()` | 并行 BFGS worker 数（fork；skeleton 去重后的唯一候选并行优化常数） |
| `--n_ode_steps` | 100 | ODE Euler 步数（t:0→1，diffusion 采样步数；多更准但慢） |
| `--max_length` | 128 | target 序列长度（同训练） |
| `--max_rows` / `--max_input_points` | 200 / 200 | pmlb 行数上限 / 喂 cond 的数值点上限 |
| `--rescale` | True | StandardScaler 标准化 X（训练数值点已标准化到 O(1)，必要；`--no-rescale` 关） |
| `--noise_seed` / `--seed` | 0 / 0 | 噪声种子 / 采样种子 |
| `--dataset_limit` | None | 只跑前 N 个数据集（smoke 用，如 `--dataset_limit 2`） |
| `--output_csv` | `experiments/pmlb/results/pmlb_m3_adaptive_noise_{ns}.csv` | 结果 CSV（默认按 noise_strength 自动分文件；含 `beam_size`/`attempt` 列） |

> 流程对齐细节：`apply_target_noise` 加噪→`y_to_fit`（BFGS 拟合目标）；`StandardScaler` 只标准化 X、不动 y；BFGS(Nelder-Mead) 在 scaled_X 空间拟合常数、reference=`y_to_fit`；`rescale_function` 把树里 `x_k` 包 `add(b_k,mul(a_k,x_k))`（常数不变）；报告口径=rescale 后树在原 X 求值 vs 干净 y。`refinement_type` 取 NoRef/BFGS 中 r² 较优者。try-except 仅包 BFGS（Nelder-Mead 失败 / 非有限 → 回退 raw）。**自适应**：cond 只算一次（与采样数无关），每 attempt 仅重跑 ODE 采样 + 并行 BFGS；CSV 的 `beam_size`/`attempt` 记录命中最优的采样规模与轮次。smoke（best.pth step10000, n_samples=16 / max_retries=1 / 8 worker）：1027_ESL r2=0.866 beam=16 attempt=1、1028_SWD r2=0.333 beam=16 attempt=1（R²<0.9 已翻倍到 32 但未超过 16 的结果，故 attempt=1）。
