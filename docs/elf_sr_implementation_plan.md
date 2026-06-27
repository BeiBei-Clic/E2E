# ELF 连续流匹配符号回归 —— 实现计划

> 基于 E2E 符号回归基础设施，移植 ELF-pytorch 的连续流匹配（Flow Matching）方法，
> 构建「数值点 (x,y) → 表达式」的连续扩散符号回归模型。
> 本计划是 grilling 会话达成的架构共识落地的实现指南。

## 0. 方案一句话

保留符号回归仓库的**数据生成 / 词汇表 / 数值点编码 / 评估**链路，
用 ELF-pytorch 的**连续流匹配 denoiser** 替换原有的自回归 decoder。
表达式 token 经一个 frozen 的双向 encoder 映射到连续 embedding 空间，
在该空间做 flow matching 去噪，仅最后一步 unembed 回离散 token。

---

## 1. 架构总览

### 1.1 三个组件

| 组件 | 来源 | 词汇表 | 训练状态 | 推理时 |
|---|---|---|---|---|
| **Expression encoder** | 新建 `TransformerModel(is_encoder=True)`，随机初始化 | `equation_words` | 阶段 A 先 MLM 预训练 → **freeze** | **不用** |
| **数值点 encoder** | 加载现有 `model.pt` 的 `embedder`+`encoder` | `float_words` | 直接 **freeze** | 用（编码输入数值点） |
| **Denoiser** | 移植 `ELF-pytorch_elf/src/modules/{model,layers}.py`，随机初始化 | — | 从头训（flow matching） | 用（去噪 + decode） |

### 1.2 两阶段训练

- **阶段 A**：expression encoder 的 MLM 预训练（独立，两 encoder 尚未接入 denoiser）
- **阶段 B**：freeze 两 encoder，按 ELF 原版条件生成配方训练 denoiser

### 1.3 关键设计决策（grilling 共识）

1. **denoiser 直接移植 ELF-pytorch**，不改造现有因果 decoder —— 两者在输入类型（连续向量 vs token id）、block 结构（RMSNorm+RoPE+SwiGLU+qk-norm vs LayerNorm+手写 MHA）、条件注入（prepend vs cross-attn）、输出头（x_pred+unembed vs 单 proj）上全面不同，`causal=False` 远不够。
2. **expression encoder 从头预训练再 freeze**（放弃 warm-start）—— 最干净，无因果/next-token 历史包袱，无需处理权重结构不匹配的拆解。
3. **归一化用 ELF 原版 `latent_mean/std`**（离线统计手填）—— 因为 encoder freeze，输出分布固定，固定常数合法；放弃 LayerNorm/running stats 方案。
4. **数值点 encoder 直接 freeze 现有 `model.pt`** —— condition 不参与去噪数学，对表示质量要求低于 target，白捡现成权重。
5. **condition 注入用 ELF 原版 prepend clean cond + `cond_seq_mask`** —— 机制现成（`sampling_utils.restore_cond`、`generation.test_generation_cond`），直接搬。

---

## 2. 组件规格

### 2.1 Expression encoder

- **架构**：复用 `symbolicregression/model/transformer.py:TransformerModel`，`is_encoder=True`（双向、非因果，`transformer.py:210-211`）。`dim = enc_emb_dim = 512`，`share_inout_emb=False`。
- **输入**：一条表达式的 token id 序列 `(S_eq, B)`（现有 `(slen, bs)` 布局）。
  - 产生链路：表达式树 → `Equation.encode`（`encoders.py:129`）→ prefix token list → `equation_word2id`（`environment.py:155`）→ `LongTensor`。
  - 例：`mul(x_1, sin(x_2))` → `["mul","x_1","sin","x_2"]` → `[14,5,9,6]`。
- **输出**：`(S_eq, B, 512)` → transpose → `(B, S_eq, 512)`，每个 token 位置一个双向 contextual embedding 向量。**即 ELF 的 clean embedding `x`。**
- **预训练任务（阶段 A）**：标准 MLM —— 随机 mask 一定比例 token，临时预测头对 `equation_words` 做 cross-entropy 预测。预测头预训练后丢弃。
- **为什么 MLM 能塑造合适的去噪目标空间**：MLM 是自监督，标签 = 被 mask 的原 token（无需外部标注）。关键在于 encoder 输出的 512 维向量**本身没有直接监督**——监督作用在"向量经预测头后能否还原 token"上，梯度穿过预测头倒灌，间接把向量塑造为"能还原 token 的结构化表示"。要还原 token，向量必须编码运算符 arity / 变量绑定 / 嵌套关系，于是空间变得结构化，这正是 flow matching 能画平滑去噪轨迹的前提（随机 encoder 的无结构空间无法去噪）。这与 ELF 里 T5 的角色一致：T5 也是"填空"（span corruption）预训练，任务与去噪无关，但表示质量高就够（论文 Fig 5a）。实证非保证，**M2（无条件去噪跑通）是检验关卡**；若不足，备选树结构掩码或放弃 freeze 改联合训练（后者引入归一化张力，需换 LayerNorm）。
- **freeze 后**：`requires_grad_(False)`，离线统计输出的 `mean/std`（见 3.2）。

### 2.2 数值点 encoder

- **来源**：现有 `model.pt` 中的 `embedder`（数值点→float token）+ `encoder`（float token→embedding）。
- **加载**：参考 `Example.ipynb` / `evaluate.py` 的 checkpoint 加载方式，取出对应 module 权重，新建同名实例载入。
- **输入**：数值点 bag `(x,y)` → `embedder` → float token → `encoder`（`causal=False`）。
- **输出**：`(B, S_cond, 512)` 的 condition embedding 序列。
- **freeze**：`requires_grad_(False)`，**不做 mean/std 归一化**（cond 不参与加噪/去噪数学；freeze 后尺度已固定，差异由 denoiser 的 `BottleneckTextProj` 吸收）。

### 2.3 Denoiser

- **来源**：移植 `ELF-pytorch_elf/src/modules/model.py:ELF` + `layers.py`，纯 PyTorch，仅依赖 `torch`+`einops`。
- **关键参数**：
  - `text_encoder_dim = 512`（= 两 encoder 输出维度，也是 flow matching 空间维度）
  - `vocab_size = len(equation_words)`（**不是 T5 的 32128**）
  - `bottleneck_dim = 128`、`num_time_tokens = 4`、`num_self_cond_cfg_tokens = 4`、`num_model_mode_tokens = 4`（ELF 默认）
  - 规模：起步用 `ELF-B`（depth=12, hidden=768, heads=12）
- **输入**：`concat[cond_emb, z_t]`，其中 `z_t` 可能带 self-conditioning（`[z_t, x_pred_prev]` → `self_cond_proj` → 512）。cond 部分由 `cond_seq_mask` 标记。
- **输出**：`(x_pred, decoder_logits)`。`x_pred` 经 `FinalLayer`（zero-init）→ 连续 embedding（denoise 分支用）；`decoder_logits` 经 factored unembed（decode 分支、`t=1` 时用）。

---

## 3. 数据流

### 3.1 阶段 A：expression encoder MLM 预训练

```
RandomFunctions 生成表达式树
   → Equation.encode → prefix token → equation_word2id → token id (S_eq, B)
   → 随机 mask（如 15%）→ expression encoder（trainable）
   → 临时预测头 → CE on masked tokens
   → 收敛后：丢预测头、freeze、保存权重
```

数据无限（`generators.RandomFunctions` 在线生成），无需离线语料。

### 3.2 离线统计 mean/std（阶段 A 结束、阶段 B 开始前）

```
采样一批表达式 → freeze 的 expression encoder 前向 → 收集输出 (B, S_eq, 512)
→ 计算全局 mean / std → 手填 config（latent_mean, latent_std）
```

### 3.3 阶段 B：denoiser flow matching 训练（条件生成）

对应 `ELF-pytorch_elf/src/train_step.py` 的条件生成逻辑。

```
batch = (表达式树, 对应数值点)
  │
  ├─ 数值点 → embedder → float token → [数值点 encoder·freeze]
  │                                   → cond_emb（clean，全程不加噪）
  │
  ├─ 表达式 → eq token → [expression encoder·freeze] → x
  │                       → 归一化 (x - latent_mean) / latent_std
  │                       → z_t = t·x_norm + (1-t)·ε·noise_scale   (t ~ logit-normal)
  │
  ├─（可选）self-conditioning：先跑一次得 x_pred'，concat [z_t, x_pred'] → self_cond_proj
  │
  └─ denoiser 输入 = concat[cond_emb, z_t]，cond 用 cond_seq_mask 保护（不移动、不进 loss）
       │
       ├─ denoise 分支（~80%）：net 预测 x_pred → v_pred=(x_pred-z_t)/(1-t) → MSE(v_pred, v_target)
       └─ decode 分支（~20%, t=1）：net decode mode → unembed → CE on GT token
```

`v_target`、self-conditioning、training-time CFG 的具体公式见 `train_step.py:120-178`。

### 3.4 推理

```
输入数值点 → embedder → 数值点 encoder → cond_emb
z_0 ~ N(0, I)
  → denoiser ODE/SDE 采样 t: 0→1（每步 cond 始终 restore 为 clean）
       每步：net denoise mode → x_pred → v → z ← z + dt·v
  → 末步 t=1：net decode mode → unembed → argmax → expression token id
  → idx_to_infix（environment.py:192）→ 表达式树
  → 复用 evaluate.py 评估（R²/RMSE/复杂度）
```

注意：**推理时不跑 expression encoder**（表达式未知，由 denoiser 生成）。

---

## 4. 实现步骤（依赖顺序）

### Step 1 — 移植 ELF-pytorch denoiser 骨架
- 拷贝 `ELF-pytorch_elf/src/modules/{model.py, layers.py}` 到 `symbolicregression/flow/`（新目录）。
- 参数化 `vocab_size`（构造时传入 `len(equation_words)`），确认 `text_encoder_dim=512`。
- 依赖：仅需 `einops`（已装）。
- **产物**：能前向 `(x, t, ...)` → `(x_pred, decoder_logits)` 的 `ELF` 模型。

### Step 2 — 搭建 expression encoder + MLM 预训练脚手架
- 新建 `TransformerModel(is_encoder=True)` 实例，绑定 `equation_words`。
- 加随机 mask 工具 + 临时预测头（参考 `transformer.py:444 predict` 的 CE 逻辑）。
- 写阶段 A 训练入口（独立脚本，只训 encoder）。

### Step 3 — 阶段 A：MLM 预训练 expression encoder  → **里程碑 M1**
- 训练到 MLM loss 收敛、masked token 重建合理。
- freeze、保存权重。

### Step 4 — 离线统计 mean/std  ✅
- 跑一批表达式过 freeze 的 encoder，统计全局 `mean/std`，填入阶段 B 的 config。
- **完成（2026-06-27）**：`compute_latent_stats.py`，best.pth（step 57999）→ `latent_mean=-0.0004, latent_std=0.9942`（≈0/1，LayerNorm 决定；目标空间已 well-scaled，归一化近恒等，与 ELF 原版 T5 std≈0.2 不同）。

### Step 5 — 加载 + freeze 数值点 encoder  ✅
- 从 `model.pt` 取 `embedder`+`encoder` 权重，载入新实例，`requires_grad_(False)`。
- **完成**：验证通过（embedder+encoder 18.8M 全 freeze；数值点 → cond_emb `(bs,slen,512)`，std≈0.29；`mw.env.params` 精简坑见项目记忆）。

### Step 6 — 阶段 B：flow matching 训练 denoiser
- 移植 `ELF-pytorch_elf/src/train_step.py`（PyTorch 版），改：
  - encoder 调用：`encode_text` 拆成 expression encoder（+归一化）和数值点 encoder 两路；
  - 数据来源：换成 `RandomFunctions` 的 `(表达式, 数值点)` batch（复用 `trainer.get_batch`）；
  - 词汇表：token id 来自 `equation_words`，unembed 已参数化。
- **先无条件验证（cond=None）→ 里程碑 M2**：denoiser 能在 expression embedding 空间去噪，末步 unembed 产出合法 expression token（先证明 flow matching 本身跑通，降低风险）。
- **加 condition → 里程碑 M3**：数值点 → 表达式，R² 起步。

### Step 7 — 采样器 + 推理
- 移植 `ELF-pytorch_elf/src/utils/sampling_utils.py`（ODE/SDE step）+ `generation.py`（条件生成主循环）。
- 末步 decode + unembed + argmax → `idx_to_infix` → 表达式。

### Step 8 — 评估  → **里程碑 M4**
- 复用 `evaluate.py` 的 R²/RMSE/复杂度 + SymPy 化简。
- 接入 `experiments/pmlb/` 批量推理框架。

---

## 5. 词汇表 / 数据对接

| ELF-pytorch 概念 | 符号回归对应 |
|---|---|
| T5 tokenizer | `equation_word2id` / `float_word2id`（`environment.py`） |
| `input_ids`（target） | 表达式 prefix token id |
| `condition_input_ids`（source） | 数值点 float token id |
| `vocab_size = len(tokenizer)` | `len(equation_words)` |
| T5 encoder（frozen） | expression encoder（阶段 A 预训练后 freeze）+ 数值点 encoder（freeze model.pt） |
| `latent_mean/std`（T5 输出统计） | expression encoder 输出统计（离线，阶段 4） |

数据生成无需改造：`RandomFunctions` 已同时产出表达式树与对应数值点，`trainer.get_batch` 即可复用。

---

## 6. 超参

### 复用 ELF-pytorch `config.py`（flow matching 相关）
- 时间调度：`denoiser_p_mean`, `denoiser_p_std`, `time_schedule='logit_normal'`
- 噪声：`denoiser_noise_scale=2.0`, `t_eps=5e-2`
- 双分支比例：`decoder_prob=0.2`（decode CE 20% / denoise MSE 80%）
- decode 分支：`decoder_p_mean=0.8`, `decoder_p_std=0.8`, `decoder_noise_scale`
- self-conditioning：`self_cond_prob=0.5`, `self_cond_cfg_min=0.5`, `self_cond_cfg_max=5.0`
- 采样：`sampling_method∈{ode,sde}`, `num_sampling_steps`, `cfgs`, `self_cond_cfg_scales`, `sde_gamma`
- EMA：`ema_decay1=0.9999`

### 符号回归特化
- `text_encoder_dim = 512`，`latent_mean=-0.0004 / latent_std=0.9942`（Step 4 实测，≈0/1）
- `max_length`：表达式 prefix 最大长度（看现有生成配置，一般几十 token）
- 数值点 condition 序列长度：数值点数 × 每点 float token 数
- denoiser 规模：起步 `ELF-B`，视显存调整

---

## 7. 验证里程碑

| 里程碑 | 阶段 | 验证标准 |
|---|---|---|
| **M1** | Step 3 后 | MLM loss 正常下降，expression encoder 能合理重建 masked token |
| **M2** | Step 6 中期（无条件） | denoiser 在 expression embedding 空间去噪跑通，末步 unembed 产出**合法** expression token（语法可解析） |
| **M3** | Step 6 + condition | 数值点 → 表达式条件生成，R² 开始有意义 |
| **M4** | Step 8 | PMLB 数据集上 R²/复原率对齐评估，与原 E2E 自回归基线对比 |

**M2 是关键去风险点**：先不加 condition、单独证明 flow matching 能在 expression embedding 空间工作，再叠加 condition，避免一旦失败无从定位。

---

## 8. 风险与备选

| 风险 | 触发条件 | 备选 |
|---|---|---|
| MLM 预训练的 encoder 表示不够好 | M1 重建质量差，或 M2 去噪不稳 | 加大树结构掩码（mask 子树）；或拉长预训练 |
| 数值点 condition 注入乏力 | M3 R² 很低、生成与输入无关 | 放开数值点 encoder 联合训练；或给 cond emb 也统计 mean/std |
| 表达式生成语法不合法 | M2/M3 unembed 后 `idx_to_infix` 解析失败 | decode 分支权重/比例调整；后处理语法约束 |
| ODE/SDE 采样步数不足 | 生成质量差但训练 loss 正常 | 加采样步数；切 SDE 采样器 |
