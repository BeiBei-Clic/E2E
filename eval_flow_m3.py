"""M3 采样推理测试: 从纯噪声走完整 ODE 轨迹 -> 末步 decode -> token -> 表达式。
回答: 权重有没有塌陷、能不能解码出合法表达式。

测 A (oracle-length): valid mask 用 ground truth 长度, 隔离长度预测, 纯看映射能力。
测 B (真实采样): valid 全 1, 全 128 位置去噪 + decode, EOS 截断解析, 看合法率 + 多样性(塌陷)。

加载 best.pth, 单卡推理。采样: z=randn*scale (t=0) -> Euler ODE t:0->1 -> decode(t=1)。
"""
import argparse
import copy
import re

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize as _scipy_min

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.transformer import TransformerModel
from symbolicregression.flow import ELF_models


ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--ckpt", default="checkpoints/flow_m3/best.pth")
ap.add_argument("--enc_ckpt", default="checkpoints/expression_encoder/best.pth")
ap.add_argument("--point_ckpt", default="model.pt")
ap.add_argument("--n_samples", type=int, default=128)
ap.add_argument("--n_ode_steps", type=int, default=100)
ap.add_argument("--max_length", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

# flow matching 超参 (与训练一致)
LATENT_MEAN, LATENT_STD = -0.0004, 0.9942
NOISE_SCALE, T_EPS = 1.0, 5e-2

device = torch.device(args.device)
use_amp = device.type == "cuda"
torch.manual_seed(args.seed)
np.random.seed(args.seed)

params = get_parser().parse_args([])
params.fp16 = False                                   # simplifier 在 __init__ 总建; use_sympy 只控 gen_expr simplify, 保持 False 让数据生成快
env = build_env(params)
env.rng = np.random.RandomState(args.seed + 100)
n_words = env.n_words
pad_id = env.float_word2id["<PAD>"]
eos_id = env.equation_word2id["<EOS>"]

# expression encoder (freeze)
ep = copy.copy(params)
ep.enc_emb_dim = params.dec_emb_dim
ep.n_enc_layers = params.n_dec_layers
ep.n_enc_heads = params.n_dec_heads
ep.n_enc_hidden_layers = params.n_dec_hidden_layers
expr_enc = TransformerModel(ep, env.equation_id2word, True, True, False,
                            params.dec_positional_embeddings).to(device).eval()
expr_enc.load_state_dict(torch.load(args.enc_ckpt, map_location="cpu", weights_only=False)["model"])
for p in expr_enc.parameters():
    p.requires_grad_(False)

# 数值点 encoder (freeze)
mw = torch.load(args.point_ckpt, map_location="cpu", weights_only=False)
embedder = mw.embedder.to(device).eval()
point_enc = mw.encoder.to(device).eval()
for m in (embedder, point_enc):
    for p in m.parameters():
        p.requires_grad_(False)

# denoiser (加载 best.pth)
denoiser = ELF_models["ELF-B"](
    text_encoder_dim=ep.enc_emb_dim, max_length=args.max_length,
    vocab_size=n_words, num_self_cond_cfg_tokens=0).to(device).eval()
del denoiser.self_cond_proj
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
denoiser.load_state_dict(ck["model"])
print(f"loaded {args.ckpt}: step {ck['step']+1} val_l2 {ck.get('val_loss', '?'):.4f}", flush=True)

# ---- 生成 n_samples 个 (表达式, 数值点) ----
tree_tokens_list, bags, xs_list, ys_list, gt_trees = [], [], [], [], []
for _ in range(args.n_samples):
    expr, _ = env.gen_expr(train=True)
    tree_tokens_list.append(expr["tree_encoded"])
    xs_list.append(expr["X_to_fit"][0]); ys_list.append(expr["Y_to_fit"][0])
    gt_trees.append(expr["tree"])
    bags.append(list(zip(expr["X_to_fit"][0], expr["Y_to_fit"][0])))

# ground truth token id (batch_equations 加 [EOS, prefix..., EOS] 包装), (slen, bs)
gt_tok, gt_lengths = env.batch_equations(env.word_to_idx(tree_tokens_list, float_input=False))
gt_tok, gt_lengths = gt_tok.to(device), gt_lengths.to(device)

# ---- encode target x0 (提供 valid mask + oracle 对比基准) ----
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
    enc_out = expr_enc("fwd", x=gt_tok, lengths=gt_lengths, causal=False)
enc_out = enc_out.float()
pad_len = args.max_length - enc_out.shape[0]
if pad_len > 0:
    enc_out = F.pad(enc_out, (0, 0, 0, 0, 0, pad_len))
    gt_tok = F.pad(gt_tok, (0, 0, 0, pad_len), value=pad_id)
elif pad_len < 0:
    enc_out = enc_out[:args.max_length]
    gt_tok = gt_tok[:args.max_length]
    gt_lengths = gt_lengths.clamp(max=args.max_length)
x0 = (enc_out.transpose(0, 1) - LATENT_MEAN) / LATENT_STD
gt_ids = gt_tok.transpose(0, 1).long()                                  # (bs, max_length) 含 EOS + pad
valid = (torch.arange(args.max_length, device=device).unsqueeze(0) < gt_lengths.unsqueeze(1))  # bool

# ---- encode cond (数值点 -> cond_emb) ----
seq_tok, seq_cond_len = embedder.batch(embedder.encode(bags))
seq_tok, seq_cond_len = seq_tok.to(device), seq_cond_len.to(device)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
    seq_emb = embedder.compress(embedder.embed(seq_tok))
    cond_emb = point_enc("fwd", x=seq_emb, lengths=seq_cond_len, causal=False)
cond_emb = cond_emb.transpose(0, 1).float()
slen_max = cond_emb.shape[1]
cond_mask = (torch.arange(slen_max, device=device).unsqueeze(0) < seq_cond_len.unsqueeze(1)).float()

bs = x0.shape[0]
t_steps = torch.linspace(0.0, 1.0, args.n_ode_steps + 1, device=device)


def ode_sample_decode(attn_mask, gamma=0.0):
    """z=randn*scale -> t:0->1 -> 末步 decode -> argmax ids (bs, max_length)。
    gamma=0: 纯 ODE Euler; gamma>0: ELF SDE (每步往回注噪 z_in=alpha*z+(1-alpha)*eps, t_in=alpha*t),
    论文 Fig5c 证明 SDE 靠减少误差累积在少步数下远优于 ODE。"""
    z = torch.randn_like(x0) * NOISE_SCALE
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for i in range(args.n_ode_steps):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            if gamma > 0:                                  # SDE: 往噪声端偏一步再前向
                alpha = max(0.0, min(1.0, 1.0 - gamma * (t_next - t)))
                t_in = alpha * t
                z_in = alpha * z + (1 - alpha) * torch.randn_like(z) * NOISE_SCALE
            else:
                t_in, z_in = t, z
            x_pred, _ = denoiser(z_in, torch.full((bs,), t_in, device=device),
                                 attention_mask=attn_mask, cond=cond_emb, cond_mask=cond_mask)
            v = (x_pred.float() - z_in) / max(1.0 - t_in, T_EPS)
            z = z_in + (t_next - t_in) * v
        z_final = z
        _, logits = denoiser(z, torch.ones(bs, device=device),
                             attention_mask=attn_mask, cond=cond_emb, cond_mask=cond_mask,
                             decoder_step_active=True)
    return logits.argmax(-1).cpu(), z_final.float()


def parse_seq(ids_row):
    """对一个样本的 decode id 序列, 取 [1:首个尾部EOS] 去掉首 EOS 后解析。
    返回 infix 字符串 (None=非法)。"""
    seq = ids_row.tolist()
    cut = len(seq)
    for j in range(1, len(seq)):              # 跳过首 EOS, 找首个 EOS 作结尾
        if seq[j] == eos_id:
            cut = j
            break
    return env.idx_to_infix(seq[1:cut], is_float=False)


# ---- 测 A: oracle-length ----
print(f"\n=== 测 A: oracle-length valid, {args.n_ode_steps} ODE steps ===", flush=True)
ids_a, z_final_a = ode_sample_decode(valid)
valid_cpu = valid.cpu()
match = ((ids_a == gt_ids.cpu()) & valid_cpu)
x0c = x0.cpu()
_diff2 = (z_final_a.cpu() - x0c).pow(2).mean(-1)                                    # (bs, max_length)
print(f"ODE 收敛 rmse → valid {_diff2[valid_cpu].mean().sqrt():.3f} | pad {_diff2[~valid_cpu].mean().sqrt():.3f} "
      f"| (x0 valid std≈{x0c[valid_cpu].std():.3f})", flush=True)
print(f"token exact-match (valid 位置): {match.sum().item() / valid_cpu.sum().item():.3f}", flush=True)

# denoiser 单步去噪诊断: 直接 add_noise(x0, noise, t) -> denoiser -> x_pred, 看 ‖x_pred-x0‖
# 区分 "单步去噪不准" vs "ODE 多步累积漂移"。t 覆盖低/中/高三段。
print("--- denoiser 单步去噪 (add_noise 流形上, ‖x_pred-x0‖_rmse @ valid) ---", flush=True)
for t_val in (0.02, 0.1, 0.5, 0.9):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        z_t = t_val * x0 + (1 - t_val) * torch.randn_like(x0) * NOISE_SCALE
        x_pred, _ = denoiser(z_t, torch.full((bs,), t_val, device=device),
                             attention_mask=valid, cond=cond_emb, cond_mask=cond_mask)
    err = (x_pred.float().cpu() - x0c).pow(2).mean(-1)[valid_cpu].mean().sqrt()
    print(f"  t={t_val:.2f}: {err.item():.3f}", flush=True)
legal_a = sum(1 for i in range(bs) if parse_seq(ids_a[i]) is not None)
print(f"解析合法率 (去首 EOS+尾部截断): {legal_a}/{bs} = {legal_a/bs:.3f}", flush=True)
gt_legal = sum(1 for i in range(bs) if env.word_to_infix(tree_tokens_list[i], is_float=False) is not None)
print(f"(自检: ground truth prefix 解析合法率 {gt_legal}/{bs} = {gt_legal/bs:.3f})", flush=True)

# ---- 测 B: 真实采样 (valid 全 1) ----
print(f"\n=== 测 B: 真实采样 valid 全 1, {args.n_ode_steps} ODE steps ===", flush=True)
ids_b, z_final_b = ode_sample_decode(torch.ones(bs, args.max_length, dtype=torch.bool, device=device))
_diff2b = (z_final_b.cpu() - x0c).pow(2).mean(-1)
print(f"ODE 收敛 rmse → valid {_diff2b[valid_cpu].mean().sqrt():.3f} | pad {_diff2b[~valid_cpu].mean().sqrt():.3f}", flush=True)
preds_b = [parse_seq(ids_b[i]) for i in range(bs)]
legal_b = sum(1 for p in preds_b if p is not None)
unique = {str(p) for p in preds_b if p is not None}
print(f"解析合法率: {legal_b}/{bs} = {legal_b/bs:.3f}", flush=True)
print(f"多样性: {len(unique)} unique / {legal_b} 合法 = {len(unique)/max(legal_b,1):.3f} (低=塌陷)", flush=True)

# ---- 样例对比 (前 15 个) ----
print("\n=== 样例 (GT vs 测A pred vs 测B pred) ===", flush=True)
for i in range(min(15, bs)):
    gt = env.word_to_infix(tree_tokens_list[i], is_float=False)
    print(f"[{i}] GT  : {str(gt)[:88]}", flush=True)
    print(f"    A   : {str(parse_seq(ids_a[i]))[:88]}", flush=True)
    print(f"    B   : {str(preds_b[i])[:88]}", flush=True)

# decode token 直查 (前 3 个测 B, 看模型实际输出 token, 不依赖 idx_to_infix)
print("\n=== 测 B 原始 token (前 3 个样本, 截到首个 EOS) ===", flush=True)
for i in range(3):
    seq = ids_b[i].tolist()
    cut = next((j for j in range(1, len(seq)) if seq[j] == eos_id), len(seq))
    words = [env.equation_id2word[t] for t in seq[:cut]]
    print(f"[{i}] {' '.join(words)[:100]}", flush=True)

# ---- 测 D: 单步 decode (decoder_z 接近 x0, 不走 ODE) — decode 头能力上限 ----
print(f"\n=== 测 D: 单步 decode (decoder_z, 不走 ODE; valid=GT长度) ===", flush=True)
lam = torch.sigmoid(torch.randn(bs, args.max_length, 1, device=device) * 0.8 + 0.8)
decoder_z = lam * x0 + (1 - lam) * torch.randn_like(x0) * NOISE_SCALE
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
    _, logits_d = denoiser(decoder_z, torch.ones(bs, device=device),
                           attention_mask=valid, cond=cond_emb, cond_mask=cond_mask,
                           decoder_step_active=True)
ids_d = logits_d.argmax(-1).cpu()


def ids_to_tree(ids_row):
    """ids 行 -> 去 [首 EOS, 首个尾 EOS] 的 word list -> tree (None=非法)。"""
    seq = ids_row.tolist()
    cut = next((j for j in range(1, len(seq)) if seq[j] == eos_id), len(seq))
    words = [env.equation_id2word[int(t)] for t in seq[1:cut]]
    return env.word_to_infix(words, is_float=False, str_array=False)


def tree_r2(pred_tree, gt_tree, x_fit):
    prefix = pred_tree.prefix()
    dims = [int(d) for d in re.findall(r"x_(\d+)", prefix)]
    if dims and max(dims) >= x_fit.shape[1]:
        return float("nan")                        # 生成维度 > 输入维度, 无法求值
    yh = pred_tree.val(x_fit)
    y = gt_tree.val(x_fit)                          # 干净 reference: 原 GT 表达式求值 (ys_list 含训练噪声, 弃用)
    if not (np.all(np.isfinite(yh)) and np.all(np.isfinite(y))):
        return float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - yh) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


def tree_to_skeleton(tree):
    """数值叶子 -> CONSTANT_k 占位符 (BFGSRefinement 要求); 返回 (skeleton_node, coeffs0 初值)。
    NodeList (多输出包装) 取 nodes[0] (M3 单输出)。"""
    if hasattr(tree, "nodes"):                        # NodeList -> 取单输出 Node
        tree = tree.nodes[0]
    coeffs = []
    def visit(node):
        if len(node.children) == 0:
            try:
                float(node.value)
                k = len(coeffs); coeffs.append(float(node.value))
                node.value = f"CONSTANT_{k}"
            except ValueError:
                pass
        for c in node.children:
            visit(c)
    visit(tree)
    return tree, np.array(coeffs)


def _val1d(tree, x):
    """tree.val(x) -> (n,); NodeList.val 返回 (n, n_out), 取第 0 输出。"""
    yh = tree.val(x)
    return yh[:, 0] if (isinstance(yh, np.ndarray) and yh.ndim == 2) else yh


def _set_constants(skeleton, coeffs):
    """skeleton 的 CONSTANT_k 叶子填 coeffs[k] (返回新 tree, 不改原)。"""
    sk = copy.deepcopy(skeleton)
    k = [0]
    def visit(node):
        if len(node.children) == 0 and str(node.value).startswith("CONSTANT"):
            node.value = str(coeffs[k[0]]); k[0] += 1
        for c in node.children:
            visit(c)
    visit(sk)
    return sk


def tree_r2_bfgs(pred_tree, gt_tree, x_fit):
    """BFGS(Nelder-Mead) 重拟合常数; 拟合+R² 都用 Node.val (与 raw 同求值, 避免 sympytorch/Node.val
    数值差异导致 BFGS R² 反而 < raw)。结构错则救不了。"""
    prefix = pred_tree.prefix()
    dims = [int(d) for d in re.findall(r"x_(\d+)", prefix)]
    if dims and max(dims) >= x_fit.shape[1]:
        return float("nan")
    y = _val1d(gt_tree, x_fit)
    if not np.all(np.isfinite(y)):
        return float("nan")
    skeleton, coeffs0 = tree_to_skeleton(copy.deepcopy(pred_tree))
    if len(coeffs0) == 0:
        yh = _val1d(pred_tree, x_fit)
    else:
        def _obj(c):
            yh_c = _val1d(_set_constants(skeleton, c), x_fit)
            if not np.all(np.isfinite(yh_c)):
                return 1e10
            return float(((y - yh_c) ** 2).mean())
        res = _scipy_min(_obj, coeffs0, method="Nelder-Mead",
                         options={"maxiter": 200 * len(coeffs0), "xatol": 1e-5, "fatol": 1e-9})
        yh = _val1d(_set_constants(skeleton, res.x), x_fit)
    if not np.all(np.isfinite(yh)):
        return float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - yh) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")


# 测 E: SDE 采样 (gamma=1.5, ELF OWT 无条件配方) — 论文 Fig5c: SDE 减少误差累积
print(f"\n=== 测 E: SDE 采样 valid 全 1, gamma=1.5, {args.n_ode_steps} 步 ===", flush=True)
ids_e, z_final_e = ode_sample_decode(torch.ones(bs, args.max_length, dtype=torch.bool, device=device), gamma=1.5)
_diff2e = (z_final_e.cpu() - x0c).pow(2).mean(-1)
print(f"SDE 收敛 rmse → valid {_diff2e[valid_cpu].mean().sqrt():.3f} | pad {_diff2e[~valid_cpu].mean().sqrt():.3f}", flush=True)

print("\n=== R² 分布 (raw vs BFGS 常数优化) ===", flush=True)
results = {}
for tag, ids in [("测D 单步decode", ids_d), ("测B ODE真实采样", ids_b)]:
    rows = []
    for i in range(bs):
        tree = ids_to_tree(ids[i])
        if tree is None:
            continue
        r_raw = tree_r2(tree, gt_trees[i], xs_list[i])
        r_bfgs = tree_r2_bfgs(tree, gt_trees[i], xs_list[i])
        rows.append((i, r_raw, r_bfgs, tree.infix()))
    raw_fin = [r for _, r, _, _ in rows if np.isfinite(r)]
    bfgs_fin = [r for _, _, r, _ in rows if np.isfinite(r)]
    results[tag] = rows
    print(f"  [{tag}] n={len(rows)} | raw  R²>0.9 {sum(r>0.9 for r in raw_fin)}/{len(raw_fin)} 中位 {np.median(raw_fin):.3f} | "
          f"BFGS R²>0.9 {sum(r>0.9 for r in bfgs_fin)}/{len(bfgs_fin)} 中位 {np.median(bfgs_fin):.3f} | "
          f"BFGS R²>0.5 {sum(r>0.5 for r in bfgs_fin)}/{len(bfgs_fin)}", flush=True)

print("\n=== 高 R² 样例 (测D 单步decode, BFGS R² 最高的前 10) ===", flush=True)
rows_d = sorted(results["测D 单步decode"],
                key=lambda r: r[2] if np.isfinite(r[2]) else -9, reverse=True)
for i, r_raw, r_bfgs, infix in rows_d[:10]:
    gt = env.word_to_infix(tree_tokens_list[i], is_float=False)
    print(f"  [{i}] raw {r_raw:+.3f} BFGS {r_bfgs:+.3f} | GT: {str(gt)[:50]}", flush=True)
    print(f"        PRED: {infix[:62]}", flush=True)
