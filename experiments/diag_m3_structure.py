"""诊断: 训练分布上 M3 表达式结构正确率 (skeleton match)。判定 self-cond 是否对症。

三口径对照 (都看 skeleton 结构正确率, 忽略常数; R² 会骗人故只作参考):
  测D 单步 decode (decoder_z≈x0, oracle-length): decode 头 + 接近真实 latent 的能力上限
  测B 多步 ODE 采样 (z=randn→100步 Euler→decode, pmlb 同款路径): 真实采样
  测E SDE 采样 (gamma=1.5, 每步回注噪减累积误差): 减漂移对照

判读:
  D高 B低 E高 → 累积漂移实锤 (ODE 漂移, SDE/self-cond 减漂移对症)
  D B E 都低  → 结构先验本身差 (decode/latent 没学好, self-cond 边际有限)
  D B E 都高  → 训练分布结构对, pmlb 差是分布外 (self-cond 无关)
"""
import argparse, copy
import numpy as np
import torch
import torch.nn.functional as F
from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.transformer import TransformerModel
from symbolicregression.flow import ELF_models

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--ckpt", default="checkpoints/m3/best.pth")
ap.add_argument("--enc_ckpt", default="checkpoints/expression_encoder/best.pth")
ap.add_argument("--point_ckpt", default="model.pt")
ap.add_argument("--n_samples", type=int, default=256)
ap.add_argument("--n_ode_steps", type=int, default=100)
ap.add_argument("--max_length", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:3")
args = ap.parse_args()

LATENT_MEAN, LATENT_STD = -0.0004, 0.9942
NOISE_SCALE, T_EPS = 1.0, 5e-2

device = torch.device(args.device)
use_amp = device.type == "cuda"
torch.manual_seed(args.seed); np.random.seed(args.seed)

params = get_parser().parse_args([])
params.fp16 = False
env = build_env(params)
env.rng = np.random.RandomState(args.seed + 100)
n_words = env.n_words
pad_id = env.float_word2id["<PAD>"]
eos_id = env.equation_word2id["<EOS>"]

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

mw = torch.load(args.point_ckpt, map_location="cpu", weights_only=False)
embedder = mw.embedder.to(device).eval()
point_enc = mw.encoder.to(device).eval()
for m in (embedder, point_enc):
    for p in m.parameters():
        p.requires_grad_(False)

denoiser = ELF_models["ELF-B"](
    text_encoder_dim=ep.enc_emb_dim, max_length=args.max_length,
    vocab_size=n_words, num_self_cond_cfg_tokens=0).to(device).eval()
del denoiser.self_cond_proj
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
denoiser.load_state_dict(ck["model"])
print(f"loaded {args.ckpt}: step {ck['step']+1} val_l2 {ck.get('val_loss','?'):.4f} | n={args.n_samples}", flush=True)

# 生成 n_samples 个 (表达式, 数值点) — train=True 即训练分布
tree_tokens_list, bags, gt_trees = [], [], []
for _ in range(args.n_samples):
    expr, _ = env.gen_expr(train=True)
    tree_tokens_list.append(expr["tree_encoded"])
    gt_trees.append(expr["tree"])
    bags.append(list(zip(expr["X_to_fit"][0], expr["Y_to_fit"][0])))

gt_tok, gt_lengths = env.batch_equations(env.word_to_idx(tree_tokens_list, float_input=False))
gt_tok, gt_lengths = gt_tok.to(device), gt_lengths.to(device)

with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
    enc_out = expr_enc("fwd", x=gt_tok, lengths=gt_lengths, causal=False).float()
pad_len = args.max_length - enc_out.shape[0]
if pad_len > 0:
    enc_out = F.pad(enc_out, (0, 0, 0, 0, 0, pad_len))
    gt_tok = F.pad(gt_tok, (0, 0, 0, pad_len), value=pad_id)
elif pad_len < 0:
    enc_out = enc_out[:args.max_length]
    gt_tok = gt_tok[:args.max_length]
    gt_lengths = gt_lengths.clamp(max=args.max_length)
x0 = (enc_out.transpose(0, 1) - LATENT_MEAN) / LATENT_STD
valid = (torch.arange(args.max_length, device=device).unsqueeze(0) < gt_lengths.unsqueeze(1))
x0c = x0.cpu(); valid_cpu = valid.cpu()

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
    z = torch.randn_like(x0) * NOISE_SCALE
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for i in range(args.n_ode_steps):
            t = t_steps[i].item(); t_next = t_steps[i + 1].item()
            if gamma > 0:
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


def ids_to_tree(ids_row):
    seq = ids_row.tolist()
    cut = next((j for j in range(1, len(seq)) if seq[j] == eos_id), len(seq))
    words = [env.equation_id2word[int(t)] for t in seq[1:cut]]
    return env.word_to_infix(words, is_float=False, str_array=False)


def skeleton_prefix(tree):
    """tree -> skeleton prefix (数值叶子→CONSTANT_k, 忽略常数); None=非法。"""
    if tree is None:
        return None
    if hasattr(tree, "nodes"):
        tree = tree.nodes[0]
    t = copy.deepcopy(tree)
    k = [0]
    def visit(node):
        if len(node.children) == 0:
            try:
                float(node.value)
            except ValueError:
                pass
            else:
                node.value = f"CONSTANT_{k[0]}"; k[0] += 1
        for c in node.children:
            visit(c)
    visit(t)
    return t.prefix()


gt_sk = [skeleton_prefix(t) for t in gt_trees]
print(f"GT skeleton 合法: {sum(1 for s in gt_sk if s is not None)}/{bs}", flush=True)


def struct_stats(tag, ids):
    n_legal = n_match = 0
    samples = []
    for i in range(bs):
        tree = ids_to_tree(ids[i])
        if tree is None:
            continue
        n_legal += 1
        sk = skeleton_prefix(tree)
        hit = (gt_sk[i] is not None and sk == gt_sk[i])
        n_match += int(hit)
        if len(samples) < 12:
            samples.append((i, hit, tree))
    print(f"[{tag}] 结构正确 {n_match}/{bs} = {n_match/bs:.3f} | 合法 {n_legal}/{bs} = {n_legal/bs:.3f}", flush=True)
    return n_match / bs, samples


# 单步去噪 rmse (单步能力基准)
print("--- 单步去噪 rmse @ valid (单步能力; 对比 ODE 累积) ---", flush=True)
for t_val in (0.02, 0.1, 0.5, 0.9):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        z_t = t_val * x0 + (1 - t_val) * torch.randn_like(x0) * NOISE_SCALE
        x_pred, _ = denoiser(z_t, torch.full((bs,), t_val, device=device),
                             attention_mask=valid, cond=cond_emb, cond_mask=cond_mask)
    err = (x_pred.float().cpu() - x0c).pow(2).mean(-1)[valid_cpu].mean().sqrt()
    print(f"  t={t_val:.2f}: {err.item():.3f}", flush=True)

# 测D 单步 decode (oracle-length, decode 头上限)
lam = torch.sigmoid(torch.randn(bs, args.max_length, 1, device=device) * 0.8 + 0.8)
decoder_z = lam * x0 + (1 - lam) * torch.randn_like(x0) * NOISE_SCALE
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
    _, logits_d = denoiser(decoder_z, torch.ones(bs, device=device),
                           attention_mask=valid, cond=cond_emb, cond_mask=cond_mask,
                           decoder_step_active=True)
ids_d = logits_d.argmax(-1).cpu()

print(f"\n=== 结构正确率对照 (skeleton match, 忽略常数) ===", flush=True)
rate_d, samp_d = struct_stats("测D 单步decode   (oracle-len, decode头上限)", ids_d)

ones = torch.ones(bs, args.max_length, dtype=torch.bool, device=device)
ids_b, z_b = ode_sample_decode(ones)
_diff2 = (z_b.cpu() - x0c).pow(2).mean(-1)
print(f"测B ODE 收敛 rmse → valid {_diff2[valid_cpu].mean().sqrt():.3f} | pad {_diff2[~valid_cpu].mean().sqrt():.3f}", flush=True)
rate_b, samp_b = struct_stats("测B 多步ODE采样  (z=randn→100步, 真实路径) ", ids_b)

ids_e, z_e = ode_sample_decode(ones, gamma=1.5)
_diff2e = (z_e.cpu() - x0c).pow(2).mean(-1)
print(f"测E SDE 收敛 rmse → valid {_diff2e[valid_cpu].mean().sqrt():.3f} | pad {_diff2e[~valid_cpu].mean().sqrt():.3f}", flush=True)
rate_e, samp_e = struct_stats("测E SDE采样      (gamma=1.5, 减累积漂移)   ", ids_e)

print(f"\n=== 判读 ===", flush=True)
print(f"D(单步上限)={rate_d:.3f}  B(ODE真实)={rate_b:.3f}  E(SDE减漂移)={rate_e:.3f}", flush=True)

print("\n=== 测B (ODE) 样例: GT vs PRED (前12, ✓/✗=结构) ===", flush=True)
for idx, hit, tree in samp_b:
    gt_str = env.word_to_infix(tree_tokens_list[idx], is_float=False)
    print(f"  [{'✓' if hit else '✗'}] GT:{str(gt_str)[:46]:48s} PRED:{tree.infix()[:46]}", flush=True)
