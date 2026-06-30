"""self-cond 模型诊断: 轨迹追踪 + 起点消融 + cosine。对比 best.pth (无 self-cond)。

--self_cond 开关控制:
  on : 不 del self_cond_proj; 采样 denoise 步输入 [z, x_pred_prev] (第一步 zeros), decode 步 [z, 0]
       (对齐训练: decode 行 sc_half 置零); cosine 测 [z, 0] uncond forward (第一步行为)
  off: del self_cond_proj; 采样单 dim; cosine 测 z (同 diag_m3_cosine/trajectory, 复现 best.pth 基线)

用法:
  best.pth 基线:   --ckpt checkpoints/m3/best.pth (默认 --self_cond 关, 因 num_self_cond_cfg_tokens=0)
  self-cond 模型:  --ckpt checkpoints/m3_self_cond/best.pth --self_cond
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
ap.add_argument("--ckpt", default="checkpoints/m3_self_cond/best.pth")
ap.add_argument("--enc_ckpt", default="checkpoints/expression_encoder/best.pth")
ap.add_argument("--point_ckpt", default="model.pt")
ap.add_argument("--self_cond", action="store_true", help="采样/cosine 走 2C self-cond 路径")
ap.add_argument("--n_samples", type=int, default=256)
ap.add_argument("--n_ode_steps", type=int, default=100)
ap.add_argument("--max_length", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:3")
args = ap.parse_args()

LATENT_MEAN, LATENT_STD, NOISE_SCALE, T_EPS = -0.0004, 0.9942, 1.0, 5e-2
device = torch.device(args.device); use_amp = device.type == "cuda"
torch.manual_seed(args.seed); np.random.seed(args.seed)
params = get_parser().parse_args([]); params.fp16 = False
env = build_env(params); env.rng = np.random.RandomState(args.seed + 100)
n_words = env.n_words; pad_id = env.float_word2id["<PAD>"]; eos_id = env.equation_word2id["<EOS>"]
ep = copy.copy(params)
ep.enc_emb_dim = params.dec_emb_dim; ep.n_enc_layers = params.n_dec_layers
ep.n_enc_heads = params.n_dec_heads; ep.n_enc_hidden_layers = params.n_dec_hidden_layers
expr_enc = TransformerModel(ep, env.equation_id2word, True, True, False, params.dec_positional_embeddings).to(device).eval()
expr_enc.load_state_dict(torch.load(args.enc_ckpt, map_location="cpu", weights_only=False)["model"])
for p in expr_enc.parameters(): p.requires_grad_(False)
mw = torch.load(args.point_ckpt, map_location="cpu", weights_only=False)
embedder = mw.embedder.to(device).eval(); point_enc = mw.encoder.to(device).eval()
for m in (embedder, point_enc):
    for p in m.parameters(): p.requires_grad_(False)
denoiser = ELF_models["ELF-B"](text_encoder_dim=ep.enc_emb_dim, max_length=args.max_length,
                                vocab_size=n_words, num_self_cond_cfg_tokens=0).to(device).eval()
if not args.self_cond:
    del denoiser.self_cond_proj
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
denoiser.load_state_dict(ck["model"])
print(f"loaded {args.ckpt}: step {ck['step']+1} val_l2 {ck.get('val_loss','?'):.4f} | self_cond={args.self_cond} | n={args.n_samples}", flush=True)

tree_tokens_list, bags, gt_trees = [], [], []
for _ in range(args.n_samples):
    expr, _ = env.gen_expr(train=True)
    tree_tokens_list.append(expr["tree_encoded"]); gt_trees.append(expr["tree"])
    bags.append(list(zip(expr["X_to_fit"][0], expr["Y_to_fit"][0])))
gt_tok, gt_lengths = env.batch_equations(env.word_to_idx(tree_tokens_list, float_input=False))
gt_tok, gt_lengths = gt_tok.to(device), gt_lengths.to(device)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
    enc_out = expr_enc("fwd", x=gt_tok, lengths=gt_lengths, causal=False).float()
pad_len = args.max_length - enc_out.shape[0]
if pad_len > 0:
    enc_out = F.pad(enc_out, (0, 0, 0, 0, 0, pad_len))
elif pad_len < 0:
    enc_out = enc_out[:args.max_length]; gt_lengths = gt_lengths.clamp(max=args.max_length)
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


def denoise_fwd(z, t, attn):
    """denoise forward; self_cond on 时输入 [z, x_pred_prev] 由调用方拼。这里只接已拼好的 z 或单 dim。"""
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        xp, _ = denoiser(z, torch.full((bs,), t, device=device), attention_mask=attn,
                         cond=cond_emb, cond_mask=cond_mask)
    return xp.float()


def skeleton_prefix(tree):
    if tree is None: return None
    if hasattr(tree, "nodes"): tree = tree.nodes[0]
    t = copy.deepcopy(tree); k = [0]
    def visit(node):
        if len(node.children) == 0:
            try: float(node.value)
            except ValueError: pass
            else: node.value = f"CONSTANT_{k[0]}"; k[0] += 1
        for c in node.children: visit(c)
    visit(t); return t.prefix()


gt_sk = [skeleton_prefix(t) for t in gt_trees]


def ids_to_tree(ids_row):
    seq = ids_row.tolist()
    cut = next((j for j in range(1, len(seq)) if seq[j] == eos_id), len(seq))
    return env.word_to_infix([env.equation_id2word[int(t)] for t in seq[1:cut]], is_float=False, str_array=False)


def struct_rate(ids):
    return sum(1 for i in range(bs) if ids_to_tree(ids[i]) is not None
               and skeleton_prefix(ids_to_tree(ids[i])) == gt_sk[i]) / bs


def z_rmse(z):
    return (z.float().cpu() - x0c).pow(2).mean(-1)[valid_cpu].mean().sqrt().item()


def ode_track(attn, t_start, n_steps):
    z = t_start * x0 + (1 - t_start) * torch.randn_like(x0) * NOISE_SCALE
    ts = torch.linspace(t_start, 1.0, n_steps + 1, device=device)
    x_prev = torch.zeros_like(z)  # self-cond 第一步无先验 (off 模式不用)
    track = [(t_start, z_rmse(z))]
    for i in range(n_steps):
        t = ts[i].item(); t_next = ts[i + 1].item()
        zin = torch.cat([z, x_prev], dim=-1) if args.self_cond else z
        xp = denoise_fwd(zin, t, attn)
        v = (xp - z) / max(1.0 - t, T_EPS)
        z = z + (t_next - t) * v
        x_prev = xp
        track.append((t_next, z_rmse(z)))
    # 终点 decode: self-cond on 时 [z, 0] (对齐训练 decode 行 sc_half=0); off 时 z
    zfinal = torch.cat([z, torch.zeros_like(z)], dim=-1) if args.self_cond else z
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        _, logits = denoiser(zfinal, torch.ones(bs, device=device), attention_mask=attn,
                             cond=cond_emb, cond_mask=cond_mask, decoder_step_active=True)
    return track, logits.argmax(-1).cpu()


ones = torch.ones(bs, args.max_length, dtype=torch.bool, device=device)

print(f"\n=== Part1: 低 t 端 x0 预测方向 (cosine) + rmse @ valid, 5 次平均 ===", flush=True)
print(f"(self_cond={'[z,0] uncond (第一步行为)' if args.self_cond else 'z 单dim'}; 随机 cosine≈0)", flush=True)
for t_val in (0.0, 0.02, 0.05, 0.1, 0.3, 0.5, 0.9):
    cs, rs = [], []
    for _ in range(5):
        z_t = t_val * x0 + (1 - t_val) * torch.randn_like(x0) * NOISE_SCALE
        zin = torch.cat([z_t, torch.zeros_like(z_t)], dim=-1) if args.self_cond else z_t
        xp = denoise_fwd(zin, t_val, valid)
        cs.append(F.cosine_similarity(xp.cpu(), x0c, dim=-1)[valid_cpu].mean().item())
        rs.append((xp.cpu() - x0c).pow(2).mean(-1)[valid_cpu].mean().sqrt().item())
    print(f"  t={t_val:.2f}: cosine={np.mean(cs):+.3f}±{np.std(cs):.3f} | rmse={np.mean(rs):.3f}", flush=True)

print(f"\n=== Part2: 轨迹追踪 (t_start=0, 每步 ‖z-x0‖) + 起点消融 ===", flush=True)
tr0, ids0 = ode_track(ones, 0.0, args.n_ode_steps)
for t, r in tr0[::10]:
    print(f"  t={t:.3f}: ‖z-x0‖={r:.3f}", flush=True)
print(f"  → t_start=0 终点结构正确率: {struct_rate(ids0):.3f}", flush=True)
print(f"\n=== 起点消融 (不同 t_start, 终点 rmse + 结构正确率) ===", flush=True)
for t_start in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
    tr, ids = ode_track(ones, t_start, args.n_ode_steps)
    print(f"  t_start={t_start:.2f}: 终点 ‖z-x0‖={tr[-1][1]:.3f} | 结构正确率={struct_rate(ids):.3f}", flush=True)
