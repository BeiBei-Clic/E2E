"""Step 4: 离线统计 expression encoder 输出的 mean/std (供阶段 B flow matching 归一化)。

加载 best.pth 的 expression encoder (freeze), 前向一批原 token 表达式 (不加 mask,
统计的是 clean embedding x), 累积全局标量 mean/std (ELF 原版 latent_mean/std 风格),
填入阶段 B config。归一化 (x - latent_mean) / latent_std。
"""
import argparse
import copy

import numpy as np
import torch

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.transformer import TransformerModel
from train_expression_encoder import gen_batch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/expression_encoder/best.pth")
    ap.add_argument("--n_batches", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"ckpt: step={ck['step']} best_val={ck['best_val']:.4f} | device={device}", flush=True)

    params = get_parser().parse_args([])
    params.fp16 = False
    env = build_env(params)
    ep = copy.copy(params)
    ep.enc_emb_dim = params.dec_emb_dim
    ep.n_enc_layers = params.n_dec_layers
    ep.n_enc_heads = params.n_dec_heads
    ep.n_enc_hidden_layers = params.n_dec_hidden_layers
    enc = TransformerModel(ep, env.equation_id2word, True, True, False,
                           params.dec_positional_embeddings).to(device).eval()
    enc.load_state_dict(ck["model"])
    for p in enc.parameters():
        p.requires_grad_(False)

    # 在线累积全局 sum / sumsq / count (double 精度), 只统计有效 (非 pad) token 位置
    tot_sum = torch.zeros((), device=device, dtype=torch.float64)
    tot_sumsq = torch.zeros((), device=device, dtype=torch.float64)
    tot_count = 0
    env.rng = np.random.RandomState(args.seed)
    with torch.no_grad():
        for i in range(args.n_batches):
            x, lengths = gen_batch(env, args.batch_size)
            x, lengths = x.to(device), lengths.to(device)
            tensor = enc("fwd", x=x, lengths=lengths, causal=False)  # (slen, bs, dim)
            slen = tensor.shape[0]
            valid = (torch.arange(slen, device=device).unsqueeze(1) < lengths.unsqueeze(0))  # (slen, bs)
            vals = tensor[valid].double()  # (n_valid, dim) → 展平所有元素统计
            tot_sum += vals.sum()
            tot_sumsq += (vals ** 2).sum()
            tot_count += vals.numel()
            if (i + 1) % 10 == 0:
                print(f"  batch {i+1}/{args.n_batches} | running mean={tot_sum.item()/tot_count:.4f} "
                      f"std={torch.sqrt((tot_sumsq/tot_count - (tot_sum/tot_count)**2).clamp(min=0)).item():.4f}",
                      flush=True)

    mean = (tot_sum / tot_count).item()
    std = torch.sqrt((tot_sumsq / tot_count - mean ** 2).clamp(min=0.0)).item()
    print(f"\nsampled: {args.n_batches * args.batch_size} exprs | {tot_count} elements", flush=True)
    print(f"latent_mean = {mean:.6f}")
    print(f"latent_std  = {std:.6f}")
    print(f"\n# 填入阶段 B config:")
    print(f"latent_mean: {mean:.4f}")
    print(f"latent_std: {std:.4f}")


if __name__ == "__main__":
    main()
