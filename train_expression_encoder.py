"""阶段 A: expression encoder MLM 预训练 (ELF-SR Step 3)。

完整训练: 在线生成训练数据 + 预生成独立验证集 (固定 mask) + val loss 早停 +
warmup/cosine lr + best/last checkpoint。

收敛判据: 验证集 MLM loss 早停 (patience 次 eval 无 best 更新)。验证集用独立
rng 预生成、mask 位置与目标 token 固定, val loss 曲线逐点可比。

注意: TransformerModel 的 is_encoder 开关同时绑定参数族 (is_encoder=True 读
enc_*, 默认仅 2 层)。想要「双向 + 深」, 需把 enc_* 对齐到 dec_*。
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.transformer import TransformerModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val_size", type=int, default=1024)
    ap.add_argument("--val_batch", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="checkpoints/expression_encoder")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    # 环境 + 词汇表
    params = get_parser().parse_args([])
    params.fp16 = False
    env = build_env(params)
    n_words = env.n_words
    print(f"device={device} | equation vocab: {n_words} words", flush=True)

    # expression encoder: is_encoder=True (双向), 借 dec 深参数族
    enc_params = copy.copy(params)
    enc_params.enc_emb_dim = params.dec_emb_dim
    enc_params.n_enc_layers = params.n_dec_layers
    enc_params.n_enc_heads = params.n_dec_heads
    enc_params.n_enc_hidden_layers = params.n_dec_hidden_layers
    encoder = TransformerModel(
        enc_params,
        env.equation_id2word,
        is_encoder=True,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    ).to(device).train()
    print(f"expression encoder: {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params", flush=True)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- 预生成独立验证集 (固定 mask): 独立 rng, 与训练数据不同 seed ----
    mask_gen = torch.Generator(device=device).manual_seed(args.seed + 2)
    env.rng = np.random.RandomState(args.seed + 1)
    val_chunks = []
    n_val = 0
    while n_val < args.val_size:
        bs = min(args.val_batch, args.val_size - n_val)
        tokens = [env.gen_expr(train=True)[0]["tree_encoded"] for _ in range(bs)]
        x, lengths = env.batch_equations(env.word_to_idx(tokens, float_input=False))
        x, lengths = x.to(device), lengths.to(device)
        slen = x.shape[0]
        alen = torch.arange(slen, device=device).unsqueeze(1)
        valid = (alen > 0) & (alen < lengths.unsqueeze(0) - 1)  # 排除首尾 <EOS> 与 pad
        chosen = valid & (torch.rand(slen, bs, generator=mask_gen, device=device) < args.mask_prob)
        noisy = x.clone()
        noisy[chosen] = torch.randint(0, n_words, (slen, bs), generator=mask_gen, device=device)[chosen]
        val_chunks.append((noisy, lengths, chosen, x[chosen]))
        n_val += bs
    env.rng = np.random.RandomState(args.seed)  # 恢复为训练 rng
    n_masked_val = sum(int(c[2].sum()) for c in val_chunks)
    print(f"val set: {n_val} exprs, {n_masked_val} masked tokens (fixed)", flush=True)

    def evaluate():
        encoder.eval()
        tot_loss, tot_masked = 0.0, 0
        with torch.no_grad():
            for noisy, lengths, chosen, y in val_chunks:
                tensor = encoder("fwd", x=noisy, lengths=lengths, causal=False)
                _, loss = encoder("predict", tensor=tensor, pred_mask=chosen,
                                  y=y, get_scores=False)
                n = int(chosen.sum())
                tot_loss += loss.item() * n
                tot_masked += n
        encoder.train()
        return tot_loss / tot_masked

    # ---- 训练循环 ----
    best_val = float("inf")
    best_step = -1
    bad = 0
    train_ema = None
    t0 = time.time()
    for step in range(args.max_steps):
        tokens = [env.gen_expr(train=True)[0]["tree_encoded"] for _ in range(args.batch_size)]
        x, lengths = env.batch_equations(env.word_to_idx(tokens, float_input=False))
        x, lengths = x.to(device), lengths.to(device)
        slen, bs = x.shape
        alen = torch.arange(slen, device=device).unsqueeze(1)
        valid = (alen > 0) & (alen < lengths.unsqueeze(0) - 1)
        chosen = valid & (torch.rand(slen, bs, device=device) < args.mask_prob)
        noisy = x.clone()
        noisy[chosen] = torch.randint(0, n_words, (slen, bs), device=device)[chosen]
        y = x[chosen]

        tensor = encoder("fwd", x=noisy, lengths=lengths, causal=False)
        _, loss = encoder("predict", tensor=tensor, pred_mask=chosen,
                          y=y, get_scores=False)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_ema = loss.item() if train_ema is None else 0.95 * train_ema + 0.05 * loss.item()

        if (step + 1) % args.eval_every == 0:
            val_loss = evaluate()
            if val_loss < best_val:
                best_val, best_step, bad = val_loss, step, 0
                torch.save({"model": encoder.state_dict(), "val_loss": val_loss,
                            "step": step, "config": vars(args)},
                           os.path.join(args.out_dir, "best.pth"))
            else:
                bad += 1
            torch.save({"model": encoder.state_dict(), "val_loss": val_loss,
                        "step": step, "config": vars(args)},
                       os.path.join(args.out_dir, "last.pth"))
            print(f"step {step+1:6d} | train_ema {train_ema:.4f} | val {val_loss:.4f} "
                  f"| best {best_val:.4f}@{best_step+1} | bad {bad}/{args.patience} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} | {(time.time()-t0)/60:.1f}min",
                  flush=True)
            if bad >= args.patience:
                print(f"early stop: val loss no improvement for {args.patience} evals", flush=True)
                break

    print(f"\ndone. best val_loss={best_val:.4f} @ step {best_step+1}", flush=True)
    print(f"checkpoints: {args.out_dir}/{{best,last}}.pth", flush=True)


if __name__ == "__main__":
    main()
