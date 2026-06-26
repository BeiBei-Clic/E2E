"""阶段 A: expression encoder MLM 预训练 — 分布式 DDP 版 (ELF-SR Step 3)。

单机多卡 DDP 加速。启动:
  torchrun --nproc_per_node=4 train_expression_encoder.py [args]   # 4 卡
  torchrun --nproc_per_node=1 train_expression_encoder.py [args]   # 单卡(走 DDP)
  python train_expression_encoder.py [args]                         # 单卡(非 DDP)

DDP 要求一次 forward 用到全部参数, 故把 fwd + predict 合并进 MLMEncoder 单次
forward。每 rank 用 seed+rank 独立生成训练数据 (有效 batch = batch_size ×
world_size); val set 固定 (所有 rank 相同), 仅 rank0 打印/保存。
梯度按 DDP 默认 SUM, 配合 lr 即线性 scaling (有效 batch ×world_size)。

收敛判据: 验证集 MLM loss 早停 (patience 次 eval 无 best 更新)。
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.transformer import TransformerModel


def mlm_mask(x, lengths, n_words, mask_prob, device, generator=None):
    """对 (slen, bs) token 做 MLM 遮挡: 排除首尾 <EOS> 与 pad, 随机替换为随机 token。

    返回 (noisy, chosen, y): y 为被遮挡位置的原 token, 按 row-major 展平,
    与 TransformerModel.predict 的取值顺序对齐。generator 固定时结果确定。
    """
    slen = x.shape[0]
    alen = torch.arange(slen, device=device).unsqueeze(1)
    valid = (alen > 0) & (alen < lengths.unsqueeze(0) - 1)
    chosen = valid & (torch.rand(slen, x.shape[1], generator=generator, device=device) < mask_prob)
    noisy = x.clone()
    noisy[chosen] = torch.randint(0, n_words, x.shape, generator=generator, device=device)[chosen]
    return noisy, chosen, x[chosen]


class MLMEncoder(nn.Module):
    """fwd + predict 合并为单次 forward (DDP 要求一次 forward 覆盖全部参数)。"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x, lengths, pred_mask, y):
        tensor = self.encoder("fwd", x=x, lengths=lengths, causal=False)
        _, loss = self.encoder("predict", tensor=tensor, pred_mask=pred_mask,
                               y=y, get_scores=False)
        return loss


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

    # ---- DDP 初始化 (torchrun 设 LOCAL_RANK; 否则单卡非 DDP) ----
    ddp = ("LOCAL_RANK" in os.environ) and torch.cuda.is_available() and not args.cpu
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world_size = 0, 1
        device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    is_main = (rank == 0)

    if is_main:
        os.makedirs(args.out_dir, exist_ok=True)
    np.random.seed(args.seed)

    # 环境 + 词汇表 (每 rank 独立 build, 配置相同)
    params = get_parser().parse_args([])
    params.fp16 = False
    env = build_env(params)
    n_words = env.n_words

    # expression encoder: is_encoder=True (双向), 借 dec 深参数族; 所有 rank 同 seed 保证初始权重一致
    enc_params = copy.copy(params)
    enc_params.enc_emb_dim = params.dec_emb_dim
    enc_params.n_enc_layers = params.n_dec_layers
    enc_params.n_enc_heads = params.n_dec_heads
    enc_params.n_enc_hidden_layers = params.n_dec_hidden_layers
    torch.manual_seed(args.seed)
    encoder = TransformerModel(
        enc_params,
        env.equation_id2word,
        is_encoder=True,
        with_output=True,
        use_prior_embeddings=False,
        positional_embeddings=params.dec_positional_embeddings,
    ).to(device)
    enc = encoder  # 底层 encoder (eval / 保存 用), 与 mlm 共享
    mlm = MLMEncoder(encoder)
    if ddp:
        mlm = DDP(mlm, device_ids=[local_rank])
    if is_main:
        print(f"device={device} | world_size={world_size} | vocab {n_words} | "
              f"encoder {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params | "
              f"effective batch = {args.batch_size} x {world_size}", flush=True)

    optimizer = torch.optim.Adam(mlm.parameters(), lr=args.lr)

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- 预生成验证集 (固定, 所有 rank 相同): 独立 rng + 固定 mask ----
    mask_gen = torch.Generator(device=device).manual_seed(args.seed + 2)
    env.rng = np.random.RandomState(args.seed + 1)
    val_chunks = []
    n_val = 0
    while n_val < args.val_size:
        bs = min(args.val_batch, args.val_size - n_val)
        tokens = [env.gen_expr(train=True)[0]["tree_encoded"] for _ in range(bs)]
        x, lengths = env.batch_equations(env.word_to_idx(tokens, float_input=False))
        x, lengths = x.to(device), lengths.to(device)
        noisy, chosen, y = mlm_mask(x, lengths, n_words, args.mask_prob, device, mask_gen)
        val_chunks.append((noisy, lengths, chosen, y))
        n_val += bs
    env.rng = np.random.RandomState(args.seed + rank)  # 训练数据 per-rank 不同

    def evaluate():
        enc.eval()
        tot_loss, tot_masked = 0.0, 0
        with torch.no_grad():
            for noisy, lengths, chosen, y in val_chunks:
                tensor = enc("fwd", x=noisy, lengths=lengths, causal=False)
                _, loss = enc("predict", tensor=tensor, pred_mask=chosen,
                              y=y, get_scores=False)
                n = int(chosen.sum())
                tot_loss += loss.item() * n
                tot_masked += n
        enc.train()
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
        noisy, chosen, y = mlm_mask(x, lengths, n_words, args.mask_prob, device)

        loss = mlm(noisy, lengths, chosen, y)  # DDP forward + 自动 all-reduce 梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_ema = loss.item() if train_ema is None else 0.95 * train_ema + 0.05 * loss.item()

        if (step + 1) % args.eval_every == 0:
            if ddp:
                dist.barrier()
            val_loss = evaluate()
            if val_loss < best_val:
                best_val, best_step, bad = val_loss, step, 0
                if is_main:
                    torch.save({"model": enc.state_dict(), "val_loss": val_loss,
                                "step": step, "config": vars(args)},
                               os.path.join(args.out_dir, "best.pth"))
            else:
                bad += 1
            if is_main:
                torch.save({"model": enc.state_dict(), "val_loss": val_loss,
                            "step": step, "config": vars(args)},
                           os.path.join(args.out_dir, "last.pth"))
                print(f"step {step+1:6d} | train_ema {train_ema:.4f} | val {val_loss:.4f} "
                      f"| best {best_val:.4f}@{best_step+1} | bad {bad}/{args.patience} "
                      f"| lr {optimizer.param_groups[0]['lr']:.2e} | {(time.time()-t0)/60:.1f}min",
                      flush=True)
            if bad >= args.patience:
                if is_main:
                    print(f"early stop: val loss no improvement for {args.patience} evals", flush=True)
                break

    if is_main:
        print(f"\ndone. best val_loss={best_val:.4f} @ step {best_step+1}", flush=True)
        print(f"checkpoints: {args.out_dir}/{{best,last}}.pth", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
