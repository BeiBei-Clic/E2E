"""Step 6 / M2: denoiser flow matching 训练 (无条件最小版)。

不加 condition、不加 self-cond/CFG。验证 flow matching 在 expression embedding
空间能训起来 (denoise loss 下降 + decode acc 上升 → 产出合法 token)。

数据: 表达式 token -> expression encoder (best.pth, freeze, 归一化) -> x0
  x0 (bs, max_length, dim), pad 到 max_length (denoiser RoPE 固定长度约束)
训练: 双分支 (每 example bernoulli(decoder_prob) 选)
  denoise (1-p): z=add_noise(x0); denoiser(z,t)->x_pred; MSE((x_pred-z)/(1-t), (x0-z)/(1-t))
  decode  (p)  : decoder_z=lambda*x0+(1-lambda)*noise (logit-normal); CE(unembed, input_ids)
eval: val_l2 (denoise MSE) + decode_acc (decode 分支 argmax vs input_ids, M2 达标指标)
启动同 train_expression_encoder.py (torchrun --nproc_per_node=N)。
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.transformer import TransformerModel
from symbolicregression.flow import ELF_models
from symbolicregression.flow.flow_matching import add_noise, sample_timesteps
from train_expression_encoder import gen_batch, TreeDataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enc_ckpt", default="checkpoints/expression_encoder/best.pth")
    ap.add_argument("--max_steps", type=int, default=200000)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val_size", type=int, default=1024)
    ap.add_argument("--val_batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--resume", default="")
    ap.add_argument("--out_dir", default="checkpoints/flow_m2")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    # flow matching 超参 (ELF 实际默认)
    ap.add_argument("--latent_mean", type=float, default=-0.0004)
    ap.add_argument("--latent_std", type=float, default=0.9942)
    ap.add_argument("--p_mean", type=float, default=0.8)
    ap.add_argument("--p_std", type=float, default=0.8)
    ap.add_argument("--noise_scale", type=float, default=1.0)
    ap.add_argument("--t_eps", type=float, default=5e-2)
    ap.add_argument("--decoder_prob", type=float, default=0.5)
    args = ap.parse_args()

    # ---- DDP 初始化 ----
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

    params = get_parser().parse_args([])
    params.fp16 = False
    env = build_env(params)
    n_words = env.n_words
    pad_id = env.float_word2id["<PAD>"]

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

    # denoiser (ELF-B, M2 无 self-cond)
    torch.manual_seed(args.seed)
    denoiser = ELF_models["ELF-B"](
        text_encoder_dim=ep.enc_emb_dim, max_length=args.max_length,
        vocab_size=n_words, num_self_cond_cfg_tokens=0,
        gradient_checkpointing=True).to(device).train()
    model = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=True) if ddp else denoiser

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        denoiser.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_step = ck["step"] + 1
        best_val, best_step, bad = ck["best_val"], ck["best_step"], ck["bad"]
        if is_main:
            print(f"resume from {args.resume}: step {start_step} best_val {best_val:.4f}@{best_step+1}", flush=True)
    else:
        start_step, best_val, best_step, bad = 0, float("inf"), -1, 0

    if is_main:
        print(f"device={device} | world_size={world_size} | num_workers={args.num_workers} | "
              f"vocab {n_words} | denoiser {sum(p.numel() for p in denoiser.parameters())/1e6:.1f}M | "
              f"max_length {args.max_length} | steps {start_step}->{args.max_steps}", flush=True)

    # token -> x0 (bs, max_length, dim) 归一化 + input_ids + valid
    def encode_batch(x_tok, lengths):
        with torch.no_grad():
            enc_out = expr_enc("fwd", x=x_tok, lengths=lengths, causal=False)  # (slen, bs, dim)
        pad_len = args.max_length - enc_out.shape[0]
        if pad_len > 0:
            enc_out = F.pad(enc_out, (0, 0, 0, 0, 0, pad_len))
            x_tok = F.pad(x_tok, (0, 0, 0, pad_len), value=pad_id)
        elif pad_len < 0:
            enc_out = enc_out[:args.max_length]
            x_tok = x_tok[:args.max_length]
            lengths = lengths.clamp(max=args.max_length)
        x0 = (enc_out.transpose(0, 1) - args.latent_mean) / args.latent_std
        input_ids = x_tok.transpose(0, 1).long()
        valid = (torch.arange(args.max_length, device=device).unsqueeze(1) < lengths.unsqueeze(0)).transpose(0, 1)
        return x0, input_ids, valid

    # ---- 验证集 (固定表达式 -> x0 + input_ids, 所有 rank 相同) ----
    env.rng = np.random.RandomState(args.seed + 1)
    val_chunks = []
    n_val = 0
    while n_val < args.val_size:
        bs = min(args.val_batch, args.val_size - n_val)
        x_tok, lengths = gen_batch(env, bs)
        x_tok, lengths = x_tok.to(device), lengths.to(device)
        x0, input_ids, valid = encode_batch(x_tok, lengths)
        val_chunks.append((x0, input_ids, valid))
        n_val += bs

    # ---- 训练数据 DataLoader ----
    loader_kwargs = dict(batch_size=None, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["multiprocessing_context"] = "fork"
    loader = DataLoader(TreeDataset(env, args.batch_size, args.seed + rank), **loader_kwargs)
    data_iter = iter(loader)

    def evaluate():
        """返回 (val_l2 denoise MSE, decode_acc decode 分支 argmax acc)。"""
        denoiser.eval()
        tot_l2, tot_n = 0.0, 0.0
        tot_acc, tot_acc_n = 0.0, 0.0
        with torch.no_grad():
            for x0, input_ids, valid in val_chunks:
                bs = x0.shape[0]
                # denoise val_l2
                t = sample_timesteps(bs, args.p_mean, args.p_std, device)
                z = add_noise(x0, torch.randn_like(x0), t, args.noise_scale)
                x_pred, _ = denoiser(z, t, attention_mask=valid)
                denom = torch.clamp(1.0 - t.reshape(-1, 1, 1), min=args.t_eps)
                l2 = (((x_pred - z) / denom - (x0 - z) / denom) ** 2).mean(-1)
                v = valid.float()
                tot_l2 += (l2 * v).sum().item()
                tot_n += v.sum().item()
                # decode_acc: decode 分支 (t=1) argmax vs input_ids (valid 位置)
                lam = torch.sigmoid(torch.randn(bs, args.max_length, 1, device=device) * args.p_std + args.p_mean)
                decoder_z = lam * x0 + (1 - lam) * (torch.randn_like(x0) * args.noise_scale)
                _, logits = denoiser(decoder_z, torch.ones(bs, device=device),
                                     attention_mask=valid, decoder_step_active=True)
                tot_acc += (logits.argmax(-1) == input_ids)[valid].float().sum().item()
                tot_acc_n += valid.sum().item()
        denoiser.train()
        return tot_l2 / tot_n, tot_acc / max(tot_acc_n, 1.0)

    def save_ckpt(path, step, val_loss):
        torch.save({
            "model": denoiser.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "step": step, "val_loss": val_loss,
            "best_val": best_val, "best_step": best_step, "bad": bad, "config": vars(args),
        }, path)

    # ---- 训练循环 ----
    train_ema = None
    t0 = time.time()
    for step in range(start_step, args.max_steps):
        x_tok, lengths = next(data_iter)
        x_tok, lengths = x_tok.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
        x0, input_ids, valid = encode_batch(x_tok, lengths)
        bs = x0.shape[0]

        t = sample_timesteps(bs, args.p_mean, args.p_std, device)
        denoiser_z = add_noise(x0, torch.randn_like(x0), t, args.noise_scale)
        lam = torch.sigmoid(torch.randn(bs, args.max_length, 1, device=device) * args.p_std + args.p_mean)
        decoder_z = lam * x0 + (1 - lam) * (torch.randn_like(x0) * args.noise_scale)
        ds = torch.bernoulli(torch.full((bs,), args.decoder_prob, device=device)).view(-1, 1, 1)
        z_mixed = ds * decoder_z + (1 - ds) * denoiser_z
        t_mixed = ds.view(-1) * 1.0 + (1 - ds.view(-1)) * t

        x_pred, logits = model(z_mixed, t_mixed, attention_mask=valid, decoder_step_active=ds.view(-1))
        denom = torch.clamp(1.0 - t.reshape(-1, 1, 1), min=args.t_eps)
        l2 = (((x_pred - denoiser_z) / denom - (x0 - denoiser_z) / denom) ** 2).mean(-1)
        ce = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        ds1 = ds.view(-1, 1)
        v = valid.float()
        loss = ((ce * v * ds1).sum() + (l2 * v * (1 - ds1)).sum()) / v.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        train_ema = loss.item() if train_ema is None else 0.95 * train_ema + 0.05 * loss.item()

        if (step + 1) % args.log_every == 0 and is_main:
            print(f"step {step+1:6d} | train_ema {train_ema:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | {(time.time()-t0)/60:.1f}min", flush=True)

        if (step + 1) % args.eval_every == 0:
            if ddp:
                dist.barrier()
            val_l2_val, decode_acc = evaluate()
            is_best = val_l2_val < best_val
            if is_best:
                best_val, best_step, bad = val_l2_val, step, 0
            else:
                bad += 1
            if is_main:
                save_ckpt(os.path.join(args.out_dir, "last.pth"), step, val_l2_val)
                if is_best:
                    save_ckpt(os.path.join(args.out_dir, "best.pth"), step, val_l2_val)
                print(f"step {step+1:6d} | train_ema {train_ema:.4f} | val_l2 {val_l2_val:.4f} "
                      f"| decode_acc {decode_acc:.3f} | best {best_val:.4f}@{best_step+1} | bad {bad}/{args.patience} "
                      f"| lr {optimizer.param_groups[0]['lr']:.2e} | {(time.time()-t0)/60:.1f}min", flush=True)
            if bad >= args.patience:
                if is_main:
                    print(f"early stop: val_l2 no improvement for {args.patience} evals", flush=True)
                break

    if is_main:
        print(f"\ndone. best val_l2={best_val:.4f} @ step {best_step+1}", flush=True)
        print(f"checkpoints: {args.out_dir}/{{best,last}}.pth", flush=True)
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
