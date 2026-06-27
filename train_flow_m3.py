"""Step 6 / M3: denoiser flow matching 条件训练 (数值点 -> 表达式)。

M3 在 M2 (无条件) 基础上接入 condition:
  数值点 (x,y) -> 数值点 encoder (model.pt, freeze) -> cond_emb
  -> denoiser 的 cross-attention 注入 (弃 ELF prepend, 点数无关; cond 不加噪)。
expression encoder (best.pth, freeze) 仍提供去噪目标 x0。

数据: gen_expr 同时产出 (表达式树, 数值点)。
训练: 双分支 (denoise MSE + decode CE), 同 M2; denoiser 每 block [self-attn -> cross-attn -> FFN]。
eval: val_l2 + decode_acc_correct (正确 cond) vs decode_acc_shuffled (batch 内打乱 cond);
      两者差值 = condition 有效性 (M3 达标指标: correct 超过 M2 的 0.632 且差值显著正)。
encoder 前向套 bf16 autocast (freeze + no_grad, 加速, 缓解 GPU 空闲)。从头训。
启动: torchrun --nproc_per_node=N train_flow_m3.py [args]。
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
from torch.utils.data import DataLoader, IterableDataset

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.transformer import TransformerModel
from symbolicregression.flow import ELF_models
from symbolicregression.flow.flow_matching import add_noise, sample_timesteps


def gen_expr_batch(env, batch_size):
    """gen_expr 同时产出 (表达式 token, 数值点 bag)。

    bag = List[(x_list, y_list)], 一个表达式的全部数值点 (喂数值点 embedder)。
    返回 (x_tok, lengths, bags): x_tok/lengths 是表达式 token batch (slen, bs) 布局。
    """
    tree_tokens_list, bags = [], []
    for _ in range(batch_size):
        expr, _ = env.gen_expr(train=True)
        tree_tokens_list.append(expr["tree_encoded"])
        x_fit = expr["X_to_fit"][0]   # (n_points, input_dim) ndarray
        y_fit = expr["Y_to_fit"][0]   # (n_points, output_dim) ndarray
        bag = list(zip(x_fit, y_fit))
        bags.append(bag)
    x_tok, lengths = env.batch_equations(env.word_to_idx(tree_tokens_list, float_input=False))
    return x_tok, lengths, bags


class ExprPointDataset(IterableDataset):
    """无限生成 (表达式, 数值点) batch; 每 worker 独立 rng。

    worker 里顺带跑数值点 embedder 的 CPU 编码 (encode+batch, ~2.2s/批, float_encoder 逐数值
    转 descriptor token) —— 这步是主进程 enc_c 瓶颈, 不能被 GPU 训练 overlap。挪进 worker 后
    与 gen_expr 一起被多 worker overlap, 主进程只剩 GPU embed/compress + point_enc。
    worker 只调 embedder.encode/batch (纯 CPU, 不碰 GPU 权重 embed/compress)。
    """

    def __init__(self, env, embedder, batch_size, base_seed):
        super().__init__()
        self.env = env
        self.embedder = embedder
        self.batch_size = batch_size
        self.base_seed = base_seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        self.env.rng = np.random.RandomState(self.base_seed + wid)
        while True:
            x_tok, lengths, bags = gen_expr_batch(self.env, self.batch_size)
            seq_tok, seq_cond_len = self.embedder.batch(self.embedder.encode(bags))  # CPU: (slen, bs, descriptor_len), (bs,)
            yield x_tok, lengths, seq_tok, seq_cond_len


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--enc_ckpt", default="checkpoints/expression_encoder/best.pth")
    ap.add_argument("--point_ckpt", default="model.pt")
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
    ap.add_argument("--out_dir", default="checkpoints/flow_m3")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile_steps", type=int, default=0,
                    help=">0: 跑 N 步打分段计时后退出 (诊断 GPU 负载瓶颈, 单卡)")
    # flow matching 超参 (ELF 实际默认, 同 M2)
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
    use_amp = (device.type == "cuda")

    params = get_parser().parse_args([])
    params.fp16 = False
    env = build_env(params)
    n_words = env.n_words
    pad_id = env.float_word2id["<PAD>"]

    # expression encoder (freeze) — 提供去噪目标 x0
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

    # 数值点 encoder (model.pt: embedder + encoder, freeze) — 提供 condition
    mw = torch.load(args.point_ckpt, map_location="cpu", weights_only=False)
    embedder = mw.embedder.to(device).eval()
    point_enc = mw.encoder.to(device).eval()
    for m in (embedder, point_enc):
        for p in m.parameters():
            p.requires_grad_(False)

    # denoiser (ELF-B + cross-attn, M3 无 self-cond)
    torch.manual_seed(args.seed)
    denoiser = ELF_models["ELF-B"](
        text_encoder_dim=ep.enc_emb_dim, max_length=args.max_length,
        vocab_size=n_words, num_self_cond_cfg_tokens=0,
        gradient_checkpointing=True).to(device).train()
    # M3 不用 self-conditioning (denoiser_z 永远单 dim, forward 里 self_cond_proj 分支不触发),
    # 删之让所有参数 used -> 关 find_unused_parameters (省每步 graph 遍历 + 恢复 all-reduce overlap)
    del denoiser.self_cond_proj
    model = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False,
                gradient_as_bucket_view=True, static_graph=True) if ddp else denoiser

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
              f"max_length {args.max_length} (target only) | steps {start_step}->{args.max_steps}", flush=True)

    # ---- encode: 表达式 token -> x0 (bs, max_length, dim) 归一化 + input_ids + valid ----
    def encode_target(x_tok, lengths):
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            enc_out = expr_enc("fwd", x=x_tok, lengths=lengths, causal=False)  # (slen, bs, dim)
        enc_out = enc_out.float()
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

    # ---- encode: 数值点 (worker 已 encode+batch 成 CPU token) -> cond_emb + cond_mask ----
    def encode_cond(seq_tok, seq_cond_len):
        """seq_tok: (slen, bs, descriptor_len) LongTensor (worker encode+batch 产出);
        主进程只跑 GPU embed+compress (LinearPointEmbedder) + point_enc。"""
        seq_tok = seq_tok.to(device, non_blocking=True)
        seq_cond_len = seq_cond_len.to(device, non_blocking=True)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            seq_emb = embedder.compress(embedder.embed(seq_tok))               # (slen_max, bs, 512)
            cond_emb = point_enc("fwd", x=seq_emb, lengths=seq_cond_len, causal=False)
        cond_emb = cond_emb.transpose(0, 1).float()                            # (bs, slen_max, 512)
        slen_max = cond_emb.shape[1]
        cond_mask = (torch.arange(slen_max, device=device).unsqueeze(0) < seq_cond_len.unsqueeze(1)).float()
        return cond_emb, cond_mask

    # ---- 验证集 (固定 表达式+数值点, 所有 rank 相同) ----
    env.rng = np.random.RandomState(args.seed + 1)
    val_chunks = []
    n_val = 0
    while n_val < args.val_size:
        bs = min(args.val_batch, args.val_size - n_val)
        x_tok, lengths, bags = gen_expr_batch(env, bs)
        x_tok, lengths = x_tok.to(device), lengths.to(device)
        seq_tok, seq_cond_len = embedder.batch(embedder.encode(bags))
        x0, input_ids, valid = encode_target(x_tok, lengths)
        cond_emb, cond_mask = encode_cond(seq_tok, seq_cond_len)
        val_chunks.append((x0, input_ids, valid, cond_emb, cond_mask))
        n_val += bs

    # ---- 训练数据 DataLoader ----
    loader_kwargs = dict(batch_size=None, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["multiprocessing_context"] = "fork"
    loader = DataLoader(ExprPointDataset(env, embedder, args.batch_size, args.seed + rank), **loader_kwargs)
    data_iter = iter(loader)

    def evaluate():
        """返回 (val_l2, decode_acc_correct, decode_acc_shuffled)。"""
        denoiser.eval()
        tot_l2, tot_n = 0.0, 0.0
        tot_acc_c, tot_acc_s, tot_acc_n = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x0, input_ids, valid, cond_emb, cond_mask in val_chunks:
                bs = x0.shape[0]
                # val_l2 (denoise MSE)
                t = sample_timesteps(bs, args.p_mean, args.p_std, device)
                z = add_noise(x0, torch.randn_like(x0), t, args.noise_scale)
                x_pred, _ = denoiser(z, t, attention_mask=valid, cond=cond_emb, cond_mask=cond_mask)
                denom = torch.clamp(1.0 - t.reshape(-1, 1, 1), min=args.t_eps)
                l2 = (((x_pred - z) / denom - (x0 - z) / denom) ** 2).mean(-1)
                v = valid.float()
                tot_l2 += (l2 * v).sum().item()
                tot_n += v.sum().item()
                # decode 分支 (t=1): 同一 decoder_z, 对比正确 cond vs 打乱 cond
                lam = torch.sigmoid(torch.randn(bs, args.max_length, 1, device=device) * args.p_std + args.p_mean)
                decoder_z = lam * x0 + (1 - lam) * (torch.randn_like(x0) * args.noise_scale)
                _, logits_c = denoiser(decoder_z, torch.ones(bs, device=device),
                                       attention_mask=valid, cond=cond_emb, cond_mask=cond_mask,
                                       decoder_step_active=True)
                perm = torch.randperm(bs, device=device)
                _, logits_s = denoiser(decoder_z, torch.ones(bs, device=device),
                                       attention_mask=valid, cond=cond_emb[perm], cond_mask=cond_mask[perm],
                                       decoder_step_active=True)
                tot_acc_c += (logits_c.argmax(-1) == input_ids)[valid].float().sum().item()
                tot_acc_s += (logits_s.argmax(-1) == input_ids)[valid].float().sum().item()
                tot_acc_n += valid.sum().item()
        denoiser.train()
        return tot_l2 / tot_n, tot_acc_c / max(tot_acc_n, 1.0), tot_acc_s / max(tot_acc_n, 1.0)

    def save_ckpt(path, step, val_loss):
        torch.save({
            "model": denoiser.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "step": step, "val_loss": val_loss,
            "best_val": best_val, "best_step": best_step, "bad": bad, "config": vars(args),
        }, path)

    # ---- 训练循环 ----
    train_ema = None
    t0 = time.time()
    prof = args.profile_steps > 0
    for step in range(start_step, args.max_steps):
        if prof:
            torch.cuda.synchronize(); _t = [time.perf_counter()]
        x_tok, lengths, seq_tok, seq_cond_len = next(data_iter)
        x_tok, lengths = x_tok.to(device, non_blocking=True), lengths.to(device, non_blocking=True)
        if prof:
            torch.cuda.synchronize(); _t.append(time.perf_counter())  # data(gen_expr+encode, worker overlap)
        x0, input_ids, valid = encode_target(x_tok, lengths)
        if prof:
            torch.cuda.synchronize(); _t.append(time.perf_counter())  # enc_target
        cond_emb, cond_mask = encode_cond(seq_tok, seq_cond_len)
        bs = x0.shape[0]
        if prof:
            torch.cuda.synchronize(); _t.append(time.perf_counter())  # enc_cond

        t = sample_timesteps(bs, args.p_mean, args.p_std, device)
        denoiser_z = add_noise(x0, torch.randn_like(x0), t, args.noise_scale)   # M3: cond 物理分离, 无 cond_seq_mask
        lam = torch.sigmoid(torch.randn(bs, args.max_length, 1, device=device) * args.p_std + args.p_mean)
        decoder_z = lam * x0 + (1 - lam) * (torch.randn_like(x0) * args.noise_scale)
        ds = torch.bernoulli(torch.full((bs,), args.decoder_prob, device=device)).view(-1, 1, 1)
        z_mixed = ds * decoder_z + (1 - ds) * denoiser_z
        t_mixed = ds.view(-1) * 1.0 + (1 - ds.view(-1)) * t
        if prof:
            torch.cuda.synchronize(); _t.append(time.perf_counter())  # 采样 + 加噪

        x_pred, logits = model(z_mixed, t_mixed, attention_mask=valid,
                               cond=cond_emb, cond_mask=cond_mask, decoder_step_active=ds.view(-1))
        if prof:
            torch.cuda.synchronize(); _t.append(time.perf_counter())  # denoiser fwd
        denom = torch.clamp(1.0 - t.reshape(-1, 1, 1), min=args.t_eps)
        l2 = (((x_pred - denoiser_z) / denom - (x0 - denoiser_z) / denom) ** 2).mean(-1)
        ce = F.cross_entropy(logits.transpose(1, 2), input_ids, reduction="none")
        ds1 = ds.view(-1, 1)
        v = valid.float()
        loss = ((ce * v * ds1).sum() + (l2 * v * (1 - ds1)).sum()) / v.sum().clamp(min=1.0)

        optimizer.zero_grad()
        loss.backward()
        if prof:
            torch.cuda.synchronize(); _t.append(time.perf_counter())  # bwd
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        if prof:
            torch.cuda.synchronize(); _t.append(time.perf_counter())  # opt
            d = [(_t[i + 1] - _t[i]) * 1000 for i in range(len(_t) - 1)]
            print(f"[prof] step {step+1:4d} | data {d[0]:6.1f} | enc_t {d[1]:6.1f} | enc_c {d[2]:6.1f} "
                  f"| noise {d[3]:6.1f} | fwd {d[4]:6.1f} | bwd {d[5]:6.1f} | opt {d[6]:6.1f} | "
                  f"sum {sum(d):6.1f}ms", flush=True)
            if step + 1 >= args.profile_steps:
                break
            continue
        train_ema = loss.item() if train_ema is None else 0.95 * train_ema + 0.05 * loss.item()

        if (step + 1) % args.log_every == 0 and is_main:
            print(f"step {step+1:6d} | train_ema {train_ema:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | {(time.time()-t0)/60:.1f}min", flush=True)

        if (step + 1) % args.eval_every == 0:
            if ddp:
                dist.barrier()
            val_l2_val, acc_c, acc_s = evaluate()
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
                      f"| acc_correct {acc_c:.3f} | acc_shuffled {acc_s:.3f} | Δcond {acc_c-acc_s:+.3f} "
                      f"| best {best_val:.4f}@{best_step+1} | bad {bad}/{args.patience} "
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
