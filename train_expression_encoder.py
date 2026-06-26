"""阶段 A: expression encoder MLM 预训练 — 分布式 DDP 版 (ELF-SR Step 3)。

单机多卡 DDP 加速 + DataLoader 多进程预生成 (overlap CPU 生成与 GPU 训练) +
断点续训 (--resume)。启动:
  torchrun --nproc_per_node=4 train_expression_encoder.py [args]   # 4 卡
  torchrun --nproc_per_node=1 train_expression_encoder.py [args]   # 单卡(走 DDP)
  python train_expression_encoder.py [args]                         # 单卡(非 DDP)

数据: TreeDataset 无限生成表达式 batch (gen_tree_encoded 只生成树+编码, 跳过 MLM
用不到的数值点, ~6x 快于 gen_expr); DataLoader num_workers 个进程并行预取, GPU
训练时 worker 生成后续 batch。每 rank 用 seed+rank, 每 worker 用 seed+rank+wid。
val set 固定 (所有 rank 相同), 仅 rank0 打印/保存。梯度按 DDP 默认 SUM, 配合 lr
即线性 scaling (有效 batch ×world_size)。

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
from torch.utils.data import DataLoader, IterableDataset

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


def gen_tree_encoded(env):
    """只生成表达式树 + 编码成 token (跳过数值点/skeleton — MLM 用不到, ~6x 快于 gen_expr)。"""
    while True:
        tree = env.generator.generate_multi_dimensional_tree(
            rng=env.rng, nb_unary_ops=None, nb_binary_ops=None,
            input_dimension=None, output_dimension=None)[0]
        if tree is None:
            continue
        env.generator.relabel_variables(tree)
        return env.equation_encoder.encode(tree)


def gen_batch(env, batch_size):
    """生成一个 batch 的 (x, lengths): 在线生成表达式树 -> token id -> padding 对齐。"""
    tokens = [gen_tree_encoded(env) for _ in range(batch_size)]
    return env.batch_equations(env.word_to_idx(tokens, float_input=False))


class TreeDataset(IterableDataset):
    """无限生成表达式 batch: 每 worker 独立 rng, yield (x(slen,bs), lengths)。"""

    def __init__(self, env, batch_size, base_seed):
        super().__init__()
        self.env = env
        self.batch_size = batch_size
        self.base_seed = base_seed

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker else 0
        self.env.rng = np.random.RandomState(self.base_seed + wid)  # fork 后独立, 不影响父进程
        while True:
            yield gen_batch(self.env, self.batch_size)


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
    ap.add_argument("--log_every", type=int, default=100, help="每 N 步打印 train_ema (不 eval)")
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--val_size", type=int, default=1024)
    ap.add_argument("--val_batch", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=4, help="每 rank 后台生成进程数")
    ap.add_argument("--resume", type=str, default="", help="从 checkpoint 恢复继续训练")
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

    optimizer = torch.optim.Adam(mlm.parameters(), lr=args.lr)

    def lr_lambda(step):
        if step < args.warmup:
            return step / max(1, args.warmup)
        progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---- 断点续训: 恢复 model/optimizer/scheduler/step/best ----
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        enc.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        best_val, best_step, bad = ckpt["best_val"], ckpt["best_step"], ckpt["bad"]
        if is_main:
            print(f"resume from {args.resume}: step {start_step} | "
                  f"best_val {best_val:.4f}@{best_step+1} | bad {bad}/{args.patience}", flush=True)
    else:
        start_step, best_val, best_step, bad = 0, float("inf"), -1, 0

    if is_main:
        print(f"device={device} | world_size={world_size} | num_workers={args.num_workers} | "
              f"vocab {n_words} | encoder {sum(p.numel() for p in encoder.parameters())/1e6:.1f}M params | "
              f"effective batch = {args.batch_size} x {world_size} | steps {start_step}->{args.max_steps}",
              flush=True)

    # ---- 预生成验证集 (固定, 所有 rank 相同): 独立 rng + 固定 mask (主进程, 不走 DataLoader) ----
    mask_gen = torch.Generator(device=device).manual_seed(args.seed + 2)
    env.rng = np.random.RandomState(args.seed + 1)
    val_chunks = []
    n_val = 0
    while n_val < args.val_size:
        bs = min(args.val_batch, args.val_size - n_val)
        x, lengths = gen_batch(env, bs)
        x, lengths = x.to(device), lengths.to(device)
        noisy, chosen, y = mlm_mask(x, lengths, n_words, args.mask_prob, device, mask_gen)
        val_chunks.append((noisy, lengths, chosen, y))
        n_val += bs

    # ---- 训练数据: DataLoader 后台多进程预生成 (每 rank 一个 loader, base_seed=seed+rank) ----
    loader_kwargs = dict(batch_size=None, num_workers=args.num_workers,
                         pin_memory=(device.type == "cuda"))
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["multiprocessing_context"] = "fork"
    loader = DataLoader(TreeDataset(env, args.batch_size, args.seed + rank), **loader_kwargs)
    data_iter = iter(loader)

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

    def save_ckpt(path, step, val_loss):
        torch.save({
            "model": enc.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "best_val": best_val,
            "best_step": best_step,
            "bad": bad,
            "config": vars(args),
        }, path)

    # ---- 训练循环 ----
    train_ema = None
    t0 = time.time()
    for step in range(start_step, args.max_steps):
        x, lengths = next(data_iter)
        x = x.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        noisy, chosen, y = mlm_mask(x, lengths, n_words, args.mask_prob, device)

        loss = mlm(noisy, lengths, chosen, y)  # DDP forward + 自动 all-reduce 梯度
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_ema = loss.item() if train_ema is None else 0.95 * train_ema + 0.05 * loss.item()

        if (step + 1) % args.log_every == 0 and is_main:
            print(f"step {step+1:6d} | train_ema {train_ema:.4f} | "
                  f"lr {optimizer.param_groups[0]['lr']:.2e} | {(time.time()-t0)/60:.1f}min",
                  flush=True)

        if (step + 1) % args.eval_every == 0:
            if ddp:
                dist.barrier()
            val_loss = evaluate()
            is_best = val_loss < best_val
            if is_best:
                best_val, best_step, bad = val_loss, step, 0
            else:
                bad += 1
            if is_main:
                save_ckpt(os.path.join(args.out_dir, "last.pth"), step, val_loss)
                if is_best:
                    save_ckpt(os.path.join(args.out_dir, "best.pth"), step, val_loss)
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
