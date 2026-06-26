"""阶段 A: expression encoder MLM 预训练脚手架 (ELF-SR Step 2)。

验证「双向 TransformerModel(is_encoder=True) + MLM」机制跑通且 loss 下降:
  - 数据: env.gen_expr 在线生成表达式 -> token id (slen, bs)
  - MLM: 随机遮挡有效位置 (替换成随机 token), 还原原 token; 复用
         TransformerModel 自带的 self.proj + predict() 做预测头, 无需新建临时头
  - encoder 输出的 (slen, bs, dim) 即后续 flow matching 的 clean embedding x

注意: TransformerModel 的 is_encoder 开关同时绑定参数族 (is_encoder=True 读
enc_*, 默认仅 2 层)。想要「双向 + 深」, 需把 enc_* 对齐到 dec_*。
"""
import argparse
import copy

import numpy as np
import torch

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.transformer import TransformerModel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_steps", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    # 环境 + 词汇表
    params = get_parser().parse_args([])
    params.fp16 = False
    env = build_env(params)
    env.rng = np.random.RandomState(args.seed)
    n_words = env.n_words
    print(f"device={device} | equation vocab: {n_words} words")

    # expression encoder: is_encoder=True (双向), 借 dec 的深参数族
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
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"expression encoder: {n_params/1e6:.1f}M params | "
          f"{enc_params.n_enc_layers}L x d={enc_params.enc_emb_dim} h={enc_params.n_enc_heads}")

    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    pad_index = env.equation_word2id["<PAD>"]

    losses = []
    for step in range(args.n_steps):
        # 在线生成一批表达式 token (slen, bs)
        tokens = [env.gen_expr(train=True)[0]["tree_encoded"] for _ in range(args.batch_size)]
        x, lengths = env.batch_equations(env.word_to_idx(tokens, float_input=False))
        x, lengths = x.to(device), lengths.to(device)

        # MLM mask: 排除首尾 <EOS> 与 pad, 从有效位置随机遮挡 mask_prob
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
        losses.append(loss.item())
        if step % 5 == 0 or step == args.n_steps - 1:
            print(f"step {step:3d}  mlm_loss={loss.item():.4f}  n_masked={int(chosen.sum())}")

    half = max(1, len(losses) // 2)
    first, last = float(np.mean(losses[:half])), float(np.mean(losses[half:]))
    print(f"\nloss trend: first-half={first:.4f}  last-half={last:.4f}  delta={last-first:+.4f}")
    print("SCAFFOLD OK" if last < first else "WARNING: loss not decreasing")


if __name__ == "__main__":
    main()
