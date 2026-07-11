"""双头 GRPO 后训练 (ADR 0002 修正版): flow 头 Flow-DPPO + decode 头 token-GRPO。

修复 ADR 0002 原版(纯 latent Flow-DPPO)的根因: decode 头不在梯度路径。
reward = decode(token) 后骨架的 BFGS-R², 在 token 空间 -> decode 头必须接收 reward。
双头 loss (coherent, 都是 GRPO 的 ratio·adv):
  flow 头: Flow-DPPO (latent logp, KL-ADV mask)
  decode 头: token-GRPO (logp_tok = Σ log p(token_i|z_final), ratio·adv)

rollout: flow SDE → z_final → decode 温度采样 token (τ, 记 logp_tok_old) → reward(BFGS, 32进程)
advantage: group G=32 内归一化 (group-mean + batch-std)
decode 温度采样保证 group 内 32 序列不同 -> 能 reward 排序。

Step 1-2 (当前): 加载 + 数据 + Flow-SDE rollout + decode 温度采样, 验证 group 内多样性。
后续: 3.reward 4.双头 loss 5.诊断。
"""
import argparse
import copy
import os
import re
import time
from multiprocessing import Pool

import numpy as np
import torch
from scipy.optimize import minimize as _scipy_min

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.flow import ELF_models

LATENT_MEAN, LATENT_STD = -0.0004, 0.9942
NOISE_SCALE, T_EPS = 1.0, 5e-2


# ---- reward 辅助 (worker 只用这些 + scipy, 不需 env/GPU) ----
def _val1d(tree, x):
    yh = tree.val(x)
    return yh[:, 0] if (isinstance(yh, np.ndarray) and yh.ndim == 2) else yh

def tree_to_skeleton(tree):
    if hasattr(tree, "nodes"):
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

def _set_constants(skeleton, coeffs):
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
                         options={"maxiter": 50 * len(coeffs0), "xatol": 1e-5, "fatol": 1e-9})
        yh = _val1d(_set_constants(skeleton, res.x), x_fit)
    if not np.all(np.isfinite(yh)):
        return float("nan")
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - yh) ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")

def _reward_one(task):
    pred_tree, gt_tree, x_fit = task
    if pred_tree is None:
        return 0.0
    r = tree_r2_bfgs(pred_tree, gt_tree, x_fit)
    return float(max(0.0, r)) if np.isfinite(r) else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/m3/best.pth")
    ap.add_argument("--point_ckpt", default="model.pt")
    ap.add_argument("--G", type=int, default=32)
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eta", type=float, default=0.7)
    ap.add_argument("--kl_mask_threshold", type=float, default=1e-3)
    ap.add_argument("--decode_temp", type=float, default=1.0, help="decode 温度采样 τ")
    ap.add_argument("--num_updates", type=int, default=2)
    ap.add_argument("--max_steps", type=int, default=500)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--out_dir", default="checkpoints/m3_rl")
    ap.add_argument("--resume", default="", help="resume from ckpt (model+optimizer+step+r_ema)")
    ap.add_argument("--reward_workers", type=int, default=32)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = get_parser().parse_args([])
    params.fp16 = False
    env = build_env(params)
    env.rng = np.random.RandomState(args.seed)
    text_encoder_dim = params.dec_emb_dim
    eos_id = env.equation_word2id["<EOS>"]

    mw = torch.load(args.point_ckpt, map_location="cpu", weights_only=False)
    embedder = mw.embedder.to(device).eval()
    point_enc = mw.encoder.to(device).eval()
    for m in (embedder, point_enc):
        for p in m.parameters():
            p.requires_grad_(False)

    denoiser = ELF_models["ELF-B"](
        text_encoder_dim=text_encoder_dim, max_length=args.max_length,
        vocab_size=env.n_words, num_self_cond_cfg_tokens=0).to(device)
    del denoiser.self_cond_proj
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    denoiser.load_state_dict(ck["model"])
    denoiser.eval()
    print(f"loaded {args.ckpt}: step {ck['step']+1} val_l2 {ck.get('val_loss','?'):.4f} | "
          f"denoiser {sum(p.numel() for p in denoiser.parameters())/1e6:.1f}M | "
          f"G{args.G} B{args.B} K{args.K} eta{args.eta} tau{args.kl_mask_threshold} temp{args.decode_temp}", flush=True)

    pool = Pool(args.reward_workers)
    dt = 1.0 / args.K
    record_steps = sorted(range(args.K // 2))

    def gen_prompts(n):
        prompts = []
        for _ in range(n):
            expr, _ = env.gen_expr(train=True)
            x_fit = expr["X_to_fit"][0]
            bags = list(zip(x_fit, expr["Y_to_fit"][0]))
            prompts.append((expr["tree"], x_fit, bags))
        return prompts

    def encode_cond(bags_list):
        seq_tok, seq_cond_len = embedder.batch(embedder.encode(bags_list))
        seq_tok = seq_tok.to(device); seq_cond_len = seq_cond_len.to(device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            seq_emb = embedder.compress(embedder.embed(seq_tok))
            cond_emb = point_enc("fwd", x=seq_emb, lengths=seq_cond_len, causal=False)
        cond_emb = cond_emb.transpose(0, 1).float()
        slen = cond_emb.shape[1]
        cond_mask = (torch.arange(slen, device=device).unsqueeze(0) < seq_cond_len.unsqueeze(1)).float()
        return cond_emb, cond_mask

    def rollout(cond_emb, cond_mask):
        """no_grad eval。返回 (ids_sampled, logp_tok_old, seg, cond_g, cond_mask_g)。
        seg[k]=(z_in,t_k,mu_old,sigma,z_next) for flow Flow-DPPO。
        ids_sampled: 温度采样 token (BG,ml); logp_tok_old: 采样时 Σ log p (BG,)。"""
        B = cond_emb.shape[0]
        BG = B * args.G
        cond_g = cond_emb.repeat_interleave(args.G, dim=0)
        cond_mask_g = cond_mask.repeat_interleave(args.G, dim=0)
        valid = torch.ones(BG, args.max_length, dtype=torch.bool, device=device)
        z = torch.randn(BG, args.max_length, text_encoder_dim, device=device) * NOISE_SCALE
        seg = []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            for k in range(args.K):
                t_k = k * dt
                t_vec = torch.full((BG,), t_k, device=device)
                x_pred, _ = denoiser(z, t_vec, attention_mask=valid, cond=cond_g, cond_mask=cond_mask_g)
                v = (x_pred.float() - z) / max(1.0 - t_k, T_EPS)
                mu = z + dt * v
                sigma = args.eta * (dt ** 0.5) * (max(1.0 - t_k, 0.0) ** 0.5)
                eps = torch.randn_like(z)
                z_next = mu + sigma * eps
                if k in record_steps:
                    seg.append((z, t_k, mu.float(), sigma, z_next))
                z = z_next
            # 末步 decode: 温度采样 (非 argmax) + 记 logp_tok_old
            _, logits = denoiser(z, torch.ones(BG, device=device),
                                 attention_mask=valid, cond=cond_g, cond_mask=cond_mask_g,
                                 decoder_step_active=True)
            logits = logits.float() / args.decode_temp
            probs = torch.softmax(logits, dim=-1)                      # (BG, ml, vocab)
            ids_sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(probs.shape[:-1])
            logp_tok_old = torch.log_softmax(logits, dim=-1).gather(-1, ids_sampled.unsqueeze(-1)).squeeze(-1).sum(-1)  # (BG,)
        return ids_sampled, logp_tok_old, seg, cond_g, cond_mask_g, z

    # ---- 训练循环 (双头 GRPO: flow Flow-DPPO + decode token-GRPO) ----
    def ids_to_tree(ids_row):
        seq = ids_row.tolist()
        cut = next((j for j in range(1, len(seq)) if seq[j] == eos_id), len(seq))
        words = [env.equation_id2word[int(t)] for t in seq[1:cut]]
        return env.word_to_infix(words, is_float=False, str_array=False)

    optimizer = torch.optim.Adam(denoiser.parameters(), lr=args.lr)
    init_sd = copy.deepcopy(denoiser.state_dict())           # 诊断基线 = best.pth (始终, 不随 resume 变)
    os.makedirs(args.out_dir, exist_ok=True)
    reward_ema, best_r, start_step = None, -1.0, 0
    if args.resume:
        rck = torch.load(args.resume, map_location="cpu", weights_only=False)
        denoiser.load_state_dict(rck["model"])
        start_step = rck["step"]
        reward_ema = rck.get("reward_ema")
        best_r = rck.get("reward_ema", -1.0)
        if "optimizer" in rck:
            optimizer.load_state_dict(rck["optimizer"])
        print(f"resume from {args.resume}: step {start_step} r_ema {reward_ema:.3f}", flush=True)
    t_start = time.time()
    for step in range(start_step, args.max_steps):
        t0 = time.time()
        denoiser.eval()
        prompts = gen_prompts(args.B)
        cond_emb, cond_mask = encode_cond([p[2] for p in prompts])
        ids, logp_tok_old, seg, cond_g, cond_mask_g, z_final = rollout(cond_emb, cond_mask)
        BG = args.B * args.G
        valid_g = torch.ones(BG, args.max_length, dtype=torch.bool, device=device)

        pred_trees = [ids_to_tree(ids[i]) for i in range(BG)]
        tasks = [(pred_trees[i], prompts[i // args.G][0], prompts[i // args.G][1]) for i in range(BG)]
        rewards = np.array(pool.map(_reward_one, tasks))

        rewards_pg = rewards.reshape(args.B, args.G)
        batch_std = rewards.std()
        if batch_std < 1e-6:
            print(f"step {step+1}: 全 batch reward std=0, 跳过", flush=True)
            continue
        group_mean = rewards_pg.mean(axis=1, keepdims=True)
        adv = torch.tensor(((rewards_pg - group_mean) / batch_std).reshape(-1), dtype=torch.float32, device=device)
        logp_tok_old_t = logp_tok_old.detach().to(device)
        z_final_g = z_final.detach()
        ids_g = ids.to(device)

        # prepare flow old anchor (no_grad, 冻结 π_old)
        anchor = []
        with torch.no_grad():
            for (z_in, t_k, mu_old, sigma, z_next) in seg:
                z_in_g = z_in.to(device); z_next_g = z_next.to(device); mu_old_g = mu_old.to(device)
                old_logp_lat = -((z_next_g - mu_old_g).pow(2) / (2 * sigma * sigma)).mean(dim=[1, 2])
                anchor.append((old_logp_lat, mu_old_g, z_in_g, z_next_g, sigma, t_k))

        m_ratio_f, m_kl_f, m_masked_f, m_loss_f = [], [], [], []
        m_ratio_t, m_loss_t = [], []
        for _ in range(args.num_updates):
            optimizer.zero_grad()
            # flow 头 Flow-DPPO (分块 forward+backward, 防 BG=256 OOM; 梯度累积等价全量 mean/len(anchor))
            chunk_f = 64
            for (old_logp_lat, mu_old_g, z_in_g, z_next_g, sigma, t_k) in anchor:
                for c0 in range(0, BG, chunk_f):
                    c1 = min(c0 + chunk_f, BG)
                    t_vec = torch.full((c1 - c0,), t_k, device=device)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        x_pred, _ = denoiser(z_in_g[c0:c1], t_vec, attention_mask=valid_g[c0:c1],
                                             cond=cond_g[c0:c1], cond_mask=cond_mask_g[c0:c1])
                    v = (x_pred.float() - z_in_g[c0:c1]) / max(1.0 - t_k, T_EPS)
                    mu_new = z_in_g[c0:c1] + dt * v
                    new_logp_lat = -((z_next_g[c0:c1] - mu_new).pow(2) / (2 * sigma * sigma)).mean(dim=[1, 2])
                    kl = ((mu_new - mu_old_g[c0:c1]).pow(2) / (2 * sigma * sigma)).mean(dim=[1, 2])
                    ratio_f = torch.exp(new_logp_lat - old_logp_lat[c0:c1])
                    kl_mask = kl < args.kl_mask_threshold
                    pos_rm = (~kl_mask) & (ratio_f > 1.0) & (adv[c0:c1] > 0)
                    neg_rm = (~kl_mask) & (ratio_f < 1.0) & (adv[c0:c1] < 0)
                    keep = ~(pos_rm | neg_rm)
                    loss_f_c = torch.where(keep, -adv[c0:c1] * ratio_f, torch.zeros_like(ratio_f)).sum() / (BG * len(anchor))
                    loss_f_c.backward()
                    m_ratio_f.append(ratio_f.detach().mean().item())
                    m_kl_f.append(kl.detach().mean().item())
                    m_masked_f.append((~keep).float().mean().item())
                    m_loss_f.append(loss_f_c.item())
            # decode 头 token-GRPO (分块 forward+backward, 防 BG=256 OOM; 梯度累积等价全量 mean)
            chunk = 64
            ratio_t_list, loss_t_total = [], 0.0
            for c0 in range(0, BG, chunk):
                c1 = min(c0 + chunk, BG); cn = c1 - c0
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    _, logits_c = denoiser(z_final_g[c0:c1], torch.ones(cn, device=device),
                                           attention_mask=valid_g[c0:c1], cond=cond_g[c0:c1], cond_mask=cond_mask_g[c0:c1],
                                           decoder_step_active=True)
                logp_c = torch.log_softmax(logits_c.float() / args.decode_temp, dim=-1).gather(-1, ids_g[c0:c1].unsqueeze(-1)).squeeze(-1).sum(-1)
                ratio_c = torch.exp(logp_c - logp_tok_old_t[c0:c1])
                loss_c = (-adv[c0:c1] * ratio_c).sum() / BG
                loss_c.backward()
                ratio_t_list.append(ratio_c.detach())
                loss_t_total += loss_c.item()
            m_ratio_t.append(torch.cat(ratio_t_list).mean().item())
            m_loss_t.append(loss_t_total)
            torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
            optimizer.step()

        reward_ema = rewards.mean() if reward_ema is None else 0.9 * reward_ema + 0.1 * rewards.mean()
        print(f"step {step+1:4d} | r_mean {rewards.mean():.3f} r>0.5 {(rewards>0.5).sum()}/{BG} "
              f"| ratio_f {np.mean(m_ratio_f):.3f} kl_f {np.mean(m_kl_f):.2e} masked_f {np.mean(m_masked_f):.3f} loss_f {np.mean(m_loss_f):+.4f} "
              f"| ratio_t {np.mean(m_ratio_t):.3f} loss_t {np.mean(m_loss_t):+.4f} "
              f"| r_ema {reward_ema:.3f} | {(time.time()-t0):.1f}s", flush=True)
        if (step + 1) % args.eval_every == 0:
            ck = {"model": denoiser.state_dict(), "optimizer": optimizer.state_dict(),
                  "step": step + 1, "reward_ema": reward_ema, "best_r": best_r, "config": vars(args)}
            torch.save(ck, os.path.join(args.out_dir, "last.pth"))
            if reward_ema > best_r:
                best_r = reward_ema
                torch.save(ck, os.path.join(args.out_dir, "best.pth"))
            print(f"  [ckpt] step {step+1} r_ema {reward_ema:.3f} best {best_r:.3f} -> {args.out_dir}/{{best,last}}.pth", flush=True)

    # ---- 诊断: 固定 prompt+噪声, init vs trained (R² 口径) ----
    denoiser.eval()
    torch.cuda.empty_cache()
    diag_prompts = gen_prompts(8)
    cond_d, cond_mask_d = encode_cond([p[2] for p in diag_prompts])
    def diag_eval(n_rollouts=1):
        rs = []
        for _ in range(n_rollouts):
            ids_d, _, _, _, _, _ = rollout(cond_d, cond_mask_d)
            trees = [ids_to_tree(ids_d[i]) for i in range(ids_d.shape[0])]
            tasks = [(trees[i], diag_prompts[i // args.G][0], diag_prompts[i // args.G][1]) for i in range(ids_d.shape[0])]
            rs.append(np.array(pool.map(_reward_one, tasks)))
        return np.concatenate(rs)
    torch.manual_seed(12345); np.random.seed(12345)
    r_tr = diag_eval()
    denoiser.load_state_dict(init_sd)
    torch.manual_seed(12345); np.random.seed(12345)
    r_init = diag_eval()
    print(f"\n[diag] {args.max_steps}步 | init R² {r_init.mean():.3f} (>0:{(r_init>0).sum()}/{len(r_init)}) "
          f"| trained R² {r_tr.mean():.3f} (>0:{(r_tr>0).sum()}/{len(r_tr)})", flush=True)
    print(f"[diag] trained R² > init R² = 双头 GRPO 有效(decode 脱节修复); < = 仍失败", flush=True)
    pool.close(); pool.join()
    print(f"done {args.max_steps} steps in {(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
