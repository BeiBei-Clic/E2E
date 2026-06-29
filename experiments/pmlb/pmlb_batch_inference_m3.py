"""M3 扩散模型 pmlb 自适应批量评估。对齐 pmlb_adaptive_beam_inference.py 的做法:
- 自适应采样规模: M3 无 beam, 等价物是并行采样数 n_samples; R² < r2_threshold 则 n_samples 翻倍重试,
  跨 attempt 取最优, 达阈值提前退出 (初始 --n_samples, 最多 --max_retries 次翻倍)。
- 多 worker 并行 BFGS: skeleton 去重后的唯一候选分发给 ProcessPoolExecutor(fork) 并行常数优化。
流程: 数值点 -> cond_emb -> ODE 批量采样 -> decode -> (并行) BFGS -> rescale -> R²。维度 >10 跳过。

对齐要点 (逐条核对 run_inference / refine / compute_metrics):
- apply_target_noise 给 y 加噪 -> y_to_fit; BFGS 拟合目标 = y_to_fit, R² reference = 干净 y (对齐)
- StandardScaler 只标准化 X (训练数值点经 generate_datapoints 内部已标准化到 O(1), rescale 把 pmlb X 拉回训练分布)
- BFGS 在 scaled_X 空间拟合常数 -> rescale_function 只作用变量 (包 add(b, mul(a, x))) 不改常数 -> 原空间求值
- refinement_type: NoRef(raw) 与 BFGS 取 r² 较优者 (对齐原 "NoRef 与 BFGS 候选共同排序取最优")
- _complexity = rescale 后树节点数; CSV 在端到端字段上加 beam_size/attempt 两列

try-except 仅在 BFGS (Nelder-Mead) 内: 常数优化失败/非有限 -> 回退 raw。其余错误直接报 (靠 resume 续跑)。
"""
import argparse
import copy
import csv
import os
import re
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize as _scipy_min

from parsers import get_parser
from symbolicregression.envs import build_env
from symbolicregression.model.utils_wrapper import StandardScaler
from symbolicregression.flow import ELF_models
from symbolicregression.metrics import compute_metrics
from experiments.pmlb.pmlb_inference import (
    load_pmlb_dataset,
    apply_target_noise,
    format_expr,
)
from experiments.pmlb.pmlb_batch_inference import (
    is_regression_dataset,
    list_regression_datasets,
    load_existing_results,
)
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

# 自适应评估结果字段: 端到端字段 + beam_size/attempt 记录命中的采样规模与重试轮次
RESULT_FIELDS = (
    "dataset", "status", "n_features", "beam_size", "attempt",
    "refinement_type", "r2", "rmse", "complexity", "seconds",
    "error", "noise_strength", "expr",
)

warnings.filterwarnings("ignore", category=RuntimeWarning)  # Nelder-Mead 反复 tree.val 的 overflow 不阻塞

# flow matching 超参 (与训练一致)
NOISE_SCALE, T_EPS = 1.0, 5e-2


def encode_cond(bags, embedder, point_enc, device, use_amp):
    """数值点 bags -> cond_emb (1, slen, dim), cond_mask (1, slen)。纯 inference。"""
    seq_tok, seq_cond_len = embedder.batch(embedder.encode(bags))
    seq_tok, seq_cond_len = seq_tok.to(device), seq_cond_len.to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        seq_emb = embedder.compress(embedder.embed(seq_tok))
        cond_emb = point_enc("fwd", x=seq_emb, lengths=seq_cond_len, causal=False)
    cond_emb = cond_emb.transpose(0, 1).float()
    slen_max = cond_emb.shape[1]
    cond_mask = (torch.arange(slen_max, device=device).unsqueeze(0) < seq_cond_len.unsqueeze(1)).float()
    return cond_emb, cond_mask


def ode_sample_decode(denoiser, cond_emb, cond_mask, n_samples, n_ode_steps,
                      max_length, d_model, device, use_amp):
    """z=randn*n_samples -> ODE t:0->1 -> 末步 decode -> argmax ids (n_samples, max_length)。
    n_samples 个候选在同一 batch 并行采样 (同 cond 广播), 一次轨迹出 n_samples 个表达式。"""
    cond_b = cond_emb.expand(n_samples, -1, -1)
    mask_b = cond_mask.expand(n_samples, -1)
    attn = torch.ones(n_samples, max_length, dtype=torch.bool, device=device)
    t_steps = torch.linspace(0.0, 1.0, n_ode_steps + 1, device=device)
    z = torch.randn(n_samples, max_length, d_model, device=device) * NOISE_SCALE
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        for i in range(n_ode_steps):
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            x_pred, _ = denoiser(z, torch.full((n_samples,), t, device=device),
                                 attention_mask=attn, cond=cond_b, cond_mask=mask_b)
            v = (x_pred.float() - z) / max(1.0 - t, T_EPS)
            z = z + (t_next - t) * v
        _, logits = denoiser(z, torch.ones(n_samples, device=device),
                             attention_mask=attn, cond=cond_b, cond_mask=mask_b,
                             decoder_step_active=True)
    return logits.argmax(-1).cpu()


def ids_to_tree(env, ids_row, eos_id):
    """ids 行 -> 去 [首 EOS, 首个尾 EOS] 的 word list -> tree (None=非法)。"""
    seq = ids_row.tolist()
    cut = next((j for j in range(1, len(seq)) if seq[j] == eos_id), len(seq))
    words = [env.equation_id2word[int(t)] for t in seq[1:cut]]
    tree = env.word_to_infix(words, is_float=False, str_array=False)
    if tree is None:
        return None
    return tree.nodes[0] if hasattr(tree, "nodes") else tree  # NodeList -> 单输出 Node


def tree_to_skeleton(tree):
    """数值叶子 -> CONSTANT_k 占位; 返回 (skeleton_node, coeffs0 初值)。"""
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


def _val1d(tree, x):
    """tree.val(x) -> (n,); NodeList.val 返回 (n, n_out) 取第 0 输出。"""
    yh = tree.val(x)
    return yh[:, 0] if (isinstance(yh, np.ndarray) and yh.ndim == 2) else yh


def _set_constants(skeleton, coeffs):
    """skeleton 的 CONSTANT_k 叶子填 coeffs[k] (返回新 tree, 不改原)。"""
    sk = copy.deepcopy(skeleton)
    k = [0]
    def visit(node):
        if len(node.children) == 0 and str(node.value).startswith("CONSTANT"):
            node.value = str(coeffs[k[0]]); k[0] += 1
        for c in node.children:
            visit(c)
    visit(sk)
    return sk


def _r2(yh, y):
    if not (np.all(np.isfinite(yh)) and np.all(np.isfinite(y))):
        return float("nan")
    ss = float(((y - y.mean()) ** 2).sum())
    return float(1.0 - ((y - yh) ** 2).sum() / ss) if ss > 0 else float("nan")


def tree_fit_r2(tree, x_fit, y_fit):
    """在 (x_fit, y_fit) 上常数优化 -> (r2, tree, refinement_type)。
    无常数 / BFGS 失败 -> raw (NoRef); BFGS 成功 -> 与 raw 比, 取较优 (对齐原排序)。"""
    dims = [int(d) for d in re.findall(r"x_(\d+)", tree.prefix())]
    if dims and max(dims) >= x_fit.shape[1]:
        return float("nan"), tree, "NoRef"  # 生成维度 > 输入维度, 无法求值
    r_raw = _r2(_val1d(tree, x_fit), y_fit)
    skeleton, coeffs0 = tree_to_skeleton(copy.deepcopy(tree))
    if len(coeffs0) == 0:
        return r_raw, tree, "NoRef"
    def _obj(c):
        yh_c = _val1d(_set_constants(skeleton, c), x_fit)
        if not np.all(np.isfinite(yh_c)):
            return 1e10
        return float(((y_fit - yh_c) ** 2).mean())
    try:
        res = _scipy_min(_obj, coeffs0, method="Nelder-Mead",
                         options={"maxiter": 200 * len(coeffs0), "xatol": 1e-5, "fatol": 1e-9})
        fitted = _set_constants(skeleton, res.x)
        r_bfgs = _r2(_val1d(fitted, x_fit), y_fit)
        if np.isfinite(r_bfgs) and (not np.isfinite(r_raw) or r_bfgs >= r_raw):
            return r_bfgs, fitted, "BFGS"
    except Exception:
        pass
    return r_raw, tree, "NoRef"


def _bfgs_worker(task):
    """ProcessPool worker: (tree, x, y) -> (r2, tree, rtype)。
    tree_fit_r2 纯 numpy/scipy + tree.val, 无 env/GPU 依赖 (fork COW 共享主进程已加载模块)。"""
    tree, x, y = task
    return tree_fit_r2(tree, x, y)


def run_inference_m3(dataset_name, denoiser, embedder, point_enc, env, eos_id,
                     args, device, use_amp, d_model):
    _, _, X, y = load_pmlb_dataset(dataset_name, args.datasets_dir, args.max_rows)
    y_to_fit = apply_target_noise(y, args.noise_strength, args.noise_seed)

    if args.rescale:
        scaler = StandardScaler()
        scaled_X = scaler.fit_transform(X)
        a, b = scaler.get_params()
    else:
        scaled_X, a, b = X, None, None

    n_pts = min(len(scaled_X), args.max_input_points)
    # y 扩成 (n,1) 列向量: float_encoder.encode 期望 1D array (原端到端 fit 同样 expand_dims y)
    y_col = np.asarray(y_to_fit[:n_pts]).reshape(-1, 1)
    bags = [list(zip(np.asarray(scaled_X[:n_pts]), y_col))]

    # cond 只依赖数据点, 与采样数无关 -> 循环外算一次复用 (自适应只重跑采样 + BFGS)
    cond_emb, cond_mask = encode_cond(bags, embedder, point_enc, device, use_amp)

    # 自适应采样规模 (对齐 pmlb_adaptive_beam_inference: R² 未达阈值则 n_samples 翻倍重试,
    # 跨 attempt 取最优, 达阈值提前退出)。M3 无 beam, 等价物是并行采样数 n_samples。
    best = None       # (r2_sel, tree, rtype, beam_size, attempt)
    best_r2 = -float("inf")
    current_samples = args.n_samples
    for attempt in range(args.max_retries + 1):
        ids = ode_sample_decode(denoiser, cond_emb, cond_mask, current_samples,
                                args.n_ode_steps, args.max_length, d_model, device, use_amp)

        # decode + skeleton 去重 -> 唯一结构候选 (同结构不同常数只 BFGS 一次, 对齐原 refine)
        unique_trees, seen = [], {}
        for i in range(ids.shape[0]):
            tree = ids_to_tree(env, ids[i], eos_id)
            if tree is None:
                continue
            sk = tree_to_skeleton(copy.deepcopy(tree))[0].prefix()
            if sk in seen:
                continue
            seen[sk] = True
            unique_trees.append(tree)

        # 多 worker 并行 BFGS (fork COW; task 带 tree+x+y, tree_fit_r2 纯 numpy 无 env/GPU 依赖)
        attempt_best = None
        if unique_trees:
            tasks = [(t, scaled_X, y_to_fit) for t in unique_trees]
            worker_count = min(len(tasks), args.bfgs_workers)
            with ProcessPoolExecutor(max_workers=worker_count, mp_context=get_context("fork")) as ex:
                fitted = list(ex.map(_bfgs_worker, tasks))
            attempt_best = max(fitted, key=lambda c: c[0] if np.isfinite(c[0]) else -np.inf)

        if attempt_best is not None:
            r2_a, tree_a, rtype_a = attempt_best
            if r2_a > best_r2:
                best_r2 = r2_a
                best = (r2_a, tree_a, rtype_a, current_samples, attempt + 1)
            # 达 R² 阈值提前退出 (选优口径 = scaled 空间 vs y_to_fit, 对齐 refine 排序)
            if r2_a >= args.r2_threshold:
                break
        current_samples *= 2

    if best is None:
        return {"status": "ok", "n_features": int(X.shape[1]), "beam_size": "",
                "attempt": "", "refinement_type": "", "r2": "", "rmse": "",
                "complexity": "", "expr": ""}

    r2_sel, best_tree, best_type, beam_size, attempt_no = best
    rescaled = scaler.rescale_function(env, best_tree, a, b) if args.rescale else best_tree

    # 报告口径: rescale 后树在原 X 求值 vs 干净 y (对齐 predict + compute_metrics)
    y_pred = _val1d(rescaled, X)
    metrics = compute_metrics({"true": [y], "predicted": [y_pred], "predicted_tree": [rescaled]},
                              metrics="r2,_rmse,_complexity")
    return {
        "status": "ok",
        "n_features": int(X.shape[1]),
        "beam_size": beam_size,
        "attempt": attempt_no,
        "refinement_type": best_type,
        "r2": metrics["r2"][0],
        "rmse": metrics["_rmse"][0],
        "complexity": metrics["_complexity"][0],
        "expr": format_expr(rescaled),
    }


def default_output_csv_m3(noise_strength):
    return f"experiments/pmlb/results/pmlb_m3_adaptive_noise_{noise_strength:g}.csv"


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_dir", default="pmlb/datasets")
    parser.add_argument("--ckpt", default="checkpoints/lr_search/uniform_cosine_lr2e-3/best.pth")
    parser.add_argument("--point_ckpt", default="model.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--max_rows", type=int, default=200)
    parser.add_argument("--max_input_points", type=int, default=200)
    parser.add_argument("--n_samples", type=int, default=32,
                        help="初始采样规模; R² 未达阈值时翻倍重试 (自适应 beam 等价物)")
    parser.add_argument("--r2_threshold", type=float, default=0.9,
                        help="R² 达此阈值提前退出 (选优口径 = scaled 空间)")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="R² 未达阈值的最大翻倍重试次数 (总 attempt = max_retries+1)")
    parser.add_argument("--bfgs_workers", type=int, default=os.cpu_count(),
                        help="并行 BFGS worker 数 (fork)")
    parser.add_argument("--n_ode_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--rescale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noise_strength", type=float, default=0.0)
    parser.add_argument("--noise_seed", type=int, default=0)
    parser.add_argument("--dataset_limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    if args.noise_strength < 0:
        raise ValueError("noise_strength must be non-negative.")

    device = torch.device(args.device)
    use_amp = device.type == "cuda"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    params = get_parser().parse_args([])
    params.fp16 = False
    env = build_env(params)
    env.rng = np.random.RandomState(args.seed + 100)
    n_words = env.n_words
    eos_id = env.equation_word2id["<EOS>"]

    mw = torch.load(args.point_ckpt, map_location="cpu", weights_only=False)
    embedder = mw.embedder.to(device).eval()
    point_enc = mw.encoder.to(device).eval()
    for m in (embedder, point_enc):
        for p in m.parameters():
            p.requires_grad_(False)

    denoiser = ELF_models["ELF-B"](
        text_encoder_dim=params.dec_emb_dim, max_length=args.max_length,
        vocab_size=n_words, num_self_cond_cfg_tokens=0).to(device).eval()
    del denoiser.self_cond_proj
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    denoiser.load_state_dict(ck["model"])
    d_model = params.dec_emb_dim
    print(f"loaded {args.ckpt}: step {ck.get('step', '?')+1} val_l2 {ck.get('val_loss', '?')}", flush=True)

    dataset_names = list_regression_datasets(args.datasets_dir)
    if args.dataset_limit is not None:
        dataset_names = dataset_names[: args.dataset_limit]

    output_path = args.output_csv or default_output_csv_m3(args.noise_strength)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    completed = load_existing_results(output_path)
    write_mode = "a" if os.path.exists(output_path) and os.path.getsize(output_path) > 0 else "w"

    with open(output_path, write_mode, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        if write_mode == "w":
            writer.writeheader()
        for dataset_name in dataset_names:
            if (dataset_name, args.noise_strength) in completed:
                print(f"{dataset_name}: skipped (done, noise={args.noise_strength:g})", flush=True)
                continue
            _, _, X, _ = load_pmlb_dataset(dataset_name, args.datasets_dir, args.max_rows)
            if X.shape[1] > 10:
                print(f"{dataset_name}: skipped (n_features={X.shape[1]} > 10)", flush=True)
                continue

            start = time.time()
            result = run_inference_m3(dataset_name, denoiser, embedder, point_enc, env, eos_id,
                                      args, device, use_amp, d_model)
            row = {field: "" for field in RESULT_FIELDS}
            row.update({"dataset": dataset_name, "noise_strength": args.noise_strength})
            row.update({f: result[f] for f in RESULT_FIELDS if f in result})
            row["seconds"] = f"{time.time() - start:.2f}"
            writer.writerow(row)
            handle.flush()
            print(f"{dataset_name}: {row['status']} r2={row['r2']} "
                  f"beam={row['beam_size']} attempt={row['attempt']} ({row['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
