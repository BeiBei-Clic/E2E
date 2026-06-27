"""Flow matching 核心公式 (移植自 ELF-pytorch sampling_utils.py, 去 config 依赖, 参数化)。

训练 (Step 6) 与推理 (Step 7) 共用。
  z = t*x0 + (1-t)*noise*scale
  v = (x0 - z)/(1-t)                # target
  网络预测 x_pred, v_pred = (x_pred - z)/(1-t), 监督 v_pred -> v_target (MSE)
"""

from typing import Optional, Tuple

import torch


def add_noise(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor,
              noise_scale: float, cond_seq_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """z = t*x0 + (1-t)*noise*noise_scale; cond 位置 (cond_seq_mask>0) 保持 clean x0。

    cond_seq_mask: (B, S, 1) 或 None (无条件)。
    """
    t_expanded = t.reshape(-1, 1, 1)
    z = t_expanded * x0 + (1.0 - t_expanded) * noise * noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1.0 - cond_seq_mask) * z
    return z


def sample_timesteps(batch_size: int, p_mean: float, p_std: float,
                     device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """logit-normal 时间调度: t = sigmoid(randn * p_std + p_mean), t ∈ (0,1)。"""
    return torch.sigmoid(torch.randn((batch_size,), dtype=dtype, device=device) * p_std + p_mean)


def net_out_to_v_x(net_out, z: torch.Tensor, t: torch.Tensor,
                   t_eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """网络输出 x_pred -> (v, x_pred): v = (x_pred - z) / clamp(1-t, t_eps)。"""
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    denom = torch.clamp(1.0 - t.reshape(-1, 1, 1), min=t_eps)
    return (net_out - z) / denom, net_out


def restore_cond(z_updated: torch.Tensor, cond_seq: torch.Tensor,
                 cond_seq_mask: torch.Tensor) -> torch.Tensor:
    """采样去噪一步后, 把 cond 位置恢复为 clean cond_seq。"""
    mask = cond_seq_mask
    while mask.dim() < z_updated.dim():
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def sample_cfg_scale(batch_size: int, cfg_min: float, cfg_max: float,
                     device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """log-uniform 采样 CFG scale ∈ [cfg_min, cfg_max] (训练时 classifier-free guidance 用)。"""
    u = torch.rand((batch_size,), dtype=dtype, device=device)
    a, b = 1.0 + cfg_min, 1.0 + cfg_max
    return a * torch.exp(u * torch.log(torch.tensor(b / a, dtype=dtype, device=device))) - 1.0
