import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import  Optional

def build_mixed_attn_bias(attention_mask: torch.Tensor,
                          image_token_mask: torch.Tensor,
                          dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    返回 attn_bias: (B,L,L)；允许=0.0 禁止=-inf（additive mask）
    规则：
      - 图像Q：看所有非pad的K（图像+文本），双向，无因果
      - 文本Q：可看所有图像K；对文本K必须因果（<=自身）
    """
    B, L = attention_mask.shape
    device = attention_mask.device
    dtype = dtype or torch.float32

    k_keep = attention_mask.bool()          # (B,L)
    q_is_img = image_token_mask.bool()
    q_is_txt = ~q_is_img
    k_is_img = image_token_mask.bool()
    k_is_txt = ~k_is_img

    # 先允许所有非pad的K
    allow = k_keep.unsqueeze(1).expand(B, L, L)             # (B,Lq,Lk)

    # 文本Q对文本K做因果
    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))  # (L,L)
    txt_q = q_is_txt.unsqueeze(2).expand(B, L, L)
    txt_k = k_is_txt.unsqueeze(1).expand(B, L, L)
    txt_q_txt_k = txt_q & txt_k
    allow = torch.where(txt_q_txt_k, allow & causal, allow)

    # 构造 additive mask: 允许=0, 禁止=-inf
    attn_bias = torch.zeros((B, L, L), dtype=dtype, device=device)
    neg_inf = torch.finfo(attn_bias.dtype).min
    attn_bias.masked_fill_(~allow, neg_inf)
    return attn_bias
