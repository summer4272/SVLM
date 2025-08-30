import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Optional
from SVLM_Block import TransformerBlock
from attn_bias import build_mixed_attn_bias
class MixedCausalLM(nn.Module):
    """
    一个支持图像无因果 / 文本因果的 Causal LM。
    - 若提供 image_token_mask，则自动构造 attn_bias 并传入每层 attention。
    - 若未提供 image_token_mask，则回退为标准的因果掩码 + padding 掩码。
    """
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 4096,
                 n_heads: int = 8,
                 n_layers: int = 12,
                 aux_loss_coef: float = 0.1,
                 max_positions: int = 8192,
                 tie_weights: bool = False,
                 use_learned_pos_emb: bool = True,
                 attn_dropout: float = 0.1,
                 resid_dropout: float = 0.1,
                 mlp_ratio: float = 4.0,
                 mlp_dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_learned_pos_emb = use_learned_pos_emb

        # token embedding
        self.tok_emb = nn.Embedding(vocab_size, d_model)

        # learned position embedding（如果你使用 RoPE，可将其关闭并在注意力内部做旋转）
        self.pos_emb = nn.Embedding(max_positions, d_model) if use_learned_pos_emb else None

        # blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=d_model, n_heads=n_heads,
                attn_dropout=attn_dropout, resid_dropout=resid_dropout,
                mlp_ratio=mlp_ratio, 
            ) for _ in range(n_layers)
        ])
        self.aux_loss_coef = aux_loss_coef  ######################
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.build_mixed_attn_bias = build_mixed_attn_bias

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self,
            input_ids: Optional[torch.Tensor] = None,     # (B,L)
            inputs_embeds: Optional[torch.Tensor] = None, # (B,L,D)
            attention_mask: Optional[torch.Tensor] = None,# (B,L) 1=keep,0=pad
            position_ids: Optional[torch.Tensor] = None,  # (B,L)
            image_token_mask: Optional[torch.Tensor] = None, # (B,L) True=img
            labels: Optional[torch.Tensor] = None,        # (B,L)
            is_causal: bool = True):

        # ---- Embedding ----
        if inputs_embeds is not None:
            x = inputs_embeds
            B, L, D = x.shape
            assert D == self.d_model, f"inputs_embeds dim {D} != d_model {self.d_model}"
        else:
            assert input_ids is not None, "either input_ids or inputs_embeds must be provided"
            x = self.tok_emb(input_ids)
            B, L, D = x.shape

        device = x.device

        # ---- attention_mask 默认全1 ----
        if attention_mask is None:
            attention_mask = torch.ones((B, L), dtype=torch.long, device=device)

        # ---- Position ids（若未传入）----
        if self.pos_emb is not None:
            if position_ids is None:
                position_ids = (attention_mask.cumsum(-1) - 1).clamp_min(0)
            pos = self.pos_emb(position_ids)  # (B,L,D)
            x = x + pos

        # ---- 构造 attn_bias（若提供 image_token_mask）----
        attn_bias = None
        if image_token_mask is not None:
            # FIX: 用 float32 构造 additive mask 更稳
            attn_bias = self.build_mixed_attn_bias(attention_mask, image_token_mask, dtype=torch.float32)

        # FIX: 用 fp32 聚合 aux，避免 bf16 累计误差
        aux_total = torch.zeros((), dtype=torch.float32, device=device)

        # ---- Transformer Blocks ----
        for blk in self.blocks:
            if attn_bias is None:
                aux, x = blk(x, key_padding_mask=attention_mask, is_causal=is_causal, attn_bias=None)
            else:
                aux, x = blk(x, key_padding_mask=None, is_causal=False, attn_bias=attn_bias)
            if aux is not None:
                # FIX: 乘以系数，并上浮到 fp32
                # print(type(aux))  # 查看 aux 类型
                # print(type(self.aux_loss_coef))  # 查看 self.aux_loss_coef 类型
                aux_total = aux_total + float(aux) * self.aux_loss_coef
                # aux_total = aux_total + aux.float() * float(self.aux_loss_coef)

        # ---- Head ----
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,L,V)

        # ---- Loss（shifted）----
        loss = None
        if labels is not None:
            # FIX: 仅在 CE 处上浮到 fp32
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            ce = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            loss = ce + aux_total
        else:
            loss = aux_total

        return {"logits": logits, "loss": loss, "aux_loss": aux_total}
