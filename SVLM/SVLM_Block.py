import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from MultiHeadSelfAttention import MultiHeadSelfAttention
from MOE_feedforward import MOEFeedForward


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.1, resid_dropout=0.1, mlp_ratio=4.0, mlp_dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, attn_dropout)
        self.ln2 = nn.LayerNorm(d_model)
        # self.mlp = MLP(d_model, int(d_model*mlp_ratio), mlp_dropout)
        self.mlp = MOEFeedForward(d_model, mlp_dropout=mlp_dropout, mlp_ratio=mlp_ratio, n_shared_experts=1, n_routed_experts=4, num_experts_per_tok=2) #MOE FeedForward
        self.drop = nn.Dropout(resid_dropout)

    def forward(self, x, key_padding_mask=None, is_causal=True, attn_bias=None):
        # 注意：当 attn_bias 不为 None 时，内部将关闭 is_causal，由 attn_bias 自己编码因果/非因果与padding
        attn_out = self.attn(self.ln1(x),
                             key_padding_mask=key_padding_mask,
                             is_causal=is_causal,
                             attn_bias=attn_bias)
        x = x + self.drop(attn_out)
        ausloss, y = self.mlp(self.ln2(x))
        x = x + self.drop(y)
        return ausloss, x
