import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        # 不在 __init__ 固化 head_dim，forward 内按实际 D 计算
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def _normalize_attn_bias(self, attn_bias, B, L, dtype, device):
        """
        将各种形状的 attn_bias 规范为 [B, L, L] 的 additive mask（允许=0，禁止=-inf）
        允许输入形状示例： [B,L,L] / [1,L,L] / [L,L] / [B,1,L,L] / [B,1,1,L,L]
        """
        if attn_bias is None:
            return None

        attn_bias = attn_bias.to(dtype=dtype, device=device)

        # 压掉多余的 broadcast 维
        # 例如 [B,1,L,L] / [B,1,1,L,L] -> 逐步 squeeze 到 [B,L,L]
        while attn_bias.dim() > 3:
            attn_bias = attn_bias.squeeze(1)

        if attn_bias.dim() == 2:                  # [L,L] -> [B,L,L]
            attn_bias = attn_bias.unsqueeze(0).expand(B, L, L)
        elif attn_bias.dim() == 3:
            if attn_bias.size(0) == 1 and B > 1:  # [1,L,L] -> [B,L,L]
                attn_bias = attn_bias.expand(B, L, L)

        # 最终必须是 [B,L,L]
        assert attn_bias.shape == (B, L, L), f"attn_bias shape {attn_bias.shape} != (B,{L},{L})"
        return attn_bias

    def forward(self, x, key_padding_mask=None, is_causal=True, attn_bias=None):
        """
        x: [B, L, D]
        key_padding_mask: [B, L] (1=keep, 0=pad)
        attn_bias: additive mask，允许=0，禁止=-inf；提供则关闭 is_causal
        """
        B, L, D = x.shape
        assert D % self.n_heads == 0, f"D({D}) % n_heads({self.n_heads}) != 0"
        head_dim = D // self.n_heads

        # QKV: [B,L,3D] -> [B,L,3,H,Dh] -> [3,B,H,L,Dh] -> 三个 [B,H,L,Dh]
        qkv = self.qkv(x).view(B, L, 3, self.n_heads, head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, L, Dh]

        # 规范化 attn_bias -> [B,L,L]
        bias = self._normalize_attn_bias(attn_bias, B, L, dtype=torch.float32, device=x.device) \
            if attn_bias is not None else None  # 用 fp32 更稳

        # 如果没自定义 bias，才用 padding 生成 additive mask
        additive = None  # [B*H, L, L] float32
        neg_inf = torch.finfo(torch.float32).min

        if bias is not None:
            # 自定义混合掩码：已经编码了 padding + (非)因果
            additive = bias.unsqueeze(1).expand(B, self.n_heads, L, L).reshape(B*self.n_heads, L, L).to(torch.float32)
        else:
            # 标准路径：我们需要同时支持 padding + 因果
            # 2a) padding -> additive_pad
            if key_padding_mask is not None:
                # key_padding_mask: 1=keep, 0=pad (禁止被看作K)
                pad = (~key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
                additive_pad = pad.expand(B, self.n_heads, L, L).reshape(B*self.n_heads, L, L).to(torch.float32) * neg_inf
            else:
                additive_pad = None

            # 2b) causal -> additive_causal
            if is_causal:
                causal = torch.ones((L, L), dtype=torch.bool, device=x.device).tril()  # 允许 i>=j
                # 允许=0, 禁止=-inf
                additive_causal = (~causal).to(torch.float32) * neg_inf   # [L,L]
                additive_causal = additive_causal.unsqueeze(0).expand(B*self.n_heads, L, L).contiguous()
            else:
                additive_causal = None

            # 2c) 合并（只要有一个禁止就禁止）
            if additive_pad is not None and additive_causal is not None:
                additive = torch.maximum(additive_pad, additive_causal)  # 两者都是 0/-inf，用 max/加法都行
            elif additive_pad is not None:
                additive = additive_pad
            elif additive_causal is not None:
                additive = additive_causal
            else:
                additive = None

        # 3) SDPA 的因果开关：
        #    只要我们提供了 additive，就把 is_causal 关掉（避免与 attn_mask 冲突）
        use_causal = False if (additive is not None) else is_causal

        # 4) 用 FP32 做注意力，再 cast 回原 dtype（bf16 下更稳）
        q32 = q.reshape(B*self.n_heads, L, head_dim).to(torch.float32)
        k32 = k.reshape(B*self.n_heads, L, head_dim).to(torch.float32)
        v32 = v.reshape(B*self.n_heads, L, head_dim).to(torch.float32)

        out = F.scaled_dot_product_attention(
            q32, k32, v32,
            attn_mask=additive,  # 可能是 None
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=use_causal
        )  # [B*H, L, Dh] (fp32)

        out = out.to(x.dtype).reshape(B, self.n_heads, L, head_dim).permute(0, 2, 1, 3).contiguous()
        out = out.view(B, L, D)
        return self.proj(out)