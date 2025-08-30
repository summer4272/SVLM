import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DimReducerLowRank(nn.Module):
    """
    低秩瓶颈降维：Linear(4096->rank) + GELU + Linear(rank->d_target)
    - 建议 rank << 4096，例如 128/192/256/384
    - 可选 LayerNorm 与 Dropout 提升稳定性
    """
    def __init__(self, d_in=4096, d_target=1536, rank=256, use_ln=True, p_dropout=0.0):
        super().__init__()
        self.proj_down = nn.Linear(d_in, rank, bias=False)
        self.act = nn.GELU()
        self.proj_up = nn.Linear(rank, d_target, bias=False)
        self.use_ln = use_ln
        self.ln = nn.LayerNorm(d_target) if use_ln else nn.Identity()
        self.dropout = nn.Dropout(p_dropout) if p_dropout > 0 else nn.Identity()

        # 稳定初始化：正交更稳
        nn.init.orthogonal_(self.proj_down.weight)
        nn.init.orthogonal_(self.proj_up.weight)

    def forward(self, x):  # x: [B, L, d_in]
        x = self.proj_down(x)        # [B, L, rank]
        x = self.act(x)
        x = self.proj_up(x)          # [B, L, d_target]
        x = self.ln(x)
        x = self.dropout(x)
        return x

# start 
def init_weights(model):
    for name, param in model.named_parameters():
        if "weight" in name:
            if "ln" in name:
                nn.init.normal_(param, mean=1.0, std=0.02)  # 归一化层的权重初始化
            else:
                nn.init.xavier_uniform_(param)  # 其他层的权重初始化
        elif "bias" in name:
            nn.init.constant_(param, 0)  # 偏置初始化为 0
