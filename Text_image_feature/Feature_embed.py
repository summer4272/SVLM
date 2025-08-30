import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoImageProcessor

class FrozenDINOv2EncoderBase(nn.Module):
    """
    冻结 facebook/dinov2-base 作为视觉塔：
    输入:  pixel_values (B,3,H,W), 建议 224x224（需为 14 的整数倍）
    输出:
      - 默认: (B, N, 1024)，224→N=16*16=256
      - 若启用 resampler: (B, K, 1024)
    """
    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        out_dim: int = 1024,
        use_resampler: bool = False,
        num_queries: int = 32,          # K，启用 resampler 时有效
        num_heads: int = 8,             # 跨注意力的头数
        train_projection: bool = False, # 想轻训仅开这个对齐头 True
        device: str | torch.device | None = None,
        use_half: bool = True
    ):
        super().__init__()
        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.use_half = use_half
        self.patch_size = 14

        # matmul 加速（可选）
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # ---- 加载 & 冻结 dinov2-base ----
        self.backbone = Dinov2Model.from_pretrained(model_name)
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.hidden_size = self.backbone.config.hidden_size  # base=768

        # ---- 归一化参数（注册为 buffer，随 .to() 迁移）----
        img_proc = AutoImageProcessor.from_pretrained(model_name)
        mean = torch.tensor(img_proc.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
        std  = torch.tensor(img_proc.image_std,  dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std",  std,  persistent=False)

        # ---- 可选：轻量 Resampler（K 个查询，聚合 N 个 patch token）----
        self.use_resampler = use_resampler
        if use_resampler:
            self.num_queries = num_queries
            # queries 作为 Parameter，这样 .to(device) 能一起搬；放大时初始化在 CPU 也没关系
            self.queries = nn.Parameter(
                torch.randn(num_queries, self.hidden_size) / (self.hidden_size ** 0.5),
                requires_grad=False
            )
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.hidden_size, num_heads=num_heads, batch_first=True
            )
            self.resampler_mlp = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size * 4),
                nn.GELU(),
                nn.Linear(self.hidden_size * 4, self.hidden_size),
            )

        # ---- 对齐头（768 → out_dim=1024），极小参数 ----
        self.align = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        for p in self.align.parameters():
            p.requires_grad = bool(train_projection)

        # ---- 统一把整个模块搬到指定设备（关键！避免 CPU/GPU 混用）----
        # 包括：backbone/queries/cross_attn/resampler_mlp/align/mean/std
        self.to(self.device)
        # backbone 推理用，保持 eval
        self.backbone.eval()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        返回:
          - 无 resampler: (B, N, out_dim) ；224 → N=256
          - 有 resampler: (B, K, out_dim)
        """
        # ---- 预处理 ----
        x = pixel_values.to(self.device, non_blocking=True, memory_format=torch.channels_last)

        # dtype & 归一化
        if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64) or x.max() > 1.5:
            x = x.float() / 255.0
        else:
            x = x.float()
        x = (x - self.mean) / self.std

        B, C, H, W = x.shape
        if (H % self.patch_size) or (W % self.patch_size):
            raise ValueError(f"输入尺寸需为 {self.patch_size} 的整数倍，当前=({H},{W})")

        # autocast（bf16 优先；否则 fp16；不支持半精度则禁用）
        amp_dtype = None
        if self.use_half and x.device.type == "cuda":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype is not None)):
            # ---- 视觉 backbone（冻结，no_grad）----
            with torch.no_grad():
                out = self.backbone(x, output_hidden_states=False)
                tokens_768 = out.last_hidden_state[:, 1:, :]  # 去掉 CLS -> (B, N, 768)

            if self.use_resampler:
                # queries: (K, 768) -> (B, K, 768)，并与 tokens 对齐 dtype（避免半精度/单精度不一致）
                q = self.queries.to(dtype=tokens_768.dtype).unsqueeze(0).expand(B, -1, -1)
                # Cross-Attn: Q=q, K=tokens_768, V=tokens_768
                attn_out, _ = self.cross_attn(q, tokens_768, tokens_768)  # (B, K, 768)
                # 轻量前馈 + 残差
                attn_out = attn_out + self.resampler_mlp(attn_out)        # (B, K, 768)
                feats = self.align(attn_out)                               # (B, K, out_dim)
            else:
                feats = self.align(tokens_768)                             # (B, N, out_dim)

        # 形状自检
        N_expected = (H // self.patch_size) * (W // self.patch_size)
        if not self.use_resampler:
            assert feats.shape[1] == N_expected, f"N 不符: {feats.shape[1]} vs {N_expected}"
        return feats



class TextEmbedding(nn.Module):
    def __init__(self, vocab_size=32002, v_size=4096):
        super().__init__()
        if vocab_size is None or v_size is None:
            raise ValueError("vocab_size 和 v_size 不能为空")
        self.inputs_emb = nn.Embedding(vocab_size, v_size)

    def forward(self, inputs):
        return self.inputs_emb(inputs)

# print(processor.tokenizer.vocab_size)
# print(processor.tokenizer.get_vocab().keys())
