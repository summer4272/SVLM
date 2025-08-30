import torch
import torch.nn as nn

class fix_text_img(nn.Module):
    def __init__(self, processor, im_patch_token: str | None = None):
        """
        processor: 含 tokenizer 的处理器
        im_patch_token: 若你的数据里有 <im_patch> 之类的 token，可传其字符串（可选）
        """
        super().__init__()
        self.pad_token_id = processor.tokenizer.pad_token_id or 0
        self.image_token_index = processor.tokenizer.convert_tokens_to_ids("<image>")
        self.im_patch_token_index = (
            processor.tokenizer.convert_tokens_to_ids(im_patch_token) if im_patch_token else None
        )
        self.ignore_index = -100

    def forward(self, image_features, inputs_embeds, input_ids, attention_mask, labels):
        """
        image_features: (B, K, D)
        inputs_embeds:  (B, L, D)
        input_ids:      (B, L)
        attention_mask: (B, L)
        labels:         (B, L) or None
        说明：本函数不改变序列长度 L，只在已有的 <image> 占位位置写入视觉特征；
             当占位数量与特征数量不匹配时，会“裁剪/清理多余占位”或“丢弃多余特征”。
        """
        device = inputs_embeds.device
        dtype  = inputs_embeds.dtype

        image_features = image_features.to(device)
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        B, L = input_ids.shape
        _, K, D = image_features.shape

        final_embedding      = inputs_embeds.clone()
        final_attention_mask = attention_mask.clone()
        final_labels         = labels.clone() if labels is not None else None
        final_input_ids      = input_ids.clone()

        # 逐样本对齐
        for b in range(B):
            # 先把潜在的 <im_patch> 连续段“折叠”为 <image>（可选，且不改变长度：仅把它们视为 <image>）
            if self.im_patch_token_index is not None:
                is_patch = (final_input_ids[b] == self.im_patch_token_index)
                if is_patch.any():
                    final_input_ids[b][is_patch] = self.image_token_index

            # 当前样本所有 <image> 位置
            img_pos = (final_input_ids[b] == self.image_token_index).nonzero(as_tuple=False).flatten()
            k = int(img_pos.numel())

            if k == 0:
                # 无图样本：跳过，不写入
                continue

            feats_b = image_features[b]  # (K, D)

            # 兼容“多 1 个 CLS”情形：K == k + 1 → 丢一个
            if K == k + 1:
                feats_b = feats_b[:k]
            # K > k：丢掉多余特征（保前 k 个）
            elif K > k:
                feats_b = feats_b[:k]
            # K < k：只写前 K 个占位，其余占位清为 pad（mask=0, label=-100, ids=pad）
            elif K < k:
                # 先写入前 K 个
                write_pos = img_pos[:K]
                final_embedding[b, write_pos, :]      = feats_b.to(dtype)
                final_attention_mask[b, write_pos]    = 1
                if final_labels is not None:
                    final_labels[b, write_pos] = self.ignore_index
                # 清理多出的占位
                extra_pos = img_pos[K:]
                final_input_ids[b, extra_pos]         = self.pad_token_id
                final_attention_mask[b, extra_pos]    = 0
                final_embedding[b, extra_pos, :]      = 0.0
                if final_labels is not None:
                    final_labels[b, extra_pos] = self.ignore_index
                # 本样本已处理，继续下一个
                continue

            # 此处 K <= k，且 feats_b 的长度与将要写入的占位数一致（=min(K,k)）
            write_len = feats_b.size(0)  # = min(K, k) 或修正后的 k
            write_pos = img_pos[:write_len]

            final_embedding[b, write_pos, :]   = feats_b.to(dtype)
            final_attention_mask[b, write_pos] = 1
            if final_labels is not None:
                final_labels[b, write_pos] = self.ignore_index

            # 如果 k > write_len（K<k 且上面没 continue，表示 K==k+1 的“丢1个”已处理完），这里仍可能有 extra 占位
            if k > write_len:
                extra_pos = img_pos[write_len:]
                final_input_ids[b, extra_pos]      = self.pad_token_id
                final_attention_mask[b, extra_pos] = 0
                final_embedding[b, extra_pos, :]   = 0.0
                if final_labels is not None:
                    final_labels[b, extra_pos] = self.ignore_index

        # 位置编码（简单用累加 mask 方式；按需可外部重算）
        position_ids = (final_attention_mask.cumsum(-1) - 1).clamp_min(0)

        # image_token_mask：用于上游做混合注意力时区分“图像位”
        image_token_mask = (final_input_ids == self.image_token_index)

        return final_embedding, final_attention_mask, final_labels, position_ids, image_token_mask
