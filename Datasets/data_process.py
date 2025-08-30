import os
import json
import torch

from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoProcessor

# ==== 初始化 AutoProcessor ====
MODEL_DIR = r"L:\test_code\my_llm\llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(MODEL_DIR)

# ==== 辅助函数 ====
def build_prompt(conversations: List[Dict[str, Any]]) -> str:
    """
    把多轮对话拼接成 prompt.
    User 用 "USER:", Assistant 用 "ASSISTANT:".
    <image> 占位符直接保留.
    """
    lines = []
    for turn in conversations:
        role = turn["role"].strip().upper()
        content = turn["content"]
        if role == "USER":
            lines.append(f"USER: {content}")
        elif role == "ASSISTANT":
            lines.append(f"ASSISTANT: {content}")
    return "\n".join(lines)

def encode_sample(conversations, image_path):
    """
    根据对话 + 图片 生成 input_ids, labels, pixel_values
    - input_ids: 全部拼接后的文本
    - labels: 仅保留 assistant 部分的 token, 其他是 -100
    - pixel_values: processor 处理过的图像张量
    """
    # 拼 prompt
    prompt = build_prompt(conversations)

    # 加载图像
    image = Image.open(image_path).convert("RGB")
    # image = image.resize((224, 224))

    # encode all text + image
    proc_out = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=False,  # 不截断, 保留全部对话
    )
    input_ids = proc_out["input_ids"][0]         # (L,)
    attention_mask = proc_out["attention_mask"][0]
    pixel_values = proc_out["pixel_values"]      # (1,3,H,W)

    # 构造 labels: 只保留 assistant 内容
    labels = input_ids.clone()
    labels[:] = -100
    for turn in conversations:
        if turn["role"].lower() == "assistant":
            resp = turn["content"]
            resp_ids = processor.tokenizer(
                resp, return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            # 在 input_ids 中找到 resp_ids 并打标签
            for i in range(len(input_ids) - len(resp_ids) + 1):
                if torch.equal(input_ids[i:i+len(resp_ids)], resp_ids):
                    labels[i:i+len(resp_ids)] = resp_ids
                    break
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
    }

# ==== Dataset ====
class LlavaJsonlDataset(Dataset):
    def __init__(self, data_dir: str, jsonl_name: str, image_dir: str):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_root = self.data_dir / image_dir

        jsonl_path = self.data_dir / jsonl_name
        self.rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        conversations = row["conversations"]
        image_file = row["image"]
        image_path = os.path.join(self.image_root, image_file)
        return encode_sample(conversations, image_path)

# ==== Collate Function ====
# def collate_fn(batch):
#     pad_token_id = processor.tokenizer.pad_token_id or 0
#     max_len = max(x["input_ids"].size(0) for x in batch)
#     def pad_1d(x, value):
#         out = x.new_full((max_len,), fill_value=value)
#         out[:x.size(0)] = x
#         return out
#     input_ids = torch.stack([pad_1d(x["input_ids"], pad_token_id) for x in batch])
#     attention_mask = torch.stack([pad_1d(x["attention_mask"], 0) for x in batch])
#     labels = torch.stack([pad_1d(x["labels"], -100) for x in batch])
#     pixel_values = torch.cat([x["pixel_values"] for x in batch], dim=0)
#     return {
#         "input_ids": input_ids,
#         "attention_mask": attention_mask,
#         "labels": labels,
#         "pixel_values": pixel_values,
#     }

@dataclass
class Collator:
    processor: Any
    fixed_len: int | None = None  # None 表示按 batch 内最大长度；否则强制到 fixed_len  右padding

    def __call__(self, batch):
        tok = self.processor.tokenizer
        pad_token_id = tok.pad_token_id or 0

        # 决定本 batch 的 pad 长度  
        if self.fixed_len is not None:
            max_len = self.fixed_len
        else:
            max_len = max(x["input_ids"].size(0) for x in batch)

        def pad_or_trim_1d(x, value):
            out = x.new_full((max_len,), fill_value=value)
            L = min(x.size(0), max_len)
            out[:L] = x[:L]
            return out

        input_ids = torch.stack([pad_or_trim_1d(x["input_ids"], pad_token_id) for x in batch])
        attention_mask = torch.stack([pad_or_trim_1d(x["attention_mask"], 0) for x in batch])
        labels = torch.stack([pad_or_trim_1d(x["labels"], -100) for x in batch])
        pixel_values = torch.cat([x["pixel_values"] for x in batch], dim=0)

        return {
            "input_ids": input_ids,           # (B, max_len)
            "attention_mask": attention_mask, # (B, max_len)
            "labels": labels,                 # (B, max_len)
            "pixel_values": pixel_values,     # (B, 3, H, W)
        }

if __name__ == "__main__":
    dataset = LlavaJsonlDataset(
    data_dir="G:\\test_code\\minimind-v-master\\minimind-v_dataset",
    # data_dir="G:\\test_code\\minimind-v-master\\dataset",
    # jsonl_name="pretrain_vlm_data.jsonl",
    jsonl_name="sft_vlm_data.jsonl",
    # image_dir="pretrain_images"
    image_dir="sft_images"
 )
    collator = Collator(processor=processor, fixed_len=2048) #768  2048
    loader = DataLoader(dataset, batch_size=1, collate_fn=collator)


    # batch = next(iter(loader))
    # print("== Collate 输出 ==")
    # for k, v in batch.items():
    #     print(f"{k}: shape={v.shape}, dtype={v.dtype}")
    #     # print(f"{k}: shape={v}, dtype={v.dtype}")
    #     if k == "input_ids":
    #         print(f"{k} 的值:\n{v}")
    
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    for i, batch in enumerate(loader):
        ids = batch["input_ids"]  # (B, L)
        # 统计每个样本 32000 的数量
        counts = (ids == image_token_id).sum(dim=-1)
        print(f"batch {i}: <image> token count per sample = {counts.tolist()}")
        
        if i == 2:  # 只看前3个batch
            break