#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import math
import time
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from SVLM.MixedCausalSVLM import MixedCausalLM
from Text_image_feature.Feature_embed import FrozenDINOv2EncoderBase, TextEmbedding
from SVLM.ReducerLow import DimReducerLowRank
from Datasets.data_process import Collator
from Datasets.data_Iterable import LlavaJsonlIterable
from Text_image_feature.fuse_text_img import fix_text_img

try:
    from torch.cuda.amp import GradScaler  
except Exception:
    GradScaler = None  

# ------------------------------
# Config
# ------------------------------
@dataclass
class TrainConfig:
    # Basic Path
    model_dir: str
    dinov2_path: str
    data_dir: str
    jsonl_name: str
    image_dir: str

    # Model Width
    d_in: int = 4096
    d_model: int = 2048
    rank: int = 384
    n_heads: int = 8
    n_layers: int = 16
    print_every: int = 10

    # Training switch
    stage: str = "stage1"  # stage1 / stage2
    unfreeze_lm_top_layers: int = 2
    train_text_embed: bool = True
    train_dino_proj: bool = True

    # Training parameters
    batch_size: int = 2
    epochs: int = 1
    steps_per_epoch: int = 1500  # >0 truncate; <=0 do not truncate and adapt
    lr: float = 2e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    warmup_ratio: float = 0.03
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"  # bfloat16 / float16 / float32
    accum_steps: int = 1

    # dataloader
    num_workers: int = 0
    prefetch_factor: int = 1
    pin_memory: bool = False
    persistent_workers: bool = False
    fixed_len: int = 768
    shuffle: bool = True

    # Log / Save
    wandb_project: Optional[str] = None
    run_name: Optional[str] = None
    save_dir: str = "./checkpoints"
    save_every: int = 5000  # step

    # Others
    seed: int = 42
    ckpt: Optional[str] = None


# ------------------------------
# Utils
# ------------------------------

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(module: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def init_weights_for_lm(model: nn.Module):
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "weight" in name:
            if "ln" in name or "layernorm" in name.lower():
                nn.init.normal_(p, mean=1.0, std=0.02)
            else:
                nn.init.xavier_uniform_(p)
        elif "bias" in name:
            nn.init.constant_(p, 0)


def select_autocast_dtype(amp_dtype: str):
    if amp_dtype.lower() == "bfloat16":
        return torch.bfloat16
    elif amp_dtype.lower() == "float16":
        return torch.float16
    else:
        return None  # no autocast


def make_optimizer(params, cfg: TrainConfig):
    return torch.optim.AdamW(params, lr=cfg.lr, betas=cfg.betas, weight_decay=cfg.weight_decay)


def make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))  # cosine decay 1->0
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_trainable_for_stage(lm: nn.Module, text_emb: nn.Module, dino: FrozenDINOv2EncoderBase,
                            reducer: nn.Module, stage: str, n_top_layers: int, train_text_embed: bool,
                            train_dino_proj: bool):
    # All frozen
    for p in lm.parameters():
        p.requires_grad = False
    # Only open the top n layers of LM + lm_head (if available)
    if hasattr(lm, "transformer") and hasattr(lm.transformer, "h"):
        blocks = lm.transformer.h
        n = len(blocks)
        for i in range(max(0, n - n_top_layers), n):
            for p in blocks[i].parameters():
                p.requires_grad = True
    elif hasattr(lm, "blocks"):
        blocks = lm.blocks
        n = len(blocks)
        for i in range(max(0, n - n_top_layers), n):
            for p in blocks[i].parameters():
                p.requires_grad = True
    # lm_head / ln_f ：
    for name in ["lm_head", "ln_f", "final_norm", "norm"]:
        if hasattr(lm, name):
            for p in getattr(lm, name).parameters():
                p.requires_grad = True

    # Reducer is always trained (dimensionality reduction/low-rank adaptation)
    for p in reducer.parameters():
        p.requires_grad = True

    # Whether text embedding is trained
    for p in text_emb.parameters():
        p.requires_grad = bool(train_text_embed)

    # Visual Tower: Whether to train projection
    if hasattr(dino, "set_train_projection"):
        dino.set_train_projection(bool(train_dino_proj))
    else:
        if hasattr(dino, "proj"):
            for p in dino.proj.parameters():
                p.requires_grad = bool(train_dino_proj)


def build_everything(cfg: TrainConfig, device: torch.device):
    # Processor & Tokenizer
    processor = AutoProcessor.from_pretrained(cfg.model_dir, use_fast=False)
    vocab_size = processor.tokenizer.vocab_size + 2


    dinov2 = FrozenDINOv2EncoderBase(
        model_name=cfg.dinov2_path,
        out_dim=cfg.d_in,
        use_resampler=True,
        num_queries=32,
        train_projection=cfg.train_dino_proj,
        use_half=True,
        device=device,
    )
    reducer = DimReducerLowRank(d_in=cfg.d_in, d_target=cfg.d_model, rank=cfg.rank, use_ln=True, p_dropout=0.1).to(device)
    text_emb = TextEmbedding(vocab_size=vocab_size, v_size=cfg.d_in).to(device)

    lm = MixedCausalLM(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        aux_loss_coef=0.1,
        max_positions=8192,
        tie_weights=True,
        use_learned_pos_emb=True,
        attn_dropout=0.1,
        resid_dropout=0.1,
        mlp_ratio=4,
        mlp_dropout=0.1,
    ).to(device)


    set_trainable_for_stage(lm, text_emb, dinov2, reducer, cfg.stage, cfg.unfreeze_lm_top_layers,
                            cfg.train_text_embed, cfg.train_dino_proj)

    # Datasets
    dataset = LlavaJsonlIterable(
        data_dir=cfg.data_dir,
        jsonl_name=cfg.jsonl_name,
        image_dir=cfg.image_dir,
        shuffle=cfg.shuffle,
    )
    collator = Collator(processor=processor, fixed_len=cfg.fixed_len)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
        num_workers=cfg.num_workers,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
    )

    return processor, dinov2, reducer, text_emb, lm, loader, dataset


# ------------------------------
# Helper: Inference dataset size and number of steps per round
# ------------------------------
def try_count_jsonl_lines(data_dir: str, jsonl_name: str) -> Optional[int]:
    try:
        path = os.path.join(data_dir, jsonl_name)
        cnt = 0
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                cnt += 1
        return cnt
    except Exception:
        return None


def infer_dataset_size(dataset, cfg: TrainConfig) -> Optional[int]:

    try:
        return len(dataset) 
    except Exception:
        pass

    return try_count_jsonl_lines(cfg.data_dir, cfg.jsonl_name)


def infer_steps_per_epoch_auto(dataset, cfg: TrainConfig) -> Optional[int]:
    size = infer_dataset_size(dataset, cfg)
    if size is None or size <= 0:
        return None
    steps = (size + cfg.batch_size - 1) // cfg.batch_size  # ceil
    return steps


# ------------------------------
# Train / Validate
# ------------------------------
def compute_uniform_ce(vocab_size: int, labels: torch.Tensor) -> float:

    with torch.no_grad():
        B, L = labels.shape
        V = vocab_size
        zeros = torch.zeros(B, L, V, device=labels.device, dtype=torch.float32)
        ce = F.cross_entropy(
            zeros[:, :-1, :].reshape(-1, V),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
    return float(ce)


def forward_step(batch: Dict[str, torch.Tensor], processor, dinov2, reducer, text_emb, lm,
                 device, autocast_dtype):
    pv = batch["pixel_values"].to(device, non_blocking=True)
    ids = batch["input_ids"].to(device, non_blocking=True)
    att = batch["attention_mask"].to(device, non_blocking=True)
    lbs = batch["labels"].to(device, non_blocking=True)

    # Image -> Visual features (B,K,4096)
    vision_tokens = dinov2(pv)
    # text -> Word vectors (B,L,4096)
    input_emb = text_emb(ids)
    input_emb = input_emb.to(vision_tokens.dtype)

    # Splicing Image and text
    packer = fix_text_img(processor=processor, im_patch_token=None)
    final_emb, final_attn, final_labels, position_ids, image_token_mask = packer(
        image_features=vision_tokens,
        inputs_embeds=input_emb,
        input_ids=ids,
        attention_mask=att,
        labels=lbs,
    )

    # Dimensions aligned to d_model
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
        final_emb = reducer(final_emb)  # (B,L,d_model)
        # Sanity: label 
        valid = final_labels[final_labels != -100]
        if valid.numel() > 0:
            vmin, vmax = int(valid.min()), int(valid.max())
            assert 0 <= vmin and vmax < lm.vocab_size, f"label 超出词表范围: [{vmin},{vmax}] vs V={lm.vocab_size}"
        out = lm(
            inputs_embeds=final_emb,
            attention_mask=final_attn,
            labels=final_labels,
            position_ids=position_ids,
            image_token_mask=None,
        )
    loss = out["loss"]
    aux_loss = out.get("aux_loss", torch.tensor(0.0, device=device))
    logits = out["logits"]
    return loss, aux_loss, logits, final_labels


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    device = get_device()


    processor, dinov2, reducer, text_emb, lm, loader, dataset = build_everything(cfg, device)

    if cfg.ckpt:
        print(f"[CKPT] Loading for training from: {cfg.ckpt}")
        load_checkpoint(cfg.ckpt, lm, reducer, text_emb, dinov2)
    else:
        print("[INIT] No ckpt provided. Initializing trainable parameters from scratch.")
        init_weights_for_lm(lm)  # 仅初始化 requires_grad=True 的参数

    print("LM params:", count_params(lm))
    print("Reducer params:", count_params(reducer))
    print("TextEmb params:", count_params(text_emb))

    # 解析每轮步数（支持不截断模式）
    if cfg.steps_per_epoch and cfg.steps_per_epoch > 0:
        steps_per_ep = cfg.steps_per_epoch
        samples_auto = infer_dataset_size(dataset, cfg)
        print(f"[EPOCH-STEPS] mode=truncate | steps_per_epoch={steps_per_ep} | "
              f"batch_size={cfg.batch_size} | samples(auto)={samples_auto if samples_auto is not None else 'unknown'}")
    else:
        steps_per_ep = infer_steps_per_epoch_auto(dataset, cfg)
        if steps_per_ep is not None:
            samples_auto = infer_dataset_size(dataset, cfg)
            print(f"[EPOCH-STEPS] mode=full-dataset | auto steps_per_epoch={steps_per_ep} "
                  f"(samples={samples_auto}, batch_size={cfg.batch_size})")
        else:
            print("[EPOCH-STEPS] mode=full-dataset | steps_per_epoch=unknown (Iterable dataset without length). "
                  "Will NOT truncate and will use constant LR (no scheduler).")

    # 优化器 & 调度
    params = [p for p in list(lm.parameters()) + list(reducer.parameters()) + list(text_emb.parameters())
              if p.requires_grad]
    for n, p in dinov2.named_parameters():
        if p.requires_grad:
            params.append(p)
    optimizer = make_optimizer(params, cfg)

    if steps_per_ep is not None:
        total_steps = cfg.epochs * steps_per_ep
        warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
        scheduler = make_scheduler(optimizer, total_steps, warmup_steps)
    else:
        total_steps = None
        warmup_steps = 0
        scheduler = None

    # AMP
    autocast_dtype = select_autocast_dtype(cfg.amp_dtype)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp_dtype.lower() == "float16"))


    # scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp_dtype.lower() == "float16"))

    # W&B
    use_wandb = cfg.wandb_project is not None
    # if use_wandb:
    #     import wandb
    #     wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=vars(cfg))
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg.wandb_project,
                name=cfg.run_name,
                config=vars(cfg),
                settings=wandb.Settings(init_timeout=30) 
            )
        except Exception as e:
            print(f"[W&B] 初始化失败，自动禁用：{e}")
            use_wandb = False

    # 训练循环
    lm.train()
    reducer.train()
    text_emb.train()
    dinov2.train()

    step = 0
    for epoch in range(1, cfg.epochs + 1):
        epoch_loss = 0.0
        epoch_aux = 0.0
        t0 = time.time()
        for it, batch in enumerate(loader):
            # 仅当设置了截断才 break
            if cfg.steps_per_epoch and cfg.steps_per_epoch > 0 and it >= cfg.steps_per_epoch:
                break

            # 前向
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
                loss, aux_loss, logits, labels = forward_step(batch, processor, dinov2, reducer, text_emb, lm,
                                                              device, autocast_dtype)
                total_loss = (loss + aux_loss) / cfg.accum_steps

            # 反向
            if scaler.is_enabled():
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            # 梯度累计
            do_step = ((it + 1) % cfg.accum_steps == 0)
            if do_step:
                if cfg.grad_clip is not None and cfg.grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)

                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                step += 1
                                # ==== 实时控制台打印（不依赖 W&B） ====
                if step % cfg.print_every == 0:
                    lr_now = (scheduler.get_last_lr()[0] if scheduler is not None else cfg.lr)
                    # 注意：loss/aux_loss 是当前 batch 的值；ppl 用 loss 近似
                    cur_loss = float(loss.detach())
                    cur_aux  = float(aux_loss.detach())
                    cur_total = cur_loss + cur_aux
                    cur_ppl = math.exp(max(0.0, cur_loss)) if cur_loss < 20 else float('inf')
                    # 也可以打印 it（从0开始），这里 +1 直观一些
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"epoch={epoch} | iter={it+1} | step={step} | "
                        f"lr={lr_now:.6g} | loss={cur_loss:.4f} | aux={cur_aux:.4f} | "
                        f"total={cur_total:.4f} | ppl≈{cur_ppl:.2f}",
                        flush=True
                    )


            # 统计
            epoch_loss += float(loss.detach())
            epoch_aux += float(aux_loss.detach())

            # 首次迭代：打印基线 CE
            if epoch == 1 and it == 0:
                base_ce = compute_uniform_ce(lm.vocab_size, labels)
                print(f"[BASELINE] CE@uniform = {base_ce:.4f}  (期望≈ ln(V)={math.log(lm.vocab_size):.2f})")

            # Log
            if use_wandb and do_step:
                import wandb
                lr_now = (scheduler.get_last_lr()[0] if scheduler is not None else cfg.lr)
                ppl = math.exp(max(0.0, float(loss.detach()))) if float(loss) < 20 else float('inf')
                wandb.log({
                    "train/loss_ce": float(loss.detach()),
                    "train/aux_loss": float(aux_loss.detach()),
                    "train/total_loss": float((loss+aux_loss).detach()),
                    "train/ppl": ppl,
                    "train/lr": lr_now,
                    "train/step": step,
                    "epoch": epoch,
                })

            # 定期保存
            if step > 0 and step % cfg.save_every == 0 and do_step:
                save_ckpt(cfg, epoch, step, processor, dinov2, reducer, text_emb, lm, optimizer, scheduler, scaler)

        # 该轮结束
        denom = (steps_per_ep if steps_per_ep is not None and steps_per_ep > 0 else max(1, it + 1))
        dt = time.time() - t0
        print(f"Epoch {epoch} done in {dt/60:.1f} min | loss={epoch_loss/denom:.4f} | aux={epoch_aux/denom:.4f}")
        save_ckpt(cfg, epoch, step, processor, dinov2, reducer, text_emb, lm, optimizer, scheduler, scaler)

    if use_wandb:
        import wandb
        wandb.finish()


def save_ckpt(cfg: TrainConfig, epoch: int, step: int, processor, dinov2, reducer, text_emb, lm,
              optimizer, scheduler, scaler):
    os.makedirs(os.path.join(cfg.save_dir, cfg.run_name or cfg.stage), exist_ok=True)
    path = os.path.join(cfg.save_dir, cfg.run_name or cfg.stage, f"epoch{epoch}.pt")
    ckpt = {
        "epoch": epoch,
        "step": step,
        "cfg": vars(cfg),
        "lm": lm.state_dict(),
        "reducer": reducer.state_dict(),
        "text_emb": text_emb.state_dict(),
        "dinov2": dinov2.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if hasattr(scaler, "state_dict") else None,
        "tokenizer": processor.tokenizer.get_vocab(),
    }
    torch.save(ckpt, path)
    print(f"[CKPT] Saved to {path}")


@torch.no_grad()
def validate(cfg: TrainConfig, ckpt_path: str):
    device = get_device()
    # 构建
    processor, dinov2, reducer, text_emb, lm, loader, dataset = build_everything(cfg, device)
    # 加载权重
    load_checkpoint(ckpt_path, lm, reducer, text_emb, dinov2)

    lm.eval(); reducer.eval(); text_emb.eval(); dinov2.eval()
    autocast_dtype = select_autocast_dtype(cfg.amp_dtype)

    # 打印验证用的步数策略
    if cfg.steps_per_epoch and cfg.steps_per_epoch > 0:
        steps_info = cfg.steps_per_epoch
        print(f"[VAL/EPOCH-STEPS] mode=truncate | steps_per_epoch={steps_info}")
    else:
        steps_info = infer_steps_per_epoch_auto(dataset, cfg)
        if steps_info is not None:
            samples_auto = infer_dataset_size(dataset, cfg)
            print(f"[VAL/EPOCH-STEPS] mode=full-dataset | auto steps_per_epoch={steps_info} "
                  f"(samples={samples_auto}, batch_size={cfg.batch_size})")
        else:
            print("[VAL/EPOCH-STEPS] mode=full-dataset | steps_per_epoch=unknown (Iterable). Will NOT truncate.")

    n_batches = 0
    loss_sum = 0.0
    aux_sum = 0.0

    for it, batch in enumerate(loader):
        if cfg.steps_per_epoch and cfg.steps_per_epoch > 0 and it >= cfg.steps_per_epoch:
            break
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=(autocast_dtype is not None)):
            loss, aux_loss, logits, labels = forward_step(batch, processor, dinov2, reducer, text_emb, lm,
                                                          device, autocast_dtype)
        loss_sum += float(loss)
        aux_sum += float(aux_loss)
        n_batches += 1

    if n_batches == 0:
        print("No validation batches.")
        return

    avg_loss = loss_sum / n_batches
    avg_aux = aux_sum / n_batches
    ppl = math.exp(max(0.0, avg_loss)) if avg_loss < 20 else float('inf')
    print(f"[VAL] loss_ce={avg_loss:.4f} | aux={avg_aux:.4f} | ppl={ppl:.2f}")


def load_checkpoint(ckpt_path: str, lm, reducer, text_emb, dinov2):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    lm.load_state_dict(ckpt["lm"], strict=True)
    reducer.load_state_dict(ckpt["reducer"], strict=True)
    text_emb.load_state_dict(ckpt["text_emb"], strict=True)
    try:
        dinov2.load_state_dict(ckpt["dinov2"], strict=False)
    except Exception:
        pass
    print(f"[CKPT] Loaded from {ckpt_path}")


# ------------------------------
# Inference (greedy)
# ------------------------------
@torch.no_grad()
def greedy_generate(processor, dinov2, reducer, text_emb, lm, device,
                    image_path: str, prompt: str, max_new_tokens: int = 64):

    from PIL import Image
    lm.eval(); reducer.eval(); text_emb.eval(); dinov2.eval()

    autocast_dtype = torch.bfloat16  # 推理用 bfloat16 足够
    tok = processor.tokenizer
    eos_id = tok.eos_token_id if tok.eos_token_id is not None else tok.convert_tokens_to_ids("</s>")

    # 1) 图像预处理 -> pixel_values (B=1)
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor.image_processor([img], return_tensors="pt")["pixel_values"].to(device)

    # 2) 文本 prompt -> input_ids/attention_mask
    input_ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to(device)
    attn_mask = torch.ones_like(input_ids, device=device)  # (1, L_text)

    # 3) 编码
    vision_tokens = dinov2(pixel_values)            # (1, K, 4096)
    text_inputs  = text_emb(input_ids).to(vision_tokens.dtype)  # (1, L_text, 4096)

    packer = fix_text_img(processor=processor, im_patch_token=None)
    final_emb, final_attn, _, position_ids, _ = packer(
        image_features=vision_tokens,
        inputs_embeds=text_inputs,
        input_ids=input_ids,
        attention_mask=attn_mask,
        labels=None,
    )

    # 4096 -> d_model
    with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
        seq = reducer(final_emb)  # (1, L0, d_model)

    # 4) 自回归生成
    generated_ids = []
    cur_attn = final_attn         # (1, L0)
    cur_pos  = position_ids       # (1, L0)

    for _ in range(max_new_tokens):
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            out = lm(
                inputs_embeds=seq,
                attention_mask=cur_attn,
                labels=None,
                position_ids=cur_pos,
                image_token_mask=None,
            )
            logits = out["logits"]  # (1, L, V)

        next_id = logits[:, -1, :].argmax(dim=-1).item()
        generated_ids.append(next_id)
        print(generated_ids) 
        if next_id == eos_id:
            break

        # 追加下一个 token 的 embedding
        next_token = torch.tensor([[next_id]], device=device)
        next_emb_4096 = text_emb(next_token).to(seq.dtype)
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=True):
            next_emb_model = reducer(next_emb_4096)  # (1,1,d_model)
        seq = torch.cat([seq, next_emb_model], dim=1)  # (1, L+1, d_model)

        # 更新 mask 与 position_ids —— 每次只追加 1
        next_mask = torch.ones((1, 1), device=device, dtype=cur_attn.dtype)
        cur_attn  = torch.cat([cur_attn, next_mask], dim=1)           # (1, L+1)
        next_pos  = cur_pos[:, -1:] + 1                                # (1,1)
        cur_pos   = torch.cat([cur_pos, next_pos], dim=1)              # (1, L+1)

    # 解码
    out_text = tok.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    return out_text


# ------------------------------
# CLI
# ------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train")
    pt.add_argument("--stage", type=str, default="stage1", choices=["stage1", "stage2"])
    pt.add_argument("--model_dir", type=str, required=True)
    pt.add_argument("--dinov2_path", type=str, required=True)
    pt.add_argument("--data_dir", type=str, required=True)
    pt.add_argument("--jsonl_name", type=str, required=True)
    pt.add_argument("--image_dir", type=str, required=True)

    pt.add_argument("--d_in", type=int, default=4096)
    pt.add_argument("--d_model", type=int, default=1024)
    pt.add_argument("--rank", type=int, default=384)
    pt.add_argument("--n_heads", type=int, default=4)
    pt.add_argument("--n_layers", type=int, default=14)

    pt.add_argument("--unfreeze_lm_top_layers", type=int, default=2)
    pt.add_argument("--train_text_embed", type=str, default="true")
    pt.add_argument("--train_dino_proj", type=str, default="true")

    pt.add_argument("--batch_size", type=int, default=2)
    pt.add_argument("--epochs", type=int, default=1)
    pt.add_argument("--steps_per_epoch", type=int, default=1500,
                    help=">0: 截断每轮步数；<=0: 不截断，一轮覆盖全数据（自动推断步数）")
    pt.add_argument("--lr", type=float, default=2e-4)
    pt.add_argument("--weight_decay", type=float, default=0.1)
    pt.add_argument("--warmup_ratio", type=float, default=0.03)
    pt.add_argument("--grad_clip", type=float, default=1.0)
    pt.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    pt.add_argument("--accum_steps", type=int, default=1)

    pt.add_argument("--num_workers", type=int, default=0)
    pt.add_argument("--prefetch_factor", type=int, default=1)
    pt.add_argument("--pin_memory", action="store_true")
    pt.add_argument("--persistent_workers", action="store_true")
    pt.add_argument("--fixed_len", type=int, default=768)
    pt.add_argument("--shuffle", type=str, default="true")

    pt.add_argument("--wandb_project", type=str, default=None)
    pt.add_argument("--run_name", type=str, default=None)
    pt.add_argument("--save_dir", type=str, default="./checkpoints")
    pt.add_argument("--save_every", type=int, default=1500)

    pt.add_argument("--seed", type=int, default=42)
    pt.add_argument("--ckpt", type=str, default=None, help="可选：从该 checkpoint 继续训练")

    # validate
    pv = sub.add_parser("validate")
    pv.add_argument("--ckpt", type=str, required=True)
    pv.add_argument("--model_dir", type=str, required=True)
    pv.add_argument("--dinov2_path", type=str, required=True)
    pv.add_argument("--data_dir", type=str, required=True)
    pv.add_argument("--jsonl_name", type=str, required=True)
    pv.add_argument("--image_dir", type=str, required=True)
    pv.add_argument("--fixed_len", type=int, default=768)
    pv.add_argument("--batch_size", type=int, default=2)
    pv.add_argument("--steps_per_epoch", type=int, default=300,
                    help=">0: 截断；<=0: 不截断（自动推断）")
    pv.add_argument("--amp_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # infer
    pi = sub.add_parser("infer")
    pi.add_argument("--ckpt", type=str, required=True)
    pi.add_argument("--model_dir", type=str, required=True)
    pi.add_argument("--dinov2_path", type=str, required=True)
    pi.add_argument("--image", type=str, required=True)
    pi.add_argument("--prompt", type=str, required=True)
    pi.add_argument("--d_in", type=int, default=4096)
    pi.add_argument("--d_model", type=int, default=1024)
    pi.add_argument("--rank", type=int, default=384)
    pi.add_argument("--n_heads", type=int, default=4)
    pi.add_argument("--n_layers", type=int, default=14)

    return p.parse_args()


def to_bool(x: str | bool) -> bool:
    if isinstance(x, bool):
        return x
    return x.strip().lower() in {"1", "true", "t", "yes", "y"}


def main():
    args = parse_args()

    if args.cmd == "train":
        cfg = TrainConfig(
            model_dir=args.model_dir,
            dinov2_path=args.dinov2_path,
            data_dir=args.data_dir,
            jsonl_name=args.jsonl_name,
            image_dir=args.image_dir,
            d_in=args.d_in,
            d_model=args.d_model,
            rank=args.rank,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            stage=args.stage,
            unfreeze_lm_top_layers=args.unfreeze_lm_top_layers,
            train_text_embed=to_bool(args.train_text_embed),
            train_dino_proj=to_bool(args.train_dino_proj),
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            grad_clip=args.grad_clip,
            amp_dtype=args.amp_dtype,
            accum_steps=args.accum_steps,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            pin_memory=bool(args.pin_memory),
            persistent_workers=bool(args.persistent_workers),
            fixed_len=args.fixed_len,
            shuffle=to_bool(args.shuffle),
            wandb_project=args.wandb_project,
            run_name=args.run_name,
            save_dir=args.save_dir,
            save_every=args.save_every,
            seed=args.seed,
            ckpt=args.ckpt,
        )
        assert cfg.d_model % cfg.n_heads == 0, "d_model 必须能被 n_heads 整除"
        train(cfg)

    elif args.cmd == "validate":
        cfg = TrainConfig(
            model_dir=args.model_dir,
            dinov2_path=args.dinov2_path,
            data_dir=args.data_dir,
            jsonl_name=args.jsonl_name,
            image_dir=args.image_dir,
            fixed_len=args.fixed_len,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
        )
        validate(cfg, args.ckpt)

    elif args.cmd == "infer":
        # 构建轻量 cfg
        cfg = TrainConfig(
            model_dir=args.model_dir,
            dinov2_path=args.dinov2_path,
            data_dir="", jsonl_name="", image_dir="",
            d_in=args.d_in, d_model=args.d_model, rank=args.rank, n_heads=args.n_heads, n_layers=args.n_layers,
        )
        device = get_device()
        processor, dinov2, reducer, text_emb, lm, _, _ = build_everything(cfg, device)
        # 加载 ckpt
        load_checkpoint(args.ckpt, lm, reducer, text_emb, dinov2)
        text = greedy_generate(processor, dinov2, reducer, text_emb, lm, device,
                               image_path=args.image, prompt=args.prompt)
        print("[GENERATED]", text)


if __name__ == "__main__":
    main()
