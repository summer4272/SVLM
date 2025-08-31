# Small Vision Language Model (SVLM)

## 🏗 模型结构示意 (Model Architecture)
![模型结构](Image/MODEL.png)

---
## 庖丁解牛 (Starting small, thinking big)

本项目从零开始手搓视觉-文本统一的多模态大模型，**显存 8GB 即可训练**，适合初学者和低算力人群进行实践与学习。  
This project builds a **vision-language unified multimodal model from scratch**, requiring **only 8GB GPU memory** to train. It is designed for beginners and users with limited resources.  

📌 **特点 / Features:**  
- 🔧 从零实现 / Implemented from scratch  
- 💻 低算力可跑 / Runs on limited resources  
- 📚 适合学习与实践 / Suitable for learning & practice  
- 🌐 简洁高效 / Compact & efficient  

---
## 📷 效果展示 (Project Showcase)

![结果演示](Image/videotogif.gif)


---
## 🚀 快速开始 (Quickstart)

## 🔧 第 1 步：模型配置 (Step 1: Model Setup)

```bash
# 克隆本项目代码仓库
# Clone the project repository
git clone https://github.com/summer4272/SVLM.git
cd SVLM
```
```
# 下载 DINOv2-base 模型 (视觉编码器)
# Download the DINOv2-base model (vision encoder)
git clone https://huggingface.co/facebook/dinov2-base

```
```
# 下载 AutoProcessor（仅需 tokenizer 配置，可不下载完整模型）
# Download AutoProcessor (only tokenizer is needed; model weights can be omitted)
git clone https://huggingface.co/llava-hf/llava-1.5-7b-hf

```
## ⚙️ 第 2 步：环境配置 (Step 2: Environment Setup)

项目提供了 `environment.yml` 文件，可直接基于该文件创建 Conda 环境。  
The project provides an `environment.yml` file, which allows you to create the Conda environment directly.

```bash
# 使用 environment.yml 创建名为 SVLM 的环境
# Create a new Conda environment named "SVLM" from environment.yml
conda env create -f environment.yml -n SVLM
```
## 📂 第 3 步：下载训练数据集 (Step 3: Download Training Dataset)

项目使用 **Chinese-LLaVA-Vision-Instructions** 数据集。  
You can download the **Chinese-LLaVA-Vision-Instructions** dataset as follows:

```bash
# 克隆数据集 
# Clone the dataset 
git clone https://huggingface.co/datasets/Summer1231/SVLM-Data
```
## 🎯 训练指令 (Training Commands)

本项目采用 **双阶段训练**：  
This project uses a **two-stage training** strategy.

---

### 🔵 阶段一：训练视觉塔 (Stage 1: Vision-Heavy Training)

```bash
python train.py train \
  --stage stage1 \
  --model_dir "llava-1.5-7b-hf" \
  --dinov2_path "dinov2-base" \
  --data_dir "Svlmdata" \
  --jsonl_name "pretrain_vlm_data.jsonl" \
  --image_dir "pretrain_images" \
  --unfreeze_lm_top_layers 2 \
  --train_text_embed true \
  --train_dino_proj true \
  --batch_size 2 \
  --epochs 5 \
  --steps_per_epoch 1500 \
  --wandb_project "vlm-train" \
  --run_name "s1-vision-heavy"
```

### 🟢 阶段二：训练语言侧 (Stage 2: Text-Heavy Training)
```bash
python train.py train \
  --stage stage2 \
  --model_dir "llava-1.5-7b-hf" \
  --dinov2_path "dinov2-base" \
  --data_dir "Svlmdata" \
  --jsonl_name "sft_vlm_data.jsonl" \
  --image_dir "sft_images" \
  --unfreeze_lm_top_layers 14 \
  --train_text_embed true \
  --train_dino_proj false \
  --batch_size 4 \
  --epochs 10 \
  --steps_per_epoch -1 \
  --ckpt "s1-vision-heavy/epoch.pt" \
  --wandb_project "vlm-train" \
  --run_name "s2-text-heavy"
```
### 🧪 验证模型 (Validation)
```bash
python train.py validate \
  --ckpt "s2-text-heavy/epoch.pt" \
  --model_dir "llava-1.5-7b-hf" \
  --dinov2_path "dinov2-base" \
  --data_dir "Svlmdata" \
  --jsonl_name "sft_vlm_data.jsonl" \
  --image_dir "sft_images" \
  --steps_per_epoch -1 \
  --amp_dtype bfloat16

```
### 💡 推理 (Inference)
```bash
python train.py infer \
  --ckpt "epoch.pt" \
  --model_dir "llava-1.5-7b-hf" \
  --dinov2_path "dinov2-base" \
  --image "your_image.jpg" \
  --prompt "请根据图像回答：这张图里有什么？" \
  --d_in 4096 \
  --d_model 1024 \
  --n_heads 4 \
  --n_layers 14
```
---

## 📈 训练效果曲线 (Training Curves)
```
这里仅仅展示了第一阶段训练曲线
```
![训练曲线](Image/train.png)
