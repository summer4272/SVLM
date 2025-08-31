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
---

## 📈 训练效果曲线 (Training Curves)

![训练曲线](Image/train.png)
