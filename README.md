
# 🖼️ Image Captioning using Transfer Learning and GRU
### Deep Learning | Computer Vision | Natural Language Generation

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow/Keras-orange)
![Task](https://img.shields.io/badge/Task-Image%20Captioning-green)
![Dataset](https://img.shields.io/badge/Dataset-Flickr8k-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Project Overview

This repository implements an **end‑to‑end deep learning system for automatic image caption generation**.  
The model combines **computer vision and natural language generation** using a **CNN‑RNN encoder‑decoder architecture**.

The pipeline extracts visual features using a **pretrained VGG16 convolutional neural network** and generates captions using a **GRU-based sequence decoder**.

The model is trained on the **Flickr8k dataset**, which contains images paired with human‑written descriptions.

This project demonstrates several key machine learning concepts:

- Transfer Learning
- CNN‑RNN Encoder‑Decoder architectures
- Natural Language Generation
- Tokenization and sequence modeling
- Autoregressive inference
- Efficient training with data generators

---

# 🧠 Model Architecture

The system follows a **two-stage architecture**:

```
Image → CNN Encoder → Feature Vector → GRU Decoder → Generated Caption
```

### 1️⃣ Image Encoder (CNN)

A **pre-trained VGG16 network** is used to extract high-level semantic features from images.

Instead of training the CNN from scratch, the system computes **transfer values**:

Feature vector size:

```
4096
```

These vectors encode the semantic content of the image and serve as the initial context for the language model.

---

### 2️⃣ Caption Decoder (GRU)

The decoder generates captions **word-by-word** using a stacked **GRU (Gated Recurrent Unit)** architecture.

Architecture:

```
Transfer Values (4096)
        │
Dense Layer (4096 → 512)
        │
Initial GRU State
        │
Embedding Layer (10000 → 128)
        │
GRU Layer (512)
GRU Layer (512)
GRU Layer (512)
        │
Dense + Softmax
        │
Vocabulary Prediction
```

The decoder predicts the next word conditioned on:

- the encoded image representation
- previously generated words

---

# 📂 Dataset

### Flickr8k Dataset

The dataset contains:

- **8,000 images**
- **5 captions per image**
- **40,000 total captions**

Each caption describes the visual content of an image.

Example caption:

```
ssss a dog running through the grass eeee
```

Special tokens:

| Token | Meaning |
|------|------|
ssss | Start of sentence |
eeee | End of sentence |

---

# ⚙️ Data Processing Pipeline

### Caption Processing

1. Add start/end tokens
2. Tokenize captions
3. Build vocabulary
4. Convert captions to integer sequences

### Image Feature Extraction

Images are processed using **VGG16** to compute **transfer values**, which are cached to accelerate training.

Advantages:

- Faster training
- Reduced GPU memory usage
- Reusable visual embeddings

---

# 🏋️ Model Training

The model is trained as a **sequence prediction problem**.

Example:

```
Input  : ssss a dog running
Target : a dog running eeee
```

The model learns to predict the **next word in the sequence**.

### Loss Function

```
Sparse Categorical Cross‑Entropy
```

### Optimizer

```
RMSprop
```

### Training Configuration

| Parameter | Value |
|----------|------|
Batch Size | 384 |
Embedding Dimension | 128 |
GRU Hidden Units | 512 |
Vocabulary Size | 10,000 |
Epochs | 20 |

---

# 🔁 Data Generator

A custom **data generator** dynamically creates training batches.

Steps:

1️⃣ Randomly select images  
2️⃣ Randomly select one caption per image  
3️⃣ Pad token sequences  
4️⃣ Shift input/output tokens  

Example training pair:

```
Decoder Input : ssss a dog running
Target Output : a dog running eeee
```

This method enables efficient training even with large datasets.

---

# ✨ Caption Generation

During inference the caption is generated using **autoregressive decoding**.

Algorithm:

1️⃣ Start with token `ssss`  
2️⃣ Predict next word  
3️⃣ Feed prediction back into the decoder  
4️⃣ Stop when `eeee` appears  

Example output:

```
Predicted Caption:
a dog running through the grass
```

---

# 📊 Example Predictions

Example outputs produced by the model:

### Example 1

Image: Dog running

Prediction:

```
a dog running through the grass
```

### Example 2

Image: Giraffes in field

Prediction:

```
two giraffes standing in a grassy field
```

---

# 📁 Repository Structure

```
Image_Captioning/
│
├── Image_Captioning_using_Machine_Translation.ipynb
├── README.md
│
├── dataset/
│   ├── Flickr8k_Dataset/
│   └── captions.txt
│
├── logs/
│   └── TensorBoard logs
│
└── checkpoints/
    └── model weights
```

---

# 🚀 Installation

Clone the repository:

```
git clone https://github.com/yourusername/Image_Captioning.git
cd Image_Captioning
```

Install dependencies:

```
pip install tensorflow numpy pandas matplotlib scikit-learn pillow joblib
```

---

# ▶️ Running the Project

### Step 1 — Mount Drive (if using Colab)

```
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2 — Compute Image Features

Run the VGG16 pipeline to generate transfer values.

### Step 3 — Train the Model

```
decoder_model.fit(
    x=generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    callbacks=callbacks
)
```

### Step 4 — Generate Captions

```
generate_caption("example.jpg")
```

---

# 🔬 Key Machine Learning Concepts

This project demonstrates:

- Transfer Learning with CNNs
- Sequence-to-Sequence modeling
- GRU recurrent networks
- Word embeddings
- Autoregressive language modeling
- Efficient batch generation
- Deep learning for multimodal AI

---

# 📈 Future Improvements

Potential extensions:

- Transformer-based captioning models
- Attention mechanisms
- Beam search decoding
- Training on larger datasets (Flickr30k, MS‑COCO)
- Modern encoders (ResNet, EfficientNet, Vision Transformers)

---

# 📚 References

1. Vinyals et al., **Show and Tell: A Neural Image Caption Generator**, CVPR 2015  
2. Simonyan & Zisserman, **Very Deep Convolutional Networks for Large-Scale Image Recognition**, 2015  
3. Cho et al., **Learning Phrase Representations using RNN Encoder‑Decoder**, 2014  

---

# 👨‍💻 Author

**Mohammad Hafezan**  
MSc Researcher — AI & Deep Learning  
Lakehead University

Research Interests:

- Deep Neural Networks
- AI Security
- Efficient AI Architectures
- GPU/Hardware‑Accelerated AI

---

⭐ If you find this project useful, consider starring the repository.
