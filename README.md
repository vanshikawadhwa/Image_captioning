# Image Captioning with ResNet50 & LSTM

A robust end-to-end image captioning pipeline built with TensorFlow and Keras. This model uses a pre-trained **ResNet50** as an encoder to extract visual features and a **Multi-layer LSTM** as a decoder to generate descriptive captions.

## 🚀 Performance
The model achieves a **19.12% BLEU-4 score** on the Flickr8k dataset after 50 epochs of training.

| Metric | Score |
|---|---|
| **BLEU-1** | 60.76% |
| **BLEU-4** | 19.12% |
| **METEOR** | 39.80% |
| **ROUGE-L** | 48.36% |

## 🏗️ Architecture
- **Encoder:** ResNet50 (pre-trained on ImageNet) with the top classification layer removed.
- **Decoder:** 
    - Word Embedding Layer
    - Sequential LSTM stacks (512 units)
    - Merged Image-Language processing
    - Softmax Output layer (Vocab size ~1,670)

## 📁 Repository Structure
- `config.py`: Central hyperparameters and paths.
- `data_loader.py`: Handles loading images and parsing captions.
- `preprocessor.py`: Text cleaning, vocabulary building, and encoding.
- `feature_extractor.py`: ResNet50 feature extraction with local caching.
- `dataset.py`: Training array generation using teacher-forcing.
- `model.py`: Keras functional API implementation of the model.
- `train.py`: Full training pipeline.
- `predict.py`: Inference script for generating captions on new images.
- `evaluate.py`: Evaluation suite for BLEU, METEOR, and ROUGE metrics.
- `visualise.py`: Matplotlib helpers for overlaying captions on images.

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/image-captioning.git
   cd image-captioning
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data Preparation:**
   Place the **Flickr8k** dataset inside the `data/` folder as follows:
   ```text
   data/
   └── Flickr8k_Dataset/
       └── Images/
   ```

4. **Run Training:**
   ```bash
   python train.py
   ```

5. **Run Evaluation:**
   ```bash
   python evaluate.py --n 500
   ```

6. **Generate Captions:**
   ```bash
   python predict.py --n 5
   ```

## 🖼️ Results
Sample AI-generated captions and overlays can be found in the [screenshots/](./screenshots/) folder.

---
*Developed as a modular, production-ready implementation of the Merge Architectures for Image Captioning.*
