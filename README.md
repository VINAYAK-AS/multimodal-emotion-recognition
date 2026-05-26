# 🎭 Multimodal Emotion Recognition (Speech + Text)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-F9AB00?logo=huggingface&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-success)

A comprehensive deep learning pipeline that classifies human emotion by dynamically fusing acoustic speech features with semantic text features. This project achieves **100% validation accuracy** on the TESS dataset by employing a custom cross-modal attention mechanism that intelligently weighs the importance of audio versus text inputs.

## 📑 Table of Contents
- [Overview & Key Features](#-overview--key-features)
- [Model Architecture](#-model-architecture)
- [The "Carrier Phrase" Insight](#-the-carrier-phrase-insight)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Training Pipelines](#-training-pipelines)
- [Results](#-results)

---

## 🚀 Overview & Key Features
Emotion recognition is rarely single-threaded. Humans interpret tone of voice and spoken words simultaneously. This project mimics that cognitive process by training independent neural networks on audio and text, and fusing their latent representations.

* **Dual-Modality Processing:** Handles both `.wav` audio files and their text transcripts.
* **Cross-Modal Attention:** A custom mechanism that mathematically determines whether the audio tone or the text meaning is more crucial for the final prediction.
* **Modular Codebase:** Clean separation of speech, text, and fusion pipelines for easy debugging and scaling.

---

## 🧠 Model Architecture

The system is built on a three-pronged architecture:

### 1. Speech Pipeline (Acoustic Features)
* **Input:** Raw audio waveforms.
* **Processing:** Feature extraction via `librosa` to generate Mel Spectrograms.
* **Network:** A robust **CNN + BiLSTM** (Convolutional Neural Network + Bidirectional Long Short-Term Memory) network captures spatial frequencies and sequential audio patterns over time.
* **Output:** 512-dimensional acoustic representation.

### 2. Text Pipeline (Semantic Features)
* **Input:** Text transcripts.
* **Processing:** Tokenization using the pre-trained HuggingFace tokenizer.
* **Network:** Fine-tuned **BERT** (`bert-base-uncased`) transformer model to extract semantic emotional context.
* **Output:** 768-dimensional semantic representation.

### 3. Fusion Pipeline (The Brain)
* **Input:** The 512-D speech vector and 768-D text vector.
* **Network:** Features are fed into a **Cross-Modal Multi-Head Attention** mechanism. The network dynamically attends to the most informative modality, concatenates them, and projects the features through a fully connected classifier.
* **Output:** Softmax probabilities across 7 emotion classes.

---

## 💡 The "Carrier Phrase" Insight
One of the most significant findings in this project was how the attention mechanism handled the dataset's constraints.

The dataset utilizes static "carrier phrases" (e.g., *"Say the word 'door'"*) spoken with different emotional inflections. Because the semantic meaning of the words does not change between emotions, the standalone Text Pipeline logically performs at random chance (~14.6% accuracy). 

Instead of failing, the **Multimodal Fusion model successfully learned to ignore the uninformative text features entirely** and mathematically shifted its attention weights heavily to the acoustic features. This dynamically adaptive behavior is exactly what cross-modal attention is designed to do, bringing the final multimodal accuracy to a perfect **100%**.

---

## 📊 Dataset
Trained on the **Toronto Emotional Speech Set (TESS)**. 
* **Size:** 2,800 audio samples.
* **Classes (7):** `Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`.

---

## 📂 Project Structure
```text
multimodal-emotion-recognition/
│
├── models/
│   ├── speech_pipeline/       # CNN+BiLSTM architecture & training scripts
│   ├── text_pipeline/         # BERT architecture & tokenizer
│   └── fusion_pipeline/       # Cross-modal attention & final classifier
│
├── Results/
│   ├── plots/                 # t-SNE visualizations and Confusion Matrices
│   ├── speech_report.txt      # Classification metrics for speech-only
│   ├── text_report.txt        # Classification metrics for text-only
│   └── fusion_report.txt      # Final multimodal classification metrics
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation