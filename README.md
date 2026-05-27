# 🎭 Multimodal Emotion Recognition (Speech + Text)

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-F9AB00?logo=huggingface&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-100%25-success)

A comprehensive deep learning pipeline that classifies human emotion by dynamically fusing acoustic speech features with semantic text features. This project achieves **100% validation accuracy** on the TESS dataset by employing a custom cross-modal attention mechanism that intelligently weighs the importance of audio versus text inputs.

## 📑 Table of Contents
1. [Installation & Setup (Step-by-Step)](#1-installation--setup-step-by-step)
2. [Execution: How to Run the Code](#2-execution-how-to-run-the-code)
3. [Project Structure](#3-project-structure)
4. [A. Architecture Decisions](#a-architecture-decisions)
5. [B. Experiments & System Comparison](#b-experiments--system-comparison)
6. [C. In-Depth Analysis](#c-in-depth-analysis)

---

## 🛠️ 1. Installation & Setup (Step-by-Step)

Follow these exact steps to set up the environment on your local machine.

**Step 1: Clone the repository**
Open your terminal or command prompt and run:
```bash
git clone [https://github.com/VINAYAK-AS/multimodal-emotion-recognition.git](https://github.com/VINAYAK-AS/multimodal-emotion-recognition.git)
cd multimodal-emotion-recognition
```

**Step 2: Create a Virtual Environment**
Isolating dependencies prevents conflicts with other Python projects on your machine.
* **On Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **On macOS/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

**Step 3: Install Dependencies**
Install all required libraries (PyTorch, Transformers, Librosa, etc.):
```bash
pip install -r requirements.txt
```
*(Note: If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch from the official PyTorch website to drastically speed up model training.)*

**Step 4: Prepare the Dataset**
1. Download the **TESS (Toronto Emotional Speech Set)** dataset from Kaggle or your provided source.
2. Create a folder named `data/` in the root directory of this project.
3. Extract and place all the `.wav` audio files directly into the `data/` folder.

---

## ⚙️ 2. Execution: How to Run the Code

**CRITICAL NOTE:** The pipelines **must** be trained sequentially. The final Multimodal Fusion model requires the saved `.pt` weight files from the independent Speech and Text models. 

**Step 1: Train the Speech Pipeline**
This script processes the audio files, trains the CNN+BiLSTM network, and saves the acoustic model weights.
```bash
python models/speech_pipeline/train.py
```
*Expected Output: `speech_model.pt` saved, with classification reports and t-SNE plots pushed to the `/Results` folder.*

**Step 2: Train the Text Pipeline**
This script tokenizes the transcripts, fine-tunes the BERT model, and saves the semantic model weights.
```bash
python models/text_pipeline/train.py
```
*Expected Output: `text_model.pt` saved.*

**Step 3: Train the Fusion Pipeline**
Once both independent models are trained and their `.pt` files exist, execute the fusion script. This freezes the base layers and trains the Cross-Modal Attention mechanism.
```bash
python models/fusion_pipeline/train.py
```
*Expected Output: The final `fusion_model.pt` saved, yielding 100% accuracy, alongside final multimodal confusion matrices and t-SNE plots in the `/Results` directory.*

---

## 📂 3. Project Structure
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
├── data/                      # TESS dataset audio files (User created)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🧠 A. Architecture Decisions

The architecture is constructed using highly specialized, independent neural blocks for each modality to ensure feature independence prior to fusion.

* **Speech Block (Temporal Modelling):** 16 kHz waveforms are converted to 2D Mel Spectrograms via `librosa`. These are fed into a **CNN + BiLSTM** network. 
  * *Why?* The CNN extracts local texture and pitch frequencies, while the BiLSTM captures temporal dependencies from past and future audio frames, outputting a robust 512-dimensional acoustic representation.
* **Text Block (Contextual Modelling):** Transcripts are tokenized using HuggingFace and passed through a pre-trained **BERT** model. 
  * *Why?* Transformers excel at capturing bidirectional semantic context. The `[CLS]` token serves as a dense, 768-D semantic summary vector of the utterance.
* **Fusion Block:** Cross-Modal Multi-Head Attention.
  * *Why?* Naive concatenation of a strong speech vector and a weak text vector would confuse the classifier. The attention mechanism dynamically computes alignment scores, mathematically learning to weigh the modality that contains the strongest emotion-discriminative signal before final classification.

---

## 📊 B. Experiments & System Comparison

Three independent experiments were conducted on a validation set of 560 samples to isolate the predictive power of each modality.

| Pipeline Architecture | Validation Samples | Macro F1-Score | Overall Accuracy |
| :--- | :--- | :--- | :--- |
| **Text-Only (BERT)** | 560 | 0.05 | **15.0%** |
| **Speech-Only (CNN-BiLSTM)** | 560 | 1.00 | **100.0%** |
| **Multimodal Fusion (Attention)** | 560 | 1.00 | **100.0%** |

**Conclusion:** The text pipeline operated at random chance (15%) due to dataset constraints, while the speech pipeline achieved a flawless 100%. The Multimodal Fusion model matched the 100% accuracy, proving the attention mechanism successfully filtered out the noisy text vectors.

---

## 🔍 C. In-Depth Analysis

### 1. Which emotions are easiest/hardest to classify?
* **Easiest (Angry, Fear, Sad):** These possess highly distinct acoustic signatures. "Angry" features high energy and harsh spectral texture, while "Sad" features low pitch and soft energy. The network easily isolates these extremes.
* **Hardest (Disgust vs. Surprise):** These are the most difficult to separate geometrically. Both involve high arousal and abrupt vocal gestures (sharp pitch excursions and sudden energy bursts). The boundary between intense disgust and sharp surprise is incredibly narrow, requiring deep temporal modeling to establish a clear decision boundary.

### 2. When does fusion help most?
Fusion is typically most beneficial when modalities provide complementary information. However, because the TESS dataset utilizes static carrier phrases (e.g., *"Say the word door"*), the text modality is uninformative. In this specific architecture, **fusion helps most by demonstrating fault tolerance and active degradation prevention.** The cross-modal attention mechanism identified the text as mathematical noise and reduced its weight to near-zero, proving the architecture is highly resilient to corrupted or useless data streams.

### 3. Error Analysis: Text-Baseline Failure Cases
Because the final Speech and Fusion pipelines achieved a mathematically perfect 100% accuracy, traditional error analysis is applied to the **Text-Only pipeline** to demonstrate the severe classification collapse the Fusion block was forced to resolve. The Text model failed catastrophically across 5 classes:
1. **Disgust:** 0% Recall
2. **Fear:** 0% Recall
3. **Happy:** 0% Recall
4. **Sad:** 0% Recall
5. **Surprise:** 0% Recall

*Analysis of the Failure:* The text pipeline collapsed into a degenerate state, predicting "Angry" for almost every sample. Because the lexical inputs were identical across all recordings, the BERT model had no semantic variance to learn from. The Multimodal Fusion block successfully corrected 100% of these failure cases by shifting attention exclusively to the acoustic features.

### 4. Visualizing the Separability of Emotion Clusters
The representation learning of the network is confirmed via t-SNE dimensionality reduction plots (available in the `/Results` folder):
* **Temporal Modelling Block (Speech):** The t-SNE projection reveals seven perfectly isolated and dense geometric clusters, allowing the linear classifier to draw flawless boundaries.
* **Contextual Modelling Block (Text):** The t-SNE representation of the BERT `[CLS]` tokens shows complete entanglement. The seven emotion classes form a single, overlapping cloud with zero spatial separability, visually explaining the 15% accuracy collapse.
* **The Fusion Block:** The fused representations perfectly mirror the Temporal block. The cross-modal attention mechanism successfully discarded the entangled Contextual cloud, resulting in seven distinctly separated clusters that drive the final 100% validation accuracy.