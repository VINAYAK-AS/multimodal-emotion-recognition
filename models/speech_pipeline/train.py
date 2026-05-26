"""
Speech-Only Emotion Recognition Pipeline
Architecture:
  Preprocessing  -> librosa (resample, trim silence, normalize)
  Feature Extraction -> Log-Mel Spectrogram (time_steps x 128 mel bins)
  Temporal Modelling -> CNN + BiLSTM
  Classifier -> Fully Connected + Softmax
"""

import os, sys, random, json
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# ─── Config ─────────────────────────────────────────────────────────────────
CFG = dict(
    data_root   = "/content/TESS",          # adjust if needed
    sample_rate = 22050,
    n_mels      = 128,
    hop_length  = 512,
    max_frames  = 128,                       # pad / truncate to this length
    batch_size  = 32,
    epochs      = 40,
    lr          = 1e-3,
    weight_decay= 1e-4,
    device      = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir    = "/content/drive/MyDrive/emotion_recognition/models/speech_pipeline",
    results_dir = "/content/drive/MyDrive/emotion_recognition/Results",
)
os.makedirs(CFG["save_dir"],    exist_ok=True)
os.makedirs(CFG["results_dir"], exist_ok=True)
print(f"Using device: {CFG['device']}")

# ─── 1. Build file list + labels from TESS folder structure ─────────────────
def build_dataframe(data_root):
    """
    TESS folder structure:
        <data_root>/YAF_<word>_<emotion>/<word>_<emotion>.wav
    Emotion tags in filenames: angry, disgust, fear, happy, neutral, ps (= pleasant_surprise), sad
    """
    records = []
    for folder in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".wav"):
                continue
            # emotion is the last underscore-separated token before .wav
            emotion = fname.split("_")[-1].replace(".wav", "").lower()
            if emotion == "ps":
                emotion = "surprise"
            word = "_".join(fname.split("_")[:-1])
            records.append({
                "path"   : os.path.join(folder_path, fname),
                "text"   : word.replace("_", " "),
                "emotion": emotion,
            })
    df = pd.DataFrame(records)
    print(f"Dataset: {len(df)} samples | Emotions: {sorted(df.emotion.unique())}")
    return df

# ─── 2. Preprocessing + Feature Extraction ──────────────────────────────────
def load_mel(path, cfg):
    """Load wav → log-mel spectrogram (max_frames × n_mels)."""
    y, sr = librosa.load(path, sr=cfg["sample_rate"], mono=True)
    # trim leading/trailing silence
    y, _ = librosa.effects.trim(y, top_db=20)
    # normalize amplitude
    if y.max() != 0:
        y = y / np.abs(y).max()
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=cfg["n_mels"], hop_length=cfg["hop_length"]
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)  # (n_mels, T)
    log_mel = log_mel.T                              # (T, n_mels)
    # pad or truncate along time axis
    T = log_mel.shape[0]
    if T < cfg["max_frames"]:
        pad = np.zeros((cfg["max_frames"] - T, cfg["n_mels"]))
        log_mel = np.vstack([log_mel, pad])
    else:
        log_mel = log_mel[: cfg["max_frames"]]
    return log_mel.astype(np.float32)               # (max_frames, n_mels)

# ─── 3. Dataset ──────────────────────────────────────────────────────────────
class SpeechDataset(Dataset):
    def __init__(self, df, label_encoder, cfg):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder
        self.cfg = cfg

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel  = load_mel(row["path"], self.cfg)          # (T, n_mels)
        mel  = torch.tensor(mel).unsqueeze(0)           # (1, T, n_mels)  – channel dim for CNN
        label = self.le.transform([row["emotion"]])[0]
        return mel, torch.tensor(label, dtype=torch.long)

# ─── 4. Model: CNN → BiLSTM → Classifier ────────────────────────────────────
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(3,3), pool=(2,2)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.MaxPool2d(pool),
            nn.Dropout2d(0.25),
        )
    def forward(self, x): return self.net(x)

class SpeechEmotionModel(nn.Module):
    """
    Input : (B, 1, T, n_mels)
    CNN   : local feature extraction over time×frequency
    BiLSTM: temporal modelling of CNN feature sequence
    FC    : classifier
    """
    def __init__(self, n_mels=128, n_classes=7, lstm_hidden=256, lstm_layers=2):
        super().__init__()
        # CNN Feature Extraction
        self.cnn = nn.Sequential(
            CNNBlock(1,  32, pool=(2,2)),
            CNNBlock(32, 64, pool=(2,2)),
            CNNBlock(64, 128, pool=(2,2)),
        )
        # after 3 × pool(2,2): freq dim = n_mels // 8
        cnn_freq_out = n_mels // 8
        cnn_ch_out   = 128
        self.cnn_proj = cnn_ch_out * cnn_freq_out   # flattened per time-step

        # Temporal Modelling: BiLSTM
        self.bilstm = nn.LSTM(
            input_size  = self.cnn_proj,
            hidden_size = lstm_hidden,
            num_layers  = lstm_layers,
            batch_first = True,
            bidirectional = True,
            dropout = 0.3,
        )
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x, return_repr=False):
        # x: (B, 1, T, n_mels)
        cnn_out = self.cnn(x)                        # (B, 128, T', F')
        B, C, T, F = cnn_out.shape
        cnn_seq = cnn_out.permute(0, 2, 1, 3)        # (B, T', C, F')
        cnn_seq = cnn_seq.reshape(B, T, C * F)       # (B, T', proj)
        lstm_out, _ = self.bilstm(cnn_seq)            # (B, T', 2H)
        # mean-pool across time
        repr_vec = lstm_out.mean(dim=1)               # (B, 2H)
        logits   = self.classifier(repr_vec)
        if return_repr:
            return logits, repr_vec
        return logits

# ─── 5. Training helpers ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for mel, labels in tqdm(loader, leave=False, desc="train"):
        mel, labels = mel.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(mel)
        loss   = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    all_reprs = []
    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        logits, reprs = model(mel, return_repr=True)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_reprs.append(reprs.cpu().numpy())
    reprs_arr = np.concatenate(all_reprs, axis=0)
    return total_loss / total, correct / total, all_preds, all_labels, reprs_arr

# ─── 6. Visualisations ───────────────────────────────────────────────────────
def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"],  label="train"); axes[1].plot(history["val_acc"],  label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.suptitle("Speech Pipeline – Training History")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_confusion(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names,
                cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Speech Pipeline – Confusion Matrix")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_tsne(reprs, labels, class_names, save_path, title="t-SNE"):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
    emb  = tsne.fit_transform(reprs)
    palette = sns.color_palette("tab10", len(class_names))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = np.array(labels) == i
        ax.scatter(emb[mask, 0], emb[mask, 1], label=name, color=palette[i], alpha=0.6, s=15)
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ─── 7. Main ─────────────────────────────────────────────────────────────────
def main():
    df = build_dataframe(CFG["data_root"])

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["emotion"])
    class_names  = list(le.classes_)
    n_classes    = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = SpeechDataset(train_df, le, CFG)
    val_ds   = SpeechDataset(val_df,   le, CFG)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    model     = SpeechEmotionModel(n_mels=CFG["n_mels"], n_classes=n_classes).to(CFG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc  = 0.0
    best_preds, best_labels, best_reprs = None, None, None

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, CFG["device"])
        vl_loss, vl_acc, preds, labels_list, reprs = eval_epoch(model, val_loader, criterion, CFG["device"])
        scheduler.step()

        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(vl_acc)

        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} acc={tr_acc:.4f} | val_loss={vl_loss:.4f} acc={vl_acc:.4f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_preds, best_labels, best_reprs = preds, labels_list, reprs
            torch.save(model.state_dict(), os.path.join(CFG["save_dir"], "best_speech_model.pt"))
            print(f"  ✓ Saved best model (val_acc={best_acc:.4f})")

    # ── Save artefacts ──
    report_str = classification_report(best_labels, best_preds, target_names=class_names)
    print("\nClassification Report:\n", report_str)
    with open(os.path.join(CFG["results_dir"], "speech_report.txt"), "w") as f:
        f.write(report_str)

    plot_history(history, os.path.join(CFG["results_dir"], "plots", "speech_history.png"))
    plot_confusion(best_labels, best_preds, class_names,
                   os.path.join(CFG["results_dir"], "plots", "speech_confusion.png"))
    plot_tsne(best_reprs, best_labels, class_names,
              os.path.join(CFG["results_dir"], "plots", "speech_tsne.png"),
              title="t-SNE – Speech BiLSTM Representations")

    # Save label encoder mapping
    le_map = {cls: int(i) for i, cls in enumerate(le.classes_)}
    with open(os.path.join(CFG["save_dir"], "label_encoder.json"), "w") as f:
        json.dump(le_map, f)

    print(f"\nBest Val Accuracy: {best_acc:.4f}")
    print("Done. Artefacts saved to", CFG["results_dir"])

if __name__ == "__main__":
    main()
