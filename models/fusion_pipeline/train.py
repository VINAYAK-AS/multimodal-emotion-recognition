"""
Multimodal Fusion Pipeline (Speech + Text)
Architecture:
  Speech branch : CNN + BiLSTM  (from speech_pipeline)
  Text branch   : BERT fine-tuned (from text_pipeline)
  Fusion        : Cross-modal Attention + Concatenation → projection
  Classifier    : FC → Softmax
"""

import os, sys, random, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ─── Corrected Absolute Imports ──────────────────────────────────────────────
from models.speech_pipeline.train import SpeechEmotionModel, build_dataframe as speech_build_df, load_mel, CFG as SCFG
from models.text_pipeline.train import BertEmotionModel, preprocess_text, CFG as TCFG

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─── Config ─────────────────────────────────────────────────────────────────
CFG = dict(
    data_root   = "/content/TESS",
    sample_rate = SCFG["sample_rate"],
    n_mels      = SCFG["n_mels"],
    hop_length  = SCFG["hop_length"],
    max_frames  = SCFG["max_frames"],
    bert_model  = TCFG["bert_model"],
    max_len     = TCFG["max_len"],
    batch_size  = 16,             # smaller – two heavy models
    epochs      = 25,
    lr_speech   = 1e-4,
    lr_text     = 2e-5,
    lr_fusion   = 5e-4,
    weight_decay= 1e-4,
    speech_ckpt = "/content/drive/MyDrive/emotion_recognition/models/speech_pipeline/best_speech_model.pt",
    text_ckpt   = "/content/drive/MyDrive/emotion_recognition/models/text_pipeline/best_text_model.pt",
    text_tok_dir= "/content/drive/MyDrive/emotion_recognition/models/text_pipeline",
    save_dir    = "/content/drive/MyDrive/emotion_recognition/models/fusion_pipeline",
    results_dir = "/content/drive/MyDrive/emotion_recognition/Results",
    device      = "cuda" if torch.cuda.is_available() else "cpu",
)
os.makedirs(CFG["save_dir"],    exist_ok=True)
os.makedirs(CFG["results_dir"], exist_ok=True)
print(f"Device: {CFG['device']}")

# ─── Multimodal Dataset ──────────────────────────────────────────────────────
class MultimodalDataset(Dataset):
    def __init__(self, df, label_encoder, tokenizer, cfg):
        self.df  = df.reset_index(drop=True)
        self.le  = label_encoder
        self.tok = tokenizer
        self.cfg = cfg

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = self.le.transform([row["emotion"]])[0]

        # Speech
        mel = load_mel(row["path"], self.cfg)
        mel = torch.tensor(mel).unsqueeze(0)   # (1, T, n_mels)

        # Text
        text = preprocess_text(row["text"])
        enc  = self.tok(text, max_length=self.cfg["max_len"],
                        padding="max_length", truncation=True, return_tensors="pt")
        return {
            "mel"           : mel,
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(label, dtype=torch.long),
        }

# ─── Cross-Modal Attention Fusion ────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    def __init__(self, dim_q, dim_kv, heads=4):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=dim_q, num_heads=heads,
                                          kdim=dim_kv, vdim=dim_kv, batch_first=True)
        self.norm = nn.LayerNorm(dim_q)

    def forward(self, query, key_value):
        attn_out, _ = self.mha(query, key_value, key_value)
        return self.norm(query + attn_out).squeeze(1)

class FusionModel(nn.Module):
    def __init__(self, speech_dim=512, text_dim=768, n_classes=7, proj_dim=256):
        super().__init__()
        self.speech_attn_text = CrossModalAttention(speech_dim, text_dim)
        self.text_attn_speech = CrossModalAttention(text_dim, speech_dim)
        fusion_in = speech_dim + text_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_in, proj_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, speech_repr, text_repr, return_repr=False):
        s = speech_repr.unsqueeze(1)
        t = text_repr.unsqueeze(1)
        s_attended = self.speech_attn_text(s, t)
        t_attended = self.text_attn_speech(t, s)
        fused = torch.cat([s_attended, t_attended], dim=-1)
        proj  = self.fusion_proj(fused)
        logits = self.classifier(proj)
        if return_repr:
            return logits, proj
        return logits

class MultimodalEmotionModel(nn.Module):
    def __init__(self, speech_backbone, text_backbone, n_classes, speech_dim=512, text_dim=768):
        super().__init__()
        self.speech = speech_backbone
        self.text   = text_backbone
        self.fusion = FusionModel(speech_dim, text_dim, n_classes)

    def forward(self, mel, input_ids, attention_mask, return_repr=False):
        _, speech_repr = self.speech(mel, return_repr=True)
        _, text_repr   = self.text(input_ids, attention_mask, return_repr=True)
        return self.fusion(speech_repr, text_repr, return_repr=return_repr)

# ─── Training helpers ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, leave=False, desc="train"):
        mel    = batch["mel"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(mel, ids, mask)
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
    all_preds, all_labels, all_reprs = [], [], []
    for batch in loader:
        mel    = batch["mel"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits, reprs = model(mel, ids, mask, return_repr=True)
        loss   = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds  = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_reprs.append(reprs.cpu().numpy())
    return (total_loss / total, correct / total,
            all_preds, all_labels, np.concatenate(all_reprs))

# ─── Plot helpers ─────────────────────────────────────────────────────────────
def _plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"],  label="train"); axes[1].plot(history["val_acc"],  label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.suptitle("Fusion Pipeline – Training History")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def _plot_confusion(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names,
                cmap="Purples", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Fusion Pipeline – Confusion Matrix")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def _plot_tsne(reprs, labels, class_names, save_path, title="t-SNE"):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=SEED)
    emb  = tsne.fit_transform(reprs)
    palette = sns.color_palette("tab10", len(class_names))
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        mask = np.array(labels) == i
        ax.scatter(emb[mask, 0], emb[mask, 1], label=name, color=palette[i], alpha=0.6, s=15)
    ax.legend(markerscale=2, fontsize=8); ax.set_title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    df = speech_build_df(CFG["data_root"])

    le = LabelEncoder()
    df["label"]  = le.fit_transform(df["emotion"])
    class_names  = list(le.classes_)
    n_classes    = len(class_names)
    print(f"Classes: {class_names}")

    tokenizer = BertTokenizer.from_pretrained(CFG["text_tok_dir"])

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
    train_ds = MultimodalDataset(train_df, le, tokenizer, CFG)
    val_ds   = MultimodalDataset(val_df,   le, tokenizer, CFG)
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    # Load pretrained backbones
    speech_backbone = SpeechEmotionModel(n_mels=CFG["n_mels"], n_classes=n_classes)
    speech_backbone.load_state_dict(torch.load(CFG["speech_ckpt"], map_location="cpu"))

    text_backbone = BertEmotionModel(CFG["bert_model"], n_classes)
    text_backbone.load_state_dict(torch.load(CFG["text_ckpt"], map_location="cpu"))

    model = MultimodalEmotionModel(speech_backbone, text_backbone,
                                    n_classes, speech_dim=512, text_dim=768).to(CFG["device"])

    criterion = nn.CrossEntropyLoss()
    # Differential learning rates per component
    optimizer = optim.AdamW([
        {"params": model.speech.parameters(), "lr": CFG["lr_speech"]},
        {"params": model.text.parameters(),   "lr": CFG["lr_text"]},
        {"params": model.fusion.parameters(), "lr": CFG["lr_fusion"]},
    ], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])

    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
    best_preds, best_labels, best_reprs = None, None, None

    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, CFG["device"])
        vl_loss, vl_acc, preds, labels_list, reprs = eval_epoch(model, val_loader, criterion, CFG["device"])
        scheduler.step()

        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc);   history["val_acc"].append(vl_acc)
        print(f"Epoch {epoch:03d} | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}")

        if vl_acc > best_acc:
            best_acc = vl_acc
            best_preds, best_labels, best_reprs = preds, labels_list, reprs
            torch.save(model.state_dict(), os.path.join(CFG["save_dir"], "best_fusion_model.pt"))
            print(f"  ✓ Saved best model (val_acc={best_acc:.4f})")

    report_str = classification_report(best_labels, best_preds, target_names=class_names)
    print("\nClassification Report:\n", report_str)
    with open(os.path.join(CFG["results_dir"], "fusion_report.txt"), "w") as f:
        f.write(report_str)

    plots_dir = os.path.join(CFG["results_dir"], "plots")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_history(history, os.path.join(plots_dir, "fusion_history.png"))
    _plot_confusion(best_labels, best_preds, class_names, os.path.join(plots_dir, "fusion_confusion.png"))
    _plot_tsne(best_reprs, best_labels, class_names, os.path.join(plots_dir, "fusion_tsne.png"),
               title="t-SNE – Cross-Modal Fusion Representations")

    le_map = {cls: int(i) for i, cls in enumerate(le.classes_)}
    with open(os.path.join(CFG["save_dir"], "label_encoder.json"), "w") as f:
        json.dump(le_map, f)

    print(f"\nBest Val Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()