"""
Text-Only Emotion Recognition Pipeline
Architecture:
  Preprocessing      -> lowercase, strip punctuation, WordPiece tokenizer (BERT)
  Feature Extraction -> BERT token embeddings  (tokens × 768)
  Contextual Modelling -> BERT Transformer encoder (fine-tuned)
  Classifier         -> [CLS] token → FC → Softmax
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

# ─── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ─── Config ─────────────────────────────────────────────────────────────────
CFG = dict(
    data_root    = "/content/TESS",
    bert_model   = "bert-base-uncased",
    max_len      = 32,            # TESS phrases are short (max ~4 words)
    batch_size   = 32,
    epochs       = 20,
    lr           = 2e-5,          # low LR for fine-tuning BERT
    weight_decay = 1e-2,
    unfreeze_top_layers = 4,      # unfreeze top N BERT encoder layers
    device       = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir     = "/content/drive/MyDrive/emotion_recognition/models/text_pipeline",
    results_dir  = "/content/drive/MyDrive/emotion_recognition/Results",
)
os.makedirs(CFG["save_dir"],    exist_ok=True)
os.makedirs(CFG["results_dir"], exist_ok=True)
print(f"Using device: {CFG['device']}")

# ─── 1. Build dataframe ──────────────────────────────────────────────────────
def build_dataframe(data_root):
    records = []
    for folder in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder)
        if not os.path.isdir(folder_path): continue
        for fname in os.listdir(folder_path):
            if not fname.endswith(".wav"): continue
            emotion = fname.split("_")[-1].replace(".wav", "").lower()
            if emotion == "ps": emotion = "surprise"
            word = "_".join(fname.split("_")[:-1]).replace("_", " ")
            records.append({"text": word, "emotion": emotion})
    df = pd.DataFrame(records).drop_duplicates(subset=["text","emotion"])
    print(f"Unique text samples: {len(df)} | Emotions: {sorted(df.emotion.unique())}")
    return df

# ─── 2. Preprocessing: clean text ───────────────────────────────────────────
import re
def preprocess_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

# ─── 3. Dataset ──────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, df, label_encoder, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.le = label_encoder
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = preprocess_text(row["text"])
        label = self.le.transform([row["emotion"]])[0]
        enc   = self.tok(
            text,
            max_length      = self.max_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(label, dtype=torch.long),
        }

# ─── 4. Model: BERT → Classifier ─────────────────────────────────────────────
class BertEmotionModel(nn.Module):
    """
    Feature Extraction + Contextual Modelling: BERT encoder (fine-tuned top layers)
    Classifier: linear head on [CLS] token
    """
    def __init__(self, bert_name, n_classes, unfreeze_top=4):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name)

        # Freeze all BERT params first
        for p in self.bert.parameters():
            p.requires_grad = False

        # Unfreeze top encoder layers + pooler
        total_layers = len(self.bert.encoder.layer)
        for i in range(total_layers - unfreeze_top, total_layers):
            for p in self.bert.encoder.layer[i].parameters():
                p.requires_grad = True
        for p in self.bert.pooler.parameters():
            p.requires_grad = True

        hidden = self.bert.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_classes),
        )

    def forward(self, input_ids, attention_mask, return_repr=False):
        out      = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = out.pooler_output           # (B, 768) — [CLS] token representation
        logits   = self.classifier(cls_repr)
        if return_repr:
            return logits, cls_repr
        return logits

# ─── 5. Training helpers ─────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(loader, leave=False, desc="train"):
        ids   = batch["input_ids"].to(device)
        mask  = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        logits = model(ids, mask)
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
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits, reprs = model(ids, mask, return_repr=True)
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

# ─── 6. Visualisations (re-used from speech pipeline pattern) ────────────────
def plot_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="train"); axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history["train_acc"],  label="train"); axes[1].plot(history["val_acc"],  label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.suptitle("Text Pipeline – Training History")
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()

def plot_confusion(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names,
                cmap="Greens", ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Text Pipeline – Confusion Matrix")
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
    df["label"]  = le.fit_transform(df["emotion"])
    class_names  = list(le.classes_)
    n_classes    = len(class_names)
    print(f"Classes ({n_classes}): {class_names}")

    tokenizer = BertTokenizer.from_pretrained(CFG["bert_model"])

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)

    train_ds = TextDataset(train_df, le, tokenizer, CFG["max_len"])
    val_ds   = TextDataset(val_df,   le, tokenizer, CFG["max_len"])
    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=2)

    model     = BertEmotionModel(CFG["bert_model"], n_classes, CFG["unfreeze_top_layers"]).to(CFG["device"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,
                                             total_iters=CFG["epochs"])

    history  = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc = 0.0
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
            torch.save(model.state_dict(), os.path.join(CFG["save_dir"], "best_text_model.pt"))
            tokenizer.save_pretrained(CFG["save_dir"])
            print(f"  ✓ Saved best model (val_acc={best_acc:.4f})")

    report_str = classification_report(best_labels, best_preds, target_names=class_names)
    print("\nClassification Report:\n", report_str)
    with open(os.path.join(CFG["results_dir"], "text_report.txt"), "w") as f:
        f.write(report_str)

    plot_history(history, os.path.join(CFG["results_dir"], "plots", "text_history.png"))
    plot_confusion(best_labels, best_preds, class_names,
                   os.path.join(CFG["results_dir"], "plots", "text_confusion.png"))
    plot_tsne(best_reprs, best_labels, class_names,
              os.path.join(CFG["results_dir"], "plots", "text_tsne.png"),
              title="t-SNE – BERT [CLS] Representations")

    le_map = {cls: int(i) for i, cls in enumerate(le.classes_)}
    with open(os.path.join(CFG["save_dir"], "label_encoder.json"), "w") as f:
        json.dump(le_map, f)

    print(f"\nBest Val Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    main()
