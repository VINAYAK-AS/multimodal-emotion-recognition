"""
Speech-Only Pipeline — Test / Inference
Loads the best checkpoint and evaluates on the held-out test split (or a custom folder).
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ── Import from train (same package) ─────────────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from train import (SpeechEmotionModel, SpeechDataset, build_dataframe,
                   load_mel, plot_confusion, plot_tsne, CFG)

SEED = 42

def load_model(checkpoint_path, n_classes, device):
    model = SpeechEmotionModel(n_mels=CFG["n_mels"], n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels, all_reprs, all_probs = [], [], [], []
    for mel, labels in loader:
        mel, labels = mel.to(device), labels.to(device)
        logits, reprs = model(mel, return_repr=True)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_reprs.append(reprs.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
    return (np.array(all_preds), np.array(all_labels),
            np.concatenate(all_reprs), np.concatenate(all_probs))

def infer_single_file(wav_path, model, le, device):
    """Predict emotion for a single WAV file."""
    mel  = load_mel(wav_path, CFG)
    mel  = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T,F)
    logits, _ = model(mel, return_repr=True)
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_idx   = int(probs.argmax())
    pred_label = le.inverse_transform([pred_idx])[0]
    confidence = probs[pred_idx]
    return pred_label, confidence, dict(zip(le.classes_, probs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default=CFG["data_root"])
    parser.add_argument("--save_dir",    default=CFG["save_dir"])
    parser.add_argument("--results_dir", default=CFG["results_dir"])
    parser.add_argument("--wav",         default=None, help="Optional: single WAV file to classify")
    args = parser.parse_args()

    device = CFG["device"]

    # Load label encoder
    le_path = os.path.join(args.save_dir, "label_encoder.json")
    with open(le_path) as f:
        le_map = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(sorted(le_map, key=le_map.get))
    class_names  = list(le.classes_)
    n_classes    = len(class_names)

    ckpt_path = os.path.join(args.save_dir, "best_speech_model.pt")
    model     = load_model(ckpt_path, n_classes, device)
    print(f"Loaded checkpoint: {ckpt_path}")

    # ── Single-file inference ──
    if args.wav:
        pred, conf, dist = infer_single_file(args.wav, model, le, device)
        print(f"\nFile : {args.wav}")
        print(f"Predicted emotion : {pred}  (confidence {conf:.3f})")
        print("Full distribution :", {k: f"{v:.3f}" for k, v in dist.items()})
        return

    # ── Full test-set evaluation ──
    from sklearn.model_selection import train_test_split
    df = build_dataframe(args.data_root)
    df["label"] = le.transform(df["emotion"])
    _, test_df  = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)

    test_ds     = SpeechDataset(test_df, le, CFG)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    preds, labels, reprs, probs = run_inference(model, test_loader, device)

    acc = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    report = classification_report(labels, preds, target_names=class_names)
    print(report)

    os.makedirs(os.path.join(args.results_dir, "plots"), exist_ok=True)

    with open(os.path.join(args.results_dir, "speech_test_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n{report}")

    plot_confusion(labels, preds, class_names,
                   os.path.join(args.results_dir, "plots", "speech_test_confusion.png"))
    plot_tsne(reprs, labels, class_names,
              os.path.join(args.results_dir, "plots", "speech_test_tsne.png"),
              title="t-SNE – Speech BiLSTM (Test)")

    # Per-class accuracy table
    per_class = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        per_class[name] = accuracy_score(labels[mask], preds[mask]) if mask.sum() > 0 else 0.0
    per_df = pd.DataFrame.from_dict(per_class, orient="index", columns=["accuracy"])
    per_df.index.name = "emotion"
    per_df.to_csv(os.path.join(args.results_dir, "speech_per_class_accuracy.csv"))
    print("\nPer-class accuracy:\n", per_df.to_string())

if __name__ == "__main__":
    main()
