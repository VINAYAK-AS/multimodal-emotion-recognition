"""
Fusion Pipeline — Test / Inference
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "speech_pipeline"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "text_pipeline"))
from train import SpeechEmotionModel, build_dataframe as speech_build_df, load_mel, CFG as SCFG
from train import BertEmotionModel, preprocess_text, CFG as TCFG
from train import (MultimodalEmotionModel, MultimodalDataset,
                   _plot_confusion, _plot_tsne, CFG)

SEED = 42

@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels, all_reprs = [], [], []
    for batch in loader:
        mel    = batch["mel"].to(device)
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits, reprs = model(mel, ids, mask, return_repr=True)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_reprs.append(reprs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.concatenate(all_reprs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default=CFG["data_root"])
    parser.add_argument("--save_dir",    default=CFG["save_dir"])
    parser.add_argument("--results_dir", default=CFG["results_dir"])
    args = parser.parse_args()

    device = CFG["device"]

    with open(os.path.join(args.save_dir, "label_encoder.json")) as f:
        le_map = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(sorted(le_map, key=le_map.get))
    class_names  = list(le.classes_)
    n_classes    = len(class_names)

    tokenizer = BertTokenizer.from_pretrained(CFG["text_tok_dir"])

    speech_backbone = SpeechEmotionModel(n_mels=SCFG["n_mels"], n_classes=n_classes)
    text_backbone   = BertEmotionModel(TCFG["bert_model"], n_classes)
    model = MultimodalEmotionModel(speech_backbone, text_backbone, n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, "best_fusion_model.pt"), map_location=device))
    model.eval()

    from sklearn.model_selection import train_test_split
    df = speech_build_df(args.data_root)
    df["label"] = le.transform(df["emotion"])
    _, test_df  = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)

    test_ds     = MultimodalDataset(test_df, le, tokenizer, CFG)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

    preds, labels, reprs = run_inference(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    print(f"\nFusion Test Accuracy: {acc:.4f}")
    report = classification_report(labels, preds, target_names=class_names)
    print(report)

    os.makedirs(os.path.join(args.results_dir, "plots"), exist_ok=True)
    with open(os.path.join(args.results_dir, "fusion_test_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n{report}")

    _plot_confusion(labels, preds, class_names,
                    os.path.join(args.results_dir, "plots", "fusion_test_confusion.png"))
    _plot_tsne(reprs, labels, class_names,
               os.path.join(args.results_dir, "plots", "fusion_test_tsne.png"),
               title="t-SNE – Fusion Representations (Test)")

    per_class = {name: accuracy_score(labels[labels==i], preds[labels==i])
                 for i, name in enumerate(class_names) if (labels==i).sum() > 0}
    per_df = pd.DataFrame.from_dict(per_class, orient="index", columns=["accuracy"])
    per_df.index.name = "emotion"
    per_df.to_csv(os.path.join(args.results_dir, "fusion_per_class_accuracy.csv"))
    print("\nPer-class accuracy:\n", per_df.to_string())

if __name__ == "__main__":
    main()
