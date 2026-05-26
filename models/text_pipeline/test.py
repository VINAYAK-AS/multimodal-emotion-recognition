"""
Text-Only Pipeline — Test / Inference
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.dirname(__file__))
from train import (BertEmotionModel, TextDataset, build_dataframe,
                   preprocess_text, plot_confusion, plot_tsne, CFG)

SEED = 42

def load_model(ckpt_path, tokenizer_dir, n_classes, device):
    model = BertEmotionModel(CFG["bert_model"], n_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    tok = BertTokenizer.from_pretrained(tokenizer_dir)
    return model, tok

@torch.no_grad()
def run_inference(model, loader, device):
    all_preds, all_labels, all_reprs = [], [], []
    for batch in loader:
        ids    = batch["input_ids"].to(device)
        mask   = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits, reprs = model(ids, mask, return_repr=True)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_reprs.append(reprs.cpu().numpy())
    return np.array(all_preds), np.array(all_labels), np.concatenate(all_reprs)

def infer_single(text, model, tokenizer, le, device):
    text = preprocess_text(text)
    enc  = tokenizer(text, max_length=CFG["max_len"], padding="max_length",
                     truncation=True, return_tensors="pt")
    ids  = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    logits, _ = model(ids, mask, return_repr=True)
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_idx   = int(probs.argmax())
    return le.inverse_transform([pred_idx])[0], probs[pred_idx], dict(zip(le.classes_, probs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   default=CFG["data_root"])
    parser.add_argument("--save_dir",    default=CFG["save_dir"])
    parser.add_argument("--results_dir", default=CFG["results_dir"])
    parser.add_argument("--text",        default=None, help="Single phrase to classify")
    args = parser.parse_args()

    device = CFG["device"]

    with open(os.path.join(args.save_dir, "label_encoder.json")) as f:
        le_map = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(sorted(le_map, key=le_map.get))
    class_names  = list(le.classes_)
    n_classes    = len(class_names)

    model, tokenizer = load_model(
        os.path.join(args.save_dir, "best_text_model.pt"),
        args.save_dir, n_classes, device
    )

    if args.text:
        pred, conf, dist = infer_single(args.text, model, tokenizer, le, device)
        print(f"Text   : {args.text}")
        print(f"Emotion: {pred}  ({conf:.3f})")
        print("Dist   :", {k: f"{v:.3f}" for k, v in dist.items()})
        return

    from sklearn.model_selection import train_test_split
    df = build_dataframe(args.data_root)
    df["label"] = le.transform(df["emotion"])
    _, test_df  = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)

    test_ds     = TextDataset(test_df, le, tokenizer, CFG["max_len"])
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    preds, labels, reprs = run_inference(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {acc:.4f}")
    report = classification_report(labels, preds, target_names=class_names)
    print(report)

    os.makedirs(os.path.join(args.results_dir, "plots"), exist_ok=True)
    with open(os.path.join(args.results_dir, "text_test_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {acc:.4f}\n\n{report}")

    plot_confusion(labels, preds, class_names,
                   os.path.join(args.results_dir, "plots", "text_test_confusion.png"))
    plot_tsne(reprs, labels, class_names,
              os.path.join(args.results_dir, "plots", "text_test_tsne.png"),
              title="t-SNE – BERT [CLS] (Test)")

    per_class = {}
    for i, name in enumerate(class_names):
        mask = labels == i
        per_class[name] = accuracy_score(labels[mask], preds[mask]) if mask.sum() > 0 else 0.0
    per_df = pd.DataFrame.from_dict(per_class, orient="index", columns=["accuracy"])
    per_df.index.name = "emotion"
    per_df.to_csv(os.path.join(args.results_dir, "text_per_class_accuracy.csv"))
    print("\nPer-class accuracy:\n", per_df.to_string())

if __name__ == "__main__":
    main()
