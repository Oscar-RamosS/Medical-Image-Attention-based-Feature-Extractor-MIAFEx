# -*- coding: utf-8 -*-
"""
Extract refined MIAFEx descriptors from a trained checkpoint and evaluate simple classical models.

Outputs (under output_dir):
  - extracted_features_with_labels.csv
  - miafex_features.npy
  - class_to_idx.json
  - ml_eval/<model>_metrics.json
  - ml_eval/<model>_report.txt
  - ml_eval/<model>_confusion_matrix.png
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import ViTForImageClassification
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

import matplotlib
matplotlib.use("Agg")  # headless/IDE-safe; saves figures to disk
import matplotlib.pyplot as plt

# ------------------------- Configuration -------------------------
output_dir   = r'./outputs/your_dataset/'
data_dir     = r'./data/your_dataset/test'
checkpoint_path = os.path.join(output_dir, 'miafex_checkpoint.pth')  # original name kept
csv_path     = os.path.join(output_dir, 'extracted_features.csv')
features_npy = os.path.join(output_dir, 'miafex_features.npy')
batch        = 16
fallback_num_classes = 3  # used only if checkpoint doesn't store num_classes
# -----------------------------------------------------------------

# ------------------------- Model definition ----------------------
class MIAFEx(nn.Module):
    """ViT backbone + element-wise refinement; FC kept for compatibility (not used for extraction)."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            output_hidden_states=True
        )
        self.fc = nn.Linear(768, num_classes)
        self.refinement_weights = nn.Parameter(torch.randn(768))

    def forward(self, x):
        out = self.vit(x)
        if out.hidden_states is None:
            raise ValueError("Hidden states not returned. Ensure output_hidden_states=True when creating the ViT.")
        cls_features = out.hidden_states[-1][:, 0, :]              # [B, 768]
        refined = cls_features * self.refinement_weights.view(1, -1)
        return refined, cls_features
# -----------------------------------------------------------------

def safe_load_ckpt(path, device):
    """torch.load with weights_only=True when available (PyTorch >= 2.4), fallback otherwise."""
    try:
        return torch.load(path, map_location=device, weights_only=True)  # type: ignore[arg-type]
    except TypeError:
        return torch.load(path, map_location=device)

def load_backbone_and_refinement(model: MIAFEx, ckpt: dict, device):
    """
    Load ViT weights and refinement vector from various checkpoint formats:
      - ckpt['vit_state_dict'] (recommended)
      - ckpt['model_state_dict'] filtered to 'vit.*'
      - ckpt['refinement_weights'] if present
      - sets num_classes from ckpt when available (optional)
    """
    # 1) Load ViT weights
    if 'vit_state_dict' in ckpt:
        vit_sd = ckpt['vit_state_dict']
    elif 'model_state_dict' in ckpt:
        # Filter only vit.* keys from a full model state_dict
        vit_sd = {k: v for k, v in ckpt['model_state_dict'].items() if k.startswith('vit.')}
        if not vit_sd:
            raise KeyError("No 'vit.' keys found in 'model_state_dict'.")
    else:
        raise KeyError("Checkpoint must contain 'vit_state_dict' or 'model_state_dict' with 'vit.' keys.")
    missing, unexpected = model.vit.load_state_dict(vit_sd, strict=False)
    if missing:
        print(f"[extract] Warning: missing keys in ViT: {missing}")
    if unexpected:
        print(f"[extract] Warning: unexpected keys ignored: {unexpected}")

    # 2) Load refinement weights if available
    if 'refinement_weights' in ckpt:
        with torch.no_grad():
            rw = ckpt['refinement_weights'].to(device)
            if rw.numel() != model.refinement_weights.numel():
                raise ValueError(f"refinement_weights size mismatch: ckpt={rw.numel()} vs model={model.refinement_weights.numel()}")
            model.refinement_weights.copy_(rw)

    # 3) Try to return num_classes if present (optional)
    if 'num_classes' in ckpt:
        return int(ckpt['num_classes'])
    # else try to infer from saved FC weights (common in full model checkpoints)
    if 'fc_state_dict' in ckpt and 'weight' in ckpt['fc_state_dict']:
        return int(ckpt['fc_state_dict']['weight'].shape[0])
    if 'model_state_dict' in ckpt:
        w = ckpt['model_state_dict'].get('fc.weight', None)
        if w is not None:
            return int(w.shape[0])
    return None  # unknown

def main():
    # Ensure output directory exists before saving anything
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[extract] device={device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=4)

    # Save class mapping for downstream use
    with open(os.path.join(output_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
        json.dump(dataset.class_to_idx, f, indent=2, ensure_ascii=False)

    # Load checkpoint and initialize model
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {os.path.abspath(checkpoint_path)}")

    ckpt = safe_load_ckpt(checkpoint_path, device)
    # num_classes from ckpt if available, else fallback
    num_classes = load_backbone_and_refinement(
        model := MIAFEx(fallback_num_classes).to(device), ckpt, device
    ) or fallback_num_classes
    print(f"[extract] num_classes={num_classes}")

    model.eval()

    # --------- Feature extraction ---------
    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting MIAFEx features"):
            images = images.to(device)
            refined, _ = model(images)
            all_features.append(refined.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.vstack(all_features) if all_features else np.zeros((0, 768), dtype=np.float32)
    y = np.concatenate(all_labels) if all_labels else np.zeros((0,), dtype=np.int64)

    # Save CSV + NPY artifacts
    df = pd.DataFrame(X)
    df["label"] = y
    df.to_csv(csv_path, index=False)
    np.save(features_npy, X)

    print(f"[extract] Features saved:\n  CSV: {os.path.abspath(csv_path)}\n  NPY: {os.path.abspath(features_npy)}")

    # --------- Quick classical ML sanity check ---------
    if len(np.unique(y)) >= 2 and len(y) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        classifiers = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'SVM': SVC(random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42)
        }

        ml_dir = os.path.join(output_dir, "ml_eval")
        os.makedirs(ml_dir, exist_ok=True)

        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            pr, rc, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            print(f"\n{name}")
            print(f"  Accuracy:  {acc*100:.2f}%")
            print(f"  Precision: {pr*100:.2f}%  Recall: {rc*100:.2f}%  F1: {f1*100:.2f}%")
            print(f"  Confusion matrix:\n{cm}")

            # Save metrics JSON
            metrics = {
                "accuracy": float(acc),
                "precision_weighted": float(pr),
                "recall_weighted": float(rc),
                "f1_weighted": float(f1),
            }
            safe_name = name.replace(" ", "_").lower()
            with open(os.path.join(ml_dir, f"{safe_name}_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

            # Save classification report
            with open(os.path.join(ml_dir, f"{safe_name}_report.txt"), "w", encoding="utf-8") as f:
                f.write(classification_report(y_test, y_pred))

            # Save confusion matrix plot
            from sklearn.metrics import ConfusionMatrixDisplay
            fig, ax = plt.subplots(figsize=(6, 6))
            ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues", ax=ax, colorbar=False)
            ax.set_title(f"Confusion Matrix — {name}")
            plt.tight_layout()
            plt.savefig(os.path.join(ml_dir, f"{safe_name}_confusion_matrix.png"), dpi=160)
            plt.close()

        print(f"[extract] ML artifacts saved to: {os.path.abspath(ml_dir)}")
    else:
        print("[extract] Skipping quick ML: need at least 2 classes and >= 10 samples.")

if __name__ == "__main__":
    main()
