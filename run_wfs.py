# run_wfs.py
# -*- coding: utf-8 -*-
"""
Wrapper Feature Selection (WFS) with Genetic Algorithm over extracted features.

- Reads features CSV (columns = features, last column or 'label' = target)
- Splits BEFORE feature selection (train/test, stratified)
- Runs GA jfs() from the WFS toolbox (FS.ga.jfs)
- Trains a KNN on selected features, evaluates on the hold-out test split
- Saves: selected indices (.npy), metrics.json, convergence curve, PCA plot,
         confusion matrix image, and classification report
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use("Agg")  # save-only; no GUI required
import matplotlib.pyplot as plt

# --- WFS method  ---
from FS.ga import jfs   # Change to from FS.pso import jfs for PSO, and so on.
# The whole list can be obtained from the FS folder



# ----------------------- configuration -----------------------
# Input CSV
INPUT_CSV = Path("./outputs/your_dataset/extracted_features_with_labels.csv")
LEGACY_CSV = Path("./outputs/your_dataset/extracted_features.csv")

# Output directory for artifacts
OUTPUT_DIR = Path("./outputs/your_dataset/wfs_ga_knn")

# KNN and GA hyperparameters (kept from your original script)
K = 3      # k for KNN
N = 50      # population size
T = 100     # max iterations
SEED = 42
# -------------------------------------------------------------


def _load_csv():
    """Load features/labels from CSV, supporting both new and legacy filenames."""
    csv_path = INPUT_CSV if INPUT_CSV.exists() else LEGACY_CSV
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Features CSV not found. Tried:\n  - {INPUT_CSV}\n  - {LEGACY_CSV}"
        )
    df = pd.read_csv(csv_path)
    if "label" in df.columns:
        X = df.drop(columns=["label"]).values
        y = df["label"].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
    return csv_path, np.asarray(X), np.asarray(y)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- load data (features..., label) ---
    csv_path, feat, label = _load_csv()
    print(f"[wfs] CSV: {csv_path.resolve()}")
    print(f"[wfs] Data: samples={len(label)}, features={feat.shape[1]}")

    # --- split BEFORE feature selection ---
    xtrain, xtest, ytrain, ytest = train_test_split(
        feat, label, test_size=0.2, stratify=label, random_state=SEED
    )
    fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

    # --- GA options (kept) ---
    opts = {'k': K, 'fold': fold, 'N': N, 'T': T}

    # --- run GA jfs (returns dict with 'sf' (idx), 'c' (curve), 'nf' (count)) ---
    fmdl = jfs(xtrain, ytrain, opts)
    if fmdl is None or 'sf' not in fmdl:
        raise RuntimeError("jfs() did not return a selection. Check FS.ga.jfs and parameters.")

    sf = np.asarray(fmdl['sf'], dtype=int)
    nf = int(fmdl.get('nf', len(sf)))
    print(f"[wfs] Selected features: {nf}")

    # --- train KNN on selected features ---
    x_train_sel = xtrain[:, sf] if nf > 0 else xtrain
    x_test_sel  = xtest[:, sf] if nf > 0 else xtest
    mdl = KNeighborsClassifier(n_neighbors=K)
    mdl.fit(x_train_sel, ytrain)

    # --- evaluate ---
    y_pred = mdl.predict(x_test_sel)
    acc = accuracy_score(ytest, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        ytest, y_pred, average='weighted', zero_division=0
    )

    print(f"[wfs] Acc={acc*100:.2f}%  P={pr*100:.2f}%  R={rc*100:.2f}%  F1={f1*100:.2f}%")

    # --- save selection & metrics ---
    np.save(OUTPUT_DIR / "selected_features_idx.npy", sf)
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump({
            "accuracy": float(acc),
            "precision_weighted": float(pr),
            "recall_weighted": float(rc),
            "f1_weighted": float(f1),
            "selected_features": int(nf),
            "k": K, "population": N, "iters": T, "seed": SEED,
            "csv": str(csv_path)
        }, f, indent=2)

    # --- PCA scatter over ALL samples using selected features ---
    if nf > 1:
        X_selected = feat[:, sf]
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_selected)

        le = LabelEncoder()
        label_numeric = le.fit_transform(label)

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=label_numeric, cmap='tab10', s=25)
        plt.title("PCA of Selected Features (GA + KNN)")
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(True)
        cbar = plt.colorbar(sc, ticks=range(len(le.classes_)))
        # Show class names on the colorbar ticks
        cbar.ax.set_yticklabels(le.classes_)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "pca_selected_features.png", dpi=160)
        plt.close()

    # --- Confusion matrix + report on test split ---
    cm = confusion_matrix(ytest, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues, ax=ax, colorbar=False)
    ax.set_title(f"Confusion Matrix — KNN(k={K})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(ytest, y_pred))

    # --- Convergence curve (if provided by jfs) ---
    if 'c' in fmdl and fmdl['c'] is not None:
        curve = np.asarray(fmdl['c']).reshape(-1)
        x_axis = np.arange(1, len(curve) + 1)
        plt.figure()
        plt.plot(x_axis, curve, marker='o')
        plt.xlabel('Iterations'); plt.ylabel('Fitness'); plt.title('GA Convergence')
        plt.grid(True); plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "wfs_convergence.png", dpi=160)
        plt.close()

    print(f"[wfs] Artifacts saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
