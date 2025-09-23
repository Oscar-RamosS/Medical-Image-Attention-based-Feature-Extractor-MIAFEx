# -*- coding: utf-8 -*-
"""
Train MIAFEx (ViT backbone + element-wise refinement) for classification.

Outputs (under output_dir):
  - miafex_checkpoint.pth               (vit_state_dict, fc_state_dict, refinement_weights, num_classes)
  - class_to_idx.json                   (ImageFolder class mapping)
  - metrics_curve.pkl                   (training loss & accuracy arrays)
  - loss_and_combined_metrics_curve.png (saved figure)
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TF logs
os.environ["TRANSFORMERS_NO_TF"] = "1"     # make transformers ignore TensorFlow
os.environ["TRANSFORMERS_NO_FLAX"] = "1"

from transformers import ViTForImageClassification
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe; saves figs without blocking
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json, pickle

# ----------------------- user-configurable -----------------------
output_dir   = r'./outputs/your_dataset/'
train_root   = r'./data/your_dataset/train'
num_classes  = 4
num_epochs   = 10
batch_size   = 16
learn_rate   = 1e-5
# -----------------------------------------------------------------

class MIAFEx(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224-in21k',
            output_hidden_states=True     # we will read the [CLS] from hidden states
        )
        self.fc = nn.Linear(768, num_classes)
        self.refinement_weights = nn.Parameter(torch.randn(768))  # learnable per-dim scale

    def forward(self, x):
        vit_outputs = self.vit(x)
        if vit_outputs.hidden_states is None:
            raise ValueError("Hidden states not returned. Ensure output_hidden_states=True.")
        cls_features = vit_outputs.hidden_states[-1][:, 0, :]     # [B, 768]
        refined = cls_features * self.refinement_weights.view(1, -1)
        logits = self.fc(refined)
        return logits, refined

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Ensure output directory exists before *any* save
os.makedirs(output_dir, exist_ok=True)

# Build dataset/loader
train_dataset = datasets.ImageFolder(root=train_root, transform=transform)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# If dataset classes disagree with num_classes, prefer the dataset value
inferred_classes = len(train_dataset.classes)
if inferred_classes != num_classes:
    print(f"[train] Warning: num_classes={num_classes} but dataset has {inferred_classes}. Using {inferred_classes}.")
    num_classes = inferred_classes

# Save class mapping for downstream use
with open(os.path.join(output_dir, "class_to_idx.json"), "w", encoding="utf-8") as f:
    json.dump(train_dataset.class_to_idx, f, indent=2, ensure_ascii=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MIAFEx(num_classes=num_classes).to(device)
print(f"[train] device={device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learn_rate)

loss_curve = []
acc_curve  = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    all_pred, all_lbl = [], []

    pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits, refined = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        all_pred.append(logits.argmax(1).detach().cpu().numpy())
        all_lbl.append(labels.detach().cpu().numpy())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = epoch_loss / len(train_loader)
    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_lbl)
    acc = accuracy_score(y_true, y_pred)

    loss_curve.append(avg_loss)
    acc_curve.append(acc)

    print(f"[train] epoch={epoch+1}  loss={avg_loss:.4f}  acc={acc:.4f}")

# Save metrics arrays
with open(os.path.join(output_dir, 'metrics_curve.pkl'), 'wb') as f:
    pickle.dump({'loss': loss_curve, 'accuracy': acc_curve}, f)

# Save a compact plot (no GUI)
plt.figure(figsize=(8, 5))
plt.plot(loss_curve, label='Training Loss')
plt.plot(acc_curve,  label='Training Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Value'); plt.title('MIAFEx — Loss/Accuracy')
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'loss_and_combined_metrics_curve.png'), dpi=160)
plt.close()

# Save a checkpoint that the extractor can read
ckpt_path = os.path.join(output_dir, 'miafex_checkpoint.pth')
torch.save({
    'vit_state_dict': model.vit.state_dict(),
    'fc_state_dict':  model.fc.state_dict(),
    'refinement_weights': model.refinement_weights.detach().cpu(),
    'num_classes': int(num_classes),
}, ckpt_path)
print(f"[train] checkpoint saved: {ckpt_path}")
