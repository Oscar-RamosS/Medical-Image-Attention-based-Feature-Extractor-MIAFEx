# MIAFEx: An Attention-based Feature Extraction Method for Medical Image Classification

[![Paper DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.knosys.2025.114468-blue)](https://doi.org/10.1016/j.knosys.2025.114468)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**MIAFEx** refines the Transformer [CLS] token with learnable weights to produce robust features for medical image classification, particularly effective on small/medium datasets. It ships a full pipeline: supervised training, feature extraction, wrapper-based feature selection (PSO/DE/GA), and classical classifiers (LR/SVM/RF/XGB).

![MIAFEx pipeline](./miafex_pipeline.png)

MIAFEx refines the Vision Transformer (ViT) CLS token with learnable per-dimension weights to produce robust descriptors for medical image classification.  
This repository contains two main entry-point scripts:

- **Training** → `train_miafex.py`
- **Feature Extraction** → `extract_miafex_features.py`

---

##  Features
- ViT backbone with learnable refinement weights over the CLS token
- One-file checkpoint (`miafex_checkpoint.pth`) containing ViT + refinement + FC
- Descriptor export to CSV + NPY
- Optional quick ML sanity check (LR, SVM, RF, GB, KNN)
- Generic dataset paths with ImageFolder structure

---

##  Installation

### Conda (recommended)
```bash
conda create -n miafex python=3.10 -y
conda activate miafex
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # adjust CUDA if needed
pip install numpy pandas scikit-learn xgboost matplotlib tqdm pyyaml timm transformers

./data/your_dataset/
├─ train/
│  ├─ CLASS_0/ img1.jpg, img2.jpg, ...
│  ├─ CLASS_1/ ...
│  └─ CLASS_K/ ...
└─ test/
   ├─ CLASS_0/ ...
   ├─ CLASS_1/ ...
   └─ CLASS_K/ ...
```

## MIAFEx Usage
```
python MIAFEx_descUse.py \
  --train_dir ./data/your_dataset/train \
  --output_dir ./outputs/your_dataset \
  --num_classes 5 \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-5

python MIAFEx_descGen.py \
  --data_dir ./data/your_dataset/test \
  --checkpoint ./outputs/your_dataset/miafex_checkpoint.pth \
  --output_dir ./outputs/your_dataset \
  --run_quick_ml

```
One-shot pipeline
```
You can chain training + extraction in one go:
python MIAFEx_descUse.py \
  --train_dir ./data/your_dataset/train \
  --output_dir ./outputs/your_dataset \
  --num_classes 5 --epochs 50 --batch_size 8 --lr 1e-5

python MIAFEx_descGen.py \
  --data_dir ./data/your_dataset/test \
  --checkpoint ./outputs/your_dataset/miafex_checkpoint.pth \
  --output_dir ./outputs/your_dataset \
  --run_quick_ml
```


### Tips

- Match --num_classes to your dataset

- For imbalanced datasets: add class weights to loss or sampler

- If CUDA runs OOM: reduce --batch_size

- Swap augmentations in transforms.Compose (resize, normalize, flips, etc.)


### Troubleshooting

- Hidden states missing → ensure output_hidden_states=True (already set in scripts)

- Checkpoint mismatch → make sure extractor uses checkpoint from same code version

- Slow feature extraction → increase --batch_size if GPU memory allows

  ### Citation


  If you use this code or the descriptors in your research, please cite:
```  
@article{RamosSoto2025MIAFEx,
  title   = {MIAFEx: An attention-based feature extraction method for medical image classification},
  author  = {Ramos-Soto, Oscar and Ramos-Frutos, Jorge and P{\'e}rez-Zarate, Ezequiel and Oliva, Diego and Balderas-Mata, Sandra E.},
  journal = {Knowledge-Based Systems},
  year    = {2025},
  doi     = {10.1016/j.knosys.2025.114468}
}
```

