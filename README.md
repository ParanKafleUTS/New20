# 🤟 ASL Alphabet Recognition – AWS SageMaker Notebooks

AWS SageMaker JupyterLab notebooks for the **American Sign Language (ASL) Alphabet Recognition** project.

Source project: [ParanKafleUTS/New](https://github.com/ParanKafleUTS/New)

---

## What This Repository Contains

Three self-contained Jupyter notebooks that implement a complete ML pipeline for recognising the **29-class ASL alphabet** (A–Z + del, nothing, space) on AWS SageMaker with S3 storage.

| Notebook | Purpose |
|----------|---------|
| `01_setup_and_data.ipynb` | Install packages · configure S3 · download Kaggle dataset · stratified split · upload |
| `02_train_models.ipynb` | Train Landmark MLP + 9 CNN models · upload weights to S3 |
| `03_evaluate_and_select_best.ipynb` | Evaluate all models · confusion matrices · bar chart · select best |

---

## Models Trained

| # | Approach | Architectures |
|---|----------|--------------|
| 1 | Hand Landmark MLP (63 MediaPipe features) | Custom MLP (512 → 256 → 29) |
| 2 | CNN on Raw Images | MobileNetV2, EfficientNetB0, ResNet50 |
| 3 | CNN on Cropped Hand Images | MobileNetV2, EfficientNetB0, ResNet50 |
| 4 | CNN on Hand Skeleton Images | MobileNetV2, EfficientNetB0, ResNet50 |

All CNNs use **ImageNet** pre-trained weights, a frozen-backbone phase, then fine-tuning of the top 20 layers.  
All pipelines use `EarlyStopping(patience=5)`, `ModelCheckpoint`, and `ReduceLROnPlateau`.

---

## Quick Start

### 1. Open SageMaker JupyterLab

Recommended instance: **`ml.g4dn.xlarge`** (1× NVIDIA T4 GPU, ~$0.70/hr)

### 2. Run notebooks in order

```
01_setup_and_data.ipynb   ← configure BUCKET_NAME first
02_train_models.ipynb
03_evaluate_and_select_best.ipynb
```

### 3. Required: Kaggle API token

Download the [Kaggle ASL Alphabet dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (87 000 images).  
Upload `~/.kaggle/kaggle.json` before running Notebook 1.

---

## Dataset

**Kaggle ASL Alphabet Dataset** by GrassKnoted  
→ https://www.kaggle.com/datasets/grassknoted/asl-alphabet

- 87 000 images across 29 classes (A–Z + del, nothing, space)
- 200×200 px RGB images, ~3 000 images per class
- Split: 70% train / 15% val / 15% test (stratified)

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| ML Framework | TensorFlow 2.13 / Keras |
| Landmark Detection | MediaPipe Hands |
| Data | NumPy, Pandas, Pillow, OpenCV |
| Evaluation | scikit-learn, Matplotlib, Seaborn |
| Storage | AWS S3 |
| Compute | AWS SageMaker JupyterLab |

---

## Full Documentation

See **[GUIDE.md](GUIDE.md)** for:
- Cell-by-cell instructions
- S3 file map
- Metric explanations
- Common errors & fixes
- Web app deployment steps

---

## License

Educational and research use.
