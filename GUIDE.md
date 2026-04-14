# 🤟 ASL Alphabet Recognition – AWS SageMaker Guide

This repository contains three Jupyter notebooks that implement a full **ASL (American Sign Language) alphabet recognition** ML pipeline on AWS SageMaker JupyterLab with S3 storage.

The source project (models, web app, CI/CD) lives at [ParanKafleUTS/New](https://github.com/ParanKafleUTS/New).

---

## Repository Contents

| File | Purpose |
|------|---------|
| `01_setup_and_data.ipynb` | Install packages, configure S3, download Kaggle dataset, split & upload |
| `02_train_models.ipynb` | Train Landmark MLP + 9 CNN models, upload weights to S3 |
| `03_evaluate_and_select_best.ipynb` | Evaluate all models, confusion matrices, bar chart, best model selection |
| `config.json` | Auto-generated shared config (bucket name, S3 URIs, class list) |

---

## Prerequisites

### 1. AWS SageMaker Studio (recommended)

1. Open **SageMaker Studio** in the AWS Console
2. Choose **JupyterLab 3** with a **`ml.g4dn.xlarge`** instance (1× NVIDIA T4 GPU)
3. Clone this repo or upload the three notebooks

> **Cost estimate:** approximately **$0.70/hour** for `ml.g4dn.xlarge`. A full training run takes ~3–4 hours → ~$2–3.

### 2. Kaggle API token

The dataset is downloaded from Kaggle automatically. You need a `kaggle.json` API token:

1. Go to https://www.kaggle.com/settings → **API** → **Create New Token**
2. Upload `kaggle.json` to `~/.kaggle/kaggle.json` inside JupyterLab
3. `chmod 600 ~/.kaggle/kaggle.json`

### 3. S3 bucket permissions

Your SageMaker execution role must have `s3:PutObject`, `s3:GetObject`, `s3:ListBucket`, and `s3:CreateBucket` on the bucket you configure.

---

## Step-by-Step Instructions

### Notebook 1 – Setup & Data (`01_setup_and_data.ipynb`)

| Cell | What it does |
|------|-------------|
| Step 1 | `pip install` TensorFlow, MediaPipe, OpenCV, scikit-learn, etc. |
| Step 2 | Verifies SageMaker role, region, and account ID |
| Step 3 | Sets `BUCKET_NAME` and `PREFIX` — **edit these** before running |
| Step 4 | Downloads ASL Alphabet dataset via Kaggle CLI (87 000 images, 29 classes) |
| Step 5 | Stratified 70/15/15 split into `train/`, `val/`, `test/` sub-folders |
| Step 6 | Uploads all splits to S3 |
| Step 7 | Verifies upload counts and writes `config.json` |

**Expected output:**
```
s3://<BUCKET>/asl-alphabet/train/   ~60 900 images
s3://<BUCKET>/asl-alphabet/val/     ~13 050 images
s3://<BUCKET>/asl-alphabet/test/    ~13 050 images
```

---

### Notebook 2 – Train Models (`02_train_models.ipynb`)

| Step | What it does |
|------|-------------|
| 1 | Loads `config.json`, checks TF + GPU |
| 2 | Downloads split data from S3 to `/tmp/asl_split/` |
| 3 | Defines shared helpers (callbacks, data generators) |
| 4 | **Approach 1** – Extracts 63 MediaPipe landmarks per image, trains a 2-hidden-layer MLP |
| 5 | **Approach 2** – Trains MobileNetV2/EfficientNetB0/ResNet50 on raw images (frozen → fine-tune) |
| 5 | **Approach 3** – Crops hands using MediaPipe bounding boxes, trains same 3 CNNs |
| 5 | **Approach 4** – Generates white-bg skeleton images, trains same 3 CNNs |
| 6 | Saves all 10 `.keras` files to S3 |

**Models produced (10 total):**
- `landmark_mlp.keras`
- `cnn_raw_mobilenetv2.keras`, `cnn_raw_efficientnetb0.keras`, `cnn_raw_resnet50.keras`
- `cnn_cropped_mobilenetv2.keras`, `cnn_cropped_efficientnetb0.keras`, `cnn_cropped_resnet50.keras`
- `cnn_skeleton_mobilenetv2.keras`, `cnn_skeleton_efficientnetb0.keras`, `cnn_skeleton_resnet50.keras`

**All models use:**
- `EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)`
- `ModelCheckpoint` (saves best epoch)
- `ReduceLROnPlateau(factor=0.5, patience=2)`

---

### Notebook 3 – Evaluate & Select Best (`03_evaluate_and_select_best.ipynb`)

| Step | What it does |
|------|-------------|
| 1–2 | Loads config + downloads all `.keras` files from S3 |
| 3 | Downloads test data from S3 (if not cached) |
| 4 | Evaluates each model: accuracy, precision, recall, F1 (macro + weighted) |
| 4 | Saves per-class classification report (`.txt`) and normalised confusion matrix (`.png`) per model |
| 5 | Builds ranked comparison DataFrame, saves CSV and Markdown table |
| 6 | Generates a grouped bar chart across all models |
| 7 | Identifies best model, saves `results_summary.json`, updates `config.json` |
| 8 | Uploads all result files to S3 |

---

## S3 File Map

```
s3://<BUCKET>/asl-alphabet/
├── train/   A/ … Z/  del/  nothing/  space/    (70 % of dataset)
├── val/     …                                   (15 %)
├── test/    …                                   (15 %)
├── models/
│   ├── landmark_mlp.keras
│   ├── cnn_raw_mobilenetv2.keras
│   ├── … (9 more .keras files)
│   └── training_results.json
├── results/
│   ├── model_comparison.csv
│   ├── model_comparison.md
│   ├── results_summary.json
│   ├── report_<model>.txt         (one per model)
│   └── plots/
│       ├── model_comparison_bar.png
│       └── cm_<model>.png         (one per model)
└── config.json
```

---

## Evaluation Metrics Explained

| Metric | Description |
|--------|-------------|
| **Accuracy** | Fraction of correctly classified samples |
| **Precision macro** | Average per-class precision (treats all 29 classes equally) |
| **Precision weighted** | Per-class precision weighted by class support |
| **Recall macro** | Average per-class recall |
| **Recall weighted** | Per-class recall weighted by support |
| **F1 macro** | Harmonic mean of macro precision and recall |
| **F1 weighted** | Harmonic mean of weighted precision and recall |

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `kaggle: command not found` | Run `pip install kaggle` and ensure `~/.kaggle/kaggle.json` exists |
| `NoSuchBucket` / `AccessDenied` | Check IAM role has S3 write access to the target bucket |
| CUDA OOM during training | Reduce `BATCH_SIZE` from 32 to 16 in Notebook 2 |
| `No hand detected` for many images | Normal — "nothing" class images have no hand; the model falls back to raw image |
| `FileNotFoundError: config.json` | Run Notebook 1 first |
| MediaPipe `Process finished with exit code 132` | Update mediapipe: `pip install --upgrade mediapipe` |
| TF GPU not found on SageMaker | Use a GPU instance type (`ml.g4dn.xlarge` or larger) |

---

## After Training – Running the Web App

The web app (`web_app/app.py` in the `New` repository) can be started with:

```bash
export MODEL_PATH=models/trained_models/best_model.keras
export MODEL_TYPE=raw      # raw | cropped | skeleton | landmark
export PYTHONPATH=$(pwd)
python web_app/app.py
```

Or with Docker:
```bash
docker compose up -d       # → http://localhost:5000
```

### Keyboard shortcuts in the browser UI

| Key | Action |
|-----|--------|
| **Enter** | Accept current word → append to sentence |
| **Backspace** | Delete last letter |
| **Space** | Insert space |
| **Escape** | Reset everything |

---

## ASL Classes (29 total)

```
A B C D E F G H I J K L M N O P Q R S T U V W X Y Z  del  nothing  space
```

- **del** – Delete signal
- **nothing** – No hand / background
- **space** – Word separator
