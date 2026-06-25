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

## MediaPipe Version Compatibility

### Available PyPI versions (as of 2026-04)

```
0.10.13  0.10.14  0.10.15  0.10.18  0.10.20  0.10.21
0.10.30  0.10.31  0.10.32  0.10.33
```

> `mediapipe==0.10.9` **does not exist** — you will see  
> `ERROR: No matching distribution found for mediapipe==0.10.9`.

### How the compatibility layer works (`mediapipe_utils.py`)

The helper module auto-detects which API is available and falls back gracefully:

| Mode | MediaPipe version | How it works |
|------|-------------------|-------------|
| **Tasks API** | 0.10.30 – 0.10.33 | Downloads `hand_landmarker.task` (~8 MB) on first use |
| **Solutions API** | older builds with `mp.solutions` | Uses `mp.solutions.hands.Hands()` directly |
| **Fallback** | MediaPipe broken / unavailable | Landmark MLP (Approach 1) is skipped; all 9 CNN models still run |

### NumPy version conflict

MediaPipe binary wheels are compiled against NumPy 1.x.  
Installing NumPy ≥ 2.0 causes ABI errors (`ValueError: module compiled against ABI version 0x…`).  
Notebook 1 pins `numpy<2.0` to prevent this.

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `No matching distribution found for mediapipe==0.10.9` | Use `mediapipe==0.10.33` — version 0.10.9 was never published |
| `ImportError: cannot import 'solutions'` | You have mediapipe 0.10.30+; the Solutions API was removed. `mediapipe_utils.py` switches to the Tasks API automatically |
| `ValueError: module compiled against ABI version` | NumPy ≥ 2.0 is installed. Run `pip install "numpy<2.0"` and restart the kernel |
| `mediapipe_utils.py not found` | Ensure `mediapipe_utils.py` is in the same folder as the notebooks |
| `Could not download hand-landmarker model` | The SageMaker instance may not have outbound internet access. The Tasks API model (~8 MB) must be downloadable from `storage.googleapis.com`. Check VPC/NAT gateway settings |
| `kaggle: command not found` | Run `pip install kaggle` and ensure `~/.kaggle/kaggle.json` exists |
| `NoSuchBucket` / `AccessDenied` | Check IAM role has S3 write access to the target bucket |
| CUDA OOM during training | Reduce `BATCH_SIZE` from 32 to 16 in Notebook 2 |
| `No hand detected` for many images | Normal — "nothing" class images have no hand; cropping falls back to the raw image |
| `FileNotFoundError: config.json` | Run Notebook 1 first |
| TF GPU not found on SageMaker | Use a GPU instance type (`ml.g4dn.xlarge` or larger) |

### Debugging MediaPipe manually

Run this in any notebook cell to see exactly which mode is active:

```python
import sys
sys.path.insert(0, ".")       # ensure mediapipe_utils.py is found
from mediapipe_utils import print_mediapipe_info, get_mediapipe_mode
print_mediapipe_info()
```

Expected output (Tasks API):

```
============================================================
MediaPipe Environment Info
============================================================
  Version       : 0.10.33
  Mode          : tasks
  Model file    : /tmp/mediapipe_models/hand_landmarker.task
============================================================
```

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
