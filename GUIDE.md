# 🍌 Banana Ripeness Classification — Step-by-Step Guide

> **Who is this for?** Complete beginners who want to run machine learning training and evaluation on AWS JupyterLab (Amazon SageMaker Studio) — no prior AWS experience needed.

---

## 📋 Table of Contents

1. [What Does This Project Do?](#1-what-does-this-project-do)
2. [Files in This Repo](#2-files-in-this-repo)
3. [Prerequisites](#3-prerequisites)
4. [Setting Up AWS SageMaker Studio](#4-setting-up-aws-sagemaker-studio)
5. [Getting the Dataset](#5-getting-the-dataset)
6. [Running the Notebooks — Step by Step](#6-running-the-notebooks--step-by-step)
   - [Step 1: Setup & Data (`01_setup_and_data.ipynb`)](#step-1-setup--data)
   - [Step 2: Train Models (`02_train_models.ipynb`)](#step-2-train-models)
   - [Step 3: Evaluate & Select Best (`03_evaluate_and_select_best.ipynb`)](#step-3-evaluate--select-best)
7. [Understanding the Results](#7-understanding-the-results)
8. [Where Are My Models Saved?](#8-where-are-my-models-saved)
9. [Common Errors & How to Fix Them](#9-common-errors--how-to-fix-them)
10. [Cost Estimate](#10-cost-estimate)

---

## 1. What Does This Project Do?

This project trains an AI model to look at a photo of a banana and classify its ripeness into one of **6 categories**:

| Class | Meaning |
|-------|---------|
| `freshripe` | Fresh and at peak ripeness |
| `freshunripe` | Fresh but not yet ripe |
| `overripe` | Past peak ripeness |
| `ripe` | Ripe |
| `rotten` | Spoiled / rotten |
| `unripe` | Not yet ripe |

Three state-of-the-art neural network models are trained and compared:
- **MobileNetV3** — fast and lightweight
- **EfficientNetB0** — balanced accuracy and speed
- **ResNet50** — a classic, powerful architecture

The best-performing model is automatically selected and saved to your **AWS S3 bucket** for later use.

---

## 2. Files in This Repo

```
New20/
├── 01_setup_and_data.ipynb        ← Run FIRST – installs packages, uploads dataset to S3
├── 02_train_models.ipynb          ← Run SECOND – trains all 3 models, saves to S3
├── 03_evaluate_and_select_best.ipynb  ← Run THIRD – evaluates, picks best, uploads results
├── config.json                    ← Auto-generated after Notebook 1 – stores S3 settings
└── GUIDE.md                       ← This file!
```

> ⚠️ **Important:** Always run the notebooks **in order** (01 → 02 → 03). Each notebook builds on the previous one.

---

## 3. Prerequisites

Before you start, make sure you have:

- ✅ An **AWS account** (free tier works for small tests, but GPU instances are recommended for training — see [Cost Estimate](#10-cost-estimate))
- ✅ The **banana ripeness dataset** — download from [Roboflow Universe](https://universe.roboflow.com/musa-acuminata/banana-ripeness-classification) (free account required — select "Original Images" → Export → ZIP)
- ✅ Basic knowledge of how to click buttons and upload files — that's all!

---

## 4. Setting Up AWS SageMaker Studio

> **SageMaker Studio** is Amazon's JupyterLab in the cloud — you get a familiar notebook interface with AWS credentials pre-configured.

### 4.1 Open SageMaker Studio

1. Log in to [AWS Console](https://console.aws.amazon.com)
2. In the search bar at the top, type **SageMaker** and click it
3. In the left sidebar, click **Studio**
4. If you haven't created a domain yet:
   - Click **Set up for single user (Quick setup)**
   - Click **Set up** (this takes ~5 minutes)
5. Once set up, click **Open Studio**

### 4.2 Clone This Repository

Once inside SageMaker Studio:

1. Click the **Git icon** in the left sidebar (looks like a diamond with branches)
2. Click **Clone a Repository**
3. Paste this URL: `https://github.com/ParanKafleUTS/New20.git`
4. Click **Clone**

You should now see the `New20` folder in the file browser on the left.

### 4.3 Choose the Right Instance Type

> 💡 **Why does this matter?** Training deep learning models on CPU is extremely slow. A GPU reduces training time from hours to minutes.

When opening a notebook for the first time:
1. Click the notebook file (e.g., `01_setup_and_data.ipynb`)
2. If asked to select a kernel/instance:
   - For **Notebook 1** (setup): any CPU instance is fine (e.g., `ml.t3.medium`)
   - For **Notebooks 2 & 3** (training/evaluation): choose a **GPU instance** like `ml.g4dn.xlarge`
3. Select **Python 3 (ipykernel)** as the kernel

---

## 5. Getting the Dataset

1. Go to [Roboflow Universe – Banana Ripeness](https://universe.roboflow.com/musa-acuminata/banana-ripeness-classification)
2. Click **Download Dataset**
3. Sign up for a free account if prompted
4. Select:
   - Format: **Folder Structure** (Original Images)
   - Download: **zip**
5. You'll get a file like `Banana Ripeness Classification.v1-original-images.zip`
6. Upload this zip to SageMaker Studio:
   - In the file browser on the left, navigate to the `New20` folder
   - Click the **Upload** button (up-arrow icon at the top of the file browser)
   - Select the zip file from your computer

The zip file will appear in the `New20` directory. You're ready for Step 1!

---

## 6. Running the Notebooks — Step by Step

### How to Run a Notebook Cell

- Click on a cell (grey box with code)
- Press **Shift + Enter** to run it, OR click the **▶ Run** button in the toolbar
- Wait for the `[*]` symbol to change to a number (e.g., `[1]`) before running the next cell
- A `[*]` means the cell is still running — **be patient!**

---

### Step 1: Setup & Data

**File:** `01_setup_and_data.ipynb`

**What it does:** Installs all Python packages, checks your AWS setup, creates an S3 bucket, and uploads the dataset.

**How to run:**

1. Open `01_setup_and_data.ipynb` by double-clicking it in the file browser
2. Run **Cell 1** (Install Required Packages)
   - Wait for the output to say `✅ All packages installed successfully.`
   - This may take 2–3 minutes
3. Run **Cell 2** (Verify AWS Environment)
   - You should see green ✅ checkmarks for SageMaker Role, AWS Region, PyTorch version
   - If you see a GPU listed — great! If not, training will be slower
4. Run **Cell 3** (Configure S3 Bucket)
   - Leave `CUSTOM_BUCKET = None` to use the automatic bucket
   - You'll see your bucket name printed (something like `sagemaker-us-east-1-123456789`)
   - A `config.json` file is created — **don't delete it!**
5. Run **Cell 4** (Upload Dataset)
   - Make sure `LOCAL_ZIP_PATH` matches the name of your uploaded zip file
   - The default is: `"Banana Ripeness Classification.v1-original-images.zip"`
   - If your file has a different name, edit the line: `LOCAL_ZIP_PATH = "your-file-name.zip"`
   - Wait for upload to complete (may take 5–15 minutes depending on dataset size)
6. Run **Cell 5** (Verify Dataset in S3)
   - You should see ✅ for train, valid, and test splits
   - If everything is ✅, you'll see: `🎉 Dataset looks good!`

> ✅ **Notebook 1 complete!** Proceed to Notebook 2.

---

### Step 2: Train Models

**File:** `02_train_models.ipynb`

**What it does:** Downloads the dataset from S3, trains three neural networks (MobileNetV3, EfficientNetB0, ResNet50), and saves all model weights to S3.

> ⏳ **Expected time:** 12–24 minutes on GPU (`ml.g4dn.xlarge`), much longer on CPU.

**How to run:**

1. Open `02_train_models.ipynb`
2. Run **Cell 1** (Imports & Configuration)
   - Check the printed configuration matches what you expect
   - You can change `NUM_EPOCHS` (default: 10) for more/less training
   - You can change `BATCH_SIZE` (default: 16) — reduce to 8 if you get memory errors
3. Run **Cell 2** (Download Dataset from S3)
   - This downloads the dataset to `/tmp/banana_dataset` on the instance
   - First run takes a few minutes; subsequent runs use the cache
4. Run **Cell 3** (Data Loaders)
   - You'll see the number of images per split and class names
5. Run **Cell 4** (Model Builder)
   - Just prints `✅ Model builder defined.` — nothing to configure
6. Run **Cell 5** (Training Loop)
   - Just defines the training function — nothing to configure
7. Run **Cell 6** (Train All Models) ← **This is the long one!**
   - Training starts for MobileNetV3, then EfficientNetB0, then ResNet50
   - For each model, you'll see progress bars and loss/accuracy per epoch
   - At the end you'll see a summary table with the best validation accuracy for each model
8. Run **Cell 7** (Upload Models to S3)
   - All `.pth` model files are uploaded to `s3://your-bucket/banana-ripeness-models/`
   - `config.json` is updated with the S3 URIs
9. Run **Cell 8** (Training Curves) ← Optional
   - Plots loss and accuracy graphs — good for checking if models are overfitting

> ✅ **Notebook 2 complete!** Proceed to Notebook 3.

---

### Step 3: Evaluate & Select Best

**File:** `03_evaluate_and_select_best.ipynb`

**What it does:** Loads all models from S3, runs them on the test set, generates confusion matrices and classification reports, selects the winner, and uploads all results to S3.

**How to run:**

1. Open `03_evaluate_and_select_best.ipynb`
2. Run **Cell 1** (Imports & Configuration)
   - Loads config from `config.json` — should show the S3 URIs from training
3. Run **Cell 2** (Load Model Weights from S3)
   - Downloads the `.pth` files from S3 and loads them into memory
4. Run **Cell 3** (Prepare Test DataLoader)
   - Loads the test images from the local cache
5. Run **Cell 4** (Evaluate All Models)
   - Runs inference on all test images — takes 1–3 minutes per model
6. Run **Cell 5** (Classification Reports)
   - Prints precision, recall, F1-score for each class and model
7. Run **Cell 6** (Confusion Matrices)
   - Displays side-by-side confusion matrix heatmaps
   - Darker diagonal = better model
8. Run **Cell 7** (Select Best Model)
   - Prints a ranked leaderboard — 🥇🥈🥉
9. Run **Cell 8** (Upload Best Model to S3)
   - The best model gets uploaded to `s3://your-bucket/banana-ripeness-models/best_model/`
10. Run **Cell 9** (Save Results Summary)
    - Saves a `results_summary.json` locally and to S3
    - Contains all accuracy numbers in one place
11. Run **Cell 10** (Per-Class Accuracy Bar Chart)
    - Visual comparison of how well each model does on each banana class

> 🎉 **All done!** Check the final output for the best model name and its S3 location.

---

## 7. Understanding the Results

### Accuracy
- **Test Accuracy** = percentage of test images correctly classified
- Example: `0.9698` = 96.98% correct
- Higher is better; above 0.90 is excellent for this task

### Precision, Recall, F1-Score (from classification report)
- **Precision** — when the model says "ripe", how often is it right?
- **Recall** — out of all truly "ripe" bananas, how many did the model find?
- **F1-score** — a balanced combination of precision and recall (1.0 = perfect)

### Confusion Matrix
- Each row = the **true** class of the banana
- Each column = what the model **predicted**
- The diagonal (top-left to bottom-right) = correct predictions
- Off-diagonal cells = mistakes
- Darker blue on the diagonal = better model

---

## 8. Where Are My Models Saved?

All files are saved to your S3 bucket. The bucket name is printed in Notebook 1 and stored in `config.json`.

```
s3://your-bucket/
├── banana-ripeness-dataset-original/     ← Your uploaded dataset
│   ├── train/
│   ├── valid/
│   └── test/
└── banana-ripeness-models/               ← Trained model weights
    ├── MobileNetV3_banana_ripeness.pth
    ├── EfficientNetB0_banana_ripeness.pth
    ├── ResNet50_banana_ripeness.pth
    ├── best_model/
    │   └── <BestModelName>_best.pth      ← 🏆 THE WINNER
    └── eval_artifacts/
        ├── confusion_matrices.png
        ├── per_class_accuracy.png
        └── results_summary.json
```

### How to download files from S3

1. Go to [AWS S3 Console](https://s3.console.aws.amazon.com)
2. Find your bucket (same name as in `config.json`)
3. Navigate to the folder you want
4. Click the file name → **Download**

---

## 9. Common Errors & How to Fix Them

### ❌ `FileNotFoundError: 'config.json' not found`
**Cause:** You skipped Notebook 1 or it didn't finish.
**Fix:** Go back and run all cells in `01_setup_and_data.ipynb`.

### ❌ `CUDA out of memory`
**Cause:** The GPU doesn't have enough memory for the current batch size.
**Fix:** In Notebook 2, find the line `BATCH_SIZE = 16` and change it to `BATCH_SIZE = 8` (or even `4`), then re-run from Cell 1.

### ❌ `Could not find 'train' folder under /tmp/banana_dataset`
**Cause:** The dataset wasn't downloaded, or the zip had a different internal structure.
**Fix:**
1. Check the zip structure — it should have `train/`, `valid/`, `test/` folders
2. If structured differently, update `LOCAL_DATA_DIR` in Notebook 2 to point to the folder that contains `train/valid/test`

### ❌ `NoCredentialsError` or `Unable to locate credentials`
**Cause:** AWS credentials are not configured.
**Fix:** You must run this on SageMaker Studio, which automatically provides credentials. If running locally, run `aws configure` in a terminal.

### ❌ Notebook 1 zip upload shows `⚠️ Zip file not found`
**Cause:** The zip filename in the notebook doesn't match the actual filename.
**Fix:** Update `LOCAL_ZIP_PATH` in Cell 4 to exactly match your file's name (check spelling and spaces).

### ❌ Training is extremely slow
**Cause:** Running on CPU instead of GPU.
**Fix:** Change your instance type to `ml.g4dn.xlarge` (GPU). In SageMaker Studio: click the instance type indicator at the top-right of the notebook → change instance.

### ❌ `[*]` stuck for a very long time
**Cause:** Normal during training (Cell 6 of Notebook 2 takes 12–24 minutes).
**Fix:** Wait! Check the progress bars — they should be moving. If completely frozen for >30 minutes, try restarting the kernel (Kernel menu → Restart Kernel) and re-running.

---

## 10. Cost Estimate

> ⚠️ AWS charges by the hour for compute and by GB for storage. Estimated costs below are approximate (as of 2025).

| Resource | Type | Approx Cost |
|----------|------|-------------|
| SageMaker Studio (Notebook 1) | `ml.t3.medium` ~5 min | ~$0.01 |
| SageMaker Studio (Notebooks 2 & 3) | `ml.g4dn.xlarge` ~40 min | ~$0.40 |
| S3 Storage (dataset + models) | ~2 GB | ~$0.05/month |
| **Total for one full run** | | **~$0.50** |

> 💡 **Tip:** Stop your SageMaker instance when not in use to avoid ongoing charges. In Studio, click the running instances icon (power plug) → stop unused instances.

---

## 📞 Quick Reference

| Task | File to run |
|------|-------------|
| First time setup | `01_setup_and_data.ipynb` |
| Train models | `02_train_models.ipynb` |
| Evaluate and find best model | `03_evaluate_and_select_best.ipynb` |
| Check where things are saved | `config.json` |

**Order to run:** `01` → `02` → `03`

---

*Happy banana classifying! 🍌*
