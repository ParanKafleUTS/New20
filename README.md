# 🍌 Banana Ripeness Classification — AWS JupyterLab

This repository contains three Jupyter notebooks for training and evaluating CNN models to classify banana ripeness on **AWS SageMaker JupyterLab**, with full S3 integration and error handling.

## 👉 Getting Started

**Read [`GUIDE.md`](GUIDE.md) first — it explains everything step by step, even if you have never used AWS before.**

## Notebooks (run in order)

| # | Notebook | Purpose |
|---|----------|---------|
| 1 | `01_setup_and_data.ipynb` | Install packages, configure S3, upload dataset |
| 2 | `02_train_models.ipynb` | Train MobileNetV3, EfficientNetB0, ResNet50 |
| 3 | `03_evaluate_and_select_best.ipynb` | Evaluate, compare models, save best to S3 |

## Models

- **MobileNetV3** — fast and lightweight
- **EfficientNetB0** — balanced accuracy and speed  
- **ResNet50** — powerful classic architecture

## Dataset

Banana Ripeness Classification (6 classes: freshripe, freshunripe, overripe, ripe, rotten, unripe)  
Source: [Roboflow Universe](https://universe.roboflow.com/musa-acuminata/banana-ripeness-classification)