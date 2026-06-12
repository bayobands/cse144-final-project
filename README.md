# CSE 144 Final Project — Transfer Learning Challenge

**Team:** Bayo Bandele (CruzID: 2141210), Hegen Chang (CruzID: 2164517)
**Course:** CSE 144, Spring 2026, UC Santa Cruz

## Overview

This project builds a 100-class image classifier using transfer learning. The dataset has only ~10 training images per class, so we fine-tuned a pretrained **EfficientNet-B2** (ImageNet weights) instead of training from scratch.

- **Validation Accuracy:** ~60%
- **Kaggle Public Score:** 0.60000 (Rank 30)

## Files

- `notebook.ipynb` — full training and inference pipeline
- `CSE144_Final_Report.pdf` — written project report
- `leaderboard.png` — screenshot of our Kaggle leaderboard position
- `submission.csv` — final Kaggle submission

## Model Weights

Trained model weights (`best_model.pth`) are available here:
[Download best_model.pth](https://drive.google.com/file/d/1zfCrme2KKJTZR8VzYS2gq3PWeFeR2icH/view?usp=sharing)

## Reproducibility

- Random seed: `42` (used for `random`, `numpy`, `torch`, `cuda`)
- Train/val split: 80/20, seed 42
- Environment: Google Colab, T4 GPU, Python 3.10, PyTorch + TorchVision

## Kaggle Leaderboard

![Leaderboard](<img width="1290" height="499" alt="image" src="https://github.com/user-attachments/assets/7993ef7e-c869-4bc2-bf24-021cc7705685" />
)


## Key Challenges

- **Label mapping bug:** Python sorts folders alphabetically by default, so folder `'10'` mapped to label `2` instead of `10`. Fixed with `sorted(os.listdir(...), key=lambda x: int(x))`.
- **Row count mismatch:** `sample_submission.csv` had 1,000 rows but Kaggle expected 1,036. Fixed by generating predictions for every image in `data/test/` directly.

See the full report (`CSE144_Final_Report.pdf`) for details.
