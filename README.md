# Comparative Evaluation of Deep Learning and Machine Learning Models for SWI-Based Landslide Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

This repository contains the complete code and reproducible pipeline for the paper:

> **Comparative Evaluation of Deep Learning and Machine Learning Models for Soil Water Index-based Rainfall-induced Landslide Prediction in Northeast India**

The study compares four predictive models — **LSTM**, **GRU**, **Random Forest**, and **XGBoost** — for binary landslide prediction using physics-informed features derived from a three-layer Sugawara tank model (Soil Water Index). Evaluation spans four automated weather stations in Arunachal Pradesh, India, with controlled ablation study, bootstrap confidence intervals, and independent temporal validation.

### Key Findings

- **Random Forest** is the most consistent model (F1: 0.667–1.000 across all stations)
- Deep learning models show **bimodal behaviour** — excellent or catastrophic — on small temporal datasets
- Physics-informed SWI features improve prediction (ΔAUC = +0.167; ΔF1 = +0.646)
- Class balance drives detection success more than dataset size
- Zero false positives in 15 of 16 model–station configurations

## Repository Structure

```
├── code/
│   ├── 01_SWI_Calculations.py              # Three-layer tank model → SWI computation
│   ├── 02_Cumulative_Rainfall.py            # Cumulative rainfall derivation
│   ├── 03_Landslide_Binary_Labelling.py     # SWI threshold → binary labels
│   ├── 04_Model_Comparison_All_Stations.py  # LSTM vs GRU vs RF vs XGBoost (4 stations)
│   ├── 05_Ablation_Study.py                 # 3 feature sets × 2 DL models × 4 stations
│   ├── 06_Bootstrap_Confidence_Intervals.py # 1,000 resamples, 95% CI
│   └── 07_Saved_Model_Prediction.py         # Load saved model for prediction
├── data/
│   └── sample/                              # Sample data format (see Data Availability)
├── figures/                                 # Generated figures (600 DPI)
├── models/                                  # Saved model files (.h5, .joblib)
├── requirements.txt                         # Python dependencies
├── LICENSE                                  # MIT License
└── README.md                                # This file
```

## Pipeline

The analytical pipeline follows seven sequential stages:

```
Raw Hourly Rainfall (CSV)
    │
    ▼
01_SWI_Calculations.py ─────────── Three-layer tank model → H1, H2, H3, SWI
    │
    ▼
02_Cumulative_Rainfall.py ──────── Cumulative rainfall column
    │
    ▼
03_Landslide_Binary_Labelling.py ─ SWI > threshold → Class 1 (landslide-prone)
    │
    ▼
04_Model_Comparison_All_Stations.py ── LSTM, GRU, RF, XGBoost (identical pipeline)
    │
    ├──▶ 05_Ablation_Study.py ─────── 24 configurations (3 feature sets × 2 models × 4 stations)
    │
    ├──▶ 06_Bootstrap_Confidence_Intervals.py ── 1,000 resamples, 95% CI
    │
    └──▶ 07_Saved_Model_Prediction.py ── Load and predict on new data
```

## Installation

```bash
git clone https://github.com/sagarbd33/SWI-Landslide-Prediction.git
cd SWI-Landslide-Prediction
pip install -r requirements.txt
```

## Requirements

- Python ≥ 3.8
- TensorFlow ≥ 2.10
- scikit-learn ≥ 1.3
- XGBoost ≥ 1.7
- pandas, numpy, matplotlib

## Usage

### 1. SWI Computation
```python
python code/01_SWI_Calculations.py
```
Computes SWI from raw hourly rainfall using the Sugawara tank model with parameters:
- L₁₁=15, L₁₂=60, L₂=15, L₃=15 mm
- α₁₁=0.10, α₁₂=0.15, α₂=0.05, α₃=0.01
- β₁=0.12, β₂=0.05, β₃=0.01

### 2. Model Comparison
```python
python code/04_Model_Comparison_All_Stations.py
```
Trains and evaluates all 4 models across all stations with identical pipeline:
- Sequence length: 10 hours
- Train/test: 80/20 stratified split
- DL: Adam (lr=0.0001), BCE loss, 100 epochs, batch=16
- RF: 100 trees, max_depth=10, balanced class weights
- XGBoost: 100 estimators, max_depth=6, scale_pos_weight

### 3. Ablation Study
```python
python code/05_Ablation_Study.py
```

### 4. Bootstrap CI
```python
python code/06_Bootstrap_Confidence_Intervals.py
```

## Model Architectures

| Model | Architecture | Parameters |
|-------|-------------|-----------|
| LSTM | LSTM(32)→Drop(0.3)→LSTM(16)→Drop(0.3)→Dense(8,ReLU)→Dense(1,Sigmoid) | 8,273 |
| GRU | GRU(32)→Drop(0.3)→GRU(16)→Drop(0.3)→Dense(8,ReLU)→Dense(1,Sigmoid) | 6,385 |
| RF | 100 trees, max_depth=10, min_samples_leaf=2, balanced weights | Non-parametric |
| XGBoost | 100 estimators, max_depth=6, lr=0.1, L1/L2 regularisation | Non-parametric |

## Stations

| Station | District | Coordinates | Hours | Positive % | Threshold |
|---------|----------|-------------|-------|------------|-----------|
| Changlang | Changlang | 27.17°N, 95.78°E | 48 | 22.9% | 80 mm |
| Tai | Lepa Rada | 27.98°N, 94.22°E | 145 | 21.4% | 500 mm |
| Danglat | Lohit | 27.93°N, 96.17°E | 120 | 55.8% | 100 mm |
| Manga | Upper Subansiri | 27.98°N, 94.22°E | 143 | 5.6% | 100 mm |

## Results Summary

| Station | Best Model | F1 | AUC | Zero FP? |
|---------|-----------|-----|-----|----------|
| Changlang | All tied | 0.667 | 1.000 | ✓ |
| Tai | GRU/RF/XGB | 0.909 | 1.000 | ✓ |
| Danglat | RF/XGBoost | 1.000 | 1.000 | ✓ |
| Manga | GRU/RF | 1.000 | 1.000 | ✓ |

## Data Availability

The hourly rainfall data were obtained from the Water Resources Department (WRD), Government of Arunachal Pradesh, and are subject to access restrictions. The landslide inventory was provided by the North Eastern Space Applications Centre (NESAC). Data can be made available from the corresponding author upon reasonable request with approval from the respective agencies.

## Authors

- Sagar Debbarma
- Mawilallawm
- Yiangyam Richard
- Toko Jersum
- Arnab Bandyopadhyay (corresponding author)
- Aditi Bhadra

Department of Agricultural Engineering, North Eastern Regional Institute of Science and Technology (NERIST), Nirjuli, Arunachal Pradesh, India

## Citation

If you use this code in your research, please cite:

```bibtex
@article{debbarma2026comparative,
  title={Comparative Evaluation of Deep Learning and Machine Learning Models for Soil Water Index-based Rainfall-induced Landslide Prediction in Northeast India},
  author={Debbarma, Sagar and {Mawilallawm} and Richard, Yiangyam and Jersum, Toko and Bandyopadhyay, Arnab and Bhadra, Aditi},
  journal={Under review},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

For questions or data access requests, please contact sagarbd33@gmail.com.
