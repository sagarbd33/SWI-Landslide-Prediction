# -*- coding: utf-8 -*-
"""
Bootstrap Confidence Intervals for All Models
Addresses reviewer concern: "Is 100% accuracy meaningful with N=48?"

Runs 1000 bootstrap resamples of test set predictions,
computes 95% CI for Accuracy, Precision, Recall, F1, AUC.

NOTE: Run this AFTER Script 1 (Model_Comparison). It loads the saved models
      and re-evaluates with bootstrap.

@author: AE-lab
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
import xgboost as xgb
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

STATION_FILES = {
    'Changlang': r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Changlang_SWI_results_with_landslide.csv',
    'Tai':       r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Tai_SWI_results_with_landslide.csv',
    'Danglat':   r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Danglat_SWI_results_with_landslide.csv',
    'Manga':     r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Manga_SWI_results_with_landslide.csv',
}

RESULTS_DIR = r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Bootstrap_CI_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES = ['Rain Hourly', 'H1', 'H2', 'H3', 'SWI', 'Cumulative_Rainfall']
TARGET = 'Landslide'
SEQUENCE_LENGTH = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

N_BOOTSTRAP = 1000
CI_LEVEL = 0.95

# =============================================================================
# HELPERS
# =============================================================================

def load_data(path):
    data = pd.read_csv(path)
    data['Date And Time'] = pd.to_datetime(data['Date And Time'])
    data.set_index('Date And Time', inplace=True)
    return data

def create_sequences(data, features, target, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[features].iloc[i:i+seq_len].values)
        y.append(data[target].iloc[i+seq_len])
    return np.array(X), np.array(y)

def bootstrap_ci(y_true, y_pred, y_proba, n_boot=1000, ci=0.95):
    """Compute bootstrap confidence intervals for all metrics"""
    n = len(y_true)
    alpha = (1 - ci) / 2
    metrics_boot = {m: [] for m in ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']}

    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        yprob = y_proba[idx]

        metrics_boot['Accuracy'].append(accuracy_score(yt, yp))
        metrics_boot['Precision'].append(precision_score(yt, yp, zero_division=0))
        metrics_boot['Recall'].append(recall_score(yt, yp, zero_division=0))
        metrics_boot['F1'].append(f1_score(yt, yp, zero_division=0))

        if len(np.unique(yt)) > 1:
            metrics_boot['AUC'].append(roc_auc_score(yt, yprob))
        else:
            metrics_boot['AUC'].append(np.nan)

    results = {}
    for metric, vals in metrics_boot.items():
        vals = np.array([v for v in vals if not np.isnan(v)])
        if len(vals) > 0:
            results[metric] = {
                'mean': np.mean(vals),
                'std': np.std(vals),
                'ci_lower': np.percentile(vals, alpha * 100),
                'ci_upper': np.percentile(vals, (1 - alpha) * 100),
            }
        else:
            results[metric] = {'mean': np.nan, 'std': np.nan,
                               'ci_lower': np.nan, 'ci_upper': np.nan}
    return results

# =============================================================================
# MAIN LOOP — RETRAIN + BOOTSTRAP
# =============================================================================

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam

all_ci_results = []

for station_name, file_path in STATION_FILES.items():
    print(f"\n{'='*70}")
    print(f"STATION: {station_name}")
    print(f"{'='*70}")

    data = load_data(file_path)
    scaler = MinMaxScaler()
    data[FEATURES] = scaler.fit_transform(data[FEATURES])
    X_seq, y = create_sequences(data, FEATURES, TARGET, SEQUENCE_LENGTH)

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    cw_dict = dict(enumerate(cw))
    scale_pw = sum(y_train == 0) / max(sum(y_train == 1), 1)

    models_preds = {}

    # --- LSTM ---
    print("  Training LSTM...", end=' ')
    m = Sequential([
        LSTM(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3), LSTM(16), Dropout(0.3),
        Dense(8, activation='relu'), Dense(1, activation='sigmoid')
    ])
    m.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    m.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          class_weight=cw_dict, verbose=0)
    proba = m.predict(X_test, verbose=0).flatten()
    models_preds['LSTM'] = (y_test, (proba > 0.5).astype(int), proba)
    print("done")

    # --- GRU ---
    print("  Training GRU...", end=' ')
    m = Sequential([
        GRU(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3), GRU(16), Dropout(0.3),
        Dense(8, activation='relu'), Dense(1, activation='sigmoid')
    ])
    m.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    m.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
          class_weight=cw_dict, verbose=0)
    proba = m.predict(X_test, verbose=0).flatten()
    models_preds['GRU'] = (y_test, (proba > 0.5).astype(int), proba)
    print("done")

    # --- Random Forest ---
    print("  Training RF...", end=' ')
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                 random_state=RANDOM_STATE, max_depth=10, min_samples_leaf=2)
    rf.fit(X_train_flat, y_train)
    proba = rf.predict_proba(X_test_flat)[:, 1]
    models_preds['Random Forest'] = (y_test, rf.predict(X_test_flat), proba)
    print("done")

    # --- XGBoost ---
    print("  Training XGBoost...", end=' ')
    xg = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                             scale_pos_weight=scale_pw, random_state=RANDOM_STATE,
                             eval_metric='logloss', use_label_encoder=False)
    xg.fit(X_train_flat, y_train)
    proba = xg.predict_proba(X_test_flat)[:, 1]
    models_preds['XGBoost'] = (y_test, xg.predict(X_test_flat), proba)
    print("done")

    # --- Bootstrap ---
    for model_name, (yt, yp, yprob) in models_preds.items():
        print(f"  Bootstrap {model_name} ({N_BOOTSTRAP} samples)...", end=' ')
        ci_results = bootstrap_ci(yt, yp, yprob, N_BOOTSTRAP, CI_LEVEL)

        for metric, vals in ci_results.items():
            all_ci_results.append({
                'Station': station_name,
                'Model': model_name,
                'Metric': metric,
                'Mean': round(vals['mean'], 4),
                'Std': round(vals['std'], 4),
                'CI_Lower': round(vals['ci_lower'], 4),
                'CI_Upper': round(vals['ci_upper'], 4),
                'CI_String': f"{vals['mean']:.3f} [{vals['ci_lower']:.3f}, {vals['ci_upper']:.3f}]"
            })
        print("done")

# =============================================================================
# SAVE RESULTS
# =============================================================================

ci_df = pd.DataFrame(all_ci_results)
ci_df.to_csv(os.path.join(RESULTS_DIR, 'bootstrap_confidence_intervals.csv'), index=False)

# Create a pivot table for paper — easy copy-paste
pivot = ci_df.pivot_table(
    index=['Station', 'Model'],
    columns='Metric',
    values='CI_String',
    aggfunc='first'
)
pivot = pivot[['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]
pivot.to_csv(os.path.join(RESULTS_DIR, 'bootstrap_ci_paper_table.csv'))

# =============================================================================
# PRINT SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print(f"BOOTSTRAP CONFIDENCE INTERVALS (95%, {N_BOOTSTRAP} resamples)")
print(f"{'='*70}")
print(pivot.to_string())
print(f"\n{'='*70}")
print(f"Saved to: {RESULTS_DIR}")
print(f"Files: bootstrap_confidence_intervals.csv, bootstrap_ci_paper_table.csv")
print(f"{'='*70}")
