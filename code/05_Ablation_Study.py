# -*- coding: utf-8 -*-
"""
Ablation Study: Impact of SWI Features on Model Performance
Tests 3 feature sets × 2 DL models × 4 stations

Feature Sets:
  - ALL: Rain Hourly, H1, H2, H3, SWI, Cumulative_Rainfall (6 features)
  - NO_SWI: Rain Hourly, Cumulative_Rainfall (2 features — rainfall only)
  - SWI_ONLY: H1, H2, H3, SWI (4 features — tank model outputs only)

This proves the VALUE of coupling SWI with DL.

@author: AE-lab
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from collections import Counter
import matplotlib.pyplot as plt
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

RESULTS_DIR = r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Ablation_Study_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Feature sets for ablation
FEATURE_SETS = {
    'ALL_FEATURES':   ['Rain Hourly', 'H1', 'H2', 'H3', 'SWI', 'Cumulative_Rainfall'],
    'RAINFALL_ONLY':  ['Rain Hourly', 'Cumulative_Rainfall'],
    'SWI_ONLY':       ['H1', 'H2', 'H3', 'SWI'],
}

TARGET = 'Landslide'
SEQUENCE_LENGTH = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

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


def build_model(model_type, input_shape):
    """Build LSTM or GRU with same architecture"""
    layer = LSTM if model_type == 'LSTM' else GRU
    model = Sequential([
        layer(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        layer(16, return_sequences=False),
        Dropout(0.3),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=['accuracy']
    )
    return model


def get_metrics(y_true, y_pred, y_proba):
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics['AUC'] = roc_auc_score(y_true, y_proba)
    else:
        metrics['AUC'] = np.nan
    return metrics


# =============================================================================
# MAIN ABLATION LOOP
# =============================================================================

all_results = []

for station_name, file_path in STATION_FILES.items():
    print(f"\n{'='*70}")
    print(f"STATION: {station_name}")
    print(f"{'='*70}")

    data_raw = load_data(file_path)
    print(f"Rows: {len(data_raw)}, Class: {dict(Counter(data_raw[TARGET]))}")

    for feat_set_name, features in FEATURE_SETS.items():
        print(f"\n  --- Feature Set: {feat_set_name} ({len(features)} features) ---")
        print(f"  Features: {features}")

        # Normalize
        data = data_raw.copy()
        scaler = MinMaxScaler()
        data[features] = scaler.fit_transform(data[features])

        # Sequences
        X, y = create_sequences(data, features, TARGET, SEQUENCE_LENGTH)

        if len(np.unique(y)) < 2:
            print(f"  SKIP — only one class in sequences")
            continue

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

        # Class weights
        cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        cw_dict = dict(enumerate(cw))

        for model_type in ['LSTM', 'GRU']:
            print(f"    Training {model_type}...", end=' ')

            model = build_model(model_type, (X_train.shape[1], X_train.shape[2]))
            model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=EPOCHS, batch_size=BATCH_SIZE,
                class_weight=cw_dict, verbose=0
            )

            proba = model.predict(X_test, verbose=0).flatten()
            pred = (proba > 0.5).astype(int)
            metrics = get_metrics(y_test, pred, proba)

            metrics['Station'] = station_name
            metrics['Model'] = model_type
            metrics['Feature_Set'] = feat_set_name
            metrics['Num_Features'] = len(features)
            all_results.append(metrics)

            print(f"Acc={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}, AUC={metrics['AUC']}")

# =============================================================================
# SAVE RESULTS
# =============================================================================

results_df = pd.DataFrame(all_results)
results_df = results_df[['Station', 'Model', 'Feature_Set', 'Num_Features',
                          'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]
results_df = results_df.round(4)
results_df.to_csv(os.path.join(RESULTS_DIR, 'ablation_results.csv'), index=False)

# =============================================================================
# ABLATION IMPROVEMENT TABLE
# Calculate how much SWI features improve over rainfall-only
# =============================================================================

improvement_rows = []
for station in results_df['Station'].unique():
    for model in ['LSTM', 'GRU']:
        all_feat = results_df[(results_df['Station'] == station) &
                              (results_df['Model'] == model) &
                              (results_df['Feature_Set'] == 'ALL_FEATURES')]
        rain_only = results_df[(results_df['Station'] == station) &
                               (results_df['Model'] == model) &
                               (results_df['Feature_Set'] == 'RAINFALL_ONLY')]
        if len(all_feat) > 0 and len(rain_only) > 0:
            for metric in ['Accuracy', 'F1', 'AUC']:
                val_all = all_feat[metric].values[0]
                val_rain = rain_only[metric].values[0]
                delta = val_all - val_rain
                improvement_rows.append({
                    'Station': station, 'Model': model, 'Metric': metric,
                    'ALL_FEATURES': val_all, 'RAINFALL_ONLY': val_rain,
                    'Improvement': delta
                })

improvement_df = pd.DataFrame(improvement_rows)
improvement_df.to_csv(os.path.join(RESULTS_DIR, 'swi_improvement_analysis.csv'), index=False)

# =============================================================================
# VISUALIZATION — GROUPED BAR CHART (600 DPI)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=60)
axes = axes.flatten()
stations = results_df['Station'].unique()

colors_feat = {'ALL_FEATURES': '#2ca02c', 'RAINFALL_ONLY': '#d62728', 'SWI_ONLY': '#1f77b4'}
hatches_model = {'LSTM': '', 'GRU': '///'}

for idx, station in enumerate(stations):
    ax = axes[idx]
    sdata = results_df[results_df['Station'] == station]

    x = np.arange(3)  # 3 feature sets
    width = 0.35
    feat_sets = ['ALL_FEATURES', 'RAINFALL_ONLY', 'SWI_ONLY']

    for m_idx, model_type in enumerate(['LSTM', 'GRU']):
        f1_vals = []
        for fs in feat_sets:
            row = sdata[(sdata['Model'] == model_type) & (sdata['Feature_Set'] == fs)]
            f1_vals.append(row['F1'].values[0] if len(row) > 0 else 0)

        bars = ax.bar(x + m_idx * width, f1_vals, width,
                       label=model_type, hatch=hatches_model[model_type],
                       color=[colors_feat[fs] for fs in feat_sets],
                       edgecolor='black', linewidth=0.5, alpha=0.8)

    ax.set_title(f'{station}', fontsize=13, fontweight='bold')
    ax.set_xlabel('Feature Set', fontsize=10)
    ax.set_ylabel('F1 Score', fontsize=10)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(['All (6)', 'Rain Only (2)', 'SWI Only (4)'], fontsize=9)
    ax.set_ylim([0, 1.1])
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Ablation Study: F1 Score by Feature Set × Model × Station',
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'ablation_f1_comparison_600dpi.png'),
            dpi=600, bbox_inches='tight')
plt.show()

# =============================================================================
# PRINT SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print("ABLATION STUDY SUMMARY")
print(f"{'='*70}")
print(results_df.to_string(index=False))

print(f"\n--- SWI Feature Improvement (ALL vs RAINFALL_ONLY) ---")
if len(improvement_df) > 0:
    print(improvement_df.to_string(index=False))

print(f"\n{'='*70}")
print(f"Results saved to: {RESULTS_DIR}")
print(f"Files: ablation_results.csv, swi_improvement_analysis.csv, ablation_f1_comparison_600dpi.png")
print(f"{'='*70}")
