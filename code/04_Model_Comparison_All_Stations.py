# -*- coding: utf-8 -*-
"""
Model Comparison: LSTM vs GRU vs Random Forest vs XGBoost
All 4 stations — Same pipeline, same split, same features
For Q1 Journal Paper

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score,
                             precision_score, recall_score, accuracy_score)
import xgboost as xgb
import matplotlib.pyplot as plt
from collections import Counter
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION — UPDATE THESE PATHS
# =============================================================================

# Station data paths (update to your local paths)
STATION_FILES = {
    'Changlang': r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Changlang_SWI_results_with_landslide.csv',
    'Tai':       r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Tai_SWI_results_with_landslide.csv',
    'Danglat':   r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Danglat_SWI_results_with_landslide.csv',
    'Manga':     r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Manga_SWI_results_with_landslide.csv',
}

# Output directory
RESULTS_DIR = r'C:\Users\AE-lab\Downloads\LandSlide_Early_Warning\Model_Comparison_Results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# Pipeline parameters (SAME as your LSTM script)
FEATURES = ['Rain Hourly', 'H1', 'H2', 'H3', 'SWI', 'Cumulative_Rainfall']
TARGET = 'Landslide'
SEQUENCE_LENGTH = 10
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.0001

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_and_prepare(file_path):
    """Load CSV, set datetime index"""
    data = pd.read_csv(file_path)
    data['Date And Time'] = pd.to_datetime(data['Date And Time'])
    data.set_index('Date And Time', inplace=True)
    return data


def create_sequences(data, features, target, seq_len):
    """Create sliding window sequences for DL models"""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[features].iloc[i:i+seq_len].values)
        y.append(data[target].iloc[i+seq_len])
    return np.array(X), np.array(y)


def create_flat_features(X_seq):
    """Flatten sequences for ML models (RF, XGBoost)
    Each sequence (10 timesteps × 6 features) → 60 features
    """
    return X_seq.reshape(X_seq.shape[0], -1)


def build_lstm_model(input_shape):
    """Same architecture as your LSTM_Landslide_Prediction2.py"""
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(16, return_sequences=False),
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


def build_gru_model(input_shape):
    """GRU — same architecture, GRU layers instead of LSTM"""
    model = Sequential([
        GRU(32, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        GRU(16, return_sequences=False),
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


def evaluate_model(y_true, y_pred, y_proba):
    """Compute all metrics"""
    results = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
    }
    # AUC — needs both classes in y_true
    if len(np.unique(y_true)) > 1:
        results['AUC'] = roc_auc_score(y_true, y_proba)
    else:
        results['AUC'] = np.nan
    return results


# =============================================================================
# MAIN LOOP — ALL STATIONS × ALL MODELS
# =============================================================================

all_results = []           # For summary table
all_roc_data = {}          # For ROC curves
all_confusion_matrices = {}

for station_name, file_path in STATION_FILES.items():
    print(f"\n{'='*70}")
    print(f"STATION: {station_name}")
    print(f"{'='*70}")

    # --- Load data ---
    data = load_and_prepare(file_path)
    print(f"Rows: {len(data)}, Class dist: {dict(Counter(data[TARGET]))}")

    # --- Normalize ---
    scaler = MinMaxScaler()
    data[FEATURES] = scaler.fit_transform(data[FEATURES])

    # --- Create sequences ---
    X_seq, y = create_sequences(data, FEATURES, TARGET, SEQUENCE_LENGTH)
    print(f"Sequences: {len(X_seq)}, Class dist: {dict(Counter(y))}")

    # --- Stratified split ---
    X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        X_seq, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(y_train)} (pos={sum(y_train)}), Test: {len(y_test)} (pos={sum(y_test)})")

    # --- Flat features for ML models ---
    X_train_flat = create_flat_features(X_train_seq)
    X_test_flat = create_flat_features(X_test_seq)

    # --- Class weights ---
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(enumerate(cw))
    print(f"Class weights: {class_weight_dict}")

    # --- XGBoost scale_pos_weight ---
    n_neg = sum(y_train == 0)
    n_pos = sum(y_train == 1)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # Store ROC data per station
    station_roc = {}

    # =====================================================================
    # MODEL 1: LSTM (same as your original)
    # =====================================================================
    print(f"\n--- LSTM ---")
    lstm_model = build_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    lstm_model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weight_dict, verbose=0
    )
    lstm_proba = lstm_model.predict(X_test_seq, verbose=0).flatten()
    lstm_pred = (lstm_proba > 0.5).astype(int)
    lstm_metrics = evaluate_model(y_test, lstm_pred, lstm_proba)
    lstm_metrics['Model'] = 'LSTM'
    lstm_metrics['Station'] = station_name
    all_results.append(lstm_metrics)
    station_roc['LSTM'] = (y_test, lstm_proba)
    all_confusion_matrices[f'{station_name}_LSTM'] = confusion_matrix(y_test, lstm_pred)
    print(f"  Acc={lstm_metrics['Accuracy']:.4f}, P={lstm_metrics['Precision']:.4f}, "
          f"R={lstm_metrics['Recall']:.4f}, F1={lstm_metrics['F1']:.4f}, AUC={lstm_metrics['AUC']}")

    # =====================================================================
    # MODEL 2: GRU
    # =====================================================================
    print(f"\n--- GRU ---")
    gru_model = build_gru_model((X_train_seq.shape[1], X_train_seq.shape[2]))
    gru_model.fit(
        X_train_seq, y_train,
        validation_data=(X_test_seq, y_test),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        class_weight=class_weight_dict, verbose=0
    )
    gru_proba = gru_model.predict(X_test_seq, verbose=0).flatten()
    gru_pred = (gru_proba > 0.5).astype(int)
    gru_metrics = evaluate_model(y_test, gru_pred, gru_proba)
    gru_metrics['Model'] = 'GRU'
    gru_metrics['Station'] = station_name
    all_results.append(gru_metrics)
    station_roc['GRU'] = (y_test, gru_proba)
    all_confusion_matrices[f'{station_name}_GRU'] = confusion_matrix(y_test, gru_pred)
    print(f"  Acc={gru_metrics['Accuracy']:.4f}, P={gru_metrics['Precision']:.4f}, "
          f"R={gru_metrics['Recall']:.4f}, F1={gru_metrics['F1']:.4f}, AUC={gru_metrics['AUC']}")

    # =====================================================================
    # MODEL 3: Random Forest
    # =====================================================================
    print(f"\n--- Random Forest ---")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_depth=10,          # prevent overfitting on small data
        min_samples_leaf=2
    )
    rf_model.fit(X_train_flat, y_train)
    rf_proba = rf_model.predict_proba(X_test_flat)[:, 1]
    rf_pred = rf_model.predict(X_test_flat)
    rf_metrics = evaluate_model(y_test, rf_pred, rf_proba)
    rf_metrics['Model'] = 'Random Forest'
    rf_metrics['Station'] = station_name
    all_results.append(rf_metrics)
    station_roc['Random Forest'] = (y_test, rf_proba)
    all_confusion_matrices[f'{station_name}_RF'] = confusion_matrix(y_test, rf_pred)
    print(f"  Acc={rf_metrics['Accuracy']:.4f}, P={rf_metrics['Precision']:.4f}, "
          f"R={rf_metrics['Recall']:.4f}, F1={rf_metrics['F1']:.4f}, AUC={rf_metrics['AUC']}")

    # =====================================================================
    # MODEL 4: XGBoost
    # =====================================================================
    print(f"\n--- XGBoost ---")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric='logloss',
        use_label_encoder=False
    )
    xgb_model.fit(X_train_flat, y_train)
    xgb_proba = xgb_model.predict_proba(X_test_flat)[:, 1]
    xgb_pred = xgb_model.predict(X_test_flat)
    xgb_metrics = evaluate_model(y_test, xgb_pred, xgb_proba)
    xgb_metrics['Model'] = 'XGBoost'
    xgb_metrics['Station'] = station_name
    all_results.append(xgb_metrics)
    station_roc['XGBoost'] = (y_test, xgb_proba)
    all_confusion_matrices[f'{station_name}_XGB'] = confusion_matrix(y_test, xgb_pred)
    print(f"  Acc={xgb_metrics['Accuracy']:.4f}, P={xgb_metrics['Precision']:.4f}, "
          f"R={xgb_metrics['Recall']:.4f}, F1={xgb_metrics['F1']:.4f}, AUC={xgb_metrics['AUC']}")

    # Store ROC data
    all_roc_data[station_name] = station_roc

    # =====================================================================
    # Save individual station models (GRU)
    # =====================================================================
    station_dir = os.path.join(RESULTS_DIR, station_name)
    os.makedirs(station_dir, exist_ok=True)
    gru_model.save(os.path.join(station_dir, 'gru_model.h5'))
    lstm_model.save(os.path.join(station_dir, 'lstm_model.h5'))

# =============================================================================
# SAVE COMPREHENSIVE RESULTS TABLE
# =============================================================================

results_df = pd.DataFrame(all_results)
results_df = results_df[['Station', 'Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC']]
results_df = results_df.round(4)

# Save to CSV
results_df.to_csv(os.path.join(RESULTS_DIR, 'model_comparison_all_stations.csv'), index=False)

# Save confusion matrices
cm_rows = []
for key, cm in all_confusion_matrices.items():
    station, model = key.rsplit('_', 1)
    model_map = {'LSTM': 'LSTM', 'GRU': 'GRU', 'RF': 'Random Forest', 'XGB': 'XGBoost'}
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (cm[0][0], 0, 0, 0)
    cm_rows.append({
        'Station': station, 'Model': model_map.get(model, model),
        'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp
    })
cm_df = pd.DataFrame(cm_rows)
cm_df.to_csv(os.path.join(RESULTS_DIR, 'confusion_matrices_all.csv'), index=False)

# =============================================================================
# ROC CURVES — ONE PLOT PER STATION (600 DPI)
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=60)
axes = axes.flatten()
colors = {'LSTM': '#1f77b4', 'GRU': '#ff7f0e', 'Random Forest': '#2ca02c', 'XGBoost': '#d62728'}

for idx, (station_name, roc_data) in enumerate(all_roc_data.items()):
    ax = axes[idx]
    for model_name, (y_true, y_proba) in roc_data.items():
        if len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_val = roc_auc_score(y_true, y_proba)
            ax.plot(fpr, tpr, color=colors[model_name], linewidth=2.5,
                    label=f'{model_name} (AUC={auc_val:.3f})')
        else:
            ax.plot([0, 1], [0, 1], color=colors[model_name], linewidth=2.5,
                    linestyle=':', label=f'{model_name} (AUC=N/A)')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_title(f'{station_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

plt.suptitle('ROC Curves — All Models × All Stations', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'ROC_curves_all_stations_600dpi.png'),
            dpi=600, bbox_inches='tight')
plt.show()

# =============================================================================
# BAR CHART COMPARISON (600 DPI)
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=60)
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1']
stations = results_df['Station'].unique()
models = results_df['Model'].unique()
x = np.arange(len(stations))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    for j, model in enumerate(models):
        vals = results_df[results_df['Model'] == model][metric].values
        ax.bar(x + j * width, vals, width, label=model, color=list(colors.values())[j])
    ax.set_xlabel('Station', fontsize=11)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(stations, rotation=45, ha='right')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

plt.suptitle('Model Comparison Across All Stations', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison_bars_600dpi.png'),
            dpi=600, bbox_inches='tight')
plt.show()

# =============================================================================
# PRINT FINAL SUMMARY
# =============================================================================

print(f"\n{'='*70}")
print("FINAL MODEL COMPARISON SUMMARY")
print(f"{'='*70}")
print(results_df.to_string(index=False))
print(f"\n{'='*70}")
print(f"Results saved to: {RESULTS_DIR}")
print(f"Files generated:")
print(f"  1. model_comparison_all_stations.csv")
print(f"  2. confusion_matrices_all.csv")
print(f"  3. ROC_curves_all_stations_600dpi.png")
print(f"  4. model_comparison_bars_600dpi.png")
print(f"  5. Per-station GRU + LSTM model files (.h5)")
print(f"{'='*70}")
