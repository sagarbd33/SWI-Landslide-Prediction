# -*- coding: utf-8 -*-
"""
Created on [Current Date]

@author: AE-lab
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# Load the saved model
model_path = r'C:\Users\AE-lab\Downloads\Landslide_Model_Results\landslide_lstm_model_improved.h5'
model = load_model(model_path)
print("Model loaded successfully!")

# Load your dataset
data_path = r'C:\Users\AE-lab\Downloads\2023_Changlang_SWI_results_with_landslide.csv'
data = pd.read_csv(data_path)
data['Date And Time'] = pd.to_datetime(data['Date And Time'])
data.set_index('Date And Time', inplace=True)

print(f"Dataset shape: {data.shape}")

# Select input features (same as during training)
features = ['Rain Hourly', 'H1', 'H2', 'H3', 'SWI', 'Cumulative_Rainfall']
target = 'Landslide'

print(f"Features used for prediction: {features}")

# Normalize input features
scaler = MinMaxScaler()
data_scaled = data.copy()
data_scaled[features] = scaler.fit_transform(data[features])

# Create sequences (same sequence_length as during training)
sequence_length = 10
X_pred, dates_pred = [], []

for i in range(len(data_scaled) - sequence_length):
    X_pred.append(data_scaled[features].iloc[i:i+sequence_length].values)
    dates_pred.append(data_scaled.index[i+sequence_length])

X_pred = np.array(X_pred)

print(f"Created {len(X_pred)} sequences for prediction")

# Make predictions
print("Making predictions...")
y_pred_proba = model.predict(X_pred)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

print("Predictions completed!")

# Create results dataframe
results_df = pd.DataFrame({
    'Date_And_Time': dates_pred,
    'Predicted_Probability': y_pred_proba.flatten(),
    'Predicted_Landslide': y_pred
})

# Merge with original data
results_full = data.iloc[sequence_length:].copy()
results_full = results_full.join(results_df.set_index('Date_And_Time'))

print(f"Prediction results shape: {results_full.shape}")

# =============================================================================
# SAVE LANDSLIDE_PREDICTIONS_COMPLETE.CSV
# =============================================================================

results_dir = r'C:\Users\AE-lab\Downloads\2023\Landslide_Model_Results'
os.makedirs(results_dir, exist_ok=True)

# Save complete predictions
output_path = os.path.join(results_dir, 'landslide_predictions_complete.csv')
results_full.to_csv(output_path)
print(f"\nSaved: landslide_predictions_complete.csv")

# =============================================================================
# CREATE HIGH-QUALITY VISUALIZATION (600 DPI) - 3 PANEL PLOT
# =============================================================================

plt.figure(figsize=(15, 10), dpi=600)

# Plot 1: Actual vs Predicted Landslide
plt.subplot(3, 1, 1)
if 'Landslide' in results_full.columns:
    plt.plot(results_full.index, results_full['Landslide'], 'bo-', 
             label='Actual Landslide', linewidth=2, markersize=4, alpha=0.7)
plt.plot(results_full.index, results_full['Predicted_Landslide'], 'rx--', 
         label='Predicted Landslide', linewidth=1.5, markersize=6)
plt.title('Landslide Predictions vs Actual', fontsize=14, fontweight='bold')
plt.ylabel('Landslide (0/1)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 2: Prediction probabilities
plt.subplot(3, 1, 2)
plt.plot(results_full.index, results_full['Predicted_Probability'], 'g-', 
         label='Prediction Probability', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold (0.5)', linewidth=2)
plt.title('Landslide Prediction Probabilities', fontsize=14, fontweight='bold')
plt.ylabel('Probability', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Plot 3: SWI and Rainfall (key indicators)
plt.subplot(3, 1, 3)
# Plot SWI
ax1 = plt.gca()
ax1.plot(results_full.index, results_full['SWI'], 'b-', label='SWI', linewidth=2)
ax1.set_ylabel('SWI', color='b', fontsize=12)
ax1.tick_params(axis='y', labelcolor='b')

# Create second y-axis for rainfall
ax2 = ax1.twinx()
ax2.plot(results_full.index, results_full['Rain Hourly'], 'r-', 
         label='Rainfall (mm)', linewidth=1, alpha=0.7)
ax2.set_ylabel('Rainfall (mm)', color='r', fontsize=12)
ax2.tick_params(axis='y', labelcolor='r')

plt.title('SWI and Rainfall Patterns', fontsize=14, fontweight='bold')
plt.xlabel('Date and Time', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()

# Save the high-quality visualization
plt.savefig(os.path.join(results_dir, 'prediction_results_600dpi.png'), 
            dpi=600, bbox_inches='tight')
plt.show()

print(f"Saved: prediction_results_600dpi.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(f"\n" + "="*70)
print("PREDICTION COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"Generated files:")
print(f"1. landslide_predictions_complete.csv")
print(f"2. prediction_results_600dpi.png")
print(f"\nLocation: {results_dir}")
print("="*70)

# Show first few predictions
print("\nFirst 10 predictions:")
print(results_full[['Rain Hourly', 'SWI', 'Landslide', 'Predicted_Probability', 'Predicted_Landslide']].head(10))