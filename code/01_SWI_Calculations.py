# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 19:52:46 2025

@author: AE-lab
"""

import pandas as pd
import numpy as np

# Model Parameters
L11 = 15.0   # mm
L12 = 60.0   # mm
L2 = 15.0    # mm
L3 = 15.0    # mm

alpha11 = 0.10
alpha12 = 0.15
alpha2 = 0.05
alpha3 = 0.01

beta1 = 0.12
beta2 = 0.05
beta3 = 0.01

def calculate_flows(H1, H2, H3):
    """Calculate outflows and infiltrations for current water levels"""
    # Upper tank outflows
    q11 = alpha11 * (H1 - L11) if H1 > L11 else 0.0
    q12 = alpha12 * (H1 - L12) if H1 > L12 else 0.0
    
    # Middle tank outflow
    q2 = alpha2 * (H2 - L2) if H2 > L2 else 0.0
    
    # Lower tank outflow
    q3 = alpha3 * (H3 - L3) if H3 > L3 else 0.0
    
    # Infiltrations
    I1 = beta1 * H1
    I2 = beta2 * H2
    I3 = beta3 * H3
    
    return q11, q12, q2, q3, I1, I2, I3

def update_tank_levels(H1, H2, H3, R, q11, q12, q2, q3, I1, I2, I3):
    """Update water levels in all tanks"""
    # Update upper tank (H1)
    H1_new = H1 - q11 - q12 - I1 + R
    
    # Update middle tank (H2)
    H2_new = H2 - q2 - I2 + I1
    
    # Update lower tank (H3)
    H3_new = H3 - q3 - I3 + I2
    
    # Ensure water levels don't go negative
    H1_new = max(0, H1_new)
    H2_new = max(0, H2_new)
    H3_new = max(0, H3_new)
    
    return H1_new, H2_new, H3_new

# Read the CSV file from specific path
file_path = r'C:\Users\AE-lab\Downloads\Changlang_cleaned_2023.csv'
df = pd.read_csv(file_path)

# Initialize water levels
H1_current = 0.0
H2_current = 0.0
H3_current = 0.0

# Lists to store results
h1_values = []
h2_values = []
h3_values = []
swi_values = []

# Process each row in the dataframe
for index, row in df.iterrows():
    R = row['Rain Hourly']  # Current rainfall
    
    # Calculate flows and infiltrations
    q11, q12, q2, q3, I1, I2, I3 = calculate_flows(H1_current, H2_current, H3_current)
    
    # Update tank levels
    H1_new, H2_new, H3_new = update_tank_levels(
        H1_current, H2_current, H3_current, R, q11, q12, q2, q3, I1, I2, I3
    )
    
    # Calculate SWI
    SWI = H1_current + H2_current + H3_current
    
    # Store current values (before update for this timestep)
    h1_values.append(H1_current)
    h2_values.append(H2_current)
    h3_values.append(H3_current)
    swi_values.append(SWI)
    
    # Update current water levels for next iteration
    H1_current, H2_current, H3_current = H1_new, H2_new, H3_new

# Add calculated columns to dataframe
df['H1'] = h1_values
df['H2'] = h2_values
df['H3'] = h3_values
df['SWI'] = swi_values

# Save the results to a new CSV file in the same directory
output_path = r'C:\Users\AE-lab\Downloads\2023_Changlang_SWI_results.csv'
df.to_csv(output_path, index=False)

print("Calculation completed successfully!")
print(f"Input file: {file_path}")
print(f"Output file: {output_path}")
print(f"Final water levels - H1: {H1_current:.2f} mm, H2: {H2_current:.2f} mm, H3: {H3_current:.2f} mm")
print(f"Final SWI: {H1_current + H2_current + H3_current:.2f} mm")
print(f"Total rows processed: {len(df)}")

# Display first few rows
print("\nFirst 10 rows of results:")
print(df[['Date And Time', 'Rain Hourly', 'H1', 'H2', 'H3', 'SWI']].head(10))

# Display summary statistics
print("\nSummary Statistics:")
print(f"Maximum SWI: {df['SWI'].max():.2f} mm")
print(f"Minimum SWI: {df['SWI'].min():.2f} mm")
print(f"Average SWI: {df['SWI'].mean():.2f} mm")