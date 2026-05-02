# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 20:01:14 2025

@author: AE-lab
"""

import pandas as pd

# Read the existing SWI results file
file_path = r'C:\Users\AE-lab\Downloads\2023_Changlang_SWI_results.csv'
df = pd.read_csv(file_path)

# Calculate cumulative rainfall
df['Cumulative_Rainfall'] = df['Rain Hourly'].cumsum()

# Save the updated dataframe back to CSV
output_path = r'C:\Users\AE-lab\Downloads\2023_Changlang_SWI_results_with_cumulative.csv'
df.to_csv(output_path, index=False)

print("Cumulative rainfall column added successfully!")
print(f"Updated file saved as: {output_path}")

# Display first 15 rows to show the new column
print("\nFirst 15 rows with cumulative rainfall:")
print(df[['Date And Time', 'Rain Hourly', 'Cumulative_Rainfall', 'SWI']].head(15))

# Display summary statistics
print("\nSummary Statistics:")
print(f"Total Cumulative Rainfall: {df['Cumulative_Rainfall'].iloc[-1]:.2f} mm")
print(f"Maximum Hourly Rainfall: {df['Rain Hourly'].max():.2f} mm")
print(f"Average Hourly Rainfall: {df['Rain Hourly'].mean():.2f} mm")
print(f"Maximum SWI: {df['SWI'].max():.2f} mm")
print(f"Minimum SWI: {df['SWI'].min():.2f} mm")