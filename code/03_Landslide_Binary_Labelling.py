# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 22:29:34 2025

@author: AE-lab
"""

import pandas as pd

# Read the existing CSV file
file_path = r'C:\Users\AE-lab\Downloads\2023_Changlang_SWI_results_with_cumulative.csv'
df = pd.read_csv(file_path)

# Create binary Landslide column based on SWI threshold
df['Landslide'] = (df['SWI'] > 80).astype(int)

# Save the updated dataframe
output_path = r'C:\Users\AE-lab\Downloads\2023_Changlang_SWI_results_with_landslide.csv'
df.to_csv(output_path, index=False)

print("Landslide column added successfully!")
print(f"Updated file saved as: {output_path}")

# Display the rows where landslide occurred (SWI > 80)
landslide_events = df[df['Landslide'] == 1]
print(f"\nNumber of landslide events (SWI > 80): {len(landslide_events)}")
print("\nLandslide events (first 10):")
print(landslide_events[['Date And Time', 'SWI', 'Landslide']].head(10))

# Display summary statistics
print("\nSummary Statistics:")
print(f"Total rows: {len(df)}")
print(f"Landslide events (1): {len(landslide_events)}")
print(f"Non-landslide events (0): {len(df) - len(landslide_events)}")
print(f"Percentage of landslide events: {(len(landslide_events)/len(df)*100):.2f}%")
print(f"Maximum SWI value: {df['SWI'].max():.2f}")
print(f"Minimum SWI value: {df['SWI'].min():.2f}")

# Display the transition points where landslide status changes
df['Landslide_shift'] = df['Landslide'].shift()
transition_points = df[df['Landslide'] != df['Landslide_shift']]
print("\nTransition points (where landslide status changes):")
print(transition_points[['Date And Time', 'SWI', 'Landslide']].head(10))