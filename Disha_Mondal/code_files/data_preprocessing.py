import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = './data/final_adjusted_crowd_dataset.csv'
df = pd.read_csv(file_path)
df.head()
print(df['Time'].head(10))  # Check for any inconsistencies in the time format
# Ensure that the 'Time' column is properly formatted by replacing hyphens with colons
print('Converting Time column to datetime format...')

# Replace any hyphens with colons to match the HH:MM:SS format
df['Time'] = df['Time'].str.replace('-', ':')

# Now convert the column to datetime, assuming the corrected format is HH:MM:SS
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')

# Check the conversion
print(df[['Time']].head())
