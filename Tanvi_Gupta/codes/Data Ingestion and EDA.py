# 1. Data Ingestion
import pandas as pd
import numpy as np
from scipy.stats import zscore
from google.colab import files

uploaded = files.upload()

# Define file path
data_path = 'final_adjusted_crowd_dataset.csv'

# Load the dataset
df = pd.read_csv(data_path)
print(df.head())
# 2. Exploratory Data Analysis (EDA) and Data Preprocessing
# Check for missing values and data types
print("Dataset Information:")
print(df.info())

print("\nMissing Values Count:")
print(df.isnull().sum())

# Drop or fill missing values as required (example: forward fill missing data)
df.fillna(method='ffill', inplace=True)

# Statistical summary
print("\nSummary Statistics:")
print(df.describe())

# Plotting distributions for key sensor data (Box and Whisker plot for outliers)
import seaborn as sns
import matplotlib.pyplot as plt

# Detecting outliers using separate Box and Whisker’s plot for each variable
# List of relevant columns (based on the dataset structure)
sensor_columns = ['Speed', 'Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']

# Plot separate Box and Whisker plots for each column
for col in sensor_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Box and Whisker Plot for {col}")
    plt.show()
