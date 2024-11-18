import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Step 1: Define a function for IQR-based outlier detection
def detect_outliers_iqr(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data < lower_bound) | (data > upper_bound)

# Step 2: Define a function for Z-score-based outlier detection
def detect_outliers_zscore(data, threshold=3):
    z_scores = zscore(data)
    return np.abs(z_scores) > threshold

# Step 3: Apply outlier detection
# Features with high correlation with 'label'
high_corr_with_label = [
    ('label', 'Total_Acc'), 
    ('label', 'Acc_Magnitude'),
    ('label', 'Gyro_Magnitude'),
    ('label', 'Total_Gyro_Acc')
]

# Combine the two lists for outlier detection
all_corr_pairs = high_corr_with_label 

for feature1, feature2 in all_corr_pairs:
    # Detect outliers using IQR
    df[f'{feature1}_{feature2}_IQR_Outliers'] = detect_outliers_iqr(df[feature1]) | detect_outliers_iqr(df[feature2])
    
    # Detect outliers using Z-score
    df[f'{feature1}_{feature2}_Z_Outliers'] = detect_outliers_zscore(df[feature1]) | detect_outliers_zscore(df[feature2])
    
    # Compare the outliers detected by both methods
    print(f"Outliers detected in {feature1} and {feature2} (IQR vs Z-Score):")
    comparison = df[[f'{feature1}_{feature2}_IQR_Outliers', f'{feature1}_{feature2}_Z_Outliers']].head()
    print(comparison)
#Visualize the outliers for comparison
# Scatter plot with custom colors for outliers
for feature1, feature2 in all_corr_pairs:
    # IQR Outliers (set to red)
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature1], df[feature2], c=df[f'{feature1}_{feature2}_IQR_Outliers'].apply(lambda x: 'red' if x else 'gray'), label='IQR Outliers')
    plt.title(f"Outliers in {feature1} vs {feature2} (IQR Method)")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.show()

    # Z-Score Outliers (set to green)
    plt.figure(figsize=(10, 6))
    plt.scatter(df[feature1], df[feature2], c=df[f'{feature1}_{feature2}_Z_Outliers'].apply(lambda x: 'green' if x else 'gray'), label='Z-Score Outliers')
    plt.title(f"Outliers in {feature1} vs {feature2} (Z-Score Method)")
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend()
    plt.show()
