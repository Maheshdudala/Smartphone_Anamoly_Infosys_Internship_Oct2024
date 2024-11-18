import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.preprocessing import StandardScaler

def mahalanobis_outliers(df, features, threshold=3.0):
    # Check for NaN and infinite values
    if df[features].isnull().values.any() or np.isinf(df[features]).values.any():
        print("Data contains NaN or infinite values. Please clean the data before proceeding.")
        return df

    # Remove features with low variance
    low_variance_features = df[features].var() < 0.01  # Adjust the threshold as necessary
    features = df[features].columns[~low_variance_features]

    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])

    # Calculate the mean and covariance matrix of the features
    mean = np.mean(scaled_features, axis=0)
    cov_matrix = np.cov(scaled_features, rowvar=False)

    # Check if the covariance matrix is singular
    try:
        inv_cov_matrix = np.linalg.inv(cov_matrix)
    except np.linalg.LinAlgError:
        print("Covariance matrix is singular and cannot be inverted.")
        return df

    # Calculate Mahalanobis distance
    df['mahalanobis'] = np.sqrt(np.sum((scaled_features - mean) @ inv_cov_matrix * (scaled_features - mean), axis=1))

    # Calculate the chi-squared threshold for outlier detection
    chi2_threshold = chi2.ppf(0.99, df=len(features))  # Adjust as needed

    # Identify outliers
    df['mahalanobis_outlier'] = df['mahalanobis'] > np.sqrt(chi2_threshold)

    return df

# Example usage
selected_features = [
    'Total_Acc',  # or 'Acc_Magnitude' or 'Gyro_Magnitude'
    'Acc_Y_Moving_Var',
    'Longitude',
    'Latitude',
    'Gyro_X_Moving_Var',
    'Distance',
    'minute',
    'Acc_Gyro_Interaction'
]

# Replace `df` with your DataFrame containing the data
df_with_outliers = mahalanobis_outliers(df, selected_features)

# Count the number of detected outliers
num_outliers = df_with_outliers['mahalanobis_outlier'].sum()
print(f"Number of outliers detected using Mahalanobis Distance: {num_outliers}")
