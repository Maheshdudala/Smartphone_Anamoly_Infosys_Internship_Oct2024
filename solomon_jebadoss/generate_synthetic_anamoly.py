import numpy as np

# Generate synthetic anomalies by adding noise to selected columns
def generate_synthetic_anomalies(df, num_anomalies=100, noise_level=0.5):
    # Select random indices to alter
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    
    # Define columns to add noise to
    noise_columns = ['Total_Acc', 'Acc_Magnitude', 'Gyro_Magnitude','Total_Gyro_Acc']
    
    # Create synthetic anomalies by adding noise
    for col in noise_columns:
        df.loc[anomaly_indices, col] += noise_level * np.random.normal(size=num_anomalies)
    
    # Label synthetic anomalies
    df.loc[anomaly_indices, 'label'] = 1  # Mark as anomalies
    
    return df

# Apply synthetic anomalies generation
df_synthetic = generate_synthetic_anomalies(df.copy())
