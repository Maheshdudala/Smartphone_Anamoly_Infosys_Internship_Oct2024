# Function to enhance synthetic anomalies by combining noise patterns
def enhance_synthetic_anomalies(df, num_anomalies=100, noise_level=0.5):
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    
    # Add combined noise to accelerometer and gyroscope data to create complex anomalies
    df.loc[anomaly_indices, 'Acc X'] += noise_level * np.random.normal(size=num_anomalies)
    df.loc[anomaly_indices, 'Acc Y'] += noise_level * np.random.normal(size=num_anomalies)
    df.loc[anomaly_indices, 'gyro_x'] += noise_level * np.random.normal(size=num_anomalies)
    df.loc[anomaly_indices, 'gyro_y'] += noise_level * np.random.normal(size=num_anomalies)
    
    # Mark as synthetic anomalies
    df.loc[anomaly_indices, 'label'] = 1  # Set label to 1 for anomalies
    
    return df

# Apply enhanced synthetic anomalies generation
df_enhanced = enhance_synthetic_anomalies(df.copy())
import matplotlib.pyplot as plt
import seaborn as sns

# Plot Acc X vs Gyro X to visualize enhanced anomalies
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_enhanced, x='Acc X', y='gyro_x', hue='IsoForest_Enhanced_Anomaly', palette='coolwarm')
plt.title("Isolation Forest Detection of Enhanced Synthetic Anomalies")
plt.xlabel("Acc X")
plt.ylabel("Gyro X")
plt.legend(title="Anomaly Detection")
plt.show()

