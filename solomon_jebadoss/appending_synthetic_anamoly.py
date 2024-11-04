# 5d: Data Augmentation for Anomalies

def augment_data_with_synthetic_anomalies(df, num_anomalies=50):
    np.random.seed(42)  # For reproducibility
    
    # Create a list to store synthetic anomaly rows
    synthetic_anomalies = []

    # Create synthetic anomalies (e.g., large spikes in Acc X, Y, Z)
    for _ in range(num_anomalies):
        anomaly = {
            'Acc X': np.random.uniform(-30, 30),  # Simulated large values for accelerometer
            'Acc Y': np.random.uniform(-30, 30),
            'Acc Z': np.random.uniform(-30, 30),
            'gyro_x': np.random.uniform(-20, 20),  # Simulated large values for gyroscope
            'gyro_y': np.random.uniform(-20, 20),
            'gyro_z': np.random.uniform(-20, 20),
            'Speed': np.random.uniform(0, 150),   # Simulated high-speed spikes
            'Distance': np.random.uniform(100, 500)  # Simulated long-distance spikes
        }
        synthetic_anomalies.append(anomaly)