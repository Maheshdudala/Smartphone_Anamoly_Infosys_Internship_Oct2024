# 1b. Feature Engineering and Data Augmentation

# Create new features: Total Acceleration
print('Creating new features...')
df['Total_Acc'] = np.sqrt(df['Acc X']**2 + df['Acc Y']**2 + df['Acc Z']**2)

# Time-based features: Rolling averages
df['Acc_X_Rolling_Mean'] = df['Acc X'].rolling(window=5).mean().fillna(df['Acc X'].mean())
df['Acc_Y_Rolling_Mean'] = df['Acc Y'].rolling(window=5).mean().fillna(df['Acc Y'].mean())
df['Acc_Z_Rolling_Mean'] = df['Acc Z'].rolling(window=5).mean().fillna(df['Acc Z'].mean())

# Time-based features: Moving variance for accelerometer and gyroscope data
print('Creating moving variance features...')
df['Acc_X_Moving_Var'] = df['Acc X'].rolling(window=5).var().fillna(df['Acc X'].var())
df['Acc_Y_Moving_Var'] = df['Acc Y'].rolling(window=5).var().fillna(df['Acc Y'].var())
df['Acc_Z_Moving_Var'] = df['Acc Z'].rolling(window=5).var().fillna(df['Acc Z'].var())
df['Gyro_X_Moving_Var'] = df['gyro_x'].rolling(window=5).var().fillna(df['gyro_x'].var())
df['Gyro_Y_Moving_Var'] = df['gyro_y'].rolling(window=5).var().fillna(df['gyro_y'].var())
df['Gyro_Z_Moving_Var'] = df['gyro_z'].rolling(window=5).var().fillna(df['gyro_z'].var())

# Print the new features to verify
print(df[['Acc_X_Moving_Var', 'Acc_Y_Moving_Var', 'Acc_Z_Moving_Var', 
          'Gyro_X_Moving_Var', 'Gyro_Y_Moving_Var', 'Gyro_Z_Moving_Var']].head())
# 2c. Advanced Feature Engineering

# Interaction terms between accelerometer and gyroscope readings
df['Acc_X_Gyro_X'] = df['Acc X'] * df['gyro_x']
df['Acc_Y_Gyro_Y'] = df['Acc Y'] * df['gyro_y']
df['Acc_Z_Gyro_Z'] = df['Acc Z'] * df['gyro_z']

# Create new derived features (e.g., ratios, accelerometer magnitude)
df['Acc_Magnitude'] = np.sqrt(df['Acc X']**2 + df['Acc Y']**2 + df['Acc Z']**2)  # Magnitude of acceleration
df['Gyro_Magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)  # Magnitude of angular velocity

# Add new features: Speed_Change, Acceleration_Spike, and Total_Gyro_Acc
df['Speed_Change'] = df['Speed'].diff().fillna(0)  # Calculate speed change
df['Acceleration_Spike'] = df['Total_Acc'].diff().fillna(0)  # Calculate acceleration spikes
df['Total_Gyro_Acc'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)  # Total gyroscope acceleration

# Print out the new features created
print("Advanced feature engineering complete. New features added:")
print(df[['Acc_X_Gyro_X', 'Acc_Y_Gyro_Y', 'Acc_Z_Gyro_Z', 'Acc_Magnitude', 'Gyro_Magnitude', 'Speed_Change', 'Acceleration_Spike','Total_Gyro_Acc']].head())

