# Calculate the overall acceleration magnitude from the three axes (X, Y, Z)
data['Acc_Magnitude'] = np.sqrt(data['Acc X']**2 + data['Acc Y']**2 + data['Acc Z']**2)

#Calculate the change in acceleration over time to capture sudden movements
data['Acc_Change'] = data['Acc_Magnitude'].diff().fillna(0)

#Magnitude of Angular Velocity: Calculate the overall rotational velocity magnitude
data['Gyro_Magnitude'] = np.sqrt(data['gyro_x']**2 + data['gyro_y']**2 + data['gyro_z']**2)

#Change in Gyroscopic Movement: Calculate the change in rotational velocity over time
data['Gyro_Change'] = data['Gyro_Magnitude'].diff().fillna(0)

#Net Displacement: Calculate the net displacement from the change in longitude and latitude
data['Net_Displacement'] = np.sqrt((data['Longitude'].diff()**2) + (data['Latitude'].diff()**2)).fillna(0)

#Speed Change: Capture the change in speed over time to identify sudden accelerations or decelerations.
data['Speed_Change'] = data['Speed'].diff().fillna(0)

#Heading Change: Calculate the change in heading over time to identify changes in direction.
data['Heading_Change'] = data['Heading'].diff().fillna(0)

#Rolling Mean/Standard Deviation: Calculate rolling statistics to capture trends over a specified window (e.g., last 5 observations)
data['Rolling_Acc_Mean'] = data['Acc_Magnitude'].rolling(window=5).mean()
data['Rolling_Acc_STD'] = data['Acc_Magnitude'].rolling(window=5).std()
data['acc_mean'] = data[['Acc X', 'Acc Y', 'Acc Z']].mean(axis=1)
data['acc_std'] = data[['Acc X', 'Acc Y', 'Acc Z']].std(axis=1)
data['gyro_mean'] = data[['gyro_x', 'gyro_y', 'gyro_z']].mean(axis=1)
data['gyro_std'] = data[['gyro_x', 'gyro_y', 'gyro_z']].std(axis=1)
