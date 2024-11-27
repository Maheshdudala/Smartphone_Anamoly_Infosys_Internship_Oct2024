# Visualize anomalies on time-series plot with updated colors
def plot_time_series_with_anomalies(data, column, iqr_outliers,zscore_outliers):
    plt.figure(figsize=(14, 6))
  
# Plot the time series with a light blue color
    plt.plot(data.index, data[column], label='Data', color='lightblue', alpha=0.7)
  
# Plot IQR outliers with dark blue color
    plt.scatter(iqr_outliers.index, iqr_outliers[column], color='darkblue',label='IQR Outliers', s=50, marker='o')
  
# Plot Z-score outliers with bright red color
    plt.scatter(zscore_outliers.index, zscore_outliers[column], color='red',label='Z-Score Outliers', s=50, marker='x')
    plt.title(f'Time-Series Plot with Anomalies for {column}')
    plt.xlabel('Time')
    plt.ylabel(column)
    plt.legend()
    plt.show()
  
# Example usage for each column
columns = ['Acc X','Acc Y','Acc Z','Heading','gyro_x','gyro_y','Gyro_Change','Net_Displacement','Speed_Change','Heading_Change','Rolling_Acc_Mean','Rolling_Acc_STD','acc_mean','acc_std','gyro_mean','gyro_std']
for col in columns:
  
# Calculate IQR and Z-score outliers for each column
    iqr_outliers, iqr_lower, iqr_upper = calculate_iqr_outliers(data, col)
    zscore_outliers, z_mean, z_std = calculate_zscore_outliers(data, col)
  
# Plot time-series with anomalies for each column
    print(f"Time-series plot with anomalies for {col}:")
    plot_time_series_with_anomalies(data, col, iqr_outliers, zscore_outliers)
