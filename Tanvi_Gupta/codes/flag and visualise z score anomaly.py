# Flag anomalies where Z-score > 3 or < -3
aug_df['Anomaly'] = (np.abs(z_scores) > 3).any(axis=1).astype(int)

# Save the updated dataset
aug_df.to_csv('augmented_dataset.csv', index=False)

# Display anomalies if needed
anomalies = aug_df[aug_df['Anomaly'] == 1]
print(anomalies)


# Define a function for Z-score calculation and anomaly detection
def calculate_z_scores(df, column_name):
    mean = df[column_name].mean()
    std = df[column_name].std()
    z_scores = (df[column_name] - mean) / std          # Z= (X-mean)/sd
    df[f'Z-Score_{column_name}'] = z_scores
    df[f'Anomaly_{column_name}'] = z_scores.apply(lambda x: 'Yes' if abs(x) > 3 else 'No')

# List of parameters for which to calculate Z-scores
parameters = ['Acc X', 'Acc Y', 'Acc Z', 'Speed', 'gyro_x', 'gyro_y', 'gyro_z']

# Calculate Z-scores for each parameter
for param in parameters:
    calculate_z_scores(df, param)

# Display the results
print("Z-Score Detection Results:")
print(df)

# Visualize Z-scores and anomalies for all features
for param in parameters:
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Time', y=param, hue=f'Anomaly_{param}', style=f'Anomaly_{param}', data=df)
    plt.axhline(y=df[param].mean() + 3 * df[param].std(), color='r', linestyle='--', label='Upper Threshold')
    plt.axhline(y=df[param].mean() - 3 * df[param].std(), color='r', linestyle='--', label='Lower Threshold')
    plt.title(f'Z-Score Based Outlier Detection for {param}')
    plt.xlabel('Time')
    plt.ylabel(param)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# In a normal distribution:
# Approximately 68% of the data falls within 1 standard deviation from the mean.
# Approximately 95% of the data falls within 2 standard deviations from the mean.
# Approximately 99.7% of the data falls within 3 standard deviations from the mean.
# Thus, if a data point has a Z-score greater than 3 or less than -3, it is considered an outlier since it falls outside the range where 99.7% of the data points lie.
