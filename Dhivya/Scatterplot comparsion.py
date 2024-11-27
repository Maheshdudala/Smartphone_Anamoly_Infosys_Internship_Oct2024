#Scatter plot to compare IQR vs Z-score outliers with updated colors
def plot_scatter_comparison(data, column, iqr_outliers, zscore_outliers):
    plt.figure(figsize=(10, 6))

# Plot original data with light blue
    plt.scatter(data.index, data[column], color='lightblue', label='Data',alpha=0.5, s=10)

# Plot IQR outliers with dark blue color
    plt.scatter(iqr_outliers.index, iqr_outliers[column], color='darkblue',label='IQR Outliers', s=70, marker='o')

# Plot Z-score outliers with bright red color
    plt.scatter(zscore_outliers.index, zscore_outliers[column], color='red',label='Z-Score Outliers', s=70, marker='x')
    plt.title(f'Scatter Plot Comparison for {column}')
    plt.xlabel('Index')
    plt.ylabel(column)
    plt.legend()
    plt.show()

# Example usage for each column
for col in columns:
  
# Calculate IQR and Z-score outliers for each column
    iqr_outliers, iqr_lower, iqr_upper = calculate_iqr_outliers(data, col)
    zscore_outliers, z_mean, z_std = calculate_zscore_outliers(data, col)
  
# Plot scatter comparison for each column
    print(f"Scatter plot comparison for {col}:")
    plot_scatter_comparison(data, col, iqr_outliers, zscore_outliers)
