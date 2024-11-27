# Data Preparation - Handle missing values and standardize variables if necessary
print("Standardizing numeric variables...")
numeric_df = data.select_dtypes(include=[np.number])
scaler = StandardScaler()
data_scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)
# Compute Correlations - Calculate pairwise correlation coefficients between all variables
print("Computing correlation matrix...")
corr_matrix = data_scaled.corr()
# Generate Scatter Plots for key variable pairs
# Scatter Plot for Acc X vs Acc Y
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Acc X', y='Acc Y', data=data)
plt.title("Scatter Plot: Acc X vs Acc Y")
plt.xlabel('Acc X')
plt.ylabel('Acc Y')
plt.show()
# Scatter Plots for All Combinations of Features
# Create a pair plot for visualizing relationships between all pairs of features
pair_plot_columns = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z','Speed']
sns.pairplot(data[pair_plot_columns])
plt.suptitle('Pair Plot of Sensor Features', y=1.02) # Adjust title position
plt.show()
