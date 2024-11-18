# 1d. Conduct Correlation Analysis

# i. Data Preparation - Handle missing values and standardize variables if necessary
print("Standardizing numeric variables...")
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

# ii. Compute Correlations - Calculate pairwise correlation coefficients between all variables
print("Computing correlation matrix...")
corr_matrix = df_scaled.corr()

# iii. Generate Scatter Plots for key variable pairs

#  Scatter Plot for Acc X vs Acc Y
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Acc X', y='Acc Y', data=df)
plt.title("Scatter Plot: Acc X vs Acc Y")
plt.xlabel('Acc X')
plt.ylabel('Acc Y')
plt.show()

# Scatter Plots for All Combinations of Features
# Create a pair plot for visualizing relationships between all pairs of features
pair_plot_columns = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z', 'Speed', 'Total_Acc']
sns.pairplot(df[pair_plot_columns])
plt.suptitle('Pair Plot of Sensor Features', y=1.02)  # Adjust title position
plt.show()
# Calculate the correlation matrix for numeric features
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

# Plot the heatmap for the correlation matrix
plt.figure(figsize=(14, 10))  # Increase the figure size
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
            annot_kws={"size": 8},  # Reduce annotation size
            cbar_kws={"shrink": .8},  # Adjust color bar size
            square=True,  # Make cells square-shaped
            linewidths=0.5)  # Add lines between cells

plt.title('Correlation Matrix', fontsize=16)
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
plt.yticks(rotation=0)  # Keep y labels horizontal
plt.tight_layout()  # Adjust layout to fit labels
plt.show()

#2d
# Create the correlation matrix including these features
corr_matrix = df.corr()

# Dynamically identify key features correlated with 'Total_Acc' for model development
key_features_dynamic = corr_matrix['Total_Acc'].sort_values(ascending=False)
print("Key features correlated with Total Acceleration:")
print(key_features_dynamic)

# Adjust the correlation threshold to 0.4 (or lower as needed)
key_features_high_corr = key_features_dynamic[abs(key_features_dynamic) > 0.4].index.tolist()
print("Features with moderate to high correlation to Total Acceleration (Threshold: 0.4):")
print(key_features_high_corr)

# Visualize the correlation matrix of the selected key features
corr_matrix_key_features = df[key_features_high_corr].corr()
plt.figure(figsize=(14, 10))  # Increase the figure size
sns.heatmap(corr_matrix_key_features, annot=True, fmt=".2f", cmap='coolwarm', 
            annot_kws={"size": 8},  # Reduce annotation size
            cbar_kws={"shrink": .8},  # Adjust color bar size
            square=True,  # Make cells square-shaped
            linewidths=0.5)  # Add lines between cells
plt.title('Correlation Matrix of Key Features (Threshold: 0.4)')
plt.show()

# Set correlation threshold for moderate correlation (you can adjust as needed)
lower_threshold = 0.3
upper_threshold = 0.7

# Create the correlation matrix for all features
corr_matrix = df.corr()

# Dynamically identify features correlated with 'label' within the moderate correlation range
label_corr = corr_matrix['label'].sort_values(ascending=False)
print("Correlation of all features with 'label':")
print(label_corr)

# Select features with moderate correlation (between 0.3 and 0.7 or -0.3 and -0.7)
moderate_corr_features = label_corr[(abs(label_corr) >= lower_threshold) & (abs(label_corr) <= upper_threshold)].index.tolist()
print(f"Features with moderate correlation to label (Threshold: {lower_threshold}-{upper_threshold}):")
print(moderate_corr_features)

# Visualize the correlation matrix for these moderately correlated features
corr_matrix_moderate = df[moderate_corr_features + ['label']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_moderate, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features with Moderate Correlation to Label')
plt.show()

selected_features = ['Total_Acc', 'Acc_Magnitude', 'Gyro_Magnitude', 'Total_Gyro_Acc', 'label']

# 2d. ii. Verify Correlation Consistency Across Data Subsets
# Step 1: Split the dataset into two subsets (for example, using random sampling)
subset_1 = df.sample(frac=0.5, random_state=42)  # Randomly sample 50% of the data
subset_2 = df.drop(subset_1.index)  # Use the rest of the data as the second subset

# Step 2: Calculate the correlation matrices for both subsets
corr_subset_1 = subset_1[selected_features].corr()
corr_subset_2 = subset_2[selected_features].corr()

# Step 3: Print and compare correlation matrices
print("Correlation matrix for subset 1:")
print(corr_subset_1)

print("\nCorrelation matrix for subset 2:")
print(corr_subset_2)

# Step 4: Visualize the correlation consistency for both subsets using heatmaps
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.heatmap(corr_subset_1, annot=True, cmap='coolwarm')
plt.title('Subset 1 Correlation Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(corr_subset_2, annot=True, cmap='coolwarm')
plt.title('Subset 2 Correlation Matrix')

plt.tight_layout()
plt.show()


