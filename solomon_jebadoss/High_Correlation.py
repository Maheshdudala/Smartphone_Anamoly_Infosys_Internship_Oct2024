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
