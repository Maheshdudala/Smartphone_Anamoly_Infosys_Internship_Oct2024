# 3d.i: Analyze outliers flagged by IQR and Z-score for validity

def analyze_outliers_validity(df, column, iqr_multiplier, z_threshold):
    # Get IQR and Z-score outliers
    iqr_out = iqr_outliers(df, column, iqr_multiplier)
    z_out = z_score_outliers(df, column, z_threshold)
    
    # Common outliers (flagged by both IQR and Z-score)
    common_outliers = pd.merge(iqr_out, z_out, how='inner', on=df.columns.tolist())
    
    # Analyze outliers for validity (you can replace this with actual domain validation logic)
    print(f"Total Common Outliers Detected in {column}: {len(common_outliers)}")
    print("Sample Outliers (first 5):")
    print(common_outliers.head())  # Display some outliers for a sanity check
    return common_outliers

# Example use
for feature, params in fine_tune_params.items():
    print(f"Analyzing validity of outliers for {feature}:")
    analyze_outliers_validity(df, feature, params['iqr_multiplier'], params['z_threshold'])