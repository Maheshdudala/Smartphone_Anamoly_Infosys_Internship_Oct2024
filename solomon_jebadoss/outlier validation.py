# Z-score outlier detection function
def z_score_outliers(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[np.abs(z_scores) > threshold]

# Function to flag and compare outliers for each column using both IQR and Z-score
def flag_and_compare_outliers(df, column, iqr_multiplier=1.5, z_threshold=3):
    iqr_out = iqr_outliers(df, column, iqr_multiplier)
    z_out = z_score_outliers(df, column, z_threshold)
    
    # Finding common outliers flagged by both methods
    common_outliers = pd.merge(iqr_out, z_out, how='inner', on=df.columns.tolist())
    return common_outliers, iqr_out, z_out

# 3b: Visualize anomalies using time-series and scatter plots
for feature in key_features:
    common_outliers, iqr_out, z_out = flag_and_compare_outliers(df, feature, iqr_multiplier=2.0, z_threshold=2.5)

    print(f"--- Column: {feature} ---")
    print(f"Common outliers: {len(common_outliers)}")
    print(f"Outliers by IQR: {len(iqr_out)}")
    print(f"Outliers by Z-score: {len(z_out)}")
    
# 3c: Evaluate the impact of new features on outlier detection results and propose a feature set

# Evaluate outlier detection for each feature
for feature in key_features:
    common_outliers, iqr_out, z_out = flag_and_compare_outliers(df, feature, iqr_multiplier=2.0, z_threshold=2.5)
    
    print(f"--- {feature} Evaluation ---")
    print(f"Common outliers: {len(common_outliers)}")
    print(f"Outliers by IQR: {len(iqr_out)}")
    print(f"Outliers by Z-score: {len(z_out)}")
    
# Propose a feature set for machine learning based on key features with strong correlation to 'label'
proposed_features = ['Total_Acc', 'Acc_Magnitude', 'Gyro_Magnitude', 'label']
print("Proposed Feature Set for Machine Learning:", proposed_features)

# These features can now be used to train anomaly detection models.
# Fine-tune IQR and Z-score methods for each feature
def z_score_outliers(df, column, threshold=3):
    z_scores = (df[column] - df[column].mean()) / df[column].std()
    return df[np.abs(z_scores) > threshold]

def iqr_outliers(df, column, iqr_multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Updated fine-tune parameters for each feature
fine_tune_params = {
    'Total_Acc': {'iqr_multiplier': 2.25, 'z_threshold': 2.25},
    'Acc_Magnitude': {'iqr_multiplier': 2.25, 'z_threshold': 2.25},
    'Gyro_Magnitude': {'iqr_multiplier': 2.0, 'z_threshold': 2.25},
    'label': {'iqr_multiplier': 1.5, 'z_threshold': 3.0}  # No changes for label
}

# Evaluate outliers with updated fine-tuned thresholds
def evaluate_outliers_fine_tuned(df, column, iqr_multiplier, z_threshold):
    iqr_out = iqr_outliers(df, column, iqr_multiplier)
    z_out = z_score_outliers(df, column, z_threshold)
    
    common_outliers = pd.merge(iqr_out, z_out, how='inner', on=df.columns.tolist())
    
    # False Positives and False Negatives
    false_positives_iqr = iqr_out[~iqr_out.index.isin(z_out.index)]  # Outliers by IQR but not Z-score
    false_negatives_iqr = df[(~df.index.isin(iqr_out.index)) & (df.index.isin(z_out.index))]  # Z-score outliers missed by IQR
    
    false_positives_z = z_out[~z_out.index.isin(iqr_out.index)]  # Outliers by Z-score but not IQR
    false_negatives_z = df[(~df.index.isin(z_out.index)) & (df.index.isin(iqr_out.index))]  # IQR outliers missed by Z-score

    return {
        'common_outliers': len(common_outliers),
        'false_positives_iqr': len(false_positives_iqr),
        'false_negatives_iqr': len(false_negatives_iqr),
        'false_positives_z': len(false_positives_z),
        'false_negatives_z': len(false_negatives_z)
    }

# Run the fine-tuned evaluation for each feature
for feature, params in fine_tune_params.items():
    results = evaluate_outliers_fine_tuned(df, feature, params['iqr_multiplier'], params['z_threshold'])

    print(f"--- {feature} Fine-Tuned Outlier Evaluation ---")
    print(f"Common Outliers: {results['common_outliers']}")
    print(f"False Positives (IQR): {results['false_positives_iqr']}")
    print(f"False Negatives (IQR): {results['false_negatives_iqr']}")
    print(f"False Positives (Z-score): {results['false_positives_z']}")
    print(f"False Negatives (Z-score): {results['false_negatives_z']}")
    print("\n")

