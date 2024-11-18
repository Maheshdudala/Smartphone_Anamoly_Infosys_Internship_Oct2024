# 3d.iii: Use domain knowledge to validate flagged outliers with updated thresholds

def validate_outliers_with_domain(df, common_outliers, column):
    # Update thresholds based on revised domain knowledge
    if column == 'Total_Acc':
        threshold = 4  # Updated threshold for aggressive acceleration (m/s²)
    elif column == 'Acc_Magnitude':
        threshold = 2  # Updated threshold for unusual acceleration (m/s²)
    elif column == 'Gyro_Magnitude':
        threshold = -2   # Updated threshold for rapid orientation changes (rad/s)
    else:
        print(f"Domain knowledge not available for {column}")
        return None, None
    
    # Validate outliers: Outliers that exceed the threshold are considered valid
    valid_outliers = common_outliers[common_outliers[column] > threshold]
    invalid_outliers = common_outliers[common_outliers[column] <= threshold]
    
    # Report results
    print(f"--- {column} Outlier Validation ---")
    print(f"Total Common Outliers: {len(common_outliers)}")
    print(f"Valid Outliers (Exceeding Domain Threshold): {len(valid_outliers)}")
    print(f"Invalid Outliers (Below Domain Threshold): {len(invalid_outliers)}")

    return valid_outliers, invalid_outliers

# Example use with updated thresholds
for feature, params in fine_tune_params.items():
    print(f"Validating outliers for {feature} based on domain knowledge:")
    common_outliers = analyze_outliers_validity(df, feature, params['iqr_multiplier'], params['z_threshold'])
    validate_outliers_with_domain(df, common_outliers, feature)
