from sklearn.metrics import accuracy_score

# 1. Original Dataset Accuracy
# Fit Isolation Forest on original data with selected hyperparameters
iso_forest_original = IsolationForest(contamination=0.05, n_estimators=100, max_samples=0.75, random_state=42)
iso_forest_original.fit(df_final[['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']])

# Predict on original data and convert -1 (anomaly) to 1, 1 (normal) to 0 for consistency with labels
predictions_original = iso_forest_original.predict(df_final[['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']])
predictions_original = (predictions_original == -1).astype(int)

# Calculate accuracy on original data
accuracy_original = accuracy_score(df_final['label'], predictions_original)
print(f"Accuracy on Original Data: {accuracy_original:.2f}")

# 2. Augmented Dataset Accuracy
# Fit Isolation Forest on augmented data
iso_forest_augmented = IsolationForest(contamination=0.05, n_estimators=100, max_samples=0.75, random_state=42)
iso_forest_augmented.fit(augmented_data[['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']])

# Predict on augmented data and convert predictions to match the label format (1 for anomaly, 0 for normal)
predictions_augmented = iso_forest_augmented.predict(augmented_data[['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']])
predictions_augmented = (predictions_augmented == -1).astype(int)

# Calculate accuracy on augmented data
accuracy_augmented = accuracy_score(augmented_data['label'], predictions_augmented)
print(f"Accuracy on Augmented Data: {accuracy_augmented:.2f}")

# 3. Comparison Summary
print(f"\nAccuracy Comparison:\nOriginal Dataset: {accuracy_original:.2f}\nAugmented Dataset: {accuracy_augmented:.2f}")
