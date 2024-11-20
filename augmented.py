from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Fit Isolation Forest on augmented data with chosen hyperparameters
iso_forest_augmented = IsolationForest(contamination=0.05, n_estimators=100, max_samples=0.75, random_state=42)
iso_forest_augmented.fit(augmented_data[['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']])

# Predict on augmented data and convert -1 (anomaly) to 1, 1 (normal) to 0 for consistency with labels
predictions_augmented = iso_forest_augmented.predict(augmented_data[['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']])
predictions_augmented = (predictions_augmented == -1).astype(int)  # Convert to match label format

# Calculate performance metrics
accuracy_aug = accuracy_score(augmented_data['label'], predictions_augmented)
precision_aug = precision_score(augmented_data['label'], predictions_augmented)
recall_aug = recall_score(augmented_data['label'], predictions_augmented)
f1_aug = f1_score(augmented_data['label'], predictions_augmented)

# Print final performance metrics
print("Final Performance Metrics on Augmented Dataset:")
print(f"Accuracy: {accuracy_aug:.2f}")
print(f"Precision: {precision_aug:.2f}")
print(f"Recall: {recall_aug:.2f}")
print(f"F1 Score: {f1_aug:.2f}")
