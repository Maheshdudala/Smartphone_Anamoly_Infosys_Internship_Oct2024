import numpy as np

# Introduce synthetic anomalies by adding noise to a subset of data
synthetic_anomalies = augmented_data.copy()
num_anomalies = 100
anomaly_indices = np.random.choice(synthetic_anomalies.index, num_anomalies, replace=False)

# Add noise to simulate anomalies
synthetic_anomalies.loc[anomaly_indices, 'Acc X'] += np.random.normal(5, 2, num_anomalies)
synthetic_anomalies.loc[anomaly_indices, 'gyro_x'] += np.random.normal(5, 2, num_anomalies)

# Fit models on augmented data with synthetic anomalies
iso_forest.fit(synthetic_anomalies[features])
iso_preds_synthetic = iso_forest.predict(synthetic_anomalies[features])
iso_preds_synthetic = (iso_preds_synthetic == -1).astype(int)

lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
lof_preds_synthetic = lof.fit_predict(synthetic_anomalies[features])
lof_preds_synthetic = (lof_preds_synthetic == -1).astype(int)

# Print evaluation metrics on synthetic data
print("\nEvaluation on Synthetic Anomalies (Isolation Forest):")
print(f"Accuracy: {accuracy_score(synthetic_anomalies['label'], iso_preds_synthetic):.2f}")
print(f"Precision: {precision_score(synthetic_anomalies['label'], iso_preds_synthetic):.2f}")
print(f"Recall: {recall_score(synthetic_anomalies['label'], iso_preds_synthetic):.2f}")
print(f"F1 Score: {f1_score(synthetic_anomalies['label'], iso_preds_synthetic):.2f}")

print("\nEvaluation on Synthetic Anomalies (LOF):")
print(f"Accuracy: {accuracy_score(synthetic_anomalies['label'], lof_preds_synthetic):.2f}")
print(f"Precision: {precision_score(synthetic_anomalies['label'], lof_preds_synthetic):.2f}")
print(f"Recall: {recall_score(synthetic_anomalies['label'], lof_preds_synthetic):.2f}")
print(f"F1 Score: {f1_score(synthetic_anomalies['label'], lof_preds_synthetic):.2f}")
