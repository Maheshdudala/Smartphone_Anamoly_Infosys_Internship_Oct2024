# Set up models for stress-tested data
iso_forest = IsolationForest(contamination=0.05, n_estimators=200, max_samples=0.75, random_state=42)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)

# Train Isolation Forest and make predictions
iso_forest.fit(df_stressed[features])
iso_preds = iso_forest.predict(df_stressed[features])
iso_preds = (iso_preds == -1).astype(int)  # Convert to binary format: 1 for anomaly, 0 for normal

# Track Isolation Forest detection rates
iso_accuracy = accuracy_score(df_stressed['label'], iso_preds)
iso_precision = precision_score(df_stressed['label'], iso_preds)
iso_recall = recall_score(df_stressed['label'], iso_preds)
iso_f1 = f1_score(df_stressed['label'], iso_preds)

print("\nIsolation Forest - Detection Rates on Stressed Data:")
print(f"Accuracy: {iso_accuracy:.4f}, Precision: {iso_precision:.4f}, Recall: {iso_recall:.4f}, F1 Score: {iso_f1:.4f}")

# Apply LOF and make predictions
lof_preds = lof.fit_predict(df_stressed[features])
lof_preds = (lof_preds == -1).astype(int)  # Convert to binary format

# Track LOF detection rates
lof_accuracy = accuracy_score(df_stressed['label'], lof_preds)
lof_precision = precision_score(df_stressed['label'], lof_preds)
lof_recall = recall_score(df_stressed['label'], lof_preds)
lof_f1 = f1_score(df_stressed['label'], lof_preds)

print("\nLocal Outlier Factor (LOF) - Detection Rates on Stressed Data:")
print(f"Accuracy: {lof_accuracy:.4f}, Precision: {lof_precision:.4f}, Recall: {lof_recall:.4f}, F1 Score: {lof_f1:.4f}")
