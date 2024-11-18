from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Isolation Forest
iso_preds = iso_forest.predict(augmented_data[features])
iso_preds = (iso_preds == -1).astype(int)
accuracy_iso = accuracy_score(augmented_data['label'], iso_preds)
precision_iso = precision_score(augmented_data['label'], iso_preds)
recall_iso = recall_score(augmented_data['label'], iso_preds)
f1_iso = f1_score(augmented_data['label'], iso_preds)

# LOF with novelty=False for labeled data
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=False)
lof_preds = lof.fit_predict(augmented_data[features])
lof_preds = (lof_preds == -1).astype(int)
accuracy_lof = accuracy_score(augmented_data['label'], lof_preds)
precision_lof = precision_score(augmented_data['label'], lof_preds)
recall_lof = recall_score(augmented_data['label'], lof_preds)
f1_lof = f1_score(augmented_data['label'], lof_preds)

print("Isolation Forest:")
print(f"Accuracy: {accuracy_iso:.2f}, Precision: {precision_iso:.2f}, Recall: {recall_iso:.2f}, F1 Score: {f1_iso:.2f}")

print("\nLocal Outlier Factor (LOF):")
print(f"Accuracy: {accuracy_lof:.2f}, Precision: {precision_lof:.2f}, Recall: {recall_lof:.2f}, F1 Score: {f1_lof:.2f}")
