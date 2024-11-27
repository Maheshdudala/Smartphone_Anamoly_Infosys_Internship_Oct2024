from sklearn.metrics import confusion_matrix, classification_report

# Now that we have the necessary columns, we can perform the comparison
iqr_results = data['IQR_anomaly']
z_score_results = data['Z_Score_anomaly']

# Calculate confusion matrices and classification reports
conf_matrix_iso = confusion_matrix(data['true_anomaly'], data['iso_forest_anomaly'])
conf_matrix_iqr = confusion_matrix(data['true_anomaly'], iqr_results)
conf_matrix_zscore = confusion_matrix(data['true_anomaly'], z_score_results)

class_report_iso = classification_report(data['true_anomaly'], data['iso_forest_anomaly'])
class_report_iqr = classification_report(data['true_anomaly'], iqr_results)
class_report_zscore = classification_report(data['true_anomaly'], z_score_results)

print("Isolation Forest Classification Report:\n", class_report_iso)
print("IQR Classification Report:\n", class_report_iqr)
print("Z-Score Classification Report:\n", class_report_zscore)
