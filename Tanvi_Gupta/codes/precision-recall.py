# Import necessary libraries
from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate Precision, Recall, and F1-Score
precision = precision_score(data['true_anomaly'], data['iso_forest_anomaly'])
recall = recall_score(data['true_anomaly'], data['iso_forest_anomaly'])
f1 = f1_score(data['true_anomaly'], data['iso_forest_anomaly'])

# Display the results
print("Isolation Forest Performance Metrics:")
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Generate precision-recall values for different thresholds
precision_values, recall_values, thresholds = precision_recall_curve(data['true_anomaly'], data['iso_forest_anomaly'])

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_values, precision_values, marker='.', label='Isolation Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Isolation Forest')
plt.legend()
plt.grid(True)
plt.show()