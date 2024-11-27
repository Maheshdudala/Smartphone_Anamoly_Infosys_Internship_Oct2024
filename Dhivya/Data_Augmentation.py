df_final=data
# Function to generate synthetic anomalies
def generate_synthetic_anomalies(data, num_anomalies=50):
synthetic_data = data.copy()
anomalies = []
for _ in range(num_anomalies):
random_row = data.sample(1).copy()
random_row[['Acc X', 'Acc Y', 'Acc Z']] += np.random.normal(0, 5, 3)
random_row[['gyro_x', 'gyro_y', 'gyro_z']] += np.random.normal(0, 3, 3)
anomalies.append(random_row)
synthetic_anomalies = pd.concat(anomalies, ignore_index=True)
augmented_data = pd.concat([synthetic_data, synthetic_anomalies],ignore_index=True)
# Label synthetic anomalies as 1 and real data as 0
augmented_data['label'] = [0] * len(data) + [1] * num_anomalies
return augmented_data
# Generate and integrate synthetic anomalies
augmented_data = generate_synthetic_anomalies(df_final, num_anomalies=100)
# Rerun Isolation Forest with augmented data
iso_forest_augmented = IsolationForest(contamination=0.05, n_estimators=100,max_samples=0.75, random_state=42)
iso_forest_augmented.fit(augmented_data[['Acc X', 'Acc Y', 'Acc Z', 'gyro_x','gyro_y', 'gyro_z']])
# Predictions on augmented data
predictions_augmented = iso_forest_augmented.predict(augmented_data[['Acc X','Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']])
augmented_data['anomaly'] = (predictions_augmented == -1).astype(int)
# Visualization: Scatter plot for synthetic anomalies
plt.figure(figsize=(10, 6))
sns.scatterplot(x=augmented_data.index, y=augmented_data['Acc X'],hue=augmented_data['anomaly'], palette="coolwarm")
plt.title("Visualization of Augmented Anomalies")
plt.show()
# Recalculate precision, recall, and F1-score with augmented data
accuracy_aug = accuracy_score(augmented_data['label'],augmented_data['anomaly'])
precision_aug = precision_score(augmented_data['label'],augmented_data['anomaly'])
recall_aug = recall_score(augmented_data['label'], augmented_data['anomaly'])
f1_aug = f1_score(augmented_data['label'], augmented_data['anomaly'])
print(f'Accuracy: {accuracy_aug}')
print("\nPerformance on Augmented Data:")
print(f"Precision: {precision_aug:.2f}, Recall: {recall_aug:.2f}, F1 Score:{f1_aug:.2f}")
# Compare performance
print(f"\nComparison with Original Performance:\nPrecision: {precision:.2f} vs{precision_aug:.2f}")
print(f"Recall: {recall:.2f} vs {recall_aug:.2f}")
print(f"F1 Score: {f1:.2f} vs {f1_aug:.2f}")
