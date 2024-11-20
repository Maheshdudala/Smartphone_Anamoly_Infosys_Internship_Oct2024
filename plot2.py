import matplotlib.pyplot as plt
import seaborn as sns

# Plot Acc X vs Gyro X to visualize enhanced anomalies
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_enhanced, x='Acc X', y='gyro_x', hue='IsoForest_Enhanced_Anomaly', palette='coolwarm')
plt.title("Isolation Forest Detection of Enhanced Synthetic Anomalies")
plt.xlabel("Acc X")
plt.ylabel("Gyro X")
plt.legend(title="Anomaly Detection")
plt.show()
