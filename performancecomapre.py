import seaborn as sns
import matplotlib.pyplot as plt

# Plot metrics for stress testing results
fig, ax = plt.subplots(2, 2, figsize=(16, 12))
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]
axes = ax.ravel()

for i, metric in enumerate(metrics):
    sns.barplot(data=df_stress_results, x="Model", y=metric, hue="Anomaly Type", ax=axes[i])
    axes[i].set_title(f"Model Performance on Stress Test - {metric}")
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.show()
