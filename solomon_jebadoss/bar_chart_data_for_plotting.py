# Separate data by 'Dataset' type
original_data = df_performance[df_performance["Dataset"] == "Original"]
augmented_data = df_performance[df_performance["Dataset"] == "Augmented"]

# Define width and x locations for grouped bar chart
width = 0.2
x = np.arange(len(original_data["Model"]))  # label locations
metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

# Set up figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each metric for both datasets
for i, metric in enumerate(metrics):
    ax.bar(x - width * 1.5 + i * width, original_data[metric], width, label=f"{metric} (Original)")
    ax.bar(x + width * 1.5 + i * width, augmented_data[metric], width, label=f"{metric} (Augmented)")

# Label adjustments and titles
ax.set_title("Model Performance Comparison on Original vs Augmented Dataset", fontsize=14)
ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(original_data["Model"], rotation=45, ha="right")
ax.legend(title="Performance Metrics", bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.tight_layout()
plt.show()