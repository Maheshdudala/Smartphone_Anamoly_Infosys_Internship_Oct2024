import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

# Define columns for x-axis and y-axis in the simulation
x_axis_column = 'Total_Acc'  # Using Total_Acc as the x-axis
y_axis_column = 'Anomaly_Score'  # Assuming Anomaly_Score is already calculated

# Ensure augmented_data is sorted by x-axis column if it represents progression
augmented_data = augmented_data.sort_values(by=x_axis_column)

# Identify anomalies
anomalies = augmented_data[augmented_data['anomaly'] == 1]  # Adjust flag column name as needed

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=augmented_data[x_axis_column], y=augmented_data[y_axis_column], ax=ax, label="Anomaly Score")

# Frame skip value
frame_skip = 500

# Update function for animation
def update(frame):
    ax.clear()
    frame_index = frame * frame_skip  # Calculate the index to skip frames
    if frame_index < len(augmented_data):  # Ensure the index does not exceed the data length
        sns.lineplot(x=augmented_data[x_axis_column].iloc[:frame_index], 
                     y=augmented_data[y_axis_column].iloc[:frame_index], ax=ax, label="Anomaly Score")
        sns.scatterplot(x=anomalies[x_axis_column], 
                        y=anomalies[y_axis_column], color='red', marker='o', s=30, label='Detected Anomaly', ax=ax)
    ax.set_title("Anomaly Detection Over Total Acceleration")
    ax.set_xlabel(x_axis_column)
    ax.set_ylabel(y_axis_column)
    ax.legend()
    ax.set_xlim(augmented_data[x_axis_column].min(), augmented_data[x_axis_column].max())  # Dynamic x-limits

# Create the animation with a faster interval and frame skipping
ani = animation.FuncAnimation(fig, update, frames=len(augmented_data) // frame_skip, interval=50)  # Reduced interval for faster animation
plt.show()

# Save animation if needed
ani.save("anomaly_detection_simulation_total_acc.mp4", writer='ffmpeg')
