# Select specific columns to include in the correlation matrix
columns_of_interest = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y','gyro_z','Heading']
selected_df = data[columns_of_interest]
# Calculate the correlation matrix for the selected columns
correlation_matrix = selected_df.corr()
# Plot the heatmap
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1,vmax=1)
# Rotate the labels
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=90)
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0)
plt.title('Correlation Heatmap')
plt.show()
