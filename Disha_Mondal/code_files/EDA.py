# 1a. Data Quality and Preliminary Analysis

# i. Assess data quality (missing values, duplicates)
print("\nData Info:")
df.info()

print("\nMissing Values Count:")
print(df.isnull().sum())

print("\nRemoving duplicates...")
df = df.drop_duplicates()

# Handle missing 'Time' values by forward filling (if applicable)
df['Time'].fillna(method='ffill', inplace=True)

# 1a. ii. Identify potential noise (sensor readings)
sensor_columns = ['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z']

for col in sensor_columns:
    plt.figure(figsize=(10, 4))
    plt.plot(df['Time'], df[col], label=col)
    plt.title(f"{col} Time-Series Plot (Raw Data)")
    plt.xlabel('Time')
    plt.ylabel(f'{col}')
    plt.legend()
    plt.show()

# 1a. iii. Perform basic summary statistics (mean, median, std)
print("\nSummary Statistics for All Sensors:")
print(df[sensor_columns].describe())

# 1a. iv. Visualize data patterns over time using time-series plots
plt.figure(figsize=(10, 4))
plt.plot(df['Time'], df['Speed'], label='Speed')
plt.title("Speed Time-Series Plot")
plt.xlabel('Time')
plt.ylabel('Speed')
plt.legend()
plt.show()

for col in sensor_columns:
    plt.figure(figsize=(10, 4))
    plt.plot(df['Time'], df[col], label=col)
    plt.title(f"{col} Time-Series Plot")
    plt.xlabel('Time')
    plt.ylabel(f'{col}')
    plt.legend()
    plt.show()
# 1c. Univariate Analysis (Plot Distributions and Identify Outliers)

# i. Plot distributions for each sensor using box plots and histograms

# Box Plots for Outlier Detection
print('Plotting box plots for outlier detection...')
for col in sensor_columns + ['Speed', 'Total_Acc']:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f"Box and Whisker Plot for {col}")
    plt.show()

# Histograms for Distribution Analysis
print('Plotting histograms for distribution analysis...')
for col in sensor_columns + ['Speed', 'Total_Acc']:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Histogram of {col}")
    plt.show()

# ii. Correlation Analysis Between Sensors

# Scatter plot for sensor pairs to examine relationships
print('Examining correlations between sensor readings...')
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Acc X', y='Acc Y', data=df)
plt.title("Acc X vs Acc Y")
plt.show()

# iii. Correlation Matrix and Heatmap
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()

