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