print(df['Time'].head(10))  # Check for any inconsistencies in the time format
# Ensure that the 'Time' column is properly formatted by replacing hyphens with colons
print('Converting Time column to datetime format...')

# Replace any hyphens with colons to match the HH:MM:SS format
df['Time'] = df['Time'].str.replace('-', ':')

# Now convert the column to datetime, assuming the corrected format is HH:MM:SS
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')

# Check the conversion
print(df[['Time']].head())