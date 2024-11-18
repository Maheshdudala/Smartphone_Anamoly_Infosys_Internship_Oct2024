import pandas as pd
import numpy as np
import zipfile
import os

# Unzip the uploaded file
zip_file_path = '5stn873wft-1.zip'
unzip_dir = 'infosys/data/unzipped_data/'

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(unzip_dir)

# Load the individual files
file1_path = os.path.join(unzip_dir, '1_20210317_184512.csv')
file2_path = os.path.join(unzip_dir, '2_20210317_171452.csv')

df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Merge the two files based on a common key, such as Time (assuming Time is the common column)
df_merged = pd.concat([df1, df2], ignore_index=True)

# Load the accelerometer and gyroscope data with labels from 3_FinalDatasetCsv.csv
dataset_path = os.path.join(unzip_dir, '3_FinalDatasetCsv.csv')
df_acc_gyro = pd.read_csv(dataset_path)

# Check the first few rows to ensure the data structure
print("Merged Data Preview:")
print(df_merged.head())

print("Accelerometer and Gyroscope Data Preview:")
print(df_acc_gyro.head())

# Ensure the number of rows between merged data and acc/gyro data match
# If there's a mismatch, you might need to align based on timestamps or trim rows accordingly

# Join the datasets (assuming they are aligned by row or time)
# For now, we assume they are aligned by index (row-wise) since no specific common key is given
df_final = df_merged.copy()
df_final[['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']] = df_acc_gyro[['Acc X', 'Acc Y', 'Acc Z', 'gyro_x', 'gyro_y', 'gyro_z', 'label']]

# Adjust speed for human context
speed_scale_factor = 0.2  # Adjust this based on your assumptions
df_final['Speed'] = df_final['Speed'] * speed_scale_factor

# Adjust labels (0: normal, 1: anomalous behavior)
df_final['label'] = df_final['label'].apply(lambda x: 0 if x == 0 else 1)

# Save the final adjusted dataset
adjusted_dataset_path = 'infosys/data/final_adjusted_crowd_dataset.csv'
df_final.to_csv(adjusted_dataset_path, index=False)

print(f"Final adjusted dataset saved to {adjusted_dataset_path}")

# Summary of Adjustments
print("Adjustments made:")
print("- Merged data from two files.")
print("- Added accelerometer and gyroscope data with labels.")
print("- Speed scaled down by factor of 0.2 to simulate human walking/running speeds.")
print("- Labels adjusted (0: normal behavior, 1: anomalous behavior).")
