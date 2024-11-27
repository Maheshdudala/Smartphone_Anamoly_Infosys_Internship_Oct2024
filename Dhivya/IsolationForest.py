from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
# Load the dataset
data = pd.read_csv('augmented_dataset.csv')
# Create a true anomaly label if it's not already present
# Assuming the 'Anomaly' column indicates the presence of an anomaly
data['true_anomaly'] = data['Anomaly'].astype(int)
# Feature selection
features = ['Acc X', 'Acc Y', 'Acc Z', 'Speed', 'Acceleration_Rate', 'Total_Acc']
X = data[features]
print(X.isnull().sum())  # Check for missing values
print(np.isinf(X).sum())  # Check for infinite values
from sklearn.ensemble import IsolationForest
# Initialize and fit the Isolation Forest model
iso_forest = IsolationForest(contamination=0.1, random_state=42)
data['iso_forest_anomaly'] = iso_forest.fit_predict(X)
# Convert predictions to binary (1 for anomaly, 0 for normal)
data['iso_forest_anomaly'] = (data['iso_forest_anomaly'] == -1).astype(int)
